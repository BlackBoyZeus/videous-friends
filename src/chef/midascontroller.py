import cv2
import torch
import numpy as np
import argparse
from torchvision.transforms import ToTensor # For RVM
from collections import deque
import os
import ffmpeg
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('midascontroller.log', mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DepthFieldController:
    def __init__(
        self,
        video_path,
        inserted_video_path,
        depth_bins=32,
        smoothing_window=5,
        model_path=None,
        midas_model_type="MiDaS_small"
    ):
        logger.info(f"Initializing DepthFieldController with video_path={video_path}, inserted_video_path={inserted_video_path}")
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise RuntimeError(f"Failed to open input video: {video_path}")

        self.inserted_video = cv2.VideoCapture(inserted_video_path)
        if not self.inserted_video.isOpened():
            raise RuntimeError(f"Failed to open inserted video: {inserted_video_path}")

        self.depth_bins = depth_bins
        self.smoothing_window = smoothing_window
        self.depth_history = deque(maxlen=smoothing_window)

        logger.info(f"Loading MiDaS model: {midas_model_type}")
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", midas_model_type, trust_repo=True)
            self.midas.eval().to(self.device)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if midas_model_type == "MiDaS_small":
                self.midas_transform = transforms.small_transform
            elif "dpt_large" in midas_model_type or "dpt_hybrid" in midas_model_type or \
                 "beit" in midas_model_type or "levit" in midas_model_type or "swin" in midas_model_type:
                self.midas_transform = transforms.dpt_transform
            else:
                logger.warning(f"MiDaS model type {midas_model_type} not explicitly handled for transform selection. Using default_transform. Check compatibility.")
                self.midas_transform = transforms.default_transform
        except Exception as e:
            logger.error(f"Failed to load MiDaS model or transforms: {e}", exc_info=True)
            raise

        logger.info(f"Loading RVM model, model_path={model_path}")
        try:
            self.rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading local RVM weights from {model_path}")
                self.rvm.load_state_dict(torch.load(model_path, map_location=self.device))
            self.rvm.eval().to(self.device)
        except Exception as e:
            logger.error(f"Failed to load RVM model: {e}", exc_info=True)
            raise
        self.rec = [None] * 4  # RVM recurrent state

        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS) or 30.0
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

        if self.frame_count <= 0 or self.height <= 0 or self.width <= 0:
            logger.error(f"Invalid video properties for main video: frames={self.frame_count}, H={self.height}, W={self.width}")
            raise ValueError("Main video has invalid dimensions or frame count.")

        logger.info(f"Video props: frames={self.frame_count}, fps={self.fps}, size={self.width}x{self.height}")
        self.inserted_count = int(self.inserted_video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.inserted_count == 0:
             logger.warning("Inserted video has 0 frames or metadata error. Will attempt to read dynamically and rewind.")

    def get_depth_map(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor_in = self.midas_transform(frame_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(tensor_in)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(self.height, self.width),
                mode='bicubic', align_corners=False
            ).squeeze()
        depth_np = prediction.cpu().numpy()
        self.depth_history.append(depth_np)
        if len(self.depth_history) >= self.smoothing_window:
            smoothed = np.median(np.stack(list(self.depth_history), axis=0), axis=0)
        else:
            smoothed = depth_np
        mn, mx = smoothed.min(), smoothed.max()
        return (smoothed - mn)/(mx - mn) if mx > mn else np.zeros_like(smoothed, dtype=np.float32)

    def process_frame_with_rvm(self, frame_bgr, trimap_np=None, use_rec=True):
        frame_rgb_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = ToTensor()(frame_rgb_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if use_rec:
                fgr, pha, *self.rec = self.rvm(frame_tensor, *self.rec, downsample_ratio=0.25)
            else:
                fgr, pha = self.rvm(frame_tensor, None, None, None, None, downsample_ratio=0.25)[:2]
        return pha.squeeze(0).squeeze(0).cpu().numpy()

    def build_3d_field_for_frame(self, frame_bgr_np, base_alpha_map_np):
        depth_map_np = self.get_depth_map(frame_bgr_np)
        frame_rgb_np = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb_np.astype(np.float32)/255.0).to(self.device)
        alpha_tensor = torch.from_numpy(base_alpha_map_np.astype(np.float32)).to(self.device)
        dt = torch.from_numpy(depth_map_np.astype(np.float32)).to(self.device)

        field_t = torch.zeros((self.height, self.width, self.depth_bins, 4), device=self.device, dtype=torch.float32)
        bins = (dt * (self.depth_bins - 1)).long().clamp(0, self.depth_bins-1)

        h_coords, w_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )
        field_t[h_coords, w_coords, bins, 0] = frame_tensor[..., 0]
        field_t[h_coords, w_coords, bins, 1] = frame_tensor[..., 1]
        field_t[h_coords, w_coords, bins, 2] = frame_tensor[..., 2]
        field_t[h_coords, w_coords, bins, 3] = alpha_tensor
        return field_t

    def depth_aware_blur(self, field_t):
        blurred = field_t.clone()
        depths = torch.arange(self.depth_bins, device=self.device).float()
        sigma = 1.0
        for d in range(self.depth_bins):
            weights = torch.exp(-((depths - d)**2)/(2*sigma**2)); weights /= weights.sum()
            for c in range(4):
                blurred[...,d,c] = (field_t[..., :,c] * weights.view(1,1,-1)).sum(-1)
        return blurred

    def render_novel_view(self, field_t, shift=(0.0,0.0)):
        H,W,D,_ = field_t.shape
        z_vals = torch.linspace(0.0,1.0,D,device=self.device)
        # Corrected normalization for grid coordinates to match common conventions ([-1, 1] or [0, W-1])
        # Using the [-1, 1] style which is common for ray generation
        ys = (torch.arange(H,device=self.device).float()/ H - 0.5) * 2 # Centered, range approx -1 to 1
        xs = (torch.arange(W,device=self.device).float()/ W - 0.5) * 2 # Centered, range approx -1 to 1
        
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij') # HxW
        rays_o = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)],-1) # HxWx3, (x,y,z=0)
        
        base_dir = torch.tensor([shift[0], shift[1], 1.0], device=self.device) # Base direction
        rays_d = base_dir.view(1,1,3).expand(H,W,3) # Expand to all pixels
        rays_d = rays_d / rays_d.norm(dim=-1,keepdim=True) # Normalize

        acc_rgb = torch.zeros((H,W,3),device=self.device)
        trans = torch.ones((H,W,1),device=self.device)
        dz = z_vals[1]-z_vals[0] if D>1 else 1.0
        for i,z_val in enumerate(z_vals): # Renamed z to z_val to avoid conflict
            # Points on the plane at depth z_val
            px = rays_o[...,0] + z_val*rays_d[...,0] # X coordinates on the depth plane
            py = rays_o[...,1] + z_val*rays_d[...,1] # Y coordinates on the depth plane

            # Convert from normalized coords [-1, 1] back to pixel coords [0, W-1] or [0, H-1]
            wi = ((px * 0.5 + 0.5) * W).round().long().clamp(0,W-1)
            hi = ((py * 0.5 + 0.5) * H).round().long().clamp(0,H-1)
            
            samp = field_t[hi,wi,i] # Sample from the field at the current depth bin i
            col, sig = samp[...,:3], samp[...,3:4]
            a = 1-torch.exp(-sig*dz)
            acc_rgb += trans*a*col
            trans *= torch.exp(-sig*dz)
        fin_alpha = 1-trans
        return acc_rgb, fin_alpha

    def depth_to_rvm_input(self, field_t, drange, mult=1.0):
        mn,mx = drange
        imin = int(mn*(self.depth_bins-1)); imax = int(mx*(self.depth_bins-1))
        imin,imax = max(0,imin), min(self.depth_bins-1,imax) # Clamp indices
        
        trimap = torch.zeros((self.height, self.width),dtype=torch.float32, device=self.device)
        if imin > imax: # Handle invalid range after clamping
            logger.warning(f"Invalid depth range for trimap after clamping: min_idx={imin}, max_idx={imax}.")
            return trimap.cpu().numpy()
            
        slab = field_t[...,imin:imax+1,3].sum(-1)
        mask = slab>(0.1*mult)
        
        nearer = field_t[...,imax+1:,3].sum(-1) if imax<self.depth_bins-1 else torch.zeros_like(mask)
        farther= field_t[...,:imin,3].sum(-1) if imin>0 else torch.zeros_like(mask)
        trimap[mask]=0.5
        trimap[~mask & (nearer>farther)]=1.0
        # Default is 0.0 (background) for ~mask & ~(nearer>farther)
        return trimap.cpu().numpy()

    def place_inserted_frame(self, raw_ins_frame, target_hw, pos_xy_top_left, scale):
        # raw_ins_frame is the original frame from inserted_video, potentially any size/channels
        th, tw = target_hw # Canvas dimensions (self.height, self.width)

        # 1. Ensure raw_ins_frame is BGR or BGRA
        if raw_ins_frame is None: # Should not happen if process_video handles it
            raw_ins_frame = np.zeros((100,100,3), dtype=np.uint8) # Small fallback
            logger.warning("place_inserted_frame received None, using fallback black frame.")

        if len(raw_ins_frame.shape) == 2: # Grayscale
            raw_ins_frame = cv2.cvtColor(raw_ins_frame, cv2.COLOR_GRAY2BGR)
        elif raw_ins_frame.shape[2] == 1: # Also grayscale
            raw_ins_frame = cv2.cvtColor(raw_ins_frame, cv2.COLOR_GRAY2BGR)
        # Now raw_ins_frame is BGR or BGRA

        # 2. Scale the raw inserted frame based on 'scale' relative to canvas size.
        # This maintains aspect ratio of raw_ins_frame.
        raw_h, raw_w = raw_ins_frame.shape[:2]
        
        # Determine target scaled dimensions while preserving aspect ratio
        # If we scale based on canvas height:
        scaled_h = int(th * scale)
        scaled_w = int(raw_w * (scaled_h / raw_h)) if raw_h > 0 else 0
        # Or if we scale based on canvas width (pick one or average or min/max):
        # scaled_w = int(tw * scale)
        # scaled_h = int(raw_h * (scaled_w / raw_w)) if raw_w > 0 else 0
        # For simplicity, let's scale based on the smaller dimension of the canvas after scaling
        # This is a common interpretation: scale means "make it X% of the canvas area/dimension"
        
        # Let's redefine: scale applies to the object so it occupies `scale` fraction of canvas height/width
        # For a non-square canvas and non-square object, this is ambiguous.
        # A common approach: make object's largest scaled dimension fit `scale` of canvas's corresponding dimension
        # or fit within a box of (th*scale, tw*scale).
        
        # Simpler: Scale the object's original dimensions by `scale` directly, then place.
        # This means `scale` is independent of canvas aspect ratio.
        obj_h_scaled = int(raw_h * scale)
        obj_w_scaled = int(raw_w * scale)

        if obj_h_scaled <= 0 or obj_w_scaled <= 0:
            logger.debug(f"Object scaled to zero size ({obj_w_scaled}x{obj_h_scaled}). Returning empty tensors.")
            return torch.zeros((th,tw,3),device=self.device), torch.zeros((th,tw,1),device=self.device)

        obj_for_placement = cv2.resize(raw_ins_frame, (obj_w_scaled, obj_h_scaled), interpolation=cv2.INTER_LINEAR)

        # 3. Convert to RGBA float
        if obj_for_placement.shape[2] == 4: # BGRA
            rgba = cv2.cvtColor(obj_for_placement, cv2.COLOR_BGRA2RGBA).astype(np.float32)/255.0
        else: # BGR
            rgb = cv2.cvtColor(obj_for_placement, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            alpha = np.ones((obj_h_scaled, obj_w_scaled,1),dtype=np.float32)
            rgba = np.concatenate([rgb,alpha],axis=2)
        
        # 4. Composite onto canvas
        canvas_rgb=np.zeros((th,tw,3),dtype=np.float32)
        canvas_alpha=np.zeros((th,tw,1),dtype=np.float32)
        
        x_tl_place, y_tl_place = pos_xy_top_left # Top-left position for the scaled object

        # Determine bounds on canvas
        y_start_canvas = max(0, y_tl_place)
        x_start_canvas = max(0, x_tl_place)
        y_end_canvas = min(th, y_tl_place + obj_h_scaled)
        x_end_canvas = min(tw, x_tl_place + obj_w_scaled)

        # Determine bounds for slicing the object (if it's partially off-canvas)
        y_start_obj = max(0, -y_tl_place)
        x_start_obj = max(0, -x_tl_place)
        
        slice_h = y_end_canvas - y_start_canvas
        slice_w = x_end_canvas - x_start_canvas
        y_end_obj = y_start_obj + slice_h
        x_end_obj = x_start_obj + slice_w

        if slice_h > 0 and slice_w > 0: # Check if there's a valid overlapping region
            canvas_rgb[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = rgba[y_start_obj:y_end_obj, x_start_obj:x_end_obj, :3]
            canvas_alpha[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = rgba[y_start_obj:y_end_obj, x_start_obj:x_end_obj, 3:4]
        
        return torch.from_numpy(canvas_rgb).to(self.device), torch.from_numpy(canvas_alpha).to(self.device)

    def process_video(self, output_path, insertion_params, cam_params=None, novel_view=False, depth_thresh_multiplier=1.0):
        out=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(self.width,self.height))
        if not out.isOpened(): logger.error(f"Cannot open writer {output_path}"); return
        self.rec=[None]*4 # Reset RVM state
        
        for i in range(self.frame_count):
            logger.info(f"Processing frame {i+1}/{self.frame_count}")
            ret,frame=self.video.read()
            if not ret: 
                logger.warning(f"Could not read frame {i+1} from main video. Stopping.")
                break
            
            ret2,ins_raw=self.inserted_video.read()
            if not ret2:
                if self.inserted_count > 0: # Rewind if it's a video
                    logger.info("Inserted video reached end. Rewinding.")
                    self.inserted_video.set(cv2.CAP_PROP_POS_FRAMES,0)
                    ret2,ins_raw=self.inserted_video.read()
                if not ret2: # Still no frame, or was an image
                    logger.warning("Failed to read from inserted video even after rewind. Using black frame.")
                    ins_raw=np.zeros((self.height,self.width,3),dtype=np.uint8) # Fallback
            
            if ins_raw is None: # Final safeguard for ins_raw
                logger.warning("ins_raw is None after read attempts, fallback to black frame.")
                ins_raw = np.zeros((self.height, self.width, 3), dtype=np.uint8)


            alpha=self.process_frame_with_rvm(frame,None,True)
            field=self.build_3d_field_for_frame(frame,alpha)
            field_bl=self.depth_aware_blur(field)
            
            if novel_view:
                accum,_=self.render_novel_view(field_bl, shift=(0.05,0.0)) # Example shift
            else:
                main_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                accum=torch.from_numpy(main_rgb).to(self.device)
                # Optional: make main video itself matted by RVM alpha
                # base_alpha_tensor = torch.from_numpy(alpha[...,None].astype(np.float32)).to(self.device)
                # accum = accum * base_alpha_tensor

            for p in insertion_params:
                trim=self.depth_to_rvm_input(field_bl,p['depth_range'],depth_thresh_multiplier)
                win_alpha_np=self.process_frame_with_rvm(frame,trim,False) # RVM for insertion window
                win_t=torch.from_numpy(win_alpha_np.astype(np.float32)).unsqueeze(-1).to(self.device)
                
                # Pass the raw inserted frame; place_inserted_frame handles its scaling and conversion
                prgb,palpha=self.place_inserted_frame(ins_raw,(self.height,self.width),p['position'],p['scale'])
                
                eff_alpha=palpha*win_t # Effective alpha = object_alpha * window_alpha
                accum=prgb*eff_alpha + accum*(1-eff_alpha) # Standard over compositing
            
            outf=(accum.clamp(0,1)*255).byte().cpu().numpy()
            out.write(cv2.cvtColor(outf,cv2.COLOR_RGB2BGR))
        
        out.release()
        self.video.release()
        self.inserted_video.release()
        cv2.destroyAllWindows() # Good practice
        logger.info(f"Video processing complete. Saved output to {output_path}")

def parse_args(): # Corrected indentation
    parser=argparse.ArgumentParser(description="Depth-based video compositing with MiDaS and RVM")
    parser.add_argument("--original-path", required=True, help="Path to the main video file")
    parser.add_argument("--background-path", required=True, help="Path to the video/image to be inserted")
    parser.add_argument("--model-path", help="Path to local RVM model weights (e.g., rvm_mobilenetv3.pth)")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output video")
    parser.add_argument("--use-depth", action="store_true", help="Enable novel view rendering using depth (parallax effect)")
    parser.add_argument("--depth-thresh-multiplier", type=float, default=1.0, help="Multiplier for depth threshold in trimap")
    parser.add_argument("--smooth-window", type=int, default=5, help="Temporal smoothing window for depth")
    parser.add_argument("--midas-model-type", default="MiDaS_small", help="MiDaS model type")
    parser.add_argument("--mix-audio", action="store_true", help="Mix audio from input videos")
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    params=[{'depth_range':(0.0,0.3),'position':(50,50),'scale':0.5,'mode':'blend'}] # Example
    logger.info(f"Script arguments: {args}")
    logger.info(f"Using insertion params: {params}")
    
    try:
        ctrl=DepthFieldController(
            args.original_path,args.background_path,
            depth_bins=32,smoothing_window=args.smooth_window,
            model_path=args.model_path,
            midas_model_type=args.midas_model_type
        )
        out_vid=os.path.join(args.output_dir,"composited_output.mp4")
        ctrl.process_video(out_vid,params,novel_view=args.use_depth,depth_thresh_multiplier=args.depth_thresh_multiplier)
        
        if args.mix_audio:
            logger.info("Attempting to mix audio...")
            tmp_audio_out=os.path.join(args.output_dir,"temp_audio_mixed.mp4")
            
            processed_video_input = ffmpeg.input(out_vid)
            audio_streams = []
            try:
                ffmpeg.probe(args.original_path, select_streams='a')
                audio_streams.append(ffmpeg.input(args.original_path).audio)
                logger.info("Audio stream found in original video.")
            except ffmpeg.Error:
                logger.warning(f"No audio stream in original video: {args.original_path}")

            try:
                ffmpeg.probe(args.background_path, select_streams='a')
                audio_streams.append(ffmpeg.input(args.background_path).audio)
                logger.info("Audio stream found in background video.")
            except ffmpeg.Error:
                logger.warning(f"No audio stream in background video: {args.background_path}")

            if not audio_streams:
                logger.info("No audio streams to mix. Video remains as is.")
            else:
                if len(audio_streams) == 1:
                    mixed_audio_stream = audio_streams[0]
                else:
                    mixed_audio_stream = ffmpeg.filter(audio_streams,'amix', inputs=len(audio_streams), duration='first', dropout_transition=0)
                
                try:
                    (
                        ffmpeg
                        .output(processed_video_input.video, mixed_audio_stream, tmp_audio_out, vcodec='copy', acodec='aac', strict='experimental')
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True) # Capture for logging
                    )
                    os.replace(tmp_audio_out, out_vid) # Replace original with audio-mixed one
                    logger.info(f"Audio mixed. Final video at {out_vid}")
                except ffmpeg.Error as e:
                    logger.error("FFmpeg audio mixing failed.")
                    logger.error("FFmpeg stdout:\n" + e.stdout.decode('utf8', errors='ignore'))
                    logger.error("FFmpeg stderr:\n" + e.stderr.decode('utf8', errors='ignore'))
                    if os.path.exists(tmp_audio_out): os.remove(tmp_audio_out) # Clean up

    except Exception as e:
        logger.error(f"A critical error occurred in main execution: {e}", exc_info=True)
        '''python midascontroller.py \
  --original-path "./output_results/standardized_reencoded_video.mov" \
  --background-path "./compositing_output/Screen_Recording_2025-05-03_at_9.21.22_AM.mov" \
  --model-path "./rvm_mobilenetv3.pth" \
  --output-dir "./output_results_frame_by_frame" \
  --use-depth \
  --depth-thresh-multiplier 1.0 \
  --smooth-window 5 \
  --midas-model-type "MiDaS_small" \
  --mix-audio
  conda activate video_py310'''