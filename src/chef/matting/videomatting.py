#!/usr/bin/env python
"""
Advanced Standalone Script for Robust Video Matting (RVM)
and Segment Anything 2 (SAM) Integration using a lightweight model.
"""

import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import logging
from typing import Tuple, Optional

# Import SAM modules (requires installing the segment_anything package)
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None

# Set up logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default frame size for fallback
DEFAULT_HEIGHT, DEFAULT_WIDTH = 480, 640

def load_rvm_model() -> Optional[torch.nn.Module]:
    """
    Load the Robust Video Matting model from PyTorch Hub.
    
    Returns:
        The loaded model (evaluated and moved to GPU if available), or None if failed.
    """
    try:
        model = torch.hub.load('PeterL1n/RobustVideoMatting', 'resnet50', pretrained=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("RVM model loaded on GPU.")
        else:
            logger.info("RVM model loaded on CPU.")
        return model
    except Exception as e:
        logger.error(f"Failed to load RVM model: {e}")
        return None

def load_sam_mask_generator(model_type: str = 'vit_t', checkpoint: str = 'sam_vit_t_0b3195.pth') -> Optional[SamAutomaticMaskGenerator]:
    """
    Load the Segment Anything model and return a mask generator using the lightest variant.
    
    Args:
        model_type: Model variant to use (e.g., 'vit_t' for the lightweight version).
        checkpoint: Path or URL to the checkpoint file.
    
    Returns:
        An instance of SamAutomaticMaskGenerator, or None if loading fails.
    """
    if sam_model_registry is None:
        logger.error("segment_anything package not installed. SAM mode will be unavailable.")
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        logger.info("SAM model loaded successfully using the light variant.")
        return mask_generator
    except Exception as e:
        logger.error(f"Failed to load SAM model: {e}")
        return None

class VideoMattingProcessor:
    """
    Class to process video frames with Robust Video Matting (RVM)
    and optional Segment Anything (SAM) segmentation.
    """

    def __init__(self,
                 input_source: str,
                 mode: str = 'composite',
                 downsample_ratio: float = 0.5,
                 background: Optional[np.ndarray] = None,
                 rvm_model: Optional[torch.nn.Module] = None,
                 sam_mask_generator: Optional[SamAutomaticMaskGenerator] = None):
        """
        Initialize the video matting processor.
        
        Args:
            input_source: Video source (0 for webcam, or path to a video file).
            mode: Output mode ('composite', 'foreground', 'alpha', 'green_screen', 'sam').
            downsample_ratio: Downsample ratio for faster processing (0.1 to 1.0).
            background: Optional custom background image (numpy array).
            rvm_model: The loaded RVM model.
            sam_mask_generator: The loaded SAM mask generator.
        """
        self.input_source = input_source
        self.mode = mode.lower()
        self.downsample_ratio = np.clip(downsample_ratio, 0.1, 1.0)
        self.background = background
        self.cap = None
        self.rec = [None] * 4  # Recurrent states for RVM
        self.frame_height = DEFAULT_HEIGHT
        self.frame_width = DEFAULT_WIDTH
        self.window_name = "Video Matting & SAM"
        self.rvm_model = rvm_model
        self.sam_mask_generator = sam_mask_generator

        valid_modes = {'composite', 'foreground', 'alpha', 'green_screen', 'sam'}
        if self.mode not in valid_modes:
            logger.warning(f"Invalid mode: {self.mode}. Defaulting to 'composite'.")
            self.mode = 'composite'

        if self.background is not None and not isinstance(self.background, np.ndarray):
            logger.warning("Background must be a numpy array. Ignoring custom background.")
            self.background = None

    def _initialize_capture(self) -> bool:
        """Initialize video capture from the input source."""
        logger.info(f"Initializing video capture from: {self.input_source}")
        self.cap = cv2.VideoCapture(self.input_source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.input_source}")
            return False
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.error("Failed to capture initial frame.")
            self.cap.release()
            return False
        self.frame_height, self.frame_width = frame.shape[:2]
        logger.info(f"Capture initialized: {self.frame_width}x{self.frame_height}")
        return True

    def _cleanup(self):
        """Release resources and close windows."""
        logger.info("Cleaning up resources...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete.")

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess the frame:
         - Resize according to the downsample ratio.
         - Convert from BGR to RGB.
         - Transform into a tensor.
        """
        h, w = frame.shape[:2]
        new_h, new_w = int(h * self.downsample_ratio), int(w * self.downsample_ratio)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        tensor = T.ToTensor()(frame_rgb).unsqueeze(0)  # Shape: [1, C, H, W]
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def _postprocess_output(self, fgr: torch.Tensor, pha: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the model output to numpy arrays and resize to the original frame shape.
        
        Args:
            fgr: The predicted foreground tensor.
            pha: The predicted alpha matte tensor.
        
        Returns:
            Tuple of the foreground image and the alpha matte.
        """
        fgr = fgr.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        pha = pha.squeeze(0).cpu().numpy()  # [H, W]
        fgr = cv2.resize(fgr, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR) * 255
        pha = cv2.resize(pha, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
        alpha = (pha * 255).astype(np.uint8)
        return fgr, alpha

    def _process_with_sam(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame using the Segment Anything (SAM) model.
        The SAM mask generator produces multiple masks which are then overlaid
        on the original frame for visualization.
        
        Args:
            frame: The original video frame.
        
        Returns:
            The frame with SAM segmentation masks overlaid.
        """
        if self.sam_mask_generator is None:
            logger.warning("SAM model not available. Returning original frame.")
            return frame

        try:
            # Generate masks using SAM's automatic mask generator
            masks = self.sam_mask_generator.generate(frame)
            overlay = frame.copy()
            for mask in masks:
                mask_bool = mask['segmentation'].astype(np.uint8) * 255
                # Create colored overlay for each mask
                color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
                contours, _ = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color.tolist(), 2)
            return overlay
        except Exception as e:
            logger.warning(f"SAM processing failed: {e}")
            return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame using the selected mode.
        For 'sam' mode, SAM segmentation is applied.
        For other modes, RVM is used.
        
        Args:
            frame: The original video frame.
        
        Returns:
            The processed frame.
        """
        if self.mode == 'sam':
            return self._process_with_sam(frame)

        if self.rvm_model is None:
            logger.warning("RVM model not available. Returning original frame.")
            return frame

        try:
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame received.")
                return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
            # Preprocess frame and run RVM inference
            src = self._preprocess_frame(frame)
            with torch.no_grad():
                fgr, pha, *self.rec = self.rvm_model(src, *self.rec, downsample_ratio=self.downsample_ratio)
            fgr_np, alpha_np = self._postprocess_output(fgr, pha)

            # Apply mode-specific output for RVM-based processing
            if self.mode == 'foreground':
                return fgr_np.astype(np.uint8)
            elif self.mode == 'alpha':
                return cv2.cvtColor(alpha_np, cv2.COLOR_GRAY2BGR)
            elif self.mode == 'green_screen':
                green_bg = np.full_like(frame, (0, 255, 0), dtype=np.uint8)
                return np.where(alpha_np[..., None] > 127, fgr_np, green_bg).astype(np.uint8)
            else:  # 'composite'
                bg = self.background if self.background is not None else np.full_like(frame, (255, 0, 0), dtype=np.uint8)
                if bg.shape[:2] != frame.shape[:2]:
                    bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
                alpha_float = alpha_np / 255.0
                composite = (fgr_np * alpha_float[..., None] + bg * (1 - alpha_float[..., None])).astype(np.uint8)
                return composite

        except Exception as e:
            logger.warning(f"Frame processing failed: {e}")
            return frame

    def run(self):
        """Run the video processor: capture frames, process, and display results."""
        if not self._initialize_capture():
            return

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.info("End of video or capture error.")
                break

            processed_frame = self.process_frame(frame)

            # Display instructions on the frame
            info_text = ("Mode: {} | C:Composite F:Foreground A:Alpha G:GreenScreen S:SAM | Q:Quit"
                         .format(self.mode.upper()))
            text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x, text_y = 10, self.frame_height - 20
            cv2.rectangle(processed_frame, (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(processed_frame, info_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.window_name, processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.mode = 'composite'
                logger.info("Switched to Composite mode")
            elif key == ord('f'):
                self.mode = 'foreground'
                logger.info("Switched to Foreground mode")
            elif key == ord('a'):
                self.mode = 'alpha'
                logger.info("Switched to Alpha mode")
            elif key == ord('g'):
                self.mode = 'green_screen'
                logger.info("Switched to Green Screen mode")
            elif key == ord('s'):
                self.mode = 'sam'
                logger.info("Switched to SAM segmentation mode")

        self._cleanup()

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Video Matting with RVM and SAM integration"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam or path to a video file)")
    parser.add_argument("--mode", type=str, default="composite",
                        choices=["composite", "foreground", "alpha", "green_screen", "sam"],
                        help="Output mode")
    parser.add_argument("--downsample", type=float, default=0.5,
                        help="Downsample ratio for faster processing (0.1 to 1.0)")
    parser.add_argument("--background", type=str,
                        help="Path to a custom background image (optional)")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_t_0b3195.pth",
                        help="Checkpoint for SAM (if using SAM mode)")
    args = parser.parse_args()

    # Determine the video source type
    input_source = int(args.source) if args.source.isdigit() else args.source

    # Load custom background image if provided
    background_img = None
    if args.background:
        background_img = cv2.imread(args.background)
        if background_img is None:
            logger.warning("Failed to load background image. Proceeding without custom background.")

    # Load the RVM model
    rvm_model = load_rvm_model()

    # Load the SAM mask generator using the light model if available
    sam_mask_generator = load_sam_mask_generator(model_type='vit_t', checkpoint=args.sam_checkpoint)

    # Create and run the processor
    processor = VideoMattingProcessor(
        input_source=input_source,
        mode=args.mode,
        downsample_ratio=args.downsample,
        background=background_img,
        rvm_model=rvm_model,
        sam_mask_generator=sam_mask_generator
    )
    try:
        processor.run()
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        processor._cleanup()

if __name__ == "__main__":
    main()
