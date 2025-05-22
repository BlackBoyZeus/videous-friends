
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
# *** PATCH 5: Apply NumPy patch BEFORE importing skvideo ***
# Patch for older libraries using deprecated NumPy aliases
if not hasattr(np, "float"):
    print("Patching np.float for skvideo/other library compatibility.")
    np.float = float # type: ignore
if not hasattr(np, "int"): # Also patch np.int if needed
    print("Patching np.int for skvideo/other library compatibility.")
    np.int = int # type: ignore

import torch
import subprocess
import os
import platform
import sys
from collections import deque # For temporal smoothing
import traceback # For detailed error printing
from importlib import metadata as importlib_metadata # For package version checks

# --- Optional Imports with Error Handling ---
try:
    import skvideo.io # Import happens *after* the numpy patch
    SKVIDEO_AVAILABLE = True
except ImportError:
    print("Warning: scikit-video not found. scikit-video implementation will be skipped.")
    SKVIDEO_AVAILABLE = False
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("Warning: pydub not found. scikit-video+pydub audio handling will be limited.")
    PYDUB_AVAILABLE = False

try:
    # *** PATCH 4: Use moviepy.editor imports ***
    from moviepy.editor import VideoFileClip, CompositeVideoClip, CompositeAudioClip, AudioFileClip, concatenate_audioclips # Changed import
    from moviepy.video.fx.all import loop # Added loop fx
    import moviepy
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("Warning: moviepy not found or import failed. MoviePy implementation will be skipped.")
    print("         Please ensure moviepy==1.0.3 (or compatible) is installed (`pip install moviepy==1.0.3`).")
    MOVIEPY_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    print("Warning: PyAV not found. PyAV implementation will be skipped.")
    print("         Please install it (`pip install av`).")
    AV_AVAILABLE = False

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
    GSTREAMER_AVAILABLE = True
    print("GStreamer initialized successfully.")
except ImportError:
    print("Warning: PyGObject or GStreamer not found/initialized. GStreamer implementation will be skipped.")
    print("         Ensure PyGObject and GStreamer (including plugins) are installed.")
    GSTREAMER_AVAILABLE = False
except Exception as e:
    print(f"Warning: GStreamer could not be initialized: {e}. GStreamer implementation will be skipped.")
    GSTREAMER_AVAILABLE = False
# --- End Optional Imports ---

# --- Robust Video Matting Import ---
# Adjust this path if your RVM library is located elsewhere
RVM_PATH_GUESS_1 = '../RobustVideoMatting' # Relative to script location if run from parent dir
RVM_PATH_GUESS_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RobustVideoMatting') # Sibling dir
RVM_PATH_GUESS_3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../RobustVideoMatting') # Parent dir

RVM_PATH = None
# Prioritize parent, then sibling, then relative guess
if os.path.isdir(RVM_PATH_GUESS_3): RVM_PATH = RVM_PATH_GUESS_3
elif os.path.isdir(RVM_PATH_GUESS_2): RVM_PATH = RVM_PATH_GUESS_2
elif os.path.isdir(RVM_PATH_GUESS_1): RVM_PATH = RVM_PATH_GUESS_1


RVM_AVAILABLE = False
if RVM_PATH:
    abs_rvm_path = os.path.abspath(RVM_PATH)
    print(f"Attempting to import RVM from: {abs_rvm_path}")
    if abs_rvm_path not in sys.path:
        sys.path.append(abs_rvm_path)
    try:
        from model import MattingNetwork
        try:
            # Try importing from the expected location first
            from inference import convert_video
            RVM_INFERENCE_OK = True
        except ImportError:
            print("Warning: 'convert_video' not found directly in 'inference'. Trying 'inference.inference'.")
            try:
                # Fallback for slightly different structure
                from inference.inference import convert_video
                RVM_INFERENCE_OK = True
            except ImportError:
                 print("Error: Could not find 'convert_video' in 'inference' or 'inference.inference'. Check RVM structure.")
                 RVM_INFERENCE_OK = False

        if RVM_INFERENCE_OK:
            # Optional: Runtime patch attempt for the 'rate' parameter issue in older RVM inference_utils.py
            try:
                import inspect
                import importlib.util
                spec = importlib.util.spec_from_file_location("inference_utils", os.path.join(abs_rvm_path, "inference_utils.py"))
                if spec and spec.loader:
                    inference_utils = importlib.util.module_from_spec(spec)
                    sys.modules["inference_utils"] = inference_utils # Add to sys.modules
                    spec.loader.exec_module(inference_utils) # Execute module content
                    print("Successfully imported RVM's inference_utils for patch check.")

                    init_sig = inspect.signature(inference_utils.VideoWriter.__init__)
                    params = list(init_sig.parameters.values())
                    rate_param_name = None
                    problematic_default = False
                    for param in params[1:]: # Skip 'self'
                        if 'rate' in param.name.lower(): # Find param likely related to frame rate
                            rate_param_name = param.name
                            if param.default is not inspect.Parameter.empty and isinstance(param.default, str) and '{' in param.default:
                                problematic_default = True
                            break

                    if rate_param_name and problematic_default:
                        print(f"Applying runtime patch to RVM inference_utils.VideoWriter '{rate_param_name}' parameter...")
                        original_init = inference_utils.VideoWriter.__init__
                        def patched_init(self, path, frame_rate, *args, **kwargs):
                            bound_args = init_sig.bind(self, path, float(frame_rate), *args, **kwargs)
                            bound_args.apply_defaults()
                            original_init(**bound_args.arguments)
                        inference_utils.VideoWriter.__init__ = patched_init
                        print("Runtime patch applied.")
                    elif rate_param_name:
                        print(f"RVM inference_utils.VideoWriter '{rate_param_name}' parameter looks OK, no runtime patch needed.")
                    else:
                        print("Could not find 'rate' parameter in RVM inference_utils.VideoWriter.__init__ to check.")
                else:
                    print("Could not find or load RVM's inference_utils.py for patch check.")
            except ImportError:
                 print("Could not import RVM's inference_utils to check/patch rate parameter.")
            except FileNotFoundError:
                 print("Could not find RVM's inference_utils.py at expected location for patch check.")
            except Exception as patch_e:
                 print(f"Warning: Error during RVM runtime patch check/attempt: {patch_e}")

            RVM_AVAILABLE = True
            print("RobustVideoMatting imported successfully.")
        # No else needed here, RVM_AVAILABLE remains False if RVM_INFERENCE_OK is False

    except ImportError as e:
        print(f"Error importing from RVM directory '{abs_rvm_path}': {e}")
        RVM_AVAILABLE = False # Ensure it's false
    except FileNotFoundError:
        print(f"Error: RVM directory specified but seems invalid: {abs_rvm_path}")
        RVM_AVAILABLE = False
else:
    print("Error: RobustVideoMatting directory not found at expected locations:")
    print(f"  - {os.path.abspath(RVM_PATH_GUESS_1)}")
    print(f"  - {os.path.abspath(RVM_PATH_GUESS_2)}")
    print(f"  - {os.path.abspath(RVM_PATH_GUESS_3)}")
    RVM_AVAILABLE = False

if not RVM_AVAILABLE:
    print("\nCRITICAL ERROR: RobustVideoMatting is required but could not be loaded. Exiting.")
    sys.exit(1)
# --- End RVM Import ---

# --- MiDaS Depth Estimation Imports and Setup ---
# These will be loaded later based on command-line args, declare placeholders
MIDAS_AVAILABLE = False
midas = None
midas_transform = None
# --- End MiDaS Setup ---

# --- Define Constants ---
# <<< PATHS CAN BE OVERRIDDEN BY COMMAND LINE ARGS >>>
MODEL_PATH = './rvm_mobilenetv3.pth' # Default RVM model path
ORIGINAL_PATH = './input_video.mp4' # Default original video path
BACKGROUND_PATH = './background_video.mp4' # Default background video path
OUTPUT_DIR = './compositing_output_depth_smooth' # Default output directory
# <<< END PATH DEFAULTS >>>

# --- Depth Compositing Parameters (Defaults, can be overridden by args) ---
USE_DEPTH = True # Default to True, overridden by --no-depth
DEPTH_MODE = 'adjust' # Options: 'adjust', 'threshold', 'hierarchical'
DEPTH_ADJUST_INFLUENCE = 0.3 # Tune this value (e.g., 0.1 to 0.7)
DEPTH_THRESHOLD_MEAN_MULTIPLIER = 1.0 # Tune this value (e.g., 0.8 to 1.2)
SMOOTH_WINDOW = 3  # Number of frames for temporal smoothing
DEPTH_RANGES = None  # Will be set via command-line argument: e.g., "0.1-0.4,0.4-0.8"
# --- End Depth Parameters ---

# Output directory created later in main after potential override
FOREGROUND_MATTE_PATH = "" # Will be set in main()
ALPHA_MATTE_PATH = "" # Will be set in main()
# --- End Constants ---

# --- Utility Functions ---
def get_device():
    """Gets the best available PyTorch device (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        print("CUDA is available, using CUDA.")
        return torch.device('cuda')
    # Check MPS availability and build status more robustly
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
             tensor_mps = torch.tensor([1.0], device='mps')
             _ = tensor_mps * 2 # Perform a simple operation
             print("MPS is available and functional, using MPS.")
             return torch.device('mps')
        except Exception as mps_test_e:
             print(f"MPS available but test operation failed: {mps_test_e}. Falling back.")

    print("No GPU (CUDA/functional MPS) available, falling back to CPU.")
    return torch.device('cpu')

# --- Depth Smoother Class for Temporal Consistency ---
class DepthSmoother:
    """Averages depth maps over a temporal window to reduce flickering."""
    def __init__(self, window_size=3):
        self.window_size = max(1, window_size) # Ensure window size is at least 1
        self.history = deque(maxlen=self.window_size)
        print(f"Initialized DepthSmoother with window size {self.window_size}")

    def smooth(self, depth_map):
        """Adds a depth map and returns the smoothed (averaged) map."""
        if depth_map is not None and isinstance(depth_map, np.ndarray):
            self.history.append(depth_map)
            if len(self.history) > 0:
                # Calculate mean only if history is not empty
                # Use float64 for accumulation to avoid precision issues? Might be overkill.
                smoothed = np.mean(self.history, axis=0, dtype=depth_map.dtype)
                return smoothed
            else:
                # Should not happen if input is valid, but return input as fallback
                return depth_map
        # Return None if input is None or invalid
        return None

# --- Modified estimate_depth to include smoothing ---
def estimate_depth(frame, midas_model, transform, device, smoother):
    """Estimates depth and applies temporal smoothing."""
    # 1. Get raw depth estimate for the current frame
    raw_depth = _estimate_depth_raw(frame, midas_model, transform, device)
    # 2. Apply temporal smoothing using the smoother object
    smoothed_depth = smoother.smooth(raw_depth)
    return smoothed_depth

def _estimate_depth_raw(frame, midas_model, transform, device):
    """Internal function to get single-frame depth map from MiDaS."""
    global MIDAS_AVAILABLE # Allow modification if error occurs
    if not MIDAS_AVAILABLE or midas_model is None or transform is None:
        return None
    try:
        # Remember original frame size
        original_height, original_width = frame.shape[:2]

        # Convert BGR to RGB (MiDaS expects RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform and move to device
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            # Run inference
            prediction = midas_model(input_batch)

            # Resize prediction to original frame size using bicubic interpolation
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic", # Use bicubic for smoother depth maps
                align_corners=False,
            ).squeeze() # Remove batch and channel dimensions

        # Move depth map to CPU and convert to NumPy array
        depth_map_np = prediction.cpu().numpy()
        return depth_map_np

    except Exception as e:
        print(f"Error in _estimate_depth_raw: {e}")
        # traceback.print_exc() # Uncomment for detailed debugging
        return None
# --- End MiDaS Depth Estimation Function ---

# --- Adaptive Thresholding Function (Used by 'threshold' mode logic below) ---
def adaptive_threshold(depth_map, mean_multiplier=1.0):
    """
    Applies adaptive thresholding based on depth.
    Returns a binary mask (0 or 1).
    NOTE: Assumes MiDaS output where higher values are CLOSER.
    Therefore, mask selects pixels CLOSER than threshold (depth > threshold).
    """
    if depth_map is None:
        print("Warning: Cannot apply adaptive threshold to None depth map.")
        return None
    try:
        valid_depths = depth_map[np.isfinite(depth_map)]
        if valid_depths.size == 0:
             print("Warning: Depth map contains no finite values for thresholding.")
             return np.zeros_like(depth_map, dtype=np.uint8)

        mean_depth = np.mean(valid_depths)
        threshold_value = mean_depth * mean_multiplier
        # Select pixels *closer* than the threshold (higher depth value)
        foreground_mask = depth_map > threshold_value
        return foreground_mask.astype(np.uint8)
    except Exception as e:
        print(f"Error during adaptive thresholding: {e}")
        return None

# --- Hierarchical Segmentation Function ---
def hierarchical_segmentation(depth_map, depth_ranges):
    """
    Segments depth map into multiple masks based on defined ranges.
    Assumes depth_map values and ranges are comparable (e.g., normalized 0-1).
    """
    if depth_map is None or not depth_ranges: # Check if ranges list is provided and not empty
        print("Warning: Cannot perform hierarchical segmentation without depth map or ranges.")
        return None
    masks = []
    # Ensure depth_map is float for comparison
    depth_map_float = depth_map.astype(np.float32)

    print(f"Generating masks for ranges: {depth_ranges}") # Debug print
    for i, (min_depth, max_depth) in enumerate(depth_ranges):
        # Create mask for pixels within the current range [min_depth, max_depth]
        # Note: >= and <= includes boundaries. Adjust if exclusive ranges are needed.
        mask = (depth_map_float >= min_depth) & (depth_map_float <= max_depth)
        masks.append(mask.astype(np.uint8))
        print(f"  Range {i} ({min_depth}-{max_depth}): Found {np.sum(mask)} pixels.") # Debug print
    return masks # Returns a list of numpy masks


# --- Modified Compositing Function with Depth ---
def composite_frames_torch(fg_frame, alpha_frame, bg_frame, depth_map, device,
                           use_depth=False, depth_mode='adjust', depth_influence=0.5,
                           depth_thresh_multiplier=1.0, depth_ranges=None):
    """
    Composites fg over bg using alpha matte with PyTorch, optionally incorporating depth.
    Includes 'adjust', 'threshold', and 'hierarchical' depth modes.
    """
    if fg_frame is None or alpha_frame is None or bg_frame is None:
        caller_info = traceback.extract_stack(limit=2)[0]
        print(f"Error in {caller_info.name}: Received None frame(s) for compositing.")
        raise ValueError("Received None frame(s) for compositing.")

    try:
        # --- Input Validation and Conversion ---
        if not isinstance(fg_frame, np.ndarray): fg_frame = np.array(fg_frame)
        if not isinstance(alpha_frame, np.ndarray): alpha_frame = np.array(alpha_frame)
        if not isinstance(bg_frame, np.ndarray): bg_frame = np.array(bg_frame)
        if fg_frame.size == 0 or alpha_frame.size == 0 or bg_frame.size == 0:
             raise ValueError(f"Received empty frame array. FG:{fg_frame.shape}, Alpha:{alpha_frame.shape}, BG:{bg_frame.shape}")

        if alpha_frame.ndim == 3:
            if alpha_frame.shape[2] == 3: alpha_frame = cv2.cvtColor(alpha_frame, cv2.COLOR_BGR2GRAY)
            elif alpha_frame.shape[2] == 1: alpha_frame = alpha_frame.squeeze(axis=-1)
            else: raise ValueError(f"Unexpected alpha channels: {alpha_frame.shape[2]}")
        elif alpha_frame.ndim != 2: raise ValueError(f"Unexpected alpha dimensions: {alpha_frame.ndim}")

        # Convert to PyTorch Tensors (HWC, float32, 0-1 range)
        fg = torch.from_numpy(fg_frame).to(device, non_blocking=True).float().div_(255.0)
        bg = torch.from_numpy(bg_frame).to(device, non_blocking=True).float().div_(255.0)
        # RVM Alpha (Grayscale H,W -> H,W,C=1)
        rvm_alpha = torch.from_numpy(alpha_frame).to(device, non_blocking=True).float().div_(255.0).unsqueeze_(-1)

        # --- Dimension Check & Resize (Crucial!) ---
        h_fg, w_fg = fg.shape[:2]
        h_alpha, w_alpha = rvm_alpha.shape[:2]
        h_bg, w_bg = bg.shape[:2]

        if not (h_fg == h_alpha and w_fg == w_alpha):
             print(f"Warning: Resizing RVM Alpha ({h_alpha}x{w_alpha}) to match FG ({h_fg}x{w_fg}).")
             rvm_alpha = torch.nn.functional.interpolate(rvm_alpha.permute(2, 0, 1).unsqueeze(0), size=(h_fg, w_fg), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        if not (h_fg == h_bg and w_fg == w_bg):
             print(f"Warning: Resizing BG ({h_bg}x{w_bg}) to match FG ({h_fg}x{w_fg}).")
             bg = torch.nn.functional.interpolate(bg.permute(2, 0, 1).unsqueeze(0), size=(h_fg, w_fg), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

        # --- Depth-Aware Alpha Modification ---
        final_alpha = rvm_alpha # Start with the original RVM alpha by default

        if use_depth and depth_map is not None and isinstance(depth_map, np.ndarray) and depth_map.size > 0:
            # Depth map is available and valid numpy array
            try:
                # Convert depth map (numpy float32) to tensor (float32, HWC C=1) on device
                depth_tensor = torch.from_numpy(depth_map.astype(np.float32)).to(device, non_blocking=True).unsqueeze_(-1)

                # Resize depth tensor if needed (MUST match FG/Alpha/BG now)
                if depth_tensor.shape[0] != h_fg or depth_tensor.shape[1] != w_fg:
                    print(f"Warning: Resizing depth map tensor ({depth_tensor.shape[0]}x{depth_tensor.shape[1]}) to match FG ({h_fg}x{w_fg}).")
                    depth_tensor = torch.nn.functional.interpolate(
                        depth_tensor.permute(2, 0, 1).unsqueeze(0), # HWC -> CHW -> NCHW
                        size=(h_fg, w_fg),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0) # NCHW -> CHW -> HWC

                # --- Apply Depth Mode Logic ---
                if depth_mode == 'adjust':
                    # Normalize depth map tensor (higher value = closer) to [0, 1]
                    depth_min = torch.min(depth_tensor)
                    depth_max = torch.max(depth_tensor)
                    # Add epsilon to prevent division by zero if depth map is flat
                    depth_normalized = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-6)

                    # Blend RVM alpha and normalized depth based on influence
                    # alpha * (1 - influence) + depth_normalized * influence
                    final_alpha = rvm_alpha * (1.0 - depth_influence) + depth_normalized * depth_influence
                    # print("Using depth 'adjust' mode (blend).") # Debug

                elif depth_mode == 'threshold':
                    # Calculate mask based on depth threshold (using NumPy version for simplicity)
                    # Note: adaptive_threshold returns 1 for CLOSER pixels (depth > threshold)
                    depth_mask_np = adaptive_threshold(depth_map, mean_multiplier=depth_thresh_multiplier)
                    if depth_mask_np is not None:
                        depth_mask = torch.from_numpy(depth_mask_np).to(device, non_blocking=True).float().unsqueeze_(-1)
                        # Ensure mask tensor is resized if necessary
                        if depth_mask.shape[0] != h_fg or depth_mask.shape[1] != w_fg:
                             depth_mask = torch.nn.functional.interpolate(
                                depth_mask.permute(2, 0, 1).unsqueeze(0), size=(h_fg, w_fg), mode='nearest'
                             ).squeeze(0).permute(1, 2, 0)
                        # Modulate RVM alpha by the depth mask
                        final_alpha = rvm_alpha * depth_mask
                        # print("Using depth 'threshold' mode (modulate).") # Debug
                    else:
                        print("Warning: Adaptive thresholding failed, using original RVM alpha.")
                        # final_alpha remains rvm_alpha

                elif depth_mode == 'hierarchical' and depth_ranges:
                    # Use the first defined depth range as a hard mask for foreground
                    # This replaces the RVM alpha entirely for pixels in this range.
                    # Assumes ranges correspond to MiDaS output (higher = closer)
                    min_depth, max_depth = depth_ranges[0]
                    # Create mask directly on tensor: 1 if within range, 0 otherwise
                    # Handle potential NaN/inf in depth tensor? Clamp might be safer.
                    foreground_mask = (depth_tensor >= min_depth) & (depth_tensor <= max_depth)
                    final_alpha = foreground_mask.float() # Convert boolean mask to float (0.0 or 1.0)
                    # print(f"Using depth 'hierarchical' mode (masking with range {depth_ranges[0]}).") # Debug

                else: # Unknown mode or hierarchical mode requested without ranges
                    if depth_mode == 'hierarchical':
                        print("Warning: Depth mode 'hierarchical' selected but no depth ranges provided. Using original RVM alpha.")
                    else:
                        print(f"Warning: Unknown depth mode '{depth_mode}', using original RVM alpha.")
                    # final_alpha remains rvm_alpha

            except Exception as depth_proc_e:
                print(f"Error processing depth map: {depth_proc_e}. Using original RVM alpha.")
                traceback.print_exc()
                # final_alpha remains rvm_alpha
        # else: # Depth not used or depth_map is None/invalid
            # final_alpha remains rvm_alpha

        # --- Compositing Formula ---
        # Clamp final_alpha just in case blending/ops pushed it outside [0, 1]
        final_alpha = torch.clamp(final_alpha, 0.0, 1.0)
        comp = fg * final_alpha + bg * (1.0 - final_alpha)

        # Clamp final color values and convert back to numpy uint8
        comp = torch.clamp(comp, 0.0, 1.0)
        return (comp.cpu().numpy() * 255.0).astype(np.uint8)

    except Exception as e:
        print(f"\n!!! Error in composite_frames_torch: {e} !!!")
        print(f"Inputs - FG:{fg_frame.shape if fg_frame is not None else 'None'}, Alpha:{alpha_frame.shape if alpha_frame is not None else 'None'}, BG:{bg_frame.shape if bg_frame is not None else 'None'}")
        print(f"Depth Map Shape: {depth_map.shape if depth_map is not None else 'None'}")
        print(f"Device:{device}, UseDepth:{use_depth}, Mode:{depth_mode}, Ranges:{depth_ranges}")
        traceback.print_exc()
        raise
# --- End Modified Compositing Function ---


def detect_gst_element(name):
    """Checks if a GStreamer element factory exists."""
    if not GSTREAMER_AVAILABLE: return False
    try: return Gst.ElementFactory.find(name) is not None
    except Exception as e: print(f"Error checking GStreamer element '{name}': {e}"); return False
# --- End Utility Functions ---


# --- Core RVM Function ---
def generate_foreground_and_alpha(original_path, foreground_output_path, alpha_output_path, model_path, model_variant='mobilenetv3'):
    """Uses RobustVideoMatting to generate foreground and alpha matte videos."""
    global RVM_DEVICE # Use the globally determined device
    if not RVM_AVAILABLE:
        print("Error: RVM library not available or failed to import. Cannot generate mattes.")
        return False
    if not os.path.exists(original_path):
        print(f"Error: Input video for RVM not found: {original_path}")
        return False
    if not os.path.exists(model_path):
        print(f"Error: RVM model file not found: {model_path}")
        return False

    device = RVM_DEVICE
    print(f"\n--- Starting RVM Matte Generation ---")
    print(f"Using RVM device: {device}")
    print(f"Loading RVM model '{model_variant}' from: {model_path}")

    model = None
    try:
        model = MattingNetwork(model_variant).eval()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"RVM Model successfully loaded and moved to {device}.")
    except Exception as e:
        print(f"Error loading/moving RVM model: {e}")
        traceback.print_exc()
        if 'cuda' in str(device) or 'mps' in str(device):
             print("GPU model loading failed. Attempting to load on CPU...")
             device = torch.device('cpu')
             try:
                 model = MattingNetwork(model_variant).eval()
                 model.load_state_dict(torch.load(model_path, map_location=device))
                 print("RVM Model successfully loaded on CPU.")
                 RVM_DEVICE = device # Update global device if we fell back
             except Exception as cpu_e:
                 print(f"Failed to load RVM model even on CPU: {cpu_e}")
                 return False
        else: return False # Already CPU and failed

    print(f"\nStarting RVM conversion process:")
    print(f"  Input Video:      {original_path}")
    print(f"  Output Foreground: {foreground_output_path}")
    print(f"  Output Alpha:     {alpha_output_path}")

    try:
        num_workers = max(0, (os.cpu_count() or 1) // 2)
        print(f"Using num_workers={num_workers} for RVM dataloader.")

        common_args = {
            'model': model,
            'input_source': original_path,
            'output_type': 'video',
            'output_video_mbps': 12,
            'downsample_ratio': None,
            'device': str(RVM_DEVICE),
            'seq_chunk': 4,
            'num_workers': num_workers
        }

        print("Attempting RVM conversion with 'output_alpha' and 'output_foreground'...")
        try:
            convert_video(
                **common_args,
                output_alpha=alpha_output_path,
                output_foreground=foreground_output_path
            )
            print("RVM conversion using 'output_alpha' and 'output_foreground' successful.")
            return True
        except TypeError as te_fg_alpha:
            print(f"TypeError with 'output_alpha/foreground': {te_fg_alpha}. Trying alternatives...")
            if 'output_alpha' in str(te_fg_alpha) or 'output_foreground' in str(te_fg_alpha) or 'unexpected keyword argument' in str(te_fg_alpha):
                 try:
                     print("Retrying RVM conversion with 'output_mask' and 'output_composition'...")
                     convert_video(
                         **common_args,
                         output_mask=alpha_output_path,
                         output_composition=foreground_output_path # FG * Alpha
                     )
                     print("RVM conversion using 'output_mask' and 'output_composition' successful.")
                     print("WARNING: Using 'output_composition'. The foreground file contains pre-multiplied alpha!")
                     print("         Compositing logic assumes non-premultiplied FG. Results may be incorrect.")
                     return False # Force failure as compositing expects raw FG
                 except TypeError as te_mask_comp:
                      print(f"TypeError with 'output_mask/composition': {te_mask_comp}. Trying mask only...")
                      if 'output_composition' in str(te_mask_comp):
                           try:
                               print("Retrying RVM conversion with 'output_mask' only...")
                               convert_video(
                                   **common_args,
                                   output_mask=alpha_output_path
                               )
                               print("RVM conversion using 'output_mask' (mask only) successful.")
                               print(f"CRITICAL ERROR: RVM only generated alpha mask. Foreground ({foreground_output_path}) is missing.")
                               return False
                           except Exception as e_mask_only:
                                print(f"Error during RVM conversion (mask only attempt): {e_mask_only}")
                                traceback.print_exc(); return False
                      else:
                           print(f"Unhandled TypeError with 'output_mask/composition': {te_mask_comp}"); traceback.print_exc(); return False
                 except Exception as e_mask:
                     print(f"Error during RVM conversion ('output_mask/composition' attempt): {e_mask}"); traceback.print_exc(); return False
            else:
                print(f"Unhandled TypeError during RVM conversion: {te_fg_alpha}"); traceback.print_exc(); return False
        except Exception as e_alpha:
            print(f"Error during RVM conversion ('output_alpha/foreground' attempt): {e_alpha}")
            if 'CUDA out of memory' in str(e_alpha) or 'mps' in str(e_alpha): print("\n *** RVM Out of Memory! Try reducing 'seq_chunk' or using CPU. ***\n")
            elif 'int' in str(e_alpha) and 'float' in str(e_alpha) and 'rate' in str(e_alpha): print("\n *** This might be the RVM 'rate' parameter bug. Check patches/RVM version. ***\n")
            traceback.print_exc(); return False

    except Exception as e:
        print(f"General error during RVM processing: {e}")
        traceback.print_exc()
        return False
    finally:
        del model
        if RVM_DEVICE.type == 'cuda': torch.cuda.empty_cache()
        elif RVM_DEVICE.type == 'mps': pass # MPS cache managed automatically
        print("RVM resources released.")
# --- End Core RVM Function ---

# --- Implementation Backends ---

def get_video_info(path):
    """Gets video properties using OpenCV. Returns None on failure."""
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file with OpenCV: {path}")
            return None
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        if info['width'] <= 0 or info['height'] <= 0:
             print(f"Warning: Invalid dimensions reported for {path}: {info['width']}x{info['height']}")
             return None
        if info['fps'] <= 0: print(f"Warning: Invalid FPS ({info['fps']}) reported for {path}. Using 30 as fallback.") # Provide a fallback
        if info['frame_count'] <= 0: print(f"Warning: Frame count reported as {info['frame_count']} for {path}. Will read until end.")

        fourcc_int = info['codec']
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        info['codec_str'] = fourcc_str
        # Use fallback FPS if needed
        info['fps'] = info['fps'] if info['fps'] > 0 else 30.0

        print(f"  Info for {os.path.basename(path)}: {info['width']}x{info['height']} @ {info['fps']:.2f} FPS, {info['frame_count']} frames, Codec: {fourcc_str}")
        return info
    except Exception as e:
        print(f"Error getting video info for {path}: {e}")
        return None
    finally:
        if cap is not None: cap.release()

def setup_compositing(foreground_path, alpha_path, background_path):
    """Reads video properties, validates, and prepares background frames. Returns None on fatal error."""
    print("--- Setting up compositing inputs ---")
    try:
        print("Getting video info...")
        fg_info = get_video_info(foreground_path)
        alpha_info = get_video_info(alpha_path)
        bg_info = get_video_info(background_path)

        if not fg_info or not alpha_info or not bg_info:
            print("Error: Failed to get required video information.")
            return None

        # --- Validation ---
        if fg_info['width'] != alpha_info['width'] or fg_info['height'] != alpha_info['height']:
            print(f"FATAL ERROR: FG ({fg_info['width']}x{fg_info['height']}) and Alpha ({alpha_info['width']}x{alpha_info['height']}) dimensions mismatch!")
            return None
        target_width, target_height = fg_info['width'], fg_info['height']

        target_fps = fg_info['fps'] # Already has fallback to 30 if needed
        if abs(fg_info['fps'] - alpha_info['fps']) > 1.0:
            print(f"Warning: FG FPS ({fg_info['fps']:.2f}) and Alpha FPS ({alpha_info['fps']:.2f}) differ significantly. Using FG FPS ({target_fps:.2f}).")

        target_frame_count = 0
        if fg_info['frame_count'] > 0 and alpha_info['frame_count'] > 0:
            if abs(fg_info['frame_count'] - alpha_info['frame_count']) > 5:
                 print(f"Warning: FG ({fg_info['frame_count']}) and Alpha ({alpha_info['frame_count']}) frame counts differ noticeably.")
            target_frame_count = min(fg_info['frame_count'], alpha_info['frame_count'])
            print(f"Using minimum frame count for processing limit: {target_frame_count}")
        elif fg_info['frame_count'] > 0:
            target_frame_count = fg_info['frame_count']; print(f"Using FG frame count for limit: {target_frame_count}")
        elif alpha_info['frame_count'] > 0:
            target_frame_count = alpha_info['frame_count']; print(f"Using Alpha frame count for limit: {target_frame_count}")
        else:
            print("Warning: Frame count unknown for FG/Alpha. Will attempt to read until end of streams.")
            target_frame_count = -1 # Indicate unknown limit

        # --- Background Processing ---
        bg_fps = bg_info['fps'] if bg_info['fps'] > 0 else target_fps # Use target_fps as fallback
        if bg_info['fps'] <= 0: print(f"Warning: Invalid BG FPS. Using target FPS {target_fps:.2f} for background timing.")

        bg_duration = 0
        if bg_info['frame_count'] > 0 and bg_fps > 0:
            bg_duration = bg_info['frame_count'] / bg_fps
            print(f"Background duration: {bg_duration:.2f}s ({bg_info['frame_count']} frames @ {bg_fps:.2f} FPS)")
        else: print("Warning: Cannot determine background duration accurately from info.")

        print("Loading and potentially resizing background frames...")
        bg_cap = None; bg_frames = None
        try:
            bg_cap = cv2.VideoCapture(background_path)
            if not bg_cap.isOpened():
                print(f"Error: Cannot open background video with OpenCV: {background_path}")
                return None

            needs_resize = (bg_info['width'] != target_width or bg_info['height'] != target_height)
            if needs_resize:
                print(f"Resizing background video from {bg_info['width']}x{bg_info['height']} to {target_width}x{target_height}...")
                bg_frames = resize_video_frames(bg_cap, target_width, target_height) # Uses helper function
                if bg_frames is None or len(bg_frames) == 0:
                    print("Error: Failed to read or resize background frames.")
                    return None
            else:
                print("Reading background frames (no resize needed)...")
                bg_frames_list = []
                frame_count_bg = 0
                while True:
                    ret, frame = bg_cap.read()
                    if not ret: break
                    if frame is None: print(f"Warning: Read empty frame from background (frame {frame_count_bg})."); frame_count_bg += 1; continue
                    bg_frames_list.append(frame); frame_count_bg += 1
                bg_frames = np.array(bg_frames_list)
                if bg_frames.size == 0:
                    print("Error: Failed to read any frames from the background video.")
                    return None

            actual_bg_frame_count = len(bg_frames)
            print(f"Loaded {actual_bg_frame_count} background frames into memory.")
            if actual_bg_frame_count == 0: return None # Should be caught above, but double check
            if bg_fps > 0:
                bg_duration = actual_bg_frame_count / bg_fps
                print(f"Using actual background duration: {bg_duration:.2f}s")
            else: bg_duration = 0; print("Warning: BG duration remains unknown due to invalid FPS.")
            bg_info['frame_count'] = actual_bg_frame_count

        except Exception as e:
            print(f"Error during background video processing: {e}"); traceback.print_exc(); return None
        finally:
            if bg_cap is not None: bg_cap.release()

        print("--- Setup complete ---")
        return (fg_info, bg_info, bg_frames, target_width, target_height,
                target_fps, target_frame_count, bg_duration, bg_fps)

    except Exception as setup_e:
        print(f"Critical error during setup_compositing: {setup_e}"); traceback.print_exc(); return None

def map_background_frame(current_frame_index, original_fps, background_duration_sec, background_fps):
    """Calculates the corresponding background frame index, looping the background."""
    if original_fps <= 0 or background_fps <= 0: return 0
    current_time_sec = current_frame_index / original_fps
    if background_duration_sec <= 0: # Duration unknown, simple time map
        frame_index_b = int(current_time_sec * background_fps)
    else: # Duration known, use modulo for looping
        time_in_background_loop = current_time_sec % background_duration_sec
        frame_index_b = int(time_in_background_loop * background_fps)
    return frame_index_b # Modulo by len(bg_frames) applied by caller

def resize_video_frames(cap, target_width, target_height):
    """Reads all frames from a cv2.VideoCapture and resizes them."""
    frames = []
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame is None: print(f"Warning: Read empty frame (frame {frame_count}) during resize."); frame_count += 1; continue
        try:
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                 inter_method = cv2.INTER_AREA if (frame.shape[1] > target_width or frame.shape[0] > target_height) else cv2.INTER_LINEAR
                 resized = cv2.resize(frame, (target_width, target_height), interpolation=inter_method)
                 frames.append(resized)
            else: print(f"Warning: Skipping invalid frame (shape {frame.shape}) at index {frame_count}.")
        except cv2.error as e: print(f"Error resizing frame {frame_count}: {e}. Skipping.")
        except Exception as e: print(f"Unexpected error resizing frame {frame_count}: {e}. Skipping.")
        frame_count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
    if not frames: print("Error: No frames could be read/resized from the video capture."); return None
    return np.array(frames)

# --- FFmpeg Implementation (Updated for smoother and depth ranges) ---
def ffmpeg_implementation(foreground_path, alpha_path, background_path, original_path, output_path, mix_background_audio=False):
    print("\n--- Running FFmpeg Implementation ---")
    temp_video_path = os.path.join(OUTPUT_DIR, f"temp_{os.path.splitext(os.path.basename(output_path))[0]}_ffmpeg.mp4")
    global COMPOSITE_DEVICE, USE_DEPTH, DEPTH_MODE, DEPTH_ADJUST_INFLUENCE, DEPTH_THRESHOLD_MEAN_MULTIPLIER, DEPTH_RANGES, SMOOTH_WINDOW
    global midas, midas_transform, MIDAS_AVAILABLE
    device = COMPOSITE_DEVICE
    fg_cap = None; alpha_cap = None; out_writer = None; setup_result = None; bg_frames = None

    try:
        # --- Setup ---
        setup_result = setup_compositing(foreground_path, alpha_path, background_path)
        if setup_result is None: raise RuntimeError("Compositing setup failed.")
        fg_info, bg_info, bg_frames, width, height, fps, frame_count_limit, bg_duration, bg_fps = setup_result

        # --- Initialize Smoother ---
        smoother = DepthSmoother(window_size=SMOOTH_WINDOW)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not out_writer.isOpened(): raise IOError(f"Could not open OpenCV VideoWriter for temporary file: {temp_video_path}")

        print(f"Compositing frames (Target: {frame_count_limit if frame_count_limit > 0 else 'read till end'})...")
        print(f"Depth Enabled: {USE_DEPTH and MIDAS_AVAILABLE}, Mode: {DEPTH_MODE if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}, Smoothing: {SMOOTH_WINDOW if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}")
        fg_cap = cv2.VideoCapture(foreground_path)
        alpha_cap = cv2.VideoCapture(alpha_path)
        if not fg_cap.isOpened(): raise IOError(f"Cannot open foreground matte: {foreground_path}")
        if not alpha_cap.isOpened(): raise IOError(f"Cannot open alpha matte: {alpha_path}")

        processed_count = 0
        i = 0
        while True:
            if 0 < frame_count_limit <= i:
                 print(f"\nReached target frame count limit ({frame_count_limit}). Stopping.")
                 break

            ret_fg, fg_frame = fg_cap.read()
            ret_alpha, alpha_frame = alpha_cap.read()

            if not ret_fg or not ret_alpha:
                if frame_count_limit <= 0: print(f"\nInput stream ended at frame {i}. Stopping.")
                else: print(f"\nWarning: Read failed at frame {i} (expected {frame_count_limit}). Stopping.")
                break
            if fg_frame is None or alpha_frame is None:
                print(f"\nWarning: Read None frame at index {i}. Stopping.")
                break

            # --- Estimate Depth (with smoothing) ---
            depth_map = None
            if USE_DEPTH and MIDAS_AVAILABLE:
                # Pass the smoother instance to estimate_depth
                depth_map = estimate_depth(fg_frame, midas, midas_transform, device, smoother)

            # --- Get Background Frame ---
            if len(bg_frames) == 0: raise RuntimeError("Background frames array is empty!")
            bg_idx = map_background_frame(i, fps, bg_duration, bg_fps) % len(bg_frames)
            bg_frame = bg_frames[bg_idx]

            # --- Composite (passing depth ranges) ---
            try:
                comp_frame = composite_frames_torch(fg_frame, alpha_frame, bg_frame, depth_map, device,
                                                    use_depth=(USE_DEPTH and MIDAS_AVAILABLE),
                                                    depth_mode=DEPTH_MODE,
                                                    depth_influence=DEPTH_ADJUST_INFLUENCE,
                                                    depth_thresh_multiplier=DEPTH_THRESHOLD_MEAN_MULTIPLIER,
                                                    depth_ranges=DEPTH_RANGES) # Pass ranges
                out_writer.write(comp_frame)
                processed_count += 1
            except Exception as comp_e:
                print(f"\nError during compositing frame {i}: {comp_e}. Stopping.")
                traceback.print_exc(); break

            if (i + 1) % 100 == 0: print(f"  Processed {i + 1} frames...", end='\r')
            i += 1
        # End of frame loop

        print(f"\nFinished compositing {processed_count} frames.")
        if processed_count == 0: raise RuntimeError("No frames were successfully composited.")

        print("Releasing video captures and writer...")
        if fg_cap: fg_cap.release()
        if alpha_cap: alpha_cap.release()
        if out_writer: out_writer.release()
        fg_cap, alpha_cap, out_writer = None, None, None

        # --- Audio Muxing using FFmpeg (Robust version - unchanged) ---
        print("Checking audio streams & muxing with FFmpeg...")
        # (Exact same FFmpeg audio muxing logic as before)
        inputs = []; input_files = []; filter_complex_parts = []; audio_map_labels = []
        video_maps = ["-map", "0:v"]
        inputs.extend(["-i", temp_video_path]); input_files.append(temp_video_path)

        def check_audio_stream(filepath):
            if not filepath or not os.path.exists(filepath): return False
            ffprobe_cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=index", "-select_streams", "a", "-of", "csv=p=0", filepath]
            try:
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False, timeout=15)
                has_audio = result.returncode == 0 and result.stdout.strip() != ""
                if not has_audio and result.stderr: print(f"  ffprobe check for {os.path.basename(filepath)} stderr: {result.stderr.strip()}")
                return has_audio
            except FileNotFoundError: print("Error: ffprobe command not found."); return False
            except subprocess.TimeoutExpired: print(f"Warning: ffprobe timed out for {os.path.basename(filepath)}."); return False
            except Exception as probe_e: print(f"Error running ffprobe for {os.path.basename(filepath)}: {probe_e}"); return False

        orig_has_audio = check_audio_stream(original_path)
        if orig_has_audio:
            print("  Original video has audio stream.")
            input_index = len(input_files)
            inputs.extend(["-i", original_path]); input_files.append(original_path)
            audio_map_labels.append(f"[{input_index}:a]")
        else: print("  Original video has no audio stream.")

        bg_has_audio = False
        if mix_background_audio:
            bg_has_audio = check_audio_stream(background_path)
            if bg_has_audio:
                print("  Background video has audio stream (mixing enabled).")
                input_index = len(input_files)
                inputs.extend(["-i", background_path]); input_files.append(background_path)
                bg_audio_label_in = f"[{input_index}:a]"; bg_audio_label_out = f"[a_bg{input_index}]"
                filter_complex_parts.append(f"{bg_audio_label_in}volume=0.5{bg_audio_label_out}")
                audio_map_labels.append(bg_audio_label_out)
            else: print("  Background video has no audio stream.")
        else: print("  Background audio mixing disabled.")

        ffmpeg_cmd_final = ["ffmpeg", "-y"] + inputs
        audio_maps_final = []
        if len(audio_map_labels) > 1:
            print(f"  Mixing {len(audio_map_labels)} audio streams.")
            mix_inputs = "".join(audio_map_labels)
            filter_complex_parts.append(f"{mix_inputs}amix=inputs={len(audio_map_labels)}:duration=shortest:dropout_transition=2[a_out]")
            audio_maps_final = ["-map", "[a_out]"]
        elif len(audio_map_labels) == 1:
            print("  Using single audio stream.")
            audio_maps_final = ["-map", audio_map_labels[0]]
        else: print("  No audio streams to include.")

        if filter_complex_parts: ffmpeg_cmd_final.extend(["-filter_complex", ";".join(filter_complex_parts)])
        ffmpeg_cmd_final.extend(video_maps)
        if audio_maps_final:
            ffmpeg_cmd_final.extend(audio_maps_final)
            ffmpeg_cmd_final.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        else: ffmpeg_cmd_final.extend(["-an"])
        ffmpeg_cmd_final.extend(["-c:v", "copy", "-movflags", "+faststart", output_path])

        print(f"Executing FFmpeg command for muxing:\n  {' '.join(ffmpeg_cmd_final)}")
        try:
            process = subprocess.run(ffmpeg_cmd_final, check=True, capture_output=True, text=True, timeout=600)
            print("FFmpeg muxing successful.")
            if process.stderr and "warnings" in process.stderr.lower(): print(f"FFmpeg stderr:\n{process.stderr}")
        except subprocess.CalledProcessError as e: print(f"\n!!! FFmpeg Muxing FAILED !!!\nRC: {e.returncode}\nCmd: {' '.join(e.cmd)}\nStderr:\n{e.stderr}"); raise RuntimeError("FFmpeg muxing failed.") from e
        except subprocess.TimeoutExpired as e: print(f"\n!!! FFmpeg Muxing Timed Out !!!\nCmd: {' '.join(e.cmd)}"); raise RuntimeError("FFmpeg muxing timed out.") from e

    except Exception as e:
        print(f"\n!!! Error in FFmpeg implementation: {e} !!!"); traceback.print_exc()
        if fg_cap and fg_cap.isOpened(): fg_cap.release()
        if alpha_cap and alpha_cap.isOpened(): alpha_cap.release()
        if out_writer and out_writer.isOpened(): out_writer.release()
        raise
    finally:
        if os.path.exists(temp_video_path):
            print(f"Removing temporary file: {temp_video_path}")
            try: os.remove(temp_video_path)
            except OSError as e: print(f"Warning: Could not remove temp file {temp_video_path}: {e}")
        del setup_result, bg_frames; import gc; gc.collect()

# --- GStreamer Implementation (Updated for smoother and depth ranges) ---
def gstreamer_implementation(foreground_path, alpha_path, background_path, original_path, output_path, mix_background_audio=False):
    if not GSTREAMER_AVAILABLE: print("\n--- Skipping GStreamer Implementation (Not Available) ---"); return
    print("\n--- Running GStreamer Implementation ---")
    temp_video_path = os.path.join(OUTPUT_DIR, f"temp_{os.path.splitext(os.path.basename(output_path))[0]}_gst.mp4")
    global COMPOSITE_DEVICE, USE_DEPTH, DEPTH_MODE, DEPTH_ADJUST_INFLUENCE, DEPTH_THRESHOLD_MEAN_MULTIPLIER, DEPTH_RANGES, SMOOTH_WINDOW
    global midas, midas_transform, MIDAS_AVAILABLE
    device = COMPOSITE_DEVICE
    pipeline = None; fg_cap = None; alpha_cap = None; out_writer = None; setup_result = None; bg_frames = None

    try:
        # --- Setup ---
        setup_result = setup_compositing(foreground_path, alpha_path, background_path)
        if setup_result is None: raise RuntimeError("Compositing setup failed.")
        fg_info, bg_info, bg_frames, width, height, fps, frame_count_limit, bg_duration, bg_fps = setup_result

        # --- Initialize Smoother ---
        smoother = DepthSmoother(window_size=SMOOTH_WINDOW)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not out_writer.isOpened(): raise IOError(f"Could not open OpenCV VideoWriter for GST temp file: {temp_video_path}")

        print(f"Compositing frames (Target: {frame_count_limit if frame_count_limit > 0 else 'read till end'})...")
        print(f"Depth Enabled: {USE_DEPTH and MIDAS_AVAILABLE}, Mode: {DEPTH_MODE if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}, Smoothing: {SMOOTH_WINDOW if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}")
        fg_cap = cv2.VideoCapture(foreground_path)
        alpha_cap = cv2.VideoCapture(alpha_path)
        if not fg_cap.isOpened(): raise IOError(f"Cannot open foreground matte: {foreground_path}")
        if not alpha_cap.isOpened(): raise IOError(f"Cannot open alpha matte: {alpha_path}")

        processed_count = 0
        i = 0
        while True:
            if 0 < frame_count_limit <= i: print(f"\nReached target frame count limit ({frame_count_limit}). Stopping."); break
            ret_fg, fg_frame = fg_cap.read(); ret_alpha, alpha_frame = alpha_cap.read()
            if not ret_fg or not ret_alpha:
                if frame_count_limit <= 0: print(f"\nInput stream ended at frame {i}. Stopping.")
                else: print(f"\nWarning: Read failed at frame {i} (expected {frame_count_limit}). Stopping.")
                break
            if fg_frame is None or alpha_frame is None: print(f"\nWarning: Read None frame at index {i}. Stopping."); break

            # --- Estimate Depth (with smoothing) ---
            depth_map = None
            if USE_DEPTH and MIDAS_AVAILABLE:
                depth_map = estimate_depth(fg_frame, midas, midas_transform, device, smoother)

            # --- Get Background Frame ---
            if len(bg_frames) == 0: raise RuntimeError("Background frames array is empty!")
            bg_idx = map_background_frame(i, fps, bg_duration, bg_fps) % len(bg_frames)
            bg_frame = bg_frames[bg_idx]

            # --- Composite (passing depth ranges) ---
            try:
                comp_frame = composite_frames_torch(fg_frame, alpha_frame, bg_frame, depth_map, device,
                                                    use_depth=(USE_DEPTH and MIDAS_AVAILABLE),
                                                    depth_mode=DEPTH_MODE,
                                                    depth_influence=DEPTH_ADJUST_INFLUENCE,
                                                    depth_thresh_multiplier=DEPTH_THRESHOLD_MEAN_MULTIPLIER,
                                                    depth_ranges=DEPTH_RANGES) # Pass ranges
                out_writer.write(comp_frame)
                processed_count += 1
            except Exception as comp_e: print(f"\nError during compositing frame {i}: {comp_e}. Stopping."); traceback.print_exc(); break
            if (i + 1) % 100 == 0: print(f"  Processed {i + 1} frames...", end='\r')
            i += 1
        # End frame loop

        print(f"\nFinished compositing {processed_count} frames.")
        if processed_count == 0: raise RuntimeError("No frames were successfully composited.")

        print("Releasing video captures and writer...")
        if fg_cap: fg_cap.release(); fg_cap = None
        if alpha_cap: alpha_cap.release(); alpha_cap = None
        if out_writer: out_writer.release(); out_writer = None

        # --- GStreamer Pipeline Construction (for muxing - simplified version unchanged) ---
        print("Constructing simplified GStreamer pipeline (copy video, add original audio)...")
        audio_encoder = 'avenc_aac' # Assume default
        if not detect_gst_element(audio_encoder):
             print(f"Warning: Preferred AAC encoder '{audio_encoder}' not found. Trying 'faac'.")
             audio_encoder = 'faac'
             if not detect_gst_element(audio_encoder):
                 print(f"Warning: Fallback AAC encoder 'faac' also not found. Audio encoding might fail.")
                 audio_encoder = None # Indicate failure

        safe_temp_path = temp_video_path.replace('"', '\\"'); safe_original_path = original_path.replace('"', '\\"'); safe_output_path = output_path.replace('"', '\\"')
        pipeline_str_simple = (
            f"filesrc location=\"{safe_temp_path}\" ! qtdemux name=vidsrc "
            f"filesrc location=\"{safe_original_path}\" ! decodebin name=audsrc "
            f"mp4mux name=mux ! filesink location=\"{safe_output_path}\" "
            f"vidsrc.video_0 ! queue ! mux.video_0 " # Copy video stream
            f"{f'audsrc. ! audioconvert ! audioresample ! queue ! {audio_encoder} ! queue ! mux.audio_0' if audio_encoder else ''}" # Add audio path if encoder exists
        )

        print(f"\nSimplified GStreamer Pipeline String Attempt:\n{pipeline_str_simple}\n")
        print("Launching GStreamer pipeline...")
        # --- GStreamer Execution (unchanged) ---
        main_loop = GLib.MainLoop()
        pipeline = None; bus = None
        try:
            pipeline = Gst.parse_launch(pipeline_str_simple)
        except GLib.Error as e: print(f"FATAL: GStreamer pipeline parse error: {e}"); raise RuntimeError("Failed to parse GStreamer pipeline string.") from e
        bus = pipeline.get_bus(); bus.add_signal_watch()
        def on_message(bus, message):
            mtype = message.type
            if mtype == Gst.MessageType.EOS: print("\nGStreamer: End-of-Stream reached."); pipeline.set_state(Gst.State.NULL); main_loop.quit()
            elif mtype == Gst.MessageType.ERROR:
                err, debug_info = message.parse_error(); element_name = message.src.get_name() if message.src else "UnknownElement"
                print(f"\n!!! GStreamer Pipeline ERROR from element '{element_name}' !!!\n  Error: {err.message}\n  Debug: {debug_info or 'None'}"); pipeline.set_state(Gst.State.NULL); main_loop.quit()
            elif mtype == Gst.MessageType.WARNING:
                err, debug_info = message.parse_warning(); element_name = message.src.get_name() if message.src else "UnknownElement"
                print(f"\nGStreamer WARNING from '{element_name}': {err.message} (Debug: {debug_info or 'None'})")
            return True
        bus.connect("message", on_message)
        print("Setting GStreamer pipeline to PLAYING..."); ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE: print("!!! GStreamer: Failed to set pipeline to PLAYING state."); raise RuntimeError("GStreamer pipeline could not start.")
        print("Pipeline running... Waiting for EOS or Error."); main_loop.run()
        _, current_state, _ = pipeline.get_state(Gst.CLOCK_TIME_NONE); print(f"GStreamer pipeline stopped. Final state: {current_state.value_nick}")
        if current_state != Gst.State.NULL: print("Warning: Pipeline did not reach NULL state cleanly."); pipeline.set_state(Gst.State.NULL)
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024: print(f"Warning: Output file {output_path} missing or small after GStreamer run.")

    except Exception as e:
        print(f"\n!!! Error in GStreamer implementation: {e} !!!");
        if not isinstance(e, (RuntimeError, IOError, GLib.Error)): traceback.print_exc()
        if pipeline is not None: try: print("Attempting to stop errored GStreamer pipeline..."); pipeline.set_state(Gst.State.NULL); except Exception as cleanup_e: print(f"Error during GStreamer cleanup: {cleanup_e}")
        if main_loop and main_loop.is_running(): main_loop.quit()
        raise
    finally:
        if bus: bus.remove_signal_watch()
        if os.path.exists(temp_video_path): print(f"Removing temporary file: {temp_video_path}"); try: os.remove(temp_video_path); except OSError as e: print(f"Warning: Could not remove temp file {temp_video_path}: {e}")
        del setup_result, bg_frames; import gc; gc.collect()

# --- PyAV Implementation (Updated for smoother and depth ranges) ---
def pyav_implementation(foreground_path, alpha_path, background_path, original_path, output_path, mix_background_audio=False):
    if not AV_AVAILABLE: print("\n--- Skipping PyAV Implementation (Not Available) ---"); return
    print("\n--- Running PyAV Implementation ---")
    if mix_background_audio: print("Warning: PyAV background audio mixing not implemented. Using original audio only.")

    input_fg = None; input_alpha = None; input_orig = None; output_container = None
    global COMPOSITE_DEVICE, USE_DEPTH, DEPTH_MODE, DEPTH_ADJUST_INFLUENCE, DEPTH_THRESHOLD_MEAN_MULTIPLIER, DEPTH_RANGES, SMOOTH_WINDOW
    global midas, midas_transform, MIDAS_AVAILABLE
    device = COMPOSITE_DEVICE
    bg_frames = None; setup_result = None

    try:
        # --- Setup ---
        print("Setting up using common routine...")
        setup_result = setup_compositing(foreground_path, alpha_path, background_path)
        if setup_result is None: raise RuntimeError("Compositing setup failed.")
        fg_info, bg_info, bg_frames, width, height, fps, frame_count_limit, bg_duration, bg_fps = setup_result

        # --- Initialize Smoother ---
        smoother = DepthSmoother(window_size=SMOOTH_WINDOW)

        print("Opening PyAV containers...")
        try: input_fg = av.open(foreground_path, mode='r')
        except av.AVError as e: raise IOError(f"PyAV Error opening FG '{foreground_path}': {e}")
        try: input_alpha = av.open(alpha_path, mode='r')
        except av.AVError as e: raise IOError(f"PyAV Error opening Alpha '{alpha_path}': {e}")
        try: input_orig = av.open(original_path, mode='r')
        except av.AVError as e: print(f"Warning: PyAV Error opening Original '{original_path}' (audio source): {e}"); input_orig = None
        try: output_container = av.open(output_path, mode='w')
        except av.AVError as e: raise IOError(f"PyAV Error opening Output '{output_path}': {e}")

        # --- Stream Setup ---
        in_v_stream_fg = input_fg.streams.video[0]; in_v_stream_alpha = input_alpha.streams.video[0]
        in_v_stream_fg.thread_type = "AUTO"; in_v_stream_alpha.thread_type = "AUTO"

        print(f"Setting up output video stream (Codec: h264, Rate: {fps:.2f}, Size: {width}x{height})")
        out_v_stream = output_container.add_stream('libx264', rate=str(fps), options={'crf': '23', 'preset': 'fast'})
        out_v_stream.width = width; out_v_stream.height = height; out_v_stream.pix_fmt = 'yuv420p'

        out_a_stream = None; in_a_stream_orig = None
        if input_orig and input_orig.streams.audio:
            in_a_stream_orig = input_orig.streams.audio[0]; in_a_stream_orig.thread_type = "AUTO"
            print(f"Found audio stream in original: Codec={in_a_stream_orig.codec.name}, Rate={in_a_stream_orig.rate}, Layout={in_a_stream_orig.layout}")
            try: out_a_stream = output_container.add_stream(template=in_a_stream_orig); print("  Added output audio stream using template.")
            except Exception as template_e:
                 print(f"  Warning: Adding audio stream with template failed: {template_e}. Trying explicit AAC.")
                 try:
                     rate = in_a_stream_orig.rate or 44100; layout = in_a_stream_orig.layout.name if in_a_stream_orig.layout else "stereo"
                     out_a_stream = output_container.add_stream("aac", rate=rate, layout=layout); out_a_stream.bit_rate = 192000
                     print(f"  Added output audio stream using AAC (Rate: {rate}, Layout: {layout}).")
                 except Exception as aac_e: print(f"  Error adding explicit AAC stream: {aac_e}."); out_a_stream = None
        else: print("Warning: No audio stream found in original.")

        # --- Frame Processing Loop ---
        print(f"Compositing and encoding frames (Target: {frame_count_limit if frame_count_limit > 0 else 'read till end'})...")
        print(f"Depth Enabled: {USE_DEPTH and MIDAS_AVAILABLE}, Mode: {DEPTH_MODE if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}, Smoothing: {SMOOTH_WINDOW if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}")
        fg_frame_iterator = input_fg.decode(in_v_stream_fg)
        alpha_frame_iterator = input_alpha.decode(in_v_stream_alpha)

        processed_count = 0
        i = 0
        while True:
            if 0 < frame_count_limit <= i: print(f"\nReached target frame count limit ({frame_count_limit}). Stopping video processing."); break
            try:
                fg_av_frame = next(fg_frame_iterator); alpha_av_frame = next(alpha_frame_iterator)
                fg_frame_np = fg_av_frame.to_ndarray(format='bgr24'); alpha_frame_np = alpha_av_frame.to_ndarray(format='gray')
                if fg_frame_np is None or alpha_frame_np is None: print(f"\nWarning: Decoded None frame at index {i}. Stopping."); break

                # --- Estimate Depth (with smoothing) ---
                depth_map = None
                if USE_DEPTH and MIDAS_AVAILABLE:
                    depth_map = estimate_depth(fg_frame_np, midas, midas_transform, device, smoother)

                # --- Get Background Frame ---
                if len(bg_frames) == 0: raise RuntimeError("Background frames array is empty!")
                bg_idx = map_background_frame(i, fps, bg_duration, bg_fps) % len(bg_frames)
                bg_frame_np = bg_frames[bg_idx]

                # --- Composite (passing depth ranges) ---
                try:
                    comp_frame_np = composite_frames_torch(fg_frame_np, alpha_frame_np, bg_frame_np, depth_map, device,
                                                           use_depth=(USE_DEPTH and MIDAS_AVAILABLE),
                                                           depth_mode=DEPTH_MODE,
                                                           depth_influence=DEPTH_ADJUST_INFLUENCE,
                                                           depth_thresh_multiplier=DEPTH_THRESHOLD_MEAN_MULTIPLIER,
                                                           depth_ranges=DEPTH_RANGES) # Pass ranges
                except Exception as comp_e: print(f"\nError compositing frame {i}: {comp_e}. Stopping."); traceback.print_exc(); break

                # --- Encode ---
                comp_av_frame = av.VideoFrame.from_ndarray(comp_frame_np, format='bgr24')
                comp_av_frame.pts = fg_av_frame.pts
                try:
                    for packet in out_v_stream.encode(comp_av_frame): output_container.mux(packet)
                    processed_count += 1
                except av.AVError as encode_e: print(f"\nError encoding video frame {i} (PTS: {comp_av_frame.pts}): {encode_e}. Stopping."); break

                if (i + 1) % 100 == 0: print(f"  Processed {i + 1} video frames...", end='\r')
                i += 1
            except StopIteration:
                if frame_count_limit <= 0: print(f"\nInput video stream ended at frame {i}.")
                else: print(f"\nWarning: Input video stream ended prematurely at frame {i}.")
                break
            except av.AVError as decode_e: print(f"\nError decoding video frame {i}: {decode_e}. Stopping."); break
            except Exception as loop_e: print(f"\nUnexpected error during PyAV loop (frame {i}): {loop_e}"); traceback.print_exc(); break
        # End of video processing loop
        print(f"\nFinished processing {processed_count} video frames.")

        # --- Flush Video Encoder ---
        print("Flushing video encoder...");
        try:
            for packet in out_v_stream.encode(): output_container.mux(packet)
            print("Video flush complete.")
        except av.AVError as flush_e: print(f"Error flushing video encoder: {flush_e}")
        except Exception as e: print(f"Unexpected error during video flush: {e}")

        # --- Audio Muxing (Copying packets - unchanged) ---
        if out_a_stream and in_a_stream_orig:
            print("Muxing audio stream from original (packet copy)..."); processed_audio_packets = 0
            try:
                for packet in input_orig.demux(in_a_stream_orig):
                    if packet.dts is None: continue
                    packet.stream = out_a_stream
                    try: output_container.mux(packet); processed_audio_packets += 1
                    except av.AVError as mux_e: print(f"Warning: Audio packet mux error (PTS: {packet.pts}): {mux_e}. Skipping."); continue
            except av.AVError as demux_e: print(f"Error demuxing audio packets: {demux_e}")
            except Exception as audio_mux_e: print(f"Unexpected error during audio muxing loop: {audio_mux_e}")
            print(f"Finished muxing {processed_audio_packets} audio packets.")
            print("Flushing audio stream...");
            try:
                 for packet in out_a_stream.encode(): output_container.mux(packet)
                 print("Audio flush complete.")
            except av.AVError as flush_e: print(f"Error flushing audio stream: {flush_e}")
            except Exception as e: print(f"Unexpected error during audio flush: {e}")
        else: print("Skipping audio muxing.")

    except Exception as e:
        print(f"\n!!! Error in PyAV implementation: {e} !!!"); traceback.print_exc(); raise
    finally:
        print("Closing PyAV containers...")
        if input_fg: try: input_fg.close(); except Exception as e: print(f"Error closing FG input: {e}")
        if input_alpha: try: input_alpha.close(); except Exception as e: print(f"Error closing Alpha input: {e}")
        if input_orig: try: input_orig.close(); except Exception as e: print(f"Error closing Original input: {e}")
        if output_container: try: output_container.close(); except Exception as e: print(f"Error closing Output container: {e}")
        del setup_result, bg_frames; import gc; gc.collect()

# --- MoviePy Implementation (Does NOT use depth features) ---
def moviepy_implementation(foreground_path, alpha_path, background_path, original_path, output_path, mix_background_audio=False):
    if not MOVIEPY_AVAILABLE: print("\n--- Skipping MoviePy Implementation (Not Available) ---"); return
    print("\n--- Running MoviePy Implementation ---")
    print("!!! WARNING: MoviePy implementation does NOT support frame-by-frame depth modification or smoothing. Performing standard alpha compositing. !!!")
    # (Code remains unchanged from the previous version as depth/smoothing are not applicable here)
    fg_clip=None; alpha_clip_for_mask=None; mask=None; bg_clip=None; original_clip=None; comp_clip=None; final_audio=None
    temp_audio_filename = f"temp_moviepy_audio_{os.path.splitext(os.path.basename(output_path))[0]}.m4a"
    temp_audio_path = os.path.join(OUTPUT_DIR, temp_audio_filename)

    try:
        print("Loading MoviePy clips...")
        try: fg_clip = VideoFileClip(foreground_path, audio=False)
        except Exception as e: raise IOError(f"MoviePy failed to load FG '{foreground_path}': {e}")
        try: alpha_clip_for_mask = VideoFileClip(alpha_path, audio=False).fx(lambda c: c.rgb_to_gray())
        except Exception as e: raise IOError(f"MoviePy failed to load Alpha '{alpha_path}': {e}")
        try: bg_clip = VideoFileClip(background_path)
        except Exception as e: raise IOError(f"MoviePy failed to load BG '{background_path}': {e}")
        try: original_clip = VideoFileClip(original_path)
        except Exception as e: raise IOError(f"MoviePy failed to load Original '{original_path}': {e}")

        print("Determining target duration and resolution...")
        target_duration = original_clip.duration
        if target_duration is None or target_duration <= 0:
             target_duration = fg_clip.duration
             if target_duration is None or target_duration <= 0:
                  target_duration = alpha_clip_for_mask.duration
                  if target_duration is None or target_duration <= 0: raise ValueError("Cannot determine valid target duration.")
                  else: print(f"Using Alpha clip duration: {target_duration:.2f}s")
             else: print(f"Using Foreground clip duration: {target_duration:.2f}s")
        else: print(f"Using Original clip duration: {target_duration:.2f}s")
        target_width, target_height = fg_clip.w, fg_clip.h
        print(f"Target resolution: {target_width}x{target_height}")

        print("Preparing clips (duration, size, mask)...")
        fg_clip = fg_clip.set_duration(target_duration)
        alpha_clip_for_mask = alpha_clip_for_mask.set_duration(target_duration)
        original_clip = original_clip.set_duration(target_duration)
        mask = alpha_clip_for_mask.to_mask(ismask=True, channel='lum')

        if bg_clip.w != target_width or bg_clip.h != target_height:
            print(f"Resizing background clip..."); bg_clip = bg_clip.resize((target_width, target_height))
        if bg_clip.duration is None or bg_clip.duration <= 0.1:
             print("Warning: Background clip has invalid duration. Using loop."); bg_clip = loop(bg_clip, duration=target_duration)
        elif bg_clip.duration < target_duration:
            print(f"Looping background clip..."); bg_clip = loop(bg_clip, duration=target_duration)
        elif bg_clip.duration > target_duration:
            print(f"Trimming background clip..."); bg_clip = bg_clip.subclip(0, target_duration)
        else: bg_clip = bg_clip.set_duration(target_duration)

        print("Applying mask to foreground clip...")
        fg_clip = fg_clip.set_mask(mask)
        print("Compositing video clips...")
        comp_clip = CompositeVideoClip([bg_clip, fg_clip], size=(target_width, target_height)).set_duration(target_duration)

        print("Processing audio...")
        original_audio = original_clip.audio; background_audio = bg_clip.audio
        orig_audio_valid = original_audio is not None and original_audio.duration is not None and original_audio.duration > 0.01
        bg_audio_valid = background_audio is not None and background_audio.duration is not None and background_audio.duration > 0.01

        if mix_background_audio:
            if orig_audio_valid and bg_audio_valid:
                print("  Mixing original and background audio...")
                original_audio = original_audio.set_duration(target_duration)
                background_audio = background_audio.set_duration(target_duration).volumex(0.5)
                final_audio = CompositeAudioClip([original_audio, background_audio])
            elif orig_audio_valid: print("  Using original audio only."); final_audio = original_audio.set_duration(target_duration)
            elif bg_audio_valid: print("  Using background audio only."); final_audio = background_audio.set_duration(target_duration).volumex(0.5)
            else: print("  No valid audio found."); final_audio = None
        else:
            if orig_audio_valid: print("  Using original audio only."); final_audio = original_audio.set_duration(target_duration)
            else: print("  No valid original audio found."); final_audio = None

        if final_audio: print(f"  Setting final audio."); comp_clip = comp_clip.set_audio(final_audio)
        else: print("  No final audio track."); comp_clip = comp_clip.set_audio(None)

        print(f"Writing final video to: {output_path}")
        comp_clip.write_videofile(
            output_path, codec='libx264', audio_codec='aac', temp_audiofile=temp_audio_path, remove_temp=True,
            preset='fast', ffmpeg_params=['-crf', '23'], threads=max(1, (os.cpu_count() or 2) - 1), logger=None)
        print("MoviePy processing complete.")
    except Exception as e: print(f"\n!!! Error in MoviePy implementation: {e} !!!"); traceback.print_exc(); raise
    finally:
        print("Closing MoviePy clips..."); clips_to_close = [final_audio, comp_clip, fg_clip, mask, alpha_clip_for_mask, bg_clip, original_clip]
        for clip in clips_to_close:
            if clip and hasattr(clip, 'close'): try: clip.close(); except Exception as e_close: print(f"Error closing clip: {e_close}")
        if os.path.exists(temp_audio_path): try: os.remove(temp_audio_path); except OSError as clean_e: print(f"Warning: Could not remove {temp_audio_path}: {clean_e}")


# --- scikit-video + Pydub Implementation (Updated for smoother and depth ranges) ---
def scikit_video_pydub_implementation(foreground_path, alpha_path, background_path, original_path, output_path, mix_background_audio=False):
    if not SKVIDEO_AVAILABLE: print("\n--- Skipping scikit-video Implementation (Not Available) ---"); return
    print("\n--- Running scikit-video + Pydub Implementation ---")
    print("!!! WARNING: skvideo.io.vread loads entire videos into memory. May fail for large files. !!!")
    if not PYDUB_AVAILABLE: print("Warning: Pydub not available, audio processing will be skipped.")

    temp_video_path = os.path.join(OUTPUT_DIR, f"temp_{os.path.splitext(os.path.basename(output_path))[0]}_skvideo.mp4")
    temp_audio_path = os.path.join(OUTPUT_DIR, f"temp_{os.path.splitext(os.path.basename(output_path))[0]}_pydub.mp3")
    global COMPOSITE_DEVICE, USE_DEPTH, DEPTH_MODE, DEPTH_ADJUST_INFLUENCE, DEPTH_THRESHOLD_MEAN_MULTIPLIER, DEPTH_RANGES, SMOOTH_WINDOW
    global midas, midas_transform, MIDAS_AVAILABLE
    device = COMPOSITE_DEVICE

    fg_video=None; alpha_video=None; bg_frames_np=None; composited_video_list=None; composited_video_np=None; final_audio_segment=None
    setup_result = None

    try:
        # --- Setup ---
        setup_result = setup_compositing(foreground_path, alpha_path, background_path)
        if setup_result is None: raise RuntimeError("Compositing setup failed.")
        fg_info, bg_info, bg_frames_np, width, height, fps, frame_count_limit, bg_duration, bg_fps = setup_result

        # --- Initialize Smoother ---
        smoother = DepthSmoother(window_size=SMOOTH_WINDOW)

        print("Loading foreground and alpha videos using skvideo.io.vread...")
        read_frames = None if frame_count_limit <= 0 else frame_count_limit
        print(f"Attempting to read {read_frames if read_frames else 'all'} frames.")
        try:
            fg_video = skvideo.io.vread(foreground_path, num_frames=read_frames, outputdict={"-pix_fmt": "bgr24"})
            alpha_video = skvideo.io.vread(alpha_path, num_frames=read_frames, as_grey=True)
            print(f"Loaded FG shape: {fg_video.shape}, Alpha shape: {alpha_video.shape}")
            if fg_video.ndim != 4 or fg_video.shape[3] != 3: raise ValueError(f"Unexpected FG shape: {fg_video.shape}")
            if alpha_video.ndim != 3: raise ValueError(f"Unexpected Alpha shape: {alpha_video.shape}")
            if fg_video.shape[0] != alpha_video.shape[0]: raise ValueError(f"Frame count mismatch FG ({fg_video.shape[0]}) vs Alpha ({alpha_video.shape[0]})")
            actual_frames_read = fg_video.shape[0]
            if read_frames and actual_frames_read < read_frames: print(f"Warning: Read {actual_frames_read} frames, expected {read_frames}.")
            if actual_frames_read == 0: raise RuntimeError("skvideo read 0 frames.")
            print(f"Successfully loaded {actual_frames_read} frames.")
        except MemoryError: print("\n!!! MemoryError: skvideo.io.vread failed. Video too large."); raise
        except Exception as read_e: print(f"\nError reading with skvideo: {read_e}"); traceback.print_exc(); raise IOError("skvideo failed to read.") from read_e

        # --- Frame Compositing ---
        num_frames_to_process = actual_frames_read
        print(f"Compositing {num_frames_to_process} frames...")
        print(f"Depth Enabled: {USE_DEPTH and MIDAS_AVAILABLE}, Mode: {DEPTH_MODE if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}, Smoothing: {SMOOTH_WINDOW if USE_DEPTH and MIDAS_AVAILABLE else 'N/A'}")
        composited_video_list = []

        for i in range(num_frames_to_process):
            if (i + 1) % 100 == 0: print(f"  Compositing frame {i + 1}/{num_frames_to_process}...", end='\r')
            fg_frame = fg_video[i]; alpha_frame = alpha_video[i]

            # --- Estimate Depth (with smoothing) ---
            depth_map = None
            if USE_DEPTH and MIDAS_AVAILABLE:
                depth_map = estimate_depth(fg_frame, midas, midas_transform, device, smoother)

            # --- Get Background Frame ---
            if len(bg_frames_np) == 0: raise RuntimeError("Background frames array is empty!")
            bg_idx = map_background_frame(i, fps, bg_duration, bg_fps) % len(bg_frames_np)
            bg_frame = bg_frames_np[bg_idx]

            # --- Composite (passing depth ranges) ---
            try:
                comp_frame = composite_frames_torch(fg_frame, alpha_frame, bg_frame, depth_map, device,
                                                    use_depth=(USE_DEPTH and MIDAS_AVAILABLE),
                                                    depth_mode=DEPTH_MODE,
                                                    depth_influence=DEPTH_ADJUST_INFLUENCE,
                                                    depth_thresh_multiplier=DEPTH_THRESHOLD_MEAN_MULTIPLIER,
                                                    depth_ranges=DEPTH_RANGES) # Pass ranges
                composited_video_list.append(comp_frame)
            except Exception as comp_e: print(f"\nError compositing frame {i}: {comp_e}. Stopping."); traceback.print_exc(); break

        print(f"\nFinished compositing {len(composited_video_list)} frames.")
        if not composited_video_list: raise RuntimeError("No frames were successfully composited.")

        print("Converting composite frame list to NumPy array...")
        composited_video_np = np.array(composited_video_list, dtype=np.uint8)
        print(f"Composited video array shape: {composited_video_np.shape}")

        # --- Write Temporary Video ---
        print(f"Writing temporary video ({composited_video_np.shape[0]} frames) using skvideo.io.vwrite...")
        try:
            input_dict = {'-r': str(fps), '-s': f'{width}x{height}', '-pix_fmt': 'bgr24'}
            output_dict = {'-vcodec': 'libx264', '-crf': '23', '-preset': 'fast', '-pix_fmt': 'yuv420p', '-r': str(fps)}
            skvideo.io.vwrite(temp_video_path, composited_video_np, inputdict=input_dict, outputdict=output_dict)
            print(f"Temporary video saved successfully: {temp_video_path}")
        except Exception as write_e: print(f"\nError writing temporary video with skvideo: {write_e}"); traceback.print_exc(); raise IOError("skvideo write failed.") from write_e

        # --- Clear large video arrays ---
        print("Clearing large video arrays from memory...")
        del fg_video, alpha_video, composited_video_list, composited_video_np; import gc; gc.collect()
        fg_video, alpha_video, composited_video_list, composited_video_np = None, None, None, None

        # --- Audio Processing with Pydub (unchanged) ---
        if PYDUB_AVAILABLE:
            print("Processing audio using Pydub..."); final_audio_segment = None
            target_duration_ms = int((num_frames_to_process / fps) * 1000) if fps > 0 else 0
            print(f"Target audio duration: {target_duration_ms / 1000.0:.2f}s")
            try:
                 original_audio = None; print(f"  Loading original audio...");
                 try: original_audio = AudioSegment.from_file(original_path); print(f"    Original loaded ({len(original_audio)/1000.0:.2f}s).");
                 except Exception as e: print(f"    Error loading original: {e}")
                 if original_audio and target_duration_ms > 0: original_audio = original_audio[:target_duration_ms]; print(f"    Trimmed original to {len(original_audio)/1000.0:.2f}s")

                 background_audio = None
                 if mix_background_audio:
                      print(f"  Loading background audio...");
                      try: background_audio = AudioSegment.from_file(background_path); print(f"    Background loaded ({len(background_audio)/1000.0:.2f}s).")
                      except Exception as e: print(f"    Error loading background: {e}")
                      if background_audio and target_duration_ms > 0 and len(background_audio) > 0:
                           if len(background_audio) < target_duration_ms: loops = int(np.ceil(target_duration_ms / len(background_audio))); print(f"      Looping BG {loops} times..."); background_audio = background_audio * loops
                           background_audio = background_audio[:target_duration_ms] - 6; print(f"    Trimmed/Looped BG to {len(background_audio)/1000.0:.2f}s (volume reduced)")
                      else: background_audio = None

                 if original_audio and background_audio: print("    Mixing..."); final_audio_segment = original_audio.overlay(background_audio)
                 elif original_audio: print("    Using original audio only."); final_audio_segment = original_audio
                 elif background_audio: print("    Using background audio only."); final_audio_segment = background_audio
                 else: print("    No valid audio found."); final_audio_segment = None

                 if final_audio_segment and len(final_audio_segment) > 0: print(f"  Exporting final audio: {temp_audio_path}"); final_audio_segment.export(temp_audio_path, format="mp3", bitrate="192k")
                 elif final_audio_segment: print("  Warning: Final audio segment is empty."); final_audio_segment = None
                 else: print("  No valid final audio generated.")
            except Exception as audio_e: print(f"  Error during Pydub: {audio_e}"); traceback.print_exc(); final_audio_segment = None
        else: final_audio_segment = None

        # --- Mux Video and Audio using FFmpeg (unchanged) ---
        print("Muxing final output using FFmpeg...")
        if not os.path.exists(temp_video_path): raise RuntimeError(f"Temp video not found: {temp_video_path}")
        ffmpeg_cmd_mux = ["ffmpeg", "-y", "-i", temp_video_path]
        maps = ["-map", "0:v"]
        temp_audio_exists = final_audio_segment and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 100
        if temp_audio_exists:
            print(f"  Adding audio from: {temp_audio_path}"); ffmpeg_cmd_mux.extend(["-i", temp_audio_path]); maps.extend(["-map", "1:a"])
            ffmpeg_cmd_mux.extend(maps); ffmpeg_cmd_mux.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", output_path])
        else:
            print("  Muxing video only."); ffmpeg_cmd_mux.extend(maps); ffmpeg_cmd_mux.extend(["-c:v", "copy", "-an", "-movflags", "+faststart", output_path])
        print(f"Executing FFmpeg command:\n  {' '.join(ffmpeg_cmd_mux)}")
        try:
            process = subprocess.run(ffmpeg_cmd_mux, check=True, capture_output=True, text=True, timeout=600)
            print("FFmpeg muxing successful.");
            if process.stderr and "warnings" in process.stderr.lower(): print(f"FFmpeg stderr:\n{process.stderr}")
        except subprocess.CalledProcessError as e: print(f"\n!!! FFmpeg FAILED !!!\nRC: {e.returncode}\nCmd: {' '.join(e.cmd)}\nStderr:\n{e.stderr}"); raise RuntimeError("FFmpeg muxing failed.") from e
        except subprocess.TimeoutExpired as e: print(f"\n!!! FFmpeg Timed Out !!!\nCmd: {' '.join(e.cmd)}"); raise RuntimeError("FFmpeg muxing timed out.") from e

    except MemoryError: print("\n!!! scikit-video implementation aborted due to MemoryError. !!!"); # Don't re-raise
    except Exception as e: print(f"\n!!! Error in scikit-video+pydub: {e} !!!"); if not isinstance(e, (IOError, RuntimeError)): traceback.print_exc(); raise
    finally:
        print("Cleaning up skvideo/pydub resources...")
        del fg_video, alpha_video, bg_frames_np, composited_video_list, composited_video_np, final_audio_segment, setup_result; import gc; gc.collect()
        for temp_path in [temp_video_path, temp_audio_path]:
            if os.path.exists(temp_path): try: os.remove(temp_path); except OSError as e: print(f"Warning: Could not remove {temp_path}: {e}")
# --- End Implementation Backends ---


# --- Main Benchmarking Script ---
def main():
    print("--- Video Background Replacement Benchmark (Depth+Smooth+Hierarchy) ---")

    # --- Print Library Versions ---
    print("\n--- Library Versions ---")
    print(f"Python version:       {sys.version.split()[0]}")
    print(f"NumPy version:        {np.__version__}")
    print(f"PyTorch version:      {torch.__version__}")
    try: print(f"PyTorch backend:      MPS={'Available' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'N/A'}, CUDA={'Available' if torch.cuda.is_available() else 'N/A'}")
    except: pass
    print(f"OpenCV version:       {cv2.__version__}")
    if MOVIEPY_AVAILABLE: print(f"MoviePy version:      {moviepy.__version__}")
    else: print("MoviePy:              Not Available")
    if AV_AVAILABLE: print(f"PyAV version:         {av.__version__}")
    else: print("PyAV:                 Not Available")
    if GSTREAMER_AVAILABLE: print(f"PyGObject/GStreamer:  Available")
    else: print("PyGObject/GStreamer:  Not Available")
    if SKVIDEO_AVAILABLE:
        try: print(f"scikit-video version: {importlib_metadata.version('scikit-video')}")
        except importlib_metadata.PackageNotFoundError: print("scikit-video version: Installed (metadata error)")
    else: print("scikit-video:         Not Available")
    if PYDUB_AVAILABLE:
        try: pydub_version = importlib_metadata.version('pydub'); print(f"Pydub version:        {pydub_version}")
        except importlib_metadata.PackageNotFoundError: print("Pydub version:        Installed (metadata error)")
    else: print("Pydub:                Not Available")
    print(f"MiDaS Available:      {MIDAS_AVAILABLE}") # Set after loading attempt
    print("------------------------\n")

    # --- Determine Devices ---
    global RVM_DEVICE, COMPOSITE_DEVICE
    print("--- Determining Compute Devices ---")
    preferred_device = get_device(); print(f"Preferred device: {preferred_device}")
    RVM_DEVICE = preferred_device; print(f"Attempting to use {RVM_DEVICE} for RVM.")
    COMPOSITE_DEVICE = preferred_device
    if USE_DEPTH and MIDAS_AVAILABLE:
        print(f"Attempting to move MiDaS model to {COMPOSITE_DEVICE}...")
        try:
            if midas is None: raise RuntimeError("MiDaS model object is None.")
            midas.to(COMPOSITE_DEVICE); midas.eval()
            print(f"MiDaS model successfully moved to {COMPOSITE_DEVICE}.")
        except Exception as e_move:
            print(f"Warning: Failed to move/use MiDaS on {COMPOSITE_DEVICE}: {e_move}")
            if COMPOSITE_DEVICE.type != 'cpu':
                print("Attempting fallback to CPU for MiDaS/Compositing..."); COMPOSITE_DEVICE = torch.device('cpu')
                try:
                    if midas is None: raise RuntimeError("MiDaS model object is None.")
                    midas.to(COMPOSITE_DEVICE); midas.eval(); print("MiDaS model successfully moved to CPU.")
                except Exception as e_cpu_move: print(f"FATAL: Failed MiDaS move to CPU: {e_cpu_move}. Disabling depth."); MIDAS_AVAILABLE = False
            else: print(f"FATAL: Failed MiDaS use on CPU: {e_move}. Disabling depth."); MIDAS_AVAILABLE = False
    elif USE_DEPTH and not MIDAS_AVAILABLE: print("Depth enabled, but MiDaS unavailable. Compositing on RVM device without depth."); COMPOSITE_DEVICE = RVM_DEVICE
    else: print("Depth disabled. Compositing on RVM device."); COMPOSITE_DEVICE = RVM_DEVICE

    print(f"Final RVM Device Chosen:       {RVM_DEVICE}")
    print(f"Final Composite/Depth Device: {COMPOSITE_DEVICE}")
    print(f"Depth Estimation Active:       {USE_DEPTH and MIDAS_AVAILABLE}")
    if USE_DEPTH and MIDAS_AVAILABLE:
        print(f"Depth Mode:                    {DEPTH_MODE}")
        print(f"Depth Smoothing Window:        {SMOOTH_WINDOW}")
        if DEPTH_MODE == 'adjust': print(f"Depth Adjust Influence:        {DEPTH_ADJUST_INFLUENCE}")
        elif DEPTH_MODE == 'threshold': print(f"Depth Threshold Multiplier:    {DEPTH_THRESHOLD_MEAN_MULTIPLIER}")
        elif DEPTH_MODE == 'hierarchical': print(f"Depth Hierarchical Ranges:     {DEPTH_RANGES}")
    print("----------------------------------\n")

    # --- Check Input Files ---
    print("--- Checking Input Files ---")
    input_files = {"Original Video": ORIGINAL_PATH, "Background Video": BACKGROUND_PATH, "RVM Model": MODEL_PATH}
    all_inputs_exist = True
    for name, path in input_files.items():
        exists = os.path.exists(path); status = "Found" if exists else "MISSING!"
        print(f"  {name:<18}: {status} ({os.path.abspath(path)})")
        if not exists: all_inputs_exist = False
    if not all_inputs_exist: print("\nError: Input files missing."); return
    print("------------------------\n")

    # --- Generate Mattes using RVM ---
    print("--- Step 1: RVM Matte Generation ---")
    print(f"Using device {RVM_DEVICE} for RVM inference.")
    rvm_success = False
    # Use paths set after arg parsing
    global FOREGROUND_MATTE_PATH, ALPHA_MATTE_PATH
    mattes_exist = os.path.exists(FOREGROUND_MATTE_PATH) and os.path.exists(ALPHA_MATTE_PATH)
    if mattes_exist:
         print(f"Found existing mattes:\n  FG: {FOREGROUND_MATTE_PATH}\n  Alpha: {ALPHA_MATTE_PATH}\nSkipping RVM generation.")
         rvm_success = True
    else:
        print("Generating foreground and alpha mattes...")
        rvm_start_time = time.time()
        try:
            rvm_success = generate_foreground_and_alpha(ORIGINAL_PATH, FOREGROUND_MATTE_PATH, ALPHA_MATTE_PATH, MODEL_PATH)
            rvm_end_time = time.time()
            if rvm_success:
                print(f"RVM Matte generation successful ({rvm_end_time - rvm_start_time:.2f}s).")
                if not os.path.exists(FOREGROUND_MATTE_PATH): print(f"Error: FG matte missing: {FOREGROUND_MATTE_PATH}"); rvm_success = False
                if not os.path.exists(ALPHA_MATTE_PATH): print(f"Error: Alpha matte missing: {ALPHA_MATTE_PATH}"); rvm_success = False
            else: print("RVM Matte generation failed.")
        except Exception as rvm_e: print(f"\nCRITICAL ERROR during RVM Matte Generation step: {rvm_e}"); traceback.print_exc(); rvm_success = False
    if not rvm_success: print("\nCannot proceed without mattes. Exiting."); return
    print("----------------------------------\n")

    # --- Run Compositing Implementations ---
    implementations_to_run = {
        "ffmpeg": (ffmpeg_implementation, True),
        "gstreamer": (gstreamer_implementation, GSTREAMER_AVAILABLE),
        "pyav": (pyav_implementation, AV_AVAILABLE),
        "moviepy": (moviepy_implementation, MOVIEPY_AVAILABLE), # Does not support depth features well
        "scikit_video_pydub": (scikit_video_pydub_implementation, SKVIDEO_AVAILABLE),
    }

    global MIX_AUDIO # Use global MIX_AUDIO set by args
    results = {}
    active_depth_mode_str = f"depth-{DEPTH_MODE}" if (USE_DEPTH and MIDAS_AVAILABLE) else "depth-off"
    active_smooth_str = f"smooth-{SMOOTH_WINDOW}" if (USE_DEPTH and MIDAS_AVAILABLE and SMOOTH_WINDOW > 1) else "smooth-off"

    print(f"--- Step 2: Compositing Benchmarks (Mix BG Audio: {MIX_AUDIO}, Mode: {active_depth_mode_str}, Smooth: {active_smooth_str}) ---")

    for name, (func, available) in implementations_to_run.items():
        if not available:
            print(f"\n=== Skipping Implementation: {name} (Not Available/Supported) ===")
            results[name] = {'duration': 0, 'success': False, 'output_file': 'N/A', 'skipped': True}
            continue

        output_filename = f"output_{name}_{active_depth_mode_str}_{active_smooth_str}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n=== Running Implementation: {name} ===")
        if name == "moviepy": print("    (Note: MoviePy does not use frame-by-frame depth/smoothing)")
        print(f"    Output file will be: {output_path}")

        start_time = time.time()
        success = False; error_occurred = False
        try:
            func(FOREGROUND_MATTE_PATH, ALPHA_MATTE_PATH, BACKGROUND_PATH, ORIGINAL_PATH, output_path, mix_background_audio=MIX_AUDIO)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024: success = True
            elif os.path.exists(output_path): print(f"Warning: Output file '{output_path}' small ({os.path.getsize(output_path)} bytes). Marking failed."); success = False
            else: print(f"Error: Output file not created: {output_path}"); success = False
        except MemoryError: print(f"\n!!! Implementation '{name}' FAILED: MemoryError !!!"); success = False; error_occurred = True
        except Exception as e: print(f"\n!!! Implementation '{name}' FAILED with exception !!!"); success = False; error_occurred = True # Error logged in function

        end_time = time.time(); duration = end_time - start_time
        results[name] = {'duration': duration, 'success': success, 'output_file': output_path, 'skipped': False}
        status_str = "OK" if success else ("FAILED" if error_occurred else "FAILED (Output Issue)")
        print(f"--- {name} {status_str} ({duration:.2f}s) ---")
        if success:
            try: file_size_mb = os.path.getsize(output_path) / (1024 * 1024); print(f"    Output: {os.path.basename(output_path)} ({file_size_mb:.2f} MB)")
            except Exception: print(f"    Output: {os.path.basename(output_path)} (Size check failed)")
        elif os.path.exists(output_path): print(f"    Output file exists despite error: {os.path.basename(output_path)}")

    print("\n------------------------------------------")

    # --- Print Summary ---
    print("\n--- Benchmark Summary ---")
    print(f"Depth Mode Active: {active_depth_mode_str}, Smoothing: {active_smooth_str}")
    print(f"{'Implementation':<22} | {'Status':<8} | {'Duration (s)':<12} | {'Output File'} ")
    print("-" * 80)
    for name, result in results.items():
        if result.get('skipped', False): status = "Skipped"; duration_str = "N/A"; output_info = "Not Run"
        else:
            status = "Success" if result['success'] else "Failed"
            duration_str = f"{result['duration']:.2f}"
            output_info = os.path.basename(result['output_file']) if result['output_file'] else "N/A"
            if result['success']:
                 if not os.path.exists(result['output_file']): output_info += " (MISSING!)"; status = "Failed"
                 elif os.path.getsize(result['output_file']) <= 1024: output_info += " (Empty?)"; status = "Failed"
            elif not result['success'] and os.path.exists(result['output_file']) and os.path.getsize(result['output_file']) > 1024: output_info += " (Exists Despite Error)"
        print(f"{name:<22} | {status:<8} | {duration_str:<12} | {output_info} ")
    print("-" * 80)
    print("Benchmark complete.")


# --- Main execution block ---
if __name__ == "__main__":
    # Make MiDaS resources global
    global midas, midas_transform, MIDAS_AVAILABLE
    # Global devices
    global RVM_DEVICE, COMPOSITE_DEVICE
    # Global parameters controllable by args
    global USE_DEPTH, DEPTH_MODE, DEPTH_ADJUST_INFLUENCE, DEPTH_THRESHOLD_MEAN_MULTIPLIER
    global SMOOTH_WINDOW, DEPTH_RANGES, MIX_AUDIO
    # Global paths controllable by args
    global MODEL_PATH, ORIGINAL_PATH, BACKGROUND_PATH, OUTPUT_DIR
    global FOREGROUND_MATTE_PATH, ALPHA_MATTE_PATH


    # Define Argparse BEFORE loading MiDaS
    import argparse
    parser = argparse.ArgumentParser(description="Video Background Replacement with Depth-Aware Compositing, Smoothing, and Hierarchy.")
    # --- Input/Output Args ---
    parser.add_argument('--input', type=str, default=ORIGINAL_PATH, help='Path to original input video.')
    parser.add_argument('--bg', type=str, default=BACKGROUND_PATH, help='Path to background video.')
    parser.add_argument('--rvm-model', type=str, default=MODEL_PATH, help='Path to RVM model file.')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Directory for output files.')
    # --- Depth Control Args ---
    parser.add_argument('--no-depth', action='store_true', help='Disable depth estimation and depth-aware compositing.')
    parser.add_argument('--depth-mode', type=str, default=DEPTH_MODE, choices=['adjust', 'threshold', 'hierarchical'], help="Mode for using depth ('adjust', 'threshold', 'hierarchical').")
    parser.add_argument('--depth-influence', type=float, default=DEPTH_ADJUST_INFLUENCE, help="Influence factor for 'adjust' mode (0.0-1.0).")
    parser.add_argument('--depth-thresh', type=float, default=DEPTH_THRESHOLD_MEAN_MULTIPLIER, help="Mean multiplier for 'threshold' mode.")
    parser.add_argument('--depth-ranges', type=str, default=None, help='Define depth ranges for hierarchical mode. Comma-separated MIN-MAX pairs (e.g., "0.1-0.4,0.4-0.8"). Assumes normalized depth [0,1].')
    # --- Smoothing Arg ---
    parser.add_argument('--smooth-window', type=int, default=SMOOTH_WINDOW, help='Number of frames for temporal smoothing of depth maps (1 to disable).')
    # --- Audio Arg ---
    parser.add_argument('--mix-audio', action='store_true', default=True, help='Enable mixing background audio (if available).')
    parser.add_argument('--no-mix-audio', action='store_false', dest='mix_audio', help='Disable mixing background audio.')

    args = parser.parse_args()

    # --- Override Global Constants based on Arguments ---
    USE_DEPTH = not args.no_depth
    DEPTH_MODE = args.depth_mode
    DEPTH_ADJUST_INFLUENCE = args.depth_influence
    DEPTH_THRESHOLD_MEAN_MULTIPLIER = args.depth_thresh
    SMOOTH_WINDOW = args.smooth_window
    MIX_AUDIO = args.mix_audio

    # Parse depth ranges string
    if args.depth_ranges:
        try:
            ranges_str = args.depth_ranges.split(',')
            DEPTH_RANGES = []
            for r in ranges_str:
                parts = r.split('-')
                if len(parts) == 2:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    if 0.0 <= min_val < max_val <= 1000.0: # Basic sanity check (allow unnormalized too)
                        DEPTH_RANGES.append((min_val, max_val))
                    else: print(f"Warning: Invalid range values '{r}', must be 0 <= min < max. Skipping.")
                else: print(f"Warning: Invalid range format '{r}', use MIN-MAX. Skipping.")
            if not DEPTH_RANGES: # If all ranges were invalid
                print("Warning: No valid depth ranges parsed from input.")
                DEPTH_RANGES = None
        except Exception as e:
            print(f"Error parsing depth ranges '{args.depth_ranges}': {e}. Using None.")
            DEPTH_RANGES = None
    else:
        DEPTH_RANGES = None

    MODEL_PATH = args.rvm_model
    ORIGINAL_PATH = args.input
    BACKGROUND_PATH = args.bg
    OUTPUT_DIR = args.output_dir

    # Update output directory and matte paths based on potentially overridden input/output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_basename = os.path.splitext(os.path.basename(ORIGINAL_PATH))[0]
    FOREGROUND_MATTE_PATH = os.path.join(OUTPUT_DIR, f"{input_basename}_foreground_rvm.mp4")
    ALPHA_MATTE_PATH = os.path.join(OUTPUT_DIR, f"{input_basename}_alpha_rvm.mp4")


    # --- Load MiDaS only if USE_DEPTH is True ---
    if USE_DEPTH:
        print("\n--- Attempting MiDaS Integration (Depth Enabled) ---")
        try:
            print("Loading MiDaS model (intel-isl/MiDaS, MiDaS_small)...")
            # Using trust_repo=True may be necessary for some environments
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            # Select the correct transform based on the loaded model
            model_name_check = str(type(midas)).lower()
            if "small" in model_name_check:
                midas_transform = midas_transforms.small_transform
                print("Using MiDaS small model and transform.")
            else:
                print("Warning: Assuming MiDaS large (DPT) model loaded. Using DPT transform.")
                midas_transform = midas_transforms.dpt_transform
            MIDAS_AVAILABLE = True
            print("MiDaS model and transforms loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load MiDaS model: {e}. Depth estimation will be skipped.")
            # traceback.print_exc()
            MIDAS_AVAILABLE = False; midas = None; midas_transform = None
    else:
        print("\n--- MiDaS Integration Skipped (Depth Disabled via --no-depth) ---")
        MIDAS_AVAILABLE = False; midas = None; midas_transform = None

    # --- Run Main Function ---
    try:
        main()
    except Exception as main_e:
        print(f"\n--- CRITICAL ERROR IN MAIN EXECUTION ---")
        print(f"Error: {main_e}"); traceback.print_exc()
        print("--- Benchmark Run Aborted ---")
    # --- End Main execution block ---

'''
Okay, let's pinpoint the exact changes introduced in the most recent script you provided, compared to the version just before it (which already had basic depth awareness):

Temporal Smoothing of Depth Maps:
DepthSmoother Class: A new class using collections.deque was added. Its purpose is to average depth maps over a small number of consecutive frames (SMOOTH_WINDOW) to reduce frame-to-frame flickering in the depth estimation.
estimate_depth Refactoring:
The original depth estimation logic was moved into a helper function _estimate_depth_raw.
The main estimate_depth function now requires a smoother object as an argument. It calls _estimate_depth_raw and then passes the result through smoother.smooth() before returning the averaged depth map.
Integration into Backends: Each frame-processing backend (like ffmpeg_implementation, pyav_implementation, etc.) was modified to:
Create an instance of DepthSmoother before the frame loop (smoother = DepthSmoother(window_size=SMOOTH_WINDOW)).
Pass this smoother instance when calling estimate_depth inside the loop.
--smooth-window Argument: A command-line argument was added to control the size of the smoothing window (defaulting to 3).
Hierarchical Depth Segmentation:
Concept Introduced: The idea of dividing the depth map into specific zones (ranges) was added.
DEPTH_RANGES Global: A global variable (DEPTH_RANGES) was introduced to hold the user-defined ranges (e.g., [(0.1, 0.4), (0.4, 0.8)]).
--depth-ranges Argument: A command-line argument was added to allow the user to specify these ranges as a string (e.g., "0.1-0.4,0.4-0.8"). Parsing logic was added in main() to convert this string into the list of tuples format, with error handling.
hierarchical_segmentation Function: A new helper function was defined to take a depth map and the DEPTH_RANGES list and generate a list of binary masks, one for each specified range. (Note: This function itself isn't directly called by the current compositing logic, but it's available for future expansion).
hierarchical Depth Mode: A new option ('hierarchical') was added to the DEPTH_MODE choices within composite_frames_torch. In its current implementation, this mode:
Only considers the first range specified in DEPTH_RANGES.
Creates a hard binary mask for pixels falling within that specific depth range.
Uses this binary mask directly as the final_alpha, completely replacing the original RVM alpha for pixels within that range.
Modifications to Existing Compositing Logic (composite_frames_torch):
adjust Mode Formula: The way final_alpha is calculated in 'adjust' mode changed from an additive boost to a linear interpolation: final_alpha = rvm_alpha * (1.0 - depth_influence) + depth_normalized * depth_influence. This blends the RVM alpha and the normalized depth based on the influence factor.
threshold Mode Formula: The calculation changed from replacing the alpha (final_alpha = depth_mask) to multiplying the RVM alpha by the depth mask (final_alpha = rvm_alpha * depth_mask). This means only pixels considered foreground by both RVM and the depth threshold remain opaque. (Self-correction: The underlying adaptive_threshold helper correctly uses > for closer pixels based on MiDaS output).
Parameter Passing: The composite_frames_torch function now accepts depth_ranges as an argument, which is passed down from the backend implementations.
Minor Changes:
Output Filename: The benchmark output filenames were updated to include the smoothing window size (e.g., smooth-3 or smooth-off).
Clarity: Renamed the alpha tensor inside composite_frames_torch to rvm_alpha to distinguish it better from the final_alpha. Added more debug prints within the hierarchical segmentation helper.
In short: the latest script builds upon the previous depth-aware version by adding temporal stability via smoothing and introducing the framework for multi-zone depth control via hierarchical segmentation, along with refining the logic of the existing depth modes.
'''