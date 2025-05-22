
import os
import json
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import logging
import time
import random
import math
from collections import deque
from PIL import Image
import torch
import torchvision.transforms as T

# Attempt to import optional dependencies
try:
    import mediapipe as mp

    _mediapipe_available = True
except ImportError:
    mp = None
    _mediapipe_available = False
    print("WARNING: Mediapipe not found. Effects requiring it will be disabled.")

try:
    import colour

    _colour_available = True
except ImportError:
    colour = None
    _colour_available = False
    print("WARNING: colour-science not found. LUT effect will be disabled.")


try:
    # Check for original segment-anything
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    _sam_available = True
except ImportError:
    _sam_available = False
    print("WARNING: segment_anything library not found. SAM v1 effect will be disabled.")


try:
    # Check for ultralytics (which includes their SAM implementation)
    # Check specifically for the SAM class to ensure it's the right version/module
    from ultralytics import SAM as UltralyticsSAM # Use an alias to avoid name clash

    _sam2_available = True
except ImportError:
    UltralyticsSAM = None # Define alias as None if import fails
    _sam2_available = False
    print("WARNING: ultralytics[sam] not found or SAM class missing. SAM v2 effect will be disabled.")


import argparse

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/video_effects_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
DEFAULT_FPS = 30.0
CONFIG_FILE = "config.json"
GOLD_TINT_COLOR = (210, 165, 30)  # BGR format for OpenCV - More orange-gold
# GOLD_TINT_COLOR = (0, 215, 255) # Alt: More yellow-gold BGR
TRAIL_START_ALPHA, TRAIL_END_ALPHA = 1.0, 0.2 # Make trails fade more
TRAIL_RADIUS = 2
TINT_STRENGTH = 0.15 # Slightly reduced tint strength
SEGMENTATION_THRESHOLD = 0.5
MASK_BLUR_KERNEL = (21, 21)
RVM_DEFAULT_BG_COLOR = (0, 120, 0)  # Green screen BGR
SAM_MASK_ALPHA = 0.4 # Slightly increased visibility


# --- Utility Functions ---


def validate_frame(frame: np.ndarray) -> np.ndarray:
    """Ensures the frame is valid and in BGR format."""
    if frame is None or frame.size == 0:
        h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
        logger.warning(f"Invalid frame detected, returning default {w}x{h} black frame")
        return np.zeros((h, w, 3), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning(f"Invalid frame format ({frame.shape}, ndim={frame.ndim}), attempting conversion")
        try:
            if frame.ndim == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # BGRA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[2] == 1:  # Single channel treated as gray
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                logger.error(
                    f"Unsupported frame shape {frame.shape}, returning black frame"
                )
                # Try to determine reasonable dimensions if possible
                h = frame.shape[0] if frame.ndim >= 1 else DEFAULT_HEIGHT
                w = frame.shape[1] if frame.ndim >= 2 else DEFAULT_WIDTH
                return np.zeros((h, w, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error during frame conversion: {e}. Returning black frame.")
            h = DEFAULT_HEIGHT
            w = DEFAULT_WIDTH
            return np.zeros((h, w, 3), dtype=np.uint8)

    return frame


def resize_frame(frame: np.ndarray, target_dims: Tuple[int, int]) -> np.ndarray:
    """Resizes a frame to target dimensions (height, width)."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot resize empty frame")
    # OpenCV resize takes (width, height)
    return cv2.resize(
        frame, (target_dims[1], target_dims[0]), interpolation=cv2.INTER_LANCZOS4
    )


# --- Model Loading Functions ---


def load_rvm_model(
    device: str, model_name: str = "resnet50", pretrained: bool = True
) -> Optional[torch.nn.Module]:
    """Loads the Robust Video Matting model."""
    if not torch.__version__:
        logger.warning("PyTorch unavailable, cannot load RVM model")
        return None
    try:
        logger.info(f"Loading RVM model '{model_name}' (pretrained={pretrained})")
        # Ensure Hub cache dir exists or is writable
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        logger.debug(f"PyTorch Hub directory: {hub_dir}")

        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            model_name,
            pretrained=pretrained,
            trust_repo=True # Added for newer torch versions
        )
        target_device = "cpu"  # Default to CPU
        if (
            device == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            target_device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            target_device = "cuda"
        elif device != "cpu":
            logger.warning(
                f"Requested device {device} unavailable or unsupported, falling back to CPU"
            )

        model = model.to(device=target_device).eval()
        logger.info(f"RVM model '{model_name}' loaded to {target_device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load RVM model '{model_name}': {e}", exc_info=True)
        return None


def load_sam_mask_generator(
    device: str, model_type: str = "vit_b", checkpoint_path: Optional[str] = None
) -> Optional[Any]:
    """Loads the Segment Anything (SAM v1) Automatic Mask Generator."""
    if not _sam_available:
        logger.warning("segment_anything library unavailable, cannot load SAM v1")
        return None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"SAM v1 checkpoint invalid or missing: {checkpoint_path}")
        return None
    try:
        logger.info(f"Loading SAM v1 model '{model_type}' from {checkpoint_path}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        target_device = "cpu"
        if device == "mps" and torch.backends.mps.is_available():
            target_device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            target_device = "cuda"
        else:
            logger.warning(f"Requested device {device} unavailable, using CPU")
        sam.to(device=target_device)
        generator = SamAutomaticMaskGenerator(sam)
        logger.info(f"SAM v1 generator initialized on {target_device}")
        return generator
    except Exception as e:
        logger.error(f"Failed to load SAM v1 model: {e}", exc_info=True)
        return None


def load_sam2_video_predictor(
    device: str, checkpoint_path: Optional[str] = None
) -> Optional[Any]:
    """Loads the Segment Anything (SAM v2 / ultralytics SAM) Predictor."""
    if not _sam2_available or UltralyticsSAM is None:
        logger.warning("ultralytics library or SAM class unavailable, cannot load SAM 2")
        return None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"SAM 2 checkpoint invalid or missing: {checkpoint_path}")
        return None
    try:
        logger.info(f"Loading SAM 2 model from {checkpoint_path}")
        # Use the aliased UltralyticsSAM class
        model = UltralyticsSAM(checkpoint_path)

        target_device = "cpu"  # Default to CPU
        if (
            device == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            target_device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            target_device = "cuda"
        elif device != "cpu":
            logger.warning(
                f"Requested device {device} unavailable or unsupported, falling back to CPU for SAM 2"
            )

        # Ultralytics models typically handle device setting internally or via predict args
        # model.to(device=target_device) # Might not be needed or correct API
        logger.info(
            f"SAM 2 model initialized (will use device: {target_device} during prediction)"
        )
        return model
    except Exception as e:
        logger.error(f"Failed to load SAM 2 model: {e}", exc_info=True)
        return None


# --- Effect Classes ---


class FrameEffectWrapper:
    """Wraps an effect function with enabling/disabling and dependency checks."""

    def __init__(
        self,
        func,
        name: str,
        requires_mediapipe: bool = False,
        requires_torch: bool = False,
        requires_sam: bool = False, # SAM v1 specific
        requires_sam2: bool = False, # SAM v2 specific
        requires_colour: bool = False,
    ):
        self.func = func
        self.name = name
        self.requires_mediapipe = requires_mediapipe
        self.requires_torch = requires_torch
        self.requires_sam = requires_sam
        self.requires_sam2 = requires_sam2
        self.requires_colour = requires_colour
        self.enabled = True

    def check_dependencies(self) -> bool:
        """Checks if required dependencies are available."""
        # Check specific library flags first
        if self.requires_mediapipe and not _mediapipe_available:
            logger.log(logging.DEBUG if hasattr(self, 'suppress_warning') else logging.WARNING,
                       f"Mediapipe required but unavailable for {self.name}")
            self.suppress_warning = True # Suppress subsequent warnings for this instance
            return False
        if self.requires_torch and not torch.__version__:
            logger.log(logging.DEBUG if hasattr(self, 'suppress_warning') else logging.WARNING,
                       f"Torch required but unavailable for {self.name}")
            self.suppress_warning = True
            return False
        if self.requires_sam and not _sam_available:
             logger.log(logging.DEBUG if hasattr(self, 'suppress_warning') else logging.WARNING,
                        f"segment_anything required but unavailable for {self.name}")
             self.suppress_warning = True
             return False
        if self.requires_sam2 and not _sam2_available:
             logger.log(logging.DEBUG if hasattr(self, 'suppress_warning') else logging.WARNING,
                        f"ultralytics[sam] required but unavailable for {self.name}")
             self.suppress_warning = True
             return False
        if self.requires_colour and not _colour_available:
            logger.log(logging.DEBUG if hasattr(self, 'suppress_warning') else logging.WARNING,
                       f"colour-science required but unavailable for {self.name}")
            self.suppress_warning = True
            return False

        # Reset suppression if dependencies are met now
        if hasattr(self, 'suppress_warning'):
            del self.suppress_warning

        return True

    def __call__(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Applies the effect if enabled and dependencies met."""
        if not self.enabled:
            return frame
        if not self.check_dependencies():
            # Only add visual indicator if dependencies failed
            cv2.putText(
                frame,
                f"{self.name.upper()} UNAVAILABLE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame
        try:
            # Call the actual effect function
            # Pass original frame in kwargs for potential error recovery if needed
            original_frame = kwargs.get('original_frame', frame) # Get original if passed
            return self.func(frame=frame, **kwargs)
        except Exception as e:
            logger.error(f"Error during effect '{self.name}': {e}", exc_info=True)
            # Add visual error indicator to the original frame
            error_frame = kwargs.get('original_frame', frame).copy() # Use original if possible
            cv2.putText(
                error_frame,
                f"{self.name.upper()} ERROR",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return error_frame # Return frame with error text

    def reset(self):
        """Resets the effect state (e.g., enables it)."""
        self.enabled = True
        if hasattr(self, 'suppress_warning'):
            del self.suppress_warning
        # If the wrapped function is an object with a reset method, call it
        if hasattr(self.func, "reset") and callable(self.func.reset):
            self.func.reset()


class LUTColorGradeEffect:
    """Applies Look-Up Table (LUT) color grading."""

    def __init__(self, lut_files: List[Tuple[str, str]]):
        self.lut_files = lut_files
        self.current_idx = 0
        self.luts: List[Tuple[str, Any]] = []  # Store loaded LUT objects
        self.lut_intensity = 1.0
        self._load_luts()

    def _load_luts(self):
        """Loads LUT files using the colour-science library."""
        if not _colour_available:
            return
        self.luts = []
        for lut_name, lut_path in self.lut_files:
            try:
                lut = colour.io.read_LUT(lut_path)
                if isinstance(lut, (colour.LUT3D, colour.LUT1D)):
                    self.luts.append((lut_name, lut))
                    logger.info(f"Loaded LUT: {lut_name} from {lut_path}")
                else:
                    logger.warning(f"Unsupported LUT format for {lut_name} at {lut_path}: {type(lut)}")
            except Exception as e:
                logger.error(f"Failed to load LUT {lut_name} from {lut_path}: {e}")
        if not self.luts:
            logger.warning("No valid LUTs loaded.")
        else:
            logger.info(f"Successfully loaded {len(self.luts)} LUTs.")
        self.current_idx = 0

    def process(self, frame: np.ndarray, runner: Any = None, **kwargs) -> np.ndarray:
        """Applies the currently selected LUT to the frame."""
        start_time = time.time()
        frame = validate_frame(frame)

        if not self.luts:
            cv2.putText(
                frame, "No LUTs Loaded", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
            )
            return frame

        try:
            if not (0 <= self.current_idx < len(self.luts)):
                logger.warning(f"Invalid LUT index {self.current_idx}, resetting to 0.")
                self.current_idx = 0
                if not self.luts:
                    return frame

            lut_name, lut = self.luts[self.current_idx]
            frame_rgb = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            frame_transformed = lut.apply(frame_rgb)
            frame_out = (frame_transformed * 255.0).clip(0, 255).astype(np.uint8)
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)

            intensity = runner.lut_intensity if runner else self.lut_intensity
            if intensity < 1.0:
                frame_out = cv2.addWeighted(frame, 1.0 - intensity, frame_out, intensity, 0.0)

            cv2.putText(
                frame_out, f"LUT: {lut_name[:25]}{'...' if len(lut_name)>25 else ''}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            logger.debug(f"Applied LUT: {lut_name} with intensity {intensity:.2f}")

        except Exception as e:
            logger.warning(f"LUT application failed for '{self.luts[self.current_idx][0]}': {e}", exc_info=True)
            cv2.putText(
                frame, "LUT Error", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            frame_out = frame
        return frame_out

    def reset(self):
        self.current_idx = 0
        self.lut_intensity = 1.0
        logger.info("LUT effect reset.")

    def cycle_lut(self):
        if not self.luts:
            logger.warning("Cannot cycle LUTs, none loaded.")
            return
        self.current_idx = (self.current_idx + 1) % len(self.luts)
        logger.info(f"Switched LUT to index {self.current_idx}: {self.luts[self.current_idx][0]}")


# --- Main Effect Runner Class ---


class PixelSensingEffectRunner:
    """Manages video capture, effects processing, and display."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._apply_config()

        self.face_mesh = None
        self.pose = None
        self.hands = None
        if _mediapipe_available:
            self._initialize_mediapipe()
        else:
            logger.warning("Mediapipe is unavailable. Effects relying on it will show 'UNAVAILABLE'.")

        self.cap = None
        self.out = None
        self.frame_width = DEFAULT_WIDTH
        self.frame_height = DEFAULT_HEIGHT
        self.fps = DEFAULT_FPS

        self.prev_gray = None
        self.prev_landmarks_flow = None
        self.hue_offset = 0
        self.frame_buffer = deque(maxlen=10)
        self.frame_times = deque(maxlen=100)
        self.effect_times = {}
        self.error_count = 0
        self.current_effect = "none"
        self.brightness_factor = 0.0
        self.contrast_factor = 1.0
        self.window_name = "PixelSense FX"

        self.goldenaura_variant = 0
        self.goldenaura_variant_names = ["Original", "Enhanced"]
        self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
        self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
        self.face_history = deque(maxlen=self.trail_length)
        self.pose_history = deque(maxlen=self.trail_length)

        self.lightning_style_index = 0
        self.lightning_style_names = ["Random", "Movement", "Branching"]
        self.lightning_styles = [self._apply_lightning_style_random, self._apply_lightning_style_movement, self._apply_lightning_style_branching]

        self.gesture_state = {"last_gesture": None, "last_time": 0.0, "gesture_count": 0}
        self.lut_color_grade_effect = None
        self.lut_intensity = 1.0

        self.rvm_rec = [None] * 4
        self.rvm_display_mode = 0
        self.current_rvm_model_idx = 0
        self.rvm_model_names = ["mobilenetv3", "resnet50"]
        self.rvm_models = {}
        self.rvm_background = None
        self.rvm_available = torch.__version__ is not None
        self.rvm_downsample_ratio = 0.25
        self.RVM_DISPLAY_MODES = ["Composite", "Alpha", "Foreground"]

        self.sam_available = _sam_available
        self.sam2_available = _sam2_available
        self.use_sam2_runtime = _sam2_available and self.config.get("sam2_checkpoint_path")
        self.sam_model_v1 = None
        self.sam_model_v2 = None
        self.sam_masks_cache = None

        self.LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.LK_PARAMS_GOLD = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        self._load_luts()
        self._initialize_effects()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config = {
            "input_source": "0", "output_path": "output.mp4", "lut_dir": "luts",
            "rvm_background_path": None, "sam_checkpoint_path": None, "sam2_checkpoint_path": None,
            "mode": "live", "display": True, "device": "cpu", "trail_length": 30,
            "glow_intensity": 0.5, "width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "fps": DEFAULT_FPS,
        }
        config = default_config.copy()
        actual_config_path = config_path or CONFIG_FILE
        if os.path.exists(actual_config_path):
            try:
                with open(actual_config_path, "r") as f:
                    loaded_config = json.load(f)
                    config.update(loaded_config)
                logger.info(f"Loaded config from {actual_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load/decode config file {actual_config_path}: {e}. Using defaults.")
        else:
            logger.info(f"Config file {actual_config_path} not found. Using default settings.")
        return config

    def _apply_config(self):
        self.input_source = self.config["input_source"]
        try: self.input_source = int(self.input_source)
        except (ValueError, TypeError): pass
        self.output_path = self.config["output_path"]
        self.lut_dir = self.config["lut_dir"]
        self._init_rvm_bg_path = self.config["rvm_background_path"]
        self.sam_checkpoint_path = self.config["sam_checkpoint_path"]
        self.sam2_checkpoint_path = self.config["sam2_checkpoint_path"]
        self.mode = self.config["mode"]
        self.display = self.config["display"]
        self.device = self.config["device"]
        if self.device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
            logger.warning("MPS device requested but not available/supported. Falling back to CPU.")
            self.device = "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            self.device = "cpu"
        logger.info(f"Selected device: {self.device}")
        self.trail_length = max(1, int(self.config["trail_length"]))
        self.glow_intensity = max(0.0, min(1.0, float(self.config["glow_intensity"])))
        self.desired_width = int(self.config.get("width", DEFAULT_WIDTH))
        self.desired_height = int(self.config.get("height", DEFAULT_HEIGHT))
        self.desired_fps = float(self.config.get("fps", DEFAULT_FPS))
        self.face_history = deque(maxlen=self.trail_length)
        self.pose_history = deque(maxlen=self.trail_length)
        self.use_sam2_runtime = _sam2_available and self.sam2_checkpoint_path

    def _initialize_mediapipe(self):
        if not _mediapipe_available:
            logger.warning("Cannot initialize Mediapipe - library not available.")
            return
        logger.info("Initializing Mediapipe modules...")
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, enable_segmentation=True)
            self.hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
            logger.info("Mediapipe modules initialized successfully.")
        except Exception as e:
            logger.error(f"Mediapipe initialization failed: {e}", exc_info=True)
            self.face_mesh = self.pose = self.hands = None
            #_mediapipe_available = False

    def _load_luts(self):
        self.luts = []
        if not os.path.isdir(self.lut_dir):
            logger.warning(f"LUT directory '{self.lut_dir}' not found or not a directory.")
            self.lut_color_grade_effect = None
            return
        for root, _, files in os.walk(self.lut_dir):
            for file in sorted(files):
                if file.lower().endswith(".cube"):
                    lut_path = os.path.join(root, file)
                    lut_name = os.path.splitext(file)[0]
                    self.luts.append((lut_name, lut_path))
        logger.info(f"Found {len(self.luts)} LUT files in '{self.lut_dir}' and subdirectories.")
        if self.luts and _colour_available:
            self.lut_color_grade_effect = LUTColorGradeEffect(self.luts)
        else:
            self.lut_color_grade_effect = None
            if not _colour_available: pass # Wrapper handles warning
            elif not self.luts: logger.warning("No valid .cube files found in LUT directory. LUT effect disabled.")

    def _initialize_effects(self):
        # SAM check requires BOTH library and a valid checkpoint path in config
        sam1_enabled = _sam_available and self.config.get("sam_checkpoint_path")
        sam2_enabled = _sam2_available and self.config.get("sam2_checkpoint_path")

        self.effects: Dict[str, FrameEffectWrapper] = {
            "none": FrameEffectWrapper(lambda frame, **kw: frame, "none"),
            "led_base": FrameEffectWrapper(self._apply_led_base_style, "led_base", requires_mediapipe=True),
            "led_enhanced": FrameEffectWrapper(self._apply_led_enhanced_style, "led_enhanced", requires_mediapipe=True),
            "led_hue_rotate": FrameEffectWrapper(self._apply_led_hue_rotate_style, "led_hue_rotate", requires_mediapipe=True),
            "goldenaura": FrameEffectWrapper(self._apply_goldenaura_style, "goldenaura", requires_mediapipe=True),
            "motion_blur": FrameEffectWrapper(self._apply_motion_blur_style, "motion_blur"),
            "chromatic_aberration": FrameEffectWrapper(self._apply_chromatic_aberration_style, "chromatic_aberration"),
            "lightning": FrameEffectWrapper(self._apply_lightning_cycle, "lightning", requires_mediapipe=True),
            "lut_color_grade": FrameEffectWrapper(
                (self.lut_color_grade_effect.process if self.lut_color_grade_effect else lambda frame, **kw: frame),
                "lut_color_grade", requires_colour=True
            ),
            "rvm_composite": FrameEffectWrapper(self._apply_rvm_composite_style, "rvm_composite", requires_torch=True),
             # Wrapper checks libraries, effect function checks loaded models
            "sam_segmentation": FrameEffectWrapper(
                self._apply_sam_segmentation_style, "sam_segmentation",
                requires_torch=True, requires_sam=sam1_enabled, requires_sam2=sam2_enabled
            ),
            "neon_glow": FrameEffectWrapper(self._apply_neon_glow_style, "neon_glow", requires_mediapipe=True),
            "particle_trail": FrameEffectWrapper(self._apply_particle_trail_style, "particle_trail", requires_mediapipe=True),
        }
        logger.info(f"Initialized {len(self.effects)} effects: {list(self.effects.keys())}")

    # --- Effect Implementations (with fixes) ---

    def _apply_led_base_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame) # Use original for error text
        try:
            if not self.face_mesh or not self.frame_width or not self.frame_height: return original_frame_for_error
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark:
                        landmarks.append((int(lm.x * self.frame_width), int(lm.y * self.frame_height)))
            if not landmarks:
                self.prev_gray = gray; self.prev_landmarks_flow = None
                return frame
            if (self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks)):
                curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
                prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                valid_indices = np.where((prev_pts[:, 0, 0] >= 0) & (prev_pts[:, 0, 0] < self.frame_width) & (prev_pts[:, 0, 1] >= 0) & (prev_pts[:, 0, 1] < self.frame_height))[0]
                if len(valid_indices) > 0:
                    prev_pts_valid = prev_pts[valid_indices]
                    flow, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS)
                    if flow is not None and status is not None:
                        good_new = flow[status.flatten() == 1]
                        good_old = prev_pts_valid[status.flatten() == 1]
                        if good_new.shape[0] > 0:
                            frame = self._draw_flow_trails_simple(frame, good_new, good_old, (255, 105, 180), (173, 216, 230), 1, 2)
                            frame = self._draw_motion_glow_separate(frame, landmarks, good_new, good_old)
            self.prev_landmarks_flow = landmarks
            self.prev_gray = gray
        except Exception as e:
            self.prev_gray = None; self.prev_landmarks_flow = None
            raise e # Let the wrapper handle logging and error text
        return frame

    def _apply_led_enhanced_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        try:
            if not self.face_mesh or not self.frame_width or not self.frame_height: return original_frame_for_error
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark:
                        landmarks.append((int(lm.x * self.frame_width), int(lm.y * self.frame_height)))
            if not landmarks:
                self.prev_gray = gray; self.prev_landmarks_flow = None
                # Apply distortion even if no face? Let's skip distortion if no face.
                return frame
            frame_distorted = self._apply_distortion(frame) # Apply distortion first
            if (self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks)):
                curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
                prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                valid_indices = np.where((prev_pts[:, 0, 0] >= 0) & (prev_pts[:, 0, 0] < self.frame_width) & (prev_pts[:, 0, 1] >= 0) & (prev_pts[:, 0, 1] < self.frame_height))[0]
                if len(valid_indices) > 0:
                    prev_pts_valid = prev_pts[valid_indices]
                    flow, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS) # Flow on original gray
                    if flow is not None and status is not None:
                        good_new = flow[status.flatten() == 1]
                        good_old = prev_pts_valid[status.flatten() == 1]
                        if good_new.shape[0] > 0:
                            # Draw on distorted frame
                            frame_distorted = self._draw_flow_trails_simple(frame_distorted, good_new, good_old, (255, 20, 147), (135, 206, 250), 2, 3)
                            frame_distorted = self._draw_motion_glow_separate(frame_distorted, landmarks, good_new, good_old)
            self.prev_landmarks_flow = landmarks
            self.prev_gray = gray
            frame = frame_distorted # Return distorted frame
        except Exception as e:
            self.prev_gray = None; self.prev_landmarks_flow = None
            raise e
        return frame

    def _apply_led_hue_rotate_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.int16)
            hue = (hue + self.hue_offset) % 180
            hsv[:, :, 0] = hue.astype(np.uint8)
            self.hue_offset = (self.hue_offset + 2) % 180
            frame_hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            if not self.face_mesh or not self.frame_width or not self.frame_height: return frame_hue # Return hue shifted if no face mesh
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Original gray for flow
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark:
                         landmarks.append((int(lm.x * self.frame_width), int(lm.y * self.frame_height)))
            if not landmarks:
                self.prev_gray = gray; self.prev_landmarks_flow = None
                return frame_hue
            if (self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks)):
                curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
                prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                valid_indices = np.where((prev_pts[:, 0, 0] >= 0) & (prev_pts[:, 0, 0] < self.frame_width) & (prev_pts[:, 0, 1] >= 0) & (prev_pts[:, 0, 1] < self.frame_height))[0]
                if len(valid_indices) > 0:
                    prev_pts_valid = prev_pts[valid_indices]
                    flow, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS)
                    if flow is not None and status is not None:
                        good_new = flow[status.flatten() == 1]
                        good_old = prev_pts_valid[status.flatten() == 1]
                        if good_new.shape[0] > 0:
                            frame_hue = self._draw_flow_trails_simple(frame_hue, good_new, good_old, (255, 69, 0), (255, 215, 0), 1, 2)
                            frame_hue = self._draw_motion_glow_separate(frame_hue, landmarks, good_new, good_old)
            self.prev_landmarks_flow = landmarks
            self.prev_gray = gray
            frame = frame_hue
        except Exception as e:
            self.prev_gray = None; self.prev_landmarks_flow = None
            raise e
        return frame

    def _apply_goldenaura_original(self, frame: np.ndarray, frame_time: float, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        frame_out = frame.copy()
        try:
            if not self.face_mesh or not self.pose or not self.frame_width or not self.frame_height: return original_frame_for_error
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                 results_face = self.face_mesh.process(frame_rgb)
                 results_pose = self.pose.process(frame_rgb)
            except Exception as mp_e: logger.warning(f"Goldenaura Orig: MP processing failed: {mp_e}"); return original_frame_for_error
            h, w = frame.shape[:2]
            face_landmarks_current, pose_landmarks_current = [], []
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    for lm in face_landmarks.landmark: face_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
            if results_pose.pose_landmarks:
                for lm in results_pose.pose_landmarks.landmark:
                    if lm.visibility > 0.3: pose_landmarks_current.append((int(lm.x * w), int(lm.y * h)))

            # Robust Optical Flow Update
            if self.prev_gray is not None:
                # Face
                if face_landmarks_current and self.prev_face_pts_gold.shape[0] > 0:
                    curr_pts_face = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    if curr_pts_face.shape[0] == self.prev_face_pts_gold.shape[0]:
                        try:
                            flow_face, status_face, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_face_pts_gold, curr_pts_face, **self.LK_PARAMS_GOLD)
                            if flow_face is not None and status_face is not None:
                                good_new = flow_face[status_face.flatten() == 1]; good_old = self.prev_face_pts_gold[status_face.flatten() == 1]
                                if good_new.shape[0] > 0: self.face_history.append((good_new, good_old))
                                self.prev_face_pts_gold = good_new.reshape(-1, 1, 2) if good_new.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
                            else: self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                        except cv2.error as cv_err: logger.warning(f"GO face LK failed: {cv_err}"); self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                    else: self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                elif face_landmarks_current: self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2); self.face_history.clear()
                else: self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32); self.face_history.clear()
                # Pose
                if pose_landmarks_current and self.prev_pose_pts_gold.shape[0] > 0:
                    curr_pts_pose = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    if curr_pts_pose.shape[0] == self.prev_pose_pts_gold.shape[0]:
                        try:
                            flow_pose, status_pose, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pose_pts_gold, curr_pts_pose, **self.LK_PARAMS_GOLD)
                            if flow_pose is not None and status_pose is not None:
                                good_new_pose = flow_pose[status_pose.flatten() == 1]; good_old_pose = self.prev_pose_pts_gold[status_pose.flatten() == 1]
                                if good_new_pose.shape[0] > 0: self.pose_history.append((good_new_pose, good_old_pose))
                                self.prev_pose_pts_gold = good_new_pose.reshape(-1, 1, 2) if good_new_pose.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
                            else: self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                        except cv2.error as cv_err: logger.warning(f"GO pose LK failed: {cv_err}"); self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                    else: self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                elif pose_landmarks_current: self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2); self.pose_history.clear()
                else: self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32); self.pose_history.clear()
            else:
                 if face_landmarks_current: self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                 if pose_landmarks_current: self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)

            # Draw Trails
            overlay = np.zeros_like(frame_out, dtype=np.uint8)
            if self.face_history:
                num_face = len(self.face_history); max_idx = max(num_face - 1, 1)
                for idx, item in enumerate(reversed(self.face_history)):
                    if len(item)==2 and item[0].shape[0]>0 and item[0].shape[0]==item[1].shape[0]:
                        flow, prev = item; alpha = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx)
                        color = tuple(int(c*alpha) for c in GOLD_TINT_COLOR); overlay=self._draw_flow_trails_simple(overlay,flow,prev,color,color,1,TRAIL_RADIUS)
            if self.pose_history:
                num_pose = len(self.pose_history); max_idx = max(num_pose - 1, 1)
                for idx, item in enumerate(reversed(self.pose_history)):
                    if len(item)==2 and item[0].shape[0]>0 and item[0].shape[0]==item[1].shape[0]:
                        flow, prev = item; alpha = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx)
                        color = tuple(int(c*alpha) for c in GOLD_TINT_COLOR); overlay=self._draw_flow_trails_simple(overlay,flow,prev,color,color,1,TRAIL_RADIUS)

            frame_out = cv2.add(frame_out, overlay)

            # Apply Tint
            tint_layer = np.full_like(frame_out, GOLD_TINT_COLOR, dtype=np.uint8)
            frame_out = cv2.addWeighted(frame_out, 1.0 - TINT_STRENGTH, tint_layer, TINT_STRENGTH, 0.0)

            # Apply Glow
            if hasattr(results_pose, 'segmentation_mask') and results_pose.segmentation_mask is not None:
                try:
                    mask = results_pose.segmentation_mask; condition = (mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
                    if condition.shape[:2] != frame_out.shape[:2]: condition = cv2.resize(condition, (w,h), interpolation=cv2.INTER_NEAREST)
                    mask_blurred = cv2.GaussianBlur(condition, MASK_BLUR_KERNEL, 0)
                    mask_alpha = (mask_blurred.astype(np.float32) / 255.0 * self.glow_intensity)
                    mask_alpha_3c = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)
                    glow_color_layer = np.full_like(frame_out, GOLD_TINT_COLOR, dtype=np.float32)
                    frame_float = frame_out.astype(np.float32); frame_float = (frame_float * (1.0 - mask_alpha_3c) + glow_color_layer * mask_alpha_3c)
                    frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
                except Exception as seg_e: logger.warning(f"GO seg/glow failed: {seg_e}")
            else: logger.debug("GO: No segmentation mask.")

            self.prev_gray = gray

        except Exception as e:
            self.prev_gray = None; self.prev_face_pts_gold = np.empty((0,1,2),dtype=np.float32); self.prev_pose_pts_gold = np.empty((0,1,2),dtype=np.float32)
            self.face_history.clear(); self.pose_history.clear()
            raise e # Let wrapper handle error
        return frame_out

    def _apply_goldenaura_enhanced(self, frame: np.ndarray, frame_time: float, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        frame_out = frame.copy()
        try:
            if not self.face_mesh or not self.pose or not self.frame_width or not self.frame_height: return original_frame_for_error
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                 results_face = self.face_mesh.process(frame_rgb)
                 results_pose = self.pose.process(frame_rgb)
            except Exception as mp_e: logger.warning(f"Goldenaura Enh: MP processing failed: {mp_e}"); return original_frame_for_error
            h, w = frame.shape[:2]
            face_landmarks_current, pose_landmarks_current = [], []
            if results_face.multi_face_landmarks:
                 for face_landmarks in results_face.multi_face_landmarks:
                      for lm in face_landmarks.landmark: face_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
            if results_pose.pose_landmarks:
                 for lm in results_pose.pose_landmarks.landmark:
                      if lm.visibility > 0.3: pose_landmarks_current.append((int(lm.x * w), int(lm.y * h)))

            # Robust Optical Flow Update (Identical logic to original)
            if self.prev_gray is not None:
                if face_landmarks_current and self.prev_face_pts_gold.shape[0] > 0:
                    curr_pts_face = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    if curr_pts_face.shape[0] == self.prev_face_pts_gold.shape[0]:
                        try:
                            flow_face, status_face, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_face_pts_gold, curr_pts_face, **self.LK_PARAMS_GOLD)
                            if flow_face is not None and status_face is not None: good_new = flow_face[status_face.flatten() == 1]; good_old = self.prev_face_pts_gold[status_face.flatten() == 1]; self.prev_face_pts_gold = good_new.reshape(-1, 1, 2) if good_new.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32); self.face_history.append((good_new, good_old)) if good_new.shape[0] > 0 else None
                            else: self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                        except cv2.error as cv_err: logger.warning(f"GE face LK failed: {cv_err}"); self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                    else: self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                elif face_landmarks_current: self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2); self.face_history.clear()
                else: self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32); self.face_history.clear()
                if pose_landmarks_current and self.prev_pose_pts_gold.shape[0] > 0:
                    curr_pts_pose = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    if curr_pts_pose.shape[0] == self.prev_pose_pts_gold.shape[0]:
                         try:
                            flow_pose, status_pose, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pose_pts_gold, curr_pts_pose, **self.LK_PARAMS_GOLD)
                            if flow_pose is not None and status_pose is not None: good_new_pose = flow_pose[status_pose.flatten() == 1]; good_old_pose = self.prev_pose_pts_gold[status_pose.flatten() == 1]; self.prev_pose_pts_gold = good_new_pose.reshape(-1, 1, 2) if good_new_pose.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32); self.pose_history.append((good_new_pose, good_old_pose)) if good_new_pose.shape[0] > 0 else None
                            else: self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                         except cv2.error as cv_err: logger.warning(f"GE pose LK failed: {cv_err}"); self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                    else: self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                elif pose_landmarks_current: self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2); self.pose_history.clear()
                else: self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32); self.pose_history.clear()
            else:
                 if face_landmarks_current: self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                 if pose_landmarks_current: self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)


            # Enhanced Drawing Params
            dynamic_glow_factor = 0.7 + 0.3 * math.sin(frame_time * 1.5)
            enhanced_tint_color_base = np.array([30, 180, 220])
            enhanced_tint_color = tuple(np.clip(enhanced_tint_color_base * dynamic_glow_factor, 0, 255).astype(int))
            line_thickness_face, dot_radius_face = 2, 3
            line_thickness_pose, dot_radius_pose = 3, 4

            # Draw Trails
            overlay = np.zeros_like(frame_out, dtype=np.uint8)
            if self.face_history:
                num_face = len(self.face_history); max_idx = max(num_face - 1, 1)
                for idx, item in enumerate(reversed(self.face_history)):
                    if len(item)==2 and item[0].shape[0]>0 and item[0].shape[0]==item[1].shape[0]:
                        flow, prev = item; alpha = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx)
                        color = tuple(int(c*alpha) for c in enhanced_tint_color); overlay=self._draw_flow_trails_simple(overlay,flow,prev,color,color,line_thickness_face,dot_radius_face)
            if self.pose_history:
                num_pose = len(self.pose_history); max_idx = max(num_pose - 1, 1)
                for idx, item in enumerate(reversed(self.pose_history)):
                    if len(item)==2 and item[0].shape[0]>0 and item[0].shape[0]==item[1].shape[0]:
                        flow, prev = item; alpha = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx)
                        color = tuple(int(c*alpha) for c in enhanced_tint_color); overlay=self._draw_flow_trails_simple(overlay,flow,prev,color,color,line_thickness_pose,dot_radius_pose)

            frame_out = cv2.add(frame_out, overlay)

            # Apply Enhanced Tint
            tint_strength_enhanced = min(1.0, TINT_STRENGTH * 1.2)
            tint_layer = np.full_like(frame_out, enhanced_tint_color, dtype=np.uint8)
            frame_out = cv2.addWeighted(frame_out, 1.0 - tint_strength_enhanced, tint_layer, tint_strength_enhanced, 0.0)

            # Apply Dynamic Glow
            if hasattr(results_pose, 'segmentation_mask') and results_pose.segmentation_mask is not None:
                try:
                    mask = results_pose.segmentation_mask; condition = (mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
                    if condition.shape[:2] != frame_out.shape[:2]: condition = cv2.resize(condition, (w,h), interpolation=cv2.INTER_NEAREST)
                    mask_blurred = cv2.GaussianBlur(condition, (31, 31), 0) # Wider blur
                    mask_alpha = mask_blurred.astype(np.float32) / 255.0
                    mask_alpha = (mask_alpha * self.glow_intensity * dynamic_glow_factor)
                    mask_alpha_3c = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)
                    glow_color_layer = np.full_like(frame_out, enhanced_tint_color, dtype=np.float32)
                    frame_float = frame_out.astype(np.float32); frame_float = (frame_float * (1.0 - mask_alpha_3c) + glow_color_layer * mask_alpha_3c)
                    frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
                except Exception as seg_e: logger.warning(f"GE seg/glow failed: {seg_e}")
            else: logger.debug("GE: No segmentation mask.")

            self.prev_gray = gray

        except Exception as e:
            self.prev_gray = None; self.prev_face_pts_gold = np.empty((0,1,2),dtype=np.float32); self.prev_pose_pts_gold = np.empty((0,1,2),dtype=np.float32)
            self.face_history.clear(); self.pose_history.clear()
            raise e
        return frame_out

    def _apply_goldenaura_style(self, frame: np.ndarray, frame_time: float = 0.0, **kwargs) -> np.ndarray:
        """Applies the selected Golden Aura variant."""
        original_frame = kwargs.get('original_frame', frame) # Get original frame for error text
        if self.goldenaura_variant == 0:
            frame_out = self._apply_goldenaura_original(frame, frame_time, original_frame=original_frame)
            # Check if error text was added by wrapper
            if not np.array_equal(frame_out, original_frame) or "ERROR" not in self.check_text_on_frame(frame_out, "Goldenaura"):
                 cv2.putText(frame_out, "Aura: Original", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return frame_out
        else:
            frame_out = self._apply_goldenaura_enhanced(frame, frame_time, original_frame=original_frame)
            if not np.array_equal(frame_out, original_frame) or "ERROR" not in self.check_text_on_frame(frame_out, "Goldenaura"):
                cv2.putText(frame_out, "Aura: Enhanced", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return frame_out

    # Helper to check if error text is already present (optional)
    def check_text_on_frame(self, frame, effect_name):
        # This is difficult and unreliable - better to rely on wrapper exception handling
        return ""


    def _apply_lightning_style_random(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.float32)
        lightning_color_float = np.array([255, 220, 200], dtype=np.float32) # BGR
        try:
            num_bolts = random.randint(1, 4)
            for _ in range(num_bolts):
                edge = random.choice(["top", "bottom", "left", "right"])
                if edge == "top": x0, y0 = random.randint(0, w - 1), random.randint(-20, 0)
                elif edge == "bottom": x0, y0 = random.randint(0, w - 1), random.randint(h, h + 20)
                elif edge == "left": x0, y0 = random.randint(-20, 0), random.randint(0, h - 1)
                else: x0, y0 = random.randint(w, w + 20), random.randint(0, h - 1)
                x1, y1 = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
                num_segments = random.randint(5, 15)
                px, py = float(x0), float(y0)
                for i in range(num_segments):
                    segment_len_factor = (num_segments - i) / num_segments
                    target_x = x0 + (x1 - x0) * (i + 1) / num_segments
                    target_y = y0 + (y1 - y0) * (i + 1) / num_segments
                    deviation = random.uniform(20, 60) * segment_len_factor
                    nx = target_x + random.uniform(-deviation, deviation)
                    ny = target_y + random.uniform(-deviation, deviation)
                    thickness = random.randint(1, 3)
                    cv2.line(overlay, (int(px), int(py)), (int(nx), int(ny)), tuple(lightning_color_float), thickness, cv2.LINE_AA)
                    px, py = nx, ny
            overlay_blurred = cv2.GaussianBlur(overlay, (7, 7), 0)
            overlay_glow = cv2.GaussianBlur(overlay, (25, 25), 0)
            frame_float = frame.astype(np.float32)
            frame_float = cv2.addWeighted(frame_float, 1.0, overlay_blurred, 1.5, 0.0) # Increased alpha
            frame_float = cv2.addWeighted(frame_float, 1.0, overlay_glow, 0.8, 0.0)    # Increased alpha
            frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
        except Exception as e: raise e
        return frame_out

    def _apply_lightning_style_movement(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        overlay = np.zeros_like(frame, dtype=np.float32)
        lightning_color_float = np.array([255, 220, 200], dtype=np.float32)
        try:
            if not self.face_mesh or not self.frame_width or not self.frame_height: return original_frame_for_error
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark: landmarks.append((int(lm.x * self.frame_width), int(lm.y * self.frame_height)))
            if not landmarks:
                self.prev_gray = gray; self.prev_landmarks_flow = None; return frame
            h, w = frame.shape[:2]
            if (self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks)):
                curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
                prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                valid_indices = np.where((prev_pts[:, 0, 0] >= 0) & (prev_pts[:, 0, 0] < w) & (prev_pts[:, 0, 1] >= 0) & (prev_pts[:, 0, 1] < h))[0]
                if len(valid_indices) > 0:
                    prev_pts_valid = prev_pts[valid_indices]
                    try:
                        flow, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS)
                        if flow is not None and status is not None:
                            good_new = flow[status.flatten() == 1]; good_old = prev_pts_valid[status.flatten() == 1]
                            if good_new.shape[0] > 0:
                                movement_vectors = good_new - good_old; magnitudes = np.linalg.norm(movement_vectors, axis=1)
                                movement_threshold = 5.0
                                for i in range(len(good_new)):
                                    if magnitudes[i] > movement_threshold:
                                        xn, yn = good_new[i].ravel(); xo, yo = good_old[i].ravel()
                                        thickness = random.randint(1, 2); jitter_x = random.uniform(-3, 3); jitter_y = random.uniform(-3, 3)
                                        cv2.line(overlay, (int(xo), int(yo)), (int(xn + jitter_x), int(yn + jitter_y)), tuple(lightning_color_float), thickness, cv2.LINE_AA)
                        else: logger.debug("Movement lightning: Optical flow failed.")
                    except cv2.error as cv_err: logger.warning(f"Movement lightning LK error: {cv_err}")
                else: logger.debug("Movement lightning: No valid previous points.")
            else: logger.debug("Movement lightning: No prev frame/landmarks or count mismatch.")
            overlay_blurred = cv2.GaussianBlur(overlay, (5, 5), 0)
            overlay_glow = cv2.GaussianBlur(overlay, (15, 15), 0)
            frame_float = frame.astype(np.float32)
            frame_float = cv2.addWeighted(frame_float, 1.0, overlay_blurred, 1.8, 0.0) # Increased alpha
            frame_float = cv2.addWeighted(frame_float, 1.0, overlay_glow, 1.0, 0.0)    # Increased alpha
            frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
            self.prev_landmarks_flow = landmarks
            self.prev_gray = gray
        except Exception as e:
            self.prev_gray = None; self.prev_landmarks_flow = None
            raise e
        return frame_out

    def _apply_lightning_style_branching(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.float32)
        lightning_color_float = np.array([255, 230, 210], dtype=np.float32)
        try:
            num_bolts = random.randint(1, 3)
            for _ in range(num_bolts):
                edge = random.choice(["top", "bottom", "left", "right"])
                if edge == "top": x0, y0 = random.randint(0, w - 1), random.randint(-20, 0)
                elif edge == "bottom": x0, y0 = random.randint(0, w - 1), random.randint(h, h + 20)
                elif edge == "left": x0, y0 = random.randint(-20, 0), random.randint(0, h - 1)
                else: x0, y0 = random.randint(w, w + 20), random.randint(0, h - 1)
                x1, y1 = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
                num_segments = random.randint(8, 20)
                px, py = float(x0), float(y0)
                current_angle = math.atan2(y1 - y0, x1 - x0)
                base_thickness = random.randint(2, 4)
                for i in range(num_segments):
                    dist_to_target = math.sqrt((x1-px)**2 + (y1-py)**2)
                    segment_len = max(10.0, dist_to_target / max(1, num_segments - i)) * random.uniform(0.8, 1.2)
                    current_angle += random.uniform(-0.4, 0.4)
                    nx = px + segment_len * math.cos(current_angle)
                    ny = py + segment_len * math.sin(current_angle)
                    thickness = max(1, int(base_thickness * ((num_segments - i) / num_segments)))
                    cv2.line(overlay, (int(px), int(py)), (int(nx), int(ny)), tuple(lightning_color_float), thickness, cv2.LINE_AA)
                    if random.random() < 0.3 and i < num_segments - 2:
                        branch_angle_offset = random.uniform(0.6, 1.2) * random.choice([-1, 1])
                        branch_angle = current_angle + branch_angle_offset
                        branch_length = segment_len * random.uniform(1.5, 3.0)
                        branch_end_x = nx + branch_length * math.cos(branch_angle)
                        branch_end_y = ny + branch_length * math.sin(branch_angle)
                        branch_thickness = max(1, thickness - 1)
                        cv2.line(overlay, (int(nx), int(ny)), (int(branch_end_x), int(branch_end_y)), tuple(lightning_color_float), branch_thickness, cv2.LINE_AA)
                    px, py = nx, ny
            overlay_blurred = cv2.GaussianBlur(overlay, (5, 5), 0)
            overlay_glow = cv2.GaussianBlur(overlay, (29, 29), 0)
            frame_float = frame.astype(np.float32)
            frame_float = cv2.addWeighted(frame_float, 1.0, overlay_blurred, 1.6, 0.0) # Increased alpha
            frame_float = cv2.addWeighted(frame_float, 1.0, overlay_glow, 1.0, 0.0)    # Increased alpha
            frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
        except Exception as e: raise e
        return frame_out

    def _apply_lightning_cycle(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        if not self.lightning_styles: logger.warning("No lightning styles available"); return original_frame_for_error
        try:
            style_function = self.lightning_styles[self.lightning_style_index]
            # Pass original frame in case the style function raises an error handled by wrapper
            frame_out = style_function(frame, original_frame=original_frame_for_error)
            style_name = self.lightning_style_names[self.lightning_style_index]
            # Only add text if no error occurred (wrapper adds error text)
            if "ERROR" not in self.check_text_on_frame(frame_out, "Lightning"):
                 cv2.putText(frame_out, f"Lightning: {style_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        except Exception as e: raise e # Let wrapper handle
        # Timing should ideally be inside the style function or wrapper
        return frame_out

    def _apply_neon_glow_style(self, frame: np.ndarray, frame_time: float, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        frame_out = frame.copy()
        try:
            if not self.pose or not self.frame_width or not self.frame_height: return original_frame_for_error
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = self.pose.process(frame_rgb)
            h, w = frame.shape[:2]
            overlay = np.zeros_like(frame_out, dtype=np.uint8)
            if results_pose.pose_landmarks:
                landmarks = []; connections = mp.solutions.pose.POSE_CONNECTIONS
                for id, lm in enumerate(results_pose.pose_landmarks.landmark):
                    landmarks.append((int(lm.x * w), int(lm.y * h))) if lm.visibility > 0.3 else landmarks.append(None)
                b = int(127 + 127 * math.sin(frame_time * 1.0)); g = int(127 + 127 * math.sin(frame_time * 1.0 + 2 * math.pi / 3)); r = int(127 + 127 * math.sin(frame_time * 1.0 + 4 * math.pi / 3))
                neon_color = (b, g, r); line_thickness = 3
                if connections:
                    for connection in connections:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            start_pt, end_pt = landmarks[start_idx], landmarks[end_idx]
                            if start_pt and end_pt: cv2.line(overlay, start_pt, end_pt, neon_color, line_thickness, cv2.LINE_AA)
                glow_overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
                wider_glow_overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
                frame_out = cv2.addWeighted(frame_out, 1.0, glow_overlay, 0.8, 0)
                frame_out = cv2.addWeighted(frame_out, 1.0, wider_glow_overlay, 0.4, 0)
            if hasattr(results_pose, 'segmentation_mask') and results_pose.segmentation_mask is not None:
                try:
                    mask = results_pose.segmentation_mask; condition = (mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
                    if condition.shape[:2] != frame_out.shape[:2]: condition = cv2.resize(condition, (w,h), interpolation=cv2.INTER_NEAREST)
                    mask_blurred = cv2.GaussianBlur(condition, (31, 31), 0)
                    mask_alpha = (mask_blurred.astype(np.float32) / 255.0 * self.glow_intensity * 0.5)
                    mask_alpha_3c = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)
                    seg_glow_layer = np.full_like(frame_out, neon_color, dtype=np.float32)
                    frame_float = frame_out.astype(np.float32); frame_float = (frame_float * (1.0 - mask_alpha_3c) + seg_glow_layer * mask_alpha_3c)
                    frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
                except Exception as seg_e: logger.warning(f"Neon glow seg failed: {seg_e}")
        except Exception as e: raise e
        return frame_out

    def _apply_particle_trail_style(self, frame: np.ndarray, frame_time: float, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        frame_out = frame.copy()
        hands_detected_this_frame = False
        try:
            if not self.hands or not self.frame_width or not self.frame_height: return original_frame_for_error
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(frame_rgb)
            h, w = frame.shape[:2]
            if results_hands.multi_hand_landmarks:
                hands_detected_this_frame = True
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    landmark_to_track = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    x, y = int(landmark_to_track.x * w), int(landmark_to_track.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        num_particles = 20; max_offset = 30; min_size, max_size = 1, 5
                        for _ in range(num_particles):
                            offset_dist = random.gauss(0, max_offset / 2.5); offset_angle = random.uniform(0, 2 * math.pi)
                            offset_x = int(x + offset_dist * math.cos(offset_angle)); offset_y = int(y + offset_dist * math.sin(offset_angle))
                            if 0 <= offset_x < w and 0 <= offset_y < h:
                                particle_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                                particle_size = random.randint(min_size, max_size)
                                cv2.circle(frame_out, (offset_x, offset_y), particle_size, particle_color, -1, cv2.LINE_AA)
            # Only show message if no hands were detected in this specific frame
            if not hands_detected_this_frame:
                cv2.putText(frame_out, "No Hands Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        except Exception as e: raise e
        return frame_out

    def _apply_motion_blur_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        frame_out = frame # Default if buffer empty
        try:
            self.frame_buffer.append(frame.astype(np.float32))
            if len(self.frame_buffer) > 1:
                avg_frame = np.mean(np.array(self.frame_buffer), axis=0)
                frame_out = np.clip(avg_frame, 0, 255).astype(np.uint8)
        except Exception as e: raise e
        return frame_out


    def _apply_chromatic_aberration_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        try:
            b, g, r = cv2.split(frame); shift = 3; h, w = frame.shape[:2]
            b_shifted = np.zeros_like(b); g_shifted = g.copy(); r_shifted = np.zeros_like(r)
            b_shifted[:, shift:] = b[:, :-shift]; b_shifted[:, :shift] = b[:, :shift] # Fill gap
            r_shifted[:, :-shift] = r[:, shift:]; r_shifted[:, -shift:] = r[:, -shift:] # Fill gap
            frame_out = cv2.merge((b_shifted, g_shifted, r_shifted))
        except Exception as e: raise e
        return frame_out

    def _apply_rvm_composite_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        frame_out = frame.copy()
        model_name = self.rvm_model_names[self.current_rvm_model_idx % len(self.rvm_model_names)]
        model = self.rvm_models.get(model_name)
        if not self.rvm_available or not model:
            # Caught by wrapper
             return original_frame_for_error
        try:
            src = self._preprocess_frame_rvm(frame)
            if src is None: raise ValueError("RVM Preprocessing failed")
            with torch.no_grad():
                rec = [r.to(self.device) if r is not None else None for r in self.rvm_rec]
                fgr, pha, *rec = model(src, *rec, downsample_ratio=self.rvm_downsample_ratio)
                self.rvm_rec = [r.cpu() if r is not None else None for r in rec]
            fgr_np, pha_np = self._postprocess_output_rvm(fgr, pha)
            if fgr_np is None or pha_np is None: raise ValueError("RVM Postprocessing failed")
            if fgr_np.shape[:2] != frame.shape[:2]:
                fgr_np = resize_frame(fgr_np, (frame.shape[0], frame.shape[1]))
                pha_np = cv2.resize(pha_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            mode = self.RVM_DISPLAY_MODES[self.rvm_display_mode]
            if mode == "Alpha": frame_out = cv2.cvtColor(pha_np, cv2.COLOR_GRAY2BGR)
            elif mode == "Foreground":
                black_bg = np.zeros_like(frame_out); alpha_f = pha_np.astype(np.float32)/255.0; alpha_3c = cv2.cvtColor(alpha_f, cv2.COLOR_GRAY2BGR)
                frame_out = (alpha_3c * fgr_np + (1.0 - alpha_3c) * black_bg).astype(np.uint8)
            else: # Composite
                if self.rvm_background is None: self._load_rvm_background(self._init_rvm_bg_path)
                bg = self.rvm_background
                if bg.shape[:2] != frame.shape[:2]: bg = resize_frame(bg, (frame.shape[0], frame.shape[1])); self.rvm_background = bg
                alpha_f = pha_np.astype(np.float32) / 255.0; alpha_3c = cv2.cvtColor(alpha_f, cv2.COLOR_GRAY2BGR)
                frame_out = (alpha_3c * fgr_np + (1.0 - alpha_3c) * bg).astype(np.uint8)
            # Only add text if no error occurred
            cv2.putText(frame_out, f"RVM: {model_name} ({mode})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e: raise e
        return frame_out

    def _apply_sam_segmentation_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Applies SAM-based segmentation style to the frame."""
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_for_error = kwargs.get('original_frame', frame)
        frame_out = frame.copy()
        sam1_ready = self.sam_model_v1 is not None
        sam2_ready = self.sam_model_v2 is not None
        sam_model_to_use = None
        sam_version_name = "N/A"

        # Prefer SAM v1 if loaded, else fall back to SAM v2
        if sam1_ready:
            sam_model_to_use = self.sam_model_v1
            sam_version_name = "SAM v1 (AutoMask)"
        elif sam2_ready:
            sam_model_to_use = self.sam_model_v2
            sam_version_name = "SAM v2 (Ultralytics)"
        else:
            logger.warning("No SAM model loaded")
            return original_frame_for_error

        logger.debug(f"Attempting SAM segmentation using {sam_version_name}")
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = None
            if sam_version_name == "SAM v1 (AutoMask)":
                masks = sam_model_to_use.generate(rgb_frame)
                logger.debug(f"SAM v1 generated {len(masks)} masks")
            elif sam_version_name == "SAM v2 (Ultralytics)":
                # SAM v2 requires a prompt; use full image as default
                h, w = frame.shape[:2]
                with torch.no_grad():
                    results = sam_model_to_use.predict(
                        rgb_frame, bboxes=[0, 0, w, h], device=self.device, verbose=False
                    )
                if results and len(results) > 0:
                    sam2_results = results[0]
                    if sam2_results.masks is not None:
                        masks = sam2_results.masks.data
                        logger.debug(f"SAM v2 generated {masks.shape[0]} masks")
                    else:
                        logger.debug("SAM v2 generated no masks")
                else:
                    logger.debug("SAM v2 prediction failed")

            if masks is not None:
                frame_out = self._draw_sam_masks(frame, masks)
                if not np.array_equal(frame_out, frame):
                    cv2.putText(
                        frame_out, f"{sam_version_name} Seg.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
            else:
                cv2.putText(
                    frame_out, f"{sam_version_name} No Masks", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )
        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}", exc_info=True)
            raise e
        return frame_out

    # --- Helper Methods ---

    def _draw_flow_trails_simple(self, frame: np.ndarray, flow_points: np.ndarray, prev_points: np.ndarray, line_color: Tuple[int, int, int], dot_color: Tuple[int, int, int], line_thickness: int = 1, dot_radius: int = 1) -> np.ndarray:
        try:
            if flow_points.shape[0] != prev_points.shape[0]: logger.warning("Mismatched flow/prev points"); return frame
            h, w = frame.shape[:2]
            for i in range(flow_points.shape[0]):
                xn, yn = flow_points[i].ravel(); xo, yo = prev_points[i].ravel()
                pt1 = (int(xo), int(yo)); pt2 = (int(xn), int(yn))
                cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
                cv2.circle(frame, pt2, dot_radius, dot_color, -1, cv2.LINE_AA)
        except Exception as e: logger.warning(f"Draw flow trails failed: {e}")
        return frame

    def _draw_motion_glow_separate(self, frame: np.ndarray, landmarks: List[Tuple[int, int]], flow_points: np.ndarray, prev_points: np.ndarray) -> np.ndarray:
        frame = validate_frame(frame)
        try:
            glow_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            if flow_points.shape[0] != prev_points.shape[0]: logger.warning("Mismatched glow points"); return frame
            h, w = frame.shape[:2]
            for i in range(flow_points.shape[0]):
                xn, yn = flow_points[i].ravel(); xo, yo = prev_points[i].ravel()
                pt1 = (int(xo), int(yo)); pt2 = (int(xn), int(yn))
                cv2.line(glow_mask, pt1, pt2, 255, 4, cv2.LINE_AA)
            glow_mask_blurred = cv2.GaussianBlur(glow_mask, (25, 25), 0)
            glow_alpha = (glow_mask_blurred.astype(np.float32) / 255.0 * self.glow_intensity)
            glow_alpha_3c = cv2.cvtColor(glow_alpha, cv2.COLOR_GRAY2BGR)
            glow_color_layer = np.full_like(frame, (255, 255, 255), dtype=np.float32)
            frame_float = frame.astype(np.float32); frame_float = (frame_float * (1.0 - glow_alpha_3c) + glow_color_layer * glow_alpha_3c)
            frame = np.clip(frame_float, 0, 255).astype(np.uint8)
        except Exception as e: logger.warning(f"Draw motion glow failed: {e}")
        return frame

    def _apply_distortion(self, frame: np.ndarray) -> np.ndarray:
        frame = validate_frame(frame)
        try:
            h, w = frame.shape[:2]; center_x, center_y = w / 2, h / 2
            k1 = 0.0000001; k2 = 0.0; p1 = 0.0; p2 = 0.0
            cam_matrix = np.array([[w, 0, center_x], [0, h, center_y], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)
            frame_distorted = cv2.undistort(frame, cam_matrix, dist_coeffs)
            return frame_distorted
        except Exception as e: logger.warning(f"Distortion effect failed: {e}"); return frame

    def _detect_gestures(self, frame: np.ndarray) -> Optional[str]:
        if not self.hands or not self.frame_width or not self.frame_height: return None
        gesture = None
        try:
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]; lm = hand_landmarks.landmark
                thumb_tip=lm[mp.solutions.hands.HandLandmark.THUMB_TIP]; index_tip=lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip=lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]; ring_tip=lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip=lm[mp.solutions.hands.HandLandmark.PINKY_TIP]; index_pip=lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
                middle_pip=lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]; ring_pip=lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP]
                pinky_pip=lm[mp.solutions.hands.HandLandmark.PINKY_PIP]; thumb_ip=lm[mp.solutions.hands.HandLandmark.THUMB_IP]
                thumb_mcp=lm[mp.solutions.hands.HandLandmark.THUMB_MCP]
                fingers_extended = 0
                if index_tip.y < index_pip.y: fingers_extended += 1
                if middle_tip.y < middle_pip.y: fingers_extended += 1
                if ring_tip.y < ring_pip.y: fingers_extended += 1
                if pinky_tip.y < pinky_pip.y: fingers_extended += 1
                if thumb_tip.y < thumb_ip.y and thumb_tip.x < thumb_mcp.x: fingers_extended += 1 # Basic check
                fingers_closed = 0
                if index_tip.y > index_pip.y: fingers_closed += 1
                if middle_tip.y > middle_pip.y: fingers_closed += 1
                if ring_tip.y > ring_pip.y: fingers_closed += 1
                if pinky_tip.y > pinky_pip.y: fingers_closed += 1
                if thumb_tip.y > thumb_ip.y and thumb_tip.x > index_pip.x: fingers_closed += 1 # Basic check
                if fingers_extended >= 4: gesture = "open_hand"; logger.debug("Detected gesture: Open hand")
                elif fingers_closed >= 4: gesture = "fist"; logger.debug("Detected gesture: Fist")
        except Exception as e: logger.warning(f"Gesture detection failed: {e}"); gesture = None
        return gesture

    def _load_rvm_background(self, bg_path: Optional[str]) -> None:
        if self.rvm_background is not None and bg_path == self._init_rvm_bg_path:
            if self.rvm_background.shape[:2] != (self.frame_height, self.frame_width):
                logger.info("Resizing cached RVM background."); self.rvm_background = resize_frame(self.rvm_background, (self.frame_height, self.frame_width))
            return
        loaded_bg = None
        if bg_path and os.path.exists(bg_path):
            try:
                loaded_bg = cv2.imread(bg_path)
                if loaded_bg is not None: loaded_bg = validate_frame(loaded_bg); logger.info(f"Loaded RVM background from {bg_path}")
                else: logger.warning(f"cv2.imread failed for RVM background: {bg_path}")
            except Exception as e: logger.warning(f"Failed to load RVM background image '{bg_path}': {e}"); loaded_bg = None
        if loaded_bg is None:
            h, w = self.frame_height, self.frame_width; self.rvm_background = np.full((h, w, 3), RVM_DEFAULT_BG_COLOR, dtype=np.uint8)
            logger.info(f"Using default {RVM_DEFAULT_BG_COLOR} RVM background ({w}x{h})")
        else: self.rvm_background = resize_frame(loaded_bg, (self.frame_height, self.frame_width))
        self._init_rvm_bg_path = bg_path

    def _preprocess_frame_rvm(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        try:
            frame = validate_frame(frame); frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transform = T.ToTensor(); src = transform(frame_rgb).unsqueeze(0).to(self.device)
            return src
        except Exception as e: logger.warning(f"RVM preprocess failed: {e}"); return None

    def _postprocess_output_rvm(self, fgr: torch.Tensor, pha: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            fgr_np = fgr.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
            pha_np = pha.cpu().detach().squeeze(0).squeeze(0).numpy()
            fgr_out = np.clip(fgr_np * 255.0, 0, 255).astype(np.uint8)
            pha_out = np.clip(pha_np * 255.0, 0, 255).astype(np.uint8)
            return fgr_out, pha_out
        except Exception as e: logger.warning(f"RVM postprocess failed: {e}"); return None, None

    def _draw_sam_masks(self, frame: np.ndarray, masks: Any) -> np.ndarray:
        start_time = time.time()
        frame = validate_frame(frame)
        overlay = frame.copy()
        if masks is None: logger.debug("No SAM masks to draw"); return frame
        try:
            h, w = frame.shape[:2]; num_masks_drawn = 0; masks_to_process = []
            # Standardize Mask Input
            if isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], dict) and "segmentation" in masks[0]: # SAM v1
                masks_to_process = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
                masks_to_process = [m["segmentation"] for m in masks_to_process]
            elif _sam2_available and UltralyticsSAM and isinstance(masks, UltralyticsSAM.Results) and masks.masks is not None: # SAM v2
                sam2_masks = masks.masks.data
                if isinstance(sam2_masks, torch.Tensor): masks_to_process = [m.cpu().numpy().astype(bool) for m in sam2_masks]
                else: logger.warning(f"Unexpected SAM v2 mask data type: {type(sam2_masks)}"); return frame
            elif isinstance(masks, torch.Tensor): # Raw Tensor
                if masks.ndim == 3: masks_to_process = [m.cpu().numpy().astype(bool) for m in masks]
                elif masks.ndim == 2: masks_to_process = [masks.cpu().numpy().astype(bool)]
                else: logger.warning(f"Unsupported raw tensor shape: {masks.shape}"); return frame
            elif isinstance(masks, np.ndarray): # Raw Numpy
                if masks.ndim == 3: masks_to_process = [m.astype(bool) for m in masks]
                elif masks.ndim == 2: masks_to_process = [masks.astype(bool)]
                else: logger.warning(f"Unsupported raw numpy shape: {masks.shape}"); return frame
            else: logger.warning(f"Unsupported SAM mask input type: {type(masks)}"); return frame

            if not masks_to_process: logger.debug("No valid SAM masks after processing."); return frame

            # Draw Masks
            for i, mask_data in enumerate(masks_to_process):
                if mask_data is None or mask_data.size == 0: continue
                try:
                    mask_bool = mask_data.astype(bool)
                    if mask_bool.shape != (h, w): mask_bool = cv2.resize(mask_bool.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
                    if not np.any(mask_bool): continue
                    color = [random.randint(64, 200) for _ in range(3)] # BGR
                    overlay[mask_bool] = color; num_masks_drawn += 1
                except Exception as draw_e: logger.warning(f"Failed to process/draw SAM mask {i}: {draw_e}")

            if num_masks_drawn > 0: frame_out = cv2.addWeighted(frame, 1.0 - SAM_MASK_ALPHA, overlay, SAM_MASK_ALPHA, 0); logger.debug(f"Drew {num_masks_drawn} SAM masks")
            else: logger.debug("No SAM masks were drawn."); frame_out = frame
        except Exception as e:
            logger.warning(f"SAM mask drawing failed: {e}", exc_info=True)
            cv2.putText(frame, "SAM Draw Error", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frame_out = frame
        return frame_out

    # --- Video Capture and Processing Loop ---

    def _initialize_capture(self) -> bool:
        logger.info(f"Initializing capture from: {self.input_source}")
        try:
            self.cap = cv2.VideoCapture(self.input_source)
            if not self.cap.isOpened(): logger.error(f"Failed to open input source: {self.input_source}"); return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height); self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0: logger.warning(f"Capture FPS invalid ({self.fps}). Using default: {DEFAULT_FPS}"); self.fps = DEFAULT_FPS
            logger.info(f"Capture source opened: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS (requested {self.desired_width}x{self.desired_height} @ {self.desired_fps:.2f})")
            ret, frame = self.cap.read()
            if not ret or frame is None: logger.error("Failed to read initial frame."); self.cap.release(); return False
            logger.info("Initial frame read successfully.")
            if self.mode == "record":
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True); fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
                if not self.out.isOpened(): logger.error(f"Failed to initialize video writer: {self.output_path}"); self.cap.release(); return False
                logger.info(f"Recording enabled: {self.output_path}")
            if self.display:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                cv2.createTrackbar("Brightness", self.window_name, 100, 200, lambda x: setattr(self, "brightness_factor", (x - 100) / 100.0))
                cv2.createTrackbar("Contrast", self.window_name, 100, 200, lambda x: setattr(self, "contrast_factor", x / 100.0))
                cv2.createTrackbar("LUT Intensity", self.window_name, 100, 100, lambda x: setattr(self, "lut_intensity", x / 100.0))
                logger.info("Display window initialized.")
            return True
        except Exception as e:
            logger.error(f"Capture initialization failed: {e}", exc_info=True)
            if self.cap and self.cap.isOpened(): self.cap.release()
            if self.out and self.out.isOpened(): self.out.release()
            return False

    def _cleanup(self) -> None:
        logger.info("Starting cleanup...")
        try:
            if self.cap and self.cap.isOpened(): self.cap.release(); logger.info("Video capture released.")
            if self.out and self.out.isOpened(): self.out.release(); logger.info("Video writer released.")
            if _mediapipe_available:
                if hasattr(self, 'face_mesh') and self.face_mesh and hasattr(self.face_mesh, 'close'): self.face_mesh.close()
                if hasattr(self, 'pose') and self.pose and hasattr(self.pose, 'close'): self.pose.close()
                if hasattr(self, 'hands') and self.hands and hasattr(self.hands, 'close'): self.hands.close()
                logger.info("Mediapipe resources closed.")
            cv2.destroyAllWindows(); logger.info("OpenCV windows destroyed.")
            logger.info("Cleanup completed.")
        except Exception as e: logger.warning(f"Error during cleanup: {e}", exc_info=True)

    @staticmethod
    def list_cameras() -> List[Tuple[int, str]]:
        available_cameras = []; logger.info("Scanning for available cameras (indices 0-9)...")
        for index in range(10):
            cap_test = cv2.VideoCapture(index)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret: backend = cap_test.getBackendName(); name = f"Camera {index} ({backend})"; available_cameras.append((index, name)); logger.info(f"  Found: {name}")
                else: logger.debug(f"  Index {index} opened but failed read.")
                cap_test.release()
        if not available_cameras: logger.warning("No cameras detected."); return available_cameras

    def _handle_gestures(self, gesture: Optional[str], frame_time: float) -> None:
        if not gesture: return
        current_time = time.time(); gesture_cooldown = 1.5
        if (self.gesture_state["last_gesture"] == gesture and current_time - self.gesture_state["last_time"] < gesture_cooldown): return
        logger.info(f"Gesture '{gesture}' detected."); self.gesture_state["last_gesture"] = gesture; self.gesture_state["last_time"] = current_time; self.gesture_state["gesture_count"] += 1
        if gesture == "open_hand":
            effect_keys = list(self.effects.keys()); current_idx = effect_keys.index(self.current_effect)
            next_idx = (current_idx + 1) % len(effect_keys); self.current_effect = effect_keys[next_idx]
            logger.info(f"Gesture 'open_hand': Switched to effect -> {self.current_effect}")
        elif gesture == "fist":
            if self.current_effect == "goldenaura":
                self.goldenaura_variant = (self.goldenaura_variant + 1) % len(self.goldenaura_variant_names)
                logger.info(f"Gesture 'fist': Switched Goldenaura variant -> {self.goldenaura_variant_names[self.goldenaura_variant]}")
            elif self.current_effect == "lightning":
                self.lightning_style_index = (self.lightning_style_index + 1) % len(self.lightning_styles)
                logger.info(f"Gesture 'fist': Switched Lightning style -> {self.lightning_style_names[self.lightning_style_index]}")
            elif self.current_effect == "lut_color_grade" and self.lut_color_grade_effect: self.lut_color_grade_effect.cycle_lut()
            elif self.current_effect == "rvm_composite":
                self.rvm_display_mode = (self.rvm_display_mode + 1) % len(self.RVM_DISPLAY_MODES)
                logger.info(f"Gesture 'fist': Switched RVM display mode -> {self.RVM_DISPLAY_MODES[self.rvm_display_mode]}")
            elif self.current_effect == "sam_segmentation":
                sam1_ready = self.sam_model_v1 is not None; sam2_ready = self.sam_model_v2 is not None
                if sam1_ready and sam2_ready:
                    self.use_sam2_runtime = not self.use_sam2_runtime
                    logger.info(f"Gesture 'fist': Switched SAM runtime -> {'SAM v2' if self.use_sam2_runtime else 'SAM v1'}")
                else: logger.info("Gesture 'fist': Cannot switch SAM runtime.")
            else: logger.info(f"Gesture 'fist': No action for '{self.current_effect}'.")

    def run(self) -> None:
        main_start_time = time.time()
        if not self._initialize_capture(): logger.error("Capture init failed. Exiting."); return
        # Load Models
        if self.rvm_available:
            for name in self.rvm_model_names:
                 if name not in self.rvm_models: model = load_rvm_model(self.device, name); self.rvm_models[name] = model if model else None
        if self.sam_available and self.sam_checkpoint_path and not self.sam_model_v1: self.sam_model_v1 = load_sam_mask_generator(self.device, checkpoint_path=self.sam_checkpoint_path)
        if self.sam2_available and self.sam2_checkpoint_path and not self.sam_model_v2: self.sam_model_v2 = load_sam2_video_predictor(self.device, checkpoint_path=self.sam2_checkpoint_path)
        self.use_sam2_runtime = (self.sam_model_v2 is not None) # Update preference based on loaded models

        frame_count = 0
        try:
            while True:
                loop_start_time = time.time()
                ret, frame = self.cap.read()
                if not ret or frame is None: logger.info("Input ended or read error."); break
                frame = validate_frame(frame)
                if frame.shape[0]!=self.frame_height or frame.shape[1]!=self.frame_width: frame=resize_frame(frame, (self.frame_height, self.frame_width))
                original_frame = frame.copy() # Keep a copy for error display

                # --- Apply Effect ---
                if self.current_effect not in self.effects: self.current_effect = "none"
                effect_wrapper = self.effects[self.current_effect]
                frame_out = effect_wrapper(frame=frame.copy(), frame_time=loop_start_time, runner=self, original_frame=original_frame)
                frame_out = validate_frame(frame_out) # Validate effect output

                # --- Global Adjustments ---
                if self.brightness_factor != 0.0 or self.contrast_factor != 1.0:
                    beta = self.brightness_factor * 100
                    frame_out = cv2.convertScaleAbs(frame_out, alpha=self.contrast_factor, beta=beta)

                # --- Info Overlay ---
                current_time = time.time(); self.frame_times.append(current_time - loop_start_time)
                if len(self.frame_times) > 1: avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if len(self.frame_times) > 0 else 0; fps_text = f"FPS: {avg_fps:.1f}"
                else: fps_text = "FPS: N/A"
                effect_text = f"Effect: {self.current_effect}"
                cv2.putText(frame_out, fps_text, (10, self.frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_out, effect_text, (10, self.frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- Output/Display ---
                if self.out: self.out.write(frame_out)
                if self.display:
                    try: cv2.imshow(self.window_name, frame_out)
                    except cv2.error as e: logger.warning(f"Display failed: {e}"); self.display = False

                frame_count += 1
                key = cv2.waitKey(1) & 0xFF

                # --- Handle Keys ---
                if key == ord("q"): break
                elif key in [ord("e"), ord("w")]: # Cycle effects
                    direction = 1 if key == ord("e") else -1
                    effect_keys = list(self.effects.keys()); current_idx = effect_keys.index(self.current_effect)
                    next_idx = (current_idx + direction + len(effect_keys)) % len(effect_keys)
                    self.current_effect = effect_keys[next_idx]; logger.info(f"Key '{chr(key)}': Effect -> {self.current_effect}")
                    self.rvm_rec = [None] * 4; self.face_history.clear(); self.pose_history.clear(); # Reset state on switch
                elif key == ord("v"): # Cycle variant
                     if self.current_effect == "goldenaura": self.goldenaura_variant = (self.goldenaura_variant + 1) % len(self.goldenaura_variant_names); logger.info(f"Key 'v': Goldenaura variant -> {self.goldenaura_variant_names[self.goldenaura_variant]}")
                     elif self.current_effect == "lightning": self.lightning_style_index = (self.lightning_style_index + 1) % len(self.lightning_styles); logger.info(f"Key 'v': Lightning style -> {self.lightning_style_names[self.lightning_style_index]}")
                     elif self.current_effect == "lut_color_grade" and self.lut_color_grade_effect: self.lut_color_grade_effect.cycle_lut()
                     elif self.current_effect == "rvm_composite": self.rvm_display_mode = (self.rvm_display_mode + 1) % len(self.RVM_DISPLAY_MODES); logger.info(f"Key 'v': RVM display -> {self.RVM_DISPLAY_MODES[self.rvm_display_mode]}")
                     elif self.current_effect == "sam_segmentation":
                         sam1_r, sam2_r = self.sam_model_v1 is not None, self.sam_model_v2 is not None
                         if sam1_r and sam2_r: self.use_sam2_runtime = not self.use_sam2_runtime; logger.info(f"Key 'v': SAM runtime -> {'SAM v2' if self.use_sam2_runtime else 'SAM v1'}")
                         else: logger.info("Key 'v': Cannot switch SAM (only one loaded).")
                     else: logger.info(f"Key 'v': No variant for '{self.current_effect}'.")
                elif key == ord("m"): # Cycle Model
                    if self.current_effect == "rvm_composite" and len(self.rvm_models) > 1:
                        self.current_rvm_model_idx = (self.current_rvm_model_idx + 1) % len(self.rvm_model_names)
                        model_name = self.rvm_model_names[self.current_rvm_model_idx]
                        if model_name not in self.rvm_models or self.rvm_models[model_name] is None: logger.warning(f"RVM {model_name} not loaded."); self.current_rvm_model_idx = (self.current_rvm_model_idx - 1 + len(self.rvm_model_names)) % len(self.rvm_model_names)
                        else: logger.info(f"Key 'm': RVM model -> {model_name}"); self.rvm_rec = [None] * 4
                elif key == ord("r"): # Reset
                    logger.info("Key 'r': Resetting..."); [e.reset() for e in self.effects.values()]
                    self.current_effect = "none"; self.brightness_factor = 0.0; self.contrast_factor = 1.0; self.lut_intensity = 1.0
                    self.prev_gray = None; self.prev_landmarks_flow = None; self.face_history.clear(); self.pose_history.clear(); self.frame_buffer.clear(); self.rvm_rec = [None] * 4
                    if self.display:
                        try: cv2.setTrackbarPos("Brightness",self.window_name,100); cv2.setTrackbarPos("Contrast",self.window_name,100); cv2.setTrackbarPos("LUT Intensity",self.window_name,100)
                        except cv2.error: logger.warning("Trackbar reset failed.")
                    logger.info("Reset complete.")

        except KeyboardInterrupt: logger.info("KeyboardInterrupt. Exiting.")
        except Exception as e: logger.error(f"Unhandled loop exception: {e}", exc_info=True)
        finally:
            self._cleanup()
            total_time = time.time() - main_start_time
            if frame_count > 0 and len(self.frame_times) > 0:
                avg_fps = frame_count / total_time if total_time > 0 else 0 # Overall FPS
                logger.info("--- Processing Summary ---")
                logger.info(f"Frames: {frame_count}, Time: {total_time:.2f}s, Avg FPS: {avg_fps:.2f}")
                logger.info(f"Errors: {self.error_count}")
            else: logger.info("No frames processed or loop exited early.")


# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixelSensing Video Effects Runner", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=str, default=CONFIG_FILE, help="Path to config JSON file")
    parser.add_argument("--input-source", type=str, help="Override config: Camera index or video file path")
    parser.add_argument("--output-path", type=str, help="Override config: Output video file path (for record mode)")
    parser.add_argument("--lut-dir", type=str, help="Override config: Directory containing LUT files (.cube)")
    parser.add_argument("--rvm-background-path", type=str, help="Override config: Path to RVM background image")
    parser.add_argument("--sam-checkpoint-path", type=str, help="Override config: Path to SAM v1 checkpoint (.pth)")
    parser.add_argument("--sam2-checkpoint-path", type=str, help="Override config: Path to SAM 2 (ultralytics) checkpoint (.pt)")
    parser.add_argument("--mode", type=str, choices=["live", "record"], help="Override config: Operation mode")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, help="Override config: Display video output (--display / --no-display)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Override config: Device for ML models")
    parser.add_argument("--trail-length", type=int, help="Override config: Trail length for effects like Goldenaura")
    parser.add_argument("--glow-intensity", type=float, help="Override config: Glow intensity (0.0-1.0)")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera devices and exit")
    parser.epilog = """
==============================
PixelSensingFX Controls Manual
==============================
(Same manual content as before)
... [Rest of the controls manual text] ...
"""
    args = parser.parse_args()

    if args.list_cameras: PixelSensingEffectRunner.list_cameras(); exit()

    runner = PixelSensingEffectRunner(config_path=args.config)
    config_to_save = runner.config.copy()
    overridden_keys = []
    for key, value in vars(args).items():
        if (value is not None and key in config_to_save and key != "config" and key != "list_cameras"):
            config_to_save[key] = value; overridden_keys.append(key)
    if overridden_keys:
        logger.info(f"Overriding config with CLI args: {', '.join(overridden_keys)}")
        runner.config = config_to_save; runner._apply_config()
        if "lut_dir" in overridden_keys: runner._load_luts()
        if "device" in overridden_keys:
             logger.info("Device changed via CLI, models will be reloaded.")
             runner.rvm_models = {}; runner.sam_model_v1 = None; runner.sam_model_v2 = None
             runner._initialize_effects() # Re-init effects for dep checks if needed
    runner.run()


    """
==============================
PixelSensingFX Controls Manual
==============================

General Controls:
-----------------
  Q: Quit the application.
  E: Cycle FORWARD through available effects.
  W: Cycle BACKWARD through available effects.
  R: Reset all effects, variants, trackbar settings, and internal states.

Effect-Specific Controls (Press 'V'):
-------------------------------------
  V: Cycle through variants or sub-modes for the CURRENT active effect.
     - Goldenaura: Toggles between "Original" and "Enhanced" variants.
     - Lightning: Cycles through "Random", "Movement", and "Branching" styles.
     - LUT Color Grade: Cycles to the NEXT loaded LUT file (.cube).
     - RVM Composite: Cycles through display modes: "Composite" -> "Alpha" -> "Foreground".
     - SAM Segmentation: Toggles between SAM v1 and SAM v2 models (if both checkpoints provided).
     - Other Effects: May have no variants (message will be logged).

Model Switching (Press 'M'):
---------------------------
  M: Cycle through available models for the CURRENT active effect (if applicable).
     - RVM Composite: Switches between "mobilenetv3" and "resnet50" (if both loaded).

Trackbar Controls (In Display Window):
--------------------------------------
  Brightness: Adjusts overall frame brightness (-1.0 to +1.0). Default: 0.0 (center).
  Contrast: Adjusts overall frame contrast (0.0 to 2.0). Default: 1.0 (center).
  LUT Intensity: Blends the LUT effect with the original frame (0% to 100%). Default: 100%. Affects only "lut_color_grade" effect.

Configuration (`config.json` or Command Line):
---------------------------------------------
  - See `python your_script_name.py --help` for command-line overrides.
  - Key `config.json` settings:
    - `input_source`: "0", "1", etc. for camera, or "/path/to/video.mp4".
    - `mode`: "live" (webcam/realtime) or "record" (process and save to file).
    - `output_path`: File path for "record" mode (e.g., "output.mp4").
    - `display`: `true` to show the OpenCV window, `false` to run headless.
    - `device`: "cpu", "cuda", or "mps" for ML model inference.
    - `lut_dir`: Folder containing .cube LUT files.
    - `*_checkpoint_path`: Paths to model files (RVM, SAM v1, SAM v2).
    - `trail_length`, `glow_intensity`: Parameters for specific effects.

Notes:
------
- Effects requiring specific libraries (Mediapipe, PyTorch, colour-science, segment-anything, ultralytics) or models will show an "UNAVAILABLE" message if dependencies are missing.
- Logs are saved to the `logs/` directory with timestamps. Check these for detailed information and errors.
- Performance (FPS) depends heavily on the selected effect, model, resolution, and hardware (especially the `device` setting).
- Some effects build state over time (e.g., trails, motion blur). Use 'R' to clear this state.
"""
#list cameras python video_effectsfinale4.py --list-cameras
#python video_effectsfinale4.py --lut-dir "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/Videous/70 CGC LUTs" --device mps --mode live --display --input-source 0 --sam2-checkpoint-path "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/segment-anything-2/checkpoints/sam_vit_b_01ec64.pth"