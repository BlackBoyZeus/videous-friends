# --- Import cv2 --- # noqa (Comment to satisfy linters if cv2 is conditionally imported elsewhere)
import cv2
import numpy as np
import abc
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import logging
import mediapipe as mp
from collections import deque
import os
import random # <<< ADDED: For lightning generation >>>
import torch
import torchvision.transforms as T

# --- Imports with Fallbacks ---
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _sam_available = True
except ImportError:
    _sam_available = False
    SamAutomaticMaskGenerator = None
    sam_model_registry = None

try:
    from ultralytics import SAM
    _sam2_available = True
except ImportError:
    _sam2_available = False
    SAM = None

# --- Device Setup ---
_torch_device = 'cpu'
if torch.cuda.is_available():
    _torch_device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    try:
        tensor_mps = torch.tensor([1.0], device='mps')
        _ = tensor_mps * tensor_mps
        _torch_device = 'mps'
        print("--- MPS device detected and verified. ---")
    except Exception as mps_error:
        print(f"--- MPS device detected but check failed: {mps_error}. Falling back to CPU. ---")
        _torch_device = 'cpu'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

logger.info(f"PyTorch available: {torch.__version__ is not None}")
logger.info(f"segment_anything (SAM v1) available: {_sam_available}")
logger.info(f"ultralytics.SAM (for SAM2) available: {_sam2_available}")
logger.info(f"Selected PyTorch device: {_torch_device}")

DEFAULT_HEIGHT, DEFAULT_WIDTH = 480, 640

# ========================================================================
#                       Utility Functions
# ========================================================================
def validate_frame(frame: Optional[np.ndarray], default_shape: Tuple[int, int, int] = (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)) -> np.ndarray:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return np.zeros(default_shape, dtype=np.uint8)
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        logger.warning(f"Frame with non-positive dimensions detected {frame.shape}, returning default black frame {default_shape}")
        return np.zeros(default_shape, dtype=np.uint8)
    return frame

def resize_frame(frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    frame = validate_frame(frame)
    current_shape = frame.shape[:2]
    target_height, target_width = target_shape
    if target_height <= 0 or target_width <= 0:
        logger.warning(f"Invalid target shape for resize: {(target_height, target_width)}. Returning original frame.")
        return frame
    if current_shape != target_shape:
        try:
            return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            logger.error(f"OpenCV resize failed from {current_shape} to {(target_width, target_height)}: {e}")
            return frame
    return frame

# ========================================================================
#                       Effect Base Classes
# ========================================================================
class Effect(abc.ABC):
    def __init__(self):
        self._validate_params()
    def _validate_params(self) -> None: pass
    @abc.abstractmethod
    def apply(self, *args, **kwargs) -> np.ndarray: pass

class IntraClipEffect(Effect):
    @abc.abstractmethod
    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float) -> np.ndarray: pass

class TransitionEffect(Effect):
    duration: float = 0.5
    def __init__(self, duration: float = 0.5):
        self.duration = max(0.01, duration)
        super().__init__()
    @abc.abstractmethod
    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray: pass

# ========================================================================
#               Enhanced Intra-Clip Effects
# ========================================================================
@dataclass
class BrightnessContrast(IntraClipEffect):
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    per_channel: bool = False
    fade_in: bool = False
    def _validate_params(self):
        self.brightness = np.clip(self.brightness, -1.0, 1.0)
        self.contrast = np.clip(self.contrast, 0.0, 3.0)
        self.gamma = max(0.1, self.gamma)
    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        if self.contrast == 1.0 and self.brightness == 0 and self.gamma == 1.0 and not self.fade_in:
            return frame
        try:
            factor = (frame_time / max(clip_duration, 1e-6)) if self.fade_in and clip_duration > 0 else 1.0
            curr_brightness = self.brightness * factor
            curr_contrast = 1.0 + (self.contrast - 1.0) * factor
            curr_gamma = 1.0 + (self.gamma - 1.0) * factor if self.gamma != 1.0 else 1.0

            if self.per_channel:
                result = np.zeros_like(frame, dtype=np.float32)
                for c in range(3):
                    channel = frame[:, :, c].astype(np.float32)
                    channel = (channel - 127.5) * curr_contrast + 127.5 + curr_brightness * 255
                    if curr_gamma != 1.0:
                        inv_gamma = 1.0 / curr_gamma
                        channel_norm = np.clip(channel / 255.0, 0, 1)
                        channel_gamma_corrected = np.power(channel_norm, inv_gamma) * 255.0
                        channel = channel_gamma_corrected
                    result[:, :, c] = np.clip(channel, 0, 255)
                return result.astype(np.uint8)
            else:
                frame_float = frame.astype(np.float32)
                adjusted = (frame_float - 127.5) * curr_contrast + 127.5 + curr_brightness * 255
                if curr_gamma != 1.0:
                    inv_gamma = 1.0 / curr_gamma
                    adjusted_norm = np.clip(adjusted / 255.0, 0, 1)
                    adjusted_gamma_corrected = np.power(adjusted_norm, inv_gamma) * 255.0
                    adjusted = adjusted_gamma_corrected
                return np.clip(adjusted, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.warning(f"BrightnessContrast failed: {e}")
            return frame

@dataclass
class Saturation(IntraClipEffect):
    scale: float = 1.0
    vibrance: float = 0.0
    fade_in: bool = False
    def _validate_params(self):
        self.scale = max(0.0, self.scale)
        self.vibrance = np.clip(self.vibrance, 0.0, 1.0)
    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        if abs(self.scale - 1.0) < 1e-3 and self.vibrance == 0 and not self.fade_in:
            return frame
        try:
            factor = (frame_time / max(clip_duration, 1e-6)) if self.fade_in and clip_duration > 0 else 1.0
            curr_scale = 1.0 + (self.scale - 1.0) * factor
            curr_vibrance = self.vibrance * factor

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            s = hsv[:, :, 1]
            if curr_vibrance > 0:
                s_max = np.max(s)
                if s_max > 1e-6:
                    vibrance_boost = (1.0 - s / s_max) * curr_vibrance * s_max
                    s += vibrance_boost
            hsv[:, :, 1] = np.clip(s * curr_scale, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        except Exception as e:
            logger.warning(f"Saturation failed: {e}")
            return frame

@dataclass
class Vignette(IntraClipEffect):
    strength: float = 0.5
    radius: float = 0.6
    falloff: float = 0.3
    shape: str = 'circular'
    color: Tuple[int, int, int] = (0, 0, 0)
    fade_in: bool = False
    def _validate_params(self):
        self.strength = np.clip(self.strength, 0.0, 1.0)
        self.radius = np.clip(self.radius, 0.0, 1.0)
        self.falloff = max(0.01, self.falloff)
        self.shape = self.shape if self.shape in ['circular', 'elliptical', 'rectangular'] else 'circular'
    def _create_mask(self, height: int, width: int, factor: float) -> np.ndarray:
        # ... (implementation unchanged) ...
        cx, cy = width / 2, height / 2
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        if self.shape == 'circular':
            max_dim = max(width, height)
            d = np.sqrt((x - cx)**2 + (y - cy)**2) / max(max_dim / 2, 1e-6)
        elif self.shape == 'elliptical':
            d = np.sqrt(((x - cx)/(width/2 + 1e-6))**2 + ((y - cy)/(height/2 + 1e-6))**2)
        else:  # rectangular
            d = np.maximum(np.abs(x - cx) / (width / 2 + 1e-6), np.abs(y - cy) / (height / 2 + 1e-6))
        curr_strength = self.strength * factor
        curr_radius = self.radius
        mask = 1.0 - curr_strength * (1.0 / (1.0 + np.exp(-(d - curr_radius) / self.falloff)))
        return mask[:, :, np.newaxis].astype(np.float32)
    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        if self.strength < 1e-3 and not self.fade_in:
            return frame
        try:
            factor = (frame_time / max(clip_duration, 1e-6)) if self.fade_in and clip_duration > 0 else 1.0
            mask = self._create_mask(frame.shape[0], frame.shape[1], factor)
            vignette_layer = np.full_like(frame, self.color, dtype=np.float32)
            return np.clip(frame.astype(np.float32) * mask + vignette_layer * (1.0 - mask), 0, 255).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Vignette failed: {e}")
            return frame

@dataclass
class Blur(IntraClipEffect):
    kernel_size: int = 5
    sigma: float = 0.0
    fade_in: bool = False
    def _validate_params(self):
        self.kernel_size = max(3, self.kernel_size if self.kernel_size % 2 != 0 else self.kernel_size + 1)
        self.sigma = max(0.0, self.sigma)
    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        if self.kernel_size <= 3 and self.sigma == 0 and not self.fade_in:
            return frame
        try:
            factor = (frame_time / max(clip_duration, 1e-6)) if self.fade_in and clip_duration > 0 else 1.0
            ckf = 1 + (self.kernel_size - 1) * factor
            ck = int(ckf) | 1
            ck = max(3, ck)
            if ck < 3:
                return frame
            return cv2.GaussianBlur(frame, (ck, ck), self.sigma)
        except Exception as e:
            logger.warning(f"Blur failed: {e}")
            return frame

@dataclass
class Crossfade(TransitionEffect):
    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame_a = validate_frame(frame_a)
        ts = frame_a.shape if (frame_b is None or not isinstance(frame_b, np.ndarray) or frame_b.size == 0) else frame_b.shape
        frame_b = validate_frame(frame_b, default_shape=ts)
        progress = np.clip(progress, 0.0, 1.0)
        try:
            a_r = resize_frame(frame_a, ts[:2])
            b_r = resize_frame(frame_b, ts[:2])
            return cv2.addWeighted(a_r, 1.0 - progress, b_r, progress, 0.0)
        except Exception as e:
            logger.warning(f"Crossfade failed: {e}")
            return frame_b if progress > 0.5 else frame_a

@dataclass
class Slide(TransitionEffect):
    direction: str = 'left'
    def _validate_params(self):
        vd = {'left', 'right', 'up', 'down'}
        if self.direction not in vd:
            logger.warning(f"Invalid direction: {self.direction}. Defaulting to 'left'")
            self.direction = 'left'
    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame_a = validate_frame(frame_a)
        ts = frame_a.shape if (frame_b is None or not isinstance(frame_b, np.ndarray) or frame_b.size == 0) else frame_b.shape
        frame_b = validate_frame(frame_b, default_shape=ts)
        progress = np.clip(progress, 0.0, 1.0)
        try:
            a = resize_frame(frame_a, ts[:2])
            b = resize_frame(frame_b, ts[:2])
            h, w = ts[:2]
            o = a.copy()
            if self.direction == 'left':
                off = int(w * progress)
                if off > 0: o[:, w - off:] = b[:, :off]
            elif self.direction == 'right':
                off = int(w * progress)
                if off > 0: o[:, :off] = b[:, w - off:]
            elif self.direction == 'up':
                off = int(h * progress)
                if off > 0: o[h - off:, :] = b[:off, :]
            elif self.direction == 'down':
                off = int(h * progress)
                if off > 0: o[:off, :] = b[h - off:, :]
            return o
        except Exception as e:
            logger.warning(f"Slide failed: {e}")
            return frame_b if progress > 0.5 else frame_a

# ========================================================================
#                       Effect Factory
# ========================================================================
EFFECT_REGISTRY = {
    'brightness_contrast': BrightnessContrast, 'saturation': Saturation, 'vignette': Vignette,
    'blur': Blur, 'crossfade': Crossfade, 'slide': Slide,
}
def create_effect(effect_name: str, **params: Any) -> Optional[Effect]:
    # ... (implementation unchanged) ...
    effect_class = EFFECT_REGISTRY.get(effect_name.lower())
    if not effect_class:
        logger.error(f"Unknown effect: {effect_name}")
        return None
    try:
        return effect_class(**params)
    except TypeError as e:
        logger.error(f"Invalid parameters for {effect_name}: {params}. Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create effect '{effect_name}' with params {params}: {e}")
        return None

# ========================================================================
#           Model Loaders
# ========================================================================
def load_rvm_model(device: str, model_name: str = 'resnet50', pretrained: bool = True) -> Optional[torch.nn.Module]:
    # ... (implementation unchanged) ...
    if not torch.__version__:
        logger.warning("PyTorch not available. RVM cannot be loaded.")
        return None
    try:
        logger.info(f"Attempting to load RVM model '{model_name}' from Torch Hub...")
        model = torch.hub.load('PeterL1n/RobustVideoMatting', model_name, pretrained=pretrained)
        logger.info(f"RVM model '{model_name}' loaded successfully. Moving to device: {device}")
        return model.to(device).eval()
    except Exception as e:
        logger.error(f"Failed to load RVM model '{model_name}': {e}", exc_info=True)
        return None

def load_sam_mask_generator(device: str, model_type: str = 'vit_t', checkpoint_path: Optional[str] = None) -> Optional[Any]:
    # ... (implementation unchanged) ...
    if not _sam_available:
        logger.warning("segment_anything library not found. SAM Mask Generator unavailable.")
        return None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"SAM v1 checkpoint not found or not specified: {checkpoint_path}. SAM Mask Generator unavailable.")
        return None
    if not sam_model_registry:
         logger.error("SAM v1 model registry not available (import failed). Cannot load SAM v1.")
         return None
    try:
        logger.info(f"Loading SAM v1 model type '{model_type}' from checkpoint: {checkpoint_path}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        logger.info(f"SAM v1 model loaded to {device}. Creating AutomaticMaskGenerator.")
        return SamAutomaticMaskGenerator(sam)
    except KeyError:
        logger.error(f"SAM v1 model type '{model_type}' not found in registry. Available types: {list(sam_model_registry.keys())}")
        return None
    except Exception as e:
        logger.error(f"Failed to load SAM v1 model or create generator: {e}", exc_info=True)
        return None

def load_sam2_video_predictor(device: str, checkpoint_path: Optional[str] = None) -> Optional[Any]:
    # ... (implementation unchanged) ...
    if not _sam2_available:
        logger.warning("ultralytics library not found. SAM 2 predictor unavailable.")
        return None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"SAM 2 checkpoint not found or not specified: {checkpoint_path}. SAM 2 predictor unavailable.")
        return None
    if not SAM:
        logger.error("Ultralytics SAM class not available (import failed). Cannot load SAM 2.")
        return None
    logger.debug(f"Attempting to load SAM 2 via ultralytics.SAM with checkpoint: {checkpoint_path}")
    try:
        model = SAM(checkpoint_path)
        logger.info(f"Successfully initialized ultralytics.SAM object for checkpoint: {checkpoint_path}")
        if hasattr(model, 'device'): logger.debug(f"Ultralytics SAM model reports device: {model.device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load SAM 2 model using Ultralytics SAM('{checkpoint_path}'): {e}", exc_info=True)
        return None

# ========================================================================
#             Pixel Sensing, Matting & Segmentation Runner
# ========================================================================
class PixelSensingEffectRunner:
    LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    LK_PARAMS_GOLD = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    MAX_HISTORY = 18
    TRAIL_START_ALPHA = 0.7
    TRAIL_END_ALPHA = 0.0
    TRAIL_RADIUS = 2
    TRAIL_BLUR_KERNEL = (5, 5)
    GOLD_TINT_COLOR = (30, 165, 210)  # BGR: Amber/Gold
    TINT_STRENGTH = 0.40
    MASK_BLUR_KERNEL = (15, 15)
    SEGMENTATION_THRESHOLD = 0.6
    RVM_DOWNSAMPLE_RATIO = 0.4
    RVM_DEFAULT_BG_COLOR = (0, 0, 255)  # BGR: Blue
    SAM_MASK_ALPHA = 0.5
    RVM_DISPLAY_MODES = ["Composite", "Alpha", "Foreground"]

    def __init__(self,
                 input_source: Any = 0,
                 mode: str = "live",
                 output_path: Optional[str] = None,
                 effect: str = "led_base",
                 display: Optional[bool] = None,
                 rvm_background_path: Optional[str] = None,
                 sam_checkpoint_path: Optional[str] = None,
                 use_sam2: bool = False):
        self.input_source = input_source
        self.mode = mode.lower()
        if self.mode not in ["live", "post"]:
            logger.error("Mode must be 'live' or 'post'. Defaulting to 'live'.")
            self.mode = "live"
        self.output_path = output_path if self.mode == "post" else None
        if self.mode == "post" and not self.output_path:
            logger.error("Post mode requires output_path. Output disabled.")
        self.display = display if display is not None else (self.mode == "live")
        self._init_rvm_bg_path = rvm_background_path
        self._requested_device = _torch_device
        self.device = self._requested_device
        logger.info(f"PixelSensingEffectRunner initialized. Mode: {self.mode}, Initial Device: {self.device}")

        self.cap = None
        self.out = None
        self.frame_height: Optional[int] = None
        self.frame_width: Optional[int] = None
        self.fps: float = 30.0

        # --- MediaPipe Initialization ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        self.pose: Optional[mp.solutions.pose.Pose] = None
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=5, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
                enable_segmentation=True)
            logger.info("MediaPipe Face Mesh and Pose initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe components: {e}", exc_info=True)

        # --- State Variables ---
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_landmarks_flow: Optional[List[Tuple[int, int]]] = None
        self.prev_face_pts_gold: np.ndarray = np.empty((0, 1, 2), dtype=np.float32)
        self.prev_pose_pts_gold: np.ndarray = np.empty((0, 1, 2), dtype=np.float32)
        self.flow: Optional[np.ndarray] = None
        self.hue_offset: int = 0
        self.face_history = deque(maxlen=self.MAX_HISTORY)
        self.pose_history = deque(maxlen=self.MAX_HISTORY)

        # --- RVM Initialization ---
        # ... (unchanged RVM init) ...
        self.rvm_models: Dict[str, Optional[torch.nn.Module]] = {}
        self.rvm_model_names: List[str] = ['resnet50', 'mobilenetv3']
        self.current_rvm_model_idx: int = 0
        self.rvm_available: bool = False
        for model_name in self.rvm_model_names:
            model = load_rvm_model(self._requested_device, model_name)
            if model:
                self.rvm_models[model_name] = model
                self.rvm_available = True
        if not self.rvm_available: logger.warning("No RVM models could be loaded.")
        self.rvm_rec: List[Optional[torch.Tensor]] = [None] * 4
        self.rvm_downsample_ratio: float = self.RVM_DOWNSAMPLE_RATIO
        self.rvm_background: Optional[np.ndarray] = None
        self.rvm_display_mode: int = 0

        # --- SAM/SAM2 Initialization ---
        # ... (unchanged SAM init) ...
        actual_sam_checkpoint = sam_checkpoint_path if sam_checkpoint_path else ""
        self.use_sam2_runtime = use_sam2 and _sam2_available and os.path.exists(actual_sam_checkpoint)
        self.sam_model: Optional[Any] = None
        self.sam_available: bool = False
        self.sam2_state: Optional[Any] = None
        self._sam_device = self.device
        if self.use_sam2_runtime:
            logger.info(f"Attempting to load SAM 2 model from: {actual_sam_checkpoint}")
            sam2_device_to_use = self._requested_device
            if self._requested_device == 'mps':
                logger.warning("MPS detected. Forcing SAM 2 to CPU due to compatibility issues.")
                sam2_device_to_use = 'cpu'
            self._sam_device = sam2_device_to_use
            self.sam_model = load_sam2_video_predictor(sam2_device_to_use, actual_sam_checkpoint)
            if self.sam_model:
                self.sam_available = True
                logger.info(f"SAM 2 model loaded on device '{self._sam_device}'.")
            else:
                logger.warning("Failed to load SAM 2 model.")
                self.use_sam2_runtime = False
        elif _sam_available and os.path.exists(actual_sam_checkpoint):
            logger.info(f"Attempting to load SAM v1 model from: {actual_sam_checkpoint}")
            sam_v1_model_type = 'vit_t'
            if 'vit_l' in actual_sam_checkpoint: sam_v1_model_type = 'vit_l'
            elif 'vit_h' in actual_sam_checkpoint: sam_v1_model_type = 'vit_h'
            elif 'vit_b' in actual_sam_checkpoint: sam_v1_model_type = 'vit_b'
            self._sam_device = self._requested_device
            self.sam_model = load_sam_mask_generator(self._sam_device, model_type=sam_v1_model_type, checkpoint_path=actual_sam_checkpoint)
            if self.sam_model:
                self.sam_available = True
                logger.info(f"SAM v1 Mask Generator loaded on device '{self._sam_device}'.")
            else:
                logger.warning("Failed to load SAM v1 model.")
        else:
            logger.warning("Neither SAM v1 nor SAM 2 could be loaded.")


        # --- Effect Definitions ---
        self.effects: Dict[str, Any] = {
            'led_base': self._apply_led_base_style,
            'led_enhanced': self._apply_led_enhanced_style,
            'led_hue_rotate': self._apply_led_hue_rotate_style,
            'goldenaura': self._apply_goldenaura_style,
            'none': lambda **kwargs: kwargs['frame'],
            'brightness_contrast': BrightnessContrast(brightness=0.1, contrast=1.1, gamma=1.0),
            'saturation': Saturation(scale=1.3, vibrance=0.1),
            'vignette': Vignette(strength=0.5, radius=0.6, falloff=0.3, shape='elliptical'),
            'blur': Blur(kernel_size=9, sigma=0),
            # <<< ADDED: Lightning effect registration >>>
            'lightning': self._apply_lightning_style
        }
        if self.rvm_available:
            self.effects['rvm_composite'] = self._apply_rvm_composite_style
        if self.sam_available:
            sam_effect_name = 'sam2_segmentation' if self.use_sam2_runtime else 'sam1_segmentation'
            self.effects[sam_effect_name] = self._apply_sam_segmentation_style
            logger.info(f"Registered SAM effect as: {sam_effect_name}")

        # Validate initial effect
        if effect not in self.effects:
            logger.warning(f"Initial effect '{effect}' not available. Defaulting to 'none'. Available: {list(self.effects.keys())}")
            self.current_effect = 'none'
        else:
            # Check availability for special effects
            if effect == 'rvm_composite' and not self.rvm_available:
                logger.warning("RVM effect requested but unavailable. Defaulting to 'none'.")
                self.current_effect = 'none'
            elif effect.startswith('sam') and not self.sam_available:
                 logger.warning("SAM effect requested but unavailable. Defaulting to 'none'.")
                 self.current_effect = 'none'
            else:
                 self.current_effect = effect # Set effect if available or standard

        self.window_name = f"Pixel Effects (Mode: {self.mode}, Req Device: {self._requested_device.upper()})"

    def _load_rvm_background(self, path: Optional[str]):
        # ... (implementation unchanged) ...
        loaded_bg = None
        if path and os.path.exists(path):
            try:
                loaded_bg = cv2.imread(path)
                if loaded_bg is not None and loaded_bg.size > 0:
                    logger.info(f"Loaded RVM background from {path}")
                else:
                    logger.warning(f"Failed to load RVM background from {path}")
                    loaded_bg = None
            except Exception as e:
                logger.warning(f"Error loading RVM background from {path}: {e}")
                loaded_bg = None
        if loaded_bg is None:
            logger.info(f"Using default solid color background: {self.RVM_DEFAULT_BG_COLOR}")
            h = self.frame_height if self.frame_height is not None else DEFAULT_HEIGHT
            w = self.frame_width if self.frame_width is not None else DEFAULT_WIDTH
            if h <= 0 or w <= 0:
                h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
                logger.warning(f"Invalid dimensions for default BG, using {w}x{h}")
            self.rvm_background = np.full((h, w, 3), self.RVM_DEFAULT_BG_COLOR, dtype=np.uint8)
        else:
            self.rvm_background = loaded_bg
            if self.frame_height is not None and self.frame_width is not None:
                self.rvm_background = resize_frame(self.rvm_background, (self.frame_height, self.frame_width))

    def _initialize_capture(self) -> bool:
        # ... (implementation unchanged) ...
        logger.info(f"Initializing video capture from: {self.input_source}")
        if self.mode == "post" and not isinstance(self.input_source, str):
            logger.error("Post mode needs video file path.")
            return False
        try:
            self.cap = cv2.VideoCapture(self.input_source)
        except Exception as e:
            logger.error(f"Error initializing VideoCapture: {e}")
            return False
        if not self.cap or not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.input_source}")
            return False
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.error("Failed to read first frame.")
            self.cap.release()
            return False
        self.frame_height, self.frame_width = frame.shape[:2]
        logger.info(f"Video source opened. Frame dimensions: {self.frame_width}x{self.frame_height}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            logger.warning(f"Invalid FPS ({self.fps}), defaulting to 30.0")
            self.fps = 30.0
        else:
            logger.info(f"Source FPS: {self.fps:.2f}")
        self._load_rvm_background(self._init_rvm_bg_path)
        if self.mode == "post" and self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            try:
                self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
                if not self.out.isOpened():
                    logger.error(f"Failed to open VideoWriter: {self.output_path}")
                    self.cap.release()
                    return False
                logger.info(f"VideoWriter initialized: {self.output_path}")
            except Exception as e:
                logger.error(f"Error initializing VideoWriter: {e}")
                self.cap.release()
                return False
        try:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.error(f"Failed first frame grayscale: {e}")
            self.prev_gray = None
        return True

    def _cleanup(self):
        # ... (implementation unchanged) ...
        logger.info("Cleaning up resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Capture released.")
        if self.out and self.out.isOpened():
            self.out.release()
            logger.info("Writer released.")
        if hasattr(self, 'face_mesh') and self.face_mesh:
            try: self.face_mesh.close()
            except: pass
            logger.info("Face Mesh closed.")
        if hasattr(self, 'pose') and self.pose:
            try: self.pose.close()
            except: pass
            logger.info("Pose closed.")
        if self.rvm_models:
            for model_name in list(self.rvm_models.keys()):
                if self.rvm_models[model_name] is not None:
                    try: del self.rvm_models[model_name]
                    except: pass
            self.rvm_models.clear()
            logger.info("RVM models deleted.")
        if self.sam_model:
            try: del self.sam_model
            except: pass
            self.sam_model = None
            logger.info("SAM model deleted.")
        if self._requested_device == 'cuda':
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        if self.display:
            try: cv2.destroyAllWindows()
            except: pass
            logger.info("OpenCV windows destroyed.")
        logger.info("Cleanup complete.")

    # --- Effect Application Methods ---
    def _apply_distortion(self, frame: np.ndarray) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        try:
            rows, cols, _ = frame.shape
            map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
            distort_x = np.sin(map_y / 20.0) * 5.0
            map_x = np.clip(map_x + distort_x, 0, cols - 1).astype(np.float32)
            map_y = map_y.astype(np.float32)
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logger.warning(f"Distortion failed: {e}")
            return frame

    def _draw_flow_trails_simple(self, frame: np.ndarray, flow: Optional[np.ndarray], prev_pts: Optional[np.ndarray], line_color=(255, 105, 180), dot_color=(173, 216, 230), line_thickness=1, dot_radius=3):
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        if flow is None or prev_pts is None or flow.shape[0] != prev_pts.shape[0]:
            return frame
        h, w = frame.shape[:2]
        try:
            for i, (new, old) in enumerate(zip(flow, prev_pts)):
                xn, yn = new.ravel()
                xo, yo = old.ravel()
                if 0 <= xn < w and 0 <= yn < h and 0 <= xo < w and 0 <= yo < h:
                    cv2.line(frame, (int(xn), int(yn)), (int(xo), int(yo)), line_color, line_thickness, cv2.LINE_AA)
                    cv2.circle(frame, (int(xn), int(yn)), dot_radius, dot_color, -1, cv2.LINE_AA)
        except Exception as e:
            logger.warning(f"Draw flow trails failed: {e}")
            return frame
        return frame

    def _draw_motion_glow_separate(self, frame: np.ndarray, landmarks: List[Tuple[int, int]], flow: Optional[np.ndarray], prev_pts: Optional[np.ndarray]):
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        num_landmarks = len(landmarks)
        if num_landmarks == 0:
            return frame
        magnitudes = np.zeros(num_landmarks)
        try:
            if flow is not None and prev_pts is not None and flow.shape[0] == prev_pts.shape[0] and prev_pts.shape[0] == num_landmarks:
                magnitudes = np.linalg.norm(flow - prev_pts, axis=2).flatten()
            elif flow is not None and prev_pts is not None and flow.shape[0] > 0:
                avg_mag = np.mean(np.linalg.norm(flow - prev_pts, axis=2))
                magnitudes = np.full(num_landmarks, avg_mag)
        except Exception as e:
            logger.debug(f"Glow magnitude calc failed: {e}")
        h, w = frame.shape[:2]
        try:
            for i, (x, y) in enumerate(landmarks):
                glow_intensity = min(255, int(magnitudes[i] * 15))
                glow_color = (glow_intensity // 3, glow_intensity // 2, glow_intensity)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 5, glow_color, -1, cv2.LINE_AA)
        except Exception as e:
            logger.warning(f"Draw motion glow failed: {e}")
            return frame
        return frame

    def _draw_sam_masks(self, frame: np.ndarray, masks: Any) -> np.ndarray:
        # ... (implementation unRanged) ...
        frame = validate_frame(frame)
        if masks is None:
            logger.debug("No masks provided to draw.")
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        num_masks_drawn = 0
        try:
            masks_to_process: List[Any] = []
            if isinstance(masks, torch.Tensor):
                if masks.ndim >= 3:
                    masks_to_process = list(masks.cpu().detach())
                    logger.debug(f"Processing {len(masks_to_process)} masks from tensor shape {masks.shape}")
                else:
                    logger.warning(f"Received tensor mask with unexpected shape: {masks.shape}")
            elif isinstance(masks, list):
                masks_to_process = masks
                logger.debug(f"Processing {len(masks_to_process)} masks from list")
            else:
                logger.warning(f"Unsupported mask input type: {type(masks)}")
                return frame

            if not masks_to_process:
                logger.debug("Mask list is empty.")
                return frame

            if isinstance(masks_to_process[0], dict) and 'area' in masks_to_process[0]:
                masks_to_process = sorted(masks_to_process, key=lambda x: x.get('area', 0), reverse=True)

            for i, mask_info in enumerate(masks_to_process):
                mask_data = None
                if isinstance(mask_info, dict):
                    mask_data = mask_info.get('segmentation')
                elif isinstance(mask_info, np.ndarray):
                    mask_data = mask_info
                elif isinstance(mask_info, torch.Tensor):
                    try:
                        mask_data = mask_info.cpu().numpy()
                    except Exception as tensor_e:
                        logger.warning(f"Mask {i} tensor to numpy failed: {tensor_e}")
                        continue
                else:
                    logger.warning(f"Mask {i} unsupported format: {type(mask_info)}")
                    continue

                if mask_data is None or mask_data.size == 0:
                    continue

                try:
                    if mask_data.dtype in (np.float32, np.float64, torch.float32, torch.float64):
                        mask_bool = (mask_data > 0.5)
                    else:
                        mask_bool = mask_data.astype(bool)
                except ValueError:
                    logger.warning(f"Mask {i} could not convert to bool.")
                    continue

                if mask_bool.shape != (h, w):
                    if mask_bool.ndim == 2:
                        try:
                            mask_resized = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        except cv2.error as resize_err:
                            logger.warning(f"Mask {i} resize failed: {resize_err}")
                            continue
                    else:
                        logger.warning(f"Mask {i} has unresizable shape {mask_bool.shape}.")
                        continue
                else:
                    mask_resized = mask_bool

                if not np.any(mask_resized):
                    continue

                color = [random.randint(64, 200) for _ in range(3)]
                color_overlay = np.zeros_like(frame, dtype=np.uint8)
                color_overlay[mask_resized] = color
                cv2.addWeighted(overlay[mask_resized], 1.0 - self.SAM_MASK_ALPHA, color_overlay[mask_resized], self.SAM_MASK_ALPHA, 0, dst=overlay[mask_resized])
                num_masks_drawn += 1

            logger.debug(f"Applied {num_masks_drawn} masks.")
        except Exception as e:
            logger.error(f"Error drawing SAM masks: {e}", exc_info=True)
            return frame
        return overlay

    def _preprocess_frame_rvm(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        # ... (implementation unchanged) ...
        frame = validate_frame(frame)
        try:
            h, w = frame.shape[:2]
            th = max(1, int(h * self.rvm_downsample_ratio))
            tw = max(1, int(w * self.rvm_downsample_ratio))
            fr = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            frgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            return T.ToTensor()(frgb).unsqueeze(0).to(self._requested_device)
        except Exception as e:
            logger.warning(f"RVM preprocess failed: {e}")
            return None

    def _postprocess_output_rvm(self, fgr: torch.Tensor, pha: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # ... (implementation unchanged) ...
        try:
            if fgr is None or pha is None or not isinstance(fgr, torch.Tensor) or not isinstance(pha, torch.Tensor):
                return None, None
            fgr_cpu = fgr.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            pha_cpu = pha.squeeze(0).cpu().detach().numpy()
            if pha_cpu.ndim == 3 and pha_cpu.shape[0] == 1:
                pha_cpu = pha_cpu.squeeze(0)
            tw = int(max(1, self.frame_width if self.frame_width is not None else DEFAULT_WIDTH))
            th = int(max(1, self.frame_height if self.frame_height is not None else DEFAULT_HEIGHT))
            dsize = (tw, th)
            if dsize[0] <= 0 or dsize[1] <= 0:
                logger.error(f"Invalid dsize {dsize}")
                return None, None
            fgr_r = cv2.resize(fgr_cpu, dsize, interpolation=cv2.INTER_LINEAR)
            pha_r = cv2.resize(pha_cpu, dsize, interpolation=cv2.INTER_LINEAR)
            fgr_f = np.clip(fgr_r * 255.0, 0, 255).astype(np.uint8)
            pha_f = np.clip(pha_r, 0.0, 1.0)
            return fgr_f, pha_f
        except Exception as e:
            logger.error(f"RVM postprocess failed: {e}", exc_info=True)
            return None, None

    def _apply_led_base_style(self, **kwargs) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(kwargs['frame'])
        lm = kwargs.get('landmarks', [])
        fl = kwargs.get('flow')
        pf = kwargs.get('prev_pts_flow')
        proc = frame.copy()
        proc = self._draw_flow_trails_simple(proc, fl, pf)
        try:
            co = np.full(proc.shape, (30, 0, 90), dtype=np.uint8)
            proc = cv2.addWeighted(proc, 0.85, co, 0.15, 0)
        except Exception as e:
            logger.warning(f"LED Base overlay failed: {e}")
        proc = self._draw_motion_glow_separate(proc, lm, fl, pf)
        proc = self._apply_distortion(proc)
        return proc

    def _apply_led_enhanced_style(self, **kwargs) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(kwargs['frame'])
        fl = kwargs.get('flow')
        pf = kwargs.get('prev_pts_flow')
        proc = frame.copy()
        h, w = proc.shape[:2]
        if fl is not None and pf is not None and fl.shape[0] == pf.shape[0]:
            try:
                for i, (new, old) in enumerate(zip(fl, pf)):
                    xn, yn = new.ravel()
                    xo, yo = old.ravel()
                    if 0 <= xn < w and 0 <= yn < h and 0 <= xo < w and 0 <= yo < h:
                        cv2.line(proc, (int(xo), int(yo)), (int(xn), int(yn)), (255, 105, 180), 1, cv2.LINE_AA)
                        mag = np.linalg.norm(new - old)
                        gi = min(255, int(mag * 15))
                        gc = (gi // 4, gi // 2, gi)
                        cv2.circle(proc, (int(xn), int(yn)), 3, gc, -1, cv2.LINE_AA)
            except Exception as e:
                logger.warning(f"LED Enhanced flow draw failed: {e}")
        proc = self._apply_distortion(proc)
        return proc

    def _apply_led_hue_rotate_style(self, **kwargs) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(kwargs['frame'])
        lm = kwargs.get('landmarks', [])
        fl = kwargs.get('flow')
        pf = kwargs.get('prev_pts_flow')
        proc = frame.copy()
        proc = self._draw_flow_trails_simple(proc, fl, pf)
        proc = self._draw_motion_glow_separate(proc, lm, fl, pf)
        proc = self._apply_distortion(proc)
        try:
            hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int) + self.hue_offset) % 180
            proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            logger.warning(f"LED Hue rotate failed: {e}")
        self.hue_offset = (self.hue_offset + 1) % 180
        return proc

    def _apply_goldenaura_style(self, **kwargs) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(kwargs['frame'])
        fr = kwargs.get('face_results')
        pr = kwargs.get('pose_results')
        proc = frame.copy()
        to = np.zeros_like(proc, dtype=np.uint8)
        try:
            cm = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
            if self.face_mesh and fr and fr.multi_face_landmarks:
                for flm in fr.multi_face_landmarks:
                    pts = np.array([[int(lm.x * self.frame_width), int(lm.y * self.frame_height)] for lm in flm.landmark if lm.x is not None and lm.y is not None], dtype=np.int32)
                    if len(pts) > 2:
                        try: hull = cv2.convexHull(pts); cv2.fillConvexPoly(cm, hull, 255)
                        except: pass
            if self.pose and pr and pr.segmentation_mask is not None:
                try:
                    smf = cv2.resize(pr.segmentation_mask, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                    smu = (smf > self.SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
                    cm = cv2.bitwise_or(cm, smu)
                except Exception as seg_e: logger.warning(f"Pose seg failed: {seg_e}")
            tmb = cv2.GaussianBlur(cm, self.MASK_BLUR_KERNEL, 0) if self.MASK_BLUR_KERNEL[0] > 0 else cm
            gl = np.full_like(proc, self.GOLD_TINT_COLOR, dtype=np.uint8)
            am = (tmb / 255.0)[..., np.newaxis]
            proc = (proc * (1.0 - am * self.TINT_STRENGTH) + gl * (am * self.TINT_STRENGTH)).astype(np.uint8)
            nfh = len(self.face_history)
            for i, hp in enumerate(self.face_history):
                if hp is not None and hp.size > 0:
                    prog = i / max(1, nfh - 1) if nfh > 1 else 0.0
                    ca = max(0.0, self.TRAIL_START_ALPHA + (self.TRAIL_END_ALPHA - self.TRAIL_START_ALPHA) * prog)
                    tc = tuple(int(c * ca) for c in self.GOLD_TINT_COLOR)
                    for p in hp: cv2.circle(to, (int(p[0]), int(p[1])), self.TRAIL_RADIUS, tc, -1, cv2.LINE_AA)
            nph = len(self.pose_history)
            for i, hp in enumerate(self.pose_history):
                if hp is not None and hp.size > 0:
                    prog = i / max(1, nph - 1) if nph > 1 else 0.0
                    ca = max(0.0, self.TRAIL_START_ALPHA + (self.TRAIL_END_ALPHA - self.TRAIL_START_ALPHA) * prog)
                    tc = tuple(int(c * ca * 0.8) for c in self.GOLD_TINT_COLOR)
                    for p in hp: cv2.circle(to, (int(p[0]), int(p[1])), self.TRAIL_RADIUS + 1, tc, -1, cv2.LINE_AA)
            if self.TRAIL_BLUR_KERNEL[0] > 0: to = cv2.GaussianBlur(to, self.TRAIL_BLUR_KERNEL, 0)
            proc = cv2.add(proc, to)
        except Exception as e:
            logger.error(f"GoldenAura failed: {e}", exc_info=True)
            return frame
        return proc

    def _apply_rvm_composite_style(self, **kwargs) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(kwargs['frame'])
        if not self.rvm_available or not self.rvm_models:
            cv2.putText(frame, "RVM N/A", (10, 60), 0, 0.7, (0, 0, 255), 2)
            return frame
        model_name = self.rvm_model_names[self.current_rvm_model_idx % len(self.rvm_model_names)]
        model = self.rvm_models.get(model_name)
        if not model:
            cv2.putText(frame, f"RVM '{model_name}' Fail", (10, 60), 0, 0.7, (0, 0, 255), 2)
            return frame
        if self.rvm_background is None or self.rvm_background.shape[:2] != (self.frame_height, self.frame_width):
            logger.warning("RVM BG invalid.")
            self._load_rvm_background(None)
            if self.rvm_background is None or self.rvm_background.shape[:2] != (self.frame_height, self.frame_width):
                cv2.putText(frame, "RVM BG Error", (10, 90), 0, 0.7, (0, 0, 255), 2)
                return frame
        try:
            frame_tensor = self._preprocess_frame_rvm(frame)
            if frame_tensor is None:
                cv2.putText(frame, "RVM Preproc Err", (10, 90), 0, 0.7, (0, 0, 255), 2)
                return frame
            with torch.no_grad():
                model = model.to(self._requested_device)
                current_rvm_rec = [rec.to(self._requested_device) if rec is not None else None for rec in self.rvm_rec]
                fgr_tensor, pha_tensor, *rec_out = model(frame_tensor, *current_rvm_rec, downsample_ratio=self.rvm_downsample_ratio)
                if all(r is not None for r in rec_out): self.rvm_rec = [r.detach() for r in rec_out]
                else: logger.warning("RVM None in rec state.")
            fgr_np, pha_np = self._postprocess_output_rvm(fgr_tensor, pha_tensor)
            if fgr_np is None or pha_np is None:
                cv2.putText(frame, "RVM Postproc Err", (10, 90), 0, 0.7, (0, 0, 255), 2)
                return frame
            output_frame = frame
            current_mode_str = self.RVM_DISPLAY_MODES[self.rvm_display_mode]
            if self.rvm_display_mode == 0:  # Composite
                alpha_3c = pha_np[..., np.newaxis]
                composite = (fgr_np * alpha_3c + self.rvm_background * (1.0 - alpha_3c))
                output_frame = np.clip(composite, 0, 255).astype(np.uint8)
            elif self.rvm_display_mode == 1:  # Alpha
                alpha_display = (pha_np * 255.0).astype(np.uint8); output_frame = cv2.cvtColor(alpha_display, cv2.COLOR_GRAY2BGR)
            elif self.rvm_display_mode == 2:  # Foreground
                output_frame = fgr_np
            info_txt = f"RVM: {model_name} ({current_mode_str})"
            cv2.putText(output_frame, info_txt, (10, 30), 0, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output_frame, info_txt, (10, 30), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return output_frame
        except Exception as e:
            logger.error(f"RVM effect failed ('{model_name}', mode={self.rvm_display_mode}): {e}", exc_info=True)
            self.rvm_rec = [None] * 4
            cv2.putText(frame, "RVM Error", (10, 90), 0, 0.7, (0, 0, 255), 2)
            return frame

    def _apply_sam_segmentation_style(self, **kwargs) -> np.ndarray:
        # ... (implementation unchanged) ...
        frame = validate_frame(kwargs['frame'])
        pose_results = kwargs.get('pose_results')
        if not self.sam_available or self.sam_model is None:
            cv2.putText(frame, "SAM N/A", (10, 60), 0, 0.7, (0, 0, 255), 2)
            return frame
        sam_type_str = "SAM2" if self.use_sam2_runtime else "SAMv1"
        logger.debug(f"Attempting {sam_type_str} segmentation on device '{self._sam_device}'...")
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks: Any = None
            if self.use_sam2_runtime:
                points_list = []
                if self.pose and pose_results and pose_results.pose_landmarks:
                    for lm in pose_results.pose_landmarks.landmark:
                        if lm.visibility > 0.3: points_list.append([int(lm.x * self.frame_width), int(lm.y * self.frame_height)])
                    logger.debug(f"Using {len(points_list)} pose landmarks as prompts.")
                if not points_list: points_list.append([self.frame_width // 2, self.frame_height // 2]); logger.debug("Using center point as prompt.")
                points = np.array(points_list); labels = np.ones(len(points), dtype=int)
                logger.debug(f"Calling SAM 2 predict with {points.shape[0]} points...")
                try:
                    results = self.sam_model(frame_rgb, points=points, labels=labels, device=self._sam_device)
                    logger.debug(f"SAM 2 inference completed. Result type: {type(results)}")
                    if results and isinstance(results, list) and len(results) > 0:
                        first_result = results[0]
                        if hasattr(first_result, 'masks') and first_result.masks is not None and hasattr(first_result.masks, 'data'):
                            masks_data = first_result.masks.data
                            if masks_data is not None and isinstance(masks_data, torch.Tensor) and masks_data.numel() > 0:
                                masks = masks_data; logger.info(f"SAM 2 generated {masks.shape[0]} mask(s).")
                            else: logger.warning("SAM 2 masks.data empty.")
                        else: logger.warning("SAM 2 result lacks masks.data.")
                    else: logger.warning(f"SAM 2 returned None/unexpected: {type(results)}")
                except Exception as inference_e:
                    logger.error(f"SAM 2 inference failed: {inference_e}", exc_info=True)
                    cv2.putText(frame, "SAM2 Infer Err", (10, 120), 0, 0.7, (0, 0, 255), 2); return frame
            else:
                logger.debug("Calling SAM v1 Automatic Mask Generator...")
                try:
                    masks = self.sam_model.generate(frame_rgb)
                    logger.info(f"SAM v1 generated {len(masks)} masks.")
                except Exception as generate_e:
                    logger.error(f"SAM v1 generate() failed: {generate_e}", exc_info=True)
                    cv2.putText(frame, "SAM1 Gen Err", (10, 120), 0, 0.7, (0, 0, 255), 2); return frame
            if masks is not None and ((isinstance(masks, list) and masks) or (isinstance(masks, torch.Tensor) and masks.numel() > 0)):
                num_drawn = len(masks) if isinstance(masks, list) else masks.shape[0]
                logger.debug(f"Drawing {num_drawn} masks.")
                processed_frame = self._draw_sam_masks(frame, masks)
                cv2.putText(processed_frame, f"{sam_type_str} Masks: {num_drawn}", (10, 30), 0, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(processed_frame, f"{sam_type_str} Masks: {num_drawn}", (10, 30), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                return processed_frame
            else:
                logger.debug("No valid masks generated.")
                cv2.putText(frame, f"{sam_type_str}: No Masks", (10, 90), 0, 0.7, (0, 165, 255), 2)
                return frame
        except Exception as e:
            logger.error(f"SAM segmentation error: {e}", exc_info=True)
            cv2.putText(frame, "SAM Error", (10, 90), 0, 0.7, (0, 0, 255), 2)
            return frame

    # <<< ADDED: Lightning Bolt Generator >>>
    def generate_lightning(self, start_point, num_segments=10, segment_length=20, angle_range=30):
        """Generate points for a jagged lightning bolt starting from start_point."""
        points = [start_point]
        current_point = start_point
        current_angle = random.uniform(0, 360)  # Initial random direction in degrees

        for _ in range(num_segments):
            angle_deviation = random.uniform(-angle_range, angle_range)
            new_angle = current_angle + angle_deviation
            dx = segment_length * np.cos(np.deg2rad(new_angle))
            dy = segment_length * np.sin(np.deg2rad(new_angle))
            new_point = (int(current_point[0] + dx), int(current_point[1] + dy))
            # Basic boundary check (optional, prevents going too far off screen)
            if 0 <= new_point[0] < (self.frame_width or DEFAULT_WIDTH) and \
               0 <= new_point[1] < (self.frame_height or DEFAULT_HEIGHT):
                points.append(new_point)
                current_point = new_point
                current_angle = new_angle
            else:
                break # Stop if bolt goes off-screen

        return points

    # <<< ADDED: Lightning Effect Function >>>
    def _apply_lightning_style(self, **kwargs):
        """Apply a Zeus-inspired lightning effect controlled by landmark movement."""
        frame = validate_frame(kwargs['frame'])
        flow = kwargs.get('flow')  # New landmark positions from optical flow
        prev_pts_flow = kwargs.get('prev_pts_flow')  # Previous landmark positions

        # Validate inputs needed for this effect
        if flow is None or prev_pts_flow is None or flow.shape[0] != prev_pts_flow.shape[0]:
            #logger.debug("Lightning effect skipped: Missing or mismatched flow data.")
            return frame

        if flow.shape[0] == 0: # No landmarks tracked
            return frame

        # Calculate movement magnitudes
        try:
            magnitudes = np.linalg.norm(flow - prev_pts_flow, axis=2).flatten()
        except Exception as e:
            logger.warning(f"Lightning effect: Magnitude calculation failed: {e}")
            return frame

        if len(magnitudes) == 0:
            return frame # Should be caught by flow.shape[0] check, but good practice

        # Create a blank layer for lightning (same size and type as frame)
        lightning_layer = np.zeros_like(frame, dtype=np.uint8)

        # Parameters for lightning appearance
        lightning_color = (255, 255, 200) # Light blue BGR
        glow_kernel_size = (15, 15)
        glow_alpha = 0.6 # Transparency of the glow/lightning layer
        movement_threshold = 5 # Minimum pixel movement to trigger lightning
        base_segments = 5
        segments_per_magnitude = 0.5 # How much length increases with speed
        max_segments = 25 # Cap the number of segments
        base_thickness = 5
        flicker_probability = 0.8 # Chance a bolt appears if triggered (adds flicker)


        # Draw lightning from landmarks with significant movement
        num_bolts = 0
        for i in range(flow.shape[0]):
            mag = magnitudes[i]
            if mag > movement_threshold and random.random() < flicker_probability:
                start_point = (int(flow[i, 0, 0]), int(flow[i, 0, 1]))

                # Check if start point is within frame bounds
                if not (0 <= start_point[0] < self.frame_width and 0 <= start_point[1] < self.frame_height):
                    continue

                # Scale number of segments with movement, capped
                num_segments = min(max_segments, int(base_segments + mag * segments_per_magnitude))
                points = self.generate_lightning(start_point, num_segments=num_segments)

                if len(points) < 2: continue # Need at least two points to draw a line

                # Draw the bolt with tapering thickness
                total_pts = len(points)
                for j in range(total_pts - 1):
                    # Thickness tapers from base_thickness down to 1
                    thickness = max(1, int(base_thickness * (1.0 - (j / max(1, total_pts -1 )))))
                    try:
                        # Check points before drawing
                         pt1 = (int(points[j][0]), int(points[j][1]))
                         pt2 = (int(points[j+1][0]), int(points[j+1][1]))
                         cv2.line(lightning_layer, pt1, pt2, lightning_color, thickness, cv2.LINE_AA)
                         num_bolts += 1
                    except Exception as line_err:
                        logger.debug(f"Lightning draw error: {line_err} on points {points[j]} -> {points[j+1]}")
                        break # Stop drawing this bolt if an error occurs


        # If any bolts were drawn, apply glow and blend
        if num_bolts > 0:
            try:
                # Add glow effect using Gaussian Blur
                lightning_layer_blurred = cv2.GaussianBlur(lightning_layer, glow_kernel_size, 0)

                # Blend the blurred lightning layer with the original frame
                # frame = frame * (1 - alpha) + layer * alpha
                processed_frame = cv2.addWeighted(frame, 1.0, lightning_layer_blurred, glow_alpha, 0)
                return processed_frame
            except Exception as blend_err:
                 logger.warning(f"Lightning effect: Blending/Glow failed: {blend_err}")
                 return frame # Return original frame on error
        else:
            # No bolts drawn, return original frame
            return frame


    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        # <<< MODIFIED: Added 'lightning' to needs_gray checks >>>
        processed_frame = validate_frame(frame, (self.frame_height or DEFAULT_HEIGHT, self.frame_width or DEFAULT_WIDTH, 3))
        if processed_frame.shape[:2] != (self.frame_height, self.frame_width):
            logger.warning(f"Frame size mismatch: {processed_frame.shape[:2]} vs {(self.frame_height, self.frame_width)}")
            processed_frame = resize_frame(processed_frame, (self.frame_height, self.frame_width))
            if processed_frame.shape[:2] != (self.frame_height, self.frame_width):
                logger.error("Resize failed!")
                return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        frame_time = frame_idx / self.fps if self.fps > 0 else 0.0
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) if self.cap else 0
        clip_duration = (total_frames / self.fps) if (total_frames > 0 and self.fps > 0) else 1.0

        # Determine needed preprocessing based on current effect
        needs_gray = self.current_effect in ['led_base', 'led_enhanced', 'led_hue_rotate', 'goldenaura', 'lightning']
        needs_rgb_mediapipe = self.current_effect in ['led_base', 'led_enhanced', 'led_hue_rotate', 'goldenaura', 'sam2_segmentation', 'sam1_segmentation', 'lightning']

        frame_gray: Optional[np.ndarray] = None
        frame_rgb: Optional[np.ndarray] = None
        if needs_gray:
            try: frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            except Exception as e: logger.warning(f"Grayscale conversion failed: {e}"); frame_gray = None
        if needs_rgb_mediapipe:
            try:
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False # Optimize MediaPipe processing
            except Exception as e: logger.warning(f"RGB conversion failed: {e}"); frame_rgb = None

        # --- MediaPipe Landmark Detection ---
        face_results, pose_results = None, None
        if frame_rgb is not None:
            if self.face_mesh:
                try: face_results = self.face_mesh.process(frame_rgb)
                except Exception as e: logger.warning(f"FaceMesh process failed: {e}")
            if self.pose:
                try: pose_results = self.pose.process(frame_rgb)
                except Exception as e: logger.warning(f"Pose process failed: {e}")
            # Set writeable back to True if needed elsewhere, though effects usually copy
            # frame_rgb.flags.writeable = True

        # --- Landmark Extraction ---
        current_landmarks_list: List[Tuple[int, int]] = []
        current_face_points_gold_np = np.empty((0, 1, 2), dtype=np.float32)
        current_pose_points_gold_np = np.empty((0, 1, 2), dtype=np.float32)
        # Consolidate landmark extraction logic (more robust)
        def extract_landmarks(results, is_face):
            lm_list = []
            lm_np_list = []
            landmarks_container = results.multi_face_landmarks if is_face else ([results.pose_landmarks] if results.pose_landmarks else [])
            if landmarks_container:
                 for landmarks in landmarks_container:
                     if landmarks:
                        pts_frame = []
                        for lm in landmarks.landmark:
                             if lm and hasattr(lm, 'x') and hasattr(lm, 'y') and lm.x is not None and lm.y is not None:
                                 visibility = getattr(lm, 'visibility', 1.0) # Default to visible if no visibility attr
                                 if visibility is not None and visibility > 0.1: # Use visibility threshold
                                    x, y = int(lm.x * self.frame_width), int(lm.y * self.frame_height)
                                    lm_list.append((x, y))
                                    pts_frame.append([lm.x * self.frame_width, lm.y * self.frame_height])
                        if pts_frame:
                           lm_np_list.append(np.array(pts_frame, dtype=np.float32).reshape(-1, 1, 2))
            np_concat = np.concatenate(lm_np_list, axis=0) if lm_np_list else np.empty((0, 1, 2), dtype=np.float32)
            return lm_list, np_concat

        if face_results:
            face_lm_list, current_face_points_gold_np = extract_landmarks(face_results, is_face=True)
            current_landmarks_list.extend(face_lm_list)
        if pose_results:
            pose_lm_list, current_pose_points_gold_np = extract_landmarks(pose_results, is_face=False)
            current_landmarks_list.extend(pose_lm_list)

        # Combine points for general optical flow if needed by current effect
        current_points_flow_np = np.array(current_landmarks_list, dtype=np.float32).reshape(-1, 1, 2)

        # --- Optical Flow Calculation ---
        flow_for_led: Optional[np.ndarray] = None
        prev_pts_for_led: Optional[np.ndarray] = None
        tracked_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
        tracked_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)

        if frame_gray is not None and self.prev_gray is not None:
            # General LK flow (for LED, Lightning)
            # Use the combined landmarks list from the *previous* frame
            if self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) > 0:
                 prev_pts_np = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                 if prev_pts_np.shape[0] > 0:
                    try:
                        flow_calc, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, prev_pts_np, None, **self.LK_PARAMS)
                        if flow_calc is not None and status is not None:
                            good_new = flow_calc[status.flatten() == 1]
                            good_old = prev_pts_np[status.flatten() == 1]
                            # Ensure shapes match before assigning
                            if good_new.shape[0] > 0 and good_new.shape == good_old.shape:
                                flow_for_led = good_new.reshape(-1, 1, 2)
                                prev_pts_for_led = good_old.reshape(-1, 1, 2)
                            else:
                                logger.debug("Optical flow resulted in empty or mismatched points.")
                                flow_for_led = None
                                prev_pts_for_led = None
                    except cv2.error as lk_err:
                        logger.warning(f"General LK Optical Flow failed: {lk_err}")
                        flow_for_led = None
                        prev_pts_for_led = None
                    except Exception as lk_e:
                        logger.warning(f"General LK Optical Flow unexpected error: {lk_e}")
                        flow_for_led = None
                        prev_pts_for_led = None

            # Golden Aura specific LK flow
            if self.current_effect == 'goldenaura':
                if self.prev_face_pts_gold.size > 0:
                    try:
                        nfp, sf, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_face_pts_gold, None, **self.LK_PARAMS_GOLD)
                        if nfp is not None and sf is not None: tracked_face_pts_gold = nfp[sf.flatten() == 1].reshape(-1, 1, 2)
                    except Exception as e: logger.warning(f"Gold Face LK failed: {e}")
                if self.prev_pose_pts_gold.size > 0:
                    try:
                        npp, sp, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_pose_pts_gold, None, **self.LK_PARAMS_GOLD)
                        if npp is not None and sp is not None: tracked_pose_pts_gold = npp[sp.flatten() == 1].reshape(-1, 1, 2)
                    except Exception as e: logger.warning(f"Gold Pose LK failed: {e}")

        # --- Update History for Golden Aura ---
        face_history_pts = tracked_face_pts_gold if tracked_face_pts_gold.size > 0 else current_face_points_gold_np
        pose_history_pts = tracked_pose_pts_gold if tracked_pose_pts_gold.size > 0 else current_pose_points_gold_np
        if self.current_effect == 'goldenaura':
            if face_history_pts.size > 0: self.face_history.append(face_history_pts.reshape(-1, 2))
            if pose_history_pts.size > 0: self.pose_history.append(pose_history_pts.reshape(-1, 2))

        # --- Apply Selected Effect ---
        effect_func_or_instance = self.effects.get(self.current_effect)
        if effect_func_or_instance is not None:
            try:
                if isinstance(effect_func_or_instance, IntraClipEffect):
                    processed_frame = effect_func_or_instance.apply(processed_frame, frame_time, clip_duration)
                elif callable(effect_func_or_instance):
                    # Pass the relevant data to the effect function
                    effect_kwargs = {
                        'frame': processed_frame,
                        'landmarks': current_landmarks_list, # Current frame's landmarks
                        'flow': flow_for_led,                # New points from LK
                        'prev_pts_flow': prev_pts_for_led,   # Old points from LK
                        'face_results': face_results,
                        'pose_results': pose_results,
                        'frame_idx': frame_idx,
                        'frame_time': frame_time,
                        'clip_duration': clip_duration
                    }
                    processed_frame = effect_func_or_instance(**effect_kwargs)
                else:
                    logger.warning(f"Effect '{self.current_effect}' invalid type.")
            except Exception as e:
                logger.error(f"Apply effect '{self.current_effect}' failed: {e}", exc_info=True)
                # Revert to original frame on error? Or keep potentially corrupted frame?
                # Let's revert for safety, though this might cause flickering on errors.
                processed_frame = frame.copy() # Use original frame before effect attempt
        else:
            logger.warning(f"Effect '{self.current_effect}' not found.")
            # processed_frame remains as it was before effect lookup

        # --- Update State for Next Frame ---
        self.prev_gray = frame_gray # Store current gray frame for next iteration's LK
        # Store the *detected* landmarks from the *current* frame to be used as
        # the starting points for LK in the *next* frame.
        self.prev_landmarks_flow = current_landmarks_list
        # Update gold aura previous points
        self.prev_face_pts_gold = face_history_pts.reshape(-1, 1, 2)
        self.prev_pose_pts_gold = pose_history_pts.reshape(-1, 1, 2)

        return validate_frame(processed_frame, (self.frame_height, self.frame_width, 3))


    def run(self):
        if not self._initialize_capture():
            logger.error("Failed to initialize capture. Exiting.")
            self._cleanup()
            return

        if self.frame_height is None or self.frame_width is None:
            logger.error("Frame dimensions not set after initialization. Exiting.")
            self._cleanup()
            return

        frame_idx = 0
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info(f"{'End of video.' if self.mode == 'post' else 'Camera ended.'}")
                    break

                processed_frame = self._process_frame(frame, frame_idx)

                if self.mode == "post" and self.out and self.out.isOpened():
                    try: self.out.write(processed_frame)
                    except Exception as write_e: logger.error(f"Write frame {frame_idx} failed: {write_e}")

                if self.display:
                    # <<< MODIFIED: Added 'L:Lightning' to help text >>>
                    mode_keys = "1:LED 2:LED+ 3:Hue 4:Aura L:Lightning 7:Bright 8:Sat 9:Vig 0:None"
                    rvm_status = f"5:RVM({self._requested_device.upper()})" if self.rvm_available else "5:RVM(N/A)"
                    if self.rvm_available:
                        rvm_status += f"[{self.RVM_DISPLAY_MODES[self.rvm_display_mode]}:T]"
                        if len(self.rvm_models) > 1: rvm_status += "[Mdl:R]"
                    sam_status = "6:SAM(N/A)"
                    if self.sam_available:
                        sam_effect_name = 'sam2_segmentation' if self.use_sam2_runtime else 'sam1_segmentation'
                        sam_type = "SAM2" if self.use_sam2_runtime else "SAMv1"
                        sam_status = f"6:{sam_type}({self._sam_device.upper()})" if sam_effect_name in self.effects else "6:SAM(Error?)"
                    info_text = f"FX: {self.current_effect.upper()} | Keys: {mode_keys} {rvm_status} {sam_status} Q:QUIT"

                    # --- Draw info text overlay --- (unchanged text drawing logic)
                    text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_x = 5
                    text_y = self.frame_height - 7
                    bg_y1 = max(0, text_y - text_size[1] - 3)
                    bg_y2 = min(self.frame_height, text_y + 3)
                    bg_x1 = max(0, text_x - 2)
                    bg_x2 = min(self.frame_width, text_x + text_size[0] + 4)
                    if bg_y1 < bg_y2 and bg_x1 < bg_x2:
                        try:
                            sub_img = processed_frame[bg_y1:bg_y2, bg_x1:bg_x2]
                            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                            res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
                            processed_frame[bg_y1:bg_y2, bg_x1:bg_x2] = res
                        except Exception as txt_bg_err:
                            logger.debug(f"Text background draw failed: {txt_bg_err}") # Be less verbose
                    cv2.putText(processed_frame, info_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    try: cv2.imshow(self.window_name, processed_frame)
                    except cv2.error as e: logger.warning(f"cv2.imshow failed: {e}")

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): logger.info("Quit key pressed."); break
                    elif key == ord('1'): self.current_effect = 'led_base'
                    elif key == ord('2'): self.current_effect = 'led_enhanced'
                    elif key == ord('3'): self.current_effect = 'led_hue_rotate'
                    elif key == ord('4'): self.current_effect = 'goldenaura'
                    elif key == ord('5') and self.rvm_available: self.current_effect = 'rvm_composite'
                    elif key == ord('6') and self.sam_available: self.current_effect = 'sam2_segmentation' if self.use_sam2_runtime else 'sam1_segmentation'
                    #elif key == ord('7'): self.current_effect = 'brightness_contrast'
                    #elif key == ord('8'): self.current_effect = 'saturation'
                    #elif key == ord('9'): self.current_effect = 'vignette'
                    elif key == ord('0'): self.current_effect = 'none'
                    # <<< ADDED: Key binding for Lightning effect >>>
                    elif key == ord('l'):
                        self.current_effect = 'lightning'
                        logger.info("Switched to Lightning effect.")
                    # --- RVM Toggles --- (unchanged)
                    elif key == ord('t') and self.current_effect == 'rvm_composite' and self.rvm_available:
                        self.rvm_display_mode = (self.rvm_display_mode + 1) % len(self.RVM_DISPLAY_MODES)
                        logger.info(f"RVM display mode toggled to: {self.RVM_DISPLAY_MODES[self.rvm_display_mode]}")
                    elif key == ord('r') and self.current_effect == 'rvm_composite' and self.rvm_available and len(self.rvm_models) > 1:
                        self.current_rvm_model_idx = (self.current_rvm_model_idx + 1) % len(self.rvm_model_names)
                        logger.info(f"RVM model switched to: {self.rvm_model_names[self.current_rvm_model_idx]}")

                frame_idx += 1
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Exiting.")
                break
            except Exception as e:
                logger.error(f"Frame processing loop error at frame {frame_idx}: {e}", exc_info=True)
                break # Exit loop on critical error

        self._cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pixel Sensing Effect Runner")
    parser.add_argument('--input', default='0', help="Input source (camera index or video file path)")
    parser.add_argument('--mode', type=str, default="live", choices=["live", "post"], help="Mode: live or post-processing")
    parser.add_argument('--output', type=str, default="output.mp4", help="Output video file path (for post mode)")
    # <<< MODIFIED: Default effect can now be 'lightning' >>>
    parser.add_argument('--effect', type=str, default="lightning", help="Initial effect to apply") # Default to lightning for testing
    parser.add_argument('--display', action='store_true', help="Enable display output")
    parser.add_argument('--no-display', action='store_false', dest='display', help="Disable display output") # Allow explicit disabling
    parser.set_defaults(display=True) # Default to display=True if not specified
    parser.add_argument('--rvm_background', type=str, default=None, help="Path to RVM background image")
    parser.add_argument('--sam_checkpoint', type=str, default=None, help="Path to SAM checkpoint")
    parser.add_argument('--use_sam2', action='store_true', help="Use SAM 2 instead of SAM v1")
    parser.add_argument('--log-level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")

    args = parser.parse_args()

    # Convert input to int if it's a camera index, otherwise keep as path
    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Determine if display should be active
    should_display = args.display and (args.mode == 'live' or not args.output)

    # Initialize and run the effect runner
    runner = PixelSensingEffectRunner(
        input_source=input_source,
        mode=args.mode,
        output_path=args.output,
        effect=args.effect,
        display=should_display,
        rvm_background_path=args.rvm_background,
        sam_checkpoint_path=args.sam_checkpoint,
        use_sam2=args.use_sam2
    )
    runner.run()

    # Example command to run with the new effect:
    # python video_effects.py --display --effect lightning
    # Or use the default and press 'L'
    # python video_effects.py --display