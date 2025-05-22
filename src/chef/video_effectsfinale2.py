import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import logging
import torch
import colour
from colour.models import RGB_COLOURSPACE_sRGB, RGB_to_XYZ, XYZ_to_Oklab
from colour.appearance import VIEWING_CONDITIONS_CAM16, CAM_Specification_CAM16, XYZ_to_CAM16, CAM16_to_XYZ
import torchvision.transforms as T
import argparse
import os
import random
import math
import time
from collections import deque
import mediapipe as mp

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

try:
    import colour
    _colour_science_available = True
except ImportError:
    _colour_science_available = False

# --- Device Setup ---
_torch_device = 'cpu'
if torch.cuda.is_available():
    _torch_device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    try:
        tensor_mps = torch.tensor([1.0], device='mps')
        _ = tensor_mps * tensor_mps
        _torch_device = 'mps'
    except Exception:
        _torch_device = 'cpu'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_HEIGHT, DEFAULT_WIDTH = 480, 640

# =========================================================================
# Utility Functions
# =========================================================================
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

# =========================================================================
# Effect Base Classes
# =========================================================================
class Effect(ABC):
    """Base class for all effects."""
    @abstractmethod
    def apply(self, *args, **kwargs) -> np.ndarray:
        """Apply the effect. Subclasses define specific arguments."""
        pass

class FrameEffect(Effect):
    """Effect applied to a single frame."""
    @abstractmethod
    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        pass

class TransitionEffect(Effect):
    """Effect applied between two frames."""
    @abstractmethod
    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float, **kwargs) -> np.ndarray:
        pass

class VideoEffect(Effect):
    """Effect requiring video context or state."""
    @abstractmethod
    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def reset(self) -> None:
        """Reset internal state (optional)."""
        pass

class CompositeEffect(Effect):
    """Chains multiple effects together."""
    def __init__(self, effects: List[Effect]):
        self.effects = effects

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        result = frame
        for effect in self.effects:
            if isinstance(effect, FrameEffect) or isinstance(effect, VideoEffect):
                result = effect.apply(result, **kwargs)
            elif isinstance(effect, TransitionEffect):
                frame_b = kwargs.get('frame_b')
                progress = kwargs.get('progress', 0.5)
                if frame_b is not None:
                    result = effect.apply(result, frame_b, progress, **kwargs)
                else:
                    logger.warning("Transition effect skipped: frame_b not provided")
            else:
                logger.warning(f"Unsupported effect type: {type(effect)}")
        return result

    def reset(self) -> None:
        for effect in self.effects:
            if hasattr(effect, 'reset'):
                effect.reset()

# =========================================================================
# Effect Registry and Factory
# =========================================================================
EFFECT_REGISTRY = {}

def register_effect(name: str):
    """Decorator to register an effect class."""
    def decorator(cls):
        EFFECT_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def create_effect(name: str, **params) -> Effect:
    """Factory function to create an effect instance."""
    effect_class = EFFECT_REGISTRY.get(name.lower())
    if not effect_class:
        raise ValueError(f"Unknown effect: {name}. Available: {list(EFFECT_REGISTRY.keys())}")
    try:
        return effect_class(**params)
    except TypeError as e:
        raise TypeError(f"Invalid parameters for {name}: {params}. Error: {e}")

# =========================================================================
# Effect Implementations
# =========================================================================
@register_effect('brightness_contrast')
@dataclass
class BrightnessContrast(FrameEffect):
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    per_channel: bool = False
    fade_in: bool = False

    def __post_init__(self):
        self.brightness = np.clip(self.brightness, -1.0, 1.0)
        self.contrast = np.clip(self.contrast, 0.0, 3.0)
        self.gamma = max(0.1, self.gamma)

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        frame_time = kwargs.get('frame_time', 0.0)
        clip_duration = kwargs.get('clip_duration', 1.0)
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

@register_effect('saturation')
@dataclass
class Saturation(FrameEffect):
    scale: float = 1.0
    vibrance: float = 0.0
    fade_in: bool = False

    def __post_init__(self):
        self.scale = max(0.0, self.scale)
        self.vibrance = np.clip(self.vibrance, 0.0, 1.0)

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        frame_time = kwargs.get('frame_time', 0.0)
        clip_duration = kwargs.get('clip_duration', 1.0)
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

@register_effect('vignette')
@dataclass
class Vignette(FrameEffect):
    strength: float = 0.5
    radius: float = 0.6
    falloff: float = 0.3
    shape: str = 'circular'
    color: Tuple[int, int, int] = (0, 0, 0)
    fade_in: bool = False

    def __post_init__(self):
        self.strength = np.clip(self.strength, 0.0, 1.0)
        self.radius = np.clip(self.radius, 0.0, 1.0)
        self.falloff = max(0.01, self.falloff)
        self.shape = self.shape if self.shape in ['circular', 'elliptical', 'rectangular'] else 'circular'

    def _create_mask(self, height: int, width: int, factor: float) -> np.ndarray:
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

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        frame_time = kwargs.get('frame_time', 0.0)
        clip_duration = kwargs.get('clip_duration', 1.0)
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

@register_effect('blur')
@dataclass
class Blur(FrameEffect):
    kernel_size: int = 5
    sigma: float = 0.0
    fade_in: bool = False

    def __post_init__(self):
        self.kernel_size = max(3, self.kernel_size if self.kernel_size % 2 != 0 else self.kernel_size + 1)
        self.sigma = max(0.0, self.sigma)

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        frame_time = kwargs.get('frame_time', 0.0)
        clip_duration = kwargs.get('clip_duration', 1.0)
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

@register_effect('crossfade')
@dataclass
class Crossfade(TransitionEffect):
    duration: float = 0.5

    def __post_init__(self):
        self.duration = max(0.01, self.duration)

    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float, **kwargs) -> np.ndarray:
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

@register_effect('slide')
@dataclass
class Slide(TransitionEffect):
    direction: str = 'left'
    duration: float = 0.5

    def __post_init__(self):
        vd = {'left', 'right', 'up', 'down'}
        self.duration = max(0.01, self.duration)
        if self.direction not in vd:
            logger.warning(f"Invalid direction: {self.direction}. Defaulting to 'left'")
            self.direction = 'left'

    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float, **kwargs) -> np.ndarray:
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

@register_effect('motion_blur')
class MotionBlurEffect(VideoEffect):
    def __init__(self, kernel_size: int = 19):
        self.kernel_size = max(3, kernel_size if kernel_size % 2 != 0 else kernel_size + 1)

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        k = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        k[self.kernel_size // 2, :] = 1. / self.kernel_size
        try:
            return cv2.filter2D(frame, -1, k)
        except cv2.error as e:
            logger.warning(f"MotionBlur fail: {e}")
            return frame

    def reset(self) -> None:
        pass

@register_effect('advanced_color_grade')
class AdvancedColorGradeEffect(VideoEffect):
    def __init__(self,
                 base_contrast: float = 1.3,
                 base_brightness: float = 0.1,
                 chroma_boost: float = 1.5,
                 vignette_strength: float = 0.3,
                 micro_variation: float = 0.02):
        self.base_contrast = max(0.1, base_contrast)
        self.base_brightness = base_brightness
        self.chroma_boost = max(1.0, chroma_boost)
        self.vignette_strength = max(0.0, min(1.0, vignette_strength))
        self.micro_variation = max(0.0, micro_variation)
        if not _colour_science_available:
            logger.warning("colour-science not available; falling back to basic color grading.")
        if _colour_science_available:
            self.viewing_conditions = VIEWING_CONDITIONS_CAM16["Average"]
            self.L_A = 200.0
            self.Y_b = 20.0

    def dynamic_scene_analysis(self, rgb: np.ndarray) -> Tuple[float, float]:
        XYZ = RGB_to_XYZ(rgb, RGB_COLOURSPACE_sRGB.whitepoint, RGB_COLOURSPACE_sRGB.whitepoint, RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ)
        oklab = XYZ_to_Oklab(XYZ)
        L = oklab[..., 0]
        a = oklab[..., 1]
        b = oklab[..., 2]
        chroma = np.sqrt(a**2 + b**2)
        avg_luminance = np.mean(L)
        avg_chroma = np.mean(chroma)
        return avg_luminance, avg_chroma

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        h, w = frame.shape[:2]
        try:
            if _colour_science_available:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                linear_rgb = RGB_COLOURSPACE_sRGB.cctf_decoding(rgb)
                avg_luminance, avg_chroma = self.dynamic_scene_analysis(linear_rgb)
                XYZ = RGB_to_XYZ(linear_rgb, RGB_COLOURSPACE_sRGB.whitepoint, RGB_COLOURSPACE_sRGB.whitepoint, RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ)
                cam16 = XYZ_to_CAM16(XYZ, self.viewing_conditions, L_A=self.L_A, Y_b=self.Y_b)
                J = cam16.J / 100.0
                C = cam16.C
                h_val = cam16.h
                contrast = self.base_contrast * (1.0 + 0.2 * (1.0 - avg_luminance))
                brightness = self.base_brightness * (1.0 + 0.3 * (avg_luminance - 0.5))
                J = np.clip(contrast * (J - 0.5) + 0.5 + brightness, 0.0, 1.0) * 100.0
                C = C * (self.chroma_boost + np.tanh(C / 50.0))
                shadow_mask = 1.0 - (J / 100.0)
                highlight_mask = J / 100.0
                h_adjusted = h_val + shadow_mask * 20.0 - highlight_mask * 20.0
                h_adjusted = h_adjusted % 360.0
                if self.micro_variation > 0:
                    J += np.random.uniform(-self.micro_variation, self.micro_variation, (h, w)) * 100.0
                    C += np.random.uniform(-self.micro_variation * 10, self.micro_variation * 10, (h, w))
                    J = np.clip(J, 0.0, 100.0)
                    C = np.clip(C, 0.0, None)
                x = np.tile(np.linspace(-1, 1, w), (h, 1))
                y = np.repeat(np.linspace(-1, 1, h).reshape(-1, 1), w, axis=1)
                r = np.sqrt(x**2 + y**2)
                vignette = 1.0 - self.vignette_strength * r
                J = J * vignette
                cam16_adjusted = CAM_Specification_CAM16(J=J, C=C, h=h_adjusted)
                XYZ_adjusted = CAM16_to_XYZ(cam16_adjusted, self.viewing_conditions, L_A=self.L_A, Y_b=self.Y_b)
                rgb_adjusted = colour.XYZ_to_RGB(XYZ_adjusted, RGB_COLOURSPACE_sRGB.whitepoint, RGB_COLOURSPACE_sRGB.whitepoint, RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB)
                rgb_final = RGB_COLOURSPACE_sRGB.cctf_encoding(rgb_adjusted)
                rgb_final = np.clip(rgb_final, 0.0, 1.0)
                output = (rgb_final * 255.0).astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            else:
                output = frame.astype(np.float32)
                output = output * self.base_contrast + self.base_brightness * 255.0
                output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.putText(output, "Proprietary Grade Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
            return output
        except Exception as e:
            logger.error(f"Color grading failed: {e}")
            return frame

    def reset(self) -> None:
        pass

@register_effect('chromatic_aberration')
class ChromaticAberrationEffect(VideoEffect):
    def __init__(self, base_strength: float = 0.05, center: Optional[Tuple[float, float]] = None, non_linear_exponent: float = 3.0, edge_boost: float = 4.0):
        self.base_strength = max(0.0, base_strength)
        self.center = center
        self.non_linear_exponent = max(1.0, non_linear_exponent)
        self.edge_boost = max(1.0, edge_boost)

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        h, w = frame.shape[:2]
        try:
            cx, cy = (w / 2.0, h / 2.0) if self.center is None else (self.center[0] * w, self.center[1] * h)
            cx, cy = np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)
            map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            map_y = np.repeat(np.arange(h, dtype=np.float32).reshape(-1, 1), w, axis=1)
            delta_x = map_x - cx
            delta_y = map_y - cy
            r = np.sqrt(delta_x**2 + delta_y**2)
            r_max = max(np.sqrt(cx**2 + cy**2) if (cx <= w/2 and cy <= h/2) else np.sqrt((w - cx)**2 + (h - cy)**2), 1e-6)
            r_normalized = r / r_max
            non_linear_factor = self.edge_boost * (r_normalized ** self.non_linear_exponent)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
            saturation = hsv[..., 1]
            k_map = self.base_strength * (1.5 + saturation)
            scale_r = 1.0 + k_map * non_linear_factor
            scale_b = 1.0 - k_map * non_linear_factor
            map_x_r = cx + delta_x * scale_r
            map_y_r = cy + delta_y * scale_r
            map_x_b = cx + delta_x * scale_b
            map_y_b = cy + delta_y * scale_b
            b, g, r = cv2.split(frame)
            r_shifted = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            b_shifted = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            output = cv2.merge((b_shifted, g, r_shifted))
            border_size = 5
            output[:border_size, :] = (0, 0, 255)
            output[-border_size:, :] = (0, 0, 255)
            output[:, :border_size] = (255, 0, 0)
            output[:, -border_size:] = (255, 0, 0)
            cv2.putText(output, "Chroma Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            return output
        except Exception as e:
            logger.error(f"Enhanced ChromaAberr fail: {e}")
            return frame

    def reset(self) -> None:
        pass

# =========================================================================
# Model Loaders
# =========================================================================
def load_rvm_model(device: str, model_name: str = 'resnet50', pretrained: bool = True) -> Optional[torch.nn.Module]:
    if not torch.__version__:
        logger.warning("PyTorch not available. RVM cannot be loaded.")
        return None
    try:
        model = torch.hub.load('PeterL1n/RobustVideoMatting', model_name, pretrained=pretrained)
        return model.to(device).eval()
    except Exception as e:
        logger.error(f"Failed to load RVM model '{model_name}': {e}")
        return None

def load_sam_mask_generator(device: str, model_type: str = 'vit_t', checkpoint_path: Optional[str] = None) -> Optional[Any]:
    if not _sam_available or not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        return SamAutomaticMaskGenerator(sam)
    except Exception as e:
        logger.error(f"Failed to load SAM v1 model: {e}")
        return None

def load_sam2_video_predictor(device: str, checkpoint_path: Optional[str] = None) -> Optional[Any]:
    if not _sam2_available or not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    try:
        model = SAM(checkpoint_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load SAM 2 model: {e}")
        return None

# =========================================================================
# Pixel Sensing Effect Runner
# =========================================================================
class PixelSensingEffectRunner:
    LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    LK_PARAMS_GOLD = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    MAX_HISTORY = 18
    TRAIL_START_ALPHA = 0.7
    TRAIL_END_ALPHA = 0.0
    TRAIL_RADIUS = 2
    TRAIL_BLUR_KERNEL = (5, 5)
    GOLD_TINT_COLOR = (30, 165, 210)
    TINT_STRENGTH = 0.40
    MASK_BLUR_KERNEL = (15, 15)
    SEGMENTATION_THRESHOLD = 0.6
    RVM_DOWNSAMPLE_RATIO = 0.4
    RVM_DEFAULT_BG_COLOR = (0, 0, 255)
    SAM_MASK_ALPHA = 0.5
    RVM_DISPLAY_MODES = ["Composite", "Alpha", "Foreground"]

    def __init__(self,
                 input_source: Any = 0,
                 mode: str = "live",
                 output_path: Optional[str] = None,
                 effect: str = "none",
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
        self.display = display if display is not None else (self.mode == "live")
        self._init_rvm_bg_path = rvm_background_path
        self._requested_device = _torch_device
        self.device = self._requested_device

        self.cap = None
        self.out = None
        self.frame_height: Optional[int] = None
        self.frame_width: Optional[int] = None
        self.fps: float = 30.0

        # MediaPipe Initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True)

        # State Variables
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_landmarks_flow: Optional[List[Tuple[int, int]]] = None
        self.prev_face_pts_gold: np.ndarray = np.empty((0, 1, 2), dtype=np.float32)
        self.prev_pose_pts_gold: np.ndarray = np.empty((0, 1, 2), dtype=np.float32)
        self.flow: Optional[np.ndarray] = None
        self.hue_offset: int = 0
        self.face_history = deque(maxlen=self.MAX_HISTORY)
        self.pose_history = deque(maxlen=self.MAX_HISTORY)

        # Lightning State
        self.lightning_style_index = 0
        self.lightning_style_names = ["Random", "Movement"]
        self.lightning_styles: List[callable] = []

        # RVM Initialization
        self.rvm_models: Dict[str, Optional[torch.nn.Module]] = {}
        self.rvm_model_names: List[str] = ['resnet50', 'mobilenetv3']
        self.current_rvm_model_idx: int = 0
        self.rvm_available: bool = False
        for model_name in self.rvm_model_names:
            model = load_rvm_model(self._requested_device, model_name)
            if model:
                self.rvm_models[model_name] = model
                self.rvm_available = True
        self.rvm_rec: List[Optional[torch.Tensor]] = [None] * 4
        self.rvm_downsample_ratio: float = self.RVM_DOWNSAMPLE_RATIO
        self.rvm_background: Optional[np.ndarray] = None
        self.rvm_display_mode: int = 0

        # SAM/SAM2 Initialization
        actual_sam_checkpoint = sam_checkpoint_path if sam_checkpoint_path else ""
        self.use_sam2_runtime = use_sam2 and _sam2_available and os.path.exists(actual_sam_checkpoint)
        self.sam_model: Optional[Any] = None
        self.sam_available: bool = False
        self.sam2_state: Optional[Any] = None
        self._sam_device = self.device
        if self.use_sam2_runtime:
            sam2_device_to_use = 'cpu' if self._requested_device == 'mps' else self._requested_device
            self._sam_device = sam2_device_to_use
            self.sam_model = load_sam2_video_predictor(sam2_device_to_use, actual_sam_checkpoint)
            if self.sam_model:
                self.sam_available = True
        elif _sam_available and os.path.exists(actual_sam_checkpoint):
            sam_v1_model_type = 'vit_t'
            if 'vit_l' in actual_sam_checkpoint: sam_v1_model_type = 'vit_l'
            elif 'vit_h' in actual_sam_checkpoint: sam_v1_model_type = 'vit_h'
            elif 'vit_b' in actual_sam_checkpoint: sam_v1_model_type = 'vit_b'
            self._sam_device = self._requested_device
            self.sam_model = load_sam_mask_generator(self._sam_device, model_type=sam_v1_model_type, checkpoint_path=actual_sam_checkpoint)
            if self.sam_model:
                self.sam_available = True

        # Effect Initialization
        self.current_effect = self._initialize_effect(effect)
        self.current_effect_name = effect  # Track the effect name
        self.window_name = f"Pixel Effects (Mode: {self.mode}, Device: {self._requested_device.upper()})"
        self.lightning_styles = [self._apply_lightning_style_random, self._apply_lightning_style_movement]

    def _initialize_effect(self, effect_name: str) -> Effect:
        try:
            if effect_name == 'none':
                return FrameEffectWrapper(lambda frame, **kwargs: frame)
            elif effect_name == 'motion_blur':
                return create_effect('motion_blur', kernel_size=19)
            elif effect_name == 'advanced_color_grade':
                return create_effect('advanced_color_grade', base_contrast=1.3, base_brightness=0.1, chroma_boost=1.5, vignette_strength=0.3, micro_variation=0.0)
            elif effect_name == 'chromatic_aberration':
                return create_effect('chromatic_aberration', base_strength=0.05, center=None, non_linear_exponent=3.0, edge_boost=4.0)
            elif effect_name == 'led_base':
                return FrameEffectWrapper(self._apply_led_base_style)
            elif effect_name == 'led_enhanced':
                return FrameEffectWrapper(self._apply_led_enhanced_style)
            elif effect_name == 'led_hue_rotate':
                return FrameEffectWrapper(self._apply_led_hue_rotate_style)
            elif effect_name == 'goldenaura':
                return FrameEffectWrapper(self._apply_goldenaura_style)
            elif effect_name == 'lightning':
                return FrameEffectWrapper(self._apply_lightning_cycle)
            elif effect_name == 'rvm_composite' and self.rvm_available:
                return FrameEffectWrapper(self._apply_rvm_composite_style)
            elif effect_name in ['sam1_segmentation', 'sam2_segmentation'] and self.sam_available:
                return FrameEffectWrapper(self._apply_sam_segmentation_style)
            else:
                return create_effect(effect_name)
        except ValueError as e:
            logger.warning(f"Effect error: {e}. Defaulting to 'none'.")
            return FrameEffectWrapper(lambda frame, **kwargs: frame)

    def _load_rvm_background(self, path: Optional[str]):
        loaded_bg = None
        if path and os.path.exists(path):
            try:
                loaded_bg = cv2.imread(path)
                if loaded_bg is not None and loaded_bg.size > 0:
                    logger.info(f"Loaded RVM background from {path}")
                else:
                    loaded_bg = None
            except Exception as e:
                logger.warning(f"Error loading RVM background from {path}: {e}")
                loaded_bg = None
        if loaded_bg is None:
            h = self.frame_height if self.frame_height is not None else DEFAULT_HEIGHT
            w = self.frame_width if self.frame_width is not None else DEFAULT_WIDTH
            if h <= 0 or w <= 0:
                h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
            self.rvm_background = np.full((h, w, 3), self.RVM_DEFAULT_BG_COLOR, dtype=np.uint8)
        else:
            self.rvm_background = loaded_bg
            if self.frame_height is not None and self.frame_width is not None:
                self.rvm_background = resize_frame(self.rvm_background, (self.frame_height, self.frame_width))

    def _initialize_capture(self) -> bool:
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
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._load_rvm_background(self._init_rvm_bg_path)
        if self.mode == "post" and self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            try:
                self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
                if not self.out.isOpened():
                    logger.error(f"Failed to open VideoWriter: {self.output_path}")
                    self.cap.release()
                    return False
            except Exception as e:
                logger.error(f"Error initializing VideoWriter: {e}")
                self.cap.release()
                return False
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return True

    def _cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.out and self.out.isOpened():
            self.out.release()
        if self.face_mesh:
            self.face_mesh.close()
        if self.pose:
            self.pose.close()
        if self.rvm_models:
            self.rvm_models.clear()
        if self.sam_model:
            self.sam_model = None
        if self._requested_device == 'cuda':
            torch.cuda.empty_cache()
        if self.display:
            cv2.destroyAllWindows()

    # --- Effect Application Methods ---
    def _apply_distortion(self, frame: np.ndarray) -> np.ndarray:
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
        frame = validate_frame(frame)
        if masks is None:
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        num_masks_drawn = 0
        try:
            masks_to_process: List[Any] = []
            if isinstance(masks, torch.Tensor):
                if masks.ndim >= 3:
                    masks_to_process = list(masks.cpu().detach())
                else:
                    logger.warning(f"Received tensor mask with unexpected shape: {masks.shape}")
            elif isinstance(masks, list):
                masks_to_process = masks
            else:
                logger.warning(f"Unsupported mask input type: {type(masks)}")
                return frame

            if not masks_to_process:
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
                    mask_data = mask_info.cpu().numpy()
                else:
                    continue

                if mask_data is None or mask_data.size == 0:
                    continue

                mask_bool = (mask_data > 0.5) if mask_data.dtype in (np.float32, np.float64, torch.float32, torch.float64) else mask_data.astype(bool)

                if mask_bool.shape != (h, w):
                    if mask_bool.ndim == 2:
                        mask_resized = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
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
        except Exception as e:
            logger.error(f"Error drawing SAM masks: {e}")
            return frame
        return overlay

    def _preprocess_frame_rvm(self, frame: np.ndarray) -> Optional[torch.Tensor]:
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
        try:
            if fgr is None or pha is None or not isinstance(fgr, torch.Tensor) or not isinstance(pha, torch.Tensor):
                return None, None
            fgr_cpu = fgr.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            pha_cpu = pha.squeeze(0).cpu().detach().numpy()
            if pha_cpu.ndim == 3 and pha_cpu.shape[0] == 1:
                pha_cpu = pha_cpu.squeeze(0)
            tw = int(max(1, self.frame_width if self.frame_width is not None else DEFAULT_WIDTH))
            th = int(max(1, self.frame_height if self.frame_height is not None else DEFAULT_HEIGHT))
            fgr_r = cv2.resize(fgr_cpu, (tw, th), interpolation=cv2.INTER_LINEAR)
            pha_r = cv2.resize(pha_cpu, (tw, th), interpolation=cv2.INTER_LINEAR)
            fgr_f = np.clip(fgr_r * 255.0, 0, 255).astype(np.uint8)
            pha_f = np.clip(pha_r, 0.0, 1.0)
            return fgr_f, pha_f
        except Exception as e:
            logger.error(f"RVM postprocess failed: {e}")
            return None, None

    def _apply_led_base_style(self, **kwargs) -> np.ndarray:
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
                        hull = cv2.convexHull(pts)
                        cv2.fillConvexPoly(cm, hull, 255)
            if self.pose and pr and pr.segmentation_mask is not None:
                smf = cv2.resize(pr.segmentation_mask, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                smu = (smf > self.SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
                cm = cv2.bitwise_or(cm, smu)
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
            logger.error(f"GoldenAura failed: {e}")
            return frame
        return proc

    def _apply_rvm_composite_style(self, **kwargs) -> np.ndarray:
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
            self._load_rvm_background(None)
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
            fgr_np, pha_np = self._postprocess_output_rvm(fgr_tensor, pha_tensor)
            if fgr_np is None or pha_np is None:
                cv2.putText(frame, "RVM Postproc Err", (10, 90), 0, 0.7, (0, 0, 255), 2)
                return frame
            output_frame = frame
            current_mode_str = self.RVM_DISPLAY_MODES[self.rvm_display_mode]
            if self.rvm_display_mode == 0:
                alpha_3c = pha_np[..., np.newaxis]
                composite = (fgr_np * alpha_3c + self.rvm_background * (1.0 - alpha_3c))
                output_frame = np.clip(composite, 0, 255).astype(np.uint8)
            elif self.rvm_display_mode == 1:
                alpha_display = (pha_np * 255.0).astype(np.uint8)
                output_frame = cv2.cvtColor(alpha_display, cv2.COLOR_GRAY2BGR)
            elif self.rvm_display_mode == 2:
                output_frame = fgr_np
            info_txt = f"RVM: {model_name} ({current_mode_str})"
            cv2.putText(output_frame, info_txt, (10, 30), 0, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output_frame, info_txt, (10, 30), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return output_frame
        except Exception as e:
            logger.error(f"RVM effect failed ('{model_name}', mode={self.rvm_display_mode}): {e}")
            self.rvm_rec = [None] * 4
            cv2.putText(frame, "RVM Error", (10, 90), 0, 0.7, (0, 0, 255), 2)
            return frame

    def _apply_sam_segmentation_style(self, **kwargs) -> np.ndarray:
        frame = validate_frame(kwargs['frame'])
        pose_results = kwargs.get('pose_results')
        if not self.sam_available or self.sam_model is None:
            cv2.putText(frame, "SAM N/A", (10, 60), 0, 0.7, (0, 0, 255), 2)
            return frame
        sam_type_str = "SAM2" if self.use_sam2_runtime else "SAMv1"
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks: Any = None
            if self.use_sam2_runtime:
                points_list = []
                if self.pose and pose_results and pose_results.pose_landmarks:
                    for lm in pose_results.pose_landmarks.landmark:
                        if lm.visibility > 0.3: points_list.append([int(lm.x * self.frame_width), int(lm.y * self.frame_height)])
                if not points_list: points_list.append([self.frame_width // 2, self.frame_height // 2])
                points = np.array(points_list)
                labels = np.ones(len(points), dtype=int)
                results = self.sam_model(frame_rgb, points=points, labels=labels, device=self._sam_device)
                if results and isinstance(results, list) and len(results) > 0:
                    first_result = results[0]
                    if hasattr(first_result, 'masks') and first_result.masks is not None and hasattr(first_result.masks, 'data'):
                        masks = first_result.masks.data
            else:
                masks = self.sam_model.generate(frame_rgb)
            if masks is not None and ((isinstance(masks, list) and masks) or (isinstance(masks, torch.Tensor) and masks.numel() > 0)):
                num_drawn = len(masks) if isinstance(masks, list) else masks.shape[0]
                processed_frame = self._draw_sam_masks(frame, masks)
                cv2.putText(processed_frame, f"{sam_type_str} Masks: {num_drawn}", (10, 30), 0, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(processed_frame, f"{sam_type_str} Masks: {num_drawn}", (10, 30), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                return processed_frame
            else:
                cv2.putText(frame, f"{sam_type_str}: No Masks", (10, 90), 0, 0.7, (0, 255, 255), 2)
                return frame
        except Exception as e:
            logger.error(f"{sam_type_str} effect failed: {e}")
            cv2.putText(frame, f"{sam_type_str} Err", (10, 90), 0, 0.7, (0, 0, 255), 2)
            return frame

    def _apply_lightning_style_random(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        frame = validate_frame(frame)
        h, w = frame.shape[:2]
        proc = frame.copy()
        try:
            num_bolts = random.randint(1, 3)
            for _ in range(num_bolts):
                if random.random() < 0.5 and landmarks:
                    start = random.choice(landmarks)
                else:
                    start = (random.randint(0, w - 1), random.randint(0, h - 1))
                end = (random.randint(0, w - 1), random.randint(0, h - 1))
                length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                segments = int(length / 10) + 1
                points = [start]
                for _ in range(segments - 1):
                    last = points[-1]
                    angle = math.atan2(end[1] - start[1], end[0] - start[0]) + random.uniform(-0.5, 0.5)
                    step = length / segments
                    new_x = last[0] + step * math.cos(angle) + random.uniform(-5, 5)
                    new_y = last[1] + step * math.sin(angle) + random.uniform(-5, 5)
                    points.append((int(new_x), int(new_y)))
                points.append(end)
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                        cv2.line(proc, p1, p2, (255, 245, 200), 2, cv2.LINE_AA)
                        cv2.line(proc, p1, p2, (200, 220, 255), 1, cv2.LINE_AA)
        except Exception as e:
            logger.warning(f"Random lightning failed: {e}")
        return proc

    def _apply_lightning_style_movement(self, frame: np.ndarray, flow: Optional[np.ndarray], prev_pts: Optional[np.ndarray]) -> np.ndarray:
        frame = validate_frame(frame)
        h, w = frame.shape[:2]
        proc = frame.copy()
        if flow is None or prev_pts is None or flow.shape[0] != prev_pts.shape[0]:
            return proc
        try:
            for i, (new, old) in enumerate(zip(flow, prev_pts)):
                xn, yn = new.ravel()
                xo, yo = old.ravel()
                if 0 <= xn < w and 0 <= yn < h and 0 <= xo < w and 0 <= yo < h:
                    mag = np.linalg.norm(new - old)
                    if mag > 5.0:
                        cv2.line(proc, (int(xo), int(yo)), (int(xn), int(yn)), (255, 245, 200), 2, cv2.LINE_AA)
                        cv2.line(proc, (int(xo), int(yo)), (int(xn), int(yn)), (200, 220, 255), 1, cv2.LINE_AA)
        except Exception as e:
            logger.warning(f"Movement lightning failed: {e}")
        return proc

    def _apply_lightning_cycle(self, **kwargs) -> np.ndarray:
        frame = validate_frame(kwargs['frame'])
        lm = kwargs.get('landmarks', [])
        fl = kwargs.get('flow')
        pf = kwargs.get('prev_pts_flow')
        style_func = self.lightning_styles[self.lightning_style_index % len(self.lightning_styles)]
        proc = style_func(frame, lm) if self.lightning_style_index == 0 else style_func(frame, fl, pf)
        cv2.putText(proc, f"Lightning: {self.lightning_style_names[self.lightning_style_index]}", (10, 30), 0, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(proc, f"Lightning: {self.lightning_style_names[self.lightning_style_index]}", (10, 30), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return proc

    # Continuation from provided code (all imports, utility functions, effect classes, model loaders, and PixelSensingEffectRunner up to run method remain as you shared)

    def run(self):
        if not self._initialize_capture():
            self._cleanup()
            return
        frame_count = 0
        start_time = time.time()
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    break
                frame_count += 1
                frame = validate_frame(frame)

                # Process with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_mesh.process(frame_rgb) if self.face_mesh else None
                pose_results = self.pose.process(frame_rgb) if self.pose else None

                # Optical Flow
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = []
                flow, prev_pts_flow = None, None
                if self.prev_gray is not None and self.face_mesh and face_results and face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        for lm in face_landmarks.landmark:
                            x, y = int(lm.x * self.frame_width), int(lm.y * self.frame_height)
                            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                                landmarks.append((x, y))
                    if landmarks:
                        curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
                        if self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks):
                            prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                            flow, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None, **self.LK_PARAMS)
                            good_new = flow[status == 1]
                            good_old = prev_pts[status == 1]
                            flow = good_new.reshape(-1, 1, 2)
                            prev_pts_flow = good_old.reshape(-1, 1, 2)
                self.prev_gray = curr_gray.copy()
                self.prev_landmarks_flow = landmarks if landmarks else self.prev_landmarks_flow

                # Golden Aura history
                if isinstance(self.current_effect, FrameEffectWrapper) and 'goldenaura' in self.current_effect_name:
                    face_landmarks_list = []
                    if self.face_mesh and face_results and face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            for lm in face_landmarks.landmark:
                                if lm.x is not None and lm.y is not None:
                                    face_landmarks_list.append([int(lm.x * self.frame_width), int(lm.y * self.frame_height)])
                    face_pts = np.array(face_landmarks_list, dtype=np.float32) if face_landmarks_list else np.empty((0, 2), dtype=np.float32)
                    pose_landmarks_list = []
                    if self.pose and pose_results and pose_results.pose_landmarks:
                        for lm in pose_results.pose_landmarks.landmark:
                            if lm.visibility > 0.5:
                                pose_landmarks_list.append([int(lm.x * self.frame_width), int(lm.y * self.frame_height)])
                    pose_pts = np.array(pose_landmarks_list, dtype=np.float32) if pose_landmarks_list else np.empty((0, 2), dtype=np.float32)
                    self.face_history.append(face_pts if face_pts.size > 0 else None)
                    self.pose_history.append(pose_pts if pose_pts.size > 0 else None)

                # Apply effect
                processed_frame = self.current_effect.apply(
                    frame=frame,
                    landmarks=landmarks,
                    flow=flow,
                    prev_pts_flow=prev_pts_flow,
                    face_results=face_results,
                    pose_results=pose_results
                )

                # Display FPS
                elapsed = time.time() - start_time
                fps_display = frame_count / elapsed if elapsed > 0 else 0.0
                cv2.putText(
                    processed_frame,
                    f"FPS: {fps_display:.1f}",
                    (10, processed_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

                # Write to output if in post mode
                if self.mode == "post" and self.out:
                    try:
                        self.out.write(processed_frame)
                    except Exception as e:
                        logger.error(f"Failed to write frame: {e}")
                        break

                # Display frame if enabled
                if self.display:
                    try:
                        cv2.imshow(self.window_name, processed_frame)
                    except Exception as e:
                        logger.error(f"Display failed: {e}")
                        break

                # Handle keypresses
                if self.display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('e'):
                        # Cycle through effects
                        effect_keys = [
                            'none', 'brightness_contrast', 'saturation', 'vignette', 'blur',
                            'motion_blur', 'advanced_color_grade', 'chromatic_aberration',
                            'led_base', 'led_enhanced', 'led_hue_rotate', 'goldenaura',
                            'lightning', 'rvm_composite', 'sam1_segmentation', 'sam2_segmentation'
                        ]
                        try:
                            current_idx = effect_keys.index(self.current_effect_name) if self.current_effect_name in effect_keys else 0
                            next_idx = (current_idx + 1) % len(effect_keys)
                            next_effect_name = effect_keys[next_idx]
                            self.current_effect = self._initialize_effect(next_effect_name)
                            self.current_effect_name = next_effect_name
                            logger.info(f"Switched to effect: {next_effect_name}")
                        except Exception as e:
                            logger.error(f"Error cycling effect: {e}")
                    elif key == ord('c'):
                        # Apply composite effect
                        try:
                            composite = CompositeEffect([
                                create_effect('brightness_contrast', brightness=0.1, contrast=1.2),
                                create_effect('vignette', strength=0.5),
                                create_effect('motion_blur', kernel_size=15)
                            ])
                            processed_frame = composite.apply(processed_frame)
                            if self.display:
                                cv2.imshow(self.window_name, processed_frame)
                            logger.info("Applied composite effect")
                        except Exception as e:
                            logger.error(f"Composite effect failed: {e}")
                    elif key == ord('l') and isinstance(self.current_effect, FrameEffectWrapper) and 'lightning' in self.current_effect_name:
                        # Cycle lightning styles
                        self.lightning_style_index = (self.lightning_style_index + 1) % len(self.lightning_styles)
                        logger.info(f"Switched to lightning style: {self.lightning_style_names[self.lightning_style_index]}")
                    elif key == ord('r') and self.rvm_available and isinstance(self.current_effect, FrameEffectWrapper) and 'rvm_composite' in self.current_effect_name:
                        # Cycle RVM models
                        self.current_rvm_model_idx = (self.current_rvm_model_idx + 1) % len(self.rvm_model_names)
                        self.rvm_rec = [None] * 4
                        logger.info(f"Switched to RVM model: {self.rvm_model_names[self.current_rvm_model_idx]}")
                    elif key == ord('d') and self.rvm_available and isinstance(self.current_effect, FrameEffectWrapper) and 'rvm_composite' in self.current_effect_name:
                        # Cycle RVM display modes
                        self.rvm_display_mode = (self.rvm_display_mode + 1) % len(self.RVM_DISPLAY_MODES)
                        logger.info(f"Switched to RVM display mode: {self.RVM_DISPLAY_MODES[self.rvm_display_mode]}")

        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self._cleanup()
            logger.info(f"Processed {frame_count} frames in {elapsed:.2f} seconds")

class FrameEffectWrapper(FrameEffect):
    """Wraps a legacy effect function to conform to the FrameEffect interface."""
    def __init__(self, effect_func: callable):
        self.effect_func = effect_func

    def apply(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        try:
            return self.effect_func(frame=frame, **kwargs)
        except Exception as e:
            logger.error(f"FrameEffectWrapper failed: {e}")
            return validate_frame(frame)

def list_cameras(max_cameras: int = 10) -> List[int]:
    """List available camera indices."""
    available_cameras = []
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                logger.info(f"Found camera at index {i}")
            cap.release()
        except Exception as e:
            logger.debug(f"Camera index {i} not available: {e}")
    return available_cameras

def main():
    parser = argparse.ArgumentParser(description="Pixel Sensing Effect Runner")
    parser.add_argument('--input-source', type=str, default='0', help='Input source (camera index or video file path)')
    parser.add_argument('--mode', type=str, default='live', choices=['live', 'post'], help='Mode: live or post-processing')
    parser.add_argument('--output-path', type=str, default=None, help='Output video file path (for post mode)')
    parser.add_argument('--effect', type=str, default='none', help='Effect to apply')
    parser.add_argument('--display', type=lambda x: x.lower() == 'true', default=True, help='Display output (true/false)')
    parser.add_argument('--rvm-background-path', type=str, default=None, help='Background image for RVM composite')
    parser.add_argument('--sam-checkpoint-path', type=str, default=None, help='Path to SAM/SAM2 checkpoint')
    parser.add_argument('--use-sam2', action='store_true', help='Use SAM2 instead of SAM1')
    args = parser.parse_args()

    # Convert input_source to int if it's a camera index
    try:
        input_source = int(args.input_source)
    except ValueError:
        input_source = args.input_source

    runner = PixelSensingEffectRunner(
        input_source=input_source,
        mode=args.mode,
        output_path=args.output_path,
        effect=args.effect,
        display=args.display,
        rvm_background_path=args.rvm_background_path,
        sam_checkpoint_path=args.sam_checkpoint_path,
        use_sam2=args.use_sam2
    )
    runner.run()

if __name__ == "__main__":
    main()
    
    
#take lighting from this script, add led variantsto 4, & 5.