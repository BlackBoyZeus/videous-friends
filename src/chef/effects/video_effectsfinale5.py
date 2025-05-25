
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard Library
import abc  # For Effect base classes
import argparse
from collections import deque
from dataclasses import dataclass  # For effect data classes
import inspect  # Used in FrameEffectWrapper
import json
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Libraries
import cv2
import numpy as np
import torch
import torchvision.transforms as T
# from PIL import Image # REMOVED - No direct usage found in this file

# --- Optional Dependency Imports ---
try:
    import mediapipe as mp
    _mediapipe_available = True
    logger = logging.getLogger(__name__)  # Setup logger early if MP available
except ImportError:
    mp = None
    _mediapipe_available = False
    # Setup logger here if MP failed, to ensure logging works regardless
    if 'logger' not in locals():
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger(__name__)
    logger.warning(
        "Mediapipe not found. Effects requiring it will be disabled.")

try:
    import colour
    _colour_available = True
except ImportError:
    colour = None
    _colour_available = False
    if 'logger' not in locals():
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger(__name__)
    logger.warning("colour-science not found. LUT effect will be disabled.")

try:
    # Check for original segment-anything
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator as BaseSamAutomaticMaskGenerator
    _sam_available = True
except ImportError:
    # Define base class as None if import fails
    BaseSamAutomaticMaskGenerator = None
    sam_model_registry = {}
    _sam_available = False
    if 'logger' not in locals():
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger(__name__)
    logger.warning(
        "segment_anything library not found. SAM v1 effect will be disabled.")

try:
    # Check for ultralytics (which includes their SAM implementation)
    from ultralytics import SAM as UltralyticsSAM
    _sam2_available = True
except ImportError:
    UltralyticsSAM = None  # Define alias as None if import fails
    _sam2_available = False
    if 'logger' not in locals():
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger(__name__)
    logger.warning(
        "ultralytics[sam] not found or SAM class missing. SAM v2 effect will be disabled.")


# --- Logging Setup (ensure it's set up regardless of imports) ---
if 'logger' not in locals():
    # Basic config if no warnings triggered it yet
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

# Ensure logs directory exists and add file handler
os.makedirs("logs", exist_ok=True)
# Unique log file name
log_file = f"logs/video_effects_final5_{time.strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")  # More detailed format
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Add stream handler if not already present (e.g., if basicConfig wasn't called)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)  # Use same format for console
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)  # Ensure level is set


# --- Constants ---
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
DEFAULT_FPS = 30.0
CONFIG_FILE = "config.json"
GOLD_TINT_COLOR = (30, 165, 210)  # BGR format for OpenCV - Orange-Gold
TRAIL_START_ALPHA, TRAIL_END_ALPHA = 1.0, 0.2
TRAIL_RADIUS = 2
TINT_STRENGTH = 0.15
SEGMENTATION_THRESHOLD = 0.5  # Confidence threshold for SAM masks
MASK_BLUR_KERNEL = (21, 21)
# Default green screen for RVM if no bg image
RVM_DEFAULT_BG_COLOR = (0, 120, 0)
# Opacity for SAM mask overlays (Yellow = 0, 255, 255 BGR)
SAM_MASK_ALPHA = 0.4
SAM_MASK_COLOR = (0, 255, 255)  # Yellow BGR

# List of supported SAM v2 model filenames (used for validation)
SUPPORTED_SAM2_MODELS = [
    'sam_h.pt', 'sam_l.pt', 'sam_b.pt', 'mobile_sam.pt',  # Older Ultralytics SAM
    'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt',  # SAM 2 base models
    'sam2.1_t.pt', 'sam2.1_s.pt', 'sam2.1_b.pt', 'sam2.1_l.pt'  # SAM 2.1 models
]


# --- Utility Functions ---
# ========================================================================
#                       Effect Base Classes (from video_effects.py)
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
#               Enhanced Intra-Clip Effects (from video_effects.py)
# ========================================================================
@dataclass
class BrightnessContrast(IntraClipEffect):
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    per_channel: bool = False  # Not yet controllable by UI
    fade_in: bool = False      # Not yet controllable by UI

    def _validate_params(self):
        self.brightness = np.clip(self.brightness, -1.0, 1.0)
        self.contrast = np.clip(self.contrast, 0.0, 3.0)
        self.gamma = max(0.1, self.gamma)

    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float, runner: Optional[Any] = None, **kwargs) -> np.ndarray:
        if runner and hasattr(runner, 'bc_params'):
            self.brightness = runner.bc_params.get('brightness', self.brightness)
            self.contrast = runner.bc_params.get('contrast', self.contrast)
            self.gamma = runner.bc_params.get('gamma', self.gamma)
            # self.per_channel = runner.bc_params.get('per_channel', self.per_channel) # Deferred
            # self.fade_in = runner.bc_params.get('fade_in', self.fade_in) # Deferred
            self._validate_params() # Re-validate if params changed

        frame = validate_frame(frame)
        if abs(self.contrast - 1.0) < 1e-3 and abs(self.brightness) < 1e-3 and abs(self.gamma - 1.0) < 1e-3 and not self.fade_in:
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
    fade_in: bool = False      # Not yet controllable by UI

    def _validate_params(self):
        self.scale = max(0.0, self.scale)
        self.vibrance = np.clip(self.vibrance, 0.0, 1.0)

    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float, runner: Optional[Any] = None, **kwargs) -> np.ndarray:
        if runner and hasattr(runner, 'saturation_params'):
            self.scale = runner.saturation_params.get('scale', self.scale)
            self.vibrance = runner.saturation_params.get('vibrance', self.vibrance)
            # self.fade_in = runner.saturation_params.get('fade_in', self.fade_in) # Deferred
            self._validate_params()

        frame = validate_frame(frame)
        if abs(self.scale - 1.0) < 1e-3 and abs(self.vibrance) < 1e-3 and not self.fade_in:
            return frame
        try:
            factor = (frame_time / max(clip_duration, 1e-6)) if self.fade_in and clip_duration > 0 else 1.0
            curr_scale = 1.0 + (self.scale - 1.0) * factor
            curr_vibrance = self.vibrance * factor

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            s = hsv[:, :, 1]
            if curr_vibrance > 0:
                s_max = np.max(s)
                if s_max > 1e-6: # Avoid division by zero for fully desaturated images
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
    shape: str = 'circular'  # Not yet controllable by UI
    color: Tuple[int, int, int] = (0, 0, 0)  # BGR, Not yet controllable by UI
    fade_in: bool = False      # Not yet controllable by UI

    def _validate_params(self):
        self.strength = np.clip(self.strength, 0.0, 1.0)
        self.radius = np.clip(self.radius, 0.0, 1.0)
        self.falloff = max(0.01, self.falloff)
        self.shape = self.shape if self.shape in ['circular', 'elliptical', 'rectangular'] else 'circular'

    def _create_mask(self, height: int, width: int, strength: float, radius: float, falloff: float, shape: str) -> np.ndarray:
        # Parameters are passed explicitly to use potentially updated values
        cx, cy = width / 2, height / 2
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        if shape == 'circular':
            max_dim = max(width, height)
            d = np.sqrt((x - cx)**2 + (y - cy)**2) / max(max_dim / 2, 1e-6)
        elif shape == 'elliptical':
            d = np.sqrt(((x - cx)/(width/2 + 1e-6))**2 + ((y - cy)/(height/2 + 1e-6))**2)
        else:  # rectangular
            d = np.maximum(np.abs(x - cx) / (width / 2 + 1e-6), np.abs(y - cy) / (height / 2 + 1e-6))

        mask = 1.0 - strength * (1.0 / (1.0 + np.exp(-(d - radius) / falloff)))
        return mask[:, :, np.newaxis].astype(np.float32)

    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float, runner: Optional[Any] = None, **kwargs) -> np.ndarray:
        if runner and hasattr(runner, 'vignette_params'):
            self.strength = runner.vignette_params.get('strength', self.strength)
            self.radius = runner.vignette_params.get('radius', self.radius)
            self.falloff = runner.vignette_params.get('falloff', self.falloff)
            # self.shape = runner.vignette_params.get('shape', self.shape) # Deferred
            # self.color = runner.vignette_params.get('color', self.color) # Deferred
            # self.fade_in = runner.vignette_params.get('fade_in', self.fade_in) # Deferred
            self._validate_params()

        frame = validate_frame(frame)
        if abs(self.strength) < 1e-3 and not self.fade_in:
            return frame
        try:
            factor = (frame_time / max(clip_duration, 1e-6)) if self.fade_in and clip_duration > 0 else 1.0
            current_strength = self.strength * factor # Apply fade-in to strength

            mask = self._create_mask(frame.shape[0], frame.shape[1], current_strength, self.radius, self.falloff, self.shape)
            vignette_layer = np.full_like(frame, self.color, dtype=np.float32) # Use instance color
            return np.clip(frame.astype(np.float32) * mask + vignette_layer * (1.0 - mask), 0, 255).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Vignette failed: {e}")
            return frame

@dataclass
class Blur(IntraClipEffect):
    kernel_size: int = 5
    sigma: float = 0.0         # Not yet controllable by UI
    fade_in: bool = False      # Not yet controllable by UI

    def _validate_params(self):
        self.kernel_size = max(3, self.kernel_size if self.kernel_size % 2 != 0 else self.kernel_size + 1)
        self.sigma = max(0.0, self.sigma)

    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float, runner: Optional[Any] = None, **kwargs) -> np.ndarray:
        if runner and hasattr(runner, 'blur_params'):
            self.kernel_size = runner.blur_params.get('kernel_size', self.kernel_size)
            # Ensure kernel_size is odd after updating from trackbar
            self.kernel_size = max(3, self.kernel_size if self.kernel_size % 2 != 0 else self.kernel_size -1 if self.kernel_size > 3 else self.kernel_size + 1)
            # self.sigma = runner.blur_params.get('sigma', self.sigma) # Deferred
            # self.fade_in = runner.blur_params.get('fade_in', self.fade_in) # Deferred
            self._validate_params()

        frame = validate_frame(frame)
        if self.kernel_size < 3 and abs(self.sigma) < 1e-3 and not self.fade_in:
             return frame
        try:
            factor = (frame_time / max(clip_duration, 1e-6)) if self.fade_in and clip_duration > 0 else 1.0
            # Apply fade-in to kernel size. Start from 1 (no blur) up to kernel_size.
            # Ensure kernel_size is at least 3 and odd.
            current_kernel_float = 1 + (self.kernel_size - 1) * factor
            current_kernel = int(current_kernel_float)
            current_kernel = max(3, current_kernel if current_kernel % 2 != 0 else current_kernel + 1)

            if current_kernel < 3 : # Effectively no blur
                return frame
            return cv2.GaussianBlur(frame, (current_kernel, current_kernel), self.sigma)
        except Exception as e:
            logger.warning(f"Blur failed: {e}")
            return frame

# ========================================================================
#                       Effect Base Classes (from video_effects.py)
# ========================================================================
# (Effect, IntraClipEffect, TransitionEffect are already added)

# Add VideoEffect base class from video_effectsfinale.py
class VideoEffect(abc.ABC):
    """Base class for algorithmic effects."""
    @abc.abstractmethod
    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def reset(self) -> None:
        pass

class MotionBlurEffect(VideoEffect): # Inherits from VideoEffect
    """Applies simple horizontal motion blur using OpenCV."""
    def __init__(self, kernel_size: int = 19): # Default kernel_size from video_effectsfinale.py
        # Ensure kernel_size is odd and at least 3
        self.kernel_size = max(3, kernel_size if kernel_size % 2 != 0 else kernel_size + 1)

    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        # Create a horizontal motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        kernel[self.kernel_size // 2, :] = 1.0 / self.kernel_size
        try:
            return cv2.filter2D(frame, -1, kernel)
        except cv2.error as e:
            logger.warning(f"MotionBlurEffect (kernel) failed: {e}")
            return frame

    def reset(self) -> None:
        pass

class ChromaticAberrationEffect(VideoEffect): # Inherits from VideoEffect
    """
    Enhanced radial chromatic aberration with amplified fringing for visibility:
    - Larger radial scaling with saturation-based strength.
    - Exaggerated edge effects with a higher boost.
    - Visual debugging with colored borders to highlight channel shifts (optional).
    """
    def __init__(self,
                 base_strength: float = 0.05, # Renamed from strength to base_strength for clarity
                 center: Optional[Tuple[float, float]] = None, # Normalized (0-1) center point (x,y)
                 non_linear_exponent: float = 3.0,
                 edge_boost: float = 4.0,
                 debug_borders: bool = False): # Added debug_borders
        self.base_strength = max(0.0, base_strength)
        self.center = center
        self.non_linear_exponent = max(1.0, non_linear_exponent)
        self.edge_boost = max(1.0, edge_boost)
        self.debug_borders = debug_borders

    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        h, w = frame.shape[:2]

        try:
            cx, cy = (w / 2.0, h / 2.0) if self.center is None else (self.center[0] * w, self.center[1] * h)
            cx, cy = np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)

            map_x_coords = np.tile(np.arange(w, dtype=np.float32), (h, 1)) # Renamed
            map_y_coords = np.repeat(np.arange(h, dtype=np.float32).reshape(-1, 1), w, axis=1) # Renamed

            delta_x = map_x_coords - cx
            delta_y = map_y_coords - cy
            r_dist = np.sqrt(delta_x**2 + delta_y**2) # Renamed
            
            # Max radius from center to any corner
            corners = np.array([[0,0], [w,0], [0,h], [w,h]])
            r_max = np.max(np.sqrt(np.sum((corners - np.array([cx, cy]))**2, axis=1)))
            r_max = max(r_max, 1e-6) # Avoid division by zero

            r_normalized = r_dist / r_max
            non_linear_factor = self.edge_boost * (r_normalized ** self.non_linear_exponent)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
            saturation = hsv[..., 1]
            k_map = self.base_strength * (1.5 + saturation) 

            scale_r = 1.0 + k_map * non_linear_factor
            scale_b = 1.0 - k_map * non_linear_factor

            map_x_r = np.clip(cx + delta_x * scale_r, 0, w - 1)
            map_y_r = np.clip(cy + delta_y * scale_r, 0, h - 1)
            map_x_b = np.clip(cx + delta_x * scale_b, 0, w - 1)
            map_y_b = np.clip(cy + delta_y * scale_b, 0, h - 1)

            b, g, r_channel = cv2.split(frame) # Renamed r to r_channel
            r_shifted = cv2.remap(r_channel, map_x_r, map_y_r, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            b_shifted = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

            output = cv2.merge((b_shifted, g, r_shifted))

            if self.debug_borders:
                border_size = 5
                output[:border_size, :] = (0, 0, 255)  # Top red
                output[-border_size:, :] = (0, 0, 255) # Bottom red
                output[:, :border_size] = (255, 0, 0)  # Left blue
                output[:, -border_size:] = (255, 0, 0) # Right blue
            # cv2.putText(output, "ChromaAberration Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) # Optional
            return output
        except Exception as e:
            logger.error(f"ChromaticAberrationEffect failed: {e}", exc_info=True)
            return frame

    def reset(self) -> None:
        pass

# ========================================================================
#               Transition Effects (from video_effects.py)
# ========================================================================
@dataclass
class Crossfade(TransitionEffect):
    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
        frame_a = validate_frame(frame_a)
        # Determine target shape from frame_b if valid, else from frame_a
        ts = frame_b.shape if (frame_b is not None and isinstance(frame_b, np.ndarray) and frame_b.size > 0) else frame_a.shape
        frame_b = validate_frame(frame_b, default_shape=ts) # Ensure frame_b is valid or a black frame of target size

        progress = np.clip(progress, 0.0, 1.0)
        try:
            # Resize frames to a consistent shape before blending
            # This uses the resize_frame from video_effectsfinale5.py
            a_r = resize_frame(frame_a, (ts[0], ts[1]))
            b_r = resize_frame(frame_b, (ts[0], ts[1]))
            return cv2.addWeighted(a_r, 1.0 - progress, b_r, progress, 0.0)
        except Exception as e:
            logger.warning(f"Crossfade failed: {e}")
            # Fallback logic: return one of the frames based on progress
            return frame_b if progress > 0.5 else frame_a

@dataclass
class Slide(TransitionEffect):
    direction: str = 'left' # 'left', 'right', 'up', 'down'
    def _validate_params(self):
        valid_directions = {'left', 'right', 'up', 'down'}
        if self.direction not in valid_directions:
            logger.warning(f"Invalid direction: {self.direction}. Defaulting to 'left'.")
            self.direction = 'left'

    def apply(self, frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
        frame_a = validate_frame(frame_a)
        ts = frame_b.shape if (frame_b is not None and isinstance(frame_b, np.ndarray) and frame_b.size > 0) else frame_a.shape
        frame_b = validate_frame(frame_b, default_shape=ts)

        progress = np.clip(progress, 0.0, 1.0)
        try:
            a = resize_frame(frame_a, (ts[0], ts[1]))
            b = resize_frame(frame_b, (ts[0], ts[1]))
            h, w = ts[:2]
            output_frame = a.copy() # Start with frame_a

            if self.direction == 'left':
                offset = int(w * progress)
                if offset > 0: output_frame[:, :w - offset] = a[:, offset:]; output_frame[:, w - offset:] = b[:, :offset]
            elif self.direction == 'right':
                offset = int(w * progress)
                if offset > 0: output_frame[:, offset:] = a[:, :w - offset]; output_frame[:, :offset] = b[:, w - offset:]
            elif self.direction == 'up':
                offset = int(h * progress)
                if offset > 0: output_frame[:h - offset, :] = a[offset:, :]; output_frame[h - offset:, :] = b[:offset, :]
            elif self.direction == 'down':
                offset = int(h * progress)
                if offset > 0: output_frame[offset:, :] = a[:h - offset, :]; output_frame[:offset, :] = b[h - offset:, :]
            return output_frame
        except Exception as e:
            logger.warning(f"Slide failed: {e}")
            return frame_b if progress > 0.5 else frame_a

# ========================================================================
#               Additional Advanced Effects (from video_effectsfinale.py)
# ========================================================================
class AdvancedColorGradeEffect(VideoEffect): # Inherits from newly added VideoEffect
    """
    A state-of-the-art color grading effect for cinematic, professional music video-grade visuals:
    - Utilizes CAM16 for perceptually accurate, viewing-condition-aware color adjustments.
    - Dynamically adapts contrast, brightness, and chroma based on scene content.
    - Applies non-linear chromatic enhancement for a unique 'glow' effect on vivid colors.
    - Adds split toning with warm highlights and cool shadows for a stylized look.
    - Incorporates a perceptual vignette to focus attention on the subject.
    - Introduces randomized micro-adjustments for an organic, dynamic appearance.
    """
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

        if not _colour_available: # _colour_available is already defined in video_effectsfinale5.py
            logger.warning("colour-science not available; falling back to basic color grading for AdvancedColorGrade.")
        else:
            # CAM16 viewing conditions for perceptual accuracy.
            # Ensure colour.appearance and colour.models are accessible
            try:
                self.viewing_conditions = colour.appearance.VIEWING_CONDITIONS_CAM16["Average"]
                self.L_A = 200.0  # Adapting field luminance (cd/m^2).
                self.Y_b = 20.0   # Background relative luminance.
            except Exception as e:
                logger.error(f"Failed to initialize CAM16 settings for AdvancedColorGrade: {e}")
                # Potentially disable colour-science dependent parts or set a flag
                self._colour_science_ready = False # Custom flag
            else:
                self._colour_science_ready = True


    def dynamic_scene_analysis(self, rgb: np.ndarray) -> Tuple[float, float]:
        """
        Analyze the scene to determine average luminance and chromatic content.
        """
        # Convert linear RGB to XYZ.
        XYZ = colour.RGB_to_XYZ(
            rgb,
            colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
            colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
            colour.models.RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ
        )
        # Convert from XYZ to Oklab.
        oklab = colour.XYZ_to_Oklab(XYZ)
        L = oklab[..., 0]  # Lightness.
        a = oklab[..., 1]
        b = oklab[..., 2]
        chroma = np.sqrt(a**2 + b**2)

        avg_luminance = np.mean(L)
        avg_chroma = np.mean(chroma)
        return avg_luminance, avg_chroma

    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        h, w = frame.shape[:2]

        try:
            if _colour_available and hasattr(self, '_colour_science_ready') and self._colour_science_ready:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                linear_rgb = colour.models.RGB_COLOURSPACE_sRGB.cctf_decoding(rgb)
                avg_luminance, avg_chroma = self.dynamic_scene_analysis(linear_rgb)
                XYZ = colour.RGB_to_XYZ(
                    linear_rgb,
                    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
                    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
                    colour.models.RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ
                )
                cam16 = colour.appearance.XYZ_to_CAM16(XYZ, self.viewing_conditions, L_A=self.L_A, Y_b=self.Y_b)
                J = cam16.J / 100.0
                C = cam16.C
                h_val = cam16.h
                contrast = self.base_contrast * (1.0 + 0.2 * (1.0 - avg_luminance))
                brightness_adj = self.base_brightness * (1.0 + 0.3 * (avg_luminance - 0.5)) # Renamed to avoid conflict
                J = np.clip(contrast * (J - 0.5) + 0.5 + brightness_adj, 0.0, 1.0) * 100.0
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
                x_coords = np.tile(np.linspace(-1, 1, w), (h, 1)) # Renamed to avoid conflict
                y_coords = np.repeat(np.linspace(-1, 1, h).reshape(-1, 1), w, axis=1) # Renamed to avoid conflict
                r_dist = np.sqrt(x_coords**2 + y_coords**2) # Renamed to avoid conflict
                vignette_mask = 1.0 - self.vignette_strength * r_dist # Renamed to avoid conflict
                J = J * vignette_mask
                cam16_adjusted = colour.appearance.CAM_Specification_CAM16(J=J, C=C, h=h_adjusted)
                XYZ_adjusted = colour.appearance.CAM16_to_XYZ(cam16_adjusted, self.viewing_conditions, L_A=self.L_A, Y_b=self.Y_b)
                rgb_adjusted = colour.XYZ_to_RGB(
                    XYZ_adjusted,
                    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
                    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
                    colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB
                )
                rgb_final = colour.models.RGB_COLOURSPACE_sRGB.cctf_encoding(rgb_adjusted)
                rgb_final = np.clip(rgb_final, 0.0, 1.0)
                output = (rgb_final * 255.0).astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            else:
                output = frame.astype(np.float32)
                output = output * self.base_contrast + self.base_brightness * 255.0
                output = np.clip(output, 0, 255).astype(np.uint8)
            # cv2.putText(output, "AdvGrade Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2) # Optional: for debugging
            return output
        except Exception as e:
            logger.error(f"AdvancedColorGradeEffect failed: {e}", exc_info=True)
            return frame

    def reset(self) -> None:
        pass

def validate_frame(frame: np.ndarray) -> np.ndarray:
    """Ensures the frame is valid (not None, has size) and is uint8 BGR."""
    if frame is None or frame.size == 0:
        h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
        logger.warning(
            f"Invalid frame detected (None or empty), returning default {w}x{h} black frame")
        return np.zeros((h, w, 3), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning(
            f"Invalid frame format ({frame.shape}, ndim={frame.ndim}), attempting conversion")
        try:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                logger.error(
                    f"Unsupported frame shape {frame.shape}, returning black frame")
                h = frame.shape[0] if frame.ndim >= 1 else DEFAULT_HEIGHT
                w = frame.shape[1] if frame.ndim >= 2 else DEFAULT_WIDTH
                return np.zeros((h, w, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(
                f"Error during frame conversion: {e}. Returning black frame.")
            h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
            return np.zeros((h, w, 3), dtype=np.uint8)
    if frame.dtype != np.uint8:
        try:
            logger.warning(
                f"Frame received with unexpected dtype: {frame.dtype}. Converting to uint8.")
            frame = frame.astype(np.uint8)
        except ValueError as e:
            logger.error(
                f"Failed to convert frame dtype {frame.dtype} to uint8: {e}. Returning black frame.")
            h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
            return np.zeros((h, w, 3), dtype=np.uint8)
    return frame


def resize_frame(frame: np.ndarray, target_dims: Tuple[int, int]) -> np.ndarray:
    """Resizes a frame to target dimensions (height, width)."""
    if frame is None or frame.size == 0:
        logger.error("Cannot resize empty frame")
        return np.zeros((target_dims[0], target_dims[1], 3), dtype=np.uint8)
    return cv2.resize(
        frame, (target_dims[1], target_dims[0]
                ), interpolation=cv2.INTER_LANCZOS4
    )


# --- MPS Compatible SAM Generator Class ---
if BaseSamAutomaticMaskGenerator is not None:
    class MPSCompatibleSamAutomaticMaskGenerator(BaseSamAutomaticMaskGenerator):
        """
        Custom SamAutomaticMaskGenerator compatible with MPS backend for SAM v1.
        Overrides _process_batch to ensure points are float32 for MPS compatibility.
        """

        def _process_batch(self, points, cropped_im_size, crop_box, orig_size):
            try:
                transformed_points = points.astype(np.float32)
                return super()._process_batch(transformed_points, cropped_im_size, crop_box, orig_size)
            except Exception as e:
                logger.error(
                    f"Error in MPSCompatibleSamAutomaticMaskGenerator._process_batch: {e}", exc_info=True)
                return None  # Return None or appropriate error indicator
else:
    class MPSCompatibleSamAutomaticMaskGenerator:  # Dummy class if base unavailable
        def __init__(self, *args, **kwargs):
            logger.error(
                "Attempted to use MPSCompatibleSamAutomaticMaskGenerator, but base class is unavailable.")

        def generate(self, *args, **kwargs): return []


# --- Model Loading Functions ---
def load_rvm_model(
    device: str, model_name: str = "resnet50", pretrained: bool = True
) -> Optional[torch.nn.Module]:
    """Loads the Robust Video Matting model."""
    if not torch.__version__:
        logger.warning("PyTorch unavailable, cannot load RVM model")
        return None
    try:
        logger.info(
            f"Loading RVM model '{model_name}' (pretrained={pretrained})")
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        model = torch.hub.load("PeterL1n/RobustVideoMatting",
                               model_name, pretrained=pretrained, trust_repo=True)
        target_device = "cpu"
        if (device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            target_device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            target_device = "cuda"
        elif device != "cpu":
            logger.warning(
                f"Requested device {device} unavailable or unsupported, falling back to CPU for RVM")
        model = model.to(device=target_device).eval()
        logger.info(f"RVM model '{model_name}' loaded to {target_device}")
        return model
    except Exception as e:
        logger.error(
            f"Failed to load RVM model '{model_name}': {e}", exc_info=True)
        return None


def load_sam_mask_generator(
    device: str, model_type: str = "vit_h", checkpoint_path: Optional[str] = None
) -> Optional[Any]:
    """Loads the Segment Anything (SAM v1) Automatic Mask Generator using the MPSCompatible class."""
    if not _sam_available:
        logger.warning(
            "segment_anything library unavailable, cannot load SAM v1")
        return None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(
            f"SAM v1 checkpoint path invalid or missing: {checkpoint_path}")
        return None
    if not checkpoint_path.lower().endswith(".pth"):
        logger.warning(
            f"Expected SAM v1 checkpoint .pth, got: {checkpoint_path}. Loading might fail.")

    try:
        ckpt_name = os.path.basename(checkpoint_path).lower()
        inferred_type = model_type
        if "vit_h" in ckpt_name:
            inferred_type = "vit_h"
        elif "vit_l" in ckpt_name:
            inferred_type = "vit_l"
        elif "vit_b" in ckpt_name:
            inferred_type = "vit_b"
        if inferred_type != model_type:
            logger.info(
                f"Inferred SAM v1 type '{inferred_type}' from filename, overriding '{model_type}'.")
            model_type = inferred_type
        if model_type not in sam_model_registry:
            logger.error(
                f"Invalid SAM v1 type '{model_type}'. Available: {list(sam_model_registry.keys())}")
            return None

        logger.info(
            f"Loading SAM v1 model '{model_type}' from {checkpoint_path}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        target_device = "cpu"
        mps_available = hasattr(
            torch.backends, "mps") and torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()
        if device == "mps" and mps_available:
            target_device = "mps"
        elif device == "cuda" and cuda_available:
            target_device = "cuda"
        elif device != "cpu":
            logger.warning(
                f"Requested device '{device}' unavailable or unsupported, falling back to CPU for SAM v1")

        sam.to(device=target_device)
        logger.info(f"SAM v1 model moved to {target_device}")

        GeneratorClass = MPSCompatibleSamAutomaticMaskGenerator if target_device == "mps" else BaseSamAutomaticMaskGenerator
        if GeneratorClass == MPSCompatibleSamAutomaticMaskGenerator:
            logger.info(
                "Using MPSCompatibleSamAutomaticMaskGenerator for SAM v1.")
        else:
            logger.info(
                "Using standard BaseSamAutomaticMaskGenerator for SAM v1.")

        generator = GeneratorClass(model=sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95,
                                   crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100)
        logger.info(
            f"SAM v1 generator initialized on {target_device} with custom settings using {GeneratorClass.__name__}")
        return generator
    except KeyError:
        logger.error(
            f"Invalid SAM v1 model type '{model_type}'. Available: {list(sam_model_registry.keys())}")
        return None
    except Exception as e:
        logger.error(f"Failed to load SAM v1 model: {e}", exc_info=True)
        return None


def load_sam2_video_predictor(
    device: str, checkpoint_path: Optional[str] = None
) -> Optional[Any]:
    """Loads the Segment Anything (SAM v2 / ultralytics SAM) Predictor with filename validation."""
    if not _sam2_available or UltralyticsSAM is None:
        logger.warning(
            "ultralytics library or SAM class unavailable, cannot load SAM 2")
        return None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(
            f"SAM 2 checkpoint path invalid or missing: {checkpoint_path}")
        return None
    if not checkpoint_path.lower().endswith(".pt"):
        logger.warning(
            f"Expected SAM 2 checkpoint .pt, got: {checkpoint_path}. Loading might fail.")

    checkpoint_name = os.path.basename(checkpoint_path)
    if checkpoint_name not in SUPPORTED_SAM2_MODELS:
        logger.error(
            f"Checkpoint filename '{checkpoint_name}' is not recognized as a supported Ultralytics SAM model.")
        logger.error(
            f"Please use one of the supported names: {SUPPORTED_SAM2_MODELS}")
        logger.error(
            "If you downloaded the file, ensure it's renamed correctly (e.g., sam2.1_hiera_tiny.pt -> sam2.1_t.pt).")
        return None

    try:
        logger.info(
            f"Loading SAM 2 (Ultralytics) model '{checkpoint_name}' from {checkpoint_path}")
        # Initialize with the validated path/filename
        model = UltralyticsSAM(checkpoint_path)

        target_device = "cpu"  # Determine target device for logging
        if (device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            target_device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            target_device = "cuda"
        elif device != "cpu":
            logger.warning(
                f"Requested device '{device}' unavailable or unsupported, SAM 2 will likely use CPU for prediction")

        logger.info(
            f"SAM 2 (Ultralytics) model '{checkpoint_name}' initialized. Target device for prediction: {target_device}")
        return model
    except Exception as e:
        if ".pth" in str(e).lower():
            logger.error(
                f"Failed to load SAM 2 model from {checkpoint_path}: Looks like a SAM v1 (.pth) file. Use --sam-checkpoint-path.", exc_info=False)
        else:
            logger.error(
                f"Failed to load SAM 2 model '{checkpoint_name}': {e}", exc_info=True)
        return None


# --- Effect Classes ---
# Note: The new effect classes (BrightnessContrast, etc.) are defined above.
# The FrameEffectWrapper might need adjustments later to properly use these class-based effects
# or a new registry similar to EFFECT_REGISTRY in video_effects.py might be introduced.

class FrameEffectWrapper:
    """Wraps an effect function with enabling/disabling and dependency checks."""

    def __init__(
            self, func, name: str, requires_mediapipe: bool = False, requires_torch: bool = False,
            requires_sam: bool = False, requires_sam2: bool = False, requires_colour: bool = False):
        self.func = func
        self.name = name
        self.requires_mediapipe = requires_mediapipe
        self.requires_torch = requires_torch
        self.requires_sam = requires_sam
        self.requires_sam2 = requires_sam2
        self.requires_colour = requires_colour
        self.enabled = True
        self._suppress_warn = False

    # *** CORRECTED check_dependencies method with proper indentation ***
    def check_dependencies(self) -> bool:
        """Checks if required libraries are available."""
        # Mediapipe Check
        if self.requires_mediapipe and not _mediapipe_available:
            if not self._suppress_warn:
                logger.warning(
                    f"Mediapipe required but unavailable for '{self.name}'")
            self._suppress_warn = True
            return False

        # Torch Check
        if self.requires_torch and not torch.__version__:
            if not self._suppress_warn:
                logger.warning(
                    f"Torch required but unavailable for '{self.name}'")
            self._suppress_warn = True
            return False

        # SAM v1 Check
        if self.requires_sam and not _sam_available:
            if not self._suppress_warn:
                logger.warning(
                    f"segment_anything required but unavailable for '{self.name}'")
            self._suppress_warn = True
            return False

        # SAM v2 Check
        if self.requires_sam2 and not _sam2_available:
            if not self._suppress_warn:
                logger.warning(
                    f"ultralytics[sam] required but unavailable for '{self.name}'")
            self._suppress_warn = True
            return False

        # Colour Check
        if self.requires_colour and not _colour_available:
            if not self._suppress_warn:
                logger.warning(
                    f"colour-science required but unavailable for '{self.name}'")
            self._suppress_warn = True
            return False

        # If all checks passed:
        self._suppress_warn = False
        return True

    def __call__(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Applies the effect if enabled and dependencies met."""
        if not self.enabled:
            return frame
        if not self.check_dependencies():
            cv2.putText(frame, f"{self.name.upper()} UNAVAILABLE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, )
            return frame
        try:
            # Safely get original frame
            original_frame = kwargs.get('original_frame', frame.copy())
            sig = inspect.signature(self.func)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            call_args = {'frame': frame}
            if 'original_frame' in sig.parameters or accepts_kwargs:
                call_args['original_frame'] = original_frame
            if accepts_kwargs:
                call_args.update(kwargs)
            else:
                for k, v in kwargs.items():
                    if k in sig.parameters and k not in ['frame', 'original_frame']:
                        call_args[k] = v
            return self.func(**call_args)  # Call with prepared arguments
        except Exception as e:
            logger.error(
                f"Error during effect '{self.name}': {e}", exc_info=True)
            # Use original if available
            error_frame = kwargs.get("original_frame", frame).copy()
            cv2.putText(error_frame, f"{self.name.upper()} ERROR",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, )
            return error_frame

    def reset(self):
        self.enabled = True
        self._suppress_warn = False
        if hasattr(self.func, "reset") and callable(self.func.reset):
            try:
                self.func.reset()
                logger.debug(f"Called reset() on function for '{self.name}'")
            except Exception as e:
                logger.warning(
                    f"Error calling reset() on function for '{self.name}': {e}")


class LUTColorGradeEffect:
    """Applies Look-Up Table (LUT) color grading."""

    def __init__(self, lut_files: List[Tuple[str, str]]):
        self.lut_files = lut_files
        self.current_idx = 0
        self.luts: List[Tuple[str, Any]] = []
        self.lut_intensity = 1.0
        self._load_luts()

    def _load_luts(self):
        if not _colour_available:
            logger.warning("Cannot load LUTs, colour-science unavailable.")
            return
        self.luts = []
        for lut_name, lut_path in self.lut_files:
            try:
                lut = colour.io.read_LUT(lut_path)
                if isinstance(lut, (colour.LUT3D, colour.LUT1D)):
                    self.luts.append((lut_name, lut))
                    logger.info(f"Loaded LUT: {lut_name} from {lut_path}")
                else:
                    logger.warning(
                        f"Unsupported LUT format/type for {lut_name} at {lut_path}: {type(lut)}")
            except Exception as e:
                logger.error(
                    f"Failed to load LUT {lut_name} from {lut_path}: {e}")
        if not self.luts:
            logger.warning("No valid LUTs were loaded.")
        else:
            logger.info(f"Successfully loaded {len(self.luts)} LUTs.")
        self.current_idx = 0

    def process(self, frame: np.ndarray, runner: Any = None, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame
        if not self.luts:
            cv2.putText(frame, "No LUTs Loaded", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return original_frame_fallback
        try:
            if not (0 <= self.current_idx < len(self.luts)):
                logger.warning(
                    f"Invalid LUT index {self.current_idx}, resetting to 0.")
                self.current_idx = 0
            if not self.luts:
                return original_frame_fallback
            lut_name, lut = self.luts[self.current_idx]
            frame_rgb = (cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            frame_transformed = lut.apply(frame_rgb)
            frame_out = (frame_transformed * 255.0).clip(0,
                                                         255).astype(np.uint8)
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
            intensity = runner.lut_intensity if runner else self.lut_intensity
            if intensity < 1.0:
                frame_out = cv2.addWeighted(
                    original_frame_fallback, 1.0 - intensity, frame_out, intensity, 0.0)
            display_name = lut_name[:25] + \
                ('...' if len(lut_name) > 25 else '')
            cv2.putText(frame_out, f"LUT: {display_name}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, )
        except Exception as e:
            logger.warning(
                f"LUT application failed for '{self.luts[self.current_idx][0]}': {e}", exc_info=True, )
            error_display_frame = original_frame_fallback.copy()
            cv2.putText(error_display_frame, "LUT Error", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_display_frame
        return frame_out

    def reset(self): self.current_idx = 0; self.lut_intensity = 1.0; logger.info(
        "LUT effect reset.")

    def cycle_lut(self):
        if not self.luts:
            logger.warning("Cannot cycle LUTs, none loaded.")
            return
        self.current_idx = (self.current_idx + 1) % len(self.luts)
        logger.info(
            f"Switched LUT to index {self.current_idx}: {self.luts[self.current_idx][0]}")


# --- Main Effect Runner Class ---
class PixelSensingEffectRunner:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._apply_config()  # Apply initial config

        self.face_mesh = None
        self.pose = None
        self.hands = None
        if _mediapipe_available:
            self._initialize_mediapipe()
        else:
            logger.warning("Mediapipe unavailable.")

        self.cap = None
        self.out = None
        self.frame_width = self.desired_width
        self.frame_height = self.desired_height
        self.fps = self.desired_fps

        self.prev_gray = None
        self.prev_landmarks_flow = None
        self.hue_offset = 0
        self.frame_buffer = deque(maxlen=10)
        self.frame_times = deque(maxlen=100)
        self.effect_times = {}
        self.error_count = 0
        self.current_effect = "sam_segmentation"  # Default to SAM
        self.use_sam2_runtime = False  # Updated after config/loading
        self.brightness_factor = 0.0
        self.contrast_factor = 1.0
        self.window_name = "PixelSense FX"

        # Parameters for dataclass effects
        self.bc_params = {'brightness': 0.0, 'contrast': 1.0, 'gamma': 1.0, 'per_channel': False, 'fade_in': False}
        self.saturation_params = {'scale': 1.0, 'vibrance': 0.0, 'fade_in': False}
        self.vignette_params = {'strength': 0.3, 'radius': 0.7, 'falloff': 0.2, 'shape': 'circular', 'color': (0,0,0), 'fade_in': False}
        self.blur_params = {'kernel_size': 5, 'sigma': 0.0, 'fade_in': False}
        # For LUT intensity, already have self.lut_intensity

        self.goldenaura_variant = 0
        self.goldenaura_variant_names = ["Original", "Enhanced"]
        self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
        self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
        self.face_history = deque(maxlen=self.trail_length)
        self.pose_history = deque(maxlen=self.trail_length)
        self.lightning_style_index = 0
        self.lightning_style_names = ["Random", "Movement", "Branching"]
        self.lightning_styles = [self._apply_lightning_style_random,
                                 self._apply_lightning_style_movement, self._apply_lightning_style_branching]
        self.gesture_state = {"last_gesture": None,
                              "last_time": 0.0, "gesture_count": 0, }

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
        self.sam_model_v1 = None
        self.sam_model_v2 = None
        self.sam_masks_cache = None

        self.LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), )
        self.LK_PARAMS_GOLD = dict(winSize=(15, 15), maxLevel=2, criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03), )

        self._load_luts()
        self._load_sam_models()  # Load SAM models based on config paths
        self._initialize_effects()  # Define effects dictionary

        logger.info(
            f"PixelSensingEffectRunner initialized. Default effect: '{self.current_effect}'")
        logger.info(
            f"Initial SAM preference: {'SAM v2' if self.use_sam2_runtime else 'SAM v1'} (based on config/loading)")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config = {"input_source": "0", "output_path": "output.mp4", "lut_dir": "luts", "rvm_background_path": None, "sam_checkpoint_path": None, "sam2_checkpoint_path": None,
                          "mode": "live", "display": True, "device": "cpu", "trail_length": 30, "glow_intensity": 0.5, "width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "fps": DEFAULT_FPS, }
        config = default_config.copy()
        actual_config_path = config_path or CONFIG_FILE
        if os.path.exists(actual_config_path):
            try:
                with open(actual_config_path, "r") as f:
                    config.update(json.load(f))
                logger.info(f"Loaded config from {actual_config_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load/decode config {actual_config_path}: {e}. Using defaults.")
        else:
            logger.info(
                f"Config file '{actual_config_path}' not found. Using defaults.")
        return config

    def _apply_config(self):
        """Applies settings from the self.config dictionary."""
        self.input_source = self.config["input_source"]
        try:
            self.input_source = int(self.input_source)
            logger.info(f"Using camera index {self.input_source}")
        except (ValueError, TypeError):
            logger.info(f"Using input source path: {self.input_source}")

        self.output_path = self.config["output_path"]
        self.lut_dir = self.config["lut_dir"]
        self._init_rvm_bg_path = self.config["rvm_background_path"]
        self.sam_checkpoint_path = self.config["sam_checkpoint_path"]
        self.sam2_checkpoint_path = self.config["sam2_checkpoint_path"]
        self.mode = self.config["mode"]
        self.display = self.config["display"]

        requested_device = self.config["device"]
        target_device = "cpu"
        if requested_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            target_device = "mps"
        elif requested_device == "cuda" and torch.cuda.is_available():
            target_device = "cuda"
        elif requested_device != "cpu":
            logger.warning(
                f"Device '{requested_device}' unavailable/unsupported. Using CPU.")
        self.device = target_device
        logger.info(f"Selected device for ML models: {self.device}")

        self.trail_length = max(1, int(self.config["trail_length"]))
        self.glow_intensity = max(
            0.0, min(1.0, float(self.config["glow_intensity"])))
        self.desired_width = int(self.config.get("width", DEFAULT_WIDTH))
        self.desired_height = int(self.config.get("height", DEFAULT_HEIGHT))
        self.desired_fps = float(self.config.get("fps", DEFAULT_FPS))

        if hasattr(self, 'face_history') and self.face_history.maxlen != self.trail_length:
            self.face_history = deque(
                self.face_history, maxlen=self.trail_length)
        if hasattr(self, 'pose_history') and self.pose_history.maxlen != self.trail_length:
            self.pose_history = deque(
                self.pose_history, maxlen=self.trail_length)

        # Set SAM v2 preference based on validated config
        sam2_path = self.config.get("sam2_checkpoint_path")
        sam2_filename = os.path.basename(sam2_path) if sam2_path else None
        self.use_sam2_runtime = (
            _sam2_available and sam2_path is not None and sam2_filename in SUPPORTED_SAM2_MODELS)
        if sam2_path and sam2_filename not in SUPPORTED_SAM2_MODELS:
            logger.warning(
                f"SAM v2 Checkpoint '{sam2_filename}' is not supported. SAM v2 will be disabled.")
        logger.debug(
            f"Applied config. SAM runtime preference set: {'SAM v2' if self.use_sam2_runtime else 'SAM v1'}")

    def _initialize_mediapipe(self):
        if not _mediapipe_available:
            logger.warning(
                "Cannot initialize Mediapipe - library not available.")
            return
        logger.info("Initializing Mediapipe modules...")
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, )
            self.pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, enable_segmentation=True, )
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, )
            logger.info("Mediapipe modules initialized successfully.")
        except Exception as e:
            logger.error(
                f"Mediapipe initialization failed: {e}", exc_info=True)
            self.face_mesh = self.pose = self.hands = None

    def _load_luts(self):
        self.luts = []
        if not os.path.isdir(self.lut_dir):
            logger.warning(f"LUT directory '{self.lut_dir}' not found.")
            self.lut_color_grade_effect = None
            return
        found_luts = []
        for root, _, files in os.walk(self.lut_dir):
            [found_luts.append((os.path.splitext(f)[0], os.path.join(root, f)))
             for f in sorted(files) if f.lower().endswith(".cube")]
        logger.info(
            f"Found {len(found_luts)} potential LUT files in '{self.lut_dir}'.")
        if found_luts and _colour_available:
            if self.lut_color_grade_effect is None:
                self.lut_color_grade_effect = LUTColorGradeEffect(found_luts)
            else:
                self.lut_color_grade_effect.lut_files = found_luts
                self.lut_color_grade_effect._load_luts()
        elif not found_luts:
            logger.warning("No .cube files found. LUT effect disabled.")
            self.lut_color_grade_effect = None
        elif not _colour_available:
            logger.warning(
                "LUT files found, but colour-science unavailable. LUT effect disabled.")
            self.lut_color_grade_effect = None

    def _initialize_effects(self):
        """Defines the available effects and their requirements."""
        sam1_configured = self.sam_checkpoint_path is not None
        sam2_path = self.config.get("sam2_checkpoint_path")
        sam2_filename = os.path.basename(sam2_path) if sam2_path else None
        sam2_configured_and_valid = sam2_path is not None and sam2_filename in SUPPORTED_SAM2_MODELS

        self.effects: Dict[str, FrameEffectWrapper] = {
            "none": FrameEffectWrapper(lambda frame, **kw: frame, "none"),
            "led_base": FrameEffectWrapper(self._apply_led_base_style, "led_base", requires_mediapipe=True),
            "led_enhanced": FrameEffectWrapper(self._apply_led_enhanced_style, "led_enhanced", requires_mediapipe=True),
            "led_hue_rotate": FrameEffectWrapper(self._apply_led_hue_rotate_style, "led_hue_rotate", requires_mediapipe=True),
            "goldenaura": FrameEffectWrapper(self._apply_goldenaura_style, "goldenaura", requires_mediapipe=True),
            "motion_blur": FrameEffectWrapper(self._apply_motion_blur_style, "motion_blur"), # Existing temporal blur
            "motion_blur_kernel": FrameEffectWrapper(MotionBlurEffect(kernel_size=19).process, "motion_blur_kernel"), # New kernel-based blur
            "chromatic_aberration": FrameEffectWrapper(ChromaticAberrationEffect(base_strength=0.03, debug_borders=False).process, "chromatic_aberration"), 
            "advanced_color_grade": FrameEffectWrapper(AdvancedColorGradeEffect().process, "advanced_color_grade", requires_colour=True),
            "lightning": FrameEffectWrapper(self._apply_lightning_cycle, "lightning", requires_mediapipe=True),
            "lut_color_grade": FrameEffectWrapper((self.lut_color_grade_effect.process if self.lut_color_grade_effect else lambda frame, **kw: frame), "lut_color_grade", requires_colour=True),
            "rvm_composite": FrameEffectWrapper(self._apply_rvm_composite_style, "rvm_composite", requires_torch=True),
            "sam_segmentation": FrameEffectWrapper(self._apply_sam_segmentation_style, "sam_segmentation", requires_torch=True, requires_sam=sam1_configured, requires_sam2=sam2_configured_and_valid),
            "neon_glow": FrameEffectWrapper(self._apply_neon_glow_style, "neon_glow", requires_mediapipe=True),
            "particle_trail": FrameEffectWrapper(self._apply_particle_trail_style, "particle_trail", requires_mediapipe=True),

            # Newly integrated dataclass effects
            "adj_brightness_contrast": FrameEffectWrapper(BrightnessContrast(**self.bc_params).apply, "adj_brightness_contrast"),
            "adj_saturation": FrameEffectWrapper(Saturation(**self.saturation_params).apply, "adj_saturation"),
            "adj_vignette": FrameEffectWrapper(Vignette(**self.vignette_params).apply, "adj_vignette"),
            "adj_blur": FrameEffectWrapper(Blur(**self.blur_params).apply, "adj_blur"),
            "trans_crossfade": FrameEffectWrapper(Crossfade(duration=1.0).apply, "trans_crossfade"), # Example duration
            "trans_slide": FrameEffectWrapper(Slide(direction='left', duration=0.5).apply, "trans_slide"), # Example params
        }
        if self.current_effect not in self.effects:
            logger.warning(
                f"Default effect '{self.current_effect}' invalid. Falling back to 'none'.")
            self.current_effect = "none"
        logger.info(
            f"Initialized {len(self.effects)} effects: {list(self.effects.keys())}")
        logger.debug(
            f"SAM v1 Available: {_sam_available}, Configured: {sam1_configured}")
        logger.debug(
            f"SAM v2 Available: {_sam2_available}, Configured & Valid: {sam2_configured_and_valid}")

    def _load_sam_models(self):
        """Loads SAM models based on config paths and validation."""
        # SAM v1 Model Loading
        if self.sam_available and self.sam_checkpoint_path:
            if self.sam_model_v1 is None:
                logger.info(
                    f"Attempting to load SAM v1 from: {self.sam_checkpoint_path}")
                self.sam_model_v1 = load_sam_mask_generator(
                    self.device, checkpoint_path=self.sam_checkpoint_path)
                logger.info(
                    f"SAM v1 Model Loaded: {self.sam_model_v1 is not None}")
            else:
                logger.debug("SAM v1 model already loaded.")
        else:
            logger.info("SAM v1 Model Loaded: False")
        if self.sam_checkpoint_path and not self.sam_available:
            logger.warning(
                "SAM v1 path provided, but 'segment_anything' unavailable.")

        # SAM v2 Model Loading
        sam2_path = self.config.get("sam2_checkpoint_path")
        if self.sam2_available and sam2_path:  # Validation happens in load_sam2_video_predictor now
            if self.sam_model_v2 is None:
                logger.info(f"Attempting to load SAM v2 from: {sam2_path}")
                self.sam_model_v2 = load_sam2_video_predictor(
                    self.device, checkpoint_path=sam2_path)
                # Log result
                logger.info(
                    f"SAM v2 Model Loaded: {self.sam_model_v2 is not None}")
            else:
                logger.debug("SAM v2 model already loaded.")
        else:
            logger.info("SAM v2 Model Loaded: False")
        if sam2_path and not self.sam2_available:
            logger.warning(
                "SAM v2 path provided, but 'ultralytics' unavailable.")

        # Final update to runtime preference based on actual loaded state
        self.use_sam2_runtime = (self.sam_model_v2 is not None)
        logger.info(
            f"SAM preference updated based on *actual* loaded models: Use {'SAM v2' if self.use_sam2_runtime else 'SAM v1'} if available.")

    # --- Effect Implementations ---

    # Updated _apply_led_base_style to align with video_effects.py logic
    # It will use its internally managed landmarks (lm) and flow (fl, pf) for now.
    def _apply_led_base_style(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        
        # kwarg extraction similar to video_effects.py _process_frame, adapted for internal state
        # These would be derived from self.face_mesh.process, self.pose.process, and optical flow calculations
        # which are typically done before calling an effect method in video_effects.py's PixelSensingEffectRunner.
        # For video_effectsfinale5.py, these are often done *inside* the effect method.
        # We will continue this pattern for now but use the logic from video_effects.py's _apply_led_base_style.

        # Simulate kwarg extraction based on typical data flow in video_effectsfinale5.py's _process_frame
        # before it calls an effect. This is a bit of a hybrid approach.
        
        # 1. Get landmarks and flow (specific to how video_effectsfinale5.py currently does it)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        face_results = self.face_mesh.process(frame_rgb) if self.face_mesh else None
        # pose_results = self.pose.process(frame_rgb) if self.pose else None # Not directly used by this style in ref
        frame_rgb.flags.writeable = True

        lm = []
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    if landmark.x is not None and landmark.y is not None:
                         x, y = int(landmark.x * self.frame_width), int(landmark.y * self.frame_height)
                         if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                            lm.append((x,y))
        
        fl, pf = None, None
        if self.prev_gray is not None and self.prev_landmarks_flow is not None and lm:
            curr_pts_np = np.array(lm, dtype=np.float32).reshape(-1, 1, 2)
            # Ensure prev_landmarks_flow is also suitable for calcOpticalFlowPyrLK
            # It should be a list of tuples, then converted to numpy array
            if isinstance(self.prev_landmarks_flow, list) and len(self.prev_landmarks_flow) > 0:
                prev_pts_np = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                if prev_pts_np.shape[0] > 0 and curr_pts_np.shape[0] == prev_pts_np.shape[0]: # Check for valid shapes
                    try:
                        # Using frame_gray (current) and self.prev_gray
                        flow_calc, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts_np, None, **self.LK_PARAMS)
                        if flow_calc is not None and status is not None:
                            good_new = flow_calc[status.flatten() == 1]
                            good_old = prev_pts_np[status.flatten() == 1]
                            if good_new.shape[0] > 0 and good_new.shape == good_old.shape:
                                fl = good_new.reshape(-1, 1, 2)
                                pf = good_old.reshape(-1, 1, 2)
                    except cv2.error as lk_err:
                        logger.warning(f"LK Optical Flow in _apply_led_base_style failed: {lk_err}")
                    except Exception as lk_e:
                         logger.warning(f"LK Optical Flow unexpected error in _apply_led_base_style: {lk_e}")
        
        self.prev_gray = gray.copy()
        self.prev_landmarks_flow = lm # Update for next frame

        # 2. Apply effects as per video_effects.py's _apply_led_base_style
        proc = frame.copy() # Start with a copy of the current frame

        # Call to _draw_flow_trails_simple (now uses updated version)
        proc = self._draw_flow_trails_simple(proc, fl, pf) # Uses fl, pf derived above

        # Color overlay
        try:
            co = np.full(proc.shape, (30, 0, 90), dtype=np.uint8) # BGR color
            proc = cv2.addWeighted(proc, 0.85, co, 0.15, 0)
        except Exception as e:
            logger.warning(f"LED Base style color overlay failed: {e}")
        
        # Call to _draw_motion_glow_separate (now uses updated version)
        # Pass lm (current landmarks) and the flow data (fl, pf)
        proc = self._draw_motion_glow_separate(proc, lm, fl, pf) 

        # Call to _apply_distortion (now uses updated version)
        proc = self._apply_distortion(proc)
        
        return proc


    # Updated _apply_led_enhanced_style to align with video_effects.py logic
    def _apply_led_enhanced_style(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        proc = frame.copy() # Start with a copy
        h, w = proc.shape[:2]

        # 1. Get landmarks and flow (consistent with how video_effectsfinale5.py currently operates internally for these styles)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        face_results = self.face_mesh.process(frame_rgb) if self.face_mesh else None
        frame_rgb.flags.writeable = True
        
        lm = []
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark_obj in face_landmarks.landmark: # Renamed 'landmark' to 'landmark_obj' to avoid conflict
                    if landmark_obj.x is not None and landmark_obj.y is not None:
                         x, y = int(landmark_obj.x * self.frame_width), int(landmark_obj.y * self.frame_height)
                         if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                            lm.append((x,y))

        fl, pf = None, None
        if self.prev_gray is not None and self.prev_landmarks_flow is not None and lm:
            curr_pts_np = np.array(lm, dtype=np.float32).reshape(-1, 1, 2)
            if isinstance(self.prev_landmarks_flow, list) and len(self.prev_landmarks_flow) > 0:
                prev_pts_np = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                if prev_pts_np.shape[0] > 0 and curr_pts_np.shape[0] == prev_pts_np.shape[0]:
                    try:
                        flow_calc, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts_np, None, **self.LK_PARAMS)
                        if flow_calc is not None and status is not None:
                            good_new = flow_calc[status.flatten() == 1]
                            good_old = prev_pts_np[status.flatten() == 1]
                            if good_new.shape[0] > 0 and good_new.shape == good_old.shape:
                                fl = good_new.reshape(-1, 1, 2)
                                pf = good_old.reshape(-1, 1, 2)
                    except cv2.error as lk_err:
                        logger.warning(f"LK Optical Flow in _apply_led_enhanced_style failed: {lk_err}")
                    except Exception as lk_e:
                        logger.warning(f"LK Optical Flow unexpected error in _apply_led_enhanced_style: {lk_e}")
        
        self.prev_gray = gray.copy()
        self.prev_landmarks_flow = lm

        # 2. Apply line drawing and glow as per video_effects.py
        if fl is not None and pf is not None and fl.shape[0] == pf.shape[0]:
            try:
                for i, (new, old) in enumerate(zip(fl, pf)): # Use fl, pf from above
                    xn, yn = new.ravel()
                    xo, yo = old.ravel()
                    if 0 <= xn < w and 0 <= yn < h and 0 <= xo < w and 0 <= yo < h:
                        cv2.line(proc, (int(xo), int(yo)), (int(xn), int(yn)), (255, 105, 180), 1, cv2.LINE_AA) # Hot pink lines
                        # Magnitude-based glow
                        mag = np.linalg.norm(new - old)
                        gi = min(255, int(mag * 15)) # Glow intensity
                        gc = (gi // 4, gi // 2, gi) # Glow color (blueish-purple tones)
                        cv2.circle(proc, (int(xn), int(yn)), 3, gc, -1, cv2.LINE_AA) # Glow dots
            except Exception as e:
                logger.warning(f"LED Enhanced style flow draw failed: {e}")
        
        # 3. Apply distortion at the end
        proc = self._apply_distortion(proc)
        
        return proc

    # Updated _apply_led_hue_rotate_style to align with video_effects.py logic
    def _apply_led_hue_rotate_style(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        proc = frame.copy()

        # 1. Get landmarks and flow (consistent with how video_effectsfinale5.py currently operates internally)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        face_results = self.face_mesh.process(frame_rgb) if self.face_mesh else None
        frame_rgb.flags.writeable = True

        lm = []
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark_obj in face_landmarks.landmark:
                    if landmark_obj.x is not None and landmark_obj.y is not None:
                        x, y = int(landmark_obj.x * self.frame_width), int(landmark_obj.y * self.frame_height)
                        if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                            lm.append((x,y))
        
        fl, pf = None, None
        if self.prev_gray is not None and self.prev_landmarks_flow is not None and lm:
            curr_pts_np = np.array(lm, dtype=np.float32).reshape(-1, 1, 2)
            if isinstance(self.prev_landmarks_flow, list) and len(self.prev_landmarks_flow) > 0:
                prev_pts_np = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)
                if prev_pts_np.shape[0] > 0 and curr_pts_np.shape[0] == prev_pts_np.shape[0]:
                    try:
                        flow_calc, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts_np, None, **self.LK_PARAMS)
                        if flow_calc is not None and status is not None:
                            good_new = flow_calc[status.flatten() == 1]
                            good_old = prev_pts_np[status.flatten() == 1]
                            if good_new.shape[0] > 0 and good_new.shape == good_old.shape:
                                fl = good_new.reshape(-1, 1, 2)
                                pf = good_old.reshape(-1, 1, 2)
                    except cv2.error as lk_err:
                        logger.warning(f"LK Optical Flow in _apply_led_hue_rotate_style failed: {lk_err}")
                    except Exception as lk_e:
                        logger.warning(f"LK Optical Flow unexpected error in _apply_led_hue_rotate_style: {lk_e}")

        self.prev_gray = gray.copy()
        self.prev_landmarks_flow = lm
        
        # 2. Apply trail, glow, distortion first (using updated helper methods)
        proc = self._draw_flow_trails_simple(proc, fl, pf) # Uses fl, pf derived above
        proc = self._draw_motion_glow_separate(proc, lm, fl, pf) # Uses lm, fl, pf
        proc = self._apply_distortion(proc)

        # 3. Apply Hue Rotation
        try:
            hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int) + self.hue_offset) % 180
            proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            logger.warning(f"LED Hue rotate style actual hue rotation failed: {e}")
            return original_frame_fallback # Return original if hue rotation fails

        self.hue_offset = (self.hue_offset + 1) % 180
        return proc

    def _apply_goldenaura_original(self, frame: np.ndarray, frame_time: float, original_frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        frame_out = frame.copy()
        try:
            if not self.face_mesh or not self.pose or not self.frame_width or not self.frame_height:
                return original_frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = self.face_mesh.process(frame_rgb)
            results_pose = self.pose.process(frame_rgb)
            frame.flags.writeable = True
            h, w = frame.shape[:2]
            face_landmarks_current, pose_landmarks_current = [], []
            if results_face.multi_face_landmarks:
                for face_lm_set in results_face.multi_face_landmarks: # Iterate over each detected face
                    for lm in face_lm_set.landmark: face_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
            if results_pose.pose_landmarks:
                for lm in results_pose.pose_landmarks.landmark:
                    if lm.visibility > 0.3: pose_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
            
            overlay = np.zeros_like(frame_out, dtype=np.uint8)

            # Robust Optical Flow Update from video_effectsfinale4.py
            if self.prev_gray is not None:
                # Face
                if face_landmarks_current and self.prev_face_pts_gold.shape[0] > 0:
                    curr_pts_face = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    # Ensure shapes are compatible for flow calculation, may need to select corresponding prev_pts if landmark count changes
                    if curr_pts_face.shape[0] == self.prev_face_pts_gold.shape[0]:
                        try:
                            flow_face, status_face, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_face_pts_gold, curr_pts_face, **self.LK_PARAMS_GOLD)
                            if flow_face is not None and status_face is not None:
                                good_new = flow_face[status_face.flatten() == 1]
                                good_old = self.prev_face_pts_gold[status_face.flatten() == 1]
                                if good_new.shape[0] > 0: self.face_history.append((good_new, good_old))
                                self.prev_face_pts_gold = good_new.reshape(-1, 1, 2) if good_new.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
                            else: # Flow failed
                                self.face_history.clear()
                                self.prev_face_pts_gold = curr_pts_face # Re-init with current points
                        except cv2.error as cv_err: logger.warning(f"GO_Orig face LK failed: {cv_err}"); self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                    else: # Landmark count changed
                         self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                elif face_landmarks_current: # No previous points, but current points exist
                    self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    self.face_history.clear()
                else: # No current face landmarks
                    self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
                    self.face_history.clear()

                # Pose
                if pose_landmarks_current and self.prev_pose_pts_gold.shape[0] > 0:
                    curr_pts_pose = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    if curr_pts_pose.shape[0] == self.prev_pose_pts_gold.shape[0]:
                        try:
                            flow_pose, status_pose, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pose_pts_gold, curr_pts_pose, **self.LK_PARAMS_GOLD)
                            if flow_pose is not None and status_pose is not None:
                                good_new_pose = flow_pose[status_pose.flatten() == 1]
                                good_old_pose = self.prev_pose_pts_gold[status_pose.flatten() == 1]
                                if good_new_pose.shape[0] > 0: self.pose_history.append((good_new_pose, good_old_pose))
                                self.prev_pose_pts_gold = good_new_pose.reshape(-1, 1, 2) if good_new_pose.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
                            else: # Flow failed
                                self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                        except cv2.error as cv_err: logger.warning(f"GO_Orig pose LK failed: {cv_err}"); self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                    else: # Landmark count changed
                        self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                elif pose_landmarks_current: # No previous points, but current points exist
                    self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    self.pose_history.clear()
                else: # No current pose landmarks
                    self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
                    self.pose_history.clear()
            else: # No previous gray frame
                 if face_landmarks_current: self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                 if pose_landmarks_current: self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                 self.face_history.clear(); self.pose_history.clear()

            # Draw Trails (logic from video_effectsfinale4.py)
            if self.face_history:
                num_face_hist = len(self.face_history)
                max_idx_face = max(num_face_hist - 1, 1)
                for idx, item_hist in enumerate(reversed(self.face_history)): # Iterate in reverse for older trails first
                    if isinstance(item_hist, tuple) and len(item_hist) == 2 and item_hist[0].shape[0] > 0 and item_hist[0].shape[0] == item_hist[1].shape[0]:
                        flow_hist, prev_hist = item_hist
                        alpha_hist = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx_face)
                        color_hist = tuple(int(c_val * alpha_hist) for c_val in GOLD_TINT_COLOR)
                        overlay = self._draw_flow_trails_simple(overlay, flow_hist, prev_hist, color_hist, color_hist, 1, TRAIL_RADIUS)
            if self.pose_history:
                num_pose_hist = len(self.pose_history)
                max_idx_pose = max(num_pose_hist - 1, 1)
                for idx, item_hist in enumerate(reversed(self.pose_history)):
                     if isinstance(item_hist, tuple) and len(item_hist) == 2 and item_hist[0].shape[0] > 0 and item_hist[0].shape[0] == item_hist[1].shape[0]:
                        flow_hist, prev_hist = item_hist
                        alpha_hist = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx_pose)
                        color_hist = tuple(int(c_val * alpha_hist) for c_val in GOLD_TINT_COLOR)
                        overlay = self._draw_flow_trails_simple(overlay, flow_hist, prev_hist, color_hist, color_hist, 1, TRAIL_RADIUS)
            
            frame_out = cv2.add(frame_out, overlay) # Add trails to the output frame

            # Apply Tint
            tint_layer = np.full_like(
                frame_out, GOLD_TINT_COLOR, dtype=np.uint8)
            frame_out = cv2.addWeighted(
                frame_out, 1.0 - TINT_STRENGTH, tint_layer, TINT_STRENGTH, 0.0)
            if (hasattr(results_pose, "segmentation_mask") and results_pose.segmentation_mask is not None):
                try:
                    mask = results_pose.segmentation_mask
                    condition = (mask > SEGMENTATION_THRESHOLD).astype(
                        np.uint8) * 255
                    if condition.shape[:2] != frame_out.shape[:2]:
                        condition = cv2.resize(
                            condition, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_blurred = cv2.GaussianBlur(
                        condition, MASK_BLUR_KERNEL, 0)
                    mask_alpha = (mask_blurred.astype(
                        np.float32) / 255.0 * self.glow_intensity)
                    mask_alpha_3c = cv2.cvtColor(
                        mask_alpha, cv2.COLOR_GRAY2BGR)
                    glow_color_layer = np.full_like(
                        frame_out, GOLD_TINT_COLOR, dtype=np.float32)
                    frame_float = frame_out.astype(np.float32)
                    frame_float = (
                        frame_float * (1.0 - mask_alpha_3c) + glow_color_layer * mask_alpha_3c)
                    frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
                except Exception as seg_e:
                    logger.warning(f"GO segmentation/glow failed: {seg_e}")
            self.prev_gray = gray.copy()
        except Exception as e:
            logger.error(
                f"Error in _apply_goldenaura_original: {e}", exc_info=True)
            self.prev_gray = None
            self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
            self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
            self.face_history.clear()
            self.pose_history.clear()
            return original_frame
        return frame_out

    def _apply_goldenaura_enhanced(self, frame: np.ndarray, frame_time: float, original_frame: np.ndarray, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        frame_out = frame.copy()
        try:
            if not self.face_mesh or not self.pose or not self.frame_width or not self.frame_height:
                return original_frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = self.face_mesh.process(frame_rgb)
            results_pose = self.pose.process(frame_rgb)
            frame.flags.writeable = True
            h, w = frame.shape[:2]
            face_landmarks_current, pose_landmarks_current = [], []
            if results_face.multi_face_landmarks:
                for face_lm_set in results_face.multi_face_landmarks:
                    for lm in face_lm_set.landmark: face_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
            if results_pose.pose_landmarks:
                for lm in results_pose.pose_landmarks.landmark:
                    if lm.visibility > 0.3: pose_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
            
            overlay = np.zeros_like(frame_out, dtype=np.uint8)

            # Robust Optical Flow Update (similar to _apply_goldenaura_original, using logic from video_effectsfinale4.py)
            if self.prev_gray is not None:
                # Face
                if face_landmarks_current and self.prev_face_pts_gold.shape[0] > 0:
                    curr_pts_face = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                    if curr_pts_face.shape[0] == self.prev_face_pts_gold.shape[0]:
                        try:
                            flow_face, status_face, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_face_pts_gold, curr_pts_face, **self.LK_PARAMS_GOLD)
                            if flow_face is not None and status_face is not None:
                                good_new = flow_face[status_face.flatten() == 1]
                                good_old = self.prev_face_pts_gold[status_face.flatten() == 1]
                                if good_new.shape[0] > 0: self.face_history.append((good_new, good_old))
                                self.prev_face_pts_gold = good_new.reshape(-1, 1, 2) if good_new.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
                            else: self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
                        except cv2.error as cv_err: logger.warning(f"GO_Enh face LK failed: {cv_err}"); self.face_history.clear(); self.prev_face_pts_gold = curr_pts_face
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
                                good_new_pose = flow_pose[status_pose.flatten() == 1]
                                good_old_pose = self.prev_pose_pts_gold[status_pose.flatten() == 1]
                                if good_new_pose.shape[0] > 0: self.pose_history.append((good_new_pose, good_old_pose))
                                self.prev_pose_pts_gold = good_new_pose.reshape(-1, 1, 2) if good_new_pose.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
                            else: self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                        except cv2.error as cv_err: logger.warning(f"GO_Enh pose LK failed: {cv_err}"); self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                    else: self.pose_history.clear(); self.prev_pose_pts_gold = curr_pts_pose
                elif pose_landmarks_current: self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2); self.pose_history.clear()
                else: self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32); self.pose_history.clear()
            else: # No previous gray frame
                 if face_landmarks_current: self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                 if pose_landmarks_current: self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
                 self.face_history.clear(); self.pose_history.clear()

            # Enhanced Drawing Params (from video_effectsfinale4.py)
            dynamic_glow_factor = 0.7 + 0.3 * math.sin(frame_time * 1.5) # Pulsating effect
            enhanced_tint_color_base = np.array([30, 180, 220]) # Slightly different gold/amber
            enhanced_tint_color = tuple(np.clip(enhanced_tint_color_base * dynamic_glow_factor, 0, 255).astype(int))
            line_thickness_face, dot_radius_face = 2, 3 # Thicker trails
            line_thickness_pose, dot_radius_pose = 3, 4

            # Draw Trails (using enhanced params)
            if self.face_history:
                num_face_hist = len(self.face_history); max_idx_face = max(num_face_hist - 1, 1)
                for idx, item_hist in enumerate(reversed(self.face_history)):
                    if isinstance(item_hist, tuple) and len(item_hist) == 2 and item_hist[0].shape[0] > 0 and item_hist[0].shape[0] == item_hist[1].shape[0]:
                        flow_hist, prev_hist = item_hist; alpha_hist = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx_face)
                        color_hist = tuple(int(c_val * alpha_hist) for c_val in enhanced_tint_color)
                        overlay = self._draw_flow_trails_simple(overlay, flow_hist, prev_hist, color_hist, color_hist, line_thickness_face, dot_radius_face)
            if self.pose_history:
                num_pose_hist = len(self.pose_history); max_idx_pose = max(num_pose_hist - 1, 1)
                for idx, item_hist in enumerate(reversed(self.pose_history)):
                     if isinstance(item_hist, tuple) and len(item_hist) == 2 and item_hist[0].shape[0] > 0 and item_hist[0].shape[0] == item_hist[1].shape[0]:
                        flow_hist, prev_hist = item_hist; alpha_hist = TRAIL_END_ALPHA + (TRAIL_START_ALPHA - TRAIL_END_ALPHA) * (idx / max_idx_pose)
                        color_hist = tuple(int(c_val * alpha_hist) for c_val in enhanced_tint_color)
                        overlay = self._draw_flow_trails_simple(overlay, flow_hist, prev_hist, color_hist, color_hist, line_thickness_pose, dot_radius_pose)
            
            frame_out = cv2.add(frame_out, overlay)

            # Apply Enhanced Tint
            tint_strength_enhanced = min(1.0, TINT_STRENGTH * 1.2) # Slightly stronger tint
            tint_layer_enhanced = np.full_like(
                frame_out, enhanced_tint_color, dtype=np.uint8)
            frame_out = cv2.addWeighted(
                frame_out, 1.0 - tint_strength_enhanced, tint_layer_enhanced, tint_strength_enhanced, 0.0, )
            if (hasattr(results_pose, "segmentation_mask") and results_pose.segmentation_mask is not None):
                try:
                    mask = results_pose.segmentation_mask
                    condition = (mask > SEGMENTATION_THRESHOLD).astype(
                        np.uint8) * 255
                    if condition.shape[:2] != frame_out.shape[:2]:
                        condition = cv2.resize(
                            condition, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_blurred = cv2.GaussianBlur(condition, (31, 31), 0)
                    mask_alpha = (mask_blurred.astype(
                        np.float32) / 255.0 * self.glow_intensity * dynamic_glow_factor)
                    mask_alpha_3c = cv2.cvtColor(
                        mask_alpha, cv2.COLOR_GRAY2BGR)
                    glow_color_layer = np.full_like(
                        frame_out, enhanced_tint_color, dtype=np.float32)
                    frame_float = frame_out.astype(np.float32)
                    frame_float = (
                        frame_float * (1.0 - mask_alpha_3c) + glow_color_layer * mask_alpha_3c)
                    frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
                except Exception as seg_e:
                    logger.warning(f"GE segmentation/glow failed: {seg_e}")
            self.prev_gray = gray.copy()
        except Exception as e:
            logger.error(
                f"Error in _apply_goldenaura_enhanced: {e}", exc_info=True)
            self.prev_gray = None
            self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
            self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
            self.face_history.clear()
            self.pose_history.clear()
            return original_frame
        return frame_out

    def _apply_goldenaura_style(self, frame: np.ndarray, frame_time: float = 0.0, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        if self.goldenaura_variant == 0:
            frame_out = self._apply_goldenaura_original(
                frame, frame_time, original_frame=original_frame_fallback, **kwargs)
            if not np.array_equal(frame_out, original_frame_fallback):
                cv2.putText(frame_out, "Aura: Original", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, )
        else:
            frame_out = self._apply_goldenaura_enhanced(
                frame, frame_time, original_frame=original_frame_fallback, **kwargs)
            if not np.array_equal(frame_out, original_frame_fallback):
                cv2.putText(frame_out, "Aura: Enhanced", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, )
        return frame_out

    # _apply_motion_blur_style will be removed/replaced next
    def _apply_motion_blur_style(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        frame_out = frame
        try:
            self.frame_buffer.append(frame.astype(np.float32))
        except Exception as e:
            logger.error(f"Error adding frame to motion blur buffer: {e}")
            return original_frame_fallback
        try:
            if len(self.frame_buffer) > 1:
                frame_out = np.clip(
                    np.mean(np.array(self.frame_buffer), axis=0), 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error calculating motion blur avg: {e}")
            self.frame_buffer.clear()
            return original_frame_fallback
    # This _apply_chromatic_aberration_style method will be removed.
    # The ChromaticAberrationEffect class is defined above.

    def _apply_lightning_style_random(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.float32)
        lightning_color_int = (200, 220, 255)  # BGR int
        try:
            num_bolts = random.randint(1, 4)
            for _ in range(num_bolts):
                edge = random.choice(["top", "bottom", "left", "right"])
                if edge == "top":
                    x0, y0 = random.randint(0, w - 1), random.randint(-20, 0)
                elif edge == "bottom":
                    x0, y0 = random.randint(
                        0, w - 1), random.randint(h, h + 20)
                elif edge == "left":
                    x0, y0 = random.randint(-20, 0), random.randint(0, h - 1)
                else:
                    x0, y0 = random.randint(
                        w, w + 20), random.randint(0, h - 1)
                x1, y1 = random.randint(
                    w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
                num_segments = random.randint(5, 15)
                px, py = float(x0), float(y0)
                for i in range(num_segments):
                    target_x = x0 + (x1 - x0) * (i + 1) / num_segments
                    target_y = y0 + (y1 - y0) * (i + 1) / num_segments
                    deviation = random.uniform(
                        20, 60) * ((num_segments - i) / num_segments)
                    nx = target_x + random.uniform(-deviation, deviation)
                    ny = target_y + random.uniform(-deviation, deviation)
                    thickness = random.randint(1, 3)
                    cv2.line(overlay, (int(px), int(py)), (int(nx), int(
                        ny)), lightning_color_int, thickness, cv2.LINE_AA)
                    px, py = nx, ny
            overlay_blurred = cv2.GaussianBlur(overlay, (7, 7), 0)
            overlay_glow = cv2.GaussianBlur(overlay, (25, 25), 0)
            frame_float = original_frame_fallback.astype(np.float32)
            frame_float = cv2.addWeighted(
                frame_float, 1.0, overlay_blurred, 1.5, 0.0)
            frame_float = cv2.addWeighted(
                frame_float, 1.0, overlay_glow, 0.8, 0.0)
            frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in lightning random: {e}")
            return original_frame_fallback
        return frame_out

    def _apply_lightning_style_movement(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        overlay = np.zeros_like(frame, dtype=np.float32)
        lightning_color_main = (255, 245, 200)
        lightning_color_glow = (200, 220, 255)
        try:
            if not self.face_mesh or not self.frame_width or not self.frame_height:
                return original_frame_fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame.flags.writeable = False
            results = self.face_mesh.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.flags.writeable = True
            landmarks = []
            if results.multi_face_landmarks:
                landmarks = [(int(lm.x * self.frame_width), int(lm.y * self.frame_height))
                             for fl in results.multi_face_landmarks for lm in fl.landmark]
            if not landmarks:
                self.prev_gray = gray
                self.prev_landmarks_flow = None
                return original_frame_fallback
            h, w = frame.shape[:2]
            frame_out = original_frame_fallback.copy()
            if self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks):
                curr_pts = np.array(
                    landmarks, dtype=np.float32).reshape(-1, 1, 2)
                prev_pts = np.array(self.prev_landmarks_flow,
                                    dtype=np.float32).reshape(-1, 1, 2)
                valid_indices = np.where((prev_pts[:, 0, 0] >= 0) & (prev_pts[:, 0, 0] < w) & (
                    prev_pts[:, 0, 1] >= 0) & (prev_pts[:, 0, 1] < h))[0]
                if len(valid_indices) > 0:
                    prev_pts_valid = prev_pts[valid_indices]
                    try:
                        flow, status, _ = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS)
                    except cv2.error as e:
                        logger.warning(f"LK Error: {e}")
                        flow, status = None, None
                    if flow is not None and status is not None:
                        good_new = flow[status.flatten() == 1]
                        good_old = prev_pts_valid[status.flatten() == 1]
                        if good_new.shape[0] > 0:
                            good_new_2d = good_new.reshape(-1, 2)
                            good_old_2d = good_old.reshape(-1, 2)
                            movement_vectors = good_new_2d - good_old_2d
                            magnitudes = np.linalg.norm(
                                movement_vectors, axis=1)
                            movement_threshold = 5.0
                            for i in range(len(good_new_2d)):
                                # *** Corrected comparison ***
                                if float(magnitudes[i]) > movement_threshold:
                                    xn, yn = good_new_2d[i]
                                    xo, yo = good_old_2d[i]
                                    pt1 = (int(xo), int(yo))
                                    pt2 = (int(xn), int(yn))
                                    cv2.line(
                                        overlay, pt1, pt2, lightning_color_main, 2, cv2.LINE_AA)
                                    cv2.line(
                                        overlay, pt1, pt2, lightning_color_glow, 1, cv2.LINE_AA)
            if np.any(overlay > 0):
                overlay_blurred = cv2.GaussianBlur(overlay, (5, 5), 0)
                overlay_glow = cv2.GaussianBlur(overlay, (15, 15), 0)
                frame_float = frame_out.astype(np.float32)
                frame_float = cv2.addWeighted(
                    frame_float, 1.0, overlay_blurred, 1.8, 0.0)
                frame_float = cv2.addWeighted(
                    frame_float, 1.0, overlay_glow, 1.0, 0.0)
                frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
            self.prev_landmarks_flow = landmarks
            self.prev_gray = gray.copy()
        except Exception as e:
            logger.error(f"Error in lightning movement: {e}", exc_info=True)
            self.prev_gray = None
            self.prev_landmarks_flow = None
            frame_out = original_frame_fallback
        return frame_out

    def _apply_lightning_style_branching(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.float32)
        lightning_color_int = (210, 230, 255)  # BGR
        try:
            num_bolts = random.randint(1, 3)
            for _ in range(num_bolts):
                edge = random.choice(["top", "bottom", "left", "right"])
                if edge == "top":
                    x0, y0 = random.randint(0, w - 1), random.randint(-20, 0)
                elif edge == "bottom":
                    x0, y0 = random.randint(
                        0, w - 1), random.randint(h, h + 20)
                elif edge == "left":
                    x0, y0 = random.randint(-20, 0), random.randint(0, h - 1)
                else:
                    x0, y0 = random.randint(
                        w, w + 20), random.randint(0, h - 1)
                x1, y1 = random.randint(
                    w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
                num_segments = random.randint(8, 20)
                px, py = float(x0), float(y0)
                current_angle = math.atan2(y1 - y0, x1 - x0)
                base_thickness = random.randint(2, 4)
                for i in range(num_segments):
                    dist_to_target = math.sqrt((x1-px)**2 + (y1-py)**2)
                    segment_len = max(
                        10.0, dist_to_target / max(1, num_segments - i)) * random.uniform(0.8, 1.2)
                    current_angle += random.uniform(-0.4, 0.4)
                    nx = px + segment_len * math.cos(current_angle)
                    ny = py + segment_len * math.sin(current_angle)
                    thickness = max(
                        1, int(base_thickness * ((num_segments - i) / num_segments)))
                    cv2.line(overlay, (int(px), int(py)), (int(nx), int(
                        ny)), lightning_color_int, thickness, cv2.LINE_AA)
                    if random.random() < 0.3 and i < num_segments - 2:
                        branch_angle_offset = random.uniform(
                            0.6, 1.2) * random.choice([-1, 1])
                        branch_angle = current_angle + branch_angle_offset
                        branch_length = segment_len * random.uniform(1.5, 3.0)
                        branch_end_x = nx + branch_length * \
                            math.cos(branch_angle)
                        branch_end_y = ny + branch_length * \
                            math.sin(branch_angle)
                        branch_thickness = max(1, thickness - 1)
                        cv2.line(overlay, (int(nx), int(ny)), (int(branch_end_x), int(
                            branch_end_y)), lightning_color_int, branch_thickness, cv2.LINE_AA)
                    px, py = nx, ny
            overlay_blurred = cv2.GaussianBlur(overlay, (5, 5), 0)
            overlay_glow = cv2.GaussianBlur(overlay, (29, 29), 0)
            frame_float = original_frame_fallback.astype(np.float32)
            frame_float = cv2.addWeighted(
                frame_float, 1.0, overlay_blurred, 1.6, 0.0)
            frame_float = cv2.addWeighted(
                frame_float, 1.0, overlay_glow, 1.0, 0.0)
            frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in lightning branching: {e}")
            return original_frame_fallback
        return frame_out

    def _apply_lightning_cycle(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        if not self.lightning_styles:
            cv2.putText(frame, "Lightning: No Styles", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            return original_frame_fallback
        try:
            style_function = self.lightning_styles[self.lightning_style_index % len(
                self.lightning_styles)]
            style_name = self.lightning_style_names[self.lightning_style_index % len(
                self.lightning_style_names)]
            frame_out = style_function(
                frame=frame, original_frame=original_frame_fallback, **kwargs)
            if not np.array_equal(frame_out, original_frame_fallback):
                cv2.putText(frame_out, f"Lightning: {style_name}", (
                    10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, )
        except Exception as e:
            logger.error(
                f"Error applying lightning style '{style_name}': {e}", exc_info=True)
            error_display_frame = original_frame_fallback.copy()
            cv2.putText(error_display_frame, f"Lightning Error: {style_name}", (
                10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, )
            return error_display_frame
        return frame_out

    def _apply_neon_glow_style(self, frame: np.ndarray, frame_time: float, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        frame_out = frame.copy()
        try:
            if not self.pose or not self.frame_width or not self.frame_height:
                return original_frame_fallback
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = self.pose.process(frame_rgb)
            frame.flags.writeable = True
            h, w = frame.shape[:2]
            overlay = np.zeros_like(frame_out, dtype=np.uint8)
            if results_pose.pose_landmarks:
                landmarks = []
                connections = mp.solutions.pose.POSE_CONNECTIONS
                for id, lm in enumerate(results_pose.pose_landmarks.landmark):
                    landmarks.append((int(lm.x * w), int(lm.y * h))
                                     if lm.visibility > 0.3 else None)
                b = int(127 + 127 * math.sin(frame_time * 1.0))
                g = int(127 + 127 * math.sin(frame_time * 1.0 + 2 * math.pi / 3))
                r = int(127 + 127 * math.sin(frame_time * 1.0 + 4 * math.pi / 3))
                neon_color = (b, g, r)
                if connections:
                    [cv2.line(overlay, landmarks[s], landmarks[e], neon_color, 3, cv2.LINE_AA) for s, e in connections if s < len(
                        landmarks) and e < len(landmarks) and landmarks[s] and landmarks[e]]
                glow_overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
                wider_glow_overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
                frame_out = cv2.addWeighted(
                    frame_out, 1.0, glow_overlay, 0.8, 0)
                frame_out = cv2.addWeighted(
                    frame_out, 1.0, wider_glow_overlay, 0.4, 0)
            if (hasattr(results_pose, "segmentation_mask") and results_pose.segmentation_mask is not None):
                try:
                    mask = results_pose.segmentation_mask
                    condition = (mask > SEGMENTATION_THRESHOLD).astype(
                        np.uint8) * 255
                    if condition.shape[:2] != frame_out.shape[:2]:
                        condition = cv2.resize(
                            condition, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_blurred = cv2.GaussianBlur(condition, (31, 31), 0)
                    mask_alpha = (mask_blurred.astype(
                        np.float32) / 255.0 * self.glow_intensity * 0.5)
                    mask_alpha_3c = cv2.cvtColor(
                        mask_alpha, cv2.COLOR_GRAY2BGR)
                    seg_glow_layer = np.full_like(
                        frame_out, neon_color, dtype=np.float32)
                    frame_float = frame_out.astype(np.float32)
                    frame_float = (
                        frame_float * (1.0 - mask_alpha_3c) + seg_glow_layer * mask_alpha_3c)
                    frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
                except Exception as seg_e:
                    logger.warning(
                        f"Neon glow segmentation blend failed: {seg_e}")
        except Exception as e:
            logger.error(f"Error in neon glow: {e}")
            return original_frame_fallback
        return frame_out

    def _apply_particle_trail_style(self, frame: np.ndarray, frame_time: float, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        frame_out = frame.copy()
        hands_detected = False
        try:
            if not self.hands or not self.frame_width or not self.frame_height:
                return original_frame_fallback
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(frame_rgb)
            frame.flags.writeable = True
            h, w = frame.shape[:2]
            if results_hands.multi_hand_landmarks:
                hands_detected = True
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    lm_wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    if lm_wrist.visibility > 0.1 and lm_wrist.presence > 0.1:
                        x, y = int(lm_wrist.x * w), int(lm_wrist.y * h)
                        if 0 <= x < w and 0 <= y < h:
                            num_particles = 20
                            max_offset = 30
                            min_size, max_size = 1, 5
                            for _ in range(num_particles):
                                offset_dist = abs(
                                    random.gauss(0, max_offset / 2.5))
                                offset_angle = random.uniform(0, 2 * math.pi)
                                ox, oy = int(x + offset_dist * math.cos(offset_angle)
                                             ), int(y + offset_dist * math.sin(offset_angle))
                                if 0 <= ox < w and 0 <= oy < h:
                                    pc = (random.randint(100, 255), random.randint(
                                        100, 255), random.randint(100, 255), )
                                    ps = random.randint(min_size, max_size)
                                    cv2.circle(frame_out, (ox, oy),
                                               ps, pc, -1, cv2.LINE_AA, )
            if not hands_detected:
                cv2.putText(frame_out, "No Hands Detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, )
        except Exception as e:
            logger.error(f"Error in particle trail: {e}")
            return original_frame_fallback
        return frame_out

    def _apply_rvm_composite_style(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()
        frame_out = frame.copy()
        model_name = self.rvm_model_names[self.current_rvm_model_idx % len(
            self.rvm_model_names)]
        model = self.rvm_models.get(model_name)
        if not self.rvm_available or not model:
            cv2.putText(frame_out, "RVM Unavailable", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return original_frame_fallback
        try:
            src = self._preprocess_frame_rvm(frame)
            rec = [r.to(self.device)
                   if r is not None else None for r in self.rvm_rec]
            with torch.no_grad():
                fgr, pha, * \
                    rec = model(
                        src, *rec, downsample_ratio=self.rvm_downsample_ratio)
            self.rvm_rec = [r.cpu() if r is not None else None for r in rec]
            fgr_np, pha_np = self._postprocess_output_rvm(fgr, pha)
            if fgr_np is None or pha_np is None:
                raise ValueError("RVM Postprocessing failed")
            if fgr_np.shape[:2] != frame.shape[:2]:
                fgr_np = resize_frame(fgr_np, (frame.shape[0], frame.shape[1]))
            if pha_np.shape[:2] != frame.shape[:2]:
                pha_np = cv2.resize(
                    pha_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR, )
            mode = self.RVM_DISPLAY_MODES[self.rvm_display_mode % len(
                self.RVM_DISPLAY_MODES)]
            if mode == "Alpha":
                frame_out = cv2.cvtColor(pha_np, cv2.COLOR_GRAY2BGR)
            elif mode == "Foreground":
                black_bg = np.zeros_like(frame_out)
                alpha_f = pha_np.astype(np.float32) / 255.0
                alpha_3c = cv2.cvtColor(alpha_f, cv2.COLOR_GRAY2BGR)
                frame_out = (alpha_3c * fgr_np + (1.0 - alpha_3c)
                             * black_bg).astype(np.uint8)
            else:
                if self.rvm_background is None:
                    self._load_rvm_background(self._init_rvm_bg_path)
                bg = self.rvm_background
                if bg.shape[:2] != frame.shape[:2]:
                    bg = resize_frame(bg, (frame.shape[0], frame.shape[1]))
                    self.rvm_background = bg
                alpha_f = pha_np.astype(np.float32) / 255.0
                alpha_3c = cv2.cvtColor(alpha_f, cv2.COLOR_GRAY2BGR)
                frame_out = (alpha_3c * fgr_np + (1.0 - alpha_3c)
                             * bg).astype(np.uint8)
            cv2.putText(frame_out, f"RVM: {model_name} ({mode})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, )
        except Exception as e:
            logger.error(f"Error in RVM composite: {e}")
            error_display_frame = original_frame_fallback.copy()
            cv2.putText(error_display_frame, "RVM Error", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_display_frame
        return frame_out

    # *** CORRECTED SAM Segmentation Style (Removed obsolete _draw_sam_masks call) ***
    def _apply_sam_segmentation_style(self, frame: np.ndarray, original_frame: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Applies SAM-based segmentation, preferring SAM v2 for live video with center point prompt."""
        start_time = time.time()
        frame = validate_frame(frame)
        original_frame_fallback = original_frame if original_frame is not None else frame.copy()

        use_v2 = self.use_sam2_runtime and self.sam_model_v2 is not None
        use_v1 = self.sam_model_v1 is not None

        if not use_v2 and not use_v1:
            logger.warning(
                "SAM Segmentation: No SAM model loaded/available. Skipping.")
            cv2.putText(original_frame_fallback, "SAM UNAVAILABLE",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return original_frame_fallback

        sam_version_name = "SAM v2 (Ultralytics)" if use_v2 else "SAM v1 (AutoMask)"
        sam_model_to_use = self.sam_model_v2 if use_v2 else self.sam_model_v1
        logger.debug(f"Attempting SAM segmentation using {sam_version_name}")

        try:
            h, w = frame.shape[:2]
            combined_mask_resized = None

            if use_v2:
                scale_factor = 0.5
                small_h, small_w = int(h * scale_factor), int(w * scale_factor)
                small_frame = cv2.resize(frame, (small_w, small_h))
                small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                logger.debug(
                    f"SAM v2 using downsampled frame: {small_rgb.shape}")
                center_point = [small_w // 2, small_h // 2]
                logger.debug(
                    f"SAM v2 using center point prompt: {center_point}")
                with torch.no_grad():
                    results = sam_model_to_use(small_rgb, points=[center_point], labels=[
                                               1], device=self.device, conf=SEGMENTATION_THRESHOLD, verbose=False)
                if results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                    masks_tensor = results[0].masks.data
                    logger.debug(
                        f"SAM v2 generated {masks_tensor.shape[0]} masks from center prompt.")
                    combined_mask_small = np.zeros(
                        (small_h, small_w), dtype=np.uint8)
                    for mask in masks_tensor:
                        combined_mask_small = np.maximum(
                            combined_mask_small, mask.cpu().numpy().astype(np.uint8) * 255)
                    combined_mask_resized = cv2.resize(
                        combined_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    logger.debug(
                        "SAM v2 generated no masks from center prompt.")
            elif use_v1:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                logger.debug(f"SAM v1 using full frame: {rgb_frame.shape}")
                with torch.no_grad():
                    masks_list = sam_model_to_use.generate(rgb_frame)
                if masks_list:
                    logger.debug(
                        f"SAM v1 generated {len(masks_list)} raw masks.")
                    combined_mask_full = np.zeros((h, w), dtype=np.uint8)
                    for mask_info in masks_list:
                        mask_np = mask_info['segmentation'].astype(
                            np.uint8) * 255
                        combined_mask_full = np.maximum(
                            combined_mask_full, mask_np) if mask_np.shape == combined_mask_full.shape else combined_mask_full
                    combined_mask_resized = combined_mask_full
                else:
                    logger.debug("SAM v1 generated no masks.")

            if combined_mask_resized is not None and np.any(combined_mask_resized):
                combined_mask_blurred = cv2.GaussianBlur(
                    combined_mask_resized, MASK_BLUR_KERNEL, 0)
                final_mask = (combined_mask_blurred >
                              127).astype(np.uint8) * 255
                mask_3d = final_mask[:, :, np.newaxis] / 255.0
                overlay_color = np.array(SAM_MASK_COLOR, dtype=np.float32)
                frame_float = frame.astype(np.float32)
                masked_frame = frame_float * \
                    (1.0 - mask_3d) + overlay_color * mask_3d * SAM_MASK_ALPHA
                frame_out = np.clip(masked_frame, 0, 255).astype(np.uint8)
                self.sam_masks_cache = final_mask
                # *** OBSOLETE CALL REMOVED HERE ***
                cv2.putText(frame_out, f"{sam_version_name} Seg.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, )
            else:
                cv2.putText(frame, f"{sam_version_name} No Masks", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, )
                frame_out = original_frame_fallback  # Return original if no masks

            elapsed = time.time() - start_time
            logger.debug(
                f"{sam_version_name} segmentation took {elapsed:.3f} seconds")
            return frame_out

        except Exception as e:
            logger.error(
                f"SAM segmentation error ({sam_version_name}): {e}", exc_info=True)
            error_display_frame = original_frame_fallback.copy()
            cv2.putText(error_display_frame, f"{sam_version_name} ERROR", (
                10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, )
            return error_display_frame

    # --- Helper Methods ---

    def _draw_flow_trails_simple(self, frame: np.ndarray, flow_points: np.ndarray, prev_points: np.ndarray, line_color: Tuple[int, int, int], dot_color: Tuple[int, int, int], line_thickness: int = 1, dot_radius: int = 1, ) -> np.ndarray:
        try:
            if flow_points.ndim == 3:
                flow_points = flow_points.reshape(-1, 2)
            if prev_points.ndim == 3:
                prev_points = prev_points.reshape(-1, 2)
            if flow_points.shape[0] != prev_points.shape[0]:
                return frame
            for i in range(flow_points.shape[0]):
                pt1 = (int(prev_points[i, 0]), int(prev_points[i, 1]))
                pt2 = (int(flow_points[i, 0]), int(flow_points[i, 1]))
                cv2.line(frame, pt1, pt2, line_color,
                         line_thickness, cv2.LINE_AA)
                cv2.circle(frame, pt2, dot_radius, dot_color, -1, cv2.LINE_AA)
        except Exception as e:
            logger.warning(f"Draw flow trails failed: {e}", exc_info=False)
        return frame

    def _draw_motion_glow_separate(self, frame: np.ndarray, landmarks: List[Tuple[int, int]], flow_points: np.ndarray, prev_points: np.ndarray, ) -> np.ndarray:
        frame = validate_frame(frame)
        try:
            if flow_points.ndim == 3:
                flow_points = flow_points.reshape(-1, 2)
            if prev_points.ndim == 3:
                prev_points = prev_points.reshape(-1, 2)
            if flow_points.shape[0] != prev_points.shape[0]:
                return frame
            glow_mask = np.zeros(
                (frame.shape[0], frame.shape[1]), dtype=np.uint8)
            for i in range(flow_points.shape[0]):
                pt1 = (int(prev_points[i, 0]), int(prev_points[i, 1]))
                pt2 = (int(flow_points[i, 0]), int(flow_points[i, 1]))
                cv2.line(glow_mask, pt1, pt2, 255, 4, cv2.LINE_AA)
            glow_mask_blurred = cv2.GaussianBlur(glow_mask, (25, 25), 0)
            glow_alpha = (glow_mask_blurred.astype(
                np.float32) / 255.0 * self.glow_intensity)
            glow_alpha_3c = cv2.cvtColor(glow_alpha, cv2.COLOR_GRAY2BGR)
            glow_color_layer = np.full_like(
                frame, (255, 255, 255), dtype=np.float32)
            frame_float = frame.astype(np.float32)
            frame_float = (frame_float * (1.0 - glow_alpha_3c) +
                          glow_color_layer * glow_alpha_3c) # This was part of the old _draw_motion_glow_separate
            frame = np.clip(frame_float, 0, 255).astype(np.uint8) # This was part of the old _draw_motion_glow_separate
        except Exception as e:
            logger.warning(f"Draw motion glow failed: {e}", exc_info=False) # This was part of the old _draw_motion_glow_separate
        return frame # This was part of the old _draw_motion_glow_separate

    # Updated _draw_motion_glow_separate to match video_effects.py
    def _draw_motion_glow_separate(self, frame: np.ndarray, landmarks: List[Tuple[int, int]], flow: Optional[np.ndarray], prev_pts: Optional[np.ndarray]) -> np.ndarray:
        frame = validate_frame(frame)
        num_landmarks = len(landmarks)
        if num_landmarks == 0:
            return frame
        
        magnitudes = np.zeros(num_landmarks)
        try:
            if flow is not None and prev_pts is not None and flow.shape[0] == prev_pts.shape[0] and prev_pts.shape[0] == num_landmarks:
                # Ensure flow and prev_pts are correctly shaped for subtraction if they are (N, 1, 2)
                flow_reshaped = flow.reshape(-1, 2)
                prev_pts_reshaped = prev_pts.reshape(-1, 2)
                magnitudes = np.linalg.norm(flow_reshaped - prev_pts_reshaped, axis=1)
            elif flow is not None and prev_pts is not None and flow.shape[0] > 0: # If landmark count mismatch, use average
                flow_reshaped = flow.reshape(-1, 2)
                prev_pts_reshaped = prev_pts.reshape(-1, 2)
                if flow_reshaped.shape[0] == prev_pts_reshaped.shape[0] and flow_reshaped.shape[0] > 0: # Check if shapes match for avg calc
                    avg_mag = np.mean(np.linalg.norm(flow_reshaped - prev_pts_reshaped, axis=1))
                    magnitudes = np.full(num_landmarks, avg_mag)
                else: # Fallback if shapes still mismatch for avg calc
                    logger.debug("Flow and prev_pts shape mismatch for average magnitude calculation.")
                    # Keep magnitudes as zeros, so no glow if points don't align
        except Exception as e:
            logger.debug(f"Glow magnitude calc failed: {e}")
            # Keep magnitudes as zeros

        h, w = frame.shape[:2]
        try:
            for i, (x, y) in enumerate(landmarks):
                if i < len(magnitudes): # Ensure index is within bounds for magnitudes
                    glow_intensity = min(255, int(magnitudes[i] * 15)) # As per video_effects.py
                    glow_color = (glow_intensity // 3, glow_intensity // 2, glow_intensity) # As per video_effects.py
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 5, glow_color, -1, cv2.LINE_AA)
                else: # Should not happen if logic is correct, but as a safeguard
                    logger.debug(f"Landmark index {i} out of bounds for magnitudes array (len: {len(magnitudes)})")

        except Exception as e:
            logger.warning(f"Draw motion glow failed: {e}")
            return frame
        return frame

    # Updated _apply_distortion to match video_effects.py
    def _apply_distortion(self, frame: np.ndarray) -> np.ndarray:
        """Applies a horizontal sine wave distortion effect to the frame."""
        frame = validate_frame(frame)
        try:
            rows, cols, _ = frame.shape
            map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
            distort_x = np.sin(map_y / 20.0) * 5.0 # Sine wave distortion
            map_x = np.clip(map_x + distort_x, 0, cols - 1).astype(np.float32)
            map_y = map_y.astype(np.float32) # Y map remains unchanged for this specific distortion
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logger.warning(f"Distortion effect failed: {e}")
            return frame

    def _detect_gestures(self, frame: np.ndarray) -> Optional[str]:
        """Detects simple hand gestures (fist, open hand) using Mediapipe Hands."""
        if not self.hands or not self.frame_width or not self.frame_height:
            return None
        gesture = None
        try:
            frame.flags.writeable = False
            results = self.hands.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.flags.writeable = True
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                tips = [lm[i] for i in [mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                                        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.PINKY_TIP]]
                pips = [lm[i] for i in [mp.solutions.hands.HandLandmark.THUMB_IP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
                                        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, mp.solutions.hands.HandLandmark.RING_FINGER_PIP, mp.solutions.hands.HandLandmark.PINKY_PIP]]
                fingers_extended = sum(1 for i in range(
                    1, 5) if tips[i].y < pips[i].y) + (1 if tips[0].y < pips[0].y else 0)
                fingers_closed = sum(1 for i in range(
                    1, 5) if tips[i].y > pips[i].y) + (1 if tips[0].y > pips[0].y else 0)
                if fingers_extended >= 4:
                    gesture = "open_hand"
                elif fingers_closed >= 4:
                    gesture = "fist"
        except Exception as e:
            logger.warning(f"Gesture detection failed: {e}")
            gesture = None
        return gesture

    def _load_rvm_background(self, bg_path: Optional[str]) -> None:
        """Loads or sets the background for RVM, resizing if necessary."""
        needs_resize = False
        if self.rvm_background is not None:
            if self.rvm_background.shape[:2] != (self.frame_height, self.frame_width):
                needs_resize = True
            if bg_path == self._init_rvm_bg_path and not needs_resize:
                return
        logger.info(
            f"Loading/Updating RVM background. Path: {bg_path}. Resize needed: {needs_resize}")
        loaded_bg = None
        if bg_path and os.path.exists(bg_path):
            try:
                loaded_bg = cv2.imread(bg_path)
                loaded_bg = validate_frame(
                    loaded_bg) if loaded_bg is not None else None
                if loaded_bg is not None:
                    logger.info(f"Loaded RVM background image from {bg_path}")
                else:
                    logger.warning(
                        f"cv2.imread failed for RVM background: {bg_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load RVM background image '{bg_path}': {e}")
                loaded_bg = None
        if loaded_bg is None:
            h, w = self.frame_height, self.frame_width
            self.rvm_background = np.full(
                (h, w, 3), RVM_DEFAULT_BG_COLOR, dtype=np.uint8)
            logger.info(
                f"Using default {RVM_DEFAULT_BG_COLOR} RVM background ({w}x{h})")
            self._init_rvm_bg_path = None
        else:
            self.rvm_background = resize_frame(
                loaded_bg, (self.frame_height, self.frame_width))
            self._init_rvm_bg_path = bg_path

    def _preprocess_frame_rvm(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Prepares a frame for RVM model input."""
        try:
            frame = validate_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return T.ToTensor()(frame_rgb).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.warning(f"RVM preprocess failed: {e}")
            return None

    def _postprocess_output_rvm(self, fgr: torch.Tensor, pha: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Converts RVM model output tensors to NumPy arrays."""
        try:
            fgr_np = fgr.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
            pha_np = pha.cpu().detach().squeeze(0).squeeze(0).numpy()
            fgr_out = np.clip(fgr_np * 255.0, 0, 255).astype(np.uint8)
            pha_out = np.clip(pha_np * 255.0, 0, 255).astype(np.uint8)
            # Return BGR foreground
            return cv2.cvtColor(fgr_out, cv2.COLOR_RGB2BGR), pha_out
        except Exception as e:
            logger.warning(f"RVM postprocess failed: {e}")
            return None, None

    def _draw_sam_masks_unified(self, frame: np.ndarray, masks: Any, original_shape: Tuple[int, int], downsampled_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Draws segmentation masks from SAM v1 or v2 onto the frame."""
        # Note: This function is currently NOT called by the corrected _apply_sam_segmentation_style
        # It remains here for potential future use or alternative drawing methods.
        overlay = frame.copy()
        h_orig, w_orig = original_shape
        num_masks_drawn = 0
        masks_to_process = []
        if isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], np.ndarray):
            masks_to_process = masks
            needs_resize = False
        elif isinstance(masks, torch.Tensor):
            if masks.ndim == 3:
                masks_to_process = [m.cpu().numpy() for m in masks]
            elif masks.ndim == 2:
                masks_to_process = [masks.cpu().numpy()]
            else:
                logger.warning(f"Unsupported SAM tensor shape: {masks.shape}")
                return frame
            needs_resize = (downsampled_shape is not None)
        else:
            logger.warning(f"Unsupported SAM mask type: {type(masks)}")
            return frame
        if not masks_to_process:
            return frame
        for i, mask_data in enumerate(masks_to_process):
            if not isinstance(mask_data, np.ndarray) or mask_data.ndim != 2 or mask_data.size == 0:
                continue
            try:
                mask_bool = mask_data.astype(bool)
                if needs_resize and mask_bool.shape != (h_orig, w_orig):
                    mask_bool = cv2.resize(mask_bool.astype(
                        np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST).astype(bool)
                if mask_bool.shape != (h_orig, w_orig) or not np.any(mask_bool):
                    continue
                color = [random.randint(64, 200) for _ in range(3)]
                overlay[mask_bool] = color
                num_masks_drawn += 1
            except Exception as draw_e:
                logger.warning(
                    f"Failed to draw SAM mask {i}: {draw_e}", exc_info=False)
        if num_masks_drawn > 0:
            frame_out = cv2.addWeighted(
                frame, 1.0 - SAM_MASK_ALPHA, overlay, SAM_MASK_ALPHA, 0)
        else:
            frame_out = frame
        return frame_out

    # --- Video Capture and Processing Loop ---

    # *** CORRECTED _initialize_capture ***
    def _initialize_capture(self) -> bool:
        """Initializes the video capture source and optional writer/display."""
        logger.info(f"Initializing capture from: {self.input_source}")
        try:  # <<< MOVED to its own line
            self.cap = cv2.VideoCapture(self.input_source)
            if not self.cap.isOpened():
                logger.error(
                    f"Failed to open input source: {self.input_source}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            if self.frame_width <= 0 or self.frame_height <= 0:
                logger.error(
                    f"Capture returned invalid dimensions: {self.frame_width}x{self.frame_height}. Cannot proceed.")
                self.cap.release()
                return False
            if self.fps <= 0:
                logger.warning(
                    f"Invalid source FPS ({self.fps}). Using default: {DEFAULT_FPS}")
                self.fps = DEFAULT_FPS

            logger.info(
                f"Capture opened: Actual {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS (Requested {self.desired_width}x{self.desired_height} @ {self.desired_fps:.2f} FPS)")

            ret, frame = self.cap.read()
            assert ret and frame is not None, "Failed to read initial frame."
            logger.info("Initial frame read ok.")

            if self.mode == "record":
                output_dir = os.path.dirname(self.output_path)
                os.makedirs(output_dir, exist_ok=True)
                try:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.out = cv2.VideoWriter(
                        self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height), )
                    assert self.out.isOpened(
                    ), f"Failed video writer init: {self.output_path}"
                    logger.info(f"Recording enabled: {self.output_path}")
                except Exception as writer_e:
                    logger.error(
                        f"Failed to create video writer '{self.output_path}': {writer_e}. Recording disabled.")
                    self.mode = "live"  # Fallback

            if self.display:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                cv2.createTrackbar("Brightness", self.window_name, 100, 200, lambda x: self.bc_params.update({'brightness': (x - 100) / 100.0}))
                cv2.createTrackbar("Contrast", self.window_name, 100, 200, lambda x: self.bc_params.update({'contrast': x / 100.0}))
                cv2.createTrackbar("Gamma", self.window_name, 33, 100, lambda x: self.bc_params.update({'gamma': x / 33.0 if x > 0 else 0.1})) # Map 33 to 1.0 gamma
                
                cv2.createTrackbar("Sat. Scale", self.window_name, 100, 200, lambda x: self.saturation_params.update({'scale': x / 100.0}))
                cv2.createTrackbar("Vibrance", self.window_name, 0, 100, lambda x: self.saturation_params.update({'vibrance': x / 100.0}))
                
                cv2.createTrackbar("Vign. Strength", self.window_name, 30, 100, lambda x: self.vignette_params.update({'strength': x / 100.0}))
                cv2.createTrackbar("Vign. Radius", self.window_name, 70, 100, lambda x: self.vignette_params.update({'radius': x / 100.0}))
                # Vignette falloff, shape, color deferred for trackbar simplicity

                # Kernel size must be odd. Trackbar range 3-51, step 2. Default 5.
                cv2.createTrackbar("Blur Kernel", self.window_name, 2, 24, lambda x: self.blur_params.update({'kernel_size': x * 2 + 1})) # (0*2+1=1 -> bad, 1*2+1=3, 2*2+1=5 ... 24*2+1=49)
                                                                                                                                         # Correcting lambda for blur_kernel to ensure it starts at 3
                cv2.setTrackbarMin("Blur Kernel", self.window_name, 1) # Min val 1 means 1*2+1 = 3
                cv2.setTrackbarPos("Blur Kernel", self.window_name, 2) # Initial position for kernel size 5 (2*2+1)


                cv2.createTrackbar("LUT Intensity", self.window_name, 100, 100, lambda x: setattr(self, "lut_intensity", x / 100.0))
                logger.info("Display window initialized with all trackbars.")
            return True
        except Exception as e:
            logger.error(f"Capture init failed: {e}", exc_info=True)
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if self.out and self.out.isOpened():
                self.out.release()
            return False

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources...")
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                logger.info("Capture released.")
            if self.out and self.out.isOpened():
                self.out.release()
                logger.info("Writer released.")
            if _mediapipe_available:
                if hasattr(self, "face_mesh") and self.face_mesh:
                    self.face_mesh.close()
                if hasattr(self, "pose") and self.pose:
                    self.pose.close()
                if hasattr(self, "hands") and self.hands:
                    self.hands.close()
                logger.info("Mediapipe resources closed.")
            if self.display:
                cv2.destroyAllWindows()
                logger.info("Windows destroyed.")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}", exc_info=True)

    @staticmethod
    def list_cameras() -> List[Tuple[int, str]]:
        available_cameras = []
        logger.info("Scanning for cameras (0-9)...")
        for index in range(10):
            cap_test = cv2.VideoCapture(index)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret:
                    name = f"Camera {index} ({cap_test.getBackendName()})"
                    available_cameras.append((index, name))
                    logger.info(f" Found: {name}")
                cap_test.release()
        if not available_cameras:
            logger.warning("No cameras detected.")
        return available_cameras

    def _handle_gestures(self, frame: np.ndarray, frame_time: float) -> None:
        """Detects gestures and triggers corresponding actions."""
        gesture = self._detect_gestures(frame)
        if not gesture:
            return
        current_time = time.time()
        gesture_cooldown = 1.5
        if (self.gesture_state["last_gesture"] == gesture and
                current_time - self.gesture_state["last_time"] < gesture_cooldown):
            return
        logger.info(f"Gesture '{gesture}' detected. Triggering action.")
        self.gesture_state["last_gesture"] = gesture
        self.gesture_state["last_time"] = current_time
        self.gesture_state["last_gesture_count"] += 1
        if gesture == "open_hand":
            effect_keys = list(self.effects.keys())
            try:
                current_idx = effect_keys.index(self.current_effect)
                next_idx = (current_idx + 1) % len(effect_keys)
                self.current_effect = effect_keys[next_idx]
                logger.info(
                    f"Gesture 'open_hand': Switched to effect -> {self.current_effect}")
                self._reset_effect_states()
            except ValueError:
                self.current_effect = effect_keys[0] if effect_keys else "none"
                logger.warning(
                    f"Effect error, reset to {self.current_effect}.")
        elif gesture == "fist":
            logger.info(
                f"Gesture 'fist': Cycling variant for '{self.current_effect}'...")
            if self.current_effect == "goldenaura":
                self.goldenaura_variant = (
                    self.goldenaura_variant + 1) % len(self.goldenaura_variant_names)
                logger.info(
                    f" -> Switched Aura to {self.goldenaura_variant_names[self.goldenaura_variant]}")
            elif self.current_effect == "lightning":
                self.lightning_style_index = (
                    self.lightning_style_index + 1) % len(self.lightning_styles)
                logger.info(
                    f" -> Switched Lightning to {self.lightning_style_names[self.lightning_style_index]}")
            elif (self.current_effect == "lut_color_grade" and self.lut_color_grade_effect):
                self.lut_color_grade_effect.cycle_lut()
            elif self.current_effect == "rvm_composite":
                self.rvm_display_mode = (
                    self.rvm_display_mode + 1) % len(self.RVM_DISPLAY_MODES)
                logger.info(
                    f" -> Switched RVM mode to {self.RVM_DISPLAY_MODES[self.rvm_display_mode]}")
                self.rvm_rec = [None] * 4
            elif self.current_effect == "sam_segmentation":
                sam1_rdy = self.sam_model_v1 is not None
                sam2_rdy = self.sam_model_v2 is not None
                if sam1_rdy and sam2_rdy:
                    self.use_sam2_runtime = not self.use_sam2_runtime
                    logger.info(
                        f" -> Toggled SAM pref to {'SAM v2' if self.use_sam2_runtime else 'SAM v1'}")
                else:
                    logger.info(" -> Cannot switch SAM (only one loaded).")
            else:
                logger.info(
                    f" -> No variant action for '{self.current_effect}'.")

    def _reset_effect_states(self):
        logger.debug("Resetting common effect states.")
        self.rvm_rec = [None] * 4
        self.face_history.clear()
        self.pose_history.clear()
        self.frame_buffer.clear()
        self.sam_masks_cache = None
        self.prev_gray = None
        self.prev_landmarks_flow = None
        self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
        self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)

    def run(self) -> None:
        """Main video processing loop."""
        main_start_time = time.time()
        frame_count = 0
        total_time = 0.0
        if not self._initialize_capture():
            logger.error("Capture init failed. Exiting.")
            return
        if self.rvm_available:
            [self.rvm_models.update({name: model}) for name in self.rvm_model_names if (
                model := load_rvm_model(self.device, name))]

        try:
            while True:
                loop_start_time = time.time()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.info("End of stream or read error.")
                    break
                frame = validate_frame(frame)
                target_h, target_w = self.frame_height, self.frame_width
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = resize_frame(frame, (target_h, target_w))

                original_frame = frame.copy()
                self._handle_gestures(original_frame, loop_start_time)

                if self.current_effect not in self.effects:
                    self.current_effect = "none"
                    logger.warning("Invalid effect, reset to 'none'.")
                effect_wrapper = self.effects[self.current_effect]
                effect_kwargs = {"frame_time": loop_start_time,
                             "runner": self, "original_frame": original_frame}

                # Apply effect with error handling
                try:
                    processed_frame = effect_wrapper(frame=frame, **effect_kwargs)
                    processed_frame = validate_frame(processed_frame)
                except ValueError as ve:
                    logger.error(f"ValueError in effect processing: {ve}")
                    continue  # Skip to next frame

                if self.brightness_factor != 0.0 or self.contrast_factor != 1.0:
                    processed_frame = cv2.convertScaleAbs(
                        processed_frame, alpha=self.contrast_factor, beta=self.brightness_factor * 100)

                frame_duration = time.time() - loop_start_time
                self.frame_times.append(frame_duration)
                avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if len(
                    self.frame_times) > 1 else (1.0 / frame_duration if frame_duration > 0 else 0)
                fps_text = f"FPS: {avg_fps:.1f}"
                cv2.putText(processed_frame, fps_text, (10, target_h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.out:
                    self.out.write(processed_frame)
                if self.display:
                    cv2.imshow(self.window_name, processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in [ord("e"), ord("w")]:
                    direction = 1 if key == ord("e") else -1
                    effect_keys = list(self.effects.keys())
                    if effect_keys:
                        try:
                            current_idx = effect_keys.index(self.current_effect)
                            next_idx = (current_idx + direction + len(effect_keys)) % len(effect_keys)
                            self.current_effect = effect_keys[next_idx]
                            logger.info(f"Key '{chr(key)}': Switched effect -> {self.current_effect}")
                            self._reset_effect_states()
                        except ValueError:
                            self.current_effect = effect_keys[0]
                            logger.warning("Effect error, reset.")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Exiting loop.")
        except Exception as e:
            logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
        finally:
            total_time = time.time() - main_start_time
            self._cleanup()
            if frame_count > 0 and total_time > 0:
                avg_fps_overall = frame_count / total_time
                logger.info(f"Processed {frame_count} frames in {total_time:.2f}s, Avg FPS: {avg_fps_overall:.2f}")
# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PixelSensing Video Effects Runner - Apply real-time effects to video.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
============================== PixelSensingFX Controls Manual ==============================
General: Q(Quit), E(Next Effect), W(Prev Effect), R(Reset All)
Variants (V): Cycles modes (Aura, Lightning, LUT, RVM Mode, SAM Pref).
Models (M): Cycles RVM model.
Trackbars: Brightness, Contrast, LUT Intensity (0-100%).
See config.json or --help for config options. Check logs/ for details.
============================================================================================
""")
    parser.add_argument("--config", type=str,
                        default=CONFIG_FILE, help="Path to config JSON.")
    parser.add_argument("--input-source", type=str,
                        help="Override: Camera index or video path.")
    parser.add_argument("--output-path", type=str,
                        help="Override: Output video path.")
    parser.add_argument("--lut-dir", type=str,
                        help="Override: LUT directory path.")
    parser.add_argument("--rvm-background-path", type=str,
                        help="Override: RVM background image path.")
    parser.add_argument("--sam-checkpoint-path", type=str,
                        help="Override: SAM v1 checkpoint path (.pth).")
    parser.add_argument("--sam2-checkpoint-path", type=str,
                        help="Override: SAM v2 checkpoint path (.pt).")
    parser.add_argument(
        "--mode", type=str, choices=["live", "record"], help="Override: 'live' or 'record' mode.")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction,
                        default=None, help="Override: --display/--no-display.")
    parser.add_argument("--device", type=str, choices=[
                        "cpu", "cuda", "mps"], help="Override: ML device (cpu, cuda, mps).")
    parser.add_argument("--trail-length", type=int,
                        help="Override: Trail effect length (frames).")
    parser.add_argument("--glow-intensity", type=float,
                        help="Override: Glow effect intensity (0.0-1.0).")
    parser.add_argument("--width", type=int, help="Override: Video width.")
    parser.add_argument("--height", type=int, help="Override: Video height.")
    parser.add_argument("--fps", type=float, help="Override: Video FPS.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List cameras and exit.")
    args = parser.parse_args()

    if args.list_cameras:
        PixelSensingEffectRunner.list_cameras()
        exit()

    logger.info("Initializing PixelSensingEffectRunner...")
    runner = PixelSensingEffectRunner(config_path=args.config)
    config_changed = False
    overridden_keys = []
    for key, value in vars(args).items():
        is_provided = False
        if isinstance(parser.get_default(key), bool) or isinstance(value, bool):
            if value is not None:
                is_provided = True
        elif value is not None:
            default_val = parser.get_default(key)
            is_provided = (
                value != default_val) if default_val is not None else True
        if is_provided and key in runner.config and key not in ["config", "list_cameras"] and value != runner.config.get(key):
            logger.info(f"CLI override: '{key}' = '{value}'")
            runner.config[key] = value
            overridden_keys.append(key)
            config_changed = True

    if config_changed:
        logger.info(
            f"Re-applying config due to overrides: {', '.join(overridden_keys)}")
        runner._apply_config()
        if "lut_dir" in overridden_keys:
            logger.info("Reloading LUTs...")
            runner._load_luts()
            runner._initialize_effects()
        model_keys = {"device", "sam_checkpoint_path",
                      "sam2_checkpoint_path", "rvm_background_path"}
        if any(key in overridden_keys for key in model_keys):
            logger.info("Reloading relevant models due to CLI change...")
            runner.rvm_models = {}
            runner.sam_model_v1 = None
            runner.sam_model_v2 = None
            runner.rvm_background = None
            runner._load_sam_models()
            if runner.rvm_available:
                [runner.rvm_models.update({name: model}) for name in runner.rvm_model_names if (
                    model := load_rvm_model(runner.device, name))]
            runner._load_rvm_background(
                runner.config.get("rvm_background_path"))
            runner._initialize_effects()

    logger.info("Starting main processing loop...")
    runner.run()
    logger.info("Application finished.")

# Corrected Example command:
# python video_effectsfinale5.py --lut-dir "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/Videous/70 CGC LUTs" --device mps --mode live --display --input-source 0 --sam2-checkpoint-path "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/sam2.1_t.pt"
# (Ensure the sam2.1_t.pt file exists at the specified path)
# python video_effectsfinale5.py --lut-dir "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/Videous/70 CGC LUTs" --device cpu --mode live --display --input-source 0 --sam2-checkpoint-path "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/sam2.1_t.pt"