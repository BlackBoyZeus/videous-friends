

# -*- coding: utf-8 -*-
# ========================================================================
#      Unified Autonomous Video Editor - Deep Integration v2.0
# ========================================================================
# This script integrates multiple advanced concepts into a single, comprehensive
# autonomous video editing workflow.
# - Planners: MathematicalGenius (SDE/OT/DRL) & ParetoRoulette
# - Effects: Integrates PixelSensingEffectRunner style effects.
# - Config: Uses unified AnalysisConfig & RenderConfig.
# - Rendering: Employs MoviePy's make_frame with dynamic effect application.
# ========================================================================

# ------------------------------------------------------------------------
#                       IMPORTS (Massive & Combined)
# ------------------------------------------------------------------------
import tkinter # For fallback messages only
from tkinter import messagebox
import cv2
import math
import numpy as np
import time
import os
import json
import threading
# import concurrent.futures # Removing parallel processing for single script simplicity for now
import traceback
import sys
import shlex
import random
from collections import defaultdict, namedtuple, deque
from math import exp, log, sqrt, cos, sin, pi
import subprocess
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set, Union, Callable
import abc
import tracemalloc

# --- Core Analysis & Editing ---
import mediapipe as mp
import librosa
import soundfile
from scipy import signal as scipy_signal, stats as scipy_stats, spatial, interpolate

# --- MoviePy Imports ---
from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip
from moviepy.video.fx.all import resize as moviepy_resize # Example fx

# --- Advanced Algorithms ---
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

# Optimal Transport (requires POT library: pip install pot)
try:
    from ot import sinkhorn
    _pot_available = True
except ImportError:
    logger.warning("Python Optimal Transport (POT) library not found. Optimal Transport planner disabled. `pip install POT`")
    _pot_available = False
    sinkhorn = None

# --- Effects Dependencies (Conditional & Logging) ---
_sam1_available = False
_sam2_available = False
_rvm_available = False # Will be checked in loader

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _sam1_available = True
    logger.info("Segment Anything (SAM v1) library found.")
except ImportError:
    logger.warning("Segment Anything (SAM v1) library not found. SAM v1 effects disabled.")
    SamAutomaticMaskGenerator = None; sam_model_registry = None

try:
    from ultralytics import SAM as UltralyticsSAM
    # Check if it's actually the SAM model class, not a generic SAM object if Ultralytics changes API
    if isinstance(UltralyticsSAM, type):
         _sam2_available = True
         logger.info("Ultralytics SAM (for SAM 2) library found.")
    else:
         logger.warning("Ultralytics SAM found, but import is not the expected class type. SAM 2 effects might be disabled.")
         UltralyticsSAM = None
except ImportError:
    logger.warning("Ultralytics library not found or SAM class unavailable. SAM 2 effects disabled.")
    UltralyticsSAM = None

# RVM check is deferred to its loader function

# --- MediaPipe Solutions ---
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    _mediapipe_available = True
    logger.info("MediaPipe libraries loaded.")
except ImportError:
    logger.error("MediaPipe library not found. Face/Pose features disabled. `pip install mediapipe`")
    _mediapipe_available = False
    mp_face_mesh = None; mp_pose = None; mp_drawing = None; mp_drawing_styles = None


# --- Setup & Constants ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s [%(threadName)s]")
logger = logging.getLogger("AutonomousEditor")

DEFAULT_ANALYSIS_HEIGHT, DEFAULT_ANALYSIS_WIDTH = 256, 256
DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH = 720, 1280

# --- Device Setup ---
_torch_device = 'cpu'
if torch.cuda.is_available():
    _torch_device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and sys.platform == "darwin":
    try:
        # Simple check if MPS works without causing immediate issues
        tensor_mps = torch.tensor([1.0], device='mps')
        _ = tensor_mps * tensor_mps
        _torch_device = 'mps'
        logger.info("--- MPS device detected and verified (Apple Silicon). ---")
    except Exception as mps_error:
        logger.warning(f"--- MPS device detected but check failed: {mps_error}. Falling back to CPU. ---")
        _torch_device = 'cpu'
logger.info(f"Selected PyTorch device: {_torch_device}")


# ------------------------------------------------------------------------
#               CONFIGURATION DATACLASSES (Comprehensive)
# ------------------------------------------------------------------------
@dataclass
class EffectParamsConfig:
    """ Parameters specific to an effect instance in the plan """
    intensity: float = 1.0 # General intensity factor (0 to 1+)
    # Effect-specific parameters are stored in a dict
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisConfig:
    """ Unified configuration for Analysis & Planning """
    # General
    planner_type: str = "MathematicalGenius" # "MathematicalGenius" or "ParetoRoulette"
    analysis_resolution_height: int = DEFAULT_ANALYSIS_HEIGHT
    analysis_resolution_width: int = DEFAULT_ANALYSIS_WIDTH
    output_dir: str = "output_autonomous"
    save_analysis_data: bool = False # Option to save frame features

    # Analysis Features
    min_face_confidence: float = 0.5
    min_pose_confidence: float = 0.5
    pose_model_complexity: int = 1 # MediaPipe Pose model complexity (0, 1, 2)
    motion_method: str = 'farneback' # 'farneback' or 'lk' (LK needs features)
    lk_max_corners: int = 100
    lk_quality_level: float = 0.1
    lk_min_distance: int = 7

    # SDE/OT/DRL Planner ('MathematicalGenius')
    sde_alpha: float = 0.6 # Weight for motion vs beats in SDE drift
    sde_beta: float = 0.3 # Volatility scaling factor in SDE diffusion
    sde_threshold: float = 1.1 # Threshold for interval selection from SDE process
    sde_min_interval_sec: float = 0.4 # Minimum duration for an SDE-selected interval
    ot_enabled: bool = _pot_available # Enable OT only if library is available
    ot_regularization: float = 0.01 # Sinkhorn regularization parameter
    ot_cost_motion_weight: float = 1.0 # How much motion influences OT cost
    ot_cost_complexity_bias: float = 0.2 # Added cost bias for complex effects
    drl_enabled: bool = True # Enable DRL refinement step (uses simplified agent)
    drl_state_dim: int = 5 # Example state: [norm_avg_motion, norm_beat_prox, time_progress, last_effect_idx, num_effects_used]
    drl_action_dim: int = 0 # Set dynamically based on available effects
    drl_hidden_dim: int = 64
    drl_gamma: float = 0.95 # Discount factor
    drl_learning_rate: float = 0.0005
    drl_reward_weights: Tuple[float, float, float] = (0.5, 0.2, 0.3) # visual_impact, sync_penalty, creativity_bonus
    drl_mock_train_epochs: int = 10 # Simplified refinement loop iterations

    # Pareto/Roulette Planner
    pareto_num_candidates: int = 75 # Number of edit plans to generate for Pareto front
    pareto_min_interval_duration: float = 0.6 # Min duration for stochastically pooled intervals
    pareto_max_interval_duration: float = 2.5 # Max duration
    pareto_motion_weight: float = 0.6 # Weight for motion in stochastic interval pooling
    pareto_beat_weight: float = 0.4 # Weight for beat proximity in pooling
    # Pareto objective weights (used for final random selection from front, if weighting desired)
    pareto_select_weight_visual: float = 1.0
    pareto_select_weight_sync: float = 1.0
    pareto_select_weight_creativity: float = 1.0

@dataclass
class RenderConfig:
    """ Configuration for Rendering """
    resolution_width: int = DEFAULT_RENDER_WIDTH
    resolution_height: int = DEFAULT_RENDER_HEIGHT
    fps: int = 30
    video_codec: str = 'libx264'
    preset: str = 'medium' # 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'
    crf: int = 23 # Constant Rate Factor (0-51, lower is better quality, 18-28 is typical range)
    audio_codec: str = 'aac'
    audio_bitrate: str = '192k'
    threads: int = max(1, (os.cpu_count() or 2) - 1) # Leave one core free
    # Default parameters for effects (can be overridden by planner's EffectParamsConfig)
    default_effect_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "brightness_contrast": {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0},
        "saturation": {"scale": 1.0, "vibrance": 0.0},
        "vignette": {"strength": 0.4, "radius": 0.7, "falloff": 0.3, "shape": 'elliptical', "color": (0,0,0)},
        "blur": {"kernel_size": 5, "sigma": 0.0},
        "lightning": {"movement_threshold": 4.0, "glow_alpha": 0.65, "flicker_probability": 0.8, "base_segments": 7, "angle_range": 45, "segment_length": 20},
        "rvm_composite": {"downsample_ratio": 0.4, "display_mode": 0}, # display_mode: 0=Composite, 1=Alpha, 2=Foreground
        "sam1_segmentation": {"alpha": 0.5},
        "sam2_segmentation": {"alpha": 0.5},
        "crossfade": {"duration": 0.5},
        "slide": {"duration": 0.5, "direction": "left"},
        "none": {}
    })
    # Paths for models (can be set externally)
    rvm_resnet50_path: Optional[str] = None # Or URL if using torch.hub directly without explicit path
    rvm_mobilenet_path: Optional[str] = None
    sam_checkpoint_path: Optional[str] = None # Path to SAM checkpoint (.pth)
    sam_model_type: str = 'vit_b' # 'vit_b', 'vit_l', 'vit_h', 'vit_t' (for SAM v1)
    use_sam2: bool = False # Prefer SAM2 if available and checkpoint provided?

# ------------------------------------------------------------------------
#                       HELPER FUNCTIONS
# ------------------------------------------------------------------------
def validate_frame(frame: Optional[np.ndarray], default_shape: Tuple[int, int, int] = (DEFAULT_ANALYSIS_HEIGHT, DEFAULT_ANALYSIS_WIDTH, 3)) -> np.ndarray:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return np.zeros(default_shape, dtype=np.uint8)
    if len(frame.shape) < 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
        logger.warning(f"Frame with invalid dimensions detected {frame.shape}, returning default {default_shape}")
        return np.zeros(default_shape, dtype=np.uint8)
    return frame

def resize_frame(frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    frame = validate_frame(frame, default_shape=(*target_shape, 3)) # Use target shape for default if needed
    current_height, current_width = frame.shape[:2]
    if target_height <= 0 or target_width <= 0: return frame
    if current_height != target_height or current_width != target_width:
        try:
            # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
            inter = cv2.INTER_AREA if target_width < current_width or target_height < current_height else cv2.INTER_LINEAR
            return cv2.resize(frame, (target_width, target_height), interpolation=inter)
        except cv2.error as e:
            logger.error(f"OpenCV resize failed from {(current_height, current_width)} to {target_shape}: {e}")
            # Return a blank frame of the target size on error
            return np.zeros((*target_shape, 3), dtype=np.uint8)
    return frame

def sigmoid(x, k=1):
    try: x_clamped = np.clip(x, -700, 700); return 1 / (1 + np.exp(-k * x_clamped))
    except OverflowError: logger.warning(f"Sigmoid overflow for {x}. Clamping."); return 0.0 if x < 0 else 1.0

# ------------------------------------------------------------------------
#                   VIDEO & AUDIO ANALYSIS (Detailed)
# ------------------------------------------------------------------------
class BasicAudioUtils:
    """ Simplified audio utilities """
    def extract_audio(self, video_path, audio_output_path="temp_audio.wav"):
        logger.info(f"Extracting audio from '{os.path.basename(video_path)}'...")
        try:
            command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -y "{audio_output_path}" -hide_banner -loglevel error'
            logger.debug(f"Running FFmpeg: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False, encoding='utf-8', timeout=60)
            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction failed (Code {result.returncode}):\n{result.stderr}")
                return None
            if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) < 100: # Check size
                 logger.error(f"FFmpeg ran but output file invalid: {audio_output_path}\nStderr:{result.stderr}")
                 return None
            logger.info("Audio extracted successfully.")
            return audio_output_path
        except subprocess.TimeoutExpired:
             logger.error("FFmpeg audio extraction timed out.")
             return None
        except Exception as e:
            logger.error(f"Error during FFmpeg audio extraction: {e}", exc_info=True)
            return None

    def analyze_audio(self, audio_path: str, target_sr: int = 22050):
        logger.info(f"Analyzing audio features: {os.path.basename(audio_path)}")
        try:
            y, sr = librosa.load(audio_path, sr=target_sr)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration <= 0:
                logger.error("Audio file has zero duration.")
                return None

            # Tempo and Beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            # Onset Strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_times = librosa.times_like(onset_env, sr=sr)

            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            rms_times = librosa.times_like(rms, sr=sr)

            logger.info(f"Audio analysis complete. Duration: {duration:.2f}s, Tempo: {tempo:.2f} BPM, Beats: {len(beat_times)}")
            # Return data structured for easy access
            return {
                "duration": float(duration),
                "beat_times": beat_times.tolist(),
                "onset_env": onset_env.tolist(),
                "onset_times": onset_times.tolist(),
                "rms": rms.tolist(),
                "rms_times": rms_times.tolist(),
                "sr": sr
            }
        except Exception as e:
            logger.error(f"Error during Librosa audio analysis: {e}", exc_info=True)
            return None

def analyze_video_content(video_path: str, fps: float, config: AnalysisConfig) -> List[Dict]:
    """ Analyzes video for motion, landmarks, etc., frame by frame """
    logger.info(f"Analyzing content for: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}"); return []

    target_h, target_w = config.analysis_resolution_height, config.analysis_resolution_width
    all_frame_features = []
    prev_gray = None
    frame_idx = 0
    lk_prev_points = None

    # Initialize MediaPipe components if available
    face_mesh = None
    pose = None
    if _mediapipe_available:
        try:
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True, # Only track one face for simplicity
                min_detection_confidence=config.min_face_confidence,
                min_tracking_confidence=0.5)
            pose = mp_pose.Pose(
                model_complexity=config.pose_model_complexity,
                min_detection_confidence=config.min_pose_confidence,
                min_tracking_confidence=0.5,
                enable_segmentation=True) # Keep segmentation for potential future use
            logger.info("MediaPipe components initialized for analysis.")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe for analysis: {e}")
            face_mesh = None; pose = None

    logger.info(f"Starting frame-by-frame analysis (Resolution: {target_w}x{target_h})...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_time = frame_idx / fps if fps > 0 else 0.0
            features = {'frame_idx': frame_idx, 'time': frame_time}

            frame_resized = resize_frame(frame, (target_h, target_w))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            motion_mag = 0.0

            # Motion Analysis
            if prev_gray is not None:
                try:
                    if config.motion_method == 'farneback':
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        if flow is not None: motion_mag = np.mean(np.linalg.norm(flow, axis=2))
                    elif config.motion_method == 'lk' and _mediapipe_available: # Use LK on tracked points
                        if lk_prev_points is not None and lk_prev_points.shape[0] > 0:
                            lk_next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, lk_prev_points, None)
                            if lk_next_points is not None and status is not None:
                                good_new = lk_next_points[status.flatten() == 1]
                                good_old = lk_prev_points[status.flatten() == 1]
                                if good_new.shape[0] > 0:
                                    displacements = np.linalg.norm(good_new - good_old, axis=1)
                                    motion_mag = np.mean(displacements) * fps # Motion per second approx
                                lk_prev_points = good_new.reshape(-1, 1, 2) # Update points for next frame
                            else: lk_prev_points = None
                        else: # Find new points if none exist
                            corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=config.lk_max_corners,
                                                               qualityLevel=config.lk_quality_level,
                                                               minDistance=config.lk_min_distance, blockSize=7)
                            lk_prev_points = corners if corners is not None else None

                except Exception as e: logger.warning(f"Motion analysis failed frame {frame_idx}: {e}")
            features['motion'] = motion_mag

            # MediaPipe Landmarks
            face_landmarks = None
            pose_landmarks = None
            pose_segmentation = None
            landmarks_for_lk = []
            if _mediapipe_available:
                try:
                    # Process with MediaPipe (RGB needed)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False # Optimization
                    if face_mesh:
                         results_face = face_mesh.process(frame_rgb)
                         if results_face.multi_face_landmarks:
                             face_landmarks = results_face.multi_face_landmarks[0] # Take first face
                             landmarks_for_lk.extend([(lm.x * target_w, lm.y * target_h) for lm in face_landmarks.landmark])
                    if pose:
                         results_pose = pose.process(frame_rgb)
                         if results_pose.pose_landmarks:
                             pose_landmarks = results_pose.pose_landmarks
                             landmarks_for_lk.extend([(lm.x * target_w, lm.y * target_h) for lm in pose_landmarks.landmark if lm.visibility > 0.1])
                         if results_pose.segmentation_mask is not None:
                             pose_segmentation = results_pose.segmentation_mask # Store raw mask
                    frame_rgb.flags.writeable = True
                except Exception as mp_err: logger.warning(f"MediaPipe processing failed frame {frame_idx}: {mp_err}")

            # Store landmark data (can be simplified based on actual usage)
            # Storing raw landmarks might consume too much memory; store key points or derived features instead.
            # features['face_landmarks'] = face_landmarks # Store raw MediaPipe object (careful with memory)
            # features['pose_landmarks'] = pose_landmarks # Store raw MediaPipe object
            # features['pose_segmentation'] = pose_segmentation # Store raw mask (large!)
            # For LK motion, update the points if using that method
            if config.motion_method == 'lk' and landmarks_for_lk:
                 lk_prev_points = np.array(landmarks_for_lk, dtype=np.float32).reshape(-1, 1, 2)


            all_frame_features.append(features)
            prev_gray = gray
            frame_idx += 1
            if frame_idx % 100 == 0: logger.debug(f"Analyzed frame {frame_idx}...")

    except Exception as e:
        logger.error(f"Error during video content analysis loop: {e}", exc_info=True)
    finally:
        if cap: cap.release()
        if face_mesh: face_mesh.close()
        if pose: pose.close()
        logger.info(f"Content analysis complete. Processed {frame_idx} frames.")

    # Save analysis data if requested
    if config.save_analysis_data:
        try:
            analysis_file = os.path.join(config.output_dir, f"{os.path.basename(video_path)}_analysis.json")
            # Sanitize data for JSON (convert numpy arrays etc.)
            sanitized_features = []
            for feat in all_frame_features:
                 s_feat = {}
                 for k, v in feat.items():
                      if isinstance(v, (np.ndarray, np.generic)): s_feat[k] = v.tolist() # Simple list conversion
                      # Add handling for MediaPipe objects if stored (e.g., extract key points)
                      elif isinstance(v, (int, float, str, bool)) or v is None: s_feat[k] = v
                      else: s_feat[k] = repr(v)[:100] # Fallback string representation
                 sanitized_features.append(s_feat)

            with open(analysis_file, 'w') as f:
                json.dump(sanitized_features, f, indent=2)
            logger.info(f"Saved analysis data to {analysis_file}")
        except Exception as save_err:
            logger.error(f"Failed to save analysis data: {save_err}")

    return all_frame_features


# ------------------------------------------------------------------------
#                       MODEL LOADERS (Placeholders)
# ------------------------------------------------------------------------
def load_rvm_model(device: str, config: RenderConfig) -> Optional[torch.nn.Module]:
    global _rvm_available
    model = None
    # Prefer ResNet50 if path provided, then MobileNet
    model_path_or_name = config.rvm_resnet50_path if config.rvm_resnet50_path else 'resnet50' # Use name for hub
    hub_repo = 'PeterL1n/RobustVideoMatting'
    try:
        logger.info(f"Attempting to load RVM model '{model_path_or_name}' from {hub_repo if '/' in model_path_or_name else 'local'}...")
        # This assumes torch.hub usage, adjust if loading from a local file path
        model = torch.hub.load(hub_repo, model_path_or_name, pretrained=True) # Adjust 'pretrained' based on path/name
        logger.info(f"RVM model '{model_path_or_name}' loaded. Moving to device: {device}")
        model = model.to(device).eval()
        _rvm_available = True
    except Exception as e:
        logger.error(f"Failed to load RVM model '{model_path_or_name}': {e}", exc_info=True)
        _rvm_available = False
        # Try MobileNet as fallback if ResNet failed
        if model_path_or_name != 'mobilenetv3' and config.rvm_mobilenet_path:
             logger.info("Trying RVM MobileNet fallback...")
             try:
                 model = torch.hub.load(hub_repo, 'mobilenetv3', pretrained=True)
                 model = model.to(device).eval()
                 _rvm_available = True; logger.info("RVM MobileNet loaded successfully.")
             except Exception as e_mb: logger.error(f"Failed to load RVM MobileNet: {e_mb}")

    return model

def load_sam_model(device: str, config: RenderConfig) -> Optional[Any]:
    if not config.sam_checkpoint_path or not os.path.exists(config.sam_checkpoint_path):
        logger.warning("SAM checkpoint path not specified or invalid. SAM effects disabled.")
        return None

    sam_model = None
    use_sam2 = config.use_sam2 and _sam2_available
    sam_checkpoint = config.sam_checkpoint_path
    sam_type = config.sam_model_type

    try:
        if use_sam2:
            logger.info(f"Loading SAM 2 model from: {sam_checkpoint}")
            sam_device_to_use = device
            # SAM2 + MPS known issues, force CPU (check Ultralytics compatibility notes)
            if device == 'mps':
                logger.warning("MPS detected. Forcing SAM 2 to CPU due to potential compatibility issues.")
                sam_device_to_use = 'cpu'
            if UltralyticsSAM:
                 sam_model = UltralyticsSAM(sam_checkpoint)
                 # model.to(sam_device_to_use) # Ultralytics models often handle device internally or via predict args
                 logger.info(f"Ultralytics SAM (SAM 2) model initialized from {os.path.basename(sam_checkpoint)}.")
            else: raise RuntimeError("UltralyticsSAM class not available.")
        elif _sam1_available:
            logger.info(f"Loading SAM v1 model type '{sam_type}' from: {sam_checkpoint}")
            if sam_model_registry and sam_type in sam_model_registry:
                sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                sam_model = SamAutomaticMaskGenerator(sam) # Create generator
                logger.info(f"SAM v1 AutomaticMaskGenerator created on device '{device}'.")
            else: raise KeyError(f"SAM v1 model type '{sam_type}' not found or registry failed.")
        else:
             logger.warning("Neither SAM v1 nor SAM v2 library is available.")

    except Exception as e:
        logger.error(f"Failed to load SAM model (Type: {'SAM2' if use_sam2 else 'SAMv1'}, Path: {sam_checkpoint}): {e}", exc_info=True)
        sam_model = None

    return sam_model


# ------------------------------------------------------------------------
#                       EFFECT DEFINITIONS (Deep Integration)
# ------------------------------------------------------------------------
# --- Base Classes (Unchanged) ---
class Effect(abc.ABC): ... # As defined before
class IntraClipEffect(Effect): ... # As defined before
class TransitionEffect(Effect): ... # As defined before

# --- Specific Effects (Revisiting complex ones) ---
@dataclass
class LightningEffect(IntraClipEffect):
    # Using defaults from RenderConfig if not overridden by EffectParamsConfig
    def generate_lightning(self, start_point, num_segments, width, height, angle_range, segment_length):
        points = [start_point]
        current_point = start_point
        current_angle = random.uniform(0, 360)
        for _ in range(num_segments):
            angle_deviation = random.uniform(-angle_range, angle_range)
            new_angle = current_angle + angle_deviation
            dx = segment_length * cos(pi * new_angle / 180.0) # Use math.cos/sin for clarity
            dy = segment_length * sin(pi * new_angle / 180.0)
            new_point = (int(current_point[0] + dx), int(current_point[1] + dy))
            if 0 <= new_point[0] < width and 0 <= new_point[1] < height:
                points.append(new_point)
                current_point = new_point
                current_angle = new_angle
            else: break
        return points

    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float, effect_config: EffectParamsConfig, analysis_data: Optional[Dict] = None) -> np.ndarray:
        frame = validate_frame(frame)
        intensity = effect_config.intensity
        cfg = effect_config.custom_params # Specific params for this instance
        render_defaults = RenderConfig().default_effect_params['lightning'] # Access global defaults

        flicker_prob = cfg.get('flicker_probability', render_defaults['flicker_probability'])
        base_segments = cfg.get('base_segments', render_defaults['base_segments'])
        glow_alpha = cfg.get('glow_alpha', render_defaults['glow_alpha'])
        angle_range = cfg.get('angle_range', render_defaults['angle_range'])
        segment_length = cfg.get('segment_length', render_defaults['segment_length'])
        movement_threshold = cfg.get('movement_threshold', render_defaults['movement_threshold'])

        # --- Get Motion Data ---
        # analysis_data should contain motion for the current frame_time/frame_idx
        motion = analysis_data.get('motion', 0.0) if analysis_data else 0.0
        trigger_value = motion * intensity # Scale trigger potential by motion and intensity

        if trigger_value > movement_threshold and random.random() < flicker_prob:
            lightning_layer = np.zeros_like(frame, dtype=np.uint8)
            h, w = frame.shape[:2]

            # Determine start point - Use landmarks if available in analysis_data, else random
            start_point = (random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4))
            # TODO: Enhance start_point selection based on landmarks in analysis_data if passed

            num_segments = int(base_segments + trigger_value * 1.5) # More segments for more motion
            num_segments = min(35, max(5, num_segments)) # Clamp
            points = self.generate_lightning(start_point, num_segments, w, h, angle_range, segment_length)

            if len(points) > 1:
                lightning_color = (255, 255, int(180 + 75 * intensity)) # Brighter with intensity
                base_thickness = int(2 + 5 * intensity) # Thicker with intensity
                try:
                    for j in range(len(points) - 1):
                        thickness = max(1, int(base_thickness * (1.0 - j / max(1, len(points) - 1))))
                        pt1 = tuple(map(int, points[j]))
                        pt2 = tuple(map(int, points[j+1]))
                        cv2.line(lightning_layer, pt1, pt2, lightning_color, thickness, cv2.LINE_AA)

                    glow_k = max(5, int(11 + 10 * intensity) | 1) # Scale glow kernel size
                    lightning_layer_blurred = cv2.GaussianBlur(lightning_layer, (glow_k, glow_k), 0)
                    adj_glow_alpha = np.clip(glow_alpha * intensity, 0.1, 0.9) # Modulated alpha
                    frame = cv2.addWeighted(frame, 1.0, lightning_layer_blurred, adj_glow_alpha, 0)
                    # Add a brief screen flash for high intensity
                    if intensity > 1.1:
                         flash_intensity = min(0.4, (intensity - 1.1) * 0.5)
                         frame = cv2.addWeighted(frame, 1.0 - flash_intensity, np.full_like(frame, 255, dtype=np.uint8), flash_intensity, 0)

                except Exception as e: logger.warning(f"Lightning draw/blend failed: {e}")
        return frame

@dataclass
class RVMEffect(IntraClipEffect):
    # Needs access to the loaded model and state
    rvm_model: Optional[torch.nn.Module] = None
    rvm_rec: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None] * 4)
    rvm_background: Optional[np.ndarray] = None
    device: str = _torch_device

    def _preprocess(self, frame: np.ndarray, downsample_ratio: float) -> Optional[torch.Tensor]:
        frame = validate_frame(frame)
        try:
            h, w = frame.shape[:2]
            th = max(1, int(h * downsample_ratio)); tw = max(1, int(w * downsample_ratio))
            fr = resize_frame(frame, (th, tw)) # Use robust resize
            frgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            return T.ToTensor()(frgb).unsqueeze(0).to(self.device)
        except Exception as e: logger.warning(f"RVM preprocess failed: {e}"); return None

    def _postprocess(self, fgr: torch.Tensor, pha: torch.Tensor, target_h: int, target_w: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            fgr_cpu = fgr.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            pha_cpu = pha.squeeze(0).cpu().detach().numpy().squeeze() # Remove channel dim if present
            fgr_r = resize_frame((fgr_cpu * 255.0).astype(np.uint8), (target_h, target_w)) # Resize after conversion
            pha_r = resize_frame(pha_cpu, (target_h, target_w)) # Resize alpha
            pha_f = np.clip(pha_r, 0.0, 1.0)
            return fgr_r, pha_f
        except Exception as e: logger.error(f"RVM postprocess failed: {e}"); return None, None

    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float, effect_config: EffectParamsConfig, analysis_data: Optional[Dict] = None) -> np.ndarray:
        frame = validate_frame(frame)
        if not self.rvm_model: return frame # Model not loaded

        downsample_ratio = effect_config.custom_params.get('downsample_ratio', 0.4)
        display_mode = effect_config.custom_params.get('display_mode', 0) # 0=Composite, 1=Alpha, 2=FG
        target_h, target_w = frame.shape[:2]

        # Handle background loading/resizing (should ideally be done once)
        if self.rvm_background is None or self.rvm_background.shape[:2] != (target_h, target_w):
             self.rvm_background = np.full((target_h, target_w, 3), (0, 0, 255), dtype=np.uint8) # Default blue BG
             logger.debug("RVM using default background.")
             # TODO: Add logic to load background from RenderConfig path if provided

        try:
            frame_tensor = self._preprocess(frame, downsample_ratio)
            if frame_tensor is None: return frame

            with torch.no_grad():
                # Ensure rec state is on the correct device
                rec_device = [r.to(self.device) if r is not None else None for r in self.rvm_rec]
                fgr_t, pha_t, *rec_out = self.rvm_model(frame_tensor, *rec_device, downsample_ratio=downsample_ratio)
                # Update rec state (move back to CPU if storing long term?)
                self.rvm_rec = [r.detach().cpu() if r is not None else None for r in rec_out]

            fgr_np, pha_np = self._postprocess(fgr_t, pha_t, target_h, target_w)

            if fgr_np is None or pha_np is None: return frame

            if display_mode == 0: # Composite
                alpha_3c = pha_np[..., np.newaxis]
                composite = (fgr_np.astype(np.float32) * alpha_3c + self.rvm_background.astype(np.float32) * (1.0 - alpha_3c))
                return np.clip(composite, 0, 255).astype(np.uint8)
            elif display_mode == 1: # Alpha
                return cv2.cvtColor((pha_np * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif display_mode == 2: # Foreground
                return fgr_np
            else: return frame
        except Exception as e:
            logger.error(f"RVM apply failed: {e}", exc_info=True)
            self.rvm_rec = [None] * 4 # Reset state on error
            return frame

@dataclass
class SAMEffect(IntraClipEffect):
    # Needs access to the loaded model
    sam_model: Optional[Any] = None
    is_sam2: bool = False
    device: str = _torch_device

    def _draw_masks(self, frame: np.ndarray, masks: Any, alpha: float) -> np.ndarray:
        # Slightly simplified drawing logic
        if masks is None: return frame
        h, w = frame.shape[:2]; overlay = frame.copy()
        num_masks_drawn = 0
        try:
            masks_list = []
            if isinstance(masks, list) and masks and isinstance(masks[0], dict): # SAM v1 format
                masks_list = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
                masks_data = [m['segmentation'] for m in masks_list]
            elif isinstance(masks, torch.Tensor): # SAM 2 format (potentially)
                masks_data = masks.cpu().numpy() # B x H x W or B x 1 x H x W
                if masks_data.ndim == 4: masks_data = masks_data.squeeze(1)
            else:
                logger.warning(f"Unsupported SAM mask format: {type(masks)}")
                return frame

            for mask_bool in masks_data:
                if mask_bool.dtype != bool: mask_bool = mask_bool > 0.5 # Threshold if not boolean
                if mask_bool.shape != (h, w):
                     mask_resized = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else: mask_resized = mask_bool
                if not np.any(mask_resized): continue

                color = [random.randint(64, 200) for _ in range(3)]
                overlay[mask_resized] = cv2.addWeighted(overlay[mask_resized], 1.0 - alpha, np.array(color, dtype=np.uint8), alpha, 0)
                num_masks_drawn += 1
            logger.debug(f"Drew {num_masks_drawn} SAM masks.")
        except Exception as e: logger.error(f"Error drawing SAM masks: {e}", exc_info=True)
        return overlay

    def apply(self, frame: np.ndarray, frame_time: float, clip_duration: float, effect_config: EffectParamsConfig, analysis_data: Optional[Dict] = None) -> np.ndarray:
        frame = validate_frame(frame)
        if not self.sam_model: return frame

        alpha = effect_config.custom_params.get('alpha', 0.5) * effect_config.intensity
        alpha = np.clip(alpha, 0.1, 0.9)

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = None
            if self.is_sam2 and isinstance(self.sam_model, UltralyticsSAM):
                # SAM2 needs prompts (points, boxes) - use pose landmarks if available
                points = None; labels = None
                # TODO: Extract points from analysis_data if available
                if points is None: # Default to center point if no landmarks
                    points = np.array([[frame.shape[1] // 2, frame.shape[0] // 2]])
                    labels = np.array([1])
                results = self.sam_model.predict(frame_rgb, points=points, labels=labels, device=self.device) # Check predict signature
                if results and results[0].masks: masks = results[0].masks.data
            elif not self.is_sam2 and isinstance(self.sam_model, SamAutomaticMaskGenerator): # SAM v1
                masks = self.sam_model.generate(frame_rgb)
            else:
                 logger.warning("SAM model type mismatch or unavailable.")
                 return frame

            if masks is not None:
                return self._draw_masks(frame, masks, alpha)
            else: return frame # No masks generated
        except Exception as e:
            logger.error(f"SAM apply failed: {e}", exc_info=True)
            return frame


# --- Other Effects (BrightnessContrast, Saturation, Vignette, Blur, Crossfade, Slide, None) ---
# Assuming implementations are similar to the previous combined script,
# just ensure they use `effect_config.custom_params` and `effect_config.intensity`.
# (Definitions omitted for brevity, but should be included as above)
BrightnessContrastEffect = BrightnessContrastEffect # Defined earlier
SaturationEffect = SaturationEffect # Defined earlier
VignetteEffect = VignetteEffect # Defined earlier
BlurEffect = BlurEffect # Defined earlier
CrossfadeEffect = CrossfadeEffect # Defined earlier
SlideEffect = SlideEffect # Defined earlier
NoneEffect = NoneEffect # Defined earlier


# --- Effect Registry (Ensure all are included) ---
EFFECT_REGISTRY: Dict[str, Type[Effect]] = { # Use Type[Effect] for clarity
    'brightness_contrast': BrightnessContrastEffect,
    'saturation': SaturationEffect,
    'vignette': VignetteEffect,
    'blur': BlurEffect,
    'lightning': LightningEffect,
    'rvm_composite': RVMEffect,
    'sam1_segmentation': SAMEffect,
    'sam2_segmentation': SAMEffect,
    'crossfade': CrossfadeEffect,
    'slide': SlideEffect,
    'none': NoneEffect,
}

TRANSITION_EFFECTS = {'crossfade', 'slide'} # Set of transition effect names

# ------------------------------------------------------------------------
#           CORE ALGORITHM SERIES (SDE, OT, DRL) PLANNER (Enhanced)
# ------------------------------------------------------------------------
# --- DRL Agent Placeholder ---
class SimpleDRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim) # Outputs Q-values per action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        q_values = self.output_layer(x)
        return q_values

class MathematicalGeniusPlanner:
    def __init__(self, config: AnalysisConfig, frame_features: List[Dict], audio_data: Dict, available_effects: List[str], fps: float):
        self.config = config
        self.frame_features = frame_features # Expects list of dicts from analyze_video_content
        self.audio_data = audio_data
        self.effects = available_effects # Full list including 'none'
        self.num_effects = len(self.effects)
        self.fps = fps
        self.video_duration = audio_data.get('duration', 0)
        self.config.drl_action_dim = self.num_effects # Set DRL output size

        # Initialize DRL agent (simple version, no training implemented here)
        self.drl_agent = None
        if config.drl_enabled:
            try:
                self.drl_agent = SimpleDRLAgent(config.drl_state_dim, config.drl_action_dim, config.drl_hidden_dim).to(_torch_device)
                # Load pre-trained weights if available (placeholder)
                # self.drl_agent.load_state_dict(torch.load("drl_agent_weights.pth"))
                self.drl_agent.eval() # Set to evaluation mode
                logger.info("DRL Agent initialized (mock/untrained).")
            except Exception as drl_init_e:
                logger.error(f"Failed to initialize DRL agent: {drl_init_e}. DRL disabled.")
                self.config.drl_enabled = False

        # Precompute interpolated features for SDE/OT speedup
        self._precompute_interpolated_features()

    def _precompute_interpolated_features(self):
        logger.debug("Precomputing interpolated features...")
        self.times = np.array([f['time'] for f in self.frame_features])
        # Motion
        motions = np.array([f['motion'] for f in self.frame_features])
        self.motion_interp_fn = interpolate.interp1d(self.times, motions, kind='linear', bounds_error=False, fill_value=(motions[0], motions[-1]))
        # Beat Proximity
        beats = self.audio_data.get('beat_times', [])
        if beats:
             beat_prox = np.array([1.0 / (min(abs(t - b) for b in beats) + 0.05) for t in self.times])
             self.beat_prox_interp_fn = interpolate.interp1d(self.times, beat_prox, kind='linear', bounds_error=False, fill_value=(beat_prox[0], beat_prox[-1]))
        else: self.beat_prox_interp_fn = lambda t: 0.0 # No beats, proximity is zero
        logger.debug("Interpolated features precomputed.")

    def _get_feature_at_time(self, feature_fn: Callable, time: float) -> float:
        try: return float(feature_fn(time))
        except Exception as e: logger.warning(f"Interpolation failed at time {time}: {e}"); return 0.0

    def sde_interval_selection(self) -> List[Tuple[float, float]]:
        logger.info("Running SDE for interval selection...")
        dt = 1.0 / self.fps; T = self.video_duration
        if T <= 0 or dt <= 0: return []
        num_steps = int(T / dt)
        if num_steps <= 1: return []
        X = np.zeros(num_steps); X[0] = 0
        alpha, beta, theta = self.config.sde_alpha, self.config.sde_beta, self.config.sde_threshold
        sde_times = np.linspace(0, T, num_steps)

        # Use interpolated functions
        motions_t = np.array([self._get_feature_at_time(self.motion_interp_fn, t) for t in sde_times])
        beat_prox_t = np.array([self._get_feature_at_time(self.beat_prox_interp_fn, t) for t in sde_times])

        # Normalize
        max_motion = np.max(motions_t) + 1e-6; max_beat_prox = np.max(beat_prox_t) + 1e-6
        norm_motions = motions_t / max_motion; norm_beat_prox = beat_prox_t / max_beat_prox

        for i in range(num_steps - 1):
            m_t = norm_motions[i]; b_t = norm_beat_prox[i]
            mu = alpha * m_t + (1 - alpha) * b_t
            sigma = beta * sqrt(max(0, m_t)) # Ensure non-negative input to sqrt
            dW = np.random.randn() * sqrt(dt)
            X[i+1] = X[i] + mu * dt + sigma * dW

        # Find intervals where X > threshold
        intervals = []; start_idx = -1
        min_len_frames = int(self.config.sde_min_interval_sec * self.fps)
        for i in range(num_steps):
            if X[i] > theta and start_idx == -1: start_idx = i
            elif (X[i] <= theta or i == num_steps - 1) and start_idx != -1:
                end_idx = i + 1 if i == num_steps - 1 and X[i] > theta else i # Include last frame if still above
                if (end_idx - start_idx) >= min_len_frames:
                    intervals.append((sde_times[start_idx], sde_times[min(end_idx, num_steps-1)])) # Ensure end time is valid
                start_idx = -1
        if start_idx != -1 and (num_steps - start_idx) >= min_len_frames: # Catch trailing interval
            intervals.append((sde_times[start_idx], sde_times[-1]))

        # Merge overlapping/adjacent intervals (optional refinement)
        if not intervals: return []
        merged = []
        intervals.sort(key=lambda x: x[0])
        current_start, current_end = intervals[0]
        for next_start, next_end in intervals[1:]:
             if next_start <= current_end + dt * 2: # Merge if very close or overlapping
                  current_end = max(current_end, next_end)
             else:
                  merged.append((current_start, current_end))
                  current_start, current_end = next_start, next_end
        merged.append((current_start, current_end))

        logger.info(f"SDE selected {len(merged)} intervals after merging.")
        return merged

    def optimal_transport_effect_assignment(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float, str, Dict]]:
        if not self.config.ot_enabled:
            logger.warning("Optimal Transport disabled (POT library missing or config). Using random assignment.")
            return [(start, end, random.choice(self.effects), {}) for start, end in intervals]
        if not intervals or not self.effects: return []
        logger.info("Running Optimal Transport for effect assignment...")
        num_intervals = len(intervals); num_effects = len(self.effects)

        # Cost function C[i, j]: Cost of effect j on interval i
        C = np.zeros((num_intervals, num_effects))
        complex_effects = {'lightning', 'rvm_composite', 'sam1_segmentation', 'sam2_segmentation'}
        max_avg_motion = 0
        avg_motions = []
        for start, end in intervals:
             motion = np.mean([f['motion'] for f in self.frame_features if start <= f['time'] < end]) if any(start <= f['time'] < end for f in self.frame_features) else 0
             avg_motions.append(motion)
             if motion > max_avg_motion: max_avg_motion = motion
        max_avg_motion += 1e-6

        for i in range(num_intervals):
            norm_motion = avg_motions[i] / max_avg_motion
            for j, effect_name in enumerate(self.effects):
                is_complex = effect_name in complex_effects
                # Cost: Lower cost for complex effects in high motion, higher cost in low motion
                cost = (1.0 - norm_motion) if is_complex else norm_motion
                # Add complexity bias
                cost += self.config.ot_cost_complexity_bias if is_complex else 0
                # Penalize 'none' slightly?
                if effect_name == 'none': cost += 0.1
                C[i, j] = cost

        # Uniform distributions
        mu = np.ones(num_effects) / num_effects; nu = np.ones(num_intervals) / num_intervals
        try:
            gamma = sinkhorn(mu, nu, C, reg=self.config.ot_regularization) # OT plan
            assignments = []
            interval_effect_probs = gamma * num_intervals # Scale rows to sum to 1 (approx)
            for i in range(num_intervals):
                 probs = interval_effect_probs[i, :] / (np.sum(interval_effect_probs[i, :]) + 1e-9)
                 chosen_effect_idx = np.random.choice(num_effects, p=probs)
                 assignments.append((intervals[i][0], intervals[i][1], self.effects[chosen_effect_idx], {}))
            logger.info("Optimal Transport assignment complete.")
            return assignments
        except Exception as e:
            logger.error(f"Optimal Transport failed: {e}. Falling back to random assignment.")
            return [(start, end, random.choice(self.effects), {}) for start, end in intervals]

    def _get_drl_state(self, current_time: float, last_effect_idx: int, effect_counts: Dict[str, int]) -> torch.Tensor:
        """ Constructs the state vector for the DRL agent """
        norm_motion = self._get_feature_at_time(self.motion_interp_fn, current_time)
        norm_beat_prox = self._get_feature_at_time(self.beat_prox_interp_fn, current_time)
        time_progress = current_time / self.video_duration if self.video_duration > 0 else 0
        num_effects_used = len(effect_counts)

        # Example state: [motion, beat_prox, time_progress, last_effect_idx (norm), num_effects_used (norm)]
        state_list = [
            norm_motion,
            norm_beat_prox,
            time_progress,
            last_effect_idx / self.num_effects if self.num_effects > 0 else 0,
            num_effects_used / self.num_effects if self.num_effects > 0 else 0
        ]
        # Pad state if state_dim is larger
        if len(state_list) < self.config.drl_state_dim:
             state_list.extend([0.0] * (self.config.drl_state_dim - len(state_list)))

        return torch.tensor(state_list[:self.config.drl_state_dim], dtype=torch.float32).to(_torch_device)

    def drl_refinement(self, initial_plan: List[Tuple[float, float, str, Dict]]) -> List[Tuple[float, float, str, Dict]]:
        if not self.config.drl_enabled or not self.drl_agent:
            logger.info("DRL refinement skipped (disabled or agent failed).")
            return initial_plan
        if not initial_plan: return []

        logger.info("Running DRL refinement step...")
        refined_plan = list(initial_plan) # Work on a copy
        effect_counts = defaultdict(int)
        last_effect_idx = self.effects.index('none') if 'none' in self.effects else 0

        # Simplified refinement loop (no actual training, just inference)
        for i in range(len(refined_plan)):
            start, end, old_effect, params = refined_plan[i]
            current_time = start

            # Get state
            state_tensor = self._get_drl_state(current_time, last_effect_idx, effect_counts)

            # Agent selects action (effect index)
            with torch.no_grad():
                q_values = self.drl_agent(state_tensor)
                action_idx = torch.argmax(q_values).item()

            new_effect_name = self.effects[action_idx]

            # Update plan (and state tracking for next iteration)
            # TODO: DRL could also output intensity or modify params
            refined_plan[i] = (start, end, new_effect_name, params)
            effect_counts[new_effect_name] += 1
            last_effect_idx = action_idx

            # Simulate reward calculation (for logging/debugging, not training here)
            # visual_impact = 1.0 if new_effect_name == 'lightning' else 0.5 # Example
            # sync_penalty = -min(abs(start - b) for b in self.audio_data['beat_times']) if self.audio_data['beat_times'] else 0
            # creativity_bonus = 1.0 / (effect_counts[new_effect_name]) # Penalize repetition
            # reward = visual_impact * w1 + sync_penalty * w2 + creativity_bonus * w3
            # logger.debug(f"DRL Step {i}: State={state_tensor.cpu().numpy()}, Action={new_effect_name}, Q-Vals={q_values.cpu().numpy()}")

        logger.info("DRL refinement inference complete.")
        return refined_plan

    def plan_edits(self) -> List[Tuple[float, float, str, EffectParamsConfig]]:
        if self.video_duration <= 0: raise RuntimeError("Video duration is zero.")
        if not self.frame_features: raise RuntimeError("Frame features not analyzed.")

        intervals = self.sde_interval_selection()
        if not intervals: logger.warning("SDE selected no intervals."); return []

        ot_plan = self.optimal_transport_effect_assignment(intervals)
        if not ot_plan: logger.warning("Optimal transport failed."); return []

        final_plan_raw = self.drl_refinement(ot_plan)

        # Convert to final format with EffectParamsConfig
        final_plan = []
        for start, end, effect_name, custom_params in final_plan_raw:
             intensity = custom_params.pop('intensity', 1.0) # Extract if DRL set it
             effect_config = EffectParamsConfig(intensity=intensity, custom_params=custom_params)
             final_plan.append((start, end, effect_name, effect_config))
        return final_plan

# ------------------------------------------------------------------------
#           ALTERNATIVE PLANNER (Pareto Roulette - Enhanced)
# ------------------------------------------------------------------------
class ParetoRoulettePlanner:
    def __init__(self, config: AnalysisConfig, frame_features: List[Dict], audio_data: Dict, available_effects: List[str], fps: float):
        self.config = config
        self.frame_features = frame_features
        self.audio_data = audio_data
        # Exclude 'none' effect from random assignment during candidate generation
        self.effects_for_roulette = [e for e in available_effects if e != 'none']
        if not self.effects_for_roulette: # Ensure there's at least one effect
             self.effects_for_roulette = ['brightness_contrast'] # Fallback
             logger.warning("No effects available for Roulette planner besides 'none'. Using fallback.")
        self.fps = fps
        self.video_duration = audio_data.get('duration', 0)
        # Precompute features similar to the other planner
        self._precompute_interpolated_features()

    # Using same precomputation and interpolation helpers as MathematicalGeniusPlanner
    _precompute_interpolated_features = MathematicalGeniusPlanner._precompute_interpolated_features
    _get_feature_at_time = MathematicalGeniusPlanner._get_feature_at_time

    def stochastic_interval_pooling_weighted(self) -> List[Tuple[float, float]]:
        logger.info("Running weighted stochastic interval pooling...")
        intervals = []; current_time = 0
        min_dur, max_dur = self.config.pareto_min_interval_duration, self.config.pareto_max_interval_duration
        motion_w = self.config.pareto_motion_weight; beat_w = self.config.pareto_beat_weight
        times = np.array([f['time'] for f in self.frame_features])

        # Calculate weights for all time points
        weights = np.zeros_like(times)
        motions = np.array([self._get_feature_at_time(self.motion_interp_fn, t) for t in times])
        beats_prox = np.array([self._get_feature_at_time(self.beat_prox_interp_fn, t) for t in times])
        max_motion = np.max(motions) + 1e-6; max_beat_prox = np.max(beats_prox) + 1e-6
        weights = (motions / max_motion) * motion_w + (beats_prox / max_beat_prox) * beat_w
        weights = np.maximum(0.01, weights) # Ensure minimum weight

        while current_time < self.video_duration:
            # Consider indices starting from current_time
            valid_indices = np.where(times >= current_time)[0]
            if len(valid_indices) == 0: break

            # Sample start index based on weights
            current_weights = weights[valid_indices]
            probs = current_weights / (np.sum(current_weights) + 1e-9)
            try:
                 start_idx_relative = np.random.choice(len(valid_indices), p=probs)
                 start_idx = valid_indices[start_idx_relative]
            except ValueError as e: # Handle potential issue with probabilities
                 logger.warning(f"Probability sampling error: {e}. Falling back to uniform.")
                 start_idx = np.random.choice(valid_indices)

            start_time = times[start_idx]

            # Determine duration
            max_possible_dur = self.video_duration - start_time
            duration = random.uniform(min_dur, min(max_dur, max_possible_dur))
            if duration < min_dur * 0.1: continue # Skip tiny intervals

            end_time = start_time + duration
            intervals.append((start_time, end_time))
            current_time = end_time

        logger.info(f"Weighted pooling selected {len(intervals)} intervals.")
        return intervals

    def evaluate_pareto_objectives(self, edit_plan: List[Tuple[float, float, str, EffectParamsConfig]], beats: List[float]) -> List[float]:
        if not edit_plan: return [0, 0, 0]
        num_intervals = len(edit_plan)

        # Visual Appeal: Average intensity of applied effects (complex effects get higher base intensity)
        visual_score = 0
        complex_effects = {'lightning', 'rvm_composite', 'sam1_segmentation', 'sam2_segmentation'}
        for _, _, eff_name, eff_cfg in edit_plan:
            base_impact = 1.0 if eff_name in complex_effects else (0.5 if eff_name != 'none' else 0.1)
            visual_score += base_impact * eff_cfg.intensity
        norm_visual = visual_score / num_intervals if num_intervals > 0 else 0

        # Sync Score: Average alignment quality (e.g., gaussian decay around beats)
        sync_score = 0
        if beats:
            beat_sync_quality = 0
            for start, end, _, _ in edit_plan:
                 # Score based on proximity of start OR end to a beat
                 mid_time = (start + end) / 2 # Or score mid-point? Let's score start/end
                 min_dist_start = min(abs(start - b) for b in beats)
                 min_dist_end = min(abs(end - b) for b in beats)
                 min_dist = min(min_dist_start, min_dist_end)
                 # Gaussian score: higher closer to beat (e.g., std dev = 0.1s)
                 beat_sync_quality += exp(-(min_dist**2) / (2 * 0.1**2))
            sync_score = beat_sync_quality / num_intervals if num_intervals > 0 else 0

        # Creativity: Entropy of effect distribution
        creativity_score = 0
        if num_intervals > 0:
            counts = defaultdict(int)
            for _, _, eff_name, _ in edit_plan: counts[eff_name] += 1
            probs = np.array([c / num_intervals for c in counts.values()])
            creativity_score = scipy_stats.entropy(probs) # Higher entropy means more variety

        return [norm_visual, sync_score, creativity_score]

    def dominates(self, scores1: List[float], scores2: List[float]) -> bool: # Identical to other planner
        if len(scores1) != len(scores2): return False
        at_least_one_better = False
        for s1, s2 in zip(scores1, scores2):
            if s1 < s2 - 1e-9: return False
            if s1 > s2 + 1e-9: at_least_one_better = True
        return at_least_one_better

    def plan_edits(self) -> List[Tuple[float, float, str, EffectParamsConfig]]:
        if self.video_duration <= 0: raise RuntimeError("Video duration is zero.")
        if not self.frame_features: raise RuntimeError("Frame features not analyzed.")

        intervals = self.stochastic_interval_pooling_weighted()
        if not intervals: return []

        candidates = []; beats = self.audio_data.get('beat_times', [])
        num_candidates = self.config.pareto_num_candidates
        logger.info(f"Generating {num_candidates} candidates for Pareto Roulette...")

        for _ in range(num_candidates):
            edit_plan_raw = []
            for start, end in intervals:
                effect_name = random.choice(self.effects_for_roulette)
                intensity = random.uniform(0.6, 1.3) # Random intensity
                # Could add random parameter variations here too
                custom_params = {}
                effect_config = EffectParamsConfig(intensity=intensity, custom_params=custom_params)
                edit_plan_raw.append((start, end, effect_name, effect_config))

            scores = self.evaluate_pareto_objectives(edit_plan_raw, beats)
            candidates.append({'plan': edit_plan_raw, 'scores': scores})

        # Find Pareto front
        pareto_front = []
        for i in range(num_candidates):
            if not candidates[i]['plan']: continue # Skip empty plans
            is_dominated = False
            for j in range(num_candidates):
                if i != j and candidates[j]['plan'] and self.dominates(candidates[j]['scores'], candidates[i]['scores']):
                    is_dominated = True; break
            if not is_dominated: pareto_front.append(candidates[i])

        if not pareto_front:
            logger.warning("Pareto front is empty! Selecting best based on weighted sum (or random).")
            if not candidates: return []
            # Simple weighted sum for fallback selection
            weights = np.array([self.config.pareto_select_weight_visual,
                                self.config.pareto_select_weight_sync,
                                self.config.pareto_select_weight_creativity])
            best_candidate_idx = np.argmax([np.dot(c['scores'], weights) for c in candidates])
            selected_candidate = candidates[best_candidate_idx]
        else:
            # Select randomly from the Pareto front
            selected_candidate = random.choice(pareto_front)
            logger.info(f"Selected from Pareto front ({len(pareto_front)} options). Chosen scores: {['{:.2f}'.format(s) for s in selected_candidate['scores']]}")

        # Return the plan part
        return selected_candidate['plan']


# ------------------------------------------------------------------------
#           UNIFIED MOVIEPY RENDERING FUNCTION (Deep Integration)
# ------------------------------------------------------------------------
def buildSequenceVideo_unified(
    final_edit_plan: List[Tuple[float, float, str, EffectParamsConfig]],
    source_video_path: str,
    output_video_path: str,
    master_audio_path: str,
    render_config: RenderConfig,
    analysis_data: List[Dict], # Pass the detailed frame features
    # --- Pass Loaded Models ---
    rvm_model: Optional[torch.nn.Module] = None,
    sam_model: Optional[Any] = None,
    is_sam2: bool = False
    ):
    logger.info(f"Rendering video to {output_video_path} with audio {master_audio_path}")
    start_time = time.time()
    tracemalloc.start()

    if not final_edit_plan: raise ValueError("Edit plan is empty.")
    if not os.path.exists(source_video_path): raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not os.path.exists(master_audio_path): raise FileNotFoundError(f"Master audio not found: {master_audio_path}")

    width = render_config.resolution_width
    height = render_config.resolution_height
    fps = render_config.fps
    if fps <= 0: fps = 30; logger.warning("Invalid FPS, defaulting to 30.")

    source_clip: Optional[VideoFileClip] = None
    try:
        logger.info(f"Loading source video: {os.path.basename(source_video_path)}")
        source_clip = VideoFileClip(source_video_path, audio=False, target_resolution=(height, width))
        if source_clip.w != width or source_clip.h != height:
            source_clip = source_clip.resize(height=height) # Let moviepy handle aspect ratio if only height is given
            # Force width if needed after resize
            if source_clip.w != width: source_clip = source_clip.resize(width=width)
        if source_clip.duration is None or source_clip.duration <= 0: raise ValueError("Source clip has zero duration.")
        video_total_duration = source_clip.duration
        logger.info(f"Source video loaded. Duration: {video_total_duration:.2f}s")
        sequence_duration = max(end for _, end, _, _ in final_edit_plan) if final_edit_plan else 0
        if sequence_duration <= 0: raise ValueError("Calculated sequence duration is zero.")
        sequence_duration = min(sequence_duration, video_total_duration) # Ensure sequence doesn't exceed source

    except Exception as load_err:
        if tracemalloc.is_tracing(): tracemalloc.stop()
        logger.error(f"Failed to load source clip {source_video_path}: {load_err}", exc_info=True)
        if source_clip: source_clip.close()
        raise

    # --- Create Stateful Effect Instances (if needed) ---
    # Pass models to relevant effect instances
    effect_instances = {}
    if rvm_model:
        # Need to manage RVM background and state ('rec')
        rvm_bg = None # TODO: Load background from config path
        effect_instances['rvm_composite'] = RVMEffect(rvm_model=rvm_model, rvm_background=rvm_bg, device=_torch_device)
    if sam_model:
        sam_effect = SAMEffect(sam_model=sam_model, is_sam2=is_sam2, device=_torch_device)
        effect_instances['sam1_segmentation'] = sam_effect # Use same instance
        effect_instances['sam2_segmentation'] = sam_effect

    # Precompute analysis data lookup (frame index -> features) for faster access in make_frame
    analysis_lookup = {f['frame_idx']: f for f in analysis_data}

    # --- Define make_frame Function ---
    def make_frame(t):
        current_time = np.clip(t, 0, sequence_duration - 1e-6)
        frame_idx = int(current_time * fps)

        # Determine active effect segment(s) at time 't'
        active_effects_info = [] # Stores (effect_name, effect_config, interval_info)
        transition_info = None # Stores (effect_name, effect_config, progress, frame_b)

        for i, (start, end, effect_name, effect_config) in enumerate(final_edit_plan):
            is_transition = effect_name in TRANSITION_EFFECTS
            duration = effect_config.custom_params.get('duration', 0.5) if is_transition else (end - start)

            if is_transition:
                # Transition happens at the *start* of the interval in this plan structure
                transition_end_time = start + duration
                if start <= current_time < transition_end_time:
                     # Need frame from the *next* interval/segment
                     next_interval_start_time = end # End time of transition is start of next effect
                     next_source_time = np.clip(next_interval_start_time, 0, video_total_duration - 1e-6)
                     try:
                          frame_b = source_clip.get_frame(next_source_time)
                          if frame_b.shape[0] != height or frame_b.shape[1] != width:
                              frame_b = resize_frame(frame_b, (height, width))
                     except Exception: frame_b = None # Handle error fetching next frame

                     if frame_b is not None:
                          progress = (current_time - start) / duration if duration > 0 else 1.0
                          transition_info = (effect_name, effect_config, progress, frame_b)
                          # Don't apply other effects during transition? Or maybe allow base effect?
                          # Let's prioritize transition.
                          break # Found the active transition

            elif start <= current_time < end: # Standard intra-clip effect interval
                 interval_info = {'start': start, 'end': end, 'duration': duration}
                 active_effects_info.append((effect_name, effect_config, interval_info))
                 # Don't break, allow overlapping effects if structure permits (though current plan is sequential)

        # --- Get Base Frame ---
        source_time = np.clip(current_time, 0, video_total_duration - 1e-6)
        try:
            base_frame = source_clip.get_frame(source_time)
            if base_frame.shape[0] != height or base_frame.shape[1] != width:
                base_frame = resize_frame(base_frame, (height, width))
        except Exception as frame_err:
            logger.error(f"Error getting frame at source_time {source_time:.4f} (t={t:.4f}): {frame_err}")
            return np.zeros((height, width, 3), dtype=np.uint8)

        processed_frame = base_frame.copy()

        # --- Apply Effects ---
        current_frame_analysis = analysis_lookup.get(frame_idx) # Get analysis data for this frame

        if transition_info:
            # Apply Transition Effect
            effect_name, effect_config, progress, frame_b = transition_info
            effect_class = EFFECT_REGISTRY.get(effect_name)
            if effect_class and issubclass(effect_class, TransitionEffect):
                 try:
                     # Use default params if needed
                     full_params = {**render_config.default_effect_params.get(effect_name, {}), **effect_config.custom_params}
                     effect_instance = effect_class(**full_params) # Instantiate with combined params
                     processed_frame = effect_instance.apply(processed_frame, frame_b, progress, effect_config)
                 except Exception as apply_err:
                      logger.warning(f"Failed to apply transition '{effect_name}' at t={t:.3f}: {apply_err}")
            else: logger.warning(f"Transition effect '{effect_name}' not found or invalid type.")
        else:
            # Apply Intra-Clip Effects (if any active)
            if not active_effects_info: # If no interval matches, apply 'none'
                 active_effects_info.append(('none', EffectParamsConfig(), {'start': current_time, 'end': current_time + 1/fps, 'duration': 1/fps}))

            for effect_name, effect_config, interval_info in active_effects_info:
                effect_class = EFFECT_REGISTRY.get(effect_name)
                if effect_class and issubclass(effect_class, IntraClipEffect):
                    try:
                        frame_time_in_interval = current_time - interval_info['start']
                        interval_duration = interval_info['duration']

                        # Get or create effect instance (handle stateful ones)
                        if effect_name in effect_instances:
                             effect_instance = effect_instances[effect_name]
                        else:
                            # Use default params merged with specific config
                            full_params = {**render_config.default_effect_params.get(effect_name, {}), **effect_config.custom_params}
                            effect_instance = effect_class(**full_params)
                            # Store if potentially stateful (simple check, refine if needed)
                            if effect_name in ['rvm_composite']: effect_instances[effect_name] = effect_instance

                        # Pass analysis data for effects that need it (like Lightning)
                        processed_frame = effect_instance.apply(
                             processed_frame,
                             frame_time_in_interval,
                             interval_duration,
                             effect_config,
                             analysis_data=current_frame_analysis # Pass frame-specific analysis
                        )
                    except Exception as apply_err:
                        logger.warning(f"Failed to apply intra-clip effect '{effect_name}' at t={t:.3f}: {apply_err}")
                elif effect_name != 'none':
                    logger.warning(f"Intra-clip effect '{effect_name}' not found or invalid type.")

        # Final validation
        if processed_frame.shape[:2] != (height, width):
             processed_frame = resize_frame(processed_frame, (height, width))
        return processed_frame

    # --- Create and Write VideoClip (Code largely same as previous) ---
    master_audio = None; sequence_clip = None; temp_audio_filepath = None
    try:
        logger.info(f"Creating MoviePy VideoClip with duration {sequence_duration:.2f}s")
        # Use threads=1 in make_frame processing if parallel issues arise, but MoviePy write handles threading.
        sequence_clip = VideoClip(make_frame, duration=sequence_duration, ismask=False)

        logger.debug(f"Loading master audio: {master_audio_path}")
        master_audio = AudioFileClip(master_audio_path)

        # Adjust audio/video duration
        final_duration = sequence_duration
        if master_audio.duration > final_duration:
            master_audio = master_audio.subclip(0, final_duration)
        elif master_audio.duration < final_duration - 1e-3:
             final_duration = master_audio.duration
             sequence_clip = sequence_clip.set_duration(final_duration)
             logger.warning(f"Audio ({master_audio.duration:.2f}s) shorter than video ({sequence_duration:.2f}s). Trimming video to {final_duration:.2f}s.")

        if master_audio: sequence_clip = sequence_clip.set_audio(master_audio)
        else: logger.warning("No master audio. Video will be silent.")

        # Prepare FFmpeg params
        temp_audio_filename = f"temp-audio_{int(time.time())}.m4a"; temp_audio_dir = os.path.dirname(output_video_path) or "."; os.makedirs(temp_audio_dir, exist_ok=True); temp_audio_filepath = os.path.join(temp_audio_dir, temp_audio_filename)
        ffmpeg_params_list = ["-preset", str(render_config.preset), "-crf", str(render_config.crf)]
        # Add `-tune animation` or `-tune film`? Depends on content.
        # ffmpeg_params_list.extend(["-tune", "film"])

        write_params = {
            "codec": render_config.video_codec, "audio_codec": render_config.audio_codec,
            "temp_audiofile": temp_audio_filepath, "remove_temp": True,
            "threads": render_config.threads, "logger": 'bar', "write_logfile": False,
            "audio_bitrate": render_config.audio_bitrate, "fps": fps,
            "ffmpeg_params": ffmpeg_params_list
        }

        logger.info(f"Writing final video (FPS={fps}, Size={width}x{height}, Threads={render_config.threads})...")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        sequence_clip.write_videofile(output_video_path, **write_params)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"MoviePy Render Performance: Time: {time.time() - start_time:.2f}s, Peak Mem: {peak_mem / 1024**2:.2f} MB")
        logger.info(f"Rendering successful: {output_video_path}")

    except Exception as e:
        if tracemalloc.is_tracing(): tracemalloc.stop()
        logger.error(f"MoviePy rendering failed: {e}", exc_info=True)
        # Cleanup failed files
        if os.path.exists(output_video_path): try: os.remove(output_video_path) except OSError: pass
        if temp_audio_filepath and os.path.exists(temp_audio_filepath): try: os.remove(temp_audio_filepath) except OSError: pass
        raise
    finally:
        logger.debug("Cleaning up MoviePy objects...")
        if sequence_clip: try: sequence_clip.close() catch Exception: pass
        if master_audio: try: master_audio.close() catch Exception: pass
        if source_clip: try: source_clip.close() catch Exception: pass
        import gc; gc.collect()
        logger.debug("MoviePy cleanup finished.")


# ------------------------------------------------------------------------
#                 MAIN ORCHESTRATION CLASS (Refined)
# ------------------------------------------------------------------------
class AutonomousVideoEditor:
    def __init__(self, video_path: str, output_path: str, analysis_config: AnalysisConfig, render_config: RenderConfig):
        self.video_path = video_path
        self.output_path = output_path
        self.analysis_config = analysis_config
        self.render_config = render_config
        self.analysis_results: List[Dict] = [] # Stores detailed frame features
        self.audio_data: Optional[Dict] = None
        self.fps: float = 30.0
        self.edit_plan: List[Tuple[float, float, str, EffectParamsConfig]] = []
        self.audio_utils = BasicAudioUtils()

        # --- Load Models ---
        self.rvm_model = load_rvm_model(_torch_device, self.render_config)
        self.sam_model = load_sam_model(_torch_device, self.render_config)
        self.is_sam2 = self.render_config.use_sam2 and _sam2_available

        # Ensure output directories exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(self.analysis_config.output_dir, exist_ok=True)

    def analyze(self):
        logger.info("--- Starting Analysis Phase ---")
        cap = cv2.VideoCapture(self.video_path);
        if not cap.isOpened(): raise IOError(f"Cannot open video: {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if not self.fps or self.fps <= 0: self.fps = 30.0; logger.warning("Invalid FPS, using 30.")
        # Use actual FPS for render config if not set otherwise
        if self.render_config.fps <= 0: self.render_config.fps = int(round(self.fps))

        # Analyze Audio
        audio_out = os.path.join(self.analysis_config.output_dir, f"{os.path.basename(self.video_path)}_audio.wav")
        extracted_audio_path = self.audio_utils.extract_audio(self.video_path, audio_out)
        if extracted_audio_path:
            self.audio_data = self.audio_utils.analyze_audio(extracted_audio_path)
            try: os.remove(extracted_audio_path) catch OSError: pass
        if not self.audio_data: raise RuntimeError("Audio analysis failed.")

        # Analyze Video Content (Motion, Landmarks etc.)
        self.analysis_results = analyze_video_content(self.video_path, self.fps, self.analysis_config)
        if not self.analysis_results: raise RuntimeError("Video content analysis failed.")
        logger.info("--- Analysis Phase Complete ---")

    def plan_edits(self):
        logger.info(f"--- Starting Planning Phase ({self.analysis_config.planner_type}) ---")
        available_effects = list(EFFECT_REGISTRY.keys())
        # Filter effects based on model availability
        if not self.rvm_model: available_effects = [e for e in available_effects if e != 'rvm_composite']
        if not self.sam_model: available_effects = [e for e in available_effects if not e.startswith('sam')]
        logger.info(f"Available effects for planning: {available_effects}")

        planner: Union[MathematicalGeniusPlanner, ParetoRoulettePlanner]
        if self.analysis_config.planner_type == "MathematicalGenius":
            planner = MathematicalGeniusPlanner(self.analysis_config, self.analysis_results, self.audio_data, available_effects, self.fps)
        elif self.analysis_config.planner_type == "ParetoRoulette":
            planner = ParetoRoulettePlanner(self.analysis_config, self.analysis_results, self.audio_data, available_effects, self.fps)
        else: raise ValueError(f"Unknown planner type: {self.analysis_config.planner_type}")

        self.edit_plan = planner.plan_edits()
        if not self.edit_plan: raise RuntimeError("Edit planning failed to produce a plan.")

        # Log the plan
        logger.info("Generated Edit Plan:")
        for i, (start, end, effect, config) in enumerate(self.edit_plan):
            logger.info(f"  {i+1}: [{start:.2f}s - {end:.2f}s] Effect: {effect}, Intensity: {config.intensity:.2f}")
        logger.info("--- Planning Phase Complete ---")

    def render(self):
        logger.info("--- Starting Rendering Phase ---")
        audio_out_render = os.path.join(self.analysis_config.output_dir, f"{os.path.basename(self.video_path)}_render_audio.wav")
        master_audio_path = self.audio_utils.extract_audio(self.video_path, audio_out_render)
        if not master_audio_path: raise RuntimeError("Failed to extract audio for rendering.")

        try:
            buildSequenceVideo_unified(
                final_edit_plan=self.edit_plan,
                source_video_path=self.video_path,
                output_video_path=self.output_path,
                master_audio_path=master_audio_path,
                render_config=self.render_config,
                analysis_data=self.analysis_results, # Pass analysis data
                # Pass loaded models
                rvm_model=self.rvm_model,
                sam_model=self.sam_model,
                is_sam2=self.is_sam2
            )
        finally:
            if master_audio_path and os.path.exists(master_audio_path): try: os.remove(master_audio_path) catch OSError: pass
        logger.info("--- Rendering Phase Complete ---")

    def run_workflow(self):
        logger.info(f"Starting Autonomous Workflow for {os.path.basename(self.video_path)}")
        workflow_start_time = time.time()
        try:
            self.analyze()
            self.plan_edits()
            self.render()
            total_time = time.time() - workflow_start_time
            logger.info(f"--- Autonomous Editing Workflow Completed Successfully ({total_time:.2f}s) ---")
        except Exception as e:
            total_time = time.time() - workflow_start_time
            logger.critical(f"--- Autonomous Editing Workflow FAILED after {total_time:.2f}s ---", exc_info=True)
            try: # Fallback message box
                 root = tkinter.Tk(); root.withdraw(); messagebox.showerror("Workflow Error", f"Error: {e}\nCheck logs."); root.destroy()
            except Exception: pass # Ignore if UI fails

# ------------------------------------------------------------------------
#                       APPLICATION ENTRY POINT
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Unified Autonomous Video Editor")
    parser.add_argument('--input', type=str, required=True, help="Input video file path")
    parser.add_argument('--output', type=str, default=None, help="Output video file path")
    parser.add_argument('--planner', type=str, default="ParetoRoulette", choices=["MathematicalGenius", "ParetoRoulette"], help="Planning strategy")
    parser.add_argument('--log-level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--sam-checkpoint', type=str, default=None, help="Path to SAM checkpoint (.pth)")
    parser.add_argument('--use-sam2', action='store_true', help="Prefer SAM 2 if available and checkpoint provided")
    # Add more config overrides as needed, e.g., --render-width 1920 --sde-alpha 0.5 etc.

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)
    # Ensure root logger also respects level for dependencies
    logging.getLogger().setLevel(log_level)

    logger.info("--- Autonomous Video Editor Initializing ---")
    logger.info(f"Input: {args.input}")
    logger.info(f"Planner: {args.planner}")

    # --- Determine Output Path ---
    if args.output is None:
        base, ext = os.path.splitext(os.path.basename(args.input))
        output_dir = "output_autonomous"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{base}_edited_{args.planner}{ext if ext else '.mp4'}")
    logger.info(f"Output: {args.output}")

    # --- Create Configurations (Allow overrides later) ---
    analysis_cfg = AnalysisConfig(
        planner_type=args.planner,
        output_dir=os.path.dirname(args.output) or "." # Use output dir for temp files
    )
    render_cfg = RenderConfig(
         sam_checkpoint_path=args.sam_checkpoint,
         use_sam2=args.use_sam2
    )
    # TODO: Apply specific overrides from args to analysis_cfg and render_cfg if added to parser

    # --- Check FFmpeg ---
    try:
        result = subprocess.run("ffmpeg -version", shell=True, capture_output=True, text=True, check=False, timeout=5)
        if result.returncode != 0: raise RuntimeError("FFmpeg check failed")
        logger.info("FFmpeg found.")
    except Exception as ffmpeg_e:
        logger.critical(f"FFmpeg not found or failed check: {ffmpeg_e}. FFmpeg is required. Exiting.")
        sys.exit(1)

    # --- Instantiate and Run Editor ---
    try:
        editor = AutonomousVideoEditor(
            video_path=args.input,
            output_path=args.output,
            analysis_config=analysis_cfg,
            render_config=render_cfg
        )
        editor.run_workflow()
    except Exception as main_err:
        logger.critical(f"Editor execution failed: {main_err}", exc_info=True)
        sys.exit(1)

    logger.info("--- Script Finished ---")

# ------------------------------------------------------------------------
#                       REQUIREMENTS.TXT (Comprehensive)
# ------------------------------------------------------------------------
"""
# requirements.txt for Unified Autonomous Video Editor v2.0

# Core
opencv-python>=4.6.0
numpy>=1.21.0,<2.0.0
scipy>=1.8.0

# Audio/Video
librosa>=0.9.1,<0.11.0 # Version 0.9.1+ preferred for some features
soundfile>=0.11.0
moviepy>=1.0.3

# Machine Learning / Advanced Algos
torch>=1.12.0 # Check CUDA/MPS compatibility for your system
torchvision>=0.13.0

# Optimal Transport (Required for MathematicalGenius Planner)
# Install separately: pip install POT
# pot>=0.8.0

# MediaPipe
mediapipe>=0.10.0,<0.11.0

# Effects Dependencies (Install based on needs)
# For SAM v1:
# pip install segment-anything
# For SAM v2:
# pip install ultralytics>=8.0.0
# For RVM (check torch hub requirements, might include timm):
# pip install timm>=0.6.0

# NOTE 1: Ensure FFmpeg executable is installed and in your system's PATH.
# Download from: https://ffmpeg.org/download.html

# NOTE 2: Download required model checkpoints (SAM, RVM if not using direct hub loading)
# and provide paths via arguments or modify RenderConfig defaults.
# - SAM v1/v2 checkpoints: Check Meta AI / Ultralytics resources.
# - RVM checkpoints: Check PeterL1n's repository or use Torch Hub names.

# NOTE 3: For GPU acceleration (CUDA/MPS), ensure PyTorch is installed correctly
# for your specific hardware and drivers. Check PyTorch official website.
"""
