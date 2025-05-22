
# -*- coding: utf-8 -*-
# ========================================================================
#                       IMPORTS (Comprehensive & Updated)
# ========================================================================
import tkinter
from tkinter import filedialog, Listbox, Scrollbar, END, MULTIPLE, Frame, messagebox, BooleanVar
import customtkinter
import cv2 # <<< Still needed for properties, image ops, optical flow
import math
import numpy as np
import time
import os
import json
import threading
import concurrent.futures
import traceback
import sys
import shlex
import random # For probabilistic selection and simulation
from collections import defaultdict, namedtuple
from math import exp, log # For sigmoid and entropy
import subprocess # For FFmpeg execution (audio extraction, potentially fallback)
import logging # <<< Added for structured logging
from dataclasses import dataclass, field # <<< Added for structured config
from typing import List, Tuple, Dict, Any, Optional, Set, Union # <<< Added Union for typing
import tracemalloc # For memory tracking during render

# --- Core Analysis & Editing ---
import mediapipe as mp
import librosa
# import librosa.display # Only for debug plotting if needed
import soundfile # Needed for audio loading/saving
# Removed moviepy imports
import scipy.signal # For audio filtering helpers
import scipy.stats # For entropy calculation

# --- Advanced Features ---
import torch # For MiDaS

# --- UI & Utilities ---
from tkinterdnd2 import DND_FILES, TkinterDnD
import multiprocessing
# from tqdm import tqdm # Import moved inside functions where needed
import matplotlib # For backend setting
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
# import matplotlib.pyplot as plt # For potential debug plotting (commented out if not used)

# --- VidGear Integration ---
from vidgear.gears import VideoGear  # <<< For efficient frame reading
# from vidgear.gears import FFmpegizer # <<< REMOVED: Deprecated
from vidgear.gears import WriteGear  # <<< ADDED: Replaces FFmpegizer for writing

# --- MediaPipe Solutions needed globally ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Constants ---
MIN_POTENTIAL_CLIP_DURATION_FRAMES = 5
# Default normalization constants (can be overridden by config)
V_MAX_EXPECTED = 50.0
A_MAX_EXPECTED = 100.0
D_MAX_EXPECTED = 0.15
E_MAX_EXPECTED = 100.0
SIGMA_M_DEFAULT = 0.2
DEFAULT_REPETITION_PENALTY = 0.3
DEFAULT_NORM_MAX_RMS = 0.5
DEFAULT_NORM_MAX_ONSET = 10.0
DEFAULT_NORM_MAX_CHROMA_VAR = 0.1

# --- Process-local Cache for Models (Attempt) ---
_midas_model_cache = None
_midas_transform_cache = None
_midas_device_cache = None

# --- Logging Setup ---
# Setup moved to the main block to ensure configuration happens early
logger = logging.getLogger(__name__) # Get logger instance

# ========================================================================
#                       DATA CLASSES FOR CONFIGURATION
# ========================================================================
@dataclass
class AnalysisConfig:
    """Configuration for the analysis phase."""
    # Shared Params
    min_sequence_clip_duration: float = 0.75
    max_sequence_clip_duration: float = 5.0
    resolution_height: int = 256
    resolution_width: int = 256
    save_analysis_data: bool = True

    # Normalization
    norm_max_velocity: float = V_MAX_EXPECTED
    norm_max_acceleration: float = A_MAX_EXPECTED
    norm_max_rms: float = DEFAULT_NORM_MAX_RMS
    norm_max_onset: float = DEFAULT_NORM_MAX_ONSET
    norm_max_chroma_var: float = DEFAULT_NORM_MAX_CHROMA_VAR
    norm_max_depth_variance: float = D_MAX_EXPECTED
    # Heuristic Normalization (potentially separate, but using shared for now)
    norm_max_kinetic: float = 50.0 # Heuristic default
    norm_max_jerk: float = 30.0 # Heuristic default
    norm_max_cam_motion: float = 5.0 # Heuristic default
    norm_max_face_size: float = 0.6 # Heuristic default

    # Detection
    min_face_confidence: float = 0.5
    mouth_open_threshold: float = 0.05
    min_pose_confidence: float = 0.5
    model_complexity: int = 1 # Pose model complexity (0, 1, 2)

    # Segment ID (Heuristic)
    use_heuristic_segment_id: bool = True # Whether to use heuristic score for potential clip ID
    score_threshold: float = 0.30
    min_potential_clip_duration_sec: float = 0.4

    # Sequencing Mode (Added here for completeness, might fit better elsewhere)
    sequencing_mode: str = "Greedy Heuristic" # or "Physics Pareto MC"

    # --- Greedy Heuristic Specific ---
    audio_weight: float = 0.3
    kinetic_weight: float = 0.25
    sharpness_weight: float = 0.1 # Jerk proxy
    camera_motion_weight: float = 0.05
    face_size_weight: float = 0.1
    percussive_weight: float = 0.05
    depth_weight: float = 0.1 # Heuristic depth use
    beat_boost: float = 0.15
    beat_boost_radius_sec: float = 0.1
    pacing_variation_factor: float = 0.25
    variety_penalty_source: float = 0.20
    variety_penalty_shot: float = 0.15
    variety_penalty_intensity: float = 0.10
    beat_sync_bonus: float = 0.15
    section_match_bonus: float = 0.10
    candidate_pool_size: int = 15
    intensity_thresholds: Tuple[float, float] = (0.3, 0.7)

    # --- Physics Pareto MC Specific ---
    # Fit Weights
    fit_weight_velocity: float = 0.3
    fit_weight_acceleration: float = 0.3
    fit_weight_mood: float = 0.4
    fit_sigmoid_steepness: float = 1.0
    # Objective Weights
    objective_weight_rhythm: float = 1.0
    objective_weight_mood: float = 1.0
    objective_weight_continuity: float = 0.8
    objective_weight_variety: float = 0.7
    objective_weight_efficiency: float = 0.5
    # Sequencing & Evaluation
    mc_iterations: int = 500
    mood_similarity_variance: float = SIGMA_M_DEFAULT
    continuity_depth_weight: float = 0.5 # k_d in DeltaE
    variety_repetition_penalty: float = DEFAULT_REPETITION_PENALTY # Lambda


@dataclass
class EffectParams:
    """Parameters for a specific effect type."""
    type: str = "cut"
    tau: float = 0.0 # Duration
    psi: float = 0.0 # Physical impact proxy
    epsilon: float = 0.0 # Perceptual gain

@dataclass
class RenderConfig:
    """Configuration for the rendering phase."""
    # Inherit or duplicate relevant normalization constants if needed by effects
    norm_max_velocity: float = V_MAX_EXPECTED
    norm_max_acceleration: float = A_MAX_EXPECTED

    # Effect definitions (could be loaded from AnalysisConfig or set separately)
    effect_settings: Dict[str, EffectParams] = field(default_factory=lambda: {
        "cut": EffectParams(type="cut"),
        "fade": EffectParams(type="fade", tau=0.2, psi=0.1, epsilon=0.2),
        "zoom": EffectParams(type="zoom", tau=0.5, psi=0.3, epsilon=0.4),
        "pan": EffectParams(type="pan", tau=0.5, psi=0.1, epsilon=0.3),
    })

    # FFmpeg Output Settings
    video_codec: str = 'libx264'
    preset: str = 'medium'
    crf: int = 23
    audio_codec: str = 'aac'
    audio_bitrate: str = '192k'
    threads: int = max(1, (os.cpu_count() or 2) // 2)
    ffmpeg_loglevel: str = 'warning' # FFmpeg's log level
    # Add any other render-specific options here

# ========================================================================
#                       HELPER FUNCTIONS (Logging Added)
# ========================================================================
def tk_write(tk_string1, parent=None, level="info"):
    """Shows message box. Logs the message. level can be 'info', 'warning', 'error'."""
    # Log the message regardless of whether the messagebox appears
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, f"Popup ({level}): {tk_string1}")

    try:
        if parent and hasattr(parent, 'winfo_exists') and parent.winfo_exists():
            root_to_use = parent
            use_messagebox = True
        else:
            logger.warning("tk_write called without a valid parent window. Message only logged.")
            use_messagebox = False

        if use_messagebox:
            title = f"Videous Chef - {level.capitalize()}"
            if level == "error":
                messagebox.showerror(title, tk_string1, parent=parent)
            elif level == "warning":
                messagebox.showwarning(title, tk_string1, parent=parent)
            else: # info
                messagebox.showinfo(title, tk_string1, parent=parent)

    except Exception as e:
        logger.error(f"tk_write internal error: {e}", exc_info=True)
        # Fallback print if logging itself failed or for visibility
        print(f"!! tk_write Error: {e}\n!! Level: {level}\n!! Message: {tk_string1}")


def get_midas_model(model_type="MiDaS_small"):
    """Loads MiDaS model and transform, attempting process-local caching."""
    global _midas_model_cache, _midas_transform_cache, _midas_device_cache
    if _midas_model_cache and _midas_transform_cache and _midas_device_cache:
        logger.debug("Using cached MiDaS model.")
        return _midas_model_cache, _midas_transform_cache, _midas_device_cache

    logger.info(f"Loading MiDaS model: {model_type}...")
    try:
        # --- Device Selection ---
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Added is_built check
             device = torch.device("mps"); logger.info("Using Apple Metal (MPS) backend.")
        elif torch.cuda.is_available():
             device = torch.device("cuda"); logger.info("Using CUDA backend.")
        else:
             device = torch.device("cpu"); logger.info("Using CPU backend.")
        _midas_device_cache = device

        # --- Load Model & Transform ---
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        model.to(device); model.eval()
        _midas_model_cache = model

        transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms_hub.small_transform if "small" in model_type else transforms_hub.dpt_transform
        _midas_transform_cache = transform

        logger.info("MiDaS model loaded successfully.")
        return _midas_model_cache, _midas_transform_cache, _midas_device_cache

    except ImportError as import_err:
        logger.error(f"Failed to load MiDaS model. Dependency 'timm' might be missing: {import_err}")
        logger.error("Please install it: pip install timm")
        _midas_model_cache, _midas_transform_cache, _midas_device_cache = None, None, None
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to load MiDaS model: {e}. Depth analysis disabled.", exc_info=True)
        _midas_model_cache, _midas_transform_cache, _midas_device_cache = None, None, None
        return None, None, None

# --- Physics/Math Helpers (Unchanged, logging not critical here) ---
def sigmoid(x, k=1):
    try:
        x_clamped = np.clip(x, -500, 500)
        return 1 / (1 + exp(-k * x_clamped))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    vec1 = np.asarray(vec1, dtype=float); vec2 = np.asarray(vec2, dtype=float)
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    return np.clip(similarity, -1.0, 1.0)


def calculate_histogram_entropy(frame):
    if frame is None or frame.size == 0: return 0.0
    try:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:
            gray = frame
        else:
            logger.warning(f"Invalid frame shape for histogram entropy: {frame.shape}")
            return 0.0

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_sum = hist.sum()
        if hist_sum == 0: return 0.0
        hist_norm = hist.ravel() / hist_sum
        entropy = scipy.stats.entropy(hist_norm)
        return entropy if np.isfinite(entropy) else 0.0
    except Exception as e:
        logger.warning(f"Histogram entropy calculation failed: {e}", exc_info=False) # Don't need full trace often
        return 0.0

# ========================================================================
#           AUDIO ANALYSIS UTILITIES (Logging Enhanced)
# ========================================================================
class PegasusAudioUtils:
    def extract_audio(self, video_path, audio_output_path="temp_audio.wav"):
        """Extracts audio from video file using FFmpeg."""
        logger.info(f"Extracting audio from '{os.path.basename(video_path)}' using FFmpeg...")
        start_time = time.time()
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found for audio extraction: {video_path}")
                raise FileNotFoundError(f"Video file not found: {video_path}")

            output_dir = os.path.dirname(audio_output_path)
            if output_dir: os.makedirs(output_dir, exist_ok=True)

            command = [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-y",
                audio_output_path,
                "-hide_banner", "-loglevel", "error" # Keep FFmpeg quiet unless error
            ]
            logger.debug(f"Executing FFmpeg: {' '.join(command)}")

            result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8') # Specify encoding

            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction failed (Code: {result.returncode}) for {os.path.basename(video_path)}.")
                logger.error(f"FFmpeg stderr:\n{result.stderr}")
                if os.path.exists(audio_output_path):
                    try:
                        os.remove(audio_output_path)
                    except OSError as del_err:
                        logger.warning(f"Could not remove failed temp audio file {audio_output_path}: {del_err}")
                return None
            else:
                if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
                    logger.error(f"FFmpeg ran but output file '{audio_output_path}' not created or is empty.")
                    logger.error(f"FFmpeg stderr (despite return code 0):\n{result.stderr}")
                    return None
                else:
                    logger.info(f"Audio extracted to '{os.path.basename(audio_output_path)}' ({time.time() - start_time:.2f}s)")
                    return audio_output_path

        except FileNotFoundError as fnf_err:
            logger.error(f"File not found during audio extraction setup: {fnf_err}")
            return None
        except Exception as e:
            logger.error(f"Error during FFmpeg audio extraction '{os.path.basename(video_path)}': {e}", exc_info=True)
            if os.path.exists(audio_output_path):
                try:
                    os.remove(audio_output_path)
                except OSError as del_err:
                    logger.warning(f"Could not remove temp audio file {audio_output_path}: {del_err}")
            return None

    # Note: analysis_config is now expected to be an AnalysisConfig object
    def analyze_audio(self, audio_path: str, analysis_config: AnalysisConfig, target_sr: int = 22050, segment_hop_factor: int = 4) -> Optional[Dict[str, Any]]:
        """Calculates enhanced audio features using AnalysisConfig."""
        logger.info(f"Analyzing audio (Physics-Based): {os.path.basename(audio_path)}")
        start_time = time.time()
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found for analysis: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            logger.debug("Loading audio file with Librosa...")
            y, sr = librosa.load(audio_path, sr=target_sr)
            duration = librosa.get_duration(y=y, sr=sr)
            hop_length = 512 # For features

            logger.debug("Computing Onset Strength...")
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

            logger.debug("Tracking Beats & Tempo...")
            tempo, beat_samples = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units='samples', tightness=100)
            beat_times = librosa.samples_to_time(beat_samples, sr=sr)

            # FIX DeprecationWarning & Ensure tempo_float is derived safely
            if isinstance(tempo, (np.ndarray, np.generic)):
                try:
                    # Use item() if it's a 0-dim array or numpy scalar
                    tempo_float = float(tempo.item())
                except ValueError:
                    # If item() fails (e.g., array with >1 element), log warning and maybe take first element or average?
                    logger.warning(f"Tempo detection returned non-scalar array: {tempo}. Using first element.")
                    tempo_float = float(tempo[0]) if tempo.size > 0 else 120.0 # Fallback tempo
            else:
                # If it's already a standard Python float/int
                tempo_float = float(tempo)

            logger.debug("Computing RMS energy...")
            rms_energy = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
            rms_times = librosa.times_like(rms_energy, sr=sr, hop_length=hop_length)

            logger.debug("Computing Chroma Features...")
            chroma_hop = hop_length * segment_hop_factor
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=chroma_hop)
            try:
                 chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=chroma_hop)
            except Exception as time_err:
                 logger.warning(f"Could not calculate precise chroma times: {time_err}. Using approximation.")
                 chroma_times = np.linspace(0, duration, chroma.shape[1])

            logger.debug("Analyzing musical structure (segmentation)...")
            bound_times = self._segment_audio(chroma, sr, chroma_hop, duration)

            segment_features = []
            logger.debug(f"Aggregating features for {len(bound_times)-1} segments...")
            for i in range(len(bound_times) - 1):
                t_start, t_end = bound_times[i], bound_times[i+1]
                seg_duration = t_end - t_start
                if seg_duration <= 0: continue

                rms_indices = np.where((rms_times >= t_start) & (rms_times < t_end))[0]
                seg_rms = np.mean(rms_energy[rms_indices]) if len(rms_indices) > 0 else 0.0
                onset_indices = np.where((onset_times >= t_start) & (onset_times < t_end))[0]
                seg_onset = np.mean(onset_env[onset_indices]) if len(onset_indices) > 0 else 0.0
                chroma_indices = np.where((chroma_times >= t_start) & (chroma_times < t_end))[0]
                seg_chroma_var = np.mean(np.var(chroma[:, chroma_indices], axis=1)) if len(chroma_indices) > 0 else 0.0

                segment_features.append({
                    'start': t_start, 'end': t_end, 'duration': seg_duration,
                    'rms_avg': seg_rms, 'onset_avg': seg_onset, 'chroma_var': seg_chroma_var
                })

            logger.info(f"Audio analysis complete ({time.time() - start_time:.2f}s). Tempo: {tempo_float:.2f} BPM, Beats: {len(beat_times)}, Segments: {len(segment_features)}")

            # Normalize using values from the AnalysisConfig dataclass
            norm_max_rms = analysis_config.norm_max_rms
            norm_max_onset = analysis_config.norm_max_onset
            norm_max_chroma_var = analysis_config.norm_max_chroma_var

            for seg in segment_features:
                 seg['b_i'] = np.clip(seg['rms_avg'] / (norm_max_rms + 1e-6), 0.0, 1.0)
                 seg['e_i'] = np.clip(seg['onset_avg'] / (norm_max_onset + 1e-6), 0.0, 1.0)
                 arousal_proxy = np.clip(tempo_float / 150.0, 0.0, 1.0) # Tempo range assumption
                 valence_proxy = np.clip(1.0 - (seg['chroma_var'] / (norm_max_chroma_var + 1e-6)), 0.0, 1.0)
                 seg['m_i'] = [arousal_proxy, valence_proxy]

            # Store raw features potentially needed by heuristic score calculation later
            # (This assumes heuristic mode might need frame-level RMS, not just segment avg)
            raw_features = {
                 'rms_energy': rms_energy.tolist(),
                 'rms_times': rms_times.tolist(),
                 # Add others like 'percussive_ratio' if calculated
            }

            return {
                "sr": sr, "duration": duration, "tempo": tempo_float,
                "beat_times": beat_times.tolist(),
                "segment_boundaries": bound_times,
                "segment_features": segment_features, # For Physics Mode
                "raw_features": raw_features # For Heuristic Mode frame scoring
            }
        except FileNotFoundError as fnf_err:
            logger.error(f"Audio analysis failed - file not found: {fnf_err}")
            return None
        except Exception as e:
            logger.error(f"Error during audio analysis: {e}", exc_info=True)
            return None

    def _segment_audio(self, features, sr, hop_length, duration, k_nn=5, backtrack=False):
        """Helper to find segment boundaries using librosa recurrence."""
        try:
            R = librosa.segment.recurrence_matrix(features, k=k_nn, width=3, mode='connectivity', sym=True)
            def medfilt2d_wrapper(data, kernel_size):
                data_float = data.astype(float) if data.dtype == bool else data
                return scipy.signal.medfilt2d(data_float, kernel_size=kernel_size)
            df = librosa.segment.timelag_filter(lambda data, **kwargs: medfilt2d_wrapper(data, kernel_size=kwargs.get('size')))
            Rf = df(R, size=(1, 7))
            # FIX: Remove 'metric' argument as it caused TypeError
            # Check librosa version for correct API usage
            if hasattr(librosa.segment, 'agglomerative') and 'metric' in librosa.segment.agglomerative.__code__.co_varnames:
                 bounds_frames = librosa.segment.agglomerative(features, k=None, metric="cosine")
                 logger.debug("Using librosa.segment.agglomerative with metric='cosine'.")
            else:
                 # Fallback for older librosa versions or if metric isn't accepted with k=None
                 logger.debug("Using librosa.segment.agglomerative without explicit metric (older librosa?).")
                 bounds_frames = librosa.segment.agglomerative(features, k=None)

            bound_times = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop_length).tolist()
            final_bounds = sorted(list(set([0.0] + bound_times + [duration])))
            min_len_sec = 0.5
            final_bounds_filtered = [final_bounds[0]]
            for i in range(1, len(final_bounds)):
                if final_bounds[i] - final_bounds_filtered[-1] >= min_len_sec:
                    final_bounds_filtered.append(final_bounds[i])
                elif i == len(final_bounds) - 1: # Ensure last boundary included if close
                    final_bounds_filtered[-1] = final_bounds[i]

            # Adjust final boundary if it doesn't reach the end
            if final_bounds_filtered[-1] < duration:
                 if duration - final_bounds_filtered[-1] >= min_len_sec / 2: # Allow slightly shorter last seg
                     final_bounds_filtered.append(duration)
                 else: # Merge last segment if too short
                     final_bounds_filtered[-1] = duration

            # Ensure first boundary is 0.0
            if not final_bounds_filtered or final_bounds_filtered[0] > 0.0:
                 final_bounds_filtered.insert(0, 0.0)

            return final_bounds_filtered

        except TypeError as te: # Catch specific TypeError
            logger.warning(f"Audio segmentation failed (TypeError: {te}). Likely metric issue. Falling back to fixed segments.", exc_info=False)
            num_segments = max(2, int(duration / 10.0)) # Ensure at least 2 segments
            return np.linspace(0, duration, num_segments + 1).tolist()
        except Exception as e:
            logger.warning(f"Audio segmentation failed ({type(e).__name__}: {e}). Falling back to fixed segments.", exc_info=False)
            num_segments = max(2, int(duration / 10.0)) # Ensure at least 2 segments
            return np.linspace(0, duration, num_segments + 1).tolist()

    # _plot_structure remains unchanged

# ========================================================================
#                    FACE UTILITIES (Logging Added)
# ========================================================================
class PegasusFaceUtils:
    def __init__(self, static_mode=False, max_faces=1, min_detect_conf=0.5, min_track_conf=0.5):
        self.face_mesh = None
        self._mp_face_mesh = mp.solutions.face_mesh
        try:
            self.face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=static_mode, max_num_faces=max_faces,
                refine_landmarks=True, min_detection_confidence=min_detect_conf,
                min_tracking_confidence=min_track_conf)
            logger.info("FaceMesh initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FaceMesh: {e}", exc_info=True)

    def get_face_features(self, image, mouth_open_threshold=0.05):
        if self.face_mesh is None: return False, 0.0, 0.5
        is_open, size_ratio, center_x = False, 0.0, 0.5
        h, w = image.shape[:2]
        if h == 0 or w == 0: return False, 0.0, 0.5

        try:
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            image.flags.writeable = True

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                lm = face_landmarks.landmark
                # Mouth
                upper_lip_y = lm[13].y * h; lower_lip_y = lm[14].y * h
                mouth_height = abs(lower_lip_y - upper_lip_y)
                # Face Height
                forehead_y = lm[10].y * h; chin_y = lm[152].y * h
                face_height = abs(chin_y - forehead_y)
                if face_height > 1e-6:
                    mouth_open_ratio = mouth_height / face_height
                    is_open = mouth_open_ratio > mouth_open_threshold
                # Face Size
                all_x = [lm_pt.x * w for lm_pt in lm]; all_y = [lm_pt.y * h for lm_pt in lm]
                face_box_w = max(all_x) - min(all_x); face_box_h = max(all_y) - min(all_y)
                face_diagonal = math.sqrt(face_box_w**2 + face_box_h**2)
                image_diagonal = math.sqrt(w**2 + h**2)
                if image_diagonal > 1e-6:
                    size_ratio = np.clip(face_diagonal / image_diagonal, 0.0, 1.0)
                # Center X
                left_cheek_x = lm[234].x; right_cheek_x = lm[454].x
                center_x = np.clip((left_cheek_x + right_cheek_x) / 2.0, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Error processing frame with FaceMesh: {e}", exc_info=False)
            return False, 0.0, 0.5

        return is_open, size_ratio, center_x

    def close(self):
        if self.face_mesh:
            try:
                self.face_mesh.close()
                logger.info("FaceMesh resources released.")
            except Exception as e:
                logger.error(f"Error closing FaceMesh: {e}")
        self.face_mesh = None

# ========================================================================
#          POSE/VISUAL UTILITIES (Logging Added for errors)
# ========================================================================
def calculate_flow_velocity(prev_gray, current_gray):
    if prev_gray is None or current_gray is None or prev_gray.shape != current_gray.shape:
        return 0.0, None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if flow is None: return 0.0, None
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        scale_factor = 10.0
        return avg_magnitude * scale_factor, flow
    except cv2.error as cv_err:
         logger.warning(f"OpenCV error during flow calculation: {cv_err}")
         return 0.0, None
    except Exception as e:
        logger.warning(f"Generic error during flow velocity calculation: {e}")
        return 0.0, None

def calculate_flow_acceleration(prev_flow, current_flow, dt):
    if prev_flow is None or current_flow is None or prev_flow.shape != current_flow.shape or dt <= 0:
        return 0.0
    try:
        flow_diff = current_flow - prev_flow
        accel_magnitude_per_pixel, _ = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
        avg_accel_magnitude = np.mean(accel_magnitude_per_pixel)
        accel = avg_accel_magnitude / dt
        scale_factor = 10.0
        return accel * scale_factor
    except Exception as e:
        logger.warning(f"Error during flow acceleration calculation: {e}")
        return 0.0

def calculate_kinetic_energy_proxy(landmarks_prev, landmarks_curr, dt):
    if landmarks_prev is None or landmarks_curr is None or dt <= 0: return 0.0
    if not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0
    lm_prev = landmarks_prev.landmark
    lm_curr = landmarks_curr.landmark
    if len(lm_prev) != len(lm_curr): return 0.0

    total_sq_velocity = 0.0
    num_valid = 0
    try:
        for i in range(len(lm_prev)):
            if i < len(lm_curr): # Safety check
                dx = lm_curr[i].x - lm_prev[i].x
                dy = lm_curr[i].y - lm_prev[i].y
                dz = lm_curr[i].z - lm_prev[i].z
                # Note: z is relative depth, might not be scaled correctly for true velocity
                # Consider visibility if available: if lm_curr[i].visibility > 0.5:
                vx = dx / dt
                vy = dy / dt
                vz = dz / dt # Less reliable component
                total_sq_velocity += vx**2 + vy**2 + vz**2 # Approximate v^2
                num_valid += 1
    except IndexError:
         logger.warning("Index error accessing landmarks in kinetic energy calc.")
         return 0.0 # Or handle differently

    if num_valid == 0: return 0.0
    # Return average squared velocity (proportional to kinetic energy if mass is constant)
    # Scale factor helps bring values to a reasonable range
    scale_factor = 1000.0
    return (total_sq_velocity / num_valid) * scale_factor


def calculate_movement_jerk_proxy(landmarks_prev_prev, landmarks_prev, landmarks_curr, dt):
    if landmarks_prev_prev is None or landmarks_prev is None or landmarks_curr is None or dt <= 0: return 0.0
    if not hasattr(landmarks_prev_prev, 'landmark') or \
       not hasattr(landmarks_prev, 'landmark') or \
       not hasattr(landmarks_curr, 'landmark'): return 0.0

    lm_pp = landmarks_prev_prev.landmark
    lm_p = landmarks_prev.landmark
    lm_c = landmarks_curr.landmark
    if len(lm_pp) != len(lm_p) or len(lm_p) != len(lm_c): return 0.0

    total_sq_jerk = 0.0
    num_valid = 0
    dt_sq = dt * dt
    try:
        for i in range(len(lm_pp)):
            if i < len(lm_p) and i < len(lm_c): # Safety check
                # Approx acceleration change (finite difference of velocity)
                ax = ((lm_c[i].x - lm_p[i].x) / dt) - ((lm_p[i].x - lm_pp[i].x) / dt)
                ay = ((lm_c[i].y - lm_p[i].y) / dt) - ((lm_p[i].y - lm_pp[i].y) / dt)
                az = ((lm_c[i].z - lm_p[i].z) / dt) - ((lm_p[i].z - lm_pp[i].z) / dt)
                jx = ax / dt
                jy = ay / dt
                jz = az / dt
                total_sq_jerk += jx**2 + jy**2 + jz**2 # Approximate jerk^2
                num_valid += 1
    except IndexError:
        logger.warning("Index error accessing landmarks in jerk calc.")
        return 0.0

    if num_valid == 0: return 0.0
    scale_factor = 10000.0 # Adjust scale factor as needed
    return (total_sq_jerk / num_valid) * scale_factor

class PegasusPoseUtils:
    def drawLandmarksOnImage(self, imageInput, poseProcessingInput):
        annotated_image = imageInput.copy()
        if poseProcessingInput and poseProcessingInput.pose_landmarks:
            try:
                mp_drawing.draw_landmarks(
                    annotated_image, poseProcessingInput.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            except Exception as e:
                logger.warning(f"Failed to draw pose landmarks: {e}")
        return annotated_image

# ========================================================================
#         DYNAMIC SEGMENT IDENTIFIER (Logging Added)
# ========================================================================
class DynamicSegmentIdentifier:
    def __init__(self, analysis_config: AnalysisConfig, fps: float):
        self.fps = fps
        self.score_threshold = analysis_config.score_threshold
        self.min_segment_len_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES,
                                           int(analysis_config.min_potential_clip_duration_sec * fps))

    def find_potential_segments(self, frame_features_list):
        potential_segments = []
        start_idx = -1
        n = len(frame_features_list)
        for i, features in enumerate(frame_features_list):
            score = features.get('boosted_score', 0.0)
            is_candidate = score >= self.score_threshold
            if is_candidate and start_idx == -1:
                start_idx = i
            elif not is_candidate and start_idx != -1:
                if (i - start_idx) >= self.min_segment_len_frames:
                    potential_segments.append({'start_frame': start_idx, 'end_frame': i})
                start_idx = -1
        if start_idx != -1 and (n - start_idx) >= self.min_segment_len_frames:
            potential_segments.append({'start_frame': start_idx, 'end_frame': n})

        logger.info(f"Identified {len(potential_segments)} potential segments (Heuristic Score >= {self.score_threshold:.2f}, MinLen={self.min_segment_len_frames / self.fps:.2f}s)")
        return potential_segments

# ========================================================================
#                      IMAGE UTILITIES (Logging Added)
# ========================================================================
class PegasusImageUtils:
    def resizeTARGET(self, image, TARGET_HEIGHT=256, TARGET_WIDTH=256):
        if image is None or image.size == 0:
            logger.warning("Resize received empty image.")
            return None
        h, w = image.shape[:2]
        if h == 0 or w == 0: return image
        if h > TARGET_HEIGHT or w > TARGET_WIDTH:
            scale = min(TARGET_HEIGHT / h, TARGET_WIDTH / w)
            new_w, new_h = int(w * scale), int(h * scale)
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            try:
                return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            except Exception as e:
                logger.warning(f"Resize failed: {e}.")
                return image # Return original on error
        return image

# ========================================================================
#               CLIP SEGMENT DATA STRUCTURE (Uses AnalysisConfig)
# ========================================================================
class ClipSegment:
    def __init__(self, source_video_path: str, start_frame: int, end_frame: int, fps: float,
                 all_frame_features: List[Dict], analysis_config: AnalysisConfig):
        self.source_video_path = source_video_path
        self.start_frame = start_frame
        self.end_frame = end_frame # Exclusive
        self.num_frames = end_frame - start_frame
        self.fps = fps if fps > 0 else 30.0
        self.start_time = start_frame / self.fps
        self.end_time = end_frame / self.fps
        self.duration = self.num_frames / self.fps
        self.analysis_config = analysis_config # Store config dataclass

        if 0 <= start_frame < len(all_frame_features) and end_frame <= len(all_frame_features):
             self.segment_frame_features = all_frame_features[start_frame:end_frame]
        else:
             logger.warning(f"Invalid frame indices for ClipSegment: {start_frame}-{end_frame} (Total: {len(all_frame_features)})")
             self.segment_frame_features = []

        # --- Heuristic Features ---
        self.avg_raw_score = 0.0; self.avg_boosted_score = 0.0; self.peak_boosted_score = 0.0
        self.avg_motion_heuristic = 0.0; self.avg_jerk_heuristic = 0.0; self.avg_camera_motion = 0.0
        self.face_presence_ratio = 0.0; self.avg_face_size = 0.0
        self.intensity_category = "Low"; self.dominant_contributor = "none"; self.contains_beat = False
        self.musical_section_indices: Set[int] = set()

        # --- Physics-based Features ---
        self.v_k = 0.0; self.a_j = 0.0; self.d_r = 0.0; self.phi = 0.0
        self.mood_vector = [0.0, 0.0]

        # --- Runtime Variables ---
        self.sequence_start_time = -1.0; self.sequence_end_time = -1.0
        self.chosen_duration = self.duration
        self.chosen_effect: Optional[EffectParams] = None # Updated type hint
        self.temp_chosen_duration = self.duration # Used by Greedy
        # Subclip timing for rendering
        self.subclip_start_time_in_source = self.start_time
        self.subclip_end_time_in_source = self.end_time

        if self.segment_frame_features:
            self._calculate_heuristic_aggregates()
            self._calculate_physics_aggregates()
        else:
             logger.warning(f"Cannot calculate aggregates for segment {start_frame}-{end_frame} due to empty features.")

    def _calculate_heuristic_aggregates(self):
        count = len(self.segment_frame_features)
        if count == 0: return

        # Access config via self.analysis_config.attribute
        norm_kinetic = self.analysis_config.norm_max_kinetic
        norm_jerk = self.analysis_config.norm_max_jerk
        norm_cam_motion = self.analysis_config.norm_max_cam_motion
        norm_face_size = self.analysis_config.norm_max_face_size

        raw_scores = [f.get('raw_score', 0.0) for f in self.segment_frame_features]
        boosted_scores = [f.get('boosted_score', 0.0) for f in self.segment_frame_features]
        # Use 'camera_motion' which is derived from flow_velocity in analyzeVideo
        cam_motions = [f.get('camera_motion', 0.0) for f in self.segment_frame_features]
        # Extract pose features safely
        motions = [f['pose_features'].get('kinetic_energy_proxy', 0.0) for f in self.segment_frame_features if 'pose_features' in f]
        jerks = [f['pose_features'].get('movement_jerk_proxy', 0.0) for f in self.segment_frame_features if 'pose_features' in f]
        face_sizes = [f['pose_features'].get('face_size_ratio', 0.0) for f in self.segment_frame_features if 'pose_features' in f and f['pose_features'].get('face_size_ratio', 0.0) > 0.01]
        face_present_frames = sum(1 for f in self.segment_frame_features if 'pose_features' in f and f['pose_features'].get('face_size_ratio', 0.0) > 0.01)

        beats = [f.get('is_beat_frame', False) for f in self.segment_frame_features]
        contributors = [f.get('dominant_contributor', 'none') for f in self.segment_frame_features]
        intensities = [f.get('intensity_category', 'Low') for f in self.segment_frame_features]
        sections = {f.get('musical_section_index', -1) for f in self.segment_frame_features if f.get('musical_section_index', -1) != -1}

        self.avg_raw_score = np.mean(raw_scores) if raw_scores else 0.0
        self.avg_boosted_score = np.mean(boosted_scores) if boosted_scores else 0.0
        self.peak_boosted_score = np.max(boosted_scores) if boosted_scores else 0.0
        # Calculate averages based on normalized values if desired, or use raw averages
        # Using raw averages here for simplicity, normalization happens elsewhere if needed
        self.avg_motion_heuristic = np.mean(motions) if motions else 0.0
        self.avg_jerk_heuristic = np.mean(jerks) if jerks else 0.0
        self.avg_camera_motion = np.mean(cam_motions) if cam_motions else 0.0

        self.face_presence_ratio = face_present_frames / count if count > 0 else 0.0
        self.avg_face_size = np.mean(face_sizes) if face_sizes else 0.0
        self.contains_beat = any(beats)
        self.musical_section_indices = sections

        if contributors:
             non_none = [c for c in contributors if c != 'none']
             if non_none: self.dominant_contributor = max(set(non_none), key=non_none.count)

        intensity_order = ['Low', 'Medium', 'High']
        if intensities:
            highest_intensity = 'Low'
            for intensity in intensities:
                 try:
                     current_idx = intensity_order.index(intensity)
                     highest_idx = intensity_order.index(highest_intensity)
                     if current_idx > highest_idx: highest_intensity = intensity
                 except ValueError: pass
            self.intensity_category = highest_intensity

    def _calculate_physics_aggregates(self):
        count = len(self.segment_frame_features)
        if count == 0: return

        # Note: flow_velocity and flow_acceleration are calculated per frame in analyzeVideo
        velocities = [f.get('flow_velocity', 0.0) for f in self.segment_frame_features if f.get('flow_velocity') is not None]
        accelerations = [f.get('flow_acceleration', 0.0) for f in self.segment_frame_features if f.get('flow_acceleration') is not None]
        depth_vars = [f.get('depth_variance', 0.0) for f in self.segment_frame_features if f.get('depth_variance') is not None]
        entropies = [f.get('histogram_entropy', 0.0) for f in self.segment_frame_features if f.get('histogram_entropy') is not None]

        self.v_k = np.mean(velocities) if velocities else 0.0
        self.a_j = np.mean(accelerations) if accelerations else 0.0

        avg_depth_var = np.mean(depth_vars) if depth_vars else 0.0
        # Use config dataclass attribute
        d_max_norm = self.analysis_config.norm_max_depth_variance
        self.d_r = np.clip(avg_depth_var / (d_max_norm + 1e-6), 0.0, 1.0) # Normalized Depth Variance
        self.phi = np.mean(entropies) if entropies else 0.0 # Average frame entropy

        # Mood Vector Proxy: [Arousal, Valence]
        v_max_norm = self.analysis_config.norm_max_velocity
        # Arousal proxy based on normalized average velocity
        arousal_proxy = np.clip(self.v_k / (v_max_norm + 1e-6), 0.0, 1.0)
        # Valence proxy based on inverse normalized depth variance (less variance -> potentially calmer/more positive?)
        valence_proxy = 1.0 - self.d_r
        self.mood_vector = [arousal_proxy, valence_proxy]

    # Note: Pass AnalysisConfig here too
    def clip_audio_fit(self, audio_segment: Dict, analysis_config: AnalysisConfig) -> float:
        """Calculates how well this visual clip fits a given audio segment (Physics mode)."""
        w_v = analysis_config.fit_weight_velocity; w_a = analysis_config.fit_weight_acceleration
        w_m = analysis_config.fit_weight_mood; v_max_norm = analysis_config.norm_max_velocity
        a_max_norm = analysis_config.norm_max_acceleration
        sigma_m_sq = analysis_config.mood_similarity_variance**2 * 2 # Variance for mood Gaussian

        audio_beat_strength = audio_segment.get('b_i', 0.0) # Normalized RMS proxy
        audio_energy = audio_segment.get('e_i', 0.0)        # Normalized Onset proxy
        audio_mood = np.asarray(audio_segment.get('m_i', [0.0, 0.0])) # [Arousal, Valence]

        # Normalize visual features based on config
        v_k_norm = np.clip(self.v_k / (v_max_norm + 1e-6), 0.0, 1.0)
        a_j_norm = np.clip(self.a_j / (a_max_norm + 1e-6), 0.0, 1.0)
        vid_mood = np.asarray(self.mood_vector)

        # Mood Similarity Term (Gaussian based on distance)
        mood_dist_sq = np.sum((vid_mood - audio_mood)**2)
        mood_term = exp(-mood_dist_sq / (sigma_m_sq + 1e-9))

        # Combine terms based on weights
        velocity_term = w_v * v_k_norm * audio_beat_strength
        acceleration_term = w_a * a_j_norm * audio_energy
        mood_term_weighted = w_m * mood_term

        score = velocity_term + acceleration_term + mood_term_weighted
        k_sigmoid = analysis_config.fit_sigmoid_steepness
        return sigmoid(score, k=k_sigmoid) # Pass through sigmoid for probability-like value

    # Note: Pass AnalysisConfig here too
    def get_feature_vector(self, analysis_config: AnalysisConfig) -> List[float]:
        """Returns a normalized feature vector [v_k_norm, a_j_norm, d_r] for continuity calc."""
        v_max_norm = analysis_config.norm_max_velocity
        a_max_norm = analysis_config.norm_max_acceleration
        v_k_norm = np.clip(self.v_k / (v_max_norm + 1e-6), 0.0, 1.0)
        a_j_norm = np.clip(self.a_j / (a_max_norm + 1e-6), 0.0, 1.0)
        # d_r is already normalized [0, 1] during calculation
        return [v_k_norm, a_j_norm, self.d_r]

    def get_shot_type(self):
        """Categorizes shot type based on face presence and size (Heuristic)."""
        if self.face_presence_ratio < 0.1: return 'wide/no_face'
        if self.avg_face_size < 0.15: return 'medium_wide'
        if self.avg_face_size < 0.35: return 'medium'
        return 'close_up'

    def __repr__(self):
        return (f"ClipSegment({os.path.basename(self.source_video_path)}, "
                f"[{self.start_time:.2f}s-{self.end_time:.2f}s], Dur:{self.duration:.2f}s)\n"
                f"  Heuristic: Score:{self.avg_boosted_score:.2f}, MotH:{self.avg_motion_heuristic:.1f}, Shot:{self.get_shot_type()}\n"
                f"  Physics: V:{self.v_k:.1f}, A:{self.a_j:.1f}, D:{self.d_r:.2f}, Phi:{self.phi:.2f}, Mood:{['{:.2f}'.format(x) for x in self.mood_vector]}")

# ========================================================================
#         MAIN ANALYSIS CLASS (VideousMain) - Uses AnalysisConfig
# ========================================================================
class VideousMain:
    def __init__(self):
        self.pegasusImageUtils = PegasusImageUtils()
        self.pegasusPoseUtils = PegasusPoseUtils()
        self.midas_model, self.midas_transform, self.midas_device = None, None, None

    def _ensure_midas_loaded(self, analysis_config: AnalysisConfig):
        # Check if MiDaS is needed based on config
        needs_midas = (analysis_config.depth_weight > 0 and analysis_config.sequencing_mode == 'Greedy Heuristic') or \
                      (analysis_config.sequencing_mode == 'Physics Pareto MC' and
                       (analysis_config.continuity_depth_weight > 0 or analysis_config.fit_weight_mood > 0)) # Mood uses depth variance

        if self.midas_model is None and needs_midas:
             logger.info("MiDaS model required by configuration, attempting load...")
             self.midas_model, self.midas_transform, self.midas_device = get_midas_model()
             if self.midas_model is None:
                 logger.warning("MiDaS failed to load. Depth features will be disabled.")

    # --- Heuristic Scoring Helpers ---
    def _determine_dominant_contributor(self, norm_features_weighted):
        # (Logic unchanged)
        if not norm_features_weighted: return "unknown"
        max_val = -float('inf'); dominant_key = "none"
        for key, value in norm_features_weighted.items():
            if value > max_val: max_val = value; dominant_key = key
        key_map = {'audio_energy': 'Audio', 'kinetic_proxy': 'Motion', 'jerk_proxy': 'Jerk',
                   'camera_motion': 'CamMove', 'face_size': 'FaceSize', 'percussive': 'Percuss',
                   'depth_variance': 'DepthVar'}
        return key_map.get(dominant_key, dominant_key) if max_val > 1e-4 else "none"

    def _categorize_intensity(self, score, thresholds=(0.3, 0.7)):
        # (Logic unchanged)
        if score < thresholds[0]: return "Low"
        if score < thresholds[1]: return "Medium"
        return "High"

    # Note: Pass AnalysisConfig
    def calculate_heuristic_score(self, frame_features: Dict, analysis_config: AnalysisConfig) -> Tuple[float, str, str, Dict]:
        # Access weights and norms from dataclass
        weights = {
            'audio_energy': analysis_config.audio_weight, 'kinetic_proxy': analysis_config.kinetic_weight,
            'jerk_proxy': analysis_config.sharpness_weight, 'camera_motion': analysis_config.camera_motion_weight,
            'face_size': analysis_config.face_size_weight, 'percussive': analysis_config.percussive_weight,
            'depth_variance': analysis_config.depth_weight }
        norm_params = {
            'rms': analysis_config.norm_max_rms + 1e-6, 'kinetic': analysis_config.norm_max_kinetic + 1e-6,
            'jerk': analysis_config.norm_max_jerk + 1e-6, 'cam_motion': analysis_config.norm_max_cam_motion + 1e-6,
            'face_size': analysis_config.norm_max_face_size + 1e-6, 'percussive_ratio': 1.0 + 1e-6, # Assume percussive is already 0-1 ratio
            'depth_variance': analysis_config.norm_max_depth_variance + 1e-6 } # Using shared depth norm

        f = frame_features; pose_f = f.get('pose_features', {})
        norm_audio = np.clip(f.get('audio_energy', 0.0) / norm_params['rms'], 0.0, 1.0)
        norm_kinetic = np.clip(pose_f.get('kinetic_energy_proxy', 0.0) / norm_params['kinetic'], 0.0, 1.0)
        norm_jerk = np.clip(pose_f.get('movement_jerk_proxy', 0.0) / norm_params['jerk'], 0.0, 1.0)
        # Use 'camera_motion' feature directly (which is based on flow_velocity)
        norm_cam_motion = np.clip(f.get('camera_motion', 0.0) / norm_params['cam_motion'], 0.0, 1.0)
        norm_face_size = np.clip(pose_f.get('face_size_ratio', 0.0) / norm_params['face_size'], 0.0, 1.0)
        norm_percussive = np.clip(f.get('percussive_ratio', 0.0) / norm_params['percussive_ratio'], 0.0, 1.0)
        norm_depth_var = np.clip(f.get('depth_variance', 0.0) / norm_params['depth_variance'], 0.0, 1.0)

        contrib = {
            'audio_energy': norm_audio * weights['audio_energy'], 'kinetic_proxy': norm_kinetic * weights['kinetic_proxy'],
            'jerk_proxy': norm_jerk * weights['jerk_proxy'], 'camera_motion': norm_cam_motion * weights['camera_motion'],
            'face_size': norm_face_size * weights['face_size'], 'percussive': norm_percussive * weights['percussive'],
            'depth_variance': norm_depth_var * weights['depth_variance'] }

        # Ensure only features with weight > 0 contribute
        weighted_contrib = {k: v for k, v in contrib.items() if weights.get(k, 0) > 0}

        score = sum(weighted_contrib.values()); final_score = np.clip(score, 0.0, 1.0)
        dominant = self._determine_dominant_contributor(weighted_contrib)
        intensity = self._categorize_intensity(final_score, thresholds=analysis_config.intensity_thresholds)
        return final_score, dominant, intensity, weighted_contrib

    # Note: Pass AnalysisConfig
    def apply_beat_boost(self, frame_features_list: List[Dict], audio_data: Dict, video_fps: float, analysis_config: AnalysisConfig):
        num_frames = len(frame_features_list)
        if num_frames == 0 or not audio_data or video_fps <= 0: return

        beat_boost = analysis_config.beat_boost
        boost_radius_sec = analysis_config.beat_boost_radius_sec
        boost_radius_frames = max(0, int(boost_radius_sec * video_fps))
        beat_times = audio_data.get('beat_times', [])
        if not beat_times and beat_boost > 0:
            logger.warning("Beat boost enabled, but no beat times found in audio data.")
            return # Exit if no beats to boost

        boost_frame_indices = set()
        if beat_boost > 0:
            for t in beat_times:
                beat_frame_center = int(round(t * video_fps))
                for r in range(-boost_radius_frames, boost_radius_frames + 1):
                    idx = beat_frame_center + r
                    if 0 <= idx < num_frames: boost_frame_indices.add(idx)

        for i, features in enumerate(frame_features_list):
            is_beat = i in boost_frame_indices
            features['is_beat_frame'] = is_beat
            boost = beat_boost if is_beat else 0.0
            raw_score = features.get('raw_score', 0.0)
            features['boosted_score'] = min(raw_score + boost, 1.0)

    def get_feature_at_time(self, times_array, values_array, target_time):
        """Finds the feature value closest to the target time."""
        if times_array is None or values_array is None or len(times_array) == 0 or len(times_array) != len(values_array):
            return 0.0
        try:
            # Find the index of the time closest to the target time
            idx = np.searchsorted(times_array, target_time, side="left")

            # Handle edge cases: target time before first sample or after last sample
            if idx == 0:
                return values_array[0]
            if idx == len(times_array):
                return values_array[-1]

            # Choose the closer of the two neighbors
            left_time = times_array[idx - 1]
            right_time = times_array[idx]
            if abs(target_time - left_time) < abs(target_time - right_time):
                return values_array[idx - 1]
            else:
                return values_array[idx]
        except IndexError:
            logger.warning(f"Index error getting feature at time {target_time:.3f}. Using last value.")
            return values_array[-1] if len(values_array) > 0 else 0.0
        except Exception as e:
             logger.error(f"Error in get_feature_at_time: {e}", exc_info=True)
             return 0.0

    # ============================================================ #
    #                 MODIFIED analyzeVideo Method                 #
    # ============================================================ #
    def analyzeVideo(self, videoPath: str, analysis_config: AnalysisConfig, audio_data: Dict) -> Tuple[Optional[List[Dict]], Optional[List[ClipSegment]]]:
        """
        Analyzes video frames using VidGear for reading and OpenCV for properties.
        Extracts visual features (motion, pose, face, depth) and aligns with audio.
        """
        logger.info(f"--- Analyzing Video Features (VidGear Read): {os.path.basename(videoPath)} ---")
        start_time = time.time()
        self._ensure_midas_loaded(analysis_config) # Load MiDaS if needed by config

        TARGET_HEIGHT = analysis_config.resolution_height
        TARGET_WIDTH = analysis_config.resolution_width

        # --- Variables Initialization ---
        capture = None # OpenCV capture for properties
        stream = None  # VidGear stream for reading
        video_fps = 30.0 # Default FPS
        total_video_frames = 0
        all_frame_features = []
        pose_detector = None
        face_detector_util = None
        prev_gray_frame = None
        prev_flow_field = None
        pose_results_buffer = [None, None, None] # Buffer for jerk calculation
        frame_count = 0 # Manual frame counter for VidGear loop

        # === Step 1: Get Video Properties using OpenCV ===
        try:
            capture = cv2.VideoCapture(videoPath)
            if not capture.isOpened():
                logger.error(f"ERROR opening video with OpenCV (for properties): {videoPath}")
                return None, None

            _fps = capture.get(cv2.CAP_PROP_FPS)
            _frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if _fps > 0: video_fps = _fps
            else: logger.warning(f"Invalid FPS ({_fps}) from OpenCV for {videoPath}. Using default {video_fps} FPS.")

            if _frames > 0: total_video_frames = _frames
            else:
                logger.error(f"Invalid frame count ({_frames}) from OpenCV for {videoPath}. Cannot proceed.")
                capture.release()
                return None, None

            logger.info(f"Video Properties (OpenCV): FPS={video_fps:.2f}, Frames={total_video_frames}, Dur={total_video_frames/video_fps:.2f}s")

        except Exception as e:
             logger.error(f"Error reading video properties with OpenCV for {videoPath}: {e}", exc_info=True)
             if capture: capture.release()
             return None, None
        finally:
            # IMPORTANT: Release OpenCV capture immediately after getting properties
            if capture: capture.release()
            logger.debug("OpenCV capture released after property retrieval.")

        if video_fps <= 0: # Should not happen if frame count check passed, but safety first
             logger.error(f"Video FPS is invalid ({video_fps}). Aborting analysis.")
             return None, None

        frame_time_diff = 1.0 / video_fps # Time difference between frames

        # === Step 2: Setup MediaPipe, Audio Refs & VidGear Stream ===
        try:
            # Initialize MediaPipe FaceMesh (if needed by config, though it's cheap)
            logger.debug("Initializing MediaPipe FaceMesh...")
            face_detector_util = PegasusFaceUtils(
                static_mode=False, max_faces=1,
                min_detect_conf=analysis_config.min_face_confidence, min_track_conf=0.5)
            if face_detector_util.face_mesh is None:
                logger.warning("FaceMesh failed to initialize. Face features disabled.")
                # Optionally disable face_size_weight here?

            # Initialize MediaPipe Pose (only if heuristic weights > 0)
            if analysis_config.kinetic_weight > 0 or analysis_config.sharpness_weight > 0:
                logger.debug(f"Initializing MediaPipe Pose (Complexity: {analysis_config.model_complexity})...")
                pose_detector = mp_pose.Pose(
                    static_image_mode=False, model_complexity=analysis_config.model_complexity,
                    min_detection_confidence=analysis_config.min_pose_confidence, min_tracking_confidence=0.5)

            # --- Audio Data References ---
            if not audio_data: # Handle case where audio analysis might have failed
                logger.error("No master audio data provided to analyzeVideo. Aborting.")
                return None, None
            audio_raw_features = audio_data.get('raw_features', {})
            audio_rms_energy = np.asarray(audio_raw_features.get('rms_energy', []))
            audio_rms_times = np.asarray(audio_raw_features.get('rms_times', []))
            # Example: Add percussive if calculated and stored in raw_features
            audio_perc_ratio = np.asarray(audio_raw_features.get('percussive_ratio', []))
            audio_perc_times = np.asarray(audio_raw_features.get('perc_times', audio_rms_times)) # Use RMS times if specific perc times absent
            segment_boundaries = audio_data.get('segment_boundaries', [0, audio_data.get('duration', float('inf'))])


            # --- Initialize VidGear Stream ---
            logger.info(f"Initializing VidGear VideoGear for frame reading: {os.path.basename(videoPath)}")
            # Enable logging=True for VidGear debugging if needed
            stream = VideoGear(source=videoPath, logging=True).start()
            if stream is None or not hasattr(stream, 'read'): # Basic check
                 raise RuntimeError("Failed to initialize VidGear VideoGear stream.")

            # === Step 3: Feature Extraction Loop using VidGear ===
            try: from tqdm import tqdm as tqdm_analyzer
            except ImportError: tqdm_analyzer = lambda x, **kwargs: x; logger.info("tqdm not found, progress bar disabled.")

            logger.info("Processing frames & generating features (using VidGear)...")
            # Use total_video_frames obtained from OpenCV for tqdm progress bar
            with tqdm_analyzer(total=total_video_frames, desc=f"Analyzing {os.path.basename(videoPath)}", unit="frame", dynamic_ncols=True, leave=False) as pbar:
                while True: # Loop until VidGear stream ends
                    # === Read Frame with VidGear ===
                    frame = stream.read()

                    # === Check for End of Stream ===
                    if frame is None:
                        logger.info(f"VidGear stream ended for {videoPath} after processing {frame_count} frames.")
                        break

                    # === Safety Break: If frame count exceeds OpenCV's count ===
                    # This can sometimes happen with slightly corrupt files or specific encodings
                    if frame_count >= total_video_frames:
                        logger.warning(f"VidGear read frame {frame_count + 1}, exceeding OpenCV's count ({total_video_frames}). Stopping analysis.")
                        break

                    current_timestamp = frame_count * frame_time_diff

                    # --- Process Frame (Existing Logic - Compatible with VidGear frames) ---
                    image_resized = self.pegasusImageUtils.resizeTARGET(frame, TARGET_HEIGHT, TARGET_WIDTH)
                    if image_resized is None or image_resized.size == 0:
                        logger.warning(f"Frame {frame_count}: Resizing failed or resulted in empty image. Skipping.")
                        # Still increment frame count and update progress even if skipped
                        frame_count += 1
                        pbar.update(1)
                        continue

                    # Convert colors AFTER resizing for efficiency
                    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

                    # --- Feature Dictionary Initialization ---
                    current_features = {'frame_index': frame_count, 'timestamp': current_timestamp}
                    pose_features_dict = {} # Store pose/face related features here

                    # --- Physics Features ---
                    flow_velocity, current_flow_field = calculate_flow_velocity(prev_gray_frame, image_gray)
                    flow_acceleration = calculate_flow_acceleration(prev_flow_field, current_flow_field, frame_time_diff)
                    current_features['flow_velocity'] = flow_velocity
                    current_features['flow_acceleration'] = flow_acceleration
                    # Update previous state for next iteration's flow calculation
                    prev_gray_frame = image_gray.copy() # Crucial copy
                    prev_flow_field = current_flow_field.copy() if current_flow_field is not None else None

                    # Depth Calculation (MiDaS)
                    depth_variance = 0.0
                    if self.midas_model and self.midas_transform and self.midas_device:
                        try:
                            with torch.no_grad():
                                # Use original frame for MiDaS for better detail? Or resized? Using original here.
                                img_midas_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                input_batch = self.midas_transform(img_midas_rgb).to(self.midas_device)
                                prediction = self.midas_model(input_batch)
                                prediction = torch.nn.functional.interpolate(
                                    prediction.unsqueeze(1), size=img_midas_rgb.shape[:2],
                                    mode="bicubic", align_corners=False).squeeze()
                                depth_map = prediction.cpu().numpy()
                                depth_min = depth_map.min(); depth_max = depth_map.max()
                                if depth_max > depth_min:
                                    depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
                                    depth_variance = np.var(depth_norm)
                                else: depth_variance = 0.0
                        except Exception as midas_e:
                            logger.warning(f"MiDaS inference failed on frame {frame_count}: {midas_e}", exc_info=False)
                            depth_variance = 0.0 # Ensure it's zero on failure
                    current_features['depth_variance'] = depth_variance # Store raw variance

                    # Histogram Entropy
                    current_features['histogram_entropy'] = calculate_histogram_entropy(image_resized)

                    # --- Heuristic & Simple Features ---
                    # Face Features
                    if face_detector_util and face_detector_util.face_mesh:
                        is_mouth_open, face_size_ratio, face_center_x = face_detector_util.get_face_features(
                           image_resized, analysis_config.mouth_open_threshold)
                        pose_features_dict['is_mouth_open'] = is_mouth_open
                        pose_features_dict['face_size_ratio'] = face_size_ratio
                        pose_features_dict['face_center_x'] = face_center_x
                    else: # Default values if no face detector
                        pose_features_dict['is_mouth_open'] = False
                        pose_features_dict['face_size_ratio'] = 0.0
                        pose_features_dict['face_center_x'] = 0.5

                    # Pose Features (Kinetic, Jerk)
                    current_pose_results = None
                    if pose_detector:
                        try:
                            image_rgb.flags.writeable = False # Optimization for MediaPipe
                            current_pose_results = pose_detector.process(image_rgb)
                            image_rgb.flags.writeable = True
                        except Exception as pose_err:
                            logger.warning(f"Pose detection failed on frame {frame_count}: {pose_err}", exc_info=False)

                    # Update buffer and calculate kinetic/jerk based on buffer history
                    pose_results_buffer.pop(0); pose_results_buffer.append(current_pose_results)
                    kinetic = 0.0; jerk = 0.0
                    lm_t1 = pose_results_buffer[1].pose_landmarks if pose_results_buffer[1] else None
                    lm_t = pose_results_buffer[2].pose_landmarks if pose_results_buffer[2] else None
                    if lm_t1 and lm_t:
                        kinetic = calculate_kinetic_energy_proxy(lm_t1, lm_t, frame_time_diff)
                        lm_t2 = pose_results_buffer[0].pose_landmarks if pose_results_buffer[0] else None
                        if lm_t2: # Need 3 frames for jerk
                             jerk = calculate_movement_jerk_proxy(lm_t2, lm_t1, lm_t, frame_time_diff)
                    pose_features_dict['kinetic_energy_proxy'] = kinetic
                    pose_features_dict['movement_jerk_proxy'] = jerk
                    current_features['pose_features'] = pose_features_dict # Add sub-dict

                    # --- Align with Audio Features ---
                    # Use timestamp slightly ahead (middle of frame duration) for better sync?
                    mid_frame_time = current_timestamp + frame_time_diff / 2.0
                    current_features['audio_energy'] = self.get_feature_at_time(audio_rms_times, audio_rms_energy, mid_frame_time)
                    current_features['percussive_ratio'] = self.get_feature_at_time(audio_perc_times, audio_perc_ratio, mid_frame_time)
                    # Use flow_velocity as camera motion proxy for heuristic score
                    current_features['camera_motion'] = flow_velocity

                    # Determine Musical Section Index
                    section_idx = -1
                    for i in range(len(segment_boundaries) - 1):
                        if segment_boundaries[i] <= mid_frame_time < segment_boundaries[i+1]:
                            section_idx = i; break
                    current_features['musical_section_index'] = section_idx

                    # --- Calculate Heuristic Score (if applicable) ---
                    # This calculates the raw score before beat boost
                    heuristic_score, dominant, intensity, _ = self.calculate_heuristic_score(current_features, analysis_config)
                    current_features['raw_score'] = heuristic_score
                    current_features['dominant_contributor'] = dominant
                    current_features['intensity_category'] = intensity
                    # Initialize beat-related fields (boost applied later)
                    current_features['boosted_score'] = heuristic_score
                    current_features['is_beat_frame'] = False

                    # --- Store Features for this Frame ---
                    all_frame_features.append(current_features)

                    # === Increment frame counter and update progress bar ===
                    frame_count += 1
                    pbar.update(1)
                    # === End of Frame Processing ===

        except Exception as loop_err:
             logger.error(f"Error during VidGear video frame processing loop: {loop_err}", exc_info=True)
             # Fall through to cleanup

        finally:
            # === Step 4: Cleanup ---
            # Stop VidGear stream first
            if stream:
                try:
                    stream.stop()
                    logger.debug("VidGear stream stopped.")
                except Exception as vg_stop_err:
                     logger.error(f"Error stopping VidGear stream: {vg_stop_err}")

            # Release MediaPipe resources
            if face_detector_util: face_detector_util.close()
            if pose_detector: pose_detector.close()
            logger.debug(f"Finished feature extraction ({len(all_frame_features)} frames processed). Releasing MediaPipe resources.")

        # === Step 5: Post-processing & Clip Identification ===
        if not all_frame_features:
            logger.error(f"No features extracted for {videoPath}. Analysis failed.")
            return None, None

        # Check if the number of processed frames is significantly different from OpenCV's count
        if abs(frame_count - total_video_frames) > max(5, total_video_frames * 0.05): # Allow small deviation
             logger.warning(f"Processed {frame_count} frames, but OpenCV reported {total_video_frames}. "
                            f"There might be a discrepancy in the video file or reading process.")
        else:
             logger.info(f"Successfully processed {frame_count} frames (matches OpenCV count: {total_video_frames}).")

        # Apply beat boost to the 'raw_score' to get 'boosted_score'
        logger.debug("Applying beat boost (for heuristic score)...")
        # Use the reliable video_fps obtained from OpenCV
        self.apply_beat_boost(all_frame_features, audio_data, video_fps, analysis_config)

        # Identify Potential Clips based on configuration
        potential_clips: List[ClipSegment] = []
        # Use the reliable video_fps obtained from OpenCV for segment identification and ClipSegment creation
        if analysis_config.use_heuristic_segment_id:
             logger.debug("Identifying potential segments using heuristic score...")
             segment_identifier = DynamicSegmentIdentifier(analysis_config, video_fps)
             potential_segment_indices = segment_identifier.find_potential_segments(all_frame_features)
             logger.info(f"Creating {len(potential_segment_indices)} ClipSegment objects from identified runs...")
             for seg_indices in potential_segment_indices:
                 start_f = seg_indices['start_frame']; end_f = seg_indices['end_frame']
                 # Add boundary checks using the length of *actual* features extracted
                 if start_f < end_f and 0 <= start_f < len(all_frame_features) and end_f <= len(all_frame_features):
                    try:
                        clip_segment = ClipSegment(videoPath, start_f, end_f, video_fps, all_frame_features, analysis_config)
                        # Filter by duration AFTER creation
                        if clip_segment.duration >= analysis_config.min_potential_clip_duration_sec:
                            potential_clips.append(clip_segment)
                        else:
                            logger.debug(f"Skipping potential heuristic segment {start_f}-{end_f}: duration {clip_segment.duration:.2f}s < min {analysis_config.min_potential_clip_duration_sec:.2f}s")
                    except Exception as clip_err:
                        logger.warning(f"Failed to create ClipSegment for heuristic segment {start_f}-{end_f}: {clip_err}", exc_info=False)
                 else:
                    logger.warning(f"Invalid segment indices {start_f}-{end_f} from DynamicSegmentIdentifier (Total features: {len(all_frame_features)}). Skipping.")
        else:
            # Fallback: Create potential segments using fixed/overlapping chunks
            logger.info("Creating potential segments using fixed/overlapping chunks...")
            min_clip_frames = int(analysis_config.min_potential_clip_duration_sec * video_fps)
            max_clip_frames = int(analysis_config.max_sequence_clip_duration * video_fps) # Use max sequence duration as upper bound
            step_frames = max(1, int(1.0 * video_fps)) # Ensure step is at least 1 frame
            actual_total_frames = len(all_frame_features) # Use the number of features actually extracted

            if actual_total_frames < min_clip_frames:
                 logger.warning(f"Video is too short ({actual_total_frames} frames < {min_clip_frames} min frames) for chunking. No clips generated.")
            else:
                for start_f in range(0, actual_total_frames - min_clip_frames + 1, step_frames):
                     # Calculate end frame based on max duration, but cap at actual total frames
                     end_f = min(start_f + max_clip_frames, actual_total_frames)
                     actual_len_frames = end_f - start_f
                     # Ensure the chunk meets the *minimum* length requirement
                     if actual_len_frames >= min_clip_frames:
                         try:
                              clip_segment = ClipSegment(videoPath, start_f, end_f, video_fps, all_frame_features, analysis_config)
                              potential_clips.append(clip_segment)
                         except Exception as clip_err:
                             logger.warning(f"Failed to create ClipSegment chunk for {start_f}-{end_f}: {clip_err}", exc_info=False)

        end_time = time.time()
        logger.info(f"--- Analysis & Clip Creation complete for {os.path.basename(videoPath)} ({end_time - start_time:.2f}s) ---")
        logger.info(f"Created {len(potential_clips)} potential clips.")

        # Pass back the list of features and the identified clips
        return all_frame_features, potential_clips

    # Note: Pass AnalysisConfig (used by ClipSegment.to_dict if implemented)
    def saveAnalysisData(self, video_path: str, frame_features: List[Dict], potential_clips: List[ClipSegment], output_dir: str, analysis_config: AnalysisConfig):
        """Saves detailed frame features and potential clips (ClipSegment data)."""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_subdir = os.path.join(output_dir, "analysis_data")
        features_path = os.path.join(out_subdir, f"{base_name}_frame_features.json")
        clips_path = os.path.join(out_subdir, f"{base_name}_potential_clips.json")
        logger.info(f"Saving analysis data for {base_name}...")

        # --- Helper to sanitize values for JSON ---
        def sanitize_for_json(value):
            if isinstance(value, (np.number, np.bool_)):
                # FIX: Check for numpy scalars like float32, int64, etc.
                return value.item() # Convert numpy types to python types
            elif isinstance(value, np.ndarray):
                return value.tolist() # Convert arrays to lists
            elif isinstance(value, set):
                return list(value) # Convert sets to lists
            elif isinstance(value, dict):
                # Recursively sanitize dictionary values
                return {k: sanitize_for_json(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                # Recursively sanitize list/tuple elements
                return [sanitize_for_json(item) for item in value]
            elif isinstance(value, (int, float, str, bool)) or value is None:
                return value # Standard JSON types are fine
            else:
                # Attempt to convert unknown types, log if fails
                try:
                    json.dumps(value) # Test if serializable
                    return value
                except TypeError:
                    logger.debug(f"Omitting non-serializable type {type(value)} during save.")
                    return None # Or return str(value) ?

        try:
            os.makedirs(out_subdir, exist_ok=True)

            # Save frame features (handle numpy types using helper)
            logger.debug(f"Saving {len(frame_features)} frame features to {features_path}")
            with open(features_path, 'w') as f:
                # Use sanitize_for_json on the entire list
                serializable_features = sanitize_for_json(frame_features)
                json.dump(serializable_features, f, indent=1) # Use less verbose indent

            # Save potential clips (handle numpy types using helper)
            logger.debug(f"Saving {len(potential_clips)} potential clips to {clips_path}")
            clips_data = []
            for clip in potential_clips:
                # Create a dictionary from the ClipSegment object
                clip_dict = {
                    'source_video_path': os.path.basename(clip.source_video_path),
                    'start_frame': clip.start_frame, 'end_frame': clip.end_frame,
                    'start_time': clip.start_time, 'end_time': clip.end_time,
                    'fps': clip.fps, 'duration': clip.duration,
                    'avg_raw_score': clip.avg_raw_score, 'avg_boosted_score': clip.avg_boosted_score,
                    'peak_boosted_score': clip.peak_boosted_score,
                    'avg_motion_heuristic': clip.avg_motion_heuristic, 'avg_jerk_heuristic': clip.avg_jerk_heuristic,
                    'avg_camera_motion': clip.avg_camera_motion,
                    'face_presence_ratio': clip.face_presence_ratio, 'avg_face_size': clip.avg_face_size,
                    'intensity_category': clip.intensity_category, 'dominant_contributor': clip.dominant_contributor,
                    'contains_beat': clip.contains_beat,
                    'musical_section_indices': list(clip.musical_section_indices), # Already converted set
                    'v_k': clip.v_k, 'a_j': clip.a_j, 'd_r': clip.d_r, 'phi': clip.phi,
                    'mood_vector': clip.mood_vector # Already list
                }
                # Sanitize the entire dictionary
                sanitized_clip_dict = sanitize_for_json(clip_dict)
                # Filter out any keys that became None during sanitization if desired
                # sanitized_clip_dict = {k: v for k, v in sanitized_clip_dict.items() if v is not None}
                clips_data.append(sanitized_clip_dict)

            with open(clips_path, 'w') as f:
                json.dump(clips_data, f, indent=2)

            logger.info(f"Saved analysis data to {os.path.basename(features_path)} and {os.path.basename(clips_path)}")
        except Exception as e:
            logger.error(f"ERROR saving analysis data for {base_name}: {e}", exc_info=True)
            # Don't raise here, just log the error in the worker

# ========================================================================
#          SEQUENCE BUILDER - GREEDY HEURISTIC (Uses AnalysisConfig)
# ========================================================================
class SequenceBuilderGreedy:
    def __init__(self, all_potential_clips: List[ClipSegment], audio_data: Dict, analysis_config: AnalysisConfig):
        self.all_clips = all_potential_clips
        self.audio_data = audio_data
        self.analysis_config = analysis_config # Store dataclass
        segment_boundaries_raw = audio_data.get('segment_boundaries', [])
        if not segment_boundaries_raw or len(segment_boundaries_raw) < 2:
             # Use audio duration as the boundary if segments invalid/missing
             aud_dur = audio_data.get('duration', 0)
             self.segment_boundaries = [0, aud_dur] if aud_dur > 0 else [0]
             logger.warning(f"Using default [0, {aud_dur:.2f}s] for audio segment boundaries.")
        else:
             self.segment_boundaries = segment_boundaries_raw

        self.beat_times = audio_data.get('beat_times', [])
        self.target_duration = audio_data.get('duration', 0)
        self.final_sequence: List[ClipSegment] = []
        self.current_time = 0.0
        self.last_clip_used: Optional[ClipSegment] = None # Type hint added

        # Get parameters from config dataclass
        self.pacing_variation = analysis_config.pacing_variation_factor
        self.variety_penalty_source = analysis_config.variety_penalty_source
        self.variety_penalty_shot = analysis_config.variety_penalty_shot
        self.variety_penalty_intensity = analysis_config.variety_penalty_intensity
        self.beat_sync_bonus = analysis_config.beat_sync_bonus
        self.section_match_bonus = analysis_config.section_match_bonus
        self.min_clip_duration = analysis_config.min_sequence_clip_duration
        self.max_clip_duration = analysis_config.max_sequence_clip_duration
        self.candidate_pool_size = analysis_config.candidate_pool_size

    def build_sequence(self) -> List[ClipSegment]:
        logger.info("--- Composing Sequence (Greedy Heuristic Mode) ---")
        if not self.all_clips:
            logger.warning("No potential clips found for Greedy Heuristic mode.")
            return []
        if self.target_duration <= 0:
             logger.warning("Target duration is zero or negative. Cannot build sequence.")
             return []
        if len(self.segment_boundaries) < 2:
             logger.warning("Insufficient audio segment boundaries. Cannot build sequence by section.")
             return []


        # Assign clips to audio sections (can belong to multiple if overlaps?)
        # Using midpoint assignment for simplicity here
        clips_by_section = defaultdict(list)
        for clip in self.all_clips:
            clip_mid_time = clip.start_time + clip.duration / 2.0
            for i in range(len(self.segment_boundaries) - 1):
                 section_start = self.segment_boundaries[i]
                 section_end = self.segment_boundaries[i+1]
                 if section_start <= clip_mid_time < section_end:
                     clips_by_section[i].append(clip)
                     break # Assign to first matching section

        available_clips = self.all_clips.copy()
        num_sections = len(self.segment_boundaries) - 1

        # --- Iterate through audio sections ---
        for section_idx in range(num_sections):
            section_start = self.segment_boundaries[section_idx]
            section_end = self.segment_boundaries[section_idx+1]
            section_duration = section_end - section_start
            if section_duration <= 0: continue

            # Determine how much time needs to be filled in this section based on current time
            target_section_fill = max(0, section_end - self.current_time)
            if target_section_fill < self.min_clip_duration * 0.5: # Skip if very little time left to fill
                 logger.debug(f"Skipping section {section_idx}: only {target_section_fill:.2f}s remaining to fill.")
                 continue

            section_context = self._get_section_context(section_idx, section_start, section_end)
            logger.info(f"Serving Course {section_idx} ({section_start:.2f}s - {section_end:.2f}s, TargetFill: {target_section_fill:.2f}s)")
            logger.info(f"  Theme: Energy={section_context['avg_energy']:.3f}, TargetClipDur={section_context['target_clip_dur']:.2f}s")

            section_filled_time = 0.0
            # Consider clips primarily from this section, but allow others if needed
            section_primary_candidates = clips_by_section.get(section_idx, [])
            # Combine primary candidates with all available clips, removing duplicates
            candidate_pool = list(dict.fromkeys(section_primary_candidates + available_clips))

            # --- Fill the current section ---
            # Stop if we exceed the section end time slightly, or run out of candidates
            while self.current_time < section_end and candidate_pool:
                best_candidate = None; max_selection_score = -float('inf')

                # Filter pool to only available clips & sort by boosted score (desc)
                eligible_candidates = [c for c in candidate_pool if c in available_clips]
                potential_candidates_sorted = sorted(eligible_candidates, key=lambda c: c.avg_boosted_score, reverse=True)

                num_considered = 0
                # Consider top N candidates based on score
                for candidate in potential_candidates_sorted[:self.candidate_pool_size]:
                    selection_score, chosen_duration = self._calculate_selection_score(candidate, section_context)
                    num_considered += 1
                    # Store temporary duration on the candidate object
                    candidate.temp_chosen_duration = chosen_duration
                    if selection_score > max_selection_score:
                        max_selection_score = selection_score
                        best_candidate = candidate
                        # No need to store duration here, it's on best_candidate.temp_chosen_duration

                if best_candidate:
                    # Use the duration calculated during scoring
                    final_chosen_duration = best_candidate.temp_chosen_duration
                    # Ensure duration respects clip limits and remaining section time
                    final_chosen_duration = np.clip(final_chosen_duration, self.min_clip_duration, self.max_clip_duration)
                    final_chosen_duration = min(final_chosen_duration, best_candidate.duration) # Can't exceed original clip length
                    final_chosen_duration = min(final_chosen_duration, max(0.01, section_end - self.current_time)) # Don't overshoot section end much
                    final_chosen_duration = max(0.01, final_chosen_duration) # Ensure positive duration

                    # Set sequence timing and subclip info
                    best_candidate.sequence_start_time = self.current_time
                    best_candidate.sequence_end_time = self.current_time + final_chosen_duration
                    best_candidate.chosen_duration = final_chosen_duration
                    best_candidate.subclip_start_time_in_source = best_candidate.start_time # Always take from start for now
                    best_candidate.subclip_end_time_in_source = best_candidate.start_time + final_chosen_duration
                    best_candidate.chosen_effect = EffectParams(type="cut") # Greedy only uses cuts for now

                    self.final_sequence.append(best_candidate)
                    logger.info(f"  + Added Clip (Greedy): {os.path.basename(best_candidate.source_video_path)} "
                                f"Src:({best_candidate.subclip_start_time_in_source:.2f}-{best_candidate.subclip_end_time_in_source:.2f}) "
                                f"Seq:({best_candidate.sequence_start_time:.2f}-{best_candidate.sequence_end_time:.2f}) "
                                f"Score: {max_selection_score:.3f}, Dur: {best_candidate.chosen_duration:.2f}s")

                    # Update state
                    self.current_time += best_candidate.chosen_duration
                    section_filled_time += best_candidate.chosen_duration
                    self.last_clip_used = best_candidate
                    if best_candidate in available_clips: available_clips.remove(best_candidate)
                    if best_candidate in candidate_pool: candidate_pool.remove(best_candidate) # Remove from section pool too
                else:
                    logger.info(f"  - No suitable clip found in this iteration for section {section_idx}. Moving to next section.")
                    break # Exit inner loop if no candidate found

            logger.info(f" Filled {section_filled_time:.2f}s for section {section_idx}. Current time: {self.current_time:.2f}s")
            # Ensure timeline progresses at least to section end if we finished early/skipped
            self.current_time = max(self.current_time, section_end)

        final_duration = self.final_sequence[-1].sequence_end_time if self.final_sequence else 0.0
        logger.info("--- Sequence Composition Complete (Greedy Heuristic) ---")
        logger.info(f"Total Duration: {final_duration:.2f}s (Target: {self.target_duration:.2f}s)")
        logger.info(f"Number of Clips: {len(self.final_sequence)}")
        return self.final_sequence

    def _get_section_context(self, section_idx, start_time, end_time):
        """Calculates context for the current audio section."""
        context = {}
        # Use config dataclass for norms
        norm_max_rms = self.analysis_config.norm_max_rms + 1e-6
        context['avg_energy'] = self._get_avg_audio_feature_in_range('rms_energy', start_time, end_time)
        norm_energy = np.clip(context['avg_energy'] / norm_max_rms, 0.0, 1.0)

        # Determine target clip duration based on energy (inverse relationship)
        context['target_clip_dur'] = self.min_clip_duration + (1.0 - norm_energy) * (self.max_clip_duration - self.min_clip_duration)
        context['target_clip_dur'] = np.clip(context['target_clip_dur'], self.min_clip_duration, self.max_clip_duration)

        context['start'] = start_time
        context['end'] = end_time
        return context

    def _calculate_selection_score(self, candidate_clip: ClipSegment, section_context: Dict) -> Tuple[float, float]:
        """Calculates the selection score for a candidate clip in the current context."""
        # Start with the clip's inherent quality score
        score = candidate_clip.avg_boosted_score

        # --- Apply Penalties for Lack of Variety ---
        if self.last_clip_used:
            if candidate_clip.source_video_path == self.last_clip_used.source_video_path:
                score -= self.variety_penalty_source
            shot_type = candidate_clip.get_shot_type()
            last_shot_type = self.last_clip_used.get_shot_type()
            if shot_type == last_shot_type and shot_type != 'wide/no_face': # Don't penalize repeating wide shots?
                score -= self.variety_penalty_shot
            if candidate_clip.intensity_category == self.last_clip_used.intensity_category:
                score -= self.variety_penalty_intensity

        # --- Apply Bonuses for Good Fits ---
        # Calculate potential duration based on section context
        # Blend target duration with clip's natural duration
        target_dur = section_context['target_clip_dur']
        effective_duration = (target_dur * 0.7) + (candidate_clip.duration * 0.3)
        effective_duration = np.clip(effective_duration, self.min_clip_duration, self.max_clip_duration)
        effective_duration = min(effective_duration, candidate_clip.duration) # Cannot exceed original length
        effective_duration = max(0.01, effective_duration)

        # Bonus for cutting near a beat
        potential_cut_time = self.current_time + effective_duration
        if self._is_near_beat(potential_cut_time, tolerance=self.analysis_config.beat_boost_radius_sec):
            score += self.beat_sync_bonus

        # Bonus for matching section energy/mood (simple proxy)
        norm_max_rms = self.analysis_config.norm_max_rms + 1e-6
        norm_max_kinetic = self.analysis_config.norm_max_kinetic + 1e-6
        norm_section_energy = np.clip(section_context['avg_energy'] / norm_max_rms, 0.0, 1.0)
        norm_clip_motion = np.clip(candidate_clip.avg_motion_heuristic / norm_max_kinetic, 0.0, 1.0)
        # Bonus if both section and clip are high energy/motion
        if norm_section_energy > 0.6 and norm_clip_motion > 0.6:
             score += self.section_match_bonus * 0.5
        # Bonus if section is low energy and clip is a close-up (calm focus)
        if norm_section_energy < 0.3 and candidate_clip.get_shot_type() == 'close_up':
             score += self.section_match_bonus

        return score, effective_duration # Return score and the calculated duration


    def _get_avg_audio_feature_in_range(self, feature_name, start_time, end_time):
        """Gets the average value of a raw audio feature within a time range."""
        # Use raw features stored during audio analysis
        audio_raw = self.audio_data.get('raw_features', {})
        # Construct key names used in audio analysis raw_features dict
        times_key = f'{feature_name}_times' if feature_name != 'rms_energy' else 'rms_times'
        values_key = feature_name

        times = np.asarray(audio_raw.get(times_key, []))
        values = np.asarray(audio_raw.get(values_key, []))

        if len(times) == 0 or len(values) == 0 or len(times) != len(values):
            logger.debug(f"Feature '{feature_name}' or its times missing/mismatched in raw audio data.")
            return 0.0

        # Find indices corresponding to the time range
        start_idx = np.searchsorted(times, start_time, side='left')
        end_idx = np.searchsorted(times, end_time, side='right')

        # If no samples fall within the range, find the closest sample
        if start_idx >= end_idx:
             mid_time = (start_time + end_time) / 2.0
             closest_idx = np.argmin(np.abs(times - mid_time))
             # Ensure index is valid
             safe_idx = min(max(0, closest_idx), len(values) - 1)
             logger.debug(f"No full samples for '{feature_name}' in {start_time:.2f}-{end_time:.2f}s. Using closest value at index {safe_idx}.")
             return values[safe_idx]

        # Calculate mean of values within the range
        section_values = values[start_idx:end_idx]
        return np.mean(section_values) if len(section_values) > 0 else 0.0

    def _is_near_beat(self, time_sec, tolerance=0.1):
        """Checks if a given time is close to a beat time."""
        if not self.beat_times: return False
        # Use numpy broadcasting for efficient calculation
        beat_times_arr = np.asarray(self.beat_times)
        if len(beat_times_arr) == 0: return False
        min_diff = np.min(np.abs(beat_times_arr - time_sec))
        return min_diff <= tolerance

# print("DEBUG: Defined SequenceBuilderGreedy") # REMOVED Debug Print

# ========================================================================
#          SEQUENCE BUILDER - PHYSICS PARETO MC (Uses AnalysisConfig)
# ========================================================================
class SequenceBuilderPhysicsMC:
    # Note: Pass AnalysisConfig
    def __init__(self, all_potential_clips: List[ClipSegment], audio_data: Dict, analysis_config: AnalysisConfig):
        self.all_clips = all_potential_clips
        self.audio_data = audio_data
        self.analysis_config = analysis_config # Store dataclass
        self.target_duration = audio_data.get('duration', 0)
        self.beat_times = audio_data.get('beat_times', [])
        self.audio_segments = audio_data.get('segment_features', []) # Pre-analyzed segments

        # Get parameters from config dataclass
        self.mc_iterations = analysis_config.mc_iterations
        self.min_clip_duration = analysis_config.min_sequence_clip_duration
        self.max_clip_duration = analysis_config.max_sequence_clip_duration
        self.w_r = analysis_config.objective_weight_rhythm
        self.w_m = analysis_config.objective_weight_mood
        self.w_c = analysis_config.objective_weight_continuity
        self.w_v = analysis_config.objective_weight_variety
        self.w_e = analysis_config.objective_weight_efficiency
        self.tempo = audio_data.get('tempo', 120.0)
        self.beat_period = 60.0 / self.tempo if self.tempo > 0 else 0.5

        # Load effects from config (assuming RenderConfig structure available or passed in)
        # This relies on RenderConfig being generated based on UI sliders before this class is called.
        # A cleaner way might be to pass RenderConfig directly, but this works if sequence builder
        # is only instantiated after render config is gathered.
        temp_render_config = RenderConfig() # Create default to get structure
        # Try to populate from analysis config sliders if needed (less ideal)
        # For now, assuming RenderConfig will be created properly based on UI sliders elsewhere
        # and we just need the basic types. We'll use defaults/placeholders if needed.
        # TODO: Pass RenderConfig directly or ensure AnalysisConfig holds effect params.
        self.effects: Dict[str, EffectParams] = {
             # Default effects, actual params (tau, psi, epsilon) should come from RenderConfig/UI
            "cut": EffectParams(type="cut"),
            "fade": EffectParams(type="fade", tau=0.2, psi=0.1, epsilon=0.2),
            "zoom": EffectParams(type="zoom", tau=0.5, psi=0.3, epsilon=0.4),
            "pan": EffectParams(type="pan", tau=0.5, psi=0.1, epsilon=0.3),
        }
        # --- Attempt to override defaults with values potentially set in AnalysisConfig ---
        # This part is a workaround - Ideally, RenderConfig is passed or effects are part of AnalysisConfig
        if hasattr(analysis_config, 'effect_settings'): # Check if AnalysisConfig was extended (unlikely)
            self.effects = analysis_config.effect_settings
        # logger.debug(f"Physics MC Initialized Effects: {self.effects}") # Log loaded effects


    def get_audio_segment_at(self, time):
        """Finds the pre-analyzed audio segment containing the given time."""
        if not self.audio_segments: return None
        for seg in self.audio_segments:
            # Use inclusive start, exclusive end
            if seg['start'] <= time < seg['end']:
                return seg
        # Handle time exactly at or past the end of the last segment
        last_seg = self.audio_segments[-1]
        if time >= last_seg['start']:
             return last_seg
        logger.warning(f"Time {time:.2f}s is before the start of the first audio segment ({self.audio_segments[0]['start']:.2f}s).")
        return None # Or return first segment?


    def build_sequence(self) -> List[ClipSegment]:
        logger.info(f"--- Composing Sequence (Physics Pareto MC Mode - {self.mc_iterations} iterations) ---")
        if not self.all_clips:
             logger.warning("No potential clips available for Physics MC mode.")
             return []
        if not self.audio_segments:
             logger.warning("No audio segment features available for Physics MC mode.")
             return []
        if self.target_duration <= 0:
             logger.warning("Target duration is zero or negative. Cannot build sequence.")
             return []

        pareto_front: List[Tuple[List[Tuple[ClipSegment, float, EffectParams]], List[float]]] = []
        # Stores tuples: (sequence_info, objective_scores)
        # sequence_info is List[Tuple[ClipSegment, chosen_duration, EffectParams]]

        # --- Run Monte Carlo Simulations ---
        try: # Use tqdm for MC iterations if available
            from tqdm import trange as trange_mc
        except ImportError: trange_mc = range; logger.info("tqdm not found, MC progress bar disabled.")

        logger.info(f"Running {self.mc_iterations} Monte Carlo simulations...")
        with trange_mc(self.mc_iterations, desc="MC Simulations", leave=False) as mc_iter_range:
            for i in mc_iter_range:
                try:
                    # Generate one stochastic sequence
                    sim_seq_info = self._run_stochastic_build()
                    if sim_seq_info:
                        # Evaluate its objectives
                        scores = self._evaluate_pareto(sim_seq_info)
                        # Update the Pareto front
                        self._update_pareto_front(pareto_front, (sim_seq_info, scores))
                except Exception as mc_err:
                     logger.error(f"Error during MC iteration {i}: {mc_err}", exc_info=False) # Log error but continue

        if not pareto_front:
             logger.error("Monte Carlo simulation yielded no valid sequences on the Pareto front.")
             tk_write("Physics MC Error: No valid sequences found after simulations. Try adjusting parameters or increasing iterations.", parent=self, level="error")
             return []

        logger.info(f"Found {len(pareto_front)} non-dominated sequences on the Pareto front.")

        # --- Select Best Sequence from Pareto Front ---
        if len(pareto_front) == 1:
            best_solution = pareto_front[0]
            logger.info("Only one non-dominated solution found.")
        else:
            # Selection Strategy: Maximize sum of normalized scores? Or prioritize one?
            # Example: Maximize Mood Coherence (index 1) + Continuity (index 2)
            def selection_metric(item):
                scores = item[1] # [neg_r, M, C, V, neg_ec]
                # Normalize scores (assuming max is approx 1, min is approx -1 or 0)
                # This is a heuristic normalization, might need adjustment
                norm_r = scores[0] / (self.w_r + 1e-6) # neg_r -> closer to 0 is better
                norm_m = scores[1] / (self.w_m + 1e-6)
                norm_c = scores[2] / (self.w_c + 1e-6)
                norm_v = scores[3] / (self.w_v + 1e-6)
                norm_ec = scores[4] / (self.w_e + 1e-6) # neg_ec -> closer to 0 is better
                # Example metric: Prioritize Mood and Continuity, penalize bad rhythm/efficiency
                return norm_m + norm_c + norm_r + norm_ec + norm_v * 0.5 # Weight variety less?

            best_solution = max(pareto_front, key=selection_metric)
            logger.info("Selected sequence from Pareto front based on custom metric (prioritizing Mood & Continuity).")

        logger.info(f"Chosen sequence objectives (R*, M, C, V, EC*): {['{:.3f}'.format(s) for s in best_solution[1]]}")
        final_sequence_info = best_solution[0] # List[Tuple[ClipSegment, duration, EffectParams]]

        # --- Convert chosen sequence info to final ClipSegment list with timing ---
        final_sequence_segments: List[ClipSegment] = []
        current_t = 0.0
        for i, (clip, duration, effect) in enumerate(final_sequence_info):
            # Make sure we have a valid ClipSegment object
            if not isinstance(clip, ClipSegment):
                 logger.error(f"Invalid item in final sequence info at index {i}: expected ClipSegment, got {type(clip)}")
                 continue

            clip.sequence_start_time = current_t
            clip.sequence_end_time = current_t + duration
            clip.chosen_duration = duration
            clip.chosen_effect = effect # Store chosen EffectParams object
            # Ensure subclip times are set correctly for rendering (take from start)
            clip.subclip_start_time_in_source = clip.start_time
            clip.subclip_end_time_in_source = clip.start_time + duration
             # Safety check: subclip end time cannot exceed original clip end time
            clip.subclip_end_time_in_source = min(clip.subclip_end_time_in_source, clip.end_time)
            final_sequence_segments.append(clip)
            current_t += duration

        logger.info("--- Sequence Composition Complete (Physics Pareto MC) ---")
        logger.info(f"Final Duration: {current_t:.2f}s (Target: {self.target_duration:.2f}s)")
        logger.info(f"Number of Clips: {len(final_sequence_segments)}")
        return final_sequence_segments

    def _run_stochastic_build(self):
        """Generates a single sequence using probabilistic selection."""
        sequence_info: List[Tuple[ClipSegment, float, EffectParams]] = []
        current_time = 0.0
        # Use indices to avoid modifying the original list directly during iteration
        available_clip_indices = list(range(len(self.all_clips)))
        random.shuffle(available_clip_indices)
        last_clip_segment: Optional[ClipSegment] = None

        while current_time < self.target_duration and available_clip_indices:
            audio_seg = self.get_audio_segment_at(current_time)
            if not audio_seg:
                # logger.debug(f"Stopping build at {current_time:.2f}s: No audio segment found.")
                break

            candidates_info = [] # Store tuples: (clip_object, original_index, probability)
            total_prob = 0

            # Consider available clips for the current audio segment
            for list_idx in available_clip_indices:
                clip = self.all_clips[list_idx]
                # Basic filtering: ensure clip meets minimum duration
                if clip.duration < self.min_clip_duration: continue

                # Calculate fit probability using AnalysisConfig
                prob = clip.clip_audio_fit(audio_seg, self.analysis_config)

                # Apply repetition penalty directly to probability
                if last_clip_segment and clip.source_video_path == last_clip_segment.source_video_path:
                     prob *= (1.0 - self.analysis_config.variety_repetition_penalty) # Use variety penalty from config

                # Only consider candidates with non-negligible probability
                if prob > 1e-5:
                    candidates_info.append((clip, list_idx, prob))
                    total_prob += prob

            if not candidates_info:
                # logger.debug(f"Stopping build at {current_time:.2f}s: No suitable candidates found.")
                break # No valid candidates found for this step

            # --- Select Clip Probabilistically ---
            if total_prob > 1e-9: # Use a small threshold to avoid division by zero
                probabilities = [p / total_prob for _, _, p in candidates_info]
                chosen_candidate_idx = random.choices(range(len(candidates_info)), weights=probabilities, k=1)[0]
            else:
                # If all probabilities are effectively zero, choose randomly among candidates
                 logger.debug(f"All candidate probabilities near zero at {current_time:.2f}s. Choosing randomly.")
                 chosen_candidate_idx = random.randrange(len(candidates_info))


            chosen_clip, original_list_idx, _ = candidates_info[chosen_candidate_idx]

            # --- Determine Clip Duration ---
            remaining_time = self.target_duration - current_time
            # Start with the clip's full duration, capped by max allowed and remaining time
            chosen_duration = min(chosen_clip.duration, remaining_time, self.max_clip_duration)
            # Ensure it meets the minimum duration requirement
            chosen_duration = max(chosen_duration, self.min_clip_duration)
            # Ensure positive duration
            chosen_duration = max(0.01, chosen_duration)


            # --- Choose Effect Probabilistically ---
            effect_options = list(self.effects.values())
            # Calculate "desirability" based on low cost (tau*psi) and high gain (epsilon)
            # Add small epsilon to avoid division by zero
            efficiencies = [(e.epsilon + 1e-6) / (e.tau * e.psi + 1e-9) for e in effect_options]

            # Slightly boost the probability of a simple 'cut'
            try:
                cut_index = next((i for i, e in enumerate(effect_options) if e.type == "cut"), -1)
                if cut_index != -1: efficiencies[cut_index] *= 1.5 # Make cuts 50% more likely than efficiency suggests
            except Exception as e: logger.warning(f"Could not find 'cut' effect for boost: {e}")

            # Normalize efficiencies to probabilities
            positive_efficiencies = [max(0, eff) for eff in efficiencies] # Ensure non-negative
            total_efficiency = sum(positive_efficiencies)
            if total_efficiency > 1e-9:
                 effect_probs = [eff / total_efficiency for eff in positive_efficiencies]
                 # Renormalize just in case of floating point issues
                 sum_probs = sum(effect_probs)
                 if sum_probs > 1e-9: effect_probs = [p / sum_probs for p in effect_probs]
                 else: effect_probs = [1.0/len(effect_options)] * len(effect_options) # Fallback to uniform
                 chosen_effect = random.choices(effect_options, weights=effect_probs, k=1)[0]
            else:
                 chosen_effect = self.effects.get('cut', EffectParams(type='cut')) # Fallback to cut


            # --- Add to Sequence ---
            sequence_info.append((chosen_clip, chosen_duration, chosen_effect))
            last_clip_segment = chosen_clip
            current_time += chosen_duration
            # Remove chosen clip index from available indices for next iteration
            available_clip_indices.remove(original_list_idx)

        # --- Final Check on Sequence Length ---
        final_sim_duration = sum(item[1] for item in sequence_info)
        # Discard sequences significantly shorter than the target (e.g., less than half)
        if final_sim_duration < self.target_duration * 0.5:
             # logger.debug(f"Discarding short sequence simulation ({final_sim_duration:.2f}s / {self.target_duration:.2f}s)")
             return None
        return sequence_info

    def _evaluate_pareto(self, seq_info: List[Tuple[ClipSegment, float, EffectParams]]) -> List[float]:
        """Calculates the 5 objective scores for a given sequence."""
        if not seq_info: return [-1e9] * 5 # Return very poor scores for empty sequence
        num_clips = len(seq_info)
        total_duration = sum(item[1] for item in seq_info)
        if total_duration <= 0: return [-1e9] * 5

        # Objective Weights from config
        w_r, w_m, w_c, w_v, w_e = (self.w_r, self.w_m, self.w_c, self.w_v, self.w_e)
        sigma_m_sq = self.analysis_config.mood_similarity_variance**2 * 2
        kd = self.analysis_config.continuity_depth_weight
        lambda_penalty = self.analysis_config.variety_repetition_penalty

        # --- R(S): Rhythm Coherence (Negative average normalized offset from beats) ---
        r_score_sum = 0.0; num_transitions_rhythm = 0; current_t = 0.0
        for i, (clip, duration, effect) in enumerate(seq_info):
            transition_time = current_t + duration
            if i < num_clips - 1: # Only score transitions between clips
                nearest_b = self._nearest_beat_time(transition_time)
                if nearest_b is not None:
                    # Normalize offset by beat period
                    offset_norm = abs(transition_time - nearest_b) / (self.beat_period + 1e-6)
                    r_score_sum += offset_norm
                    num_transitions_rhythm += 1
            current_t = transition_time
        # Average normalized offset, negated because lower offset is better
        neg_r_score = -w_r * (r_score_sum / num_transitions_rhythm if num_transitions_rhythm > 0 else 1.0) # Penalize if no transitions

        # --- M(S): Mood Coherence (Average mood similarity between video clip and audio segment) ---
        m_score_sum = 0.0; current_t = 0.0
        for clip, duration, effect in seq_info:
            mid_time = current_t + duration / 2.0 # Check mood at clip midpoint
            audio_seg = self.get_audio_segment_at(mid_time)
            if audio_seg:
                vid_mood = np.asarray(clip.mood_vector)
                aud_mood = np.asarray(audio_seg.get('m_i', [0.0, 0.0]))
                mood_dist_sq = np.sum((vid_mood - aud_mood)**2)
                # Gaussian similarity based on mood distance
                m_score_sum += exp(-mood_dist_sq / (sigma_m_sq + 1e-9))
            current_t += duration
        m_score = w_m * (m_score_sum / num_clips if num_clips > 0 else 0.0)

        # --- C(S): Visual Continuity (Average transition smoothness + effect gain) ---
        c_score_sum = 0.0; num_transitions_cont = 0
        for i in range(num_clips - 1):
            clip1, _, effect_at_transition = seq_info[i] # Effect applied *at the end* of clip1
            clip2, _, _ = seq_info[i+1]
            # Get normalized feature vectors using AnalysisConfig for normalization params
            f1 = clip1.get_feature_vector(self.analysis_config) # [v_norm, a_norm, d_r]
            f2 = clip2.get_feature_vector(self.analysis_config)
            # Weighted Euclidean distance in feature space
            delta_e_sq = (f1[0]-f2[0])**2 + (f1[1]-f2[1])**2 + kd*(f1[2]-f2[2])**2
            # Normalize distance (max possible distance squared is 1^2+1^2+kd*1^2 = 2+kd)
            delta_e_norm_sq = delta_e_sq / (2.0 + kd + 1e-6)
            # Continuity = (1 - normalized distance) + perceptual gain of effect
            c_score_sum += (1.0 - np.sqrt(delta_e_norm_sq)) + effect_at_transition.epsilon
            num_transitions_cont += 1
        c_score = w_c * (c_score_sum / num_transitions_cont if num_transitions_cont > 0 else 1.0) # Assume perfect continuity if 0/1 clips

        # --- V(S): Dynamic Variety (Average frame entropy - repetition penalty) ---
        avg_phi = np.mean([item[0].phi for item in seq_info if item[0].phi is not None and np.isfinite(item[0].phi)]) if seq_info else 0.0
        repetition_count = 0; num_transitions_var = 0
        for i in range(num_clips - 1):
            # Penalize consecutive clips from the same source video
            if seq_info[i][0].source_video_path == seq_info[i+1][0].source_video_path:
                 repetition_count += 1
            num_transitions_var +=1
        repetition_term = lambda_penalty * (repetition_count / num_transitions_var if num_transitions_var > 0 else 0)
        # Normalize average entropy (approximate max entropy for 8-bit grayscale)
        max_entropy_approx = log(256); avg_phi_norm = avg_phi / max_entropy_approx if max_entropy_approx > 0 else 0.0
        v_score = w_v * (avg_phi_norm - repetition_term)

        # --- EC(S): Effect Efficiency Cost (Negative average cost = tau*psi/epsilon) ---
        ec_score_sum = 0.0
        for _, _, effect in seq_info:
             # Cost is high duration/impact, low perceptual gain
             cost = (effect.tau * effect.psi) / (effect.epsilon + 1e-9)
             ec_score_sum += cost
        # Average cost, negated because lower cost is better
        neg_ec_score = -w_e * (ec_score_sum / num_clips if num_clips > 0 else 0.0)

        return [neg_r_score, m_score, c_score, v_score, neg_ec_score]


    def _nearest_beat_time(self, time_sec):
        """Finds the beat time closest to the given time."""
        if not self.beat_times: return None
        beat_times_arr = np.asarray(self.beat_times)
        if len(beat_times_arr) == 0: return None
        # Find index of the minimum absolute difference
        closest_beat_idx = np.argmin(np.abs(beat_times_arr - time_sec))
        return beat_times_arr[closest_beat_idx]

    def _update_pareto_front(self, front: List[Tuple[List[Any], List[float]]], new_solution: Tuple[List[Any], List[float]]):
        """Updates the Pareto front with a new solution."""
        new_seq_info, new_scores = new_solution
        dominated_indices = [] # Indices in the current front dominated by the new solution
        is_dominated_by_front = False # Flag if the new solution is dominated by anything in the front

        # Compare new solution against the current front
        for i, (existing_seq_info, existing_scores) in enumerate(front):
            if self._dominates(new_scores, existing_scores):
                # New solution dominates an existing one
                dominated_indices.append(i)
            if self._dominates(existing_scores, new_scores):
                # New solution is dominated by an existing one
                is_dominated_by_front = True
                break # No need to check further if dominated

        # If the new solution is not dominated by any existing solution...
        if not is_dominated_by_front:
            # Remove all solutions from the front that are dominated by the new one
            # Iterate in reverse order to avoid index shifting issues
            for i in sorted(dominated_indices, reverse=True):
                del front[i]
            # Add the new non-dominated solution to the front
            front.append(new_solution)


    def _dominates(self, scores1: List[float], scores2: List[float]) -> bool:
        """Checks if solution 1 dominates solution 2."""
        # Assumes all objectives are to be maximized
        # (negated objectives like rhythm/efficiency are handled before this)
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have the same length for dominance check.")

        at_least_one_better = False
        for s1, s2 in zip(scores1, scores2):
            if s1 < s2: # scores1 is worse in at least one objective
                return False
            if s1 > s2: # scores1 is better in at least one objective
                at_least_one_better = True
        # To dominate, scores1 must be >= scores2 in all objectives AND > in at least one
        return at_least_one_better

# print("DEBUG: Defined SequenceBuilderPhysicsMC") # REMOVED Debug Print

# ========================================================================
#    FFMPEG VIDEO BUILDING FUNCTION (VidGear/WriteGear) - Uses RenderConfig
# ========================================================================

# Basic input sanitization (expand as needed)
def _sanitize_ffmpeg_input(input_str: str) -> str:
    """Basic sanitization for FFmpeg path inputs."""
    if input_str is None: return ""
    safe_str = str(input_str)

    # ---> ADD THIS LINE TO FIX THE SPECIAL SPACE ISSUE <---
    safe_str = safe_str.replace('\u202f', ' ') # Replace narrow no-break space with regular space
    # ------------------------------------------------------

    # Existing sanitization for command injection risks (keep these)
    safe_str = safe_str.replace(';', '_').replace('&', '_').replace('|', '_').replace('`', '_')
    # Ensure paths don't contain newlines
    safe_str = safe_str.replace('\n', ' ').replace('\r', ' ')
    return safe_str


# Note: Pass RenderConfig
def _generate_ffmpeg_filter_for_effect(effect: Optional[EffectParams], segment_context: ClipSegment, render_config: RenderConfig, clip_duration: float, clip_fps: float) -> Optional[str]:
    """Generates FFmpeg filter string snippet for effects using RenderConfig."""
    if not effect or effect.type == "cut" or clip_duration <= 0 or clip_fps <=0:
        return None # No filter for cuts or invalid inputs

    filter_str = None
    # Get normalization values from RenderConfig (needed for intensity scaling)
    v_max_norm = render_config.norm_max_velocity + 1e-6
    a_max_norm = render_config.norm_max_acceleration + 1e-6

    try:
        # Use effect parameters (tau, psi, epsilon) from the EffectParams object stored on the clip
        tau = effect.tau # Duration of effect
        psi = effect.psi # Physical impact proxy (intensity factor)
        # Epsilon (perceptual gain) is used in scoring, not directly in filter generation here

        if effect.type == "zoom":
            # Scale zoom intensity based on normalized clip velocity (v_k)
            zoom_intensity = np.clip(segment_context.v_k / v_max_norm, 0.0, 1.0)
            # Effective duration of the zoom effect (cannot exceed clip duration)
            eff_dur_frames = max(1, int(min(tau, clip_duration) * clip_fps))
            # Calculate target scale factor based on base impact (psi) and intensity
            end_scale = 1.0 + psi * zoom_intensity
            # FFmpeg zoompan expression: linear zoom over eff_dur_frames
            # 'on' is input frame number. Zoom from 1 to end_scale over eff_dur_frames.
            # Hold zoom at end_scale after eff_dur_frames.
            # Need to escape commas and quotes carefully for filter_complex
            zoom_expr = f"'if(lte(on,{eff_dur_frames}),lerp(1,{end_scale:.4f},on/{eff_dur_frames}),{end_scale:.4f})'"
            # Center the zoom: x and y expressions position the top-left corner
            x_expr = "'iw/2-(iw/zoom/2)'"
            y_expr = "'ih/2-(ih/zoom/2)'"
            # Use d=1 for continuous zoom? fps sets the output rate.
            # Setting d to total output frames might be better if zoom should last whole clip. Test this.
            # Using d=1 and relying on the expression over 'on' is often more reliable per frame.
            filter_str = f"zoompan=z={zoom_expr}:x={x_expr}:y={y_expr}:d=1:fps={clip_fps}"

        elif effect.type == "pan":
             # Scale pan intensity based on normalized clip acceleration (a_j)
             pan_intensity = np.clip(segment_context.a_j / a_max_norm, 0.0, 1.0)
             # Effective duration of the pan
             eff_dur_sec = min(tau, clip_duration)
             if eff_dur_sec > 0:
                 # Calculate target pan distance as a fraction of width
                 pan_distance_pixels = f"iw*{psi * pan_intensity:.4f}"
                 # Linear pan over eff_dur_sec: 't' is time in seconds
                 # Pan distance = min(speed * t, max_distance)
                 # Speed = max_distance / eff_dur_sec
                 # x offset = min( (iw*psi*intensity / eff_dur_sec) * t, iw*psi*intensity )
                 # Careful with escaping for FFmpeg expression
                 x_offset_expr = f"'min(({pan_distance_pixels}/{eff_dur_sec:.4f})*t,{pan_distance_pixels})'"
                 # Use crop filter to achieve the pan by moving the crop window
                 filter_str = f"crop=iw:ih:x={x_offset_expr}:y=0"
             else: filter_str = None # No pan if effect duration is zero

        # --- Add Fade Effect ---
        # Note: This example only adds zoom/pan. Fades are often handled by FFmpeg's concat demuxer
        # transitions or specific fade filters applied *between* segments in filter_complex.
        # Implementing fades here would require more complex filtergraph generation (e.g., overlay, blend).
        # WriteGear doesn't directly support concat demuxer transitions easily with custom commands.
        # We'll stick to per-segment effects for now. Fades might need separate handling if critical.

    except Exception as e:
        logger.warning(f"Error generating filter for effect '{effect.type}': {e}", exc_info=False)
        return None

    # Ensure filter string is properly formatted if created
    return filter_str.strip() if filter_str else None


# --- Primary Render Function (using WriteGear) ---
def _render_with_vidgear_writegear(final_sequence: List[ClipSegment], output_video_path: str, audio_path: str, render_config: RenderConfig) -> bool:
    """Renders the video using FFmpeg via VidGear's WriteGear."""
    logger.info(f"Attempting render with VidGear WriteGear to {os.path.basename(output_video_path)}")
    start_time = time.time()

    # --- Build FFmpeg Command Components ---
    input_definitions = [] # Stores "-ss", "-i", path, "-to" parts
    filter_complex_parts = [] # Stores individual filter chains for each segment and concat
    input_streams_for_concat = [] # Stores names like "[v0]", "[v1_filtered]" for concat

    logger.debug("Preparing FFmpeg command components...")

    # --- DEBUG: Limit number of segments for testing ---
    # MAX_RENDER_SEGMENTS = 3 # <<< Limit to first 3 clips
    # logger.warning(f"---!!! DEBUGGING: Rendering only first {MAX_RENDER_SEGMENTS} segments !!!---")
    # segments_to_render = final_sequence[:MAX_RENDER_SEGMENTS]
    segments_to_render = final_sequence # Use full sequence
    # ----------------------------------------------------

    valid_segment_count = 0
    # Use the (potentially limited) list for generating the command
    for i, segment in enumerate(segments_to_render):
        
         # ---> ADD THESE TWO LINES <---
        logger.debug(f"DEBUG RENDER LOOP: Segment {i}, Path from object: '{segment.source_video_path}'")
        if not isinstance(segment.source_video_path, str):
             logger.error(f"DEBUG RENDER LOOP: Segment {i}, Path is NOT a string! Type: {type(segment.source_video_path)}")
        # Basic validation for segment data
        if not isinstance(segment, ClipSegment) or not hasattr(segment, 'source_video_path') or not os.path.exists(segment.source_video_path):
             logger.warning(f"Skipping invalid segment at index {i}: Invalid object or source path '{getattr(segment, 'source_video_path', 'N/A')}'.")
             continue
        if segment.chosen_duration <= 0.01 or segment.fps <= 0:
             logger.warning(f"Skipping segment {i} due to zero/negative duration ({segment.chosen_duration:.3f}s) or FPS ({segment.fps}).")
             continue

        source_path = _sanitize_ffmpeg_input(segment.source_video_path)
        start_t = segment.subclip_start_time_in_source
        duration = segment.chosen_duration # Use the chosen duration for this segment
        end_t = start_t + duration

        # --- Define Input ---
        # Use -ss before -i for faster seeking (input seeking)
        input_definitions.extend(["-ss", f"{start_t:.5f}"])
        input_definitions.extend(["-i", source_path])
        # Use -to or -t to specify duration/end time. -to is often more precise with -ss.
        input_definitions.extend(["-to", f"{end_t:.5f}"]) # Specify end time relative to original file start

        # --- Define Filter Chain for this Segment ---
        input_video_stream = f"[{valid_segment_count}:v]" # Use valid_segment_count as input index
        processed_stream_name = f"[v{valid_segment_count}]" # Base name for this segment's output

        # Set PTS to start from zero for each segment before filtering/concat
        pts_filter = f"{input_video_stream}setpts=PTS-STARTPTS[pts{valid_segment_count}]"
        filter_complex_parts.append(pts_filter)
        current_stream_name = f"[pts{valid_segment_count}]" # Output of setpts

        # Generate effect filter if applicable
        effect = segment.chosen_effect # Should be EffectParams object
        # Pass RenderConfig to filter generation
        effect_filter_str = _generate_ffmpeg_filter_for_effect(effect, segment, render_config, duration, segment.fps)

        if effect_filter_str:
            filtered_stream_name = f"[veffect{valid_segment_count}]"
            # Apply effect filter to the PTS-reset stream
            full_effect_filter = f"{current_stream_name}{effect_filter_str}{filtered_stream_name}"
            filter_complex_parts.append(full_effect_filter)
            # Update the name of the stream to be used in concat
            input_streams_for_concat.append(filtered_stream_name)
        else:
            # If no effect, the PTS-reset stream is the final one for this segment
             input_streams_for_concat.append(current_stream_name)

        valid_segment_count += 1 # Increment *only* for valid segments added

    if valid_segment_count == 0:
        logger.error("No valid clips found to include in FFmpeg command after filtering sequence.")
        return False # Indicate failure

    # --- Define Concatenation Filter ---
    concat_filter = "".join(input_streams_for_concat) + f"concat=n={valid_segment_count}:v=1:a=0[outv]"
    filter_complex_parts.append(concat_filter)

    # --- Define Audio Input ---
    audio_input_index = valid_segment_count # Audio is the next input after all video segments
    input_definitions.extend(["-i", _sanitize_ffmpeg_input(audio_path)])

    # --- Assemble Custom FFmpeg Command List for WriteGear ---
    custom_ffmpeg_cmd = []
    custom_ffmpeg_cmd.extend(input_definitions) # Add all input definitions first
    custom_ffmpeg_cmd.extend(["-filter_complex", ";".join(filter_complex_parts)]) # Add complex filtergraph
    # Map the final video output stream and the audio input stream
    custom_ffmpeg_cmd.extend(["-map", "[outv]", "-map", f"{audio_input_index}:a"])

    # --- Define Output Parameters (for WriteGear's -output_params) ---
    output_params_dict = {
        "-c:v": render_config.video_codec, "-preset": render_config.preset,
        "-crf": str(render_config.crf), "-c:a": render_config.audio_codec,
        "-b:a": render_config.audio_bitrate, "-shortest": "", # End when shortest input (audio) ends
        "-threads": str(render_config.threads), "-y": "", # Overwrite output without asking
        # WriteGear handles FFmpeg log level via its `logging` parameter
    }
    # Optional: Add pix_fmt for compatibility if needed
    if render_config.video_codec == 'libx264':
        output_params_dict["-pix_fmt"] = "yuv420p"

    # --- Prepare parameters for WriteGear initializer ---
    # Combine custom command and output parameters into a single dict for WriteGear
    writegear_params = {
        "-custom_ffmpeg": custom_ffmpeg_cmd,
        "-output_params": output_params_dict,
        # Add any other direct WriteGear parameters if needed, e.g. -input_framerate
    }

    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    except OSError as e:
         logger.error(f"Failed to create output directory {os.path.dirname(output_video_path)}: {e}")
         return False

    # --- Execute with WriteGear ---
    writer = None
    try:
        # Log parameters before initializing
        logger.debug(f"WriteGear output path: {output_video_path}")
        # Use a try-except for JSON dump in case params are not serializable (though they should be)
        try:
            # Use repr for potentially very long list, avoid excessive logging
            cmd_repr = repr(writegear_params["-custom_ffmpeg"])
            if len(cmd_repr) > 1000: cmd_repr = cmd_repr[:500] + "..." + cmd_repr[-500:]
            params_log = {
                "-custom_ffmpeg": cmd_repr, # Log truncated repr
                "-output_params": writegear_params["-output_params"]
            }
            logger.debug(f"WriteGear Params Dictionary (Truncated): {json.dumps(params_log, indent=2)}")
        except Exception: # Catch broader errors during logging prep
             logger.debug(f"WriteGear Params Dictionary (Logging Error): {writegear_params}")


        # Initialize WriteGear (logging=True helps debugging)
        # Pass the combined dictionary using **
        writer = WriteGear(
            output=output_video_path, # <<< RENAMED from output_filename
            logging=True, # Set to True to see FFmpeg output/errors in console/log
            **writegear_params
        )

        logger.info("Executing FFmpeg via VidGear WriteGear...")

        # Close WriteGear to finalize the FFmpeg process
        # When using -custom_ffmpeg like this, close() triggers the execution.
        writer.close()

        # --- Check Result ---
        # WriteGear might not raise an exception even if FFmpeg fails internally. Check output file.
        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) < 100: # Check for minimal size
            logger.error(f"WriteGear execution finished, but output file is missing or too small: {output_video_path}")
            logger.error("Check console/log output above for FFmpeg errors (ensure WriteGear logging=True).")
            # Attempt to delete potentially invalid output file
            if os.path.exists(output_video_path):
                try:
                    os.remove(output_video_path)
                except OSError as del_err:
                     logger.warning(f"Could not remove potentially invalid output file {output_video_path}: {del_err}")
            return False
        else:
            logger.info(f"Render successful via VidGear WriteGear ({time.time() - start_time:.2f}s)")
            return True

    except Exception as render_err:
        logger.error(f"Error during VidGear WriteGear setup or execution: {render_err}", exc_info=True)
        logger.error("Check console/log output above for FFmpeg errors (ensure WriteGear logging=True).")
        return False
    finally:
        if writer is not None:
            # Close should be called in the main flow, but ensure cleanup if error occurred before close
            try:
                # WriteGear's close() might have already been called or failed.
                # Calling it again might not be necessary or could raise issues if already closed.
                # Let's ensure it's called if the writer object exists.
                if not getattr(writer, "closed", False): # Check if it's not already marked closed
                     writer.close()
            except Exception as close_err:
                 logger.debug(f"Minor error during final WriteGear close: {close_err}")
            logger.debug("WriteGear resources potentially closed/terminated.")


# --- Placeholder Fallback Functions ---
def _render_with_moviepy_fallback(*args, **kwargs):
    logger.warning("MoviePy fallback rendering not implemented.")
    raise NotImplementedError("MoviePy fallback is not available.")

def _render_with_opencv_fallback(*args, **kwargs):
    logger.warning("OpenCV fallback rendering not implemented.")
    raise NotImplementedError("OpenCV fallback is not available.")

# --- Main Build Function with Fallback Structure ---
def buildSequenceVideo(final_sequence: List[ClipSegment], output_video_path: str, audio_path: str, render_config: RenderConfig):
    """Builds the final video using WriteGear, with structure for fallbacks."""
    # Enhanced input validation
    if not final_sequence:
        logger.error("Cannot build video: Input sequence is empty.")
        raise ValueError("Empty sequence: No video clips to process")
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Cannot build video: Master audio file not found at {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not output_video_path:
         logger.error("Cannot build video: Output path not specified.")
         raise ValueError("Output video path is required.")

    # Attempt rendering strategies
    rendering_strategies = [
        _render_with_vidgear_writegear, # <<< UPDATED to use WriteGear function
        # _render_with_moviepy_fallback, # Requires optional moviepy install
        # _render_with_opencv_fallback,  # Very complex
    ]

    success = False
    last_exception = None
    for strategy in rendering_strategies:
        logger.info(f"Attempting rendering strategy: {strategy.__name__}")
        try:
            # Memory/Time tracking per strategy attempt
            tracemalloc.start()
            strategy_start_time = time.time()
            start_memory = tracemalloc.get_traced_memory()[0]

            success = strategy(final_sequence, output_video_path, audio_path, render_config)

            end_memory = tracemalloc.get_traced_memory()[0]
            strategy_end_time = time.time()
            # Get peak memory usage during the strategy execution
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop() # Stop tracing *after* getting peak

            logger.info(f"Strategy {strategy.__name__} Performance:")
            logger.info(f"  Time: {strategy_end_time - strategy_start_time:.2f}s")
            # Note: tracemalloc tracks Python memory, not FFmpeg's external process memory
            logger.info(f"  Python Memory Delta (Current): {(end_memory - start_memory) / 1024**2:.2f} MB")
            logger.info(f"  Python Memory Peak: {peak_mem / 1024**2:.2f} MB")


            if success:
                logger.info(f"Rendering successful using {strategy.__name__}.")
                break # Exit loop on success
            else:
                 logger.warning(f"Strategy {strategy.__name__} reported failure.")
                 last_exception = RuntimeError(f"{strategy.__name__} reported failure.") # Store generic error

        except NotImplementedError:
             logger.info(f"Strategy {strategy.__name__} is not implemented. Skipping.")
             last_exception = NotImplementedError(f"{strategy.__name__} not implemented.")
             if tracemalloc.is_tracing(): tracemalloc.stop() # Ensure stop on skip
        except Exception as e:
            logger.error(f"Strategy {strategy.__name__} failed with exception: {e}", exc_info=True)
            last_exception = e
            # Ensure tracemalloc is stopped even if strategy fails mid-way
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            # Try next strategy

    if not success:
        logger.critical("All video rendering strategies failed.")
        # Raise the exception from the *last failed* strategy for better context
        if last_exception:
            raise RuntimeError("All video rendering strategies failed.") from last_exception
        else:
            raise RuntimeError("All video rendering strategies failed (no specific exception captured).")

# ========================================================================
#         WORKER FUNCTION FOR PARALLEL PROCESSING (Uses AnalysisConfig)
# ========================================================================
def process_single_video(video_path: str, audio_data: Dict, analysis_config: AnalysisConfig, output_dir: str) -> Tuple[str, str, List[ClipSegment]]:
    """Worker function: Analyzes video, returns potential clips."""
    start_t = time.time()
    pid = os.getpid()
    thread_name = threading.current_thread().name # Get thread name if available
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Configure logging within the worker process if needed (basic setup might inherit)
    # If using ProcessPoolExecutor, separate logger setup might be beneficial
    worker_logger = logging.getLogger(f"Worker.{pid}.{thread_name}") # More specific logger name
    worker_logger.info(f"Starting Analysis: {base_name}")

    status = "Unknown Error"
    potential_clips: List[ClipSegment] = []
    frame_features: List[Dict] = [] # Keep features local to worker

    try:
        # Instantiate analyzer *within* the worker process
        analyzer = VideousMain()
        # Analyze video using the provided config and audio data
        # The modified analyzeVideo (using VidGear) will be called here
        frame_features, potential_clips_result = analyzer.analyzeVideo(video_path, analysis_config, audio_data)

        if potential_clips_result is None:
            status = "Analysis Failed (analyzeVideo returned None)"
            potential_clips = []
        elif not potential_clips_result:
             status = "Analysis OK (0 potential clips found)"
             potential_clips = []
        else:
            # Ensure result is a list of ClipSegments before returning
            potential_clips = [clip for clip in potential_clips_result if isinstance(clip, ClipSegment)]
            status = f"Analysis OK ({len(potential_clips)} potential clips)"
            # Save analysis data if requested *and* features were generated
            if analysis_config.save_analysis_data and frame_features:
                try:
                     # Pass config for saving if needed by ClipSegment serialization
                     analyzer.saveAnalysisData(video_path, frame_features, potential_clips, output_dir, analysis_config)
                except Exception as save_err:
                     # Log save error but don't fail the whole worker process
                     worker_logger.error(f"Failed to save analysis data for {base_name}: {save_err}", exc_info=True)

    except Exception as e:
        # Log the full traceback from the worker process for debugging
        status = f"Failed: {type(e).__name__}"
        worker_logger.error(f"!!! FATAL ERROR analyzing {base_name} in worker {pid} !!!", exc_info=True)
        potential_clips = [] # Ensure empty list is returned on error
    finally:
        # Explicitly delete large objects to potentially help memory management in worker
        if 'frame_features' in locals(): del frame_features
        if 'potential_clips_result' in locals(): del potential_clips_result # If it exists
        if 'analyzer' in locals(): del analyzer # Delete the analyzer instance
        pass

    end_t = time.time()
    worker_logger.info(f"Finished Analysis {base_name} ({status}) in {end_t - start_t:.2f}s")
    # Return path, status, and the list of ClipSegment objects
    # The ClipSegment objects themselves need to be pickleable for multiprocessing
    return (video_path, status, potential_clips)

# ========================================================================
#                      APP INTERFACE - Uses Dataclasses
# ========================================================================
class VideousApp(customtkinter.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("Videous Chef - v4.5 (WriteGear Render)") # Updated Title
        self.geometry("950x850")

        # Core State
        self.video_files: List[str] = []
        self.beat_track_path: Optional[str] = None
        self.analysis_config: Optional[AnalysisConfig] = None
        self.render_config: Optional[RenderConfig] = None
        self.is_processing = False
        self.master_audio_data: Optional[Dict] = None
        self.all_potential_clips: List[ClipSegment] = []

        # Process/Thread Management
        self.processing_thread = None
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.analysis_futures: List[concurrent.futures.Future] = []
        self.futures_map: Dict[concurrent.futures.Future, str] = {}
        self.total_tasks = 0
        self.completed_tasks = 0

        # Output Directories
        self.output_dir = "output_videous_chef"
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.analysis_subdir = os.path.join(self.output_dir, "analysis_data")
            os.makedirs(self.analysis_subdir, exist_ok=True)
            self.render_subdir = os.path.join(self.output_dir, "final_renders")
            os.makedirs(self.render_subdir, exist_ok=True)
        except OSError as e:
             logger.critical(f"Failed to create output directories: {e}. Exiting.", exc_info=True)
             messagebox.showerror("Fatal Error", f"Could not create output directories in {self.output_dir}.\nPlease check permissions.\n\n{e}")
             sys.exit(1)


        # Fonts
        try:
            self.header_font = customtkinter.CTkFont(family="Garamond", size=28, weight="bold")
            self.label_font = customtkinter.CTkFont(family="Garamond", size=14)
            self.button_font = customtkinter.CTkFont(family="Garamond", size=12)
            self.dropdown_font = customtkinter.CTkFont(family="Garamond", size=12)
            self.small_font = customtkinter.CTkFont(family="Garamond", size=10)
            self.mode_font = customtkinter.CTkFont(family="Garamond", size=13, weight="bold")
            self.tab_font = customtkinter.CTkFont(family="Garamond", size=14, weight="bold")
            self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")
        except Exception as font_e:
             logger.warning(f"Garamond font not found, using defaults: {font_e}")
             self.header_font = customtkinter.CTkFont(size=28, weight="bold")
             self.label_font = customtkinter.CTkFont(size=14)
             self.button_font = customtkinter.CTkFont(size=12)
             self.dropdown_font = customtkinter.CTkFont(size=12)
             self.small_font = customtkinter.CTkFont(size=10)
             self.mode_font = customtkinter.CTkFont(size=13, weight="bold")
             self.tab_font = customtkinter.CTkFont(size=14, weight="bold")
             self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")


        # UI Setup
        self._setup_luxury_theme()
        self._build_ui() # This implicitly calls _create_config_sliders
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.info("VideousApp initialized.")

    def _setup_luxury_theme(self):
        self.diamond_white = "#F5F6F5"; self.deep_black = "#1C2526";
        self.gold_accent = "#D4AF37"; self.jewel_blue = "#2A4B7C";
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        self.configure(bg=self.deep_black)

    def _button_styles(self, border_width=1):
        return { "corner_radius": 8, "fg_color": self.jewel_blue, "hover_color": self.gold_accent,
            "border_color": self.diamond_white, "border_width": border_width, "text_color": self.diamond_white }

    def _radio_styles(self):
        return { "border_color": self.diamond_white, "fg_color": self.jewel_blue,
            "hover_color": self.gold_accent, "text_color": self.diamond_white }

    def _build_ui(self):
        # --- Grid Configuration ---
        self.grid_columnconfigure(0, weight=4); self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(1, weight=1); self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0); self.grid_rowconfigure(4, weight=0)

        # --- Header ---
        header = customtkinter.CTkLabel(self, text="Videous Chef - Remix Engine", font=self.header_font, text_color=self.gold_accent)
        header.grid(row=0, column=0, columnspan=2, pady=(15, 10), sticky="ew")

        # --- Left Column: Config Tabs ---
        config_outer_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15)
        config_outer_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nsew")
        config_outer_frame.grid_rowconfigure(0, weight=1); config_outer_frame.grid_columnconfigure(0, weight=1)
        self.tab_view = customtkinter.CTkTabview(config_outer_frame, fg_color=self.deep_black, segmented_button_fg_color=self.deep_black,
                                                 segmented_button_selected_color=self.jewel_blue, segmented_button_selected_hover_color=self.gold_accent,
                                                 segmented_button_unselected_color="#333", segmented_button_unselected_hover_color="#555",
                                                 text_color=self.diamond_white, border_color=self.diamond_white, border_width=2)
        self.tab_view.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.tab_view.add("Shared"); self.tab_view.add("Greedy Heuristic"); self.tab_view.add("Physics MC")
        self.tab_view._segmented_button.configure(font=self.tab_font)
        # Make tab frames scrollable
        self.shared_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Shared"), fg_color="transparent")
        self.shared_tab_frame.pack(expand=True, fill="both")
        self.greedy_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Greedy Heuristic"), fg_color="transparent")
        self.greedy_tab_frame.pack(expand=True, fill="both")
        self.physics_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Physics MC"), fg_color="transparent")
        self.physics_tab_frame.pack(expand=True, fill="both")
        self._create_config_sliders() # Populates tabs

        # --- Right Column: Files ---
        files_outer_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        files_outer_frame.grid(row=1, column=1, padx=(5, 10), pady=5, sticky="nsew")
        files_outer_frame.grid_rowconfigure(0, weight=0); files_outer_frame.grid_rowconfigure(1, weight=1); files_outer_frame.grid_columnconfigure(0, weight=1)
        # Master Audio Frame
        controls_frame = customtkinter.CTkFrame(files_outer_frame, fg_color=self.deep_black, border_color=self.diamond_white, border_width=2, corner_radius=10)
        controls_frame.grid(row=0, column=0, padx=0, pady=(0, 5), sticky="new")
        beat_track_frame = customtkinter.CTkFrame(controls_frame, fg_color="transparent")
        beat_track_frame.pack(pady=(10, 10), padx=10, fill="x")
        customtkinter.CTkLabel(beat_track_frame, text="2. Master Audio Track (The Base)", anchor="w", font=self.label_font, text_color=self.diamond_white).pack(pady=(0, 2), anchor="w")
        beat_btn_frame = customtkinter.CTkFrame(beat_track_frame, fg_color="transparent")
        beat_btn_frame.pack(fill="x")
        self.beat_track_button = customtkinter.CTkButton(beat_btn_frame, text="Select Audio/Video", font=self.button_font, command=self._select_beat_track, **self._button_styles())
        self.beat_track_button.pack(side="left", padx=(0, 10))
        self.beat_track_label = customtkinter.CTkLabel(beat_btn_frame, text="No master track selected.", anchor="w", wraplength=300, font=self.small_font, text_color=self.diamond_white)
        self.beat_track_label.pack(side="left", fill="x", expand=True)
        # Video Files Frame
        video_files_frame = customtkinter.CTkFrame(files_outer_frame, fg_color=self.deep_black, border_color=self.diamond_white, border_width=2, corner_radius=10)
        video_files_frame.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")
        video_files_frame.grid_rowconfigure(1, weight=1); video_files_frame.grid_columnconfigure(0, weight=1)
        customtkinter.CTkLabel(video_files_frame, text="3. Source Videos (The Ingredients)", anchor="w", font=self.label_font, text_color=self.diamond_white).grid(row=0, column=0, columnspan=2, pady=(5, 2), padx=10, sticky="ew")
        list_frame = Frame(video_files_frame, bg=self.deep_black, highlightbackground=self.diamond_white, highlightthickness=1)
        list_frame.grid(row=1, column=0, columnspan=2, pady=5, padx=10, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1); list_frame.grid_columnconfigure(0, weight=1)
        self.video_listbox = Listbox(list_frame, selectmode=MULTIPLE, bg=self.deep_black, fg=self.diamond_white, borderwidth=0, highlightthickness=0, width=50, font=("Garamond", 12), selectbackground=self.jewel_blue, selectforeground=self.gold_accent)
        self.video_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = Scrollbar(list_frame, orient="vertical", command=self.video_listbox.yview, background=self.deep_black, troughcolor=self.jewel_blue) # Style scrollbar
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.video_listbox.configure(yscrollcommand=scrollbar.set)
        # Enable Drag & Drop
        self.video_listbox.drop_target_register(DND_FILES); self.video_listbox.dnd_bind('<<Drop>>', self._handle_drop)
        # Listbox Buttons
        list_button_frame = customtkinter.CTkFrame(video_files_frame, fg_color="transparent")
        list_button_frame.grid(row=2, column=0, columnspan=2, pady=(5, 10), padx=10, sticky="ew")
        list_button_frame.grid_columnconfigure((0, 1, 2), weight=1) # Space buttons evenly
        self.add_button = customtkinter.CTkButton(list_button_frame, text="Add", width=70, font=self.button_font, command=self._add_videos_manual, **self._button_styles()); self.add_button.grid(row=0, column=0, padx=5, sticky="ew")
        self.remove_button = customtkinter.CTkButton(list_button_frame, text="Remove", width=70, font=self.button_font, command=self._remove_selected_videos, **self._button_styles()); self.remove_button.grid(row=0, column=1, padx=5, sticky="ew")
        self.clear_button = customtkinter.CTkButton(list_button_frame, text="Clear", width=70, font=self.button_font, command=self._clear_video_list, **self._button_styles()); self.clear_button.grid(row=0, column=2, padx=5, sticky="ew")

        # --- Bottom Frame: Mode Select & Action ---
        bottom_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=(5, 5), padx=10, sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1); bottom_frame.grid_columnconfigure(1, weight=1)
        mode_inner_frame = customtkinter.CTkFrame(bottom_frame, fg_color="transparent")
        mode_inner_frame.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="ew")
        customtkinter.CTkLabel(mode_inner_frame, text="Sequencing Mode:", font=self.mode_font, text_color=self.gold_accent).pack(side="left", padx=(5, 10))
        self.mode_var = tkinter.StringVar(value="Greedy Heuristic"); # Default mode
        mode_selector = customtkinter.CTkSegmentedButton(mode_inner_frame, values=["Greedy Heuristic", "Physics Pareto MC"], variable=self.mode_var,
                                                        font=self.button_font, selected_color=self.jewel_blue, selected_hover_color=self.gold_accent,
                                                        unselected_color="#333", unselected_hover_color="#555", text_color=self.diamond_white, command=self._mode_changed)
        mode_selector.pack(side="left", expand=True, fill="x")
        self.run_button = customtkinter.CTkButton(bottom_frame, text="4. Compose Video Remix", height=45, font=customtkinter.CTkFont(family="Garamond", size=16, weight="bold"), command=self._start_processing, **self._button_styles(border_width=2))
        self.run_button.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="e")

        # --- Status Label & Footer ---
        self.status_label = customtkinter.CTkLabel(self, text="Ready for Chef's command.", anchor="w", font=self.button_font, text_color=self.diamond_white, wraplength=900)
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(0, 5), padx=20, sticky="ew")
        footer = customtkinter.CTkLabel(self, text="Videous Chef v4.5 - WriteGear Render", font=self.small_font, text_color=self.gold_accent)
        footer.grid(row=4, column=1, pady=5, padx=10, sticky="se")

    def _mode_changed(self, value):
        logger.info(f"Sequencing mode changed to: {value}")
        # Update status or provide context if needed
        self.status_label.configure(text=f"Mode set to: {value}. Ready.", text_color=self.diamond_white)

    def _create_config_sliders(self):
        # --- Define Slider Variables (using defaults from dataclass where possible) ---
        # Need to instantiate AnalysisConfig with defaults to get values
        default_analysis_cfg = AnalysisConfig()
        default_render_cfg = RenderConfig() # Use for render-related defaults

        self.slider_vars = {
            # Shared
            'min_sequence_clip_duration': tkinter.DoubleVar(value=default_analysis_cfg.min_sequence_clip_duration),
            'max_sequence_clip_duration': tkinter.DoubleVar(value=default_analysis_cfg.max_sequence_clip_duration),
            'norm_max_velocity': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_velocity),
            'norm_max_acceleration': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_acceleration),
            'norm_max_rms': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_rms),
            'norm_max_onset': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_onset),
            'norm_max_chroma_var': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_chroma_var),
            'norm_max_depth_variance': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_depth_variance),
            'min_face_confidence': tkinter.DoubleVar(value=default_analysis_cfg.min_face_confidence),
            'mouth_open_threshold': tkinter.DoubleVar(value=default_analysis_cfg.mouth_open_threshold),
            'save_analysis_data': tkinter.BooleanVar(value=default_analysis_cfg.save_analysis_data),

            # Greedy
            'audio_weight': tkinter.DoubleVar(value=default_analysis_cfg.audio_weight),
            'kinetic_weight': tkinter.DoubleVar(value=default_analysis_cfg.kinetic_weight),
            'sharpness_weight': tkinter.DoubleVar(value=default_analysis_cfg.sharpness_weight),
            'camera_motion_weight': tkinter.DoubleVar(value=default_analysis_cfg.camera_motion_weight),
            'face_size_weight': tkinter.DoubleVar(value=default_analysis_cfg.face_size_weight),
            'percussive_weight': tkinter.DoubleVar(value=default_analysis_cfg.percussive_weight),
            'depth_weight': tkinter.DoubleVar(value=default_analysis_cfg.depth_weight),
            'score_threshold': tkinter.DoubleVar(value=default_analysis_cfg.score_threshold),
            'beat_boost': tkinter.DoubleVar(value=default_analysis_cfg.beat_boost),
            'beat_boost_radius_sec': tkinter.DoubleVar(value=default_analysis_cfg.beat_boost_radius_sec),
            'min_potential_clip_duration_sec': tkinter.DoubleVar(value=default_analysis_cfg.min_potential_clip_duration_sec),
            'pacing_variation_factor': tkinter.DoubleVar(value=default_analysis_cfg.pacing_variation_factor),
            'variety_penalty_source': tkinter.DoubleVar(value=default_analysis_cfg.variety_penalty_source),
            'variety_penalty_shot': tkinter.DoubleVar(value=default_analysis_cfg.variety_penalty_shot),
            'variety_penalty_intensity': tkinter.DoubleVar(value=default_analysis_cfg.variety_penalty_intensity),
            'beat_sync_bonus': tkinter.DoubleVar(value=default_analysis_cfg.beat_sync_bonus),
            'section_match_bonus': tkinter.DoubleVar(value=default_analysis_cfg.section_match_bonus),
            'candidate_pool_size': tkinter.IntVar(value=default_analysis_cfg.candidate_pool_size),
            'min_pose_confidence': tkinter.DoubleVar(value=default_analysis_cfg.min_pose_confidence),
            'model_complexity': tkinter.IntVar(value=default_analysis_cfg.model_complexity),
            'use_heuristic_segment_id': tkinter.BooleanVar(value=default_analysis_cfg.use_heuristic_segment_id),

            # Physics
            'fit_weight_velocity': tkinter.DoubleVar(value=default_analysis_cfg.fit_weight_velocity),
            'fit_weight_acceleration': tkinter.DoubleVar(value=default_analysis_cfg.fit_weight_acceleration),
            'fit_weight_mood': tkinter.DoubleVar(value=default_analysis_cfg.fit_weight_mood),
            'fit_sigmoid_steepness': tkinter.DoubleVar(value=default_analysis_cfg.fit_sigmoid_steepness),
            'objective_weight_rhythm': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_rhythm),
            'objective_weight_mood': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_mood),
            'objective_weight_continuity': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_continuity),
            'objective_weight_variety': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_variety),
            'objective_weight_efficiency': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_efficiency),
            'mc_iterations': tkinter.IntVar(value=default_analysis_cfg.mc_iterations),
            'mood_similarity_variance': tkinter.DoubleVar(value=default_analysis_cfg.mood_similarity_variance),
            'continuity_depth_weight': tkinter.DoubleVar(value=default_analysis_cfg.continuity_depth_weight),
            'variety_repetition_penalty': tkinter.DoubleVar(value=default_analysis_cfg.variety_repetition_penalty),

            # Effects (These configure RenderConfig - using defaults from RenderConfig)
            'effect_fade_duration': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('fade', EffectParams()).tau),
            'effect_zoom_duration': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('zoom', EffectParams()).tau),
            'effect_zoom_impact': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('zoom', EffectParams()).psi),
            'effect_zoom_perceptual': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('zoom', EffectParams()).epsilon),
            'effect_pan_duration': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('pan', EffectParams()).tau),
            'effect_pan_impact': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('pan', EffectParams()).psi),
            'effect_pan_perceptual': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('pan', EffectParams()).epsilon),
        }

        def add_separator(parent_frame, text):
             sep = customtkinter.CTkFrame(parent_frame, height=2, fg_color=self.diamond_white)
             sep.pack(fill="x", padx=5, pady=(15, 2))
             lab = customtkinter.CTkLabel(parent_frame, text=text, font=self.separator_font, text_color=self.gold_accent, anchor="w")
             lab.pack(fill="x", padx=5, pady=(0, 5))

        def add_checkbox(parent, label_text, variable):
             frame = customtkinter.CTkFrame(parent, fg_color="transparent")
             frame.pack(fill="x", pady=4, padx=5)
             checkbox = customtkinter.CTkCheckBox(frame, text=label_text, variable=variable, font=self.label_font,
                                                  text_color=self.diamond_white, hover_color=self.gold_accent,
                                                  fg_color=self.jewel_blue, border_color=self.diamond_white)
             checkbox.pack(side="left", padx=5)

        # --- Populate SHARED Tab ---
        parent = self.shared_tab_frame
        add_separator(parent, "--- Sequencing Constraints ---")
        self._create_single_slider(parent, "Min Clip Length in Edit (s):", self.slider_vars['min_sequence_clip_duration'], 0.2, 3.0, 28, "{:.2f}s")
        self._create_single_slider(parent, "Max Clip Length in Edit (s):", self.slider_vars['max_sequence_clip_duration'], 1.0, 10.0, 90, "{:.1f}s")
        add_separator(parent, "--- Analysis: Detection Settings ---")
        self._create_single_slider(parent, "Min Face Detection Certainty:", self.slider_vars['min_face_confidence'], 0.1, 0.9, 16, "{:.2f}")
        self._create_single_slider(parent, "Mouth 'Open' Threshold:", self.slider_vars['mouth_open_threshold'], 0.01, 0.2, 19, "{:.2f}")
        add_checkbox(parent, "Save Detailed Frame Analysis Data (.json)", self.slider_vars['save_analysis_data'])
        add_separator(parent, "--- Advanced: Normalization Calibration ---")
        self._create_single_slider(parent, "Calibrate: Max Motion Speed (V):", self.slider_vars['norm_max_velocity'], 10.0, 200.0, 38, "{:.0f}")
        self._create_single_slider(parent, "Calibrate: Max Motion Accel (A):", self.slider_vars['norm_max_acceleration'], 20.0, 500.0, 48, "{:.0f}")
        self._create_single_slider(parent, "Calibrate: Max Audio Loudness (RMS):", self.slider_vars['norm_max_rms'], 0.1, 1.0, 18, "{:.1f}")
        self._create_single_slider(parent, "Calibrate: Max Audio Attack (Onset):", self.slider_vars['norm_max_onset'], 1.0, 50.0, 49, "{:.0f}")
        self._create_single_slider(parent, "Calibrate: Max Harmonic Variance:", self.slider_vars['norm_max_chroma_var'], 0.01, 0.3, 29, "{:.2f}")
        self._create_single_slider(parent, "Calibrate: Max Scene Depth Variation:", self.slider_vars['norm_max_depth_variance'], 0.01, 0.5, 49, "{:.2f}")

        # --- Populate GREEDY HEURISTIC Tab ---
        parent = self.greedy_tab_frame
        add_separator(parent, "--- Feature Influence Weights ---")
        self._create_single_slider(parent, "Music Volume Influence:", self.slider_vars['audio_weight'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Body Motion Influence:", self.slider_vars['kinetic_weight'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Motion Sharpness Influence:", self.slider_vars['sharpness_weight'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Camera Movement Influence:", self.slider_vars['camera_motion_weight'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Close-Up Shot Bonus:", self.slider_vars['face_size_weight'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Percussive Hit Influence:", self.slider_vars['percussive_weight'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Scene Depth Influence:", self.slider_vars['depth_weight'], 0.0, 1.0, 20, "{:.2f}")
        add_separator(parent, "--- Clip Identification & Selection ---")
        add_checkbox(parent, "Identify Potential Clips using Heuristic Score Runs", self.slider_vars['use_heuristic_segment_id'])
        self._create_single_slider(parent, "Min Heuristic Score Run Threshold:", self.slider_vars['score_threshold'], 0.1, 0.8, 35, "{:.2f}")
        self._create_single_slider(parent, "Min Potential Heuristic Clip Length (s):", self.slider_vars['min_potential_clip_duration_sec'], 0.2, 2.0, 18, "{:.1f}s")
        self._create_single_slider(parent, "Clip Options Considered per Cut:", self.slider_vars['candidate_pool_size'], 5, 50, 45, "{:.0f}")
        add_separator(parent, "--- Beat Sync & Bonuses ---")
        self._create_single_slider(parent, "Beat Emphasis Strength:", self.slider_vars['beat_boost'], 0.0, 0.5, 25, "{:.2f}")
        self._create_single_slider(parent, "Beat Emphasis Window (s):", self.slider_vars['beat_boost_radius_sec'], 0.0, 0.5, 10, "{:.1f}s")
        self._create_single_slider(parent, "Cut-on-Beat Bonus:", self.slider_vars['beat_sync_bonus'], 0.0, 0.5, 25, "{:.2f}")
        self._create_single_slider(parent, "Music Section Mood Match Bonus:", self.slider_vars['section_match_bonus'], 0.0, 0.5, 25, "{:.2f}")
        add_separator(parent, "--- Sequence Pacing & Variety ---")
        self._create_single_slider(parent, "Pacing Flexibility (+/- %):", self.slider_vars['pacing_variation_factor'], 0.0, 0.7, 14, "{:.1f}")
        self._create_single_slider(parent, "Same Video Repetition Penalty:", self.slider_vars['variety_penalty_source'], 0.0, 0.5, 25, "{:.2f}")
        self._create_single_slider(parent, "Same Shot Type Penalty:", self.slider_vars['variety_penalty_shot'], 0.0, 0.5, 25, "{:.2f}")
        self._create_single_slider(parent, "Same Energy Level Penalty:", self.slider_vars['variety_penalty_intensity'], 0.0, 0.5, 25, "{:.2f}")
        add_separator(parent, "--- Pose Analysis Settings ---")
        self._create_single_slider(parent, "Min Body Pose Certainty:", self.slider_vars['min_pose_confidence'], 0.1, 0.9, 16, "{:.2f}")
        comp_frame_greedy = customtkinter.CTkFrame(parent, fg_color="transparent"); comp_frame_greedy.pack(fill="x", pady=5, padx=5)
        customtkinter.CTkLabel(comp_frame_greedy, text="Pose Model Quality:", width=190, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5, 5))
        radio_fr_greedy = customtkinter.CTkFrame(comp_frame_greedy, fg_color="transparent"); radio_fr_greedy.pack(side="left", padx=5)
        customtkinter.CTkRadioButton(radio_fr_greedy, text="Fast(0)", variable=self.slider_vars['model_complexity'], value=0, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3)
        customtkinter.CTkRadioButton(radio_fr_greedy, text="Balanced(1)", variable=self.slider_vars['model_complexity'], value=1, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3)
        customtkinter.CTkRadioButton(radio_fr_greedy, text="Accurate(2)", variable=self.slider_vars['model_complexity'], value=2, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3)

        # --- Populate PHYSICS PARETO MC Tab ---
        parent = self.physics_tab_frame
        add_separator(parent, "--- Clip-Audio Fit Weights ---")
        self._create_single_slider(parent, "Match: Motion Speed to Beat (w_v):", self.slider_vars['fit_weight_velocity'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Match: Motion Change to Energy (w_a):", self.slider_vars['fit_weight_acceleration'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Match: Video Mood to Music Mood (w_m):", self.slider_vars['fit_weight_mood'], 0.0, 1.0, 20, "{:.2f}")
        self._create_single_slider(parent, "Fit Sensitivity (Sigmoid k):", self.slider_vars['fit_sigmoid_steepness'], 0.1, 5.0, 49, "{:.1f}")
        add_separator(parent, "--- Pareto Objective Priorities ---")
        self._create_single_slider(parent, "Priority: Cut Rhythm/Timing (w_r):", self.slider_vars['objective_weight_rhythm'], 0.0, 2.0, 20, "{:.1f}")
        self._create_single_slider(parent, "Priority: Mood Consistency (w_m):", self.slider_vars['objective_weight_mood'], 0.0, 2.0, 20, "{:.1f}")
        self._create_single_slider(parent, "Priority: Smooth Transitions (w_c):", self.slider_vars['objective_weight_continuity'], 0.0, 2.0, 20, "{:.1f}")
        self._create_single_slider(parent, "Priority: Scene/Pacing Variety (w_v):", self.slider_vars['objective_weight_variety'], 0.0, 2.0, 20, "{:.1f}")
        self._create_single_slider(parent, "Priority: Effect Efficiency (w_e):", self.slider_vars['objective_weight_efficiency'], 0.0, 2.0, 20, "{:.1f}")
        add_separator(parent, "--- Sequence Evaluation ---")
        self._create_single_slider(parent, "Sequence Search Depth (MC Iterations):", self.slider_vars['mc_iterations'], 100, 5000, 490, "{:d}") # Increased range
        self._create_single_slider(parent, "Mood Matching Tolerance (σm):", self.slider_vars['mood_similarity_variance'], 0.05, 0.5, 18, "{:.2f}")
        self._create_single_slider(parent, "Transition Smoothness: Depth Weight (kd):", self.slider_vars['continuity_depth_weight'], 0.0, 1.0, 20, "{:.1f}")
        self._create_single_slider(parent, "Source Repetition Penalty (λ):", self.slider_vars['variety_repetition_penalty'], 0.0, 1.0, 20, "{:.1f}") # Renamed for clarity
        add_separator(parent, "--- Effect Tuning (Configures Rendering) ---")
        # Note: These sliders modify vars that will be used to create RenderConfig later
        self._create_single_slider(parent, "Fade Duration (τ, s):", self.slider_vars['effect_fade_duration'], 0.05, 1.0, 19, "{:.2f}s")
        self._create_single_slider(parent, "Zoom Duration (τ, s):", self.slider_vars['effect_zoom_duration'], 0.1, 2.0, 19, "{:.1f}s")
        self._create_single_slider(parent, "Zoom Max Intensity (ψ, factor):", self.slider_vars['effect_zoom_impact'], 0.05, 1.0, 19, "{:.2f}")
        self._create_single_slider(parent, "Zoom Visual Gain (ε):", self.slider_vars['effect_zoom_perceptual'], 0.0, 1.0, 20, "{:.1f}")
        self._create_single_slider(parent, "Pan Duration (τ, s):", self.slider_vars['effect_pan_duration'], 0.1, 2.0, 19, "{:.1f}s")
        self._create_single_slider(parent, "Pan Max Distance (ψ, % width):", self.slider_vars['effect_pan_impact'], 0.01, 0.5, 49, "{:.2f}")
        self._create_single_slider(parent, "Pan Visual Gain (ε):", self.slider_vars['effect_pan_perceptual'], 0.0, 1.0, 20, "{:.1f}")

    def _create_single_slider(self, parent, label_text, variable, from_val, to_val, steps, format_str="{:.2f}"):
        row = customtkinter.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", pady=4, padx=5)
        customtkinter.CTkLabel(row, text=label_text, width=300, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5, 5))
        val_lab = customtkinter.CTkLabel(row, text=format_str.format(variable.get()), width=70, anchor="e", font=self.button_font, text_color=self.gold_accent)
        val_lab.pack(side="right", padx=(5, 5))
        slider = customtkinter.CTkSlider(
             row, variable=variable, from_=from_val, to=to_val, number_of_steps=steps,
             command=lambda v, lbl=val_lab, fmt=format_str, tk_var=variable:
                 lbl.configure(text=fmt.format(int(v) if isinstance(tk_var, tkinter.IntVar) else v)),
             progress_color=self.gold_accent, button_color=self.diamond_white, button_hover_color=self.gold_accent, fg_color=self.jewel_blue)
        slider.pack(side="left", fill="x", expand=True, padx=5)

    # --- File Handling Methods (Logging Enhanced) ---
    def _select_beat_track(self):
        if self.is_processing: return
        filetypes = (("Audio/Video files", "*.wav *.mp3 *.aac *.flac *.ogg *.mp4 *.mov *.avi *.mkv"),("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select Master Audio Track Source", filetypes=filetypes)
        if filepath:
            try:
                # Basic validation: check existence and if it's a file
                if os.path.isfile(filepath):
                    self.beat_track_path = filepath
                    self.beat_track_label.configure(text=os.path.basename(filepath))
                    logger.info(f"Master audio source selected: {filepath}")
                    self.status_label.configure(text=f"Master: {os.path.basename(filepath)}", text_color=self.diamond_white)
                else:
                    self.beat_track_path = None
                    self.beat_track_label.configure(text="Selection is not a file.")
                    logger.warning(f"Selected path is not a file: {filepath}")
                    tk_write(f"The selected path does not point to a valid file:\n{filepath}", parent=self, level="warning")
            except Exception as e:
                self.beat_track_path = None
                self.beat_track_label.configure(text="Error checking file path.")
                logger.error(f"Could not verify selected path '{filepath}': {e}", exc_info=True)
                tk_write(f"Could not verify the selected file path:\n{e}", parent=self, level="error")

    def _handle_drop(self, event):
        if self.is_processing: return 'break' # Indicate event handled, do nothing
        try:
            # Attempt to parse dropped data (can be complex with spaces/special chars)
            # Using shlex can sometimes help with quoted paths
            try:
                 raw_paths = shlex.split(event.data.strip('{}'))
            except ValueError: # Fallback for simple splitting if shlex fails
                 raw_paths = self.tk.splitlist(event.data.strip('{}'))

            filepaths = [p.strip() for p in raw_paths if p.strip()]
            video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
            count = 0
            current_files_set = set(self.video_files)
            skipped_non_video = 0
            skipped_duplicates = 0

            for fp in filepaths:
                fp_clean = fp.strip("'\" ") # Clean extra quotes/spaces
                try:
                    # Check if it's a file and has a video extension
                    if fp_clean and os.path.isfile(fp_clean):
                        if fp_clean.lower().endswith(video_extensions):
                            if fp_clean not in current_files_set:
                                self.video_files.append(fp_clean)
                                self.video_listbox.insert(END, os.path.basename(fp_clean))
                                current_files_set.add(fp_clean)
                                count += 1
                                logger.debug(f"Added dropped video file: {fp_clean}")
                            else:
                                skipped_duplicates += 1
                                logger.debug(f"Skipped duplicate dropped file: {fp_clean}")
                        else:
                            skipped_non_video += 1
                            logger.debug(f"Skipped dropped item (not a video file extension): {fp_clean}")
                    elif fp_clean:
                        # Could be a folder or invalid path
                        logger.debug(f"Skipped dropped item (not a valid file path): {fp_clean}")
                except Exception as file_check_err:
                     logger.warning(f"Error checking dropped file path '{fp_clean}': {file_check_err}")

            status_parts = []
            if count > 0: status_parts.append(f"Added {count} video(s)")
            if skipped_non_video > 0: status_parts.append(f"skipped {skipped_non_video} non-video")
            if skipped_duplicates > 0: status_parts.append(f"skipped {skipped_duplicates} duplicate(s)")

            if status_parts:
                self.status_label.configure(text=f"Drop: {', '.join(status_parts)}.")
                logger.info(f"Processed drop event: {', '.join(status_parts)}.")
            else:
                self.status_label.configure(text="Drop: No valid new videos found.")
                logger.info("Processed drop event: No valid new videos found.")

        except Exception as e:
             logger.error(f"Error handling drop: {e}\nRaw Drop Data: {event.data}", exc_info=True)
             tk_write(f"An error occurred processing the dropped files:\n{e}", parent=self, level="warning")
        return event.action # Required by TkinterDnD

    def _add_videos_manual(self):
        if self.is_processing: return
        filetypes = (("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*"))
        filepaths = filedialog.askopenfilename(title="Select Source Video Ingredients", filetypes=filetypes, multiple=True)
        count = 0; current_files_set = set(self.video_files)
        if filepaths:
            for fp in filepaths:
                fp_clean = fp.strip("'\" ")
                try:
                    if fp_clean and os.path.isfile(fp_clean):
                         # Check extension again for robustness
                         video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
                         if fp_clean.lower().endswith(video_extensions):
                             if fp_clean not in current_files_set:
                                 self.video_files.append(fp_clean)
                                 self.video_listbox.insert(END, os.path.basename(fp_clean))
                                 current_files_set.add(fp_clean); count += 1
                                 logger.debug(f"Added selected file: {fp_clean}")
                             else:
                                 logger.debug(f"Skipped duplicate selected file: {fp_clean}")
                         else:
                              logger.warning(f"Selected file '{fp_clean}' is not a recognized video type. Skipping.")
                    elif fp_clean:
                         logger.warning(f"Selected item is not a valid file: {fp_clean}")
                except Exception as e:
                    logger.warning(f"Could not add selected file '{fp_clean}': {e}")

            if count > 0:
                self.status_label.configure(text=f"Added {count} video ingredient(s).")
                logger.info(f"Added {count} video(s) via selection dialog.")

    def _remove_selected_videos(self):
         if self.is_processing: return
         selected_indices = self.video_listbox.curselection()
         if not selected_indices:
             self.status_label.configure(text="Select videos from the list to remove.", text_color="yellow")
             return

         indices_to_remove = sorted(list(selected_indices), reverse=True)
         removed_count = 0
         for i in indices_to_remove:
             if 0 <= i < len(self.video_files):
                 try:
                     removed_path = self.video_files.pop(i)
                     self.video_listbox.delete(i); removed_count += 1
                     logger.debug(f"Removed video: {removed_path}")
                 except Exception as e:
                      logger.error(f"Error removing item at index {i}: {e}")
         if removed_count > 0:
             self.status_label.configure(text=f"Removed {removed_count} ingredient(s).", text_color=self.diamond_white)
             logger.info(f"Removed {removed_count} video(s).")

    def _clear_video_list(self):
        if self.is_processing or not self.video_files: return
        if messagebox.askyesno("Confirm Clear", "Discard all current video ingredients?", parent=self):
            self.video_files.clear(); self.video_listbox.delete(0, END)
            self.status_label.configure(text="Ingredient list cleared.", text_color=self.diamond_white)
            logger.info("Cleared all videos from the list.")

    # --- Configuration Management ---
    def _get_analysis_config(self) -> AnalysisConfig:
        """Gathers ANALYSIS settings from UI into AnalysisConfig dataclass."""
        logger.debug("Gathering analysis configuration from UI...")
        # Create dict from slider/checkbox/radio vars first
        config_dict = {key: var.get() for key, var in self.slider_vars.items()}
        # Add mode explicitly
        config_dict["sequencing_mode"] = self.mode_var.get()
        # Add resolution (currently fixed, could be UI elements)
        config_dict["resolution_height"] = 256
        config_dict["resolution_width"] = 256

        # Populate dataclass, handling potential type mismatches or missing keys gracefully
        try:
            # Filter dict to only include keys present in AnalysisConfig annotations
            valid_keys = {f.name for f in AnalysisConfig.__dataclass_fields__.values()}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

            # Ensure specific types are correct (e.g., int for pool size)
            for key, field_type in AnalysisConfig.__annotations__.items():
                if key in filtered_dict:
                    current_val = filtered_dict[key]
                    try:
                        if field_type == int and not isinstance(current_val, int):
                            filtered_dict[key] = int(round(current_val)) # Round float from slider before int conversion
                        elif field_type == float and not isinstance(current_val, float):
                             filtered_dict[key] = float(current_val)
                        elif field_type == bool and not isinstance(current_val, bool):
                             # BooleanVar.get() returns 0/1, convert if necessary
                             filtered_dict[key] = bool(current_val)
                        # Handle Tuple type if needed (e.g., intensity_thresholds - not a slider here currently)
                        # elif field_type == Tuple[float, float] and isinstance(current_val, list): ...
                    except (ValueError, TypeError):
                           logger.warning(f"Invalid value type for {key}: '{current_val}' (expected {field_type}). Using default.")
                           # Remove invalid entry to use dataclass default
                           if key in filtered_dict: del filtered_dict[key]


            cfg = AnalysisConfig(**filtered_dict)
            logger.debug(f"Generated AnalysisConfig: {cfg}")
            return cfg
        except Exception as e:
            logger.error(f"Failed to create AnalysisConfig from UI values: {e}", exc_info=True)
            tk_write("Error reading configuration from UI. Using defaults.", parent=self, level="error")
            return AnalysisConfig() # Return default config on error

    def _get_render_config(self) -> RenderConfig:
        """Gathers RENDERING settings into RenderConfig dataclass."""
        logger.debug("Gathering render configuration...")
        # Get analysis config to copy shared normalization values
        analysis_cfg = self.analysis_config if self.analysis_config else self._get_analysis_config()

        # Create effect params dictionary from slider values
        effect_settings = {
            "cut": EffectParams(type="cut"), # Cut always exists
            "fade": EffectParams(type="fade", tau=self.slider_vars['effect_fade_duration'].get(),
                                 psi=0.1, epsilon=0.2), # Keep psi/epsilon defaults for fade for now
            "zoom": EffectParams(type="zoom", tau=self.slider_vars['effect_zoom_duration'].get(),
                                 psi=self.slider_vars['effect_zoom_impact'].get(),
                                 epsilon=self.slider_vars['effect_zoom_perceptual'].get()),
            "pan": EffectParams(type="pan", tau=self.slider_vars['effect_pan_duration'].get(),
                                psi=self.slider_vars['effect_pan_impact'].get(),
                                epsilon=self.slider_vars['effect_pan_perceptual'].get()),
        }

        # Populate RenderConfig dataclass
        try:
            # Determine thread count (e.g., half of CPU cores, min 1, max maybe 8?)
            cpu_cores = os.cpu_count() or 1
            render_threads = max(1, min(cpu_cores // 2, 8))

            cfg = RenderConfig(
                # Copy relevant norm values from analysis config
                norm_max_velocity=analysis_cfg.norm_max_velocity,
                norm_max_acceleration=analysis_cfg.norm_max_acceleration,
                # Set effect settings from UI
                effect_settings=effect_settings,
                # --- Other FFmpeg settings (could be UI elements too) ---
                video_codec='libx264', # Common default
                preset='medium',       # Balance between speed and quality
                crf=23,                # Constant Rate Factor (lower is better quality, larger file)
                audio_codec='aac',     # Common default
                audio_bitrate='192k',
                threads=render_threads,# Use calculated threads
                ffmpeg_loglevel='warning' # FFmpeg's internal log level
            )
            logger.debug(f"Generated RenderConfig: {cfg}")
            self.render_config = cfg # Store it
            return cfg
        except Exception as e:
            logger.error(f"Failed to create RenderConfig: {e}", exc_info=True)
            tk_write("Error reading render configuration. Using defaults.", parent=self, level="error")
            self.render_config = RenderConfig() # Store default on error
            return self.render_config


    def _set_ui_processing_state(self, processing: bool):
        """Disables/Enables UI elements during processing."""
        self.is_processing = processing
        state = "disabled" if processing else "normal"
        widgets_to_toggle = [
            self.beat_track_button, self.add_button, self.remove_button,
            self.clear_button, self.run_button
        ]
        # Find mode selector robustly
        try:
             # Assumes mode_selector is the last widget in mode_inner_frame
             bottom_frame = self.grid_slaves(row=2, column=0)[0]
             mode_inner_frame = bottom_frame.grid_slaves(row=0, column=0)[0]
             mode_selector = mode_inner_frame.winfo_children()[-1]
             if isinstance(mode_selector, customtkinter.CTkSegmentedButton):
                  widgets_to_toggle.append(mode_selector)
             else: logger.warning("Could not find mode selector widget correctly.")
        except (IndexError, AttributeError) as e: logger.warning(f"Error finding mode selector: {e}")

        for widget in widgets_to_toggle:
            if widget and hasattr(widget, 'configure'):
                 try: widget.configure(state=state)
                 except Exception: pass # Ignore if widget doesn't support state

        # Toggle sliders/checkboxes/radios in tabs
        try:
             for tab_name in ["Shared", "Greedy Heuristic", "Physics MC"]:
                 tab = self.tab_view.tab(tab_name)
                 if tab and hasattr(tab, 'winfo_children') and tab.winfo_children():
                     # Assume the first child is the scrollable frame
                     scroll_frame = tab.winfo_children()[0]
                     if scroll_frame and hasattr(scroll_frame, 'winfo_children'):
                         # Iterate through the rows (frames containing label+widget)
                         for item_frame in scroll_frame.winfo_children():
                             if isinstance(item_frame, customtkinter.CTkFrame) and hasattr(item_frame, 'winfo_children'):
                                 # Find sliders, checkboxes, radiobuttons within the row frame
                                 for widget in item_frame.winfo_children():
                                     widget_type = type(widget)
                                     if widget_type in (customtkinter.CTkSlider, customtkinter.CTkCheckBox, customtkinter.CTkRadioButton, customtkinter.CTkFrame):
                                          # Handle nested radio button frames
                                          if widget_type == customtkinter.CTkFrame:
                                              for radio in widget.winfo_children():
                                                   if isinstance(radio, customtkinter.CTkRadioButton):
                                                       try: radio.configure(state=state)
                                                       except Exception: pass
                                          # Handle sliders and checkboxes
                                          elif hasattr(widget, 'configure') and 'state' in widget.configure():
                                               try: widget.configure(state=state)
                                               except Exception: pass
        except Exception as e: logger.warning(f"Error toggling config UI state in tabs: {e}", exc_info=False)

        self.run_button.configure(text="Chef is Cooking..." if processing else "4. Compose Video Remix")
        self.update_idletasks() # Force UI update

    # --- Processing Workflow (Logging Enhanced) ---
    def _start_processing(self):
        if self.is_processing:
            logger.warning("Processing already in progress. Ignoring start request.")
            return

        # --- Validation Checks ---
        if not self.beat_track_path or not os.path.isfile(self.beat_track_path):
            logger.warning("Master audio track path is invalid or not set.")
            tk_write("Chef needs a valid master audio track (the base)!\nPlease select the file.", parent=self, level="warning")
            return
        if not self.video_files:
            logger.warning("No source video files selected.")
            tk_write("Chef needs video ingredients to cook!", parent=self, level="warning")
            return
        # Further check if video files actually exist and update list if needed
        valid_video_files = [f for f in self.video_files if os.path.isfile(f)]
        invalid_count = len(self.video_files) - len(valid_video_files)
        if invalid_count > 0:
             logger.warning(f"{invalid_count} video path(s) in the list are invalid. Proceeding with {len(valid_video_files)} valid file(s).")
             # Ask user to confirm? For now, just proceed with valid ones.
             self.video_files = valid_video_files # Update internal list
             # Update listbox UI (more complex, requires tracking original indices or rebuilding)
             # Simple approach: clear and re-add valid ones
             self.video_listbox.delete(0, END)
             for vf in self.video_files: self.video_listbox.insert(END, os.path.basename(vf))
             tk_write(f"{invalid_count} invalid video path(s) were removed from the list.", parent=self, level="warning")
             if not self.video_files: # Check if list became empty
                 tk_write("No valid video ingredients remaining after check!", parent=self, level="error")
                 return

        # --- Get Config & Start Processing ---
        logger.info("Starting processing workflow...")
        self.analysis_config = self._get_analysis_config() # Get AnalysisConfig
        self.render_config = self._get_render_config()   # Get RenderConfig
        if not self.analysis_config or not self.render_config:
            logger.error("Failed to gather configuration from UI. Aborting.")
            return

        self._set_ui_processing_state(True)
        self.status_label.configure(text="Chef is prepping...", text_color=self.diamond_white)

        # Reset state variables for this run
        self.master_audio_data = None; self.all_potential_clips = []
        self.analysis_futures = []; self.futures_map = {}
        self.total_tasks = 0; self.completed_tasks = 0
        self.shutdown_executor() # Ensure clean state before starting new pool

        logger.info("Starting master audio analysis thread...")
        self.status_label.configure(text="Analyzing master audio (the base)...")
        # Run audio analysis in a separate thread to keep UI responsive
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._analyze_master_audio, name="AudioAnalysisThread", daemon=True)
            self.processing_thread.start()
        else:
            logger.warning("Processing thread already active (_start_processing). Aborting.")
            self._set_ui_processing_state(False)

    def _analyze_master_audio(self):
        """Analyzes master audio in a background thread. On success, triggers parallel video analysis."""
        try:
            audio_analyzer = PegasusAudioUtils()
            # Create a unique temp file name for master audio extraction if needed
            timestamp = time.strftime("%Y%m%d%H%M%S")
            temp_audio_for_master = os.path.join(self.output_dir, f"temp_master_audio_{timestamp}.wav")
            audio_file_to_analyze = self.beat_track_path
            needs_cleanup = False

            # Check if master source is video and needs extraction
            video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv')
            if self.beat_track_path.lower().endswith(video_extensions):
                self.after(0, lambda: self.status_label.configure(text="Extracting audio from master track..."))
                logger.info("Master track is video, extracting audio...")
                extracted_path = audio_analyzer.extract_audio(self.beat_track_path, temp_audio_for_master)
                if not extracted_path:
                    raise RuntimeError(f"Failed audio extraction from master track: {self.beat_track_path}")
                audio_file_to_analyze = extracted_path; needs_cleanup = True
                logger.info(f"Using extracted audio: {audio_file_to_analyze}")
            elif not os.path.isfile(audio_file_to_analyze):
                 raise FileNotFoundError(f"Master audio file path is invalid: {audio_file_to_analyze}")

            # Run audio analysis (using the AnalysisConfig gathered earlier)
            self.after(0, lambda: self.status_label.configure(text="Analyzing master audio features..."))
            logger.info(f"Starting Librosa audio analysis on: {os.path.basename(audio_file_to_analyze)}")
            # Pass the AnalysisConfig object stored in self.analysis_config
            self.master_audio_data = audio_analyzer.analyze_audio(audio_file_to_analyze, self.analysis_config)

            if not self.master_audio_data:
                raise RuntimeError("Audio analysis failed to return data.")

            logger.info("Master audio analysis successful.")
            # --- Proceed to parallel video analysis ---
            # Ensure we switch back to the main thread to start the next phase (which uses 'after')
            self.after(0, self._start_parallel_video_analysis) # Call helper to start next phase

        except Exception as e:
            error_msg = f"Error processing master audio: {type(e).__name__}"
            logger.error(f"Error processing master audio: {e}", exc_info=True)
            # Use 'after' to schedule UI updates from this background thread
            self.after(0, lambda: [
                self.status_label.configure(text=f"Error: {error_msg}", text_color="orange"),
                tk_write(f"Failed to process the master audio track.\n\n{e}", parent=self, level="error"),
                self._set_ui_processing_state(False) # Re-enable UI on error
            ])
        finally:
            # Cleanup extracted audio file if created
            if 'needs_cleanup' in locals() and needs_cleanup and os.path.exists(audio_file_to_analyze):
                try: os.remove(audio_file_to_analyze); logger.debug("Cleaned up temp master audio.")
                except OSError as del_err: logger.warning(f"Failed to remove temp audio {audio_file_to_analyze}: {del_err}")

    def _start_parallel_video_analysis(self):
        """Helper function called from main thread to initiate parallel video analysis."""
        self.status_label.configure(text=f"Analyzing {len(self.video_files)} video ingredients...")
        # Run the parallel setup and monitoring in a new background thread
        # to avoid blocking the UI while submitting jobs and checking status.
        analysis_pool_thread = threading.Thread(target=self._run_parallel_video_analysis_pool, name="VideoAnalysisPoolMgr", daemon=True)
        analysis_pool_thread.start()

    def _run_parallel_video_analysis_pool(self):
        """Sets up and monitors the ProcessPoolExecutor for video analysis."""
        cpu_cores = os.cpu_count() or 1
        # Adjust worker count: less likely to hit memory limits with fewer, concurrent workers
        max_workers = max(1, min(cpu_cores // 2, 6)) # Example: Half cores, max 6
        logger.info(f"Starting parallel video analysis using up to {max_workers} worker processes.")

        self.analysis_futures = []; self.futures_map = {}
        self.total_tasks = len(self.video_files); self.completed_tasks = 0

        # --- Check if data is pickleable before submitting ---
        try:
             import pickle
             # Test pickling of the objects that will be sent to workers
             pickle.dumps(self.master_audio_data)
             pickle.dumps(self.analysis_config)
             logger.debug("Audio data and analysis config appear pickleable.")
        except Exception as pickle_e:
             err_msg = f"Internal Error: Cannot send data to worker processes (pickling failed).\n\n{pickle_e}"
             logger.critical(f"Pickle Error preventing parallel processing: {pickle_e}", exc_info=True)
             # Use 'after' to update UI from this thread
             self.after(0, lambda: [ tk_write(err_msg, parent=self, level="error"), self._set_ui_processing_state(False)])
             return # Stop if data can't be sent

        # --- Submit Jobs ---
        self.shutdown_executor() # Ensure clean slate
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        submitted_count = 0
        submission_errors = 0

        for vid_path in self.video_files:
            if not os.path.exists(vid_path): # Double check path exists before submitting
                 logger.error(f"Skipping submission for non-existent file: {vid_path}")
                 submission_errors += 1
                 continue
            try:
                # Pass AnalysisConfig object to worker
                future = self.executor.submit(process_single_video, vid_path, self.master_audio_data, self.analysis_config, self.analysis_subdir)
                self.futures_map[future] = vid_path # Map future back to path for logging
                self.analysis_futures.append(future)
                submitted_count += 1
            except Exception as submit_e:
                 # Catch errors during submission itself (rare, might indicate pool issues)
                 logger.error(f"ERROR submitting job for {os.path.basename(vid_path)}: {submit_e}", exc_info=True)
                 submission_errors += 1

        # Adjust total tasks based on submission success/failures
        self.total_tasks = submitted_count

        if not self.analysis_futures:
             logger.error("No analysis jobs were successfully submitted. Aborting parallel analysis.")
             final_msg = "Error: Failed to start analysis jobs."
             if submission_errors > 0: final_msg += f" ({submission_errors} submission errors)"
             self.after(0, lambda: [ self.status_label.configure(text=final_msg, text_color="red"), self._set_ui_processing_state(False)])
             self.shutdown_executor()
             return

        logger.info(f"Submitted {submitted_count} video analysis jobs.")
        # --- Monitor Progress ---
        # Initial status update via 'after'
        self.after(0, lambda: self.status_label.configure(text=f"Analyzing ingredients... 0/{self.total_tasks} (0.0%)"))
        # Start the polling check also via 'after' to run it on the main thread
        self.after(1000, self._check_analysis_status)


    def _check_analysis_status(self):
        """Checks parallel analysis status (called via 'after'). Proceeds when done."""
        # Check if processing was cancelled or finished externally
        if not self.is_processing or not hasattr(self, 'analysis_futures') or not self.analysis_futures:
            logger.debug("Stopping analysis status check (not processing or no futures).")
            self.shutdown_executor() # Ensure pool is cleaned up if check stops unexpectedly
            return

        try:
            # Count completed futures (successfully or with exception)
            done_futures = [f for f in self.analysis_futures if f.done()]
            done_count = len(done_futures)
            self.completed_tasks = done_count

            if self.total_tasks > 0:
                progress = (done_count / self.total_tasks) * 100
                # Update status label on the main thread
                self.status_label.configure(text=f"Analyzing ingredients... {done_count}/{self.total_tasks} ({progress:.1f}%)")
            else:
                 self.status_label.configure(text="Waiting for analysis results...") # Handle case of 0 tasks

            # --- Check if All Tasks Are Done ---
            if done_count == self.total_tasks:
                logger.info("--- All Video Analyses Finished ---")
                logger.info("Shutting down analysis process pool...")
                self.shutdown_executor() # Initiate shutdown (non-blocking)

                logger.info("Collecting analysis results...")
                self.all_potential_clips = []
                success_count = 0; fail_count = 0
                failed_videos = []

                for future in self.analysis_futures: # Iterate through original list
                    vid_path_for_log = self.futures_map.get(future, "Unknown Video")
                    try:
                        # Get result from the future (this can raise exceptions from the worker)
                        video_path, status, potential_clips_result = future.result(timeout=5) # Short timeout for retrieval
                        logger.info(f"Result for {os.path.basename(video_path)}: {status}")
                        # Check status string and if clips were returned
                        if "Analysis OK" in status and potential_clips_result:
                            # Filter for actual ClipSegment objects (worker might return empty list)
                            valid_clips = [clip for clip in potential_clips_result if isinstance(clip, ClipSegment)]
                            self.all_potential_clips.extend(valid_clips)
                            if valid_clips: success_count += 1
                            elif potential_clips_result: # OK status but no valid clips after check
                                logger.warning(f"Analysis OK for {os.path.basename(video_path)} but yielded no valid ClipSegment objects.")
                                fail_count += 1 # Count as failure if no usable clips produced
                                failed_videos.append(os.path.basename(video_path))
                        elif "Analysis OK" not in status: # Explicit failure status from worker
                             fail_count += 1
                             failed_videos.append(os.path.basename(video_path))
                             logger.error(f"Analysis explicitly failed for {os.path.basename(video_path)} with status: {status}")

                    except concurrent.futures.TimeoutError:
                         logger.error(f"Timeout retrieving analysis result for {os.path.basename(vid_path_for_log)}.")
                         fail_count += 1
                         failed_videos.append(os.path.basename(vid_path_for_log))
                    except Exception as e:
                         # Exception raised *within* the worker process
                         logger.error(f"Error raised by worker analyzing {os.path.basename(vid_path_for_log)}: {type(e).__name__} - {e}", exc_info=False) # Log summary, full trace is in worker log
                         fail_count += 1
                         failed_videos.append(os.path.basename(vid_path_for_log))

                logger.info(f"Collected {len(self.all_potential_clips)} potential clips from {success_count} successfully analyzed source(s).")

                # --- Handle Analysis Failures / Lack of Clips ---
                if fail_count > 0:
                    fail_msg = f"{fail_count} video(s) failed analysis or result retrieval. Check logs for details."
                    if failed_videos: fail_msg += f"\nFailed: {', '.join(failed_videos[:5])}{'...' if len(failed_videos) > 5 else ''}"
                    tk_write(fail_msg, parent=self, level="warning")

                if not self.all_potential_clips:
                    # Use 'after' as we are still in the callback from the previous 'after'
                    self.after(0, lambda: [
                         self.status_label.configure(text="Error: No usable clip ingredients found.", text_color="orange"),
                         tk_write("Analysis finished, but no usable clips were identified. Try adjusting analysis parameters or video sources.", parent=self, level="error"),
                         self._set_ui_processing_state(False) # Re-enable UI
                    ])
                    logger.error("Aborting workflow: No potential clips identified after analysis.")
                    return # Stop the process here

                # --- Proceed to Sequence Building (in a new thread) ---
                self.status_label.configure(text="Chef is composing the final dish (sequence)...")
                logger.info("Starting sequence building thread...")
                # Ensure previous thread is finished before starting new one
                if self.processing_thread and self.processing_thread.is_alive():
                     logger.warning("Waiting for previous processing thread to finish before starting sequence build...")
                     self.processing_thread.join(timeout=5.0) # Wait briefly

                self.processing_thread = threading.Thread(target=self._build_final_sequence_and_video, name="SequenceBuildThread", daemon=True)
                self.processing_thread.start()
                # Status check loop ends here, sequence building takes over

            else:
                # Schedule the next check if not all tasks are done
                self.after(1000, self._check_analysis_status) # Check again in 1 second

        except Exception as poll_err:
             logger.error(f"Error during analysis status check/result collection: {poll_err}", exc_info=True)
             self.after(0, lambda: [ self.status_label.configure(text="Error checking analysis status.", text_color="red"), self._set_ui_processing_state(False)])
             self.shutdown_executor() # Try to shutdown pool if polling fails badly

    def _build_final_sequence_and_video(self):
        """Builds sequence and renders video in a background thread."""
        try:
            # --- Pre-checks ---
            if not self.analysis_config or not self.master_audio_data or not self.render_config:
                 logger.error("Cannot build sequence: Missing analysis config, render config, or master audio data.")
                 raise RuntimeError("Internal state error: Missing configuration or audio data.")
            if not self.all_potential_clips:
                 logger.error("Cannot build sequence: No potential clips available.")
                 raise RuntimeError("Internal state error: No potential clips.")

            selected_mode = self.analysis_config.sequencing_mode
            builder: Optional[Union[SequenceBuilderGreedy, SequenceBuilderPhysicsMC]] = None # Type hint
            logger.info(f"Instantiating sequence builder for mode: {selected_mode}")
            self.after(0, lambda: self.status_label.configure(text=f"Composing sequence ({selected_mode})..."))

            # --- Instantiate Builder ---
            if selected_mode == "Physics Pareto MC":
                # Pass AnalysisConfig (contains objective weights etc.)
                # Physics builder might internally use RenderConfig for effect params if passed,
                # but currently it uses placeholders or AnalysisConfig values as fallback.
                # Pass both configs for future flexibility? For now, only AnalysisConfig needed by builder.
                builder = SequenceBuilderPhysicsMC(self.all_potential_clips, self.master_audio_data, self.analysis_config)
                # Override the builder's default effects with those from RenderConfig
                builder.effects = self.render_config.effect_settings
                logger.debug(f"Physics MC Builder using effects from RenderConfig: {builder.effects}")
            else: # Default to Greedy Heuristic
                # Pass AnalysisConfig
                builder = SequenceBuilderGreedy(self.all_potential_clips, self.master_audio_data, self.analysis_config)

            # --- Build Sequence ---
            logger.info("Building sequence...")
            final_sequence = builder.build_sequence()

            if not final_sequence:
                logger.error(f"Sequence building failed for mode {selected_mode}.")
                # Update UI from this thread using 'after'
                self.after(0, lambda: [
                    self.status_label.configure(text=f"Error: Failed to compose sequence ({selected_mode}).", text_color="orange"),
                    tk_write(f"The Chef could not create a sequence using {selected_mode} mode. Check logs for details.", parent=self, level="error"),
                    self._set_ui_processing_state(False) # Re-enable UI
                ])
                return # Stop processing

            logger.info(f"Sequence built successfully ({len(final_sequence)} clips). Preparing for render.")
            self.after(0, lambda: self.status_label.configure(text=f"Preparing final render ({len(final_sequence)} clips)..."))

            # --- Prepare for Render ---
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            mode_tag = "Greedy" if selected_mode == "Greedy Heuristic" else "PhysicsMC"
            output_video_path = os.path.join(self.render_subdir, f"videous_chef_{mode_tag}_{timestamp}.mp4")

            # --- Get Master Audio Path for Render (May require re-extraction) ---
            master_audio_path_for_render = self.beat_track_path
            temp_audio_render = None
            audio_util = PegasusAudioUtils() # Need instance for extraction

            # Check if original master was video and needs re-extraction
            video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv')
            if self.beat_track_path.lower().endswith(video_extensions):
                self.after(0, lambda: self.status_label.configure(text="Extracting audio for final render..."))
                logger.info("Extracting master audio again for final render...")
                temp_audio_render = os.path.join(self.render_subdir, f"master_audio_render_{timestamp}.wav")
                # Use a different path for render extraction
                extracted = audio_util.extract_audio(self.beat_track_path, temp_audio_render)
                if not extracted or not os.path.exists(extracted):
                     raise RuntimeError("Failed to extract audio for final render.")
                master_audio_path_for_render = extracted
                logger.info(f"Using re-extracted audio for render: {master_audio_path_for_render}")
            elif not os.path.isfile(master_audio_path_for_render):
                 # If original was audio, but path is now invalid
                 raise FileNotFoundError(f"Master audio path for render not found or invalid: {master_audio_path_for_render}")

            # --- Build Video ---
            self.after(0, lambda: self.status_label.configure(text=f"Rendering final dish ({selected_mode})..."))
            logger.info(f"Starting final video render to {output_video_path}...")
            # Pass RenderConfig object to the build function
            buildSequenceVideo(final_sequence, output_video_path, master_audio_path_for_render, self.render_config)

            # --- Success ---
            final_msg = f"Success ({mode_tag})! Saved:\n{os.path.basename(output_video_path)}"
            final_color = "light green"
            logger.info(f"Video composition successful: {output_video_path}")
            self.after(0, lambda: self.status_label.configure(text=final_msg, text_color=final_color))
            tk_write(f"Video Remix Composition Successful!\n\nMode: {selected_mode}\nOutput:\n{output_video_path}", parent=self, level="info")

        except Exception as e:
            # --- FIX NameError ---
            # Capture exception details within the except block
            error_type_name = type(e).__name__
            error_message = str(e)
            # --- Modify lambda to use captured details via default arguments ---
            final_msg = "Error during composition or rendering. Check log."
            final_color = "orange"
            logger.critical(f"!!! FATAL ERROR during Sequence Building/Rendering !!!", exc_info=True)
            # Use 'after' for UI update, passing captured error details
            self.after(0, lambda et=error_type_name, em=error_message: [
                 self.status_label.configure(text=final_msg, text_color=final_color),
                 tk_write(f"The Chef encountered a problem during final composition or rendering:\n\n{et}: {em}\n\nPlease check the log file or console.", parent=self, level="error")
            ])
        finally:
            # Cleanup temp render audio if created
            if 'temp_audio_render' in locals() and temp_audio_render and os.path.exists(temp_audio_render):
                try: os.remove(temp_audio_render); logger.debug("Cleaned up temp render audio.")
                except OSError as del_err: logger.warning(f"Failed to remove temp render audio: {del_err}")
            # Ensure UI is re-enabled regardless of success/failure
            self.after(0, self._set_ui_processing_state, False)
            logger.info("Build/Render thread finished.")

    def on_closing(self):
        logger.info("Shutdown requested via window close...")
        if self.is_processing:
            if messagebox.askyesno("Confirm Exit", "Processing is ongoing.\nExiting now might leave unfinished processes or corrupt output.\n\nAre you sure you want to exit?", parent=self):
                 logger.warning("Forcing shutdown during processing.")
                 # Attempt to cancel ongoing threads/processes gracefully
                 self.shutdown_executor() # Request pool shutdown
                 # Consider signaling background threads to stop if possible (e.g., using threading.Event)
                 # For simplicity, we just destroy the window here, but background tasks might linger.
                 self.destroy()
            else:
                 logger.info("Shutdown cancelled by user.")
                 return # Don't close the window
        else:
            # Clean shutdown if not processing
            self.shutdown_executor()
            self.destroy()
        logger.info("Application closing.")


    def shutdown_executor(self):
         """Attempts to gracefully shut down the ProcessPoolExecutor."""
         if hasattr(self, 'executor') and self.executor: # Check if executor exists
             if not getattr(self.executor, '_shutdown', False): # Check if not already shut down
                 logger.info("Shutting down process pool executor...")
                 # Cancel pending futures before shutdown
                 cancelled_count = 0
                 if hasattr(self, 'analysis_futures'):
                     for f in self.analysis_futures:
                         if not f.done():
                             if f.cancel(): # Attempt to cancel
                                 cancelled_count += 1
                 logger.debug(f"Attempted to cancel {cancelled_count} pending future(s).")

                 try:
                     # Non-blocking shutdown with wait=False, request future cancellation
                     # On Python 3.9+, cancel_futures=True is available
                     if sys.version_info >= (3, 9):
                         self.executor.shutdown(wait=False, cancel_futures=True)
                     else:
                         self.executor.shutdown(wait=False) # Older versions rely on pre-cancellation
                     self.executor = None # Clear reference immediately
                     logger.info("Executor shutdown initiated.")
                 except Exception as e:
                      logger.error(f"Error during executor shutdown: {e}", exc_info=True)
             else:
                 logger.debug("Executor shutdown requested, but already shutdown.")
                 self.executor = None # Ensure reference is cleared
         else:
              logger.debug("Executor shutdown requested, but no executor instance exists or it's already cleared.")

# print("DEBUG: Defined VideousApp") # REMOVED Debug Print

# ========================================================================
#                      REQUIREMENTS.TXT Block
# ========================================================================
"""
# requirements.txt for Videous Chef v4.5

# Core UI & Analysis
customtkinter>=5.2.0,<6.0.0           # Check for latest stable 5.x
opencv-python>=4.6.0,<5.0.0           # Check MediaPipe/VidGear compatibility
numpy>=1.21.0,<2.0.0                  # Avoid numpy 2.0 for now
librosa>=0.9.2,<0.11.0                # Stable 0.9.x or 0.10.x recommended
mediapipe>=0.10.0,<0.11.0             # Check compatibility with OpenCV version
tkinterdnd2-universal>=2.1.0         # Use the universal fork
soundfile>=0.11.0,<0.13.0
tqdm>=4.60.0
matplotlib>=3.5.0                      # Primarily for 'Agg' backend
scipy>=1.8.0

# AI / ML Features
torch>=1.12.0                          # Check specific MiDaS requirements
torchaudio>=0.12.0
torchvision>=0.13.0
timm>=0.6.0                            # Specific requirement for MiDaS via torch.hub

# Video Reading/Rendering (VidGear)
vidgear>=0.3.0,<0.4.0                  # Use a recent stable 0.3.x version (Tested with 0.3.3)

# NOTE 1: Ensure FFmpeg (the executable) is installed separately and accessible in your system's PATH!
# Download from: https://ffmpeg.org/download.html

# NOTE 2: On macOS with Apple Silicon (M1/M2), ensure PyTorch is installed correctly
# for MPS support if desired (often requires specific build or nightly). Check PyTorch docs.

# NOTE 3: Consider creating a virtual environment (e.g., venv, conda) to manage these dependencies.
# Example setup:
# python -m venv venv
# source venv/bin/activate  # (or venv\Scripts\activate on Windows)
# pip install -r requirements.txt
"""

# ========================================================================
#                       APPLICATION ENTRY POINT
# ========================================================================
if __name__ == "__main__":
    # print("DEBUG: Entered __main__ block.") # REMOVED Debug Print

    # --- Multiprocessing Setup ---
    multiprocessing.freeze_support()
    # print("DEBUG: freeze_support() called.") # REMOVED Debug Print
    try:
        default_method = multiprocessing.get_start_method(allow_none=True)
        # print(f"DEBUG: Default multiprocessing method: {default_method}") # REMOVED Debug Print
        # --- RE-ENABLE forcing 'spawn' ---
        if default_method != 'spawn':
             multiprocessing.set_start_method('spawn', force=True)
             print(f"INFO: Set multiprocessing start method to 'spawn' (was '{default_method}').")
        else:
             print(f"INFO: Multiprocessing start method already 'spawn'.")
        # print(f"DEBUG: Using multiprocessing start method: {multiprocessing.get_start_method()}.") # REMOVED Debug Print
    except Exception as E:
         print(f"WARNING: Error during multiprocessing setup: {E}. Using default: {multiprocessing.get_start_method()}.")
    # print("DEBUG: Multiprocessing setup done.") # REMOVED Debug Print


    print("--- Videous Chef v4.5 (WriteGear Render) Starting ---")
    # Setup basic console handler for logging BEFORE UI or complex imports if possible
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Capture all levels initially
    root_logger.addHandler(console_handler)
    # print("DEBUG: Basic console logging configured.") # REMOVED Debug Print

    # Optional: Add file handler (Keep this)
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"videous_chef_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to console (INFO+) and file (DEBUG+): {log_file}")
        # print("DEBUG: File logging configured.") # REMOVED Debug Print
    except Exception as log_setup_e:
        logger.error(f"Failed to set up file logging: {log_setup_e}")
        logger.info("Logging to console only.")
        # print("DEBUG: File logging failed.") # REMOVED Debug Print


    try: # Outer try block for startup
        # print("DEBUG: Entering main try block for startup...") # REMOVED Debug Print
        # --- Dependency Checks (using logging) ---
        # --- RE-ENABLE Dependency Checks ---
        logger.info("Checking dependencies...")
        missing_deps = []
        deps = {
            "customtkinter": "customtkinter", "opencv-python": "cv2", "numpy": "numpy",
            "librosa": "librosa", "mediapipe": "mediapipe", "tkinterdnd2": "tkinterdnd2",
            "soundfile": "soundfile", "scipy": "scipy", "torch": "torch",
            "torchaudio": "torchaudio", "torchvision": "torchvision", "timm": "timm",
            "vidgear": "vidgear", "matplotlib": "matplotlib", "tqdm": "tqdm"
        }
        critical_deps = ["cv2", "numpy", "customtkinter", "vidgear", "torch", "mediapipe", "librosa"]

        for pkg_name, mod_name in deps.items():
            logger.debug(f"Checking for {mod_name} ({pkg_name})...")
            try:
                 m = __import__(mod_name)
                 # Optional: Check version?
                 # if hasattr(m, '__version__'): logger.debug(f"  {mod_name} version: {m.__version__}")
            except ImportError as imp_err:
                 logger.error(f"Dependency check FAILED for {mod_name} ({pkg_name}): {imp_err}")
                 missing_deps.append(pkg_name)
                 if mod_name in critical_deps:
                      logger.critical(f"*** Critical dependency '{mod_name}' missing. Exiting. ***")
                      # Show error message box before exiting
                      try:
                         root_err = tkinter.Tk(); root_err.withdraw()
                         messagebox.showerror("Dependency Error", f"Critical dependency missing: {pkg_name}\nPlease install it.\n\n{imp_err}")
                         root_err.destroy()
                      except Exception: pass
                      sys.exit(1)
            except Exception as other_err:
                 logger.error(f"Dependency check FAILED for {mod_name} ({pkg_name}) - Other Error: {other_err}", exc_info=True)
                 missing_deps.append(f"{pkg_name} (Import Error: {type(other_err).__name__})")
                 if mod_name in critical_deps:
                     logger.critical(f"*** Critical dependency '{mod_name}' failed check. Exiting. ***")
                     try:
                         root_err = tkinter.Tk(); root_err.withdraw()
                         messagebox.showerror("Dependency Error", f"Critical dependency failed: {pkg_name}\nError: {other_err}\nCheck logs.")
                         root_err.destroy()
                     except Exception: pass
                     sys.exit(1)

        if missing_deps:
             install_cmd = f"pip install {' '.join(missing_deps)}"
             err_msg = f"Missing/problematic non-critical dependencies: {', '.join(missing_deps)}\nPlease try installing them (e.g., '{install_cmd}')\nAlso ensure FFmpeg executable is in PATH."
             logger.warning(err_msg)
             # Don't exit for non-critical, but warn the user
             try:
                 root_err = tkinter.Tk(); root_err.withdraw()
                 messagebox.showwarning("Dependency Warning", err_msg)
                 root_err.destroy()
             except Exception: pass
             # Continue execution if only non-critical deps are missing
        else:
            logger.info("All critical dependencies checked successfully.")
        # print("DEBUG: Dependency checks skipped.") # REMOVED Debug Print


        # --- Check FFmpeg Executable ---
        # --- RE-ENABLE FFmpeg Check ---
        try:
            logger.debug("Checking for FFmpeg executable...")
            # Use subprocess to run `ffmpeg -version` and check return code
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False, timeout=5)
            if result.returncode != 0 or "ffmpeg version" not in result.stdout.lower():
                 logger.error(f"FFmpeg check failed (Code: {result.returncode}). Stdout: {result.stdout[:100]}..., Stderr: {result.stderr[:100]}...")
                 raise FileNotFoundError("FFmpeg not found or basic version check failed.")
            else:
                 logger.info("FFmpeg executable found and seems operational.")
                 # logger.debug(f"FFmpeg version output (partial): {result.stdout.splitlines()[0]}")
        except FileNotFoundError:
            err_msg = "FFmpeg executable not found in PATH.\n\nVideous Chef requires FFmpeg for audio extraction and video rendering.\nPlease install FFmpeg and ensure it's added to your system's PATH environment variable.\n\nDownload: https://ffmpeg.org/download.html"
            logger.critical(err_msg)
            try:
                root_err = tkinter.Tk(); root_err.withdraw()
                messagebox.showerror("Dependency Error", err_msg)
                root_err.destroy()
            except Exception: pass
            sys.exit(1)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg check timed out. Assuming it's not working correctly.")
            # Handle timeout as potential error? Or just warn? Warn for now.
            try:
                root_err = tkinter.Tk(); root_err.withdraw()
                messagebox.showwarning("Dependency Warning", "Checking FFmpeg timed out. Ensure it's installed correctly and working.")
                root_err.destroy()
            except Exception: pass
        except Exception as ffmpeg_e:
             logger.error(f"Error during FFmpeg check: {ffmpeg_e}", exc_info=True)
             # Warn but don't necessarily exit? Depends if core functionality relies solely on it. Audio extraction does.
             try:
                 root_err = tkinter.Tk(); root_err.withdraw()
                 messagebox.showwarning("Dependency Warning", f"Error checking FFmpeg: {ffmpeg_e}\nFunctionality might be limited.")
                 root_err.destroy()
             except Exception: pass
        # print("DEBUG: FFmpeg check skipped.") # REMOVED Debug Print

        # --- PyTorch Backend Check ---
        # --- RE-ENABLE PyTorch Check ---
        logger.info(f"PyTorch version: {torch.__version__}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): logger.info("PyTorch MPS backend available (Apple Silicon).")
        elif torch.cuda.is_available(): logger.info(f"PyTorch CUDA backend available. Devices: {torch.cuda.device_count()}")
        else: logger.info("PyTorch CPU backend.")
        # print("DEBUG: PyTorch check skipped.") # REMOVED Debug Print


        # --- Run App ---
        logger.info("Initializing Application UI...")
        # print("DEBUG: Initializing VideousApp()...") # REMOVED Debug Print
        app = VideousApp()
        # print("DEBUG: VideousApp() initialized.") # REMOVED Debug Print
        logger.info("Starting Tkinter main loop...")
        # print("DEBUG: Calling app.mainloop()...") # REMOVED Debug Print
        app.mainloop()
        # print("DEBUG: app.mainloop() finished.") # REMOVED Debug Print

    except SystemExit: # Catch explicit exits
        # print("DEBUG: SystemExit caught during startup.") # REMOVED Debug Print
        logger.warning("Application exited during startup.")
        # Optionally re-raise or handle differently if needed
        # raise # Re-raise if you want the exit code to propagate
    except Exception as e:
        # Catch any unexpected startup errors not caught by specific checks
        # Ensure logging is working at this point
        try:
            logger.critical(f"!!! UNHANDLED STARTUP ERROR !!!", exc_info=True)
            # print(f"DEBUG: Unhandled Exception during startup: {type(e).__name__}: {e}") # REMOVED Debug Print
        except Exception as log_err:
            print(f"CRITICAL ERROR during startup: {e}")
            print(f"Logging also failed: {log_err}")

        try: # Basic Tk fallback error message
            root_err = tkinter.Tk(); root_err.withdraw()
            messagebox.showerror("Startup Error", f"Application failed to start:\n\n{type(e).__name__}: {e}\n\nCheck log file or console for details.")
            root_err.destroy()
        except Exception as msg_err:
            print(f"Could not display error messagebox: {msg_err}")
        sys.exit(1) # Exit after showing error
    finally:
        # print("DEBUG: Exiting __main__ block.") # REMOVED Debug Print
        logger.info("--- Videous Chef Shutting Down ---") # Add final log message
