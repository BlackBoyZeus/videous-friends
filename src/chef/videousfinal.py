
 # -*- coding: utf-8 -*-
# ========================================================================
#                       IMPORTS (Comprehensive & Updated)
# ========================================================================
import tkinter
from tkinter import filedialog, Listbox, Scrollbar, END, MULTIPLE, Frame, messagebox, BooleanVar, IntVar # Added IntVar
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
# --- MoviePy Imports ---
from moviepy import VideoFileClip, AudioFileClip, VideoClip

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

# --- REMOVED VidGear Imports ---
# from vidgear.gears import VideoGear
# from vidgear.gears import WriteGear

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
# --- Adjusted Defaults for Sequence Building ---
# Lowered repetition penalties and relaxed mood matching
# to improve performance with limited source videos.
SIGMA_M_DEFAULT = 0.3 # Was 0.2
DEFAULT_REPETITION_PENALTY = 0.15 # Was 0.3 (Physics MC Variety Penalty)
DEFAULT_GREEDY_SOURCE_PENALTY = 0.1 # Was 0.2
DEFAULT_GREEDY_SHOT_PENALTY = 0.1 # Was 0.15
# --- End Adjusted Defaults ---
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
    resolution_height: int = 256 # Target analysis resolution
    resolution_width: int = 256 # Target analysis resolution
    save_analysis_data: bool = True

    # Normalization
    norm_max_velocity: float = V_MAX_EXPECTED
    norm_max_acceleration: float = A_MAX_EXPECTED
    norm_max_rms: float = DEFAULT_NORM_MAX_RMS
    norm_max_onset: float = DEFAULT_NORM_MAX_ONSET
    norm_max_chroma_var: float = DEFAULT_NORM_MAX_CHROMA_VAR
    norm_max_depth_variance: float = D_MAX_EXPECTED
    # Heuristic Normalization (Adjustable via UI)
    norm_max_kinetic: float = 50.0
    norm_max_jerk: float = 30.0
    norm_max_cam_motion: float = 5.0
    norm_max_face_size: float = 0.6

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
    variety_penalty_source: float = DEFAULT_GREEDY_SOURCE_PENALTY # Adjusted Default
    variety_penalty_shot: float = DEFAULT_GREEDY_SHOT_PENALTY # Adjusted Default
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
    mood_similarity_variance: float = SIGMA_M_DEFAULT # Adjusted Default
    continuity_depth_weight: float = 0.5 # k_d in DeltaE
    variety_repetition_penalty: float = DEFAULT_REPETITION_PENALTY # Adjusted Default (Lambda)


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

    # Effect definitions (will be populated from UI sliders via _get_render_config)
    effect_settings: Dict[str, EffectParams] = field(default_factory=lambda: {
        "cut": EffectParams(type="cut"),
        "fade": EffectParams(type="fade", tau=0.2, psi=0.1, epsilon=0.2), # Defaults
        "zoom": EffectParams(type="zoom", tau=0.5, psi=0.3, epsilon=0.4), # Defaults
        "pan": EffectParams(type="pan", tau=0.5, psi=0.1, epsilon=0.3),  # Defaults
    })

    # MoviePy / FFmpeg Output Settings
    video_codec: str = 'libx264'
    preset: Optional[str] = 'medium' # Used by MoviePy's libx264 (via ffmpeg_params), Make Optional
    crf: Optional[int] = 23          # Used by MoviePy's libx264 via ffmpeg_params, Make Optional
    audio_codec: str = 'aac'
    audio_bitrate: str = '192k'
    threads: int = max(1, (os.cpu_count() or 2) // 2)
    # ffmpeg_loglevel: str = 'warning' # Only relevant if directly using FFmpeg/WriteGear

    # Resolution/FPS for MoviePy rendering (Set from UI)
    resolution_width: int = 1920
    resolution_height: int = 1080
    fps: int = 30

# ========================================================================
#                       HELPER FUNCTIONS (Logging Added)
# ========================================================================
def tk_write(tk_string1, parent=None, level="info"):
    """Shows message box. Logs the message. level can be 'info', 'warning', 'error'."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, f"Popup ({level}): {tk_string1}")

    try:
        if parent and hasattr(parent, 'winfo_exists') and parent.winfo_exists():
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
        print(f"!! tk_write Error: {e}\n!! Level: {level}\n!! Message: {tk_string1}")


def get_midas_model(model_type="MiDaS_small"):
    """Loads MiDaS model and transform, attempting process-local caching."""
    global _midas_model_cache, _midas_transform_cache, _midas_device_cache
    if _midas_model_cache and _midas_transform_cache and _midas_device_cache:
        logger.debug("Using cached MiDaS model.")
        return _midas_model_cache, _midas_transform_cache, _midas_device_cache

    logger.info(f"Loading MiDaS model: {model_type}...")
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Added is_built check and hasattr
             device = torch.device("mps"); logger.info("Using Apple Metal (MPS) backend.")
        elif torch.cuda.is_available():
             device = torch.device("cuda"); logger.info("Using CUDA backend.")
        else:
             device = torch.device("cpu"); logger.info("Using CPU backend.")
        _midas_device_cache = device

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
        # Clamp input to avoid extreme values causing overflow
        x_clamped = np.clip(x, -700, 700) # Adjust range if necessary
        return 1 / (1 + np.exp(-k * x_clamped)) # Use np.exp for array compatibility
    except OverflowError:
        # This path might be less likely with clamping, but kept as fallback
        logger.warning(f"Sigmoid overflow for input {x}. Clamping result.")
        return 0.0 if x < 0 else 1.0

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    vec1 = np.asarray(vec1, dtype=float); vec2 = np.asarray(vec2, dtype=float)
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    # Clamp to handle potential floating point inaccuracies slightly outside [-1, 1]
    return np.clip(similarity, -1.0, 1.0)


def calculate_histogram_entropy(frame):
    if frame is None or frame.size == 0: return 0.0
    try:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # OpenCV uses BGR
        elif len(frame.shape) == 2:
            gray = frame
        else:
            logger.warning(f"Invalid frame shape for histogram entropy: {frame.shape}")
            return 0.0

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_sum = hist.sum()
        if hist_sum <= 0: return 0.0 # Avoid division by zero
        hist_norm = hist.ravel() / hist_sum
        # Filter out zero probabilities for entropy calculation to avoid log(0)
        hist_norm_nonzero = hist_norm[hist_norm > 0]
        if hist_norm_nonzero.size == 0: return 0.0 # All probabilities were zero? (Shouldn't happen if hist_sum > 0)

        entropy = scipy.stats.entropy(hist_norm_nonzero) # Use only non-zero probabilities
        return entropy if np.isfinite(entropy) else 0.0
    except Exception as e:
        logger.warning(f"Histogram entropy calculation failed: {e}", exc_info=False) # Don't need full trace often
        return 0.0

# ========================================================================
#           AUDIO ANALYSIS UTILITIES (Logging Enhanced)
# ========================================================================
class BBZAudioUtils:
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
                "ffmpeg", "-i", shlex.quote(video_path), # Quote input path
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-y",
                shlex.quote(audio_output_path), # Quote output path
                "-hide_banner", "-loglevel", "error" # Keep FFmpeg quiet unless error
            ]
            logger.debug(f"Executing FFmpeg: {' '.join(command)}")

            result = subprocess.run(" ".join(command), shell=True, capture_output=True, text=True, check=False, encoding='utf-8') # Use shell=True with quoted command string

            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction failed (Code: {result.returncode}) for {os.path.basename(video_path)}.")
                logger.error(f"FFmpeg stderr:\n{result.stderr}")
                if os.path.exists(audio_output_path):
                    try: os.remove(audio_output_path)
                    except OSError as del_err: logger.warning(f"Could not remove failed temp audio file {audio_output_path}: {del_err}")
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
            # This might catch ffmpeg not being found
            logger.error(f"File not found during audio extraction setup (or ffmpeg missing?): {fnf_err}")
            return None
        except Exception as e:
            logger.error(f"Error during FFmpeg audio extraction '{os.path.basename(video_path)}': {e}", exc_info=True)
            if os.path.exists(audio_output_path):
                try: os.remove(audio_output_path)
                except OSError as del_err: logger.warning(f"Could not remove temp audio file {audio_output_path}: {del_err}")
            return None

    def analyze_audio(self, audio_path: str, analysis_config: AnalysisConfig, target_sr: int = 22050, segment_hop_factor: int = 4) -> Optional[Dict[str, Any]]:
        """Calculates enhanced audio features using AnalysisConfig."""
        logger.info(f"Analyzing audio: {os.path.basename(audio_path)}")
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

            if isinstance(tempo, (np.ndarray, np.generic)):
                try: tempo_float = float(tempo.item())
                except ValueError:
                    logger.warning(f"Tempo detection returned non-scalar array: {tempo}. Using first element if available.")
                    tempo_float = float(tempo[0]) if tempo.size > 0 else 120.0 # Fallback tempo
            else: tempo_float = float(tempo)

            logger.debug("Computing RMS energy...")
            rms_energy = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
            rms_times = librosa.times_like(rms_energy, sr=sr, hop_length=hop_length)

            logger.debug("Computing Chroma Features...")
            chroma_hop = hop_length * segment_hop_factor
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=chroma_hop)
            try: chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=chroma_hop)
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
                if seg_duration <= 1e-6: continue # Avoid zero or tiny segments

                rms_indices = np.where((rms_times >= t_start) & (rms_times < t_end))[0]
                seg_rms = np.mean(rms_energy[rms_indices]) if len(rms_indices) > 0 else 0.0
                onset_indices = np.where((onset_times >= t_start) & (onset_times < t_end))[0]
                seg_onset = np.mean(onset_env[onset_indices]) if len(onset_indices) > 0 else 0.0
                chroma_indices = np.where((chroma_times >= t_start) & (chroma_times < t_end))[0]
                seg_chroma_var = np.mean(np.var(chroma[:, chroma_indices], axis=1)) if len(chroma_indices) > 0 else 0.0

                segment_features.append({
                    'start': t_start, 'end': t_end, 'duration': seg_duration,
                    'rms_avg': float(seg_rms), 'onset_avg': float(seg_onset), 'chroma_var': float(seg_chroma_var)
                })

            logger.info(f"Audio analysis complete ({time.time() - start_time:.2f}s). Tempo: {tempo_float:.2f} BPM, Beats: {len(beat_times)}, Segments: {len(segment_features)}")

            norm_max_rms = analysis_config.norm_max_rms; norm_max_onset = analysis_config.norm_max_onset
            norm_max_chroma_var = analysis_config.norm_max_chroma_var
            for seg in segment_features:
                 seg['b_i'] = np.clip(seg['rms_avg'] / (norm_max_rms + 1e-6), 0.0, 1.0)
                 seg['e_i'] = np.clip(seg['onset_avg'] / (norm_max_onset + 1e-6), 0.0, 1.0)
                 arousal_proxy = np.clip(tempo_float / 150.0, 0.0, 1.0) # Tempo range assumption
                 valence_proxy = np.clip(1.0 - (seg['chroma_var'] / (norm_max_chroma_var + 1e-6)), 0.0, 1.0)
                 seg['m_i'] = [float(arousal_proxy), float(valence_proxy)]

            # Ensure raw features are JSON serializable
            raw_features = {
                 'rms_energy': [float(x) for x in rms_energy.tolist()],
                 'rms_times': [float(x) for x in rms_times.tolist()],
                 # Add others like 'percussive_ratio' if calculated and stored here
            }

            return {
                "sr": sr, "duration": float(duration), "tempo": float(tempo_float),
                "beat_times": [float(x) for x in beat_times.tolist()],
                "segment_boundaries": [float(x) for x in bound_times],
                "segment_features": segment_features, # For Physics Mode (already contains floats)
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
            # Librosa < 0.10 compatibility check
            librosa_version_major_minor = tuple(map(int, librosa.__version__.split('.')[:2]))

            # Recurrence matrix calculation might need adjustment based on features shape/type
            try:
                R = librosa.segment.recurrence_matrix(features, k=k_nn, width=3, mode='connectivity', sym=True)
            except Exception as R_err:
                logger.warning(f"Recurrence matrix calculation failed: {R_err}. Using simple features.")
                # Fallback: use simple magnitude or similar if chroma fails
                R = librosa.segment.recurrence_matrix(np.abs(features), k=k_nn, width=3, mode='connectivity', sym=True)

            # Use a simplified filter if the custom one fails
            try:
                def medfilt2d_wrapper(data, kernel_size):
                    data_float = data.astype(float) if data.dtype == bool else data
                    return scipy.signal.medfilt2d(data_float, kernel_size=kernel_size)
                df = librosa.segment.timelag_filter(lambda data, **kwargs: medfilt2d_wrapper(data, kernel_size=kwargs.get('size')))
                Rf = df(R, size=(1, 7)) # Use recurrence matrix R here
            except Exception as filter_err:
                logger.warning(f"Timelag filter failed: {filter_err}. Skipping filtering.")
                Rf = R # Use the unfiltered matrix

            # --- Librosa Agglomerative Call (Version Aware) ---
            try:
                if librosa_version_major_minor >= (0, 10):
                    logger.debug("Using librosa.segment.agglomerative (>=0.10.0).")
                    # Pass recurrence matrix Rf, not original features
                    bounds_frames = librosa.segment.agglomerative(Rf)
                else:
                    default_k = 10 # Provide a default value for k
                    logger.debug(f"Using librosa.segment.agglomerative (<0.10.0) with k={default_k}.")
                    # Pass recurrence matrix Rf, not original features
                    bounds_frames = librosa.segment.agglomerative(Rf, k=default_k)
            except TypeError as agg_err:
                 # This specific check might be redundant if version check is accurate, but handles unexpected API shifts
                 logger.error(f"Librosa agglomerative call failed unexpectedly ({agg_err}). Trying with k=10 fallback.", exc_info=False)
                 try:
                     bounds_frames = librosa.segment.agglomerative(Rf, k=10)
                 except Exception as fallback_err:
                     logger.error(f"Agglomerative fallback also failed: {fallback_err}. Proceeding with fixed segments.", exc_info=True)
                     raise ValueError("Agglomerative segmentation failed.")
            except Exception as agg_generic_err:
                 logger.error(f"Librosa agglomerative call failed: {agg_generic_err}. Proceeding with fixed segments.", exc_info=True)
                 raise ValueError("Agglomerative segmentation failed.")

            bound_times = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop_length).tolist()
            # --- Segment boundary refinement ---
            final_bounds = sorted(list(set([0.0] + [t for t in bound_times if np.isfinite(t)] + [duration]))) # Ensure finite times
            min_len_sec = 0.5 # Minimum segment length
            final_bounds_filtered = [final_bounds[0]]
            for i in range(1, len(final_bounds)):
                # Ensure segment has minimum length AND is a valid time
                if np.isfinite(final_bounds[i]) and (final_bounds[i] - final_bounds_filtered[-1]) >= min_len_sec:
                    final_bounds_filtered.append(final_bounds[i])
                # Handle the very last boundary potentially being too close
                elif i == len(final_bounds) - 1 and np.isfinite(final_bounds[i]):
                    # Merge if the last proposed segment is too short
                    if (final_bounds[i] - final_bounds_filtered[-1]) < min_len_sec:
                        final_bounds_filtered[-1] = final_bounds[i] # Extend the previous segment to the end
                    else:
                         final_bounds_filtered.append(final_bounds[i]) # Keep it if long enough

            # Ensure the final boundary is exactly the duration
            if final_bounds_filtered and final_bounds_filtered[-1] < duration:
                if duration - final_bounds_filtered[-1] >= min_len_sec / 2: # Allow slightly shorter last seg
                    final_bounds_filtered.append(duration)
                else: # Merge last segment if too short
                    final_bounds_filtered[-1] = duration
            elif not final_bounds_filtered: # Handle empty list case
                 final_bounds_filtered = [0.0, duration]

            # Ensure start is 0.0
            if not final_bounds_filtered or final_bounds_filtered[0] > 1e-6:
                 final_bounds_filtered.insert(0, 0.0)

            # Remove duplicates that might arise from merging/clamping
            final_bounds_filtered = sorted(list(set(final_bounds_filtered)))
            # Filter out segments that are too short after deduplication
            final_bounds_cleaned = [final_bounds_filtered[0]]
            for i in range(1, len(final_bounds_filtered)):
                if final_bounds_filtered[i] - final_bounds_cleaned[-1] >= 1e-3: # Keep segments longer than ~1ms
                    final_bounds_cleaned.append(final_bounds_filtered[i])
            if final_bounds_cleaned[-1] < duration - 1e-3: # Ensure end is duration
                 final_bounds_cleaned.append(duration)

            if len(final_bounds_cleaned) < 2: # Ensure at least one segment [0, duration]
                logger.warning("Segmentation resulted in less than 2 boundaries. Using [0, duration].")
                return [0.0, duration]

            return final_bounds_cleaned

        except ValueError as ve: # Catch specific error from logs (or our raised ValueError)
             logger.warning(f"Audio segmentation failed ({type(ve).__name__}: {ve}). Falling back to fixed segments.", exc_info=False)
        except Exception as e:
            logger.warning(f"Audio segmentation failed ({type(e).__name__}: {e}). Falling back to fixed segments.", exc_info=True) # Log full trace for unexpected

        # Fallback to fixed segments
        num_segments = max(2, int(duration / 10.0)) # Ensure at least 2 segments
        return np.linspace(0, duration, num_segments + 1).tolist()


# ========================================================================
#                    FACE UTILITIES (Logging Added)
# ========================================================================
class BBZFaceUtils:
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
        """Gets face features. Expects BGR image input."""
        if self.face_mesh is None: return False, 0.0, 0.5
        is_open, size_ratio, center_x = False, 0.0, 0.5
        h, w = image.shape[:2]
        if h == 0 or w == 0: return False, 0.0, 0.5

        try:
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB for MediaPipe
            results = self.face_mesh.process(image_rgb)
            image.flags.writeable = True

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0] # Get first detected face
                lm = face_landmarks.landmark

                # Check landmark indices exist before accessing
                if len(lm) > 152: # Ensure common landmarks are present
                    # Mouth Open Ratio
                    upper_lip_y = lm[13].y * h; lower_lip_y = lm[14].y * h
                    mouth_height = abs(lower_lip_y - upper_lip_y)
                    forehead_y = lm[10].y * h; chin_y = lm[152].y * h
                    face_height = abs(chin_y - forehead_y)
                    if face_height > 1e-6:
                        mouth_open_ratio = mouth_height / face_height
                        is_open = mouth_open_ratio > mouth_open_threshold

                    # Face Size Ratio
                    all_x = [lm_pt.x * w for lm_pt in lm]; all_y = [lm_pt.y * h for lm_pt in lm]
                    # Use min/max of landmark coords to estimate bounding box diagonal
                    min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y)
                    face_box_w = max_x - min_x; face_box_h = max_y - min_y
                    face_diagonal = math.sqrt(face_box_w**2 + face_box_h**2)
                    image_diagonal = math.sqrt(w**2 + h**2)
                    if image_diagonal > 1e-6:
                        size_ratio = np.clip(face_diagonal / image_diagonal, 0.0, 1.0)

                    # Face Center X
                    if len(lm) > 454: # Ensure cheek landmarks exist
                         left_cheek_x = lm[234].x; right_cheek_x = lm[454].x
                         center_x = np.clip((left_cheek_x + right_cheek_x) / 2.0, 0.0, 1.0)
                    else: # Fallback using overall landmark center
                         center_x = np.clip(np.mean([lm_pt.x for lm_pt in lm]), 0.0, 1.0)
                else:
                     logger.warning("FaceMesh returned insufficient landmarks. Using defaults.")


        except Exception as e:
            logger.warning(f"Error processing frame with FaceMesh: {e}", exc_info=False)
            # Return defaults on error
            return False, 0.0, 0.5

        return is_open, float(size_ratio), float(center_x) # Ensure float return

    def close(self):
        if self.face_mesh:
            try: self.face_mesh.close(); logger.info("FaceMesh resources released.")
            except Exception as e: logger.error(f"Error closing FaceMesh: {e}")
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
        # Use nanmean to handle potential NaN values in flow calculation
        avg_magnitude = np.nanmean(magnitude)
        if not np.isfinite(avg_magnitude): avg_magnitude = 0.0 # Handle case where all values are NaN
        scale_factor = 10.0 # Empirical scaling factor
        return float(avg_magnitude * scale_factor), flow
    except cv2.error as cv_err:
         # Log specific OpenCV errors which can be common
         logger.warning(f"OpenCV error during flow calculation: {cv_err}")
         return 0.0, None
    except Exception as e:
        logger.warning(f"Generic error during flow velocity calculation: {e}")
        return 0.0, None

def calculate_flow_acceleration(prev_flow, current_flow, dt):
    if prev_flow is None or current_flow is None or prev_flow.shape != current_flow.shape or dt <= 1e-6:
        return 0.0 # Avoid division by zero or near-zero dt
    try:
        flow_diff = current_flow - prev_flow
        accel_magnitude_per_pixel, _ = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
        # Use nanmean for robustness
        avg_accel_magnitude = np.nanmean(accel_magnitude_per_pixel)
        if not np.isfinite(avg_accel_magnitude): avg_accel_magnitude = 0.0
        accel = avg_accel_magnitude / dt
        scale_factor = 10.0 # Empirical scaling factor
        return float(accel * scale_factor)
    except Exception as e:
        logger.warning(f"Error during flow acceleration calculation: {e}")
        return 0.0

def calculate_kinetic_energy_proxy(landmarks_prev, landmarks_curr, dt):
    if landmarks_prev is None or landmarks_curr is None or dt <= 1e-6: return 0.0
    if not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0
    lm_prev = landmarks_prev.landmark; lm_curr = landmarks_curr.landmark
    if len(lm_prev) != len(lm_curr): return 0.0

    total_sq_velocity = 0.0; num_valid = 0
    try:
        for i in range(len(lm_prev)):
            # Check visibility/presence if available (adjust threshold as needed)
            vis_prev = getattr(lm_prev[i], 'visibility', 1.0)
            vis_curr = getattr(lm_curr[i], 'visibility', 1.0)
            if vis_prev > 0.1 and vis_curr > 0.1:
                dx = lm_curr[i].x - lm_prev[i].x; dy = lm_curr[i].y - lm_prev[i].y
                dz = lm_curr[i].z - lm_prev[i].z # Z can be less reliable
                # Consider weighting Z less or ignoring it if too noisy
                # vx = dx / dt; vy = dy / dt; vz = dz / dt
                # total_sq_velocity += vx**2 + vy**2 + 0.5*(vz**2) # Example: Weight Z less
                total_sq_velocity += (dx**2 + dy**2 + dz**2) / (dt**2) # Direct v^2
                num_valid += 1
    except IndexError:
         logger.warning("Index error accessing landmarks in kinetic energy calc.")
         return 0.0

    if num_valid == 0: return 0.0
    avg_sq_velocity = total_sq_velocity / num_valid
    scale_factor = 1000.0 # Empirical scaling
    return float(avg_sq_velocity * scale_factor)


def calculate_movement_jerk_proxy(landmarks_prev_prev, landmarks_prev, landmarks_curr, dt):
    if landmarks_prev_prev is None or landmarks_prev is None or landmarks_curr is None or dt <= 1e-6: return 0.0
    if not hasattr(landmarks_prev_prev, 'landmark') or not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0
    lm_pp = landmarks_prev_prev.landmark; lm_p = landmarks_prev.landmark; lm_c = landmarks_curr.landmark
    if len(lm_pp) != len(lm_p) or len(lm_p) != len(lm_c): return 0.0

    total_sq_jerk = 0.0; num_valid = 0; dt_sq = dt * dt
    try:
        for i in range(len(lm_pp)):
            vis_pp = getattr(lm_pp[i], 'visibility', 1.0)
            vis_p = getattr(lm_p[i], 'visibility', 1.0)
            vis_c = getattr(lm_c[i], 'visibility', 1.0)
            if vis_pp > 0.1 and vis_p > 0.1 and vis_c > 0.1: # Check visibility across all three frames
                # Finite difference approximation for acceleration components
                ax = (lm_c[i].x - 2*lm_p[i].x + lm_pp[i].x) / dt_sq
                ay = (lm_c[i].y - 2*lm_p[i].y + lm_pp[i].y) / dt_sq
                az = (lm_c[i].z - 2*lm_p[i].z + lm_pp[i].z) / dt_sq # Again, Z might be noisy
                # Finite difference approximation for jerk (derivative of acceleration)
                # This requires 4 points for a centered difference, or a forward/backward difference
                # Simpler: Use squared acceleration difference as proxy for jerk magnitude
                # acc_prev = [(lm_p[i].x - 2*lm_pp[i].x + lm_ppp[i].x) / dt_sq, ...] # Needs landmarks_prev_prev_prev
                # Instead, approximate jerk as magnitude of change in velocity squared?
                # Or simply use magnitude of acceleration as a proxy related to "jerkiness"

                # Let's stick to the finite difference of velocity:
                # v_curr = [(lm_c[i].x - lm_p[i].x)/dt, ...]
                # v_prev = [(lm_p[i].x - lm_pp[i].x)/dt, ...]
                # a_approx = [(v_curr[0] - v_prev[0])/dt, ...] = [(lm_c[i].x - 2*lm_p[i].x + lm_pp[i].x)/dt_sq, ...]
                # Using acceleration magnitude directly
                acc_magnitude_sq = ax**2 + ay**2 + az**2 # Using acceleration magnitude as proxy for jerkiness
                total_sq_jerk += acc_magnitude_sq
                num_valid += 1
    except IndexError:
        logger.warning("Index error accessing landmarks in jerk calc.")
        return 0.0

    if num_valid == 0: return 0.0
    avg_sq_jerk_proxy = total_sq_jerk / num_valid
    # Adjust scale factor based on using acceleration magnitude sq as proxy
    scale_factor = 100000.0 # May need significant tuning
    return float(avg_sq_jerk_proxy * scale_factor)

class BBZPoseUtils:
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
        self.fps = fps if fps > 0 else 30.0 # Ensure valid FPS
        self.score_threshold = analysis_config.score_threshold
        self.min_segment_len_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES,
                                           int(analysis_config.min_potential_clip_duration_sec * self.fps))
        logger.debug(f"Segment Identifier Init: FPS={self.fps:.2f}, Threshold={self.score_threshold:.2f}, MinLenFrames={self.min_segment_len_frames}")


    def find_potential_segments(self, frame_features_list):
        potential_segments = []
        start_idx = -1
        n = len(frame_features_list)
        if n == 0: return []

        for i, features in enumerate(frame_features_list):
            # Ensure features is a dict and key exists
            score = features.get('boosted_score', 0.0) if isinstance(features, dict) else 0.0
            is_candidate = score >= self.score_threshold

            if is_candidate and start_idx == -1:
                start_idx = i # Start of a potential segment
            elif not is_candidate and start_idx != -1:
                # End of a potential segment
                segment_len = i - start_idx
                if segment_len >= self.min_segment_len_frames:
                    potential_segments.append({'start_frame': start_idx, 'end_frame': i}) # End frame is exclusive
                    logger.debug(f"Found potential segment: Frames {start_idx}-{i} (Len: {segment_len})")
                else:
                    logger.debug(f"Discarded short potential segment: Frames {start_idx}-{i} (Len: {segment_len}, Min: {self.min_segment_len_frames})")
                start_idx = -1 # Reset

        # Check if the video ends with a candidate segment
        if start_idx != -1:
            segment_len = n - start_idx
            if segment_len >= self.min_segment_len_frames:
                potential_segments.append({'start_frame': start_idx, 'end_frame': n})
                logger.debug(f"Found potential segment ending at video end: Frames {start_idx}-{n} (Len: {segment_len})")
            else:
                 logger.debug(f"Discarded short potential segment at video end: Frames {start_idx}-{n} (Len: {segment_len}, Min: {self.min_segment_len_frames})")


        fps_val = self.fps if self.fps > 0 else 30.0 # Use default if fps is bad
        min_len_s = self.min_segment_len_frames / fps_val
        logger.info(f"Identified {len(potential_segments)} potential segments (Heuristic Score >= {self.score_threshold:.2f}, MinLen={min_len_s:.2f}s)")
        return potential_segments

# ========================================================================
#                      IMAGE UTILITIES (Logging Added)
# ========================================================================
class BBZImageUtils:
    def resizeTARGET(self, image, TARGET_HEIGHT=256, TARGET_WIDTH=256):
        """Resizes image using OpenCV. Expects BGR input."""
        if image is None or image.size == 0:
            logger.warning("Resize received empty image.")
            return None
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            logger.warning(f"Resize received image with zero dimension: {h}x{w}")
            return None # Return None for invalid input dimensions
        # Only resize if needed
        if h != TARGET_HEIGHT or w != TARGET_WIDTH:
            # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
            interpolation = cv2.INTER_AREA if h > TARGET_HEIGHT or w > TARGET_WIDTH else cv2.INTER_LINEAR
            try:
                return cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=interpolation)
            except cv2.error as cv_err: # Catch specific OpenCV errors
                logger.warning(f"OpenCV error during resize: {cv_err}. Returning original.")
                return image
            except Exception as e:
                logger.warning(f"Generic resize failed: {e}. Returning original.")
                return image
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
        # Ensure duration is calculated correctly even if num_frames is 0
        self.duration = self.num_frames / self.fps if self.fps > 0 and self.num_frames > 0 else 0.0
        self.start_time = start_frame / self.fps if self.fps > 0 else 0.0
        self.end_time = end_frame / self.fps if self.fps > 0 else 0.0
        self.analysis_config = analysis_config # Store config dataclass

        # Safely slice features, handle edge cases
        if 0 <= start_frame < end_frame <= len(all_frame_features):
             self.segment_frame_features = all_frame_features[start_frame:end_frame]
        else:
             logger.warning(f"Invalid frame indices for ClipSegment: Start={start_frame}, End={end_frame} (Total frames: {len(all_frame_features)}) for {os.path.basename(str(source_video_path))}. Creating with empty features.")
             self.segment_frame_features = []
             self.num_frames = 0 # Correct num_frames if indices were invalid
             self.duration = 0.0 # Correct duration

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
        self.chosen_duration = self.duration # Initialize chosen duration
        self.chosen_effect: Optional[EffectParams] = None # Updated type hint
        self.temp_chosen_duration = self.duration # Used by Greedy
        # Subclip timing for rendering
        self.subclip_start_time_in_source = self.start_time
        self.subclip_end_time_in_source = self.end_time

        if self.segment_frame_features:
            self._calculate_heuristic_aggregates()
            self._calculate_physics_aggregates()
        else:
             # Log if segment was intended to have frames but ended up empty
             if self.num_frames > 0:
                 logger.warning(f"Cannot calculate aggregates for segment {start_frame}-{end_frame} ({os.path.basename(str(source_video_path))}) due to empty features despite num_frames={self.num_frames}.")

    def _calculate_heuristic_aggregates(self):
        count = len(self.segment_frame_features)
        if count == 0: return

        # Use stored config for normalization constants
        norm_kinetic = self.analysis_config.norm_max_kinetic + 1e-6
        norm_jerk = self.analysis_config.norm_max_jerk + 1e-6
        norm_cam_motion = self.analysis_config.norm_max_cam_motion + 1e-6
        norm_face_size = self.analysis_config.norm_max_face_size + 1e-6

        # Safely extract features, handling missing keys or non-dict items
        raw_scores = [f.get('raw_score', 0.0) for f in self.segment_frame_features if isinstance(f, dict)]
        boosted_scores = [f.get('boosted_score', 0.0) for f in self.segment_frame_features if isinstance(f, dict)]
        cam_motions = [f.get('camera_motion', 0.0) for f in self.segment_frame_features if isinstance(f, dict)]
        motions = [f.get('pose_features', {}).get('kinetic_energy_proxy', 0.0) for f in self.segment_frame_features if isinstance(f, dict) and 'pose_features' in f]
        jerks = [f.get('pose_features', {}).get('movement_jerk_proxy', 0.0) for f in self.segment_frame_features if isinstance(f, dict) and 'pose_features' in f]
        face_sizes = [f.get('pose_features', {}).get('face_size_ratio', 0.0) for f in self.segment_frame_features if isinstance(f, dict) and f.get('pose_features', {}).get('face_size_ratio', 0.0) > 0.01]
        face_present_frames = sum(1 for f in self.segment_frame_features if isinstance(f, dict) and f.get('pose_features', {}).get('face_size_ratio', 0.0) > 0.01)
        beats = [f.get('is_beat_frame', False) for f in self.segment_frame_features if isinstance(f, dict)]
        contributors = [f.get('dominant_contributor', 'none') for f in self.segment_frame_features if isinstance(f, dict)]
        intensities = [f.get('intensity_category', 'Low') for f in self.segment_frame_features if isinstance(f, dict)]
        sections = {f.get('musical_section_index', -1) for f in self.segment_frame_features if isinstance(f, dict) and f.get('musical_section_index', -1) != -1}

        # Use np.nanmean for robustness against potential NaN values (though less likely here)
        self.avg_raw_score = float(np.nanmean(raw_scores)) if raw_scores else 0.0
        self.avg_boosted_score = float(np.nanmean(boosted_scores)) if boosted_scores else 0.0
        self.peak_boosted_score = float(np.nanmax(boosted_scores)) if boosted_scores else 0.0
        # Aggregate heuristic proxies (normalized values not stored, so use raw averages for now)
        self.avg_motion_heuristic = float(np.nanmean(motions)) if motions else 0.0
        self.avg_jerk_heuristic = float(np.nanmean(jerks)) if jerks else 0.0
        self.avg_camera_motion = float(np.nanmean(cam_motions)) if cam_motions else 0.0
        self.face_presence_ratio = float(face_present_frames / count) if count > 0 else 0.0
        self.avg_face_size = float(np.nanmean(face_sizes)) if face_sizes else 0.0
        self.contains_beat = any(beats)
        self.musical_section_indices = sections

        if contributors:
             non_none = [c for c in contributors if c != 'none']
             if non_none: self.dominant_contributor = max(set(non_none), key=non_none.count)
        intensity_order = ['Low', 'Medium', 'High']; highest_intensity = 'Low'
        if intensities:
            # Find the highest intensity category present in the segment
            intensity_indices = [intensity_order.index(i) for i in intensities if i in intensity_order]
            if intensity_indices:
                 highest_intensity = intensity_order[max(intensity_indices)]
        self.intensity_category = highest_intensity

    def _calculate_physics_aggregates(self):
        count = len(self.segment_frame_features)
        if count == 0: return

        # Extract features safely, filtering None and non-finite values
        velocities = [f.get('flow_velocity') for f in self.segment_frame_features if isinstance(f, dict) and f.get('flow_velocity') is not None and np.isfinite(f.get('flow_velocity'))]
        accelerations = [f.get('flow_acceleration') for f in self.segment_frame_features if isinstance(f, dict) and f.get('flow_acceleration') is not None and np.isfinite(f.get('flow_acceleration'))]
        depth_vars = [f.get('depth_variance') for f in self.segment_frame_features if isinstance(f, dict) and f.get('depth_variance') is not None and np.isfinite(f.get('depth_variance'))]
        entropies = [f.get('histogram_entropy') for f in self.segment_frame_features if isinstance(f, dict) and f.get('histogram_entropy') is not None and np.isfinite(f.get('histogram_entropy'))]

        self.v_k = float(np.mean(velocities)) if velocities else 0.0
        self.a_j = float(np.mean(accelerations)) if accelerations else 0.0
        avg_depth_var = float(np.mean(depth_vars)) if depth_vars else 0.0
        d_max_norm = self.analysis_config.norm_max_depth_variance
        self.d_r = np.clip(avg_depth_var / (d_max_norm + 1e-6), 0.0, 1.0) # Normalized Depth Variance
        self.phi = float(np.mean(entropies)) if entropies else 0.0 # Average frame entropy

        v_max_norm = self.analysis_config.norm_max_velocity
        # Calculate mood vector based on aggregated physics features
        arousal_proxy = np.clip(self.v_k / (v_max_norm + 1e-6), 0.0, 1.0)
        valence_proxy = 1.0 - self.d_r # Inverse normalized depth variance proxy
        self.mood_vector = [float(arousal_proxy), float(valence_proxy)]

    def clip_audio_fit(self, audio_segment: Dict, analysis_config: AnalysisConfig) -> float:
        """Calculates how well this visual clip fits a given audio segment (Physics mode)."""
        w_v = analysis_config.fit_weight_velocity; w_a = analysis_config.fit_weight_acceleration
        w_m = analysis_config.fit_weight_mood; v_max_norm = analysis_config.norm_max_velocity
        a_max_norm = analysis_config.norm_max_acceleration
        sigma_m_sq = analysis_config.mood_similarity_variance**2 * 2 # Variance for mood Gaussian

        audio_beat_strength = audio_segment.get('b_i', 0.0) # Normalized RMS proxy
        audio_energy = audio_segment.get('e_i', 0.0)        # Normalized Onset proxy
        audio_mood = np.asarray(audio_segment.get('m_i', [0.0, 0.0])) # [Arousal, Valence]

        v_k_norm = np.clip(self.v_k / (v_max_norm + 1e-6), 0.0, 1.0)
        a_j_norm = np.clip(self.a_j / (a_max_norm + 1e-6), 0.0, 1.0)
        vid_mood = np.asarray(self.mood_vector)

        # Calculate mood similarity using Gaussian kernel
        mood_dist_sq = np.sum((vid_mood - audio_mood)**2)
        mood_term = exp(-mood_dist_sq / (sigma_m_sq + 1e-9)) # Add epsilon for stability

        # Combine weighted terms
        velocity_term = w_v * v_k_norm * audio_beat_strength
        acceleration_term = w_a * a_j_norm * audio_energy
        mood_term_weighted = w_m * mood_term

        score = velocity_term + acceleration_term + mood_term_weighted
        k_sigmoid = analysis_config.fit_sigmoid_steepness
        # Apply sigmoid for probability-like value in [0, 1]
        return float(sigmoid(score, k=k_sigmoid))

    def get_feature_vector(self, analysis_config: AnalysisConfig) -> List[float]:
        """Returns a normalized feature vector [v_k_norm, a_j_norm, d_r] for continuity calc."""
        v_max_norm = analysis_config.norm_max_velocity + 1e-6 # Add epsilon
        a_max_norm = analysis_config.norm_max_acceleration + 1e-6 # Add epsilon
        v_k_norm = np.clip(self.v_k / v_max_norm, 0.0, 1.0)
        a_j_norm = np.clip(self.a_j / a_max_norm, 0.0, 1.0)
        # d_r is already normalized [0, 1] during calculation
        return [float(v_k_norm), float(a_j_norm), float(self.d_r)]

    def get_shot_type(self):
        """Categorizes shot type based on face presence and size (Heuristic)."""
        if self.face_presence_ratio < 0.1: return 'wide/no_face'
        if self.avg_face_size < 0.15: return 'medium_wide'
        if self.avg_face_size < 0.35: return 'medium'
        return 'close_up'

    def __repr__(self):
        # Ensure source_video_path is a string for basename
        source_basename = os.path.basename(str(self.source_video_path)) if self.source_video_path else "N/A"
        return (f"ClipSegment({source_basename}, "
                f"Frames:[{self.start_frame}-{self.end_frame}], " # Show frames
                f"Time:[{self.start_time:.2f}s-{self.end_time:.2f}s], Dur:{self.duration:.2f}s)\n"
                f"  Heuristic: Score:{self.avg_boosted_score:.2f}, MotH:{self.avg_motion_heuristic:.1f}, Shot:{self.get_shot_type()}\n"
                f"  Physics: V:{self.v_k:.1f}, A:{self.a_j:.1f}, D:{self.d_r:.2f}, Phi:{self.phi:.2f}, Mood:{['{:.2f}'.format(x) for x in self.mood_vector]}")

# ========================================================================
#         MAIN ANALYSIS CLASS (VideousMain) - Uses AnalysisConfig
# ========================================================================
class VideousMain:
    def __init__(self):
        self.BBZImageUtils = BBZImageUtils()
        self.BBZPoseUtils = BBZPoseUtils()
        self.midas_model, self.midas_transform, self.midas_device = None, None, None

    def _ensure_midas_loaded(self, analysis_config: AnalysisConfig):
        needs_midas = (analysis_config.depth_weight > 0 and analysis_config.sequencing_mode == 'Greedy Heuristic') or \
                      (analysis_config.sequencing_mode == 'Physics Pareto MC' and
                       (analysis_config.continuity_depth_weight > 0 or analysis_config.fit_weight_mood > 0))

        if self.midas_model is None and needs_midas:
             logger.info("MiDaS model required by configuration, attempting load...")
             self.midas_model, self.midas_transform, self.midas_device = get_midas_model()
             if self.midas_model is None:
                 logger.warning("MiDaS failed to load. Depth features will be disabled.")

    # --- Heuristic Scoring Helpers ---
    def _determine_dominant_contributor(self, norm_features_weighted):
        if not norm_features_weighted: return "unknown"
        max_val = -float('inf'); dominant_key = "none"
        for key, value in norm_features_weighted.items():
            if value > max_val: max_val = value; dominant_key = key
        key_map = {'audio_energy': 'Audio', 'kinetic_proxy': 'Motion', 'jerk_proxy': 'Jerk',
                   'camera_motion': 'CamMove', 'face_size': 'FaceSize', 'percussive': 'Percuss',
                   'depth_variance': 'DepthVar'}
        # Return 'none' only if max contribution is very small
        return key_map.get(dominant_key, dominant_key) if max_val > 1e-4 else "none"

    def _categorize_intensity(self, score, thresholds=(0.3, 0.7)):
        if score < thresholds[0]: return "Low"
        if score < thresholds[1]: return "Medium"
        return "High"

    def calculate_heuristic_score(self, frame_features: Dict, analysis_config: AnalysisConfig) -> Tuple[float, str, str, Dict]:
        weights = { 'audio_energy': analysis_config.audio_weight, 'kinetic_proxy': analysis_config.kinetic_weight,
            'jerk_proxy': analysis_config.sharpness_weight, 'camera_motion': analysis_config.camera_motion_weight,
            'face_size': analysis_config.face_size_weight, 'percussive': analysis_config.percussive_weight,
            'depth_variance': analysis_config.depth_weight }
        norm_params = { 'rms': analysis_config.norm_max_rms + 1e-6, 'kinetic': analysis_config.norm_max_kinetic + 1e-6,
            'jerk': analysis_config.norm_max_jerk + 1e-6, 'cam_motion': analysis_config.norm_max_cam_motion + 1e-6,
            'face_size': analysis_config.norm_max_face_size + 1e-6, 'percussive_ratio': 1.0 + 1e-6, # Percussive usually 0-1
            'depth_variance': analysis_config.norm_max_depth_variance + 1e-6 }

        f = frame_features; pose_f = f.get('pose_features', {})
        # Safely get features, default to 0 if missing
        audio_energy = f.get('audio_energy', 0.0)
        kinetic_proxy = pose_f.get('kinetic_energy_proxy', 0.0)
        jerk_proxy = pose_f.get('movement_jerk_proxy', 0.0)
        camera_motion = f.get('camera_motion', 0.0)
        face_size_ratio = pose_f.get('face_size_ratio', 0.0)
        percussive_ratio = f.get('percussive_ratio', 0.0)
        depth_variance = f.get('depth_variance', 0.0)

        # Normalize features
        norm_audio = np.clip(audio_energy / norm_params['rms'], 0.0, 1.0)
        norm_kinetic = np.clip(kinetic_proxy / norm_params['kinetic'], 0.0, 1.0)
        norm_jerk = np.clip(jerk_proxy / norm_params['jerk'], 0.0, 1.0)
        norm_cam_motion = np.clip(camera_motion / norm_params['cam_motion'], 0.0, 1.0)
        norm_face_size = np.clip(face_size_ratio / norm_params['face_size'], 0.0, 1.0)
        norm_percussive = np.clip(percussive_ratio / norm_params['percussive_ratio'], 0.0, 1.0)
        norm_depth_var = np.clip(depth_variance / norm_params['depth_variance'], 0.0, 1.0)

        # Calculate weighted contributions
        contrib = {
            'audio_energy': norm_audio * weights['audio_energy'],
            'kinetic_proxy': norm_kinetic * weights['kinetic_proxy'],
            'jerk_proxy': norm_jerk * weights['jerk_proxy'],
            'camera_motion': norm_cam_motion * weights['camera_motion'],
            'face_size': norm_face_size * weights['face_size'],
            'percussive': norm_percussive * weights['percussive'],
            'depth_variance': norm_depth_var * weights['depth_variance']
        }
        # Filter out contributions where weight is zero
        weighted_contrib = {k: v for k, v in contrib.items() if weights.get(k, 0) > 1e-6}

        score = sum(weighted_contrib.values()); final_score = np.clip(score, 0.0, 1.0)
        dominant = self._determine_dominant_contributor(weighted_contrib)
        intensity = self._categorize_intensity(final_score, thresholds=analysis_config.intensity_thresholds)
        return float(final_score), dominant, intensity, weighted_contrib

    def apply_beat_boost(self, frame_features_list: List[Dict], audio_data: Dict, video_fps: float, analysis_config: AnalysisConfig):
        num_frames = len(frame_features_list)
        if num_frames == 0 or not audio_data or video_fps <= 0: return

        beat_boost = analysis_config.beat_boost
        boost_radius_sec = analysis_config.beat_boost_radius_sec
        boost_radius_frames = max(0, int(boost_radius_sec * video_fps))
        beat_times = audio_data.get('beat_times', [])
        if not beat_times and beat_boost > 0:
            logger.warning("Beat boost enabled, but no beat times found in audio data.")
            return

        boost_frame_indices = set()
        if beat_boost > 0:
            for t in beat_times:
                beat_frame_center = int(round(t * video_fps))
                # Apply boost in a window around the beat frame
                for r in range(-boost_radius_frames, boost_radius_frames + 1):
                    idx = beat_frame_center + r
                    if 0 <= idx < num_frames: boost_frame_indices.add(idx)

        # Apply boost to frame scores
        for i, features in enumerate(frame_features_list):
            if not isinstance(features, dict): continue # Skip non-dict items
            is_beat = i in boost_frame_indices
            features['is_beat_frame'] = is_beat
            boost = beat_boost if is_beat else 0.0
            raw_score = features.get('raw_score', 0.0)
            # Ensure boosted score doesn't exceed 1.0
            features['boosted_score'] = min(raw_score + boost, 1.0)

    def get_feature_at_time(self, times_array, values_array, target_time):
        """Interpolates or finds nearest feature value at a specific time."""
        if times_array is None or values_array is None or len(times_array) == 0 or len(times_array) != len(values_array):
            return 0.0
        try:
            # Use np.interp for linear interpolation (often smoother than nearest)
            # Ensure times_array is sorted for np.interp
            if not np.all(np.diff(times_array) >= 0):
                sort_indices = np.argsort(times_array)
                times_array = times_array[sort_indices]
                values_array = values_array[sort_indices]

            interpolated_value = np.interp(target_time, times_array, values_array, left=values_array[0], right=values_array[-1])
            return float(interpolated_value)

        except Exception as e:
             logger.error(f"Error in get_feature_at_time (time={target_time:.3f}): {e}", exc_info=True)
             # Fallback to nearest neighbour on error
             try:
                idx = np.searchsorted(times_array, target_time, side="left")
                if idx == 0: return float(values_array[0])
                if idx == len(times_array): return float(values_array[-1])
                # Choose the closer neighbour
                left_time = times_array[idx - 1]; right_time = times_array[idx]
                if abs(target_time - left_time) < abs(target_time - right_time):
                    return float(values_array[idx - 1])
                else:
                    return float(values_array[idx])
             except Exception as fallback_e:
                 logger.error(f"Fallback nearest neighbour failed in get_feature_at_time: {fallback_e}")
                 return 0.0


    # ============================================================ #
    #         analyzeVideo Method (Refactored for MoviePy)         #
    # ============================================================ #
    def analyzeVideo(self, videoPath: str, analysis_config: AnalysisConfig, audio_data: Dict) -> Tuple[Optional[List[Dict]], Optional[List[ClipSegment]]]:
        """ Analyzes video frames using MoviePy for reading. """
        logger.info(f"--- Analyzing Video Features (MoviePy Read): {os.path.basename(videoPath)} ---")
        start_time = time.time()
        self._ensure_midas_loaded(analysis_config)

        TARGET_HEIGHT = analysis_config.resolution_height # For analysis resize
        TARGET_WIDTH = analysis_config.resolution_width   # For analysis resize

        clip = None # MoviePy clip object
        all_frame_features = []
        pose_detector = None
        face_detector_util = None
        pose_results_buffer = [None, None, None] # For Jerk calculation [t-2, t-1, t]
        prev_gray = None
        prev_flow = None
        fps = 30.0 # Default FPS

        try:
            # === Step 1: Load with MoviePy and get properties ===
            logger.debug(f"Loading video with MoviePy: {videoPath}")
            clip = VideoFileClip(videoPath, audio=False) # Load without audio
            # Use clip.reader.fps if available and valid, otherwise clip.fps
            fps = clip.reader.fps if hasattr(clip, 'reader') and clip.reader and clip.reader.fps > 0 else (clip.fps if clip.fps and clip.fps > 0 else 30.0)
            if fps <= 0: fps = 30.0; logger.warning(f"Invalid FPS ({fps}) detected for {videoPath}, using default 30.")
            frame_time_diff = 1.0 / fps
            total_frames = int(clip.duration * fps) if clip.duration and clip.duration > 0 else 0
            logger.info(f"Video Properties (MoviePy): FPS={fps:.2f}, Approx Frames={total_frames}, Dur={clip.duration:.2f}s")
            if total_frames <= 0 or clip.duration <=0:
                raise ValueError(f"MoviePy clip has zero or negative duration/frames ({clip.duration}s, {total_frames} frames).")

            # === Step 2: Setup MediaPipe & Audio Refs ===
            logger.debug("Initializing MediaPipe FaceMesh...");
            face_detector_util = BBZFaceUtils(
                static_mode=False, max_faces=1,
                min_detect_conf=analysis_config.min_face_confidence, min_track_conf=0.5)
            if face_detector_util.face_mesh is None: logger.warning("FaceMesh failed. Face features disabled.")

            pose_context = None # Define context manager variable
            # Check if pose detection is needed based on config
            pose_needed = (analysis_config.kinetic_weight > 0 or
                           analysis_config.sharpness_weight > 0 or
                           analysis_config.face_size_weight > 0 or  # Heuristic needs pose for face size
                           analysis_config.sequencing_mode == "Physics Pareto MC") # Physics always needs pose features

            if pose_needed:
                logger.debug(f"Initializing MediaPipe Pose (Complexity: {analysis_config.model_complexity})...")
                pose_context = mp_pose.Pose(
                    static_image_mode=False, model_complexity=analysis_config.model_complexity,
                    min_detection_confidence=analysis_config.min_pose_confidence, min_tracking_confidence=0.5)
                pose_detector = pose_context.__enter__() # Manually enter context

            if not audio_data: logger.error("No master audio data provided for analysis. Aborting."); return None, None
            audio_raw_features = audio_data.get('raw_features', {})
            audio_rms_energy = np.asarray(audio_raw_features.get('rms_energy', []))
            audio_rms_times = np.asarray(audio_raw_features.get('rms_times', []))
            # Example: Add percussive feature handling if calculated in audio analysis
            audio_perc_ratio = np.asarray(audio_raw_features.get('percussive_ratio', []))
            audio_perc_times = np.asarray(audio_raw_features.get('perc_times', audio_rms_times)) # Use RMS times if specific perc times missing
            segment_boundaries = audio_data.get('segment_boundaries', [0, audio_data.get('duration', float('inf'))])
            if not segment_boundaries or len(segment_boundaries) < 2:
                 logger.warning("Invalid audio segment boundaries. Using [0, duration].")
                 segment_boundaries = [0, audio_data.get('duration', float('inf'))]


            # === Step 3: Feature Extraction Loop using MoviePy iter_frames ===
            try: from tqdm import tqdm as tqdm_analyzer
            except ImportError: tqdm_analyzer = lambda x, **kwargs: x; logger.info("tqdm not found, progress bar disabled.")

            logger.info("Processing frames & generating features (using MoviePy)...")
            frame_iterator = clip.iter_frames(fps=fps, dtype="uint8", logger=None) # logger=None suppresses MoviePy bar

            with tqdm_analyzer(total=total_frames if total_frames > 0 else None, desc=f"Analyzing {os.path.basename(videoPath)}", unit="frame", dynamic_ncols=True, leave=False) as pbar:
                for frame_idx, frame_rgb in enumerate(frame_iterator):
                    if frame_rgb is None:
                        logger.warning(f"Received None frame at index {frame_idx}. Stopping analysis.")
                        break

                    timestamp = frame_idx / fps
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # BGR for OpenCV

                    # Resize for analysis (expects BGR)
                    image_resized_bgr = self.BBZImageUtils.resizeTARGET(frame_bgr, TARGET_HEIGHT, TARGET_WIDTH)
                    if image_resized_bgr is None or image_resized_bgr.size == 0:
                        logger.warning(f"Frame {frame_idx}: Resize failed. Skipping."); pbar.update(1); continue

                    current_features = {'frame_index': frame_idx, 'timestamp': timestamp}
                    pose_features_dict = {}

                    # Grayscale for optical flow & entropy
                    current_gray = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2GRAY)

                    # Optical Flow Velocity & Acceleration
                    flow_velocity, current_flow_field = calculate_flow_velocity(prev_gray, current_gray)
                    flow_acceleration = calculate_flow_acceleration(prev_flow, current_flow_field, frame_time_diff)
                    current_features['flow_velocity'] = flow_velocity
                    current_features['flow_acceleration'] = flow_acceleration
                    current_features['camera_motion'] = flow_velocity # Alias for heuristic score

                    # Depth (requires RGB input for MiDaS)
                    depth_variance = 0.0
                    if self.midas_model and self.midas_transform and self.midas_device:
                        try:
                            with torch.no_grad():
                                image_resized_rgb = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB
                                input_batch = self.midas_transform(image_resized_rgb).to(self.midas_device)
                                prediction = self.midas_model(input_batch)
                                prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=image_resized_rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
                                depth_map = prediction.cpu().numpy()
                                depth_min = depth_map.min(); depth_max = depth_map.max()
                                if depth_max > depth_min + 1e-6: # Avoid division by zero/small numbers
                                     # Normalize depth map before variance calculation
                                     norm_depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                                     depth_variance = float(np.var(norm_depth_map))
                                else: depth_variance = 0.0
                        except Exception as midas_e: logger.warning(f"MiDaS failed frame {frame_idx}: {midas_e}", exc_info=False)
                    current_features['depth_variance'] = depth_variance

                    # Histogram Entropy (use grayscale)
                    current_features['histogram_entropy'] = calculate_histogram_entropy(current_gray)

                    # Face Features (expects BGR)
                    _face_size_ratio_fm = 0.0; _face_center_x_fm = 0.5 # Defaults
                    if face_detector_util and face_detector_util.face_mesh:
                        is_mouth_open, face_size_ratio_fm, face_center_x_fm = face_detector_util.get_face_features(image_resized_bgr, analysis_config.mouth_open_threshold)
                        pose_features_dict['is_mouth_open'] = is_mouth_open
                        _face_size_ratio_fm = face_size_ratio_fm # Store temporary results
                        _face_center_x_fm = face_center_x_fm
                    else:
                        pose_features_dict['is_mouth_open'] = False

                    # Pose Features (expects RGB)
                    current_pose_results = None
                    if pose_detector:
                        try:
                            image_resized_rgb_pose = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB) # RGB for Pose
                            image_resized_rgb_pose.flags.writeable = False
                            current_pose_results = pose_detector.process(image_resized_rgb_pose)
                            image_resized_rgb_pose.flags.writeable = True
                        except Exception as pose_err: logger.warning(f"Pose failed frame {frame_idx}: {pose_err}", exc_info=False)

                    # Update buffer [t-2, t-1, t]
                    pose_results_buffer.pop(0); pose_results_buffer.append(current_pose_results)
                    lm_t2, lm_t1, lm_t = ( (res.pose_landmarks if res else None) for res in pose_results_buffer )

                    # Calculate Pose-based features if landmarks available
                    kinetic = calculate_kinetic_energy_proxy(lm_t1, lm_t, frame_time_diff)
                    jerk = calculate_movement_jerk_proxy(lm_t2, lm_t1, lm_t, frame_time_diff)
                    pose_face_size = 0.0; pose_face_center = 0.5

                    if pose_needed and lm_t: # Estimate face size/center from pose
                        try:
                            # Use nose landmark (index 0) and shoulder landmarks (11, 12) for rough scale/position
                            nose = lm_t.landmark[0]
                            left_shoulder = lm_t.landmark[11]; right_shoulder = lm_t.landmark[12]
                            if nose.visibility > 0.1 and left_shoulder.visibility > 0.1 and right_shoulder.visibility > 0.1:
                                shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                                # Estimate face width relative to shoulder width (very rough)
                                face_width_estimate = shoulder_width * 0.5 # Empirical guess
                                pose_face_size = np.clip(face_width_estimate, 0.0, 1.0) # Normalize roughly
                                pose_face_center = np.clip(nose.x, 0.0, 1.0)
                        except IndexError: logger.debug("Pose landmarks missing for face size estimate.")
                        except Exception as pose_face_err: logger.debug(f"Error estimating face size from pose: {pose_face_err}")

                    pose_features_dict['kinetic_energy_proxy'] = kinetic
                    pose_features_dict['movement_jerk_proxy'] = jerk
                    # Use Pose face data if pose was enabled & detected, otherwise use FaceMesh data
                    pose_features_dict['face_size_ratio'] = pose_face_size if pose_needed and pose_face_size > 0 else _face_size_ratio_fm
                    pose_features_dict['face_center_x'] = pose_face_center if pose_needed and pose_face_size > 0 else _face_center_x_fm

                    current_features['pose_features'] = pose_features_dict

                    # Align with Audio Features (use mid-frame time for better alignment)
                    mid_frame_time = timestamp + (frame_time_diff / 2.0)
                    current_features['audio_energy'] = self.get_feature_at_time(audio_rms_times, audio_rms_energy, mid_frame_time)
                    current_features['percussive_ratio'] = self.get_feature_at_time(audio_perc_times, audio_perc_ratio, mid_frame_time)

                    # Determine Musical Section Index
                    section_idx = -1
                    for i in range(len(segment_boundaries) - 1):
                        if segment_boundaries[i] <= mid_frame_time < segment_boundaries[i+1]:
                            section_idx = i; break
                    current_features['musical_section_index'] = section_idx

                    # Calculate Heuristic Score & related features
                    heuristic_score, dominant, intensity, _ = self.calculate_heuristic_score(current_features, analysis_config)
                    current_features['raw_score'] = heuristic_score; current_features['dominant_contributor'] = dominant
                    current_features['intensity_category'] = intensity; current_features['boosted_score'] = heuristic_score # Initial boosted = raw
                    current_features['is_beat_frame'] = False # Will be set later by apply_beat_boost

                    all_frame_features.append(current_features)

                    # Update previous states for next iteration
                    prev_gray = current_gray.copy() # Important: copy!
                    prev_flow = current_flow_field.copy() if current_flow_field is not None else None

                    if pbar: pbar.update(1) # Update progress bar

        except ValueError as ve: # Catch specific ValueErrors like zero duration
             logger.error(f"ValueError during analysis setup for {videoPath}: {ve}")
             return None, None
        except Exception as e:
            logger.error(f"Error during MoviePy video analysis loop for {videoPath}: {e}", exc_info=True)
            return None, None # Indicate failure
        finally:
            # === Step 4: Cleanup ===
            if pose_context:
                try: pose_context.__exit__(None, None, None); logger.debug("Pose context exited.")
                except Exception as pose_close_err: logger.error(f"Error closing Pose context: {pose_close_err}")
            if face_detector_util: face_detector_util.close()
            if clip:
                try: clip.close(); logger.debug("MoviePy clip closed.")
                except Exception as clip_close_err: logger.error(f"Error closing MoviePy clip: {clip_close_err}")
            logger.debug(f"Feature extraction for {os.path.basename(videoPath)} done ({len(all_frame_features)} frames). Resources released.")

        # === Step 5: Post-processing & Clip Identification ===
        if not all_frame_features:
            logger.error(f"No features extracted for {videoPath}. Analysis failed.")
            return None, None

        logger.debug("Applying beat boost (for heuristic score)...")
        self.apply_beat_boost(all_frame_features, audio_data, fps, analysis_config) # Use actual fps

        potential_clips: List[ClipSegment] = []
        actual_total_frames = len(all_frame_features) # Use actual number processed
        if analysis_config.use_heuristic_segment_id:
             logger.debug("Identifying potential segments using heuristic score...")
             segment_identifier = DynamicSegmentIdentifier(analysis_config, fps) # Use actual fps
             potential_segment_indices = segment_identifier.find_potential_segments(all_frame_features)
             logger.info(f"Creating {len(potential_segment_indices)} ClipSegments from heuristic runs...")
             for seg_indices in potential_segment_indices:
                 start_f = seg_indices['start_frame']; end_f = seg_indices['end_frame']
                 # Add stricter validation for segment indices
                 if 0 <= start_f < end_f <= actual_total_frames:
                    try:
                        clip_seg = ClipSegment(videoPath, start_f, end_f, fps, all_frame_features, analysis_config)
                        # Filter by duration after creation
                        if clip_seg.duration >= analysis_config.min_potential_clip_duration_sec:
                            potential_clips.append(clip_seg)
                        else:
                            logger.debug(f"Skipping short heuristic segment {start_f}-{end_f}: {clip_seg.duration:.2f}s (Min: {analysis_config.min_potential_clip_duration_sec:.2f}s)")
                    except Exception as clip_err: logger.warning(f"Failed ClipSegment creation {start_f}-{end_f}: {clip_err}", exc_info=False)
                 else: logger.warning(f"Invalid segment indices {start_f}-{end_f} from DynamicSegmentIdentifier. Total frames: {actual_total_frames}. Skipping.")
        else:
            # Fallback: Create potential segments using fixed/overlapping chunks
            logger.info("Creating potential segments using fixed/overlapping chunks (heuristic ID disabled)...")
            min_clip_frames = max(1, int(analysis_config.min_potential_clip_duration_sec * fps))
            max_clip_frames = max(min_clip_frames, int(analysis_config.max_sequence_clip_duration * fps))
            step_frames = max(1, int(1.0 * fps)) # Step ~1 second
            if actual_total_frames < min_clip_frames:
                logger.warning(f"Video too short ({actual_total_frames} frames) for chunking with min clip length {min_clip_frames} frames.")
            else:
                for start_f in range(0, actual_total_frames - min_clip_frames + 1, step_frames):
                     end_f = min(start_f + max_clip_frames, actual_total_frames) # Clip end cannot exceed video length
                     if (end_f - start_f) >= min_clip_frames:
                         try:
                             clip_seg = ClipSegment(videoPath, start_f, end_f, fps, all_frame_features, analysis_config)
                             potential_clips.append(clip_seg)
                         except Exception as clip_err: logger.warning(f"Failed ClipSegment chunk creation {start_f}-{end_f}: {clip_err}", exc_info=False)

        end_time = time.time()
        logger.info(f"--- Analysis & Clip Creation complete for {os.path.basename(videoPath)} ({end_time - start_time:.2f}s) ---")
        logger.info(f"Created {len(potential_clips)} potential clips.")
        return all_frame_features, potential_clips

    def saveAnalysisData(self, video_path: str, frame_features: List[Dict], potential_clips: List[ClipSegment], output_dir: str, analysis_config: AnalysisConfig):
        """Saves detailed frame features and potential clips (ClipSegment data)."""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_subdir = os.path.join(output_dir) # Save directly in analysis_subdir passed from worker
        features_path = os.path.join(out_subdir, f"{base_name}_frame_features.json")
        clips_path = os.path.join(out_subdir, f"{base_name}_potential_clips.json")
        logger.info(f"Saving analysis data for {base_name}...")

        def sanitize_for_json(value):
            """Recursively sanitize data for JSON serialization."""
            if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, (np.complex_, np.complex64, np.complex128)):
                return {'real': float(value.real), 'imag': float(value.imag)}
            elif isinstance(value, (np.ndarray,)):
                return [sanitize_for_json(item) for item in value.tolist()] # Sanitize list elements too
            elif isinstance(value, (np.bool_)):
                return bool(value)
            elif isinstance(value, (np.void)): # Cannot be serialized
                logger.debug("Omitting non-serializable numpy void type.")
                return None
            elif isinstance(value, set):
                 return [sanitize_for_json(item) for item in list(value)] # Sanitize list elements
            elif isinstance(value, dict):
                 return {str(k): sanitize_for_json(v) for k, v in value.items()} # Ensure keys are strings
            elif isinstance(value, (list, tuple)):
                 return [sanitize_for_json(item) for item in value]
            # Basic types are fine
            elif isinstance(value, (int, float, str, bool)) or value is None:
                return value
            # Attempt to serialize unknown types, return string rep if fail
            else:
                try:
                    json.dumps(value) # Test serialization
                    return value
                except TypeError:
                    logger.debug(f"Omitting non-serializable type {type(value)} during save. Repr: {repr(value)[:50]}...")
                    return None # Return None for non-serializable types

        try:
            os.makedirs(out_subdir, exist_ok=True)
            logger.debug(f"Saving {len(frame_features)} frame features to {features_path}")
            # Sanitize the entire features list structure
            sanitized_features = sanitize_for_json(frame_features)
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_features, f, indent=1, ensure_ascii=False) # Less verbose indent

            logger.debug(f"Saving {len(potential_clips)} potential clips to {clips_path}")
            clips_data = []
            serializable_attrs = [ # Define attributes to save
                 'source_video_path', 'start_frame', 'end_frame', 'num_frames', 'fps',
                 'start_time', 'end_time', 'duration', 'avg_raw_score', 'avg_boosted_score',
                 'peak_boosted_score', 'avg_motion_heuristic', 'avg_jerk_heuristic',
                 'avg_camera_motion', 'face_presence_ratio', 'avg_face_size',
                 'intensity_category', 'dominant_contributor', 'contains_beat',
                 'musical_section_indices', 'v_k', 'a_j', 'd_r', 'phi', 'mood_vector',
                 # Runtime variables might not be needed in saved data, but can include if useful
                 # 'sequence_start_time', 'sequence_end_time', 'chosen_duration',
                 # 'subclip_start_time_in_source', 'subclip_end_time_in_source'
            ]
            for clip in potential_clips:
                 clip_dict = {}
                 for attr in serializable_attrs:
                     if hasattr(clip, attr):
                         val = getattr(clip, attr)
                         # Special handling for path and sets
                         if attr == 'source_video_path':
                              clip_dict[attr] = os.path.basename(str(val)) if val else "N/A"
                         else:
                              clip_dict[attr] = sanitize_for_json(val) # Sanitize each attribute
                 clips_data.append(clip_dict)

            # No need to sanitize clips_data again if individual attributes were sanitized
            with open(clips_path, 'w', encoding='utf-8') as f:
                json.dump(clips_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved analysis data to {os.path.basename(features_path)} and {os.path.basename(clips_path)}")
        except TypeError as te:
            logger.error(f"TYPE ERROR saving analysis data for {base_name}: {te}. Check data sanitization.", exc_info=True)
        except Exception as e:
            logger.error(f"ERROR saving analysis data for {base_name}: {e}", exc_info=True)


# ========================================================================
#          SEQUENCE BUILDER - GREEDY HEURISTIC (Uses AnalysisConfig)
# ========================================================================
class SequenceBuilderGreedy:
    def __init__(self, all_potential_clips: List[ClipSegment], audio_data: Dict, analysis_config: AnalysisConfig):
        self.all_clips = all_potential_clips; self.audio_data = audio_data; self.analysis_config = analysis_config
        segment_boundaries_raw = audio_data.get('segment_boundaries', [])
        aud_dur = audio_data.get('duration', 0)
        self.segment_boundaries = segment_boundaries_raw if segment_boundaries_raw and len(segment_boundaries_raw) >= 2 else ([0, aud_dur] if aud_dur > 0 else [0])
        if len(self.segment_boundaries) < 2: logger.warning(f"Using default [0, {aud_dur:.2f}s] audio boundaries.")
        self.beat_times = audio_data.get('beat_times', []); self.target_duration = aud_dur
        self.final_sequence: List[ClipSegment] = []; self.current_time = 0.0; self.last_clip_used: Optional[ClipSegment] = None
        # Get parameters from config
        self.pacing_variation = analysis_config.pacing_variation_factor; self.variety_penalty_source = analysis_config.variety_penalty_source
        self.variety_penalty_shot = analysis_config.variety_penalty_shot; self.variety_penalty_intensity = analysis_config.variety_penalty_intensity
        self.beat_sync_bonus = analysis_config.beat_sync_bonus; self.section_match_bonus = analysis_config.section_match_bonus
        self.min_clip_duration = analysis_config.min_sequence_clip_duration; self.max_clip_duration = analysis_config.max_sequence_clip_duration
        self.candidate_pool_size = analysis_config.candidate_pool_size

    def build_sequence(self) -> List[ClipSegment]:
        logger.info("--- Composing Sequence (Greedy Heuristic Mode) ---")
        if not self.all_clips: logger.warning("No potential clips for Greedy mode."); return []
        if self.target_duration <= 0: logger.warning("Target duration zero. Cannot build sequence."); return []
        if len(self.segment_boundaries) < 2: logger.warning("Insufficient audio boundaries."); return []

        # Pre-filter clips to ensure they meet min duration for sequence use
        eligible_clips_initial = [c for c in self.all_clips if c.duration >= self.min_clip_duration]
        if not eligible_clips_initial:
            logger.warning("No potential clips meet the minimum sequence duration criteria."); return []
        logger.info(f"Starting with {len(eligible_clips_initial)} clips meeting min duration {self.min_clip_duration:.2f}s.")


        clips_by_section = defaultdict(list)
        for clip in eligible_clips_initial:
            clip_mid_time = clip.start_time + clip.duration / 2.0
            for i in range(len(self.segment_boundaries) - 1):
                 # Assign clip to section if its midpoint falls within the section boundaries
                 if self.segment_boundaries[i] <= clip_mid_time < self.segment_boundaries[i+1]:
                      clips_by_section[i].append(clip); break
            # Handle clips potentially falling exactly on the last boundary or after
            if clip_mid_time >= self.segment_boundaries[-1] and len(self.segment_boundaries) > 1:
                 clips_by_section[len(self.segment_boundaries) - 2].append(clip) # Assign to last section

        available_clips = eligible_clips_initial.copy(); num_sections = len(self.segment_boundaries) - 1

        for section_idx in range(num_sections):
            section_start = self.segment_boundaries[section_idx]; section_end = self.segment_boundaries[section_idx+1]
            section_duration = section_end - section_start
            if section_duration <= 0: continue

            target_section_fill = max(0, section_end - self.current_time)
            # Only attempt to fill if remaining time is somewhat substantial
            if target_section_fill < self.min_clip_duration * 0.25:
                 logger.debug(f"Skipping section {section_idx}: Only {target_section_fill:.2f}s left to fill (less than 25% of min clip).")
                 continue

            section_context = self._get_section_context(section_idx, section_start, section_end)
            logger.info(f"Serving Course {section_idx} ({section_start:.2f}s-{section_end:.2f}s, TargetFill:{target_section_fill:.2f}s)")
            logger.info(f"  Theme: Energy={section_context['avg_energy']:.3f}, TargetClipDur={section_context['target_clip_dur']:.2f}s")
            section_filled_time = 0.0
            section_primary_candidates = clips_by_section.get(section_idx, [])
            # Candidate pool: prioritize clips whose midpoint is in this section, then add all other available clips
            # Use dict.fromkeys to maintain order while removing duplicates
            candidate_pool = list(dict.fromkeys(section_primary_candidates + available_clips))

            while self.current_time < section_end and candidate_pool:
                best_candidate = None; max_selection_score = -float('inf')
                # Filter candidate_pool again to ensure clips are still globally available
                eligible_candidates_in_pool = [c for c in candidate_pool if c in available_clips]
                if not eligible_candidates_in_pool:
                     logger.debug(f"  No eligible candidates remaining in pool for section {section_idx}.")
                     break

                # Sort eligible candidates by their boosted score (pre-calculated heuristic value)
                potential_candidates_sorted = sorted(eligible_candidates_in_pool, key=lambda c: c.avg_boosted_score, reverse=True)

                # Evaluate top N candidates from the sorted list
                num_to_evaluate = min(self.candidate_pool_size, len(potential_candidates_sorted))
                considered_candidates = []
                for candidate in potential_candidates_sorted[:num_to_evaluate]:
                    selection_score, potential_duration = self._calculate_selection_score(candidate, section_context)
                    candidate.temp_chosen_duration = potential_duration # Store potential duration for this context
                    considered_candidates.append((candidate, selection_score))

                # Find the best candidate among those considered
                if considered_candidates:
                     best_candidate_tuple = max(considered_candidates, key=lambda item: item[1])
                     best_candidate = best_candidate_tuple[0]
                     max_selection_score = best_candidate_tuple[1]
                else: # Should not happen if eligible_candidates_in_pool was not empty
                     logger.warning("Considered candidates list empty despite eligible candidates. Skipping section fill.")
                     break

                if best_candidate:
                    # Final duration calculation for the chosen clip
                    final_chosen_duration = best_candidate.temp_chosen_duration
                    # Apply constraints: min/max length, clip's own duration, remaining section time
                    final_chosen_duration = np.clip(final_chosen_duration, self.min_clip_duration, self.max_clip_duration)
                    final_chosen_duration = min(final_chosen_duration, best_candidate.duration) # Cannot be longer than original potential clip
                    final_chosen_duration = min(final_chosen_duration, max(0.01, section_end - self.current_time)) # Don't overshoot section
                    final_chosen_duration = max(0.01, final_chosen_duration) # Ensure positive duration

                    # Check if the final duration is too small to be useful
                    # Allow slightly shorter clips if it completes the section
                    min_useful_duration = self.min_clip_duration * 0.5
                    is_last_clip_in_section = (section_end - self.current_time) <= final_chosen_duration + 1e-3 # Check if this clip fills the section
                    if final_chosen_duration < min_useful_duration and not is_last_clip_in_section:
                         logger.debug(f"  Skipping clip {os.path.basename(str(best_candidate.source_video_path))} - duration too short ({final_chosen_duration:.2f}s) and doesn't fill section")
                         # Remove this specific candidate from the current section's pool to avoid re-selecting it immediately
                         candidate_pool.remove(best_candidate)
                         continue # Try the next best candidate in the pool

                    # --- Set final properties on the chosen clip ---
                    best_candidate.sequence_start_time = self.current_time
                    best_candidate.sequence_end_time = self.current_time + final_chosen_duration
                    best_candidate.chosen_duration = final_chosen_duration
                    # Subclip starts from the beginning of the potential ClipSegment
                    best_candidate.subclip_start_time_in_source = best_candidate.start_time
                    best_candidate.subclip_end_time_in_source = best_candidate.start_time + final_chosen_duration
                    best_candidate.chosen_effect = EffectParams(type="cut") # Greedy only uses cuts

                    self.final_sequence.append(best_candidate)
                    clip_basename = os.path.basename(str(best_candidate.source_video_path)) if best_candidate.source_video_path else "N/A"
                    logger.info(f"  + Added Clip (Greedy): {clip_basename} "
                                f"Src:({best_candidate.subclip_start_time_in_source:.2f}-{best_candidate.subclip_end_time_in_source:.2f}) "
                                f"Seq:({best_candidate.sequence_start_time:.2f}-{best_candidate.sequence_end_time:.2f}) "
                                f"Score: {max_selection_score:.3f}, Dur: {best_candidate.chosen_duration:.2f}s")

                    self.current_time += best_candidate.chosen_duration
                    section_filled_time += best_candidate.chosen_duration
                    self.last_clip_used = best_candidate

                    # Remove chosen clip from the global pool of available clips
                    if best_candidate in available_clips:
                         available_clips.remove(best_candidate)
                    # Also remove from the current section's candidate pool
                    if best_candidate in candidate_pool:
                         candidate_pool.remove(best_candidate)
                else:
                    # This condition implies considered_candidates was empty or max score was -inf
                    logger.info(f"  - No suitable candidate found for section {section_idx} at time {self.current_time:.2f}s. Moving on.")
                    break # Stop trying for this section

            logger.info(f" Filled {section_filled_time:.2f}s for section {section_idx}. Current time: {self.current_time:.2f}s")
            # No time jump needed here - let the loop naturally progress

        # Check if the sequence is completely empty
        if not self.final_sequence:
            logger.error("Greedy sequence building resulted in an empty sequence.")
            return []

        final_duration = self.final_sequence[-1].sequence_end_time
        logger.info("--- Sequence Composition Complete (Greedy Heuristic) ---")
        logger.info(f"Total Duration: {final_duration:.2f}s (Target: {self.target_duration:.2f}s), Clips: {len(self.final_sequence)}")
        return self.final_sequence

    def _get_section_context(self, section_idx, start_time, end_time):
        context = {}
        norm_max_rms = self.analysis_config.norm_max_rms + 1e-6
        # Get average audio energy for the section
        context['avg_energy'] = self._get_avg_audio_feature_in_range('rms_energy', start_time, end_time)
        norm_energy = np.clip(context['avg_energy'] / norm_max_rms, 0.0, 1.0)

        # Calculate target clip duration: Inverse relationship with energy + pacing variation
        # Base duration leans towards max for low energy, min for high energy
        base_target_dur = self.min_clip_duration + (1.0 - norm_energy) * (self.max_clip_duration - self.min_clip_duration)
        # Apply random variation around the base target
        variation_amount = base_target_dur * self.pacing_variation
        random_factor = random.uniform(-variation_amount, variation_amount)
        context['target_clip_dur'] = base_target_dur + random_factor

        # Clamp target duration within sequence limits
        context['target_clip_dur'] = np.clip(context['target_clip_dur'], self.min_clip_duration, self.max_clip_duration)
        context['start'] = start_time; context['end'] = end_time
        return context

    def _calculate_selection_score(self, candidate_clip: ClipSegment, section_context: Dict) -> Tuple[float, float]:
        # Base score: Use the clip's pre-calculated boosted score
        score = candidate_clip.avg_boosted_score

        # --- Variety Penalties (apply if not the first clip) ---
        if self.last_clip_used:
            cand_path = str(candidate_clip.source_video_path) if candidate_clip.source_video_path else None
            last_path = str(self.last_clip_used.source_video_path) if self.last_clip_used.source_video_path else None
            # Penalty for using the same source video consecutively
            if cand_path and last_path and cand_path == last_path:
                score -= self.variety_penalty_source

            # Penalty for using the same shot type consecutively (excluding wide/no face)
            shot_type = candidate_clip.get_shot_type()
            last_shot_type = self.last_clip_used.get_shot_type()
            if shot_type == last_shot_type and shot_type != 'wide/no_face':
                score -= self.variety_penalty_shot

            # Penalty for using the same intensity category consecutively
            if candidate_clip.intensity_category == self.last_clip_used.intensity_category:
                score -= self.variety_penalty_intensity

        # --- Calculate Potential Duration for this Clip ---
        target_dur = section_context['target_clip_dur']
        # Blend target duration (from audio context) with clip's inherent duration (based on analysis)
        # Pacing variation factor controls the blend
        potential_duration = (target_dur * (1.0 - self.pacing_variation)) + (candidate_clip.duration * self.pacing_variation)
        # Clamp within sequence limits and clip's actual max duration
        potential_duration = np.clip(potential_duration, self.min_clip_duration, self.max_clip_duration)
        potential_duration = min(potential_duration, candidate_clip.duration)
        potential_duration = max(0.01, potential_duration) # Ensure positive

        # --- Bonuses ---
        # Beat Sync Bonus: Check if the *potential* cut time aligns with a beat
        potential_cut_time = self.current_time + potential_duration
        if self._is_near_beat(potential_cut_time, tolerance=self.analysis_config.beat_boost_radius_sec):
            score += self.beat_sync_bonus

        # Section Mood Match Bonus (Simplified Heuristic Version)
        norm_max_rms = self.analysis_config.norm_max_rms + 1e-6
        norm_max_kinetic = self.analysis_config.norm_max_kinetic + 1e-6
        norm_section_energy = np.clip(section_context['avg_energy'] / norm_max_rms, 0.0, 1.0)
        # Use the clip's aggregated motion heuristic value
        norm_clip_motion = np.clip(candidate_clip.avg_motion_heuristic / norm_max_kinetic, 0.0, 1.0)

        # Add bonus if both audio section and clip motion are high energy
        if norm_section_energy > 0.6 and norm_clip_motion > 0.6:
            score += self.section_match_bonus * 0.5
        # Add bonus if audio section is low energy and clip is a close-up (suggests calmer shot)
        if norm_section_energy < 0.3 and candidate_clip.get_shot_type() == 'close_up':
            score += self.section_match_bonus

        # Return the final score and the calculated potential duration for this clip in this context
        return score, potential_duration

    def _get_avg_audio_feature_in_range(self, feature_name, start_time, end_time):
        audio_raw = self.audio_data.get('raw_features', {})
        # Construct keys for time and value arrays
        times_key = f'{feature_name}_times' if feature_name != 'rms_energy' else 'rms_times'
        values_key = feature_name
        times = np.asarray(audio_raw.get(times_key, [])); values = np.asarray(audio_raw.get(values_key, []))

        if len(times) == 0 or len(values) == 0 or len(times) != len(values):
            logger.debug(f"Feature '{feature_name}' missing or mismatched times/values. Returning 0.")
            return 0.0

        # Find indices corresponding to the time range
        start_idx = np.searchsorted(times, start_time, side='left')
        end_idx = np.searchsorted(times, end_time, side='right')

        # Handle empty range or out-of-bounds
        if start_idx >= end_idx:
            # If range is empty, find the single closest value
            mid_time = (start_time + end_time) / 2.0
            closest_idx = np.argmin(np.abs(times - mid_time))
            safe_idx = min(max(0, closest_idx), len(values) - 1) # Clamp index
            logger.debug(f"No samples for '{feature_name}' in range [{start_time:.2f}-{end_time:.2f}], using closest value at index {safe_idx}.")
            return float(values[safe_idx]) if len(values) > 0 else 0.0

        # Calculate mean of values within the range
        section_values = values[start_idx:end_idx]
        # Use nanmean for robustness if NaNs are possible in audio features
        mean_val = np.nanmean(section_values) if len(section_values) > 0 else 0.0
        return float(mean_val) if np.isfinite(mean_val) else 0.0 # Ensure finite float return


    def _is_near_beat(self, time_sec, tolerance=0.1):
        if not self.beat_times: return False
        beat_times_arr = np.asarray(self.beat_times)
        if len(beat_times_arr) == 0: return False
        # Calculate minimum absolute difference to any beat time
        min_diff = np.min(np.abs(beat_times_arr - time_sec))
        return min_diff <= tolerance

# ========================================================================
#          SEQUENCE BUILDER - PHYSICS PARETO MC (Uses AnalysisConfig)
# ========================================================================
class SequenceBuilderPhysicsMC:
    def __init__(self, all_potential_clips: List[ClipSegment], audio_data: Dict, analysis_config: AnalysisConfig):
        self.all_clips = all_potential_clips; self.audio_data = audio_data; self.analysis_config = analysis_config
        self.target_duration = audio_data.get('duration', 0); self.beat_times = audio_data.get('beat_times', [])
        self.audio_segments = audio_data.get('segment_features', [])
        # Get parameters from config
        self.mc_iterations = analysis_config.mc_iterations; self.min_clip_duration = analysis_config.min_sequence_clip_duration
        self.max_clip_duration = analysis_config.max_sequence_clip_duration; self.w_r = analysis_config.objective_weight_rhythm
        self.w_m = analysis_config.objective_weight_mood; self.w_c = analysis_config.objective_weight_continuity
        self.w_v = analysis_config.objective_weight_variety; self.w_e = analysis_config.objective_weight_efficiency
        self.tempo = audio_data.get('tempo', 120.0); self.beat_period = 60.0 / self.tempo if self.tempo > 0 else 0.5
        # Effects will be set from RenderConfig by the caller (_build_final_sequence_and_video)
        self.effects: Dict[str, EffectParams] = {} # Placeholder, gets populated later

    def get_audio_segment_at(self, time):
        if not self.audio_segments: return None
        for seg in self.audio_segments:
            # Check if time falls within the segment [start, end)
            if seg['start'] <= time < seg['end']: return seg
        # Handle time exactly at or past the end of the last segment
        if self.audio_segments:
            last_seg = self.audio_segments[-1]
            # Allow matching the last segment if time is exactly its end or slightly past (within epsilon)
            if time >= last_seg['start'] and time <= last_seg['end'] + 1e-6:
                 return last_seg
            # If time is beyond the last segment, return the last segment
            if time > last_seg['end']:
                logger.debug(f"Time {time:.2f}s is past last audio segment ({last_seg['end']:.2f}s), using last segment.")
                return last_seg

        logger.warning(f"Time {time:.2f}s outside audio segment boundaries [{self.audio_segments[0]['start'] if self.audio_segments else '?'}-{self.audio_segments[-1]['end'] if self.audio_segments else '?'}].")
        return None # Return None if truly before the first segment or other issues


    def build_sequence(self) -> List[ClipSegment]:
        # Pareto MC explanation integrated into info log
        logger.info(f"--- Composing Sequence (Physics Pareto MC Mode - {self.mc_iterations} iterations) ---")
        logger.info("Seeking balance between Rhythm(R), Mood(M), Continuity(C), Variety(V), Efficiency(EC) via Pareto front.")
        if not self.all_clips: logger.warning("No clips for Physics MC."); return []
        if not self.audio_segments: logger.warning("No audio segments for Physics MC."); return []
        if self.target_duration <= 0: logger.warning("Target duration zero."); return []
        if not self.effects: logger.warning("Effects dictionary not populated in Physics MC Builder!"); self.effects = RenderConfig().effect_settings # Use defaults as fallback

        # Filter clips initially based on min duration
        eligible_clips_initial = [c for c in self.all_clips if c.duration >= self.min_clip_duration]
        if not eligible_clips_initial:
            logger.warning(f"No potential clips meet min duration {self.min_clip_duration:.2f}s for Physics MC."); return []
        logger.info(f"Starting Physics MC with {len(eligible_clips_initial)} eligible clips.")
        self.all_clips = eligible_clips_initial # Use only eligible clips for building

        pareto_front: List[Tuple[List[Tuple[ClipSegment, float, EffectParams]], List[float]]] = []
        try: from tqdm import trange as trange_mc
        except ImportError: trange_mc = range; logger.info("tqdm not found, MC progress disabled.")

        logger.info(f"Running {self.mc_iterations} Monte Carlo simulations...")
        successful_sims = 0
        with trange_mc(self.mc_iterations, desc="MC Simulations", leave=False) as mc_iter_range:
            for i in mc_iter_range:
                try:
                    sim_seq_info = self._run_stochastic_build()
                    if sim_seq_info:
                        successful_sims += 1
                        scores = self._evaluate_pareto(sim_seq_info)
                        # Check for invalid scores before updating front
                        if all(np.isfinite(s) for s in scores):
                            self._update_pareto_front(pareto_front, (sim_seq_info, scores))
                        else:
                            logger.warning(f"MC iter {i}: Invalid scores calculated {scores}, discarding solution.")
                except Exception as mc_err: logger.error(f"Error in MC iteration {i}: {mc_err}", exc_info=False) # Less verbose traceback in loop

        logger.info(f"Completed {self.mc_iterations} simulations, {successful_sims} yielded sequences.")

        if not pareto_front:
             logger.error("MC simulation yielded no valid Pareto front sequences.")
             # tk_write(...) called by caller if build_sequence returns empty
             return []
        logger.info(f"Found {len(pareto_front)} non-dominated sequences on the Pareto front.")

        # Select the best solution from the Pareto front
        if len(pareto_front) == 1:
             best_solution = pareto_front[0]; logger.info("Only one non-dominated solution found.")
        else:
            # Define selection metric using configured objective weights
            def selection_metric(item):
                scores = item[1] # [neg_r, M, C, V, neg_ec]
                # Use weights from AnalysisConfig
                metric_score = (scores[0] * self.analysis_config.objective_weight_rhythm + # neg_r * w_r
                                scores[1] * self.analysis_config.objective_weight_mood +   # M * w_m
                                scores[2] * self.analysis_config.objective_weight_continuity + # C * w_c
                                scores[3] * self.analysis_config.objective_weight_variety +    # V * w_v
                                scores[4] * self.analysis_config.objective_weight_efficiency) # neg_ec * w_e
                return metric_score

            best_solution = max(pareto_front, key=selection_metric)
            logger.info("Selected best sequence from Pareto front using weighted objective metric.")

        # Log objectives of the chosen sequence (R*, M, C, V, EC*) - Note R and EC are negative scores
        log_scores = [f"{s:.3f}" for s in best_solution[1]]
        logger.info(f"Chosen sequence objectives (NegR, M, C, V, NegEC): [{', '.join(log_scores)}]")
        final_sequence_info = best_solution[0]

        # Reconstruct final sequence with ClipSegment objects and timing info
        final_sequence_segments: List[ClipSegment] = []; current_t = 0.0
        for i, (clip, duration, effect) in enumerate(final_sequence_info):
            if not isinstance(clip, ClipSegment):
                 logger.error(f"Invalid item {i} in final seq info: {type(clip)}. Skipping.")
                 continue
            clip.sequence_start_time = current_t
            clip.sequence_end_time = current_t + duration
            clip.chosen_duration = duration
            clip.chosen_effect = effect
            # Ensure subclip times are valid within the original clip's boundaries
            clip.subclip_start_time_in_source = clip.start_time # Start from beginning of potential clip
            clip.subclip_end_time_in_source = min(clip.start_time + duration, clip.end_time) # Ensure end doesn't exceed original end
            final_sequence_segments.append(clip)
            current_t += duration

        if not final_sequence_segments:
             logger.error("Physics MC failed to construct a final sequence list.")
             return []

        logger.info("--- Sequence Composition Complete (Physics Pareto MC) ---")
        logger.info(f"Final Duration: {current_t:.2f}s (Target: {self.target_duration:.2f}s), Clips: {len(final_sequence_segments)}")
        return final_sequence_segments

    def _run_stochastic_build(self):
        """Performs one Monte Carlo simulation run to build a sequence."""
        sequence_info: List[Tuple[ClipSegment, float, EffectParams]] = []
        current_time = 0.0
        # Use indices of self.all_clips (which are pre-filtered)
        available_clip_indices = list(range(len(self.all_clips)))
        random.shuffle(available_clip_indices)
        last_clip_segment: Optional[ClipSegment] = None
        num_sources = len(set(c.source_video_path for c in self.all_clips))

        while current_time < self.target_duration and available_clip_indices:
            audio_seg = self.get_audio_segment_at(current_time)
            if not audio_seg:
                logger.debug(f"Stopping build at {current_time:.2f}s: No audio segment found.")
                break

            candidates_info = []; total_prob = 0.0
            # Consider clips remaining in the shuffled available list
            for list_idx_pos, original_clip_index in enumerate(available_clip_indices):
                clip = self.all_clips[original_clip_index]
                # Clip duration check is already done by pre-filtering self.all_clips

                # Calculate fit probability
                prob = clip.clip_audio_fit(audio_seg, self.analysis_config)

                # Apply repetition penalty ONLY if there's >1 source video and it's not the first clip
                if num_sources > 1 and last_clip_segment:
                    cand_path = str(clip.source_video_path) if clip.source_video_path else None
                    last_path = str(last_clip_segment.source_video_path) if last_clip_segment.source_video_path else None
                    if cand_path and last_path and cand_path == last_path:
                        prob *= (1.0 - self.analysis_config.variety_repetition_penalty)

                # Consider candidate if probability is above a small threshold
                if prob > 1e-5:
                     # Store: (clip object, index in available_clip_indices list, probability)
                     candidates_info.append((clip, list_idx_pos, prob))
                     total_prob += prob

            if not candidates_info:
                logger.debug(f"Stopping build at {current_time:.2f}s: No suitable candidates found (fit < 1e-5 or penalty too high).")
                break

            # --- Probabilistic Selection ---
            probabilities = [p / (total_prob + 1e-9) for _, _, p in candidates_info] # Normalize probabilities
            try:
                # Ensure probabilities sum ~1 and handle potential floating point issues
                prob_sum = sum(probabilities)
                if abs(prob_sum - 1.0) > 1e-6:
                    probabilities = [p / prob_sum for p in probabilities]
                # Choose candidate index based on calculated probabilities
                chosen_candidate_local_idx = random.choices(range(len(candidates_info)), weights=probabilities, k=1)[0]
            except ValueError as e: # Catch empty weights or other issues
                logger.warning(f"random.choices failed at {current_time:.2f}s (ProbSum:{total_prob:.2E}, NumCand:{len(candidates_info)}): {e}. Choosing randomly from candidates.")
                if not candidates_info: break # Should be caught above, but safety check
                chosen_candidate_local_idx = random.randrange(len(candidates_info))

            chosen_clip, chosen_list_idx_pos, _ = candidates_info[chosen_candidate_local_idx]
            # Get the original index from the main self.all_clips list
            original_chosen_clip_index = available_clip_indices[chosen_list_idx_pos]

            # --- Determine Duration ---
            remaining_time = self.target_duration - current_time
            # Choose duration: min of clip's available duration, remaining time, max allowed clip duration
            chosen_duration = min(chosen_clip.duration, remaining_time, self.max_clip_duration)
            # Ensure duration meets minimum requirement (unless it's the very last bit)
            chosen_duration = max(chosen_duration, self.min_clip_duration if remaining_time > self.min_clip_duration else 0.01)
            chosen_duration = max(0.01, chosen_duration) # Ensure positive

            # --- Choose Effect ---
            effect_options = list(self.effects.values())
            efficiencies = []
            for e in effect_options: # Calculate efficiency robustly
                denominator = e.tau * e.psi
                numerator = e.epsilon
                if abs(denominator) < 1e-9: eff = 0.0 if abs(numerator) < 1e-9 else 1e9
                else: eff = (numerator + 1e-9) / (denominator + 1e-9)
                efficiencies.append(eff)

            # Boost 'cut' probability (less disruptive)
            try: cut_index = next((i for i, e in enumerate(effect_options) if e.type == "cut"), -1)
            except StopIteration: cut_index = -1 # Handle case where 'cut' might not be in effects
            if cut_index != -1: efficiencies[cut_index] = max(efficiencies[cut_index], 1.0) * 2.0 # Ensure cut has some base chance & boost it

            positive_efficiencies = [max(0, eff) for eff in efficiencies]; total_efficiency = sum(positive_efficiencies)
            if total_efficiency > 1e-9 and effect_options:
                 effect_probs = [eff / total_efficiency for eff in positive_efficiencies]
                 # Renormalize if needed
                 sum_probs = sum(effect_probs)
                 if abs(sum_probs - 1.0) > 1e-6: effect_probs = [p / (sum_probs+1e-9) for p in effect_probs]

                 try: chosen_effect = random.choices(effect_options, weights=effect_probs, k=1)[0]
                 except ValueError: chosen_effect = self.effects.get('cut', EffectParams(type='cut')) # Fallback on choice error
                 except IndexError: chosen_effect = self.effects.get('cut', EffectParams(type='cut')) # Fallback if lists mismatch
            else: chosen_effect = self.effects.get('cut', EffectParams(type='cut')) # Default/Fallback

            # --- Add to Sequence ---
            sequence_info.append((chosen_clip, chosen_duration, chosen_effect))
            last_clip_segment = chosen_clip
            current_time += chosen_duration
            # Remove the chosen clip's index from the available list *for this run*
            available_clip_indices.pop(chosen_list_idx_pos)

        # --- Final Check ---
        final_sim_duration = sum(item[1] for item in sequence_info)
        # Require sequence to have *some* content
        if final_sim_duration < self.min_clip_duration:
            logger.debug(f"Discarding simulation sequence (Duration: {final_sim_duration:.2f}s < Min: {self.min_clip_duration:.2f}s)")
            return None
        return sequence_info


    def _evaluate_pareto(self, seq_info: List[Tuple[ClipSegment, float, EffectParams]]) -> List[float]:
        """Evaluates a generated sequence against the Pareto objectives."""
        if not seq_info: return [-1e9] * 5 # Return worst scores if sequence is empty
        num_clips = len(seq_info); total_duration = sum(item[1] for item in seq_info)
        if total_duration <= 1e-6: return [-1e9] * 5 # Avoid division by zero

        # Config parameters
        w_r, w_m, w_c, w_v, w_e = (self.analysis_config.objective_weight_rhythm, self.analysis_config.objective_weight_mood,
                                   self.analysis_config.objective_weight_continuity, self.analysis_config.objective_weight_variety,
                                   self.analysis_config.objective_weight_efficiency)
        sigma_m_sq = self.analysis_config.mood_similarity_variance**2 * 2
        kd = self.analysis_config.continuity_depth_weight
        lambda_penalty = self.analysis_config.variety_repetition_penalty
        num_sources = len(set(item[0].source_video_path for item in seq_info))

        # R(S): Rhythm Coherence - Minimize offset from beats (Negative score, higher is better)
        r_score_sum = 0.0; num_transitions_rhythm = 0; current_t = 0.0
        for i, (clip, duration, effect) in enumerate(seq_info):
            transition_time = current_t + duration
            if i < num_clips - 1: # Only evaluate transitions between clips
                nearest_b = self._nearest_beat_time(transition_time)
                if nearest_b is not None and self.beat_period > 1e-6:
                    offset_norm = abs(transition_time - nearest_b) / self.beat_period
                    # Penalize large offsets more (e.g., square the normalized offset)
                    r_score_sum += offset_norm**2 # Accumulate squared normalized offset
                    num_transitions_rhythm += 1
            current_t = transition_time
        avg_sq_offset = (r_score_sum / num_transitions_rhythm) if num_transitions_rhythm > 0 else 1.0 # Default penalty
        neg_r_score = -w_r * avg_sq_offset # Lower offset -> higher score

        # M(S): Mood Coherence - Maximize similarity between video/audio mood
        m_score_sum = 0.0; current_t = 0.0; mood_calcs = 0
        for clip, duration, effect in seq_info:
            mid_time = current_t + duration / 2.0; audio_seg = self.get_audio_segment_at(mid_time)
            if audio_seg:
                vid_mood = np.asarray(clip.mood_vector); aud_mood = np.asarray(audio_seg.get('m_i', [0.0, 0.0]))
                mood_dist_sq = np.sum((vid_mood - aud_mood)**2)
                m_score_sum += exp(-mood_dist_sq / (sigma_m_sq + 1e-9)) # Gaussian similarity
                mood_calcs += 1
            current_t += duration
        m_score = w_m * (m_score_sum / mood_calcs if mood_calcs > 0 else 0.0)

        # C(S): Visual Continuity - Maximize similarity between consecutive clips + effect gain
        c_score_sum = 0.0; num_transitions_cont = 0
        for i in range(num_clips - 1):
            clip1, _, effect_at_transition = seq_info[i]; clip2, _, _ = seq_info[i+1]
            f1 = clip1.get_feature_vector(self.analysis_config); f2 = clip2.get_feature_vector(self.analysis_config)
            safe_kd = max(0.0, kd) # Ensure non-negative weight
            # Calculate squared Euclidean distance in feature space [v, a, d_r]
            delta_e_sq = (f1[0]-f2[0])**2 + (f1[1]-f2[1])**2 + safe_kd*(f1[2]-f2[2])**2
            # Normalize by approx max possible distance squared
            max_delta_e_sq = 1**2 + 1**2 + safe_kd*(1**2)
            delta_e_norm_sq = delta_e_sq / (max_delta_e_sq + 1e-9)
            # Continuity term: 1 - normalized distance (higher is better)
            continuity_term = (1.0 - np.sqrt(np.clip(delta_e_norm_sq, 0.0, 1.0)))
            # Add perceptual gain from the chosen effect
            c_score_sum += continuity_term + effect_at_transition.epsilon
            num_transitions_cont += 1
        c_score = w_c * (c_score_sum / num_transitions_cont if num_transitions_cont > 0 else 1.0) # Default high if 1 clip


        # V(S): Dynamic Variety - Maximize avg entropy, Penalize source repetition
        valid_phis = [item[0].phi for item in seq_info if isinstance(item[0], ClipSegment) and item[0].phi is not None and np.isfinite(item[0].phi)]
        avg_phi = np.mean(valid_phis) if valid_phis else 0.0
        repetition_count = 0; num_transitions_var = 0
        # Only apply repetition penalty if multiple sources were actually used
        if num_sources > 1:
            for i in range(num_clips - 1):
                path1 = str(seq_info[i][0].source_video_path) if seq_info[i][0].source_video_path else None
                path2 = str(seq_info[i+1][0].source_video_path) if seq_info[i+1][0].source_video_path else None
                if path1 and path2 and path1 == path2: repetition_count += 1
                num_transitions_var +=1
        repetition_term = lambda_penalty * (repetition_count / num_transitions_var if num_transitions_var > 0 else 0)
        # Normalize entropy (approximate max entropy for 8-bit grayscale)
        max_entropy_approx = log(256); avg_phi_norm = avg_phi / max_entropy_approx if max_entropy_approx > 0 else 0.0
        v_score = w_v * (avg_phi_norm - repetition_term) # Higher entropy & lower repetition is better

        # EC(S): Effect Efficiency Cost - Minimize cost (Psi*Tau / Epsilon) (Negative score, higher is better)
        ec_score_sum = 0.0; cost_calcs = 0
        for _, _, effect in seq_info:
             psi_tau = effect.psi * effect.tau # Cost proxy
             epsilon = effect.epsilon # Gain proxy
             # Calculate cost/gain ratio, handle division by zero
             cost = (psi_tau + 1e-9) / (epsilon + 1e-9) if abs(epsilon) > 1e-9 else (psi_tau + 1e-9) / 1e-9 # High cost if no gain
             # Maybe clamp cost to avoid extreme values? cost = np.clip(cost, 0, 100)
             ec_score_sum += cost
             cost_calcs += 1
        avg_cost = (ec_score_sum / cost_calcs if cost_calcs > 0 else 0.0)
        neg_ec_score = -w_e * avg_cost # Lower cost -> higher score

        # Return list of weighted scores [NegR, M, C, V, NegEC]
        final_scores = [neg_r_score, m_score, c_score, v_score, neg_ec_score]
        # Ensure all returned scores are finite floats
        return [float(s) if np.isfinite(s) else -1e9 for s in final_scores]


    def _nearest_beat_time(self, time_sec):
        """Finds the beat time closest to the given time."""
        if not self.beat_times: return None
        beat_times_arr = np.asarray(self.beat_times)
        if len(beat_times_arr) == 0: return None
        # Find index of the minimum absolute difference
        closest_beat_idx = np.argmin(np.abs(beat_times_arr - time_sec))
        return float(beat_times_arr[closest_beat_idx])

    def _update_pareto_front(self, front: List[Tuple[List[Any], List[float]]], new_solution: Tuple[List[Any], List[float]]):
        """Updates the Pareto front with a new solution."""
        new_seq_info, new_scores = new_solution
        # Ensure scores are valid floats before comparison
        if not all(isinstance(s, float) and np.isfinite(s) for s in new_scores):
             logger.warning(f"Skipping Pareto update: new solution has invalid/non-float scores {new_scores}")
             return

        dominated_indices = set() # Use set for faster lookup and avoid duplicates
        is_dominated_by_front = False

        # Check against existing solutions
        indices_to_check = list(range(len(front))) # Iterate over indices to allow deletion
        for i in reversed(indices_to_check): # Iterate backwards for safe deletion
            existing_seq_info, existing_scores = front[i]
            # Ensure existing scores are also valid
            if not all(isinstance(s, float) and np.isfinite(s) for s in existing_scores):
                logger.warning(f"Removing existing Pareto solution {i} due to invalid/non-float scores {existing_scores}")
                del front[i]
                continue

            # Check for dominance
            if self._dominates(new_scores, existing_scores):
                # New solution dominates existing one
                dominated_indices.add(i)
            if self._dominates(existing_scores, new_scores):
                # Existing solution dominates new one
                is_dominated_by_front = True
                break # No need to check further, new solution won't be added

        # If new solution is not dominated by any existing solution
        if not is_dominated_by_front:
            # Remove all solutions dominated by the new one
            if dominated_indices:
                 # Convert set to sorted list for deletion
                 for i in sorted(list(dominated_indices), reverse=True):
                     if 0 <= i < len(front): # Check index validity again just in case
                          del front[i]
                     else:
                          logger.warning(f"Attempted to delete invalid index {i} from Pareto front during update.")
            # Add the new non-dominated solution
            front.append(new_solution)


    def _dominates(self, scores1: List[float], scores2: List[float]) -> bool:
        """Checks if solution 1 dominates solution 2 (Maximization)."""
        if len(scores1) != len(scores2): raise ValueError("Score lists must have same length.")
        # Check if scores1 is strictly better in at least one objective AND not worse in any
        at_least_one_better = False
        for s1, s2 in zip(scores1, scores2):
            if s1 < s2 - 1e-9: # score1 is worse than score2 (allow for float tolerance)
                return False
            if s1 > s2 + 1e-9: # score1 is strictly better than score2
                at_least_one_better = True
        return at_least_one_better


# ========================================================================
#        MOVIEPY VIDEO BUILDING FUNCTION (Using make_frame)
# ========================================================================
def buildSequenceVideo(final_sequence: List[ClipSegment], output_video_path: str, master_audio_path: str, render_config: RenderConfig):
    """Builds the final video using MoviePy's VideoClip and make_frame."""
    logger.info(f"Rendering video to {output_video_path} with audio {master_audio_path} using MoviePy make_frame...")
    start_time = time.time()
    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]

    # --- Validate Inputs ---
    if not final_sequence:
        logger.error("Cannot build video: Input sequence is empty.")
        if tracemalloc.is_tracing(): tracemalloc.stop()
        raise ValueError("Empty sequence: No video clips to process")
    if not master_audio_path or not os.path.exists(master_audio_path):
        logger.error(f"Cannot build video: Master audio file not found at {master_audio_path}")
        if tracemalloc.is_tracing(): tracemalloc.stop()
        raise FileNotFoundError(f"Audio file not found: {master_audio_path}")
    if not output_video_path:
         logger.error("Cannot build video: Output path not specified.")
         if tracemalloc.is_tracing(): tracemalloc.stop()
         raise ValueError("Output video path is required.")

    # --- Prepare Clips and Data for make_frame ---
    source_clips_dict: Dict[str, Optional[VideoFileClip]] = {}
    width = render_config.resolution_width
    height = render_config.resolution_height
    fps = render_config.fps
    # Ensure FPS is valid
    if not isinstance(fps, (int, float)) or fps <= 0:
        logger.warning(f"Invalid FPS in render_config ({fps}). Using default 30.")
        fps = 30

    # --- Pre-load and resize source clips ---
    logger.info("Pre-loading and resizing source video clips for make_frame...")
    unique_source_paths = sorted(list(set(str(seg.source_video_path) for seg in final_sequence)))
    for source_path in unique_source_paths:
        if not os.path.exists(source_path):
             logger.error(f"Source video file not found: {source_path}. Marking as None.")
             source_clips_dict[source_path] = None
             continue
        try:
            logger.debug(f"Loading and resizing: {os.path.basename(source_path)} to {width}x{height}")
            # Specify target resolution during load for potential efficiency
            clip_obj = VideoFileClip(source_path, audio=False, target_resolution=(height, width))
            # Ensure resizing happens if target_resolution didn't achieve it
            if clip_obj.w != width or clip_obj.h != height:
                 clip_obj = clip_obj.resize((width, height))

            # Basic check if reader seems initialized
            if not hasattr(clip_obj, 'reader') or clip_obj.reader is None:
                 logger.warning(f"MoviePy reader might not be initialized for {source_path}. Accessing duration to potentially force init.")
                 _ = clip_obj.duration # Access duration to trigger reader init if needed
            source_clips_dict[source_path] = clip_obj
        except Exception as load_err:
            logger.error(f"Failed to load or resize source clip {source_path}: {load_err}", exc_info=True)
            source_clips_dict[source_path] = None

    # --- Define make_frame Function ---
    def make_frame(t):
        # Find the active segment for time t
        active_segment = None
        for segment in final_sequence:
            # Use a small tolerance for end time comparison due to potential float issues
            if segment.sequence_start_time <= t < segment.sequence_end_time + 1e-6:
                active_segment = segment
                break
        # If t is exactly the end time of the sequence, use the last segment
        if active_segment is None and final_sequence and abs(t - final_sequence[-1].sequence_end_time) < 1e-6:
             active_segment = final_sequence[-1]

        if active_segment:
            source_path = str(active_segment.source_video_path)
            source_clip = source_clips_dict.get(source_path)

            if source_clip:
                # Calculate time within the source clip corresponding to sequence time t
                clip_time_in_sequence = t - active_segment.sequence_start_time
                # Map sequence time to source video time, starting from the segment's subclip start
                source_time = active_segment.subclip_start_time_in_source + clip_time_in_sequence

                # Clamp source_time to be within the valid duration of the source clip AND the chosen subclip bounds
                source_duration = source_clip.duration if source_clip.duration else 0
                subclip_start = active_segment.subclip_start_time_in_source
                # Use the END of the potential clip segment as the max bound, not chosen_duration,
                # because chosen_duration might be shorter than the segment duration used for analysis.
                subclip_end = active_segment.subclip_end_time_in_source # End time within the original source video

                # Clamp within the source clip's actual duration first
                final_source_time = np.clip(source_time, 0, source_duration - 1e-6 if source_duration > 0 else 0)
                # Then clamp within the specific subclip bounds identified earlier
                final_source_time = np.clip(final_source_time, subclip_start, subclip_end - 1e-6 if subclip_end > subclip_start else subclip_start)


                try:
                    frame = source_clip.get_frame(final_source_time)
                    # --- Placeholder for applying effects ---
                    # Example: if active_segment.chosen_effect.type == 'fade': ... apply fade logic ...
                    # Ensure frame dimensions match target after potential effects
                    if frame.shape[0] != height or frame.shape[1] != width:
                         frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    return frame
                except IndexError as frame_idx_err:
                     logger.error(f"IndexError getting frame at final_source_time {final_source_time:.4f} (orig t={t:.4f}) for {os.path.basename(source_path)}. Source Dur: {source_duration:.4f}, Subclip: [{subclip_start:.4f}-{subclip_end:.4f}]. Error: {frame_idx_err}")
                     return np.zeros((height, width, 3), dtype=np.uint8) # Return black frame on error
                except Exception as frame_err:
                    logger.error(f"Error getting frame at final_source_time {final_source_time:.4f} (orig t={t:.4f}) for {os.path.basename(source_path)}. Error: {frame_err}")
                    return np.zeros((height, width, 3), dtype=np.uint8) # Return black frame
            else:
                 logger.warning(f"Source clip object not found or invalid for path: {source_path} at time t={t:.3f}")
                 return np.zeros((height, width, 3), dtype=np.uint8) # Black frame if source clip failed loading
        else:
            # This case might happen if t slightly exceeds the sequence duration due to float precision
            logger.warning(f"Time t={t:.4f}s outside defined sequence range [0-{final_sequence[-1].sequence_end_time if final_sequence else 0:.4f}]. Returning black frame.")
            return np.zeros((height, width, 3), dtype=np.uint8)

    # --- Create and Write VideoClip ---
    master_audio = None
    sequence_clip = None
    temp_audio_filepath = None # Initialize to None
    try:
        total_duration = final_sequence[-1].sequence_end_time if final_sequence else 0
        if total_duration <= 0:
            raise ValueError(f"Sequence has zero or negative duration ({total_duration}). Cannot render.")
        logger.info(f"Creating MoviePy VideoClip with duration {total_duration:.2f}s")

        # Create the main video clip using the make_frame function
        sequence_clip = VideoClip(make_frame, duration=total_duration, ismask=False)

        # Load and prepare master audio
        logger.debug(f"Loading master audio: {master_audio_path}")
        master_audio = AudioFileClip(master_audio_path)
        logger.debug(f"Master audio duration: {master_audio.duration:.2f}s, Target video duration: {total_duration:.2f}s")

        # Adjust audio/video duration to match
        if master_audio.duration > total_duration:
            logger.info(f"Trimming master audio from {master_audio.duration:.2f}s to video duration {total_duration:.2f}s")
            master_audio = master_audio.subclip(0, total_duration)
        elif master_audio.duration < total_duration - 1e-3: # Allow small difference
             logger.warning(f"Master audio ({master_audio.duration:.2f}s) is shorter than video ({total_duration:.2f}s). Trimming video to match audio.")
             total_duration = master_audio.duration
             sequence_clip = sequence_clip.set_duration(total_duration)

        # Set audio on the video clip
        if master_audio:
             sequence_clip = sequence_clip.set_audio(master_audio)
        else:
             logger.warning("No master audio could be loaded or prepared. Video will be silent.")

        # --- Prepare FFmpeg Write Parameters ---
        # Map internal codec names if needed (currently identity)
        codec_map = {'libx264': 'libx264'} # Add other mappings if needed
        audio_codec_map = {'aac': 'aac'} # Add other mappings if needed

        # Define temporary audio file path
        temp_audio_filename = f"temp-audio_{int(time.time())}_{random.randint(1000,9999)}.m4a" # Use common temp format
        temp_audio_dir = os.path.dirname(output_video_path) or "." # Use output dir or current dir
        os.makedirs(temp_audio_dir, exist_ok=True)
        temp_audio_filepath = os.path.join(temp_audio_dir, temp_audio_filename)

        # Build ffmpeg_params list carefully
        ffmpeg_params_list = []
        if render_config.preset:
            ffmpeg_params_list.extend(["-preset", str(render_config.preset)])
        if render_config.crf is not None:
             ffmpeg_params_list.extend(["-crf", str(render_config.crf)])
        # Add more ffmpeg params here if needed (e.g., "-tune", "film")

        # Define parameters for write_videofile
        write_params = {
            "codec": codec_map.get(render_config.video_codec, 'libx264'),
            "audio_codec": audio_codec_map.get(render_config.audio_codec, 'aac'),
            "temp_audiofile": temp_audio_filepath,
            "remove_temp": True,
            "threads": render_config.threads,
            "preset": None,  # <<< CRITICAL FIX: DO NOT pass preset here directly anymore
            "logger": 'bar', # Use tqdm progress bar
            "write_logfile": False, # Don't write separate ffmpeg log file
            "audio_bitrate": render_config.audio_bitrate,
            "fps": fps,
            "ffmpeg_params": ffmpeg_params_list if ffmpeg_params_list else None # Pass preset/crf via this list
        }

        logger.info(f"Target FPS for rendering: {fps}")

        # --- Write Video ---
        logger.info(f"Writing final video with MoviePy (using make_frame)...")
        logger.debug(f"MoviePy write_videofile parameters: {write_params}")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # ***** THE ACTUAL RENDER CALL *****
        sequence_clip.write_videofile(output_video_path, **write_params)
        # ***** ********************** *****

        # --- Post-Render Memory Logging ---
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        end_memory = current_mem
        tracemalloc.stop()
        logger.info(f"MoviePy Render Performance: Time: {time.time() - start_time:.2f}s, PyMem Δ: {(end_memory - start_memory) / 1024**2:.2f} MB, Peak: {peak_mem / 1024**2:.2f} MB")
        logger.info(f"MoviePy rendering successful: {output_video_path}")

    except Exception as e:
        if tracemalloc.is_tracing(): tracemalloc.stop()
        logger.error(f"MoviePy rendering failed: {e}", exc_info=True)
        # Attempt to clean up potentially incomplete output file
        if os.path.exists(output_video_path):
             try: os.remove(output_video_path); logger.info(f"Removed potentially incomplete output file: {output_video_path}")
             except OSError as del_err: logger.warning(f"Could not remove failed output file {output_video_path}: {del_err}")
        # Attempt to clean up temporary audio file
        if temp_audio_filepath and os.path.exists(temp_audio_filepath):
             try: os.remove(temp_audio_filepath); logger.debug(f"Removed temp audio file {temp_audio_filepath} after error.")
             except OSError as del_err: logger.warning(f"Could not remove temp audio file {temp_audio_filepath} after error: {del_err}")
        raise # Re-raise the exception to be caught by the calling thread

    finally:
        # --- Cleanup MoviePy Objects ---
        logger.debug("Cleaning up MoviePy clip objects...")
        # Use weak references or manual closing
        if sequence_clip and hasattr(sequence_clip, 'close'):
             try: sequence_clip.close()
             except Exception as e: logger.debug(f"Minor error closing sequence_clip: {e}")
        if master_audio and hasattr(master_audio, 'close'):
             try: master_audio.close()
             except Exception as e: logger.debug(f"Minor error closing master_audio: {e}")
        for clip_key, source_clip_obj in source_clips_dict.items():
            if source_clip_obj and hasattr(source_clip_obj, 'close'):
                try: source_clip_obj.close()
                except Exception as e: logger.debug(f"Minor error closing source clip {clip_key}: {e}")
        # Force garbage collection maybe?
        import gc
        gc.collect()
        logger.debug("MoviePy clip cleanup attempt finished.")


# ========================================================================
#         WORKER FUNCTION FOR PARALLEL PROCESSING (Uses AnalysisConfig)
# ========================================================================
def process_single_video(video_path: str, audio_data: Dict, analysis_config: AnalysisConfig, output_dir: str) -> Tuple[str, str, List[ClipSegment]]:
    """Worker function: Analyzes video, returns potential clips."""
    start_t = time.time(); pid = os.getpid(); thread_name = threading.current_thread().name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Setup worker-specific logging to avoid multiprocessing issues with root logger handlers
    worker_logger = logging.getLogger(f"Worker.{pid}.{thread_name}")
    if not worker_logger.hasHandlers():
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(f'%(asctime)s - %(levelname)-8s - Worker {pid} - %(message)s')
        ch.setFormatter(formatter)
        worker_logger.addHandler(ch)
        worker_logger.setLevel(logging.INFO) # Set level for worker console output

    worker_logger.info(f"Starting Analysis: {base_name}")
    status = "Unknown Error"; potential_clips: List[ClipSegment] = []; frame_features: Optional[List[Dict]] = []

    try:
        # Instantiate analyzer inside the worker process
        analyzer = VideousMain()

        # Perform the analysis
        frame_features, potential_clips_result = analyzer.analyzeVideo(video_path, analysis_config, audio_data)

        # Process results
        if potential_clips_result is None:
             status = "Analysis Failed (returned None)"
             potential_clips = []
        elif not potential_clips_result:
             status = "Analysis OK (0 potential clips)"
             potential_clips = []
        else:
            # Filter results to ensure only valid ClipSegment objects are returned
            potential_clips = [clip for clip in potential_clips_result if isinstance(clip, ClipSegment)]
            status = f"Analysis OK ({len(potential_clips)} potential clips)"

            # Save analysis data if requested and successful
            if analysis_config.save_analysis_data and frame_features and potential_clips:
                try:
                    analyzer.saveAnalysisData(video_path, frame_features, potential_clips, output_dir, analysis_config)
                except Exception as save_err:
                    worker_logger.error(f"Failed to save analysis data for {base_name}: {save_err}", exc_info=True)
            elif analysis_config.save_analysis_data and not (frame_features and potential_clips):
                 worker_logger.warning(f"Save analysis data requested for {base_name}, but features or clips were empty/invalid.")

    except Exception as e:
        status = f"Failed: {type(e).__name__}"
        worker_logger.error(f"!!! FATAL ERROR analyzing {base_name} in worker {pid} !!!", exc_info=True)
        potential_clips = []
        frame_features = None # Ensure failure state is clear
    finally:
        # Explicitly delete large objects to potentially help memory management in the worker
        if 'frame_features' in locals(): del frame_features
        if 'potential_clips_result' in locals(): del potential_clips_result
        if 'analyzer' in locals(): del analyzer
        # Consider releasing MiDaS model if loaded only in worker? (Handled by global cache for now)

    end_t = time.time()
    worker_logger.info(f"Finished Analysis {base_name} ({status}) in {end_t - start_t:.2f}s")
    # Return path, status, and the list of ClipSegment objects (ensuring it's a list)
    return (video_path, status, potential_clips if potential_clips is not None else [])


# ========================================================================
#                      APP INTERFACE - Uses Dataclasses
# ========================================================================
class VideousApp(customtkinter.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("Videous Chef - v4.7.3 (MoviePy + Fixes)") # Incremented Version
        self.geometry("950x850")

        # Core State
        self.video_files: List[str] = []; self.beat_track_path: Optional[str] = None
        self.analysis_config: Optional[AnalysisConfig] = None; self.render_config: Optional[RenderConfig] = None
        self.is_processing = False; self.master_audio_data: Optional[Dict] = None
        self.all_potential_clips: List[ClipSegment] = []

        # Process/Thread Management
        self.processing_thread = None; self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.analysis_futures: List[concurrent.futures.Future] = []; self.futures_map: Dict[concurrent.futures.Future, str] = {}
        self.total_tasks = 0; self.completed_tasks = 0

        # Output Directories
        self.output_dir = "output_videous_chef"
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            # Define subdirs relative to output_dir
            self.analysis_subdir = os.path.join(self.output_dir, "analysis_data"); os.makedirs(self.analysis_subdir, exist_ok=True)
            self.render_subdir = os.path.join(self.output_dir, "final_renders"); os.makedirs(self.render_subdir, exist_ok=True)
        except OSError as e:
             logger.critical(f"Failed to create output directories: {e}. Exiting.", exc_info=True)
             messagebox.showerror("Fatal Error", f"Could not create output directories in {self.output_dir}.\nPlease check permissions.\n\n{e}")
             sys.exit(1)

        # Fonts (Try-Except block handles missing fonts)
        try:
            self.header_font = customtkinter.CTkFont(family="Garamond", size=28, weight="bold"); self.label_font = customtkinter.CTkFont(family="Garamond", size=14)
            self.button_font = customtkinter.CTkFont(family="Garamond", size=12); self.dropdown_font = customtkinter.CTkFont(family="Garamond", size=12)
            self.small_font = customtkinter.CTkFont(family="Garamond", size=10); self.mode_font = customtkinter.CTkFont(family="Garamond", size=13, weight="bold")
            self.tab_font = customtkinter.CTkFont(family="Garamond", size=14, weight="bold"); self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")
        except Exception as font_e:
             logger.warning(f"Garamond font not found, using defaults: {font_e}")
             self.header_font = customtkinter.CTkFont(size=28, weight="bold"); self.label_font = customtkinter.CTkFont(size=14); self.button_font = customtkinter.CTkFont(size=12)
             self.dropdown_font = customtkinter.CTkFont(size=12); self.small_font = customtkinter.CTkFont(size=10); self.mode_font = customtkinter.CTkFont(size=13, weight="bold")
             self.tab_font = customtkinter.CTkFont(size=14, weight="bold"); self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")

        # UI Setup
        self._setup_luxury_theme(); self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.info("VideousApp initialized.")

    def _setup_luxury_theme(self):
        self.diamond_white = "#F5F6F5"; self.deep_black = "#1C2526"; self.gold_accent = "#D4AF37"; self.jewel_blue = "#2A4B7C";
        customtkinter.set_appearance_mode("dark"); customtkinter.set_default_color_theme("blue"); self.configure(bg=self.deep_black)
    def _button_styles(self, border_width=1): return { "corner_radius": 8, "fg_color": self.jewel_blue, "hover_color": self.gold_accent, "border_color": self.diamond_white, "border_width": border_width, "text_color": self.diamond_white }
    def _radio_styles(self): return { "border_color": self.diamond_white, "fg_color": self.jewel_blue, "hover_color": self.gold_accent, "text_color": self.diamond_white }

    def _build_ui(self):
        # Grid Config
        self.grid_columnconfigure(0, weight=4); self.grid_columnconfigure(1, weight=3); self.grid_rowconfigure(1, weight=1); # Rows 2,3,4 for bottom content
        # Header
        customtkinter.CTkLabel(self, text="Videous Chef - Remix Engine", font=self.header_font, text_color=self.gold_accent).grid(row=0, column=0, columnspan=2, pady=(15, 10), sticky="ew")
        # Left Column: Config Tabs
        config_outer_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15); config_outer_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nsew")
        config_outer_frame.grid_rowconfigure(0, weight=1); config_outer_frame.grid_columnconfigure(0, weight=1)
        self.tab_view = customtkinter.CTkTabview(config_outer_frame, fg_color=self.deep_black, segmented_button_fg_color=self.deep_black, segmented_button_selected_color=self.jewel_blue, segmented_button_selected_hover_color=self.gold_accent, segmented_button_unselected_color="#333", segmented_button_unselected_hover_color="#555", text_color=self.diamond_white, border_color=self.diamond_white, border_width=2); self.tab_view.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        # --- Added Render Settings Tab ---
        self.tab_view.add("Shared"); self.tab_view.add("Greedy Heuristic"); self.tab_view.add("Physics MC"); self.tab_view.add("Render Settings")
        self.tab_view._segmented_button.configure(font=self.tab_font)
        self.shared_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Shared"), fg_color="transparent"); self.shared_tab_frame.pack(expand=True, fill="both")
        self.greedy_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Greedy Heuristic"), fg_color="transparent"); self.greedy_tab_frame.pack(expand=True, fill="both")
        self.physics_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Physics MC"), fg_color="transparent"); self.physics_tab_frame.pack(expand=True, fill="both")
        self.render_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Render Settings"), fg_color="transparent"); self.render_tab_frame.pack(expand=True, fill="both")
        self._create_config_sliders() # Populates tabs including Render Settings
        # Right Column: Files
        files_outer_frame = customtkinter.CTkFrame(self, fg_color="transparent"); files_outer_frame.grid(row=1, column=1, padx=(5, 10), pady=5, sticky="nsew")
        files_outer_frame.grid_rowconfigure(1, weight=1); files_outer_frame.grid_columnconfigure(0, weight=1) # Make list expand
        # Master Audio Frame
        controls_frame = customtkinter.CTkFrame(files_outer_frame, fg_color=self.deep_black, border_color=self.diamond_white, border_width=2, corner_radius=10); controls_frame.grid(row=0, column=0, padx=0, pady=(0, 5), sticky="new")
        beat_track_frame = customtkinter.CTkFrame(controls_frame, fg_color="transparent"); beat_track_frame.pack(pady=(10, 10), padx=10, fill="x")
        customtkinter.CTkLabel(beat_track_frame, text="2. Master Audio Track (The Base)", anchor="w", font=self.label_font, text_color=self.diamond_white).pack(pady=(0, 2), anchor="w")
        beat_btn_frame = customtkinter.CTkFrame(beat_track_frame, fg_color="transparent"); beat_btn_frame.pack(fill="x")
        self.beat_track_button = customtkinter.CTkButton(beat_btn_frame, text="Select Audio/Video", font=self.button_font, command=self._select_beat_track, **self._button_styles()); self.beat_track_button.pack(side="left", padx=(0, 10))
        self.beat_track_label = customtkinter.CTkLabel(beat_btn_frame, text="No master track selected.", anchor="w", wraplength=300, font=self.small_font, text_color=self.diamond_white); self.beat_track_label.pack(side="left", fill="x", expand=True)
        # Video Files Frame
        video_files_frame = customtkinter.CTkFrame(files_outer_frame, fg_color=self.deep_black, border_color=self.diamond_white, border_width=2, corner_radius=10); video_files_frame.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")
        video_files_frame.grid_rowconfigure(1, weight=1); video_files_frame.grid_columnconfigure(0, weight=1)
        customtkinter.CTkLabel(video_files_frame, text="3. Source Videos (The Ingredients)", anchor="w", font=self.label_font, text_color=self.diamond_white).grid(row=0, column=0, columnspan=2, pady=(5, 2), padx=10, sticky="ew")
        list_frame = Frame(video_files_frame, bg=self.deep_black, highlightbackground=self.diamond_white, highlightthickness=1); list_frame.grid(row=1, column=0, columnspan=2, pady=5, padx=10, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1); list_frame.grid_columnconfigure(0, weight=1)
        self.video_listbox = Listbox(list_frame, selectmode=MULTIPLE, bg=self.deep_black, fg=self.diamond_white, borderwidth=0, highlightthickness=0, font=("Garamond", 12), selectbackground=self.jewel_blue, selectforeground=self.gold_accent); self.video_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = Scrollbar(list_frame, orient="vertical", command=self.video_listbox.yview, background=self.deep_black, troughcolor=self.jewel_blue); scrollbar.grid(row=0, column=1, sticky="ns"); self.video_listbox.configure(yscrollcommand=scrollbar.set)
        self.video_listbox.drop_target_register(DND_FILES); self.video_listbox.dnd_bind('<<Drop>>', self._handle_drop)
        list_button_frame = customtkinter.CTkFrame(video_files_frame, fg_color="transparent"); list_button_frame.grid(row=2, column=0, columnspan=2, pady=(5, 10), padx=10, sticky="ew")
        list_button_frame.grid_columnconfigure((0, 1, 2), weight=1) # Space buttons
        self.add_button = customtkinter.CTkButton(list_button_frame, text="Add", width=70, font=self.button_font, command=self._add_videos_manual, **self._button_styles()); self.add_button.grid(row=0, column=0, padx=5, sticky="ew")
        self.remove_button = customtkinter.CTkButton(list_button_frame, text="Remove", width=70, font=self.button_font, command=self._remove_selected_videos, **self._button_styles()); self.remove_button.grid(row=0, column=1, padx=5, sticky="ew")
        self.clear_button = customtkinter.CTkButton(list_button_frame, text="Clear", width=70, font=self.button_font, command=self._clear_video_list, **self._button_styles()); self.clear_button.grid(row=0, column=2, padx=5, sticky="ew")
        # Bottom Frame: Mode Select & Action
        self.bottom_control_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15) # Name this frame
        self.bottom_control_frame.grid(row=2, column=0, columnspan=2, pady=(5, 5), padx=10, sticky="ew")
        self.bottom_control_frame.grid_columnconfigure(0, weight=1); self.bottom_control_frame.grid_columnconfigure(1, weight=1)
        mode_inner_frame = customtkinter.CTkFrame(self.bottom_control_frame, fg_color="transparent"); mode_inner_frame.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="ew")
        customtkinter.CTkLabel(mode_inner_frame, text="Sequencing Mode:", font=self.mode_font, text_color=self.gold_accent).pack(side="left", padx=(5, 10))
        self.mode_var = tkinter.StringVar(value="Greedy Heuristic"); # Default mode
        self.mode_selector = customtkinter.CTkSegmentedButton(mode_inner_frame, values=["Greedy Heuristic", "Physics Pareto MC"], variable=self.mode_var, font=self.button_font, selected_color=self.jewel_blue, selected_hover_color=self.gold_accent, unselected_color="#333", unselected_hover_color="#555", text_color=self.diamond_white, command=self._mode_changed) # Name this widget
        self.mode_selector.pack(side="left", expand=True, fill="x")
        self.run_button = customtkinter.CTkButton(self.bottom_control_frame, text="4. Compose Video Remix", height=45, font=customtkinter.CTkFont(family="Garamond", size=16, weight="bold"), command=self._start_processing, **self._button_styles(border_width=2)); self.run_button.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="e")
        # Status Label & Footer
        self.status_label = customtkinter.CTkLabel(self, text="Ready for Chef's command.", anchor="w", font=self.button_font, text_color=self.diamond_white, wraplength=900); self.status_label.grid(row=3, column=0, columnspan=2, pady=(0, 5), padx=20, sticky="ew")
        customtkinter.CTkLabel(self, text="Videous Chef v4.7.3 - MoviePy Full Refactor", font=self.small_font, text_color=self.gold_accent).grid(row=4, column=1, pady=5, padx=10, sticky="se") # Updated footer version

    def _mode_changed(self, value):
        logger.info(f"Sequencing mode changed to: {value}")
        self.status_label.configure(text=f"Mode set to: {value}. Ready.", text_color=self.diamond_white)

    def _create_config_sliders(self):
        # Use the AnalysisConfig/RenderConfig with adjusted defaults
        default_analysis_cfg = AnalysisConfig(); default_render_cfg = RenderConfig()
        self.slider_vars = {
            # Analysis Shared
            'min_sequence_clip_duration': tkinter.DoubleVar(value=default_analysis_cfg.min_sequence_clip_duration), 'max_sequence_clip_duration': tkinter.DoubleVar(value=default_analysis_cfg.max_sequence_clip_duration),
            'norm_max_velocity': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_velocity), 'norm_max_acceleration': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_acceleration),
            'norm_max_rms': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_rms), 'norm_max_onset': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_onset),
            'norm_max_chroma_var': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_chroma_var), 'norm_max_depth_variance': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_depth_variance),
            'min_face_confidence': tkinter.DoubleVar(value=default_analysis_cfg.min_face_confidence), 'mouth_open_threshold': tkinter.DoubleVar(value=default_analysis_cfg.mouth_open_threshold),
            'save_analysis_data': tkinter.BooleanVar(value=default_analysis_cfg.save_analysis_data),
            # Heuristic Norms (New/Moved)
            'norm_max_kinetic': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_kinetic),
            'norm_max_jerk': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_jerk),
            'norm_max_cam_motion': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_cam_motion),
            'norm_max_face_size': tkinter.DoubleVar(value=default_analysis_cfg.norm_max_face_size),
            # Greedy
            'audio_weight': tkinter.DoubleVar(value=default_analysis_cfg.audio_weight), 'kinetic_weight': tkinter.DoubleVar(value=default_analysis_cfg.kinetic_weight),
            'sharpness_weight': tkinter.DoubleVar(value=default_analysis_cfg.sharpness_weight), 'camera_motion_weight': tkinter.DoubleVar(value=default_analysis_cfg.camera_motion_weight),
            'face_size_weight': tkinter.DoubleVar(value=default_analysis_cfg.face_size_weight), 'percussive_weight': tkinter.DoubleVar(value=default_analysis_cfg.percussive_weight),
            'depth_weight': tkinter.DoubleVar(value=default_analysis_cfg.depth_weight), 'score_threshold': tkinter.DoubleVar(value=default_analysis_cfg.score_threshold),
            'beat_boost': tkinter.DoubleVar(value=default_analysis_cfg.beat_boost), 'beat_boost_radius_sec': tkinter.DoubleVar(value=default_analysis_cfg.beat_boost_radius_sec),
            'min_potential_clip_duration_sec': tkinter.DoubleVar(value=default_analysis_cfg.min_potential_clip_duration_sec), 'pacing_variation_factor': tkinter.DoubleVar(value=default_analysis_cfg.pacing_variation_factor),
            'variety_penalty_source': tkinter.DoubleVar(value=default_analysis_cfg.variety_penalty_source), # Uses adjusted default
            'variety_penalty_shot': tkinter.DoubleVar(value=default_analysis_cfg.variety_penalty_shot), # Uses adjusted default
            'variety_penalty_intensity': tkinter.DoubleVar(value=default_analysis_cfg.variety_penalty_intensity), 'beat_sync_bonus': tkinter.DoubleVar(value=default_analysis_cfg.beat_sync_bonus),
            'section_match_bonus': tkinter.DoubleVar(value=default_analysis_cfg.section_match_bonus), 'candidate_pool_size': tkinter.IntVar(value=default_analysis_cfg.candidate_pool_size),
            'min_pose_confidence': tkinter.DoubleVar(value=default_analysis_cfg.min_pose_confidence), 'model_complexity': tkinter.IntVar(value=default_analysis_cfg.model_complexity),
            'use_heuristic_segment_id': tkinter.BooleanVar(value=default_analysis_cfg.use_heuristic_segment_id),
            # Physics
            'fit_weight_velocity': tkinter.DoubleVar(value=default_analysis_cfg.fit_weight_velocity), 'fit_weight_acceleration': tkinter.DoubleVar(value=default_analysis_cfg.fit_weight_acceleration),
            'fit_weight_mood': tkinter.DoubleVar(value=default_analysis_cfg.fit_weight_mood), 'fit_sigmoid_steepness': tkinter.DoubleVar(value=default_analysis_cfg.fit_sigmoid_steepness),
            'objective_weight_rhythm': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_rhythm), 'objective_weight_mood': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_mood),
            'objective_weight_continuity': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_continuity), 'objective_weight_variety': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_variety),
            'objective_weight_efficiency': tkinter.DoubleVar(value=default_analysis_cfg.objective_weight_efficiency), 'mc_iterations': tkinter.IntVar(value=default_analysis_cfg.mc_iterations),
            'mood_similarity_variance': tkinter.DoubleVar(value=default_analysis_cfg.mood_similarity_variance), # Uses adjusted default
            'continuity_depth_weight': tkinter.DoubleVar(value=default_analysis_cfg.continuity_depth_weight),
            'variety_repetition_penalty': tkinter.DoubleVar(value=default_analysis_cfg.variety_repetition_penalty), # Uses adjusted default
            # Effects (Physics Tab - Configure RenderConfig)
            'effect_fade_duration': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('fade', EffectParams()).tau), 'effect_zoom_duration': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('zoom', EffectParams()).tau),
            'effect_zoom_impact': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('zoom', EffectParams()).psi), 'effect_zoom_perceptual': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('zoom', EffectParams()).epsilon),
            'effect_pan_duration': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('pan', EffectParams()).tau), 'effect_pan_impact': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('pan', EffectParams()).psi),
            'effect_pan_perceptual': tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('pan', EffectParams()).epsilon),
            # Render Settings (New)
            'render_width': tkinter.IntVar(value=default_render_cfg.resolution_width),
            'render_height': tkinter.IntVar(value=default_render_cfg.resolution_height),
            'render_fps': tkinter.IntVar(value=default_render_cfg.fps),
        }
        def add_separator(parent, text):
             customtkinter.CTkFrame(parent, height=2, fg_color=self.diamond_white).pack(fill="x", padx=5, pady=(15, 2))
             customtkinter.CTkLabel(parent, text=text, font=self.separator_font, text_color=self.gold_accent, anchor="w").pack(fill="x", padx=5, pady=(0, 5))
        def add_checkbox(parent, label, variable):
             frame = customtkinter.CTkFrame(parent, fg_color="transparent"); frame.pack(fill="x", pady=4, padx=5)
             customtkinter.CTkCheckBox(frame, text=label, variable=variable, font=self.label_font, text_color=self.diamond_white, hover_color=self.gold_accent, fg_color=self.jewel_blue, border_color=self.diamond_white).pack(side="left", padx=5)

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
        # --- New Heuristic Normalization Calibration Section ---
        add_separator(parent, "--- Heuristic Normalization Calibration ---")
        self._create_single_slider(parent, "Max Body Motion (Kinetic):", self.slider_vars['norm_max_kinetic'], 10.0, 200.0, 38, "{:.0f}")
        self._create_single_slider(parent, "Max Motion Sharpness (Jerk):", self.slider_vars['norm_max_jerk'], 10.0, 200.0, 38, "{:.0f}")
        self._create_single_slider(parent, "Max Camera Movement:", self.slider_vars['norm_max_cam_motion'], 1.0, 20.0, 38, "{:.1f}")
        self._create_single_slider(parent, "Max Face Size Ratio:", self.slider_vars['norm_max_face_size'], 0.1, 1.0, 18, "{:.1f}")

        # --- Populate GREEDY HEURISTIC Tab ---
        parent = self.greedy_tab_frame; add_separator(parent, "--- Feature Influence Weights ---"); self._create_single_slider(parent, "Music Volume Influence:", self.slider_vars['audio_weight'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Body Motion Influence:", self.slider_vars['kinetic_weight'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Motion Sharpness Influence:", self.slider_vars['sharpness_weight'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Camera Movement Influence:", self.slider_vars['camera_motion_weight'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Close-Up Shot Bonus:", self.slider_vars['face_size_weight'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Percussive Hit Influence:", self.slider_vars['percussive_weight'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Scene Depth Influence:", self.slider_vars['depth_weight'], 0.0, 1.0, 20, "{:.2f}")
        add_separator(parent, "--- Clip Identification & Selection ---"); add_checkbox(parent, "Identify Potential Clips using Heuristic Score Runs", self.slider_vars['use_heuristic_segment_id']); self._create_single_slider(parent, "Min Heuristic Score Run Threshold:", self.slider_vars['score_threshold'], 0.1, 0.8, 35, "{:.2f}"); self._create_single_slider(parent, "Min Potential Heuristic Clip Length (s):", self.slider_vars['min_potential_clip_duration_sec'], 0.2, 2.0, 18, "{:.1f}s"); self._create_single_slider(parent, "Clip Options Considered per Cut:", self.slider_vars['candidate_pool_size'], 5, 50, 45, "{:.0f}")
        add_separator(parent, "--- Beat Sync & Bonuses ---"); self._create_single_slider(parent, "Beat Emphasis Strength:", self.slider_vars['beat_boost'], 0.0, 0.5, 25, "{:.2f}"); self._create_single_slider(parent, "Beat Emphasis Window (s):", self.slider_vars['beat_boost_radius_sec'], 0.0, 0.5, 10, "{:.1f}s"); self._create_single_slider(parent, "Cut-on-Beat Bonus:", self.slider_vars['beat_sync_bonus'], 0.0, 0.5, 25, "{:.2f}"); self._create_single_slider(parent, "Music Section Mood Match Bonus:", self.slider_vars['section_match_bonus'], 0.0, 0.5, 25, "{:.2f}")
        add_separator(parent, "--- Sequence Pacing & Variety ---"); self._create_single_slider(parent, "Pacing Flexibility (+/- %):", self.slider_vars['pacing_variation_factor'], 0.0, 0.7, 14, "{:.1f}"); self._create_single_slider(parent, "Same Video Repetition Penalty:", self.slider_vars['variety_penalty_source'], 0.0, 0.5, 25, "{:.2f}"); self._create_single_slider(parent, "Same Shot Type Penalty:", self.slider_vars['variety_penalty_shot'], 0.0, 0.5, 25, "{:.2f}"); self._create_single_slider(parent, "Same Energy Level Penalty:", self.slider_vars['variety_penalty_intensity'], 0.0, 0.5, 25, "{:.2f}")
        add_separator(parent, "--- Pose Analysis Settings ---"); self._create_single_slider(parent, "Min Body Pose Certainty:", self.slider_vars['min_pose_confidence'], 0.1, 0.9, 16, "{:.2f}")
        comp_frame_greedy = customtkinter.CTkFrame(parent, fg_color="transparent"); comp_frame_greedy.pack(fill="x", pady=5, padx=5); customtkinter.CTkLabel(comp_frame_greedy, text="Pose Model Quality:", width=190, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5, 5)); radio_fr_greedy = customtkinter.CTkFrame(comp_frame_greedy, fg_color="transparent"); radio_fr_greedy.pack(side="left", padx=5); customtkinter.CTkRadioButton(radio_fr_greedy, text="Fast(0)", variable=self.slider_vars['model_complexity'], value=0, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3); customtkinter.CTkRadioButton(radio_fr_greedy, text="Balanced(1)", variable=self.slider_vars['model_complexity'], value=1, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3); customtkinter.CTkRadioButton(radio_fr_greedy, text="Accurate(2)", variable=self.slider_vars['model_complexity'], value=2, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3)
        # --- Populate PHYSICS PARETO MC Tab ---
        parent = self.physics_tab_frame; add_separator(parent, "--- Clip-Audio Fit Weights ---"); self._create_single_slider(parent, "Match: Motion Speed to Beat (w_v):", self.slider_vars['fit_weight_velocity'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Match: Motion Change to Energy (w_a):", self.slider_vars['fit_weight_acceleration'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Match: Video Mood to Music Mood (w_m):", self.slider_vars['fit_weight_mood'], 0.0, 1.0, 20, "{:.2f}"); self._create_single_slider(parent, "Fit Sensitivity (Sigmoid k):", self.slider_vars['fit_sigmoid_steepness'], 0.1, 5.0, 49, "{:.1f}")
        add_separator(parent, "--- Pareto Objective Priorities ---"); self._create_single_slider(parent, "Priority: Cut Rhythm/Timing (w_r):", self.slider_vars['objective_weight_rhythm'], 0.0, 2.0, 20, "{:.1f}"); self._create_single_slider(parent, "Priority: Mood Consistency (w_m):", self.slider_vars['objective_weight_mood'], 0.0, 2.0, 20, "{:.1f}"); self._create_single_slider(parent, "Priority: Smooth Transitions (w_c):", self.slider_vars['objective_weight_continuity'], 0.0, 2.0, 20, "{:.1f}"); self._create_single_slider(parent, "Priority: Scene/Pacing Variety (w_v):", self.slider_vars['objective_weight_variety'], 0.0, 2.0, 20, "{:.1f}"); self._create_single_slider(parent, "Priority: Effect Efficiency (w_e):", self.slider_vars['objective_weight_efficiency'], 0.0, 2.0, 20, "{:.1f}")
        add_separator(parent, "--- Sequence Evaluation ---"); self._create_single_slider(parent, "Sequence Search Depth (MC Iterations):", self.slider_vars['mc_iterations'], 100, 5000, 490, "{:d}"); self._create_single_slider(parent, "Mood Matching Tolerance (σm):", self.slider_vars['mood_similarity_variance'], 0.05, 0.5, 18, "{:.2f}"); self._create_single_slider(parent, "Transition Smoothness: Depth Weight (kd):", self.slider_vars['continuity_depth_weight'], 0.0, 1.0, 20, "{:.1f}"); self._create_single_slider(parent, "Source Repetition Penalty (λ):", self.slider_vars['variety_repetition_penalty'], 0.0, 1.0, 20, "{:.1f}")
        add_separator(parent, "--- Effect Tuning (Configures Rendering) ---"); self._create_single_slider(parent, "Fade Duration (τ, s):", self.slider_vars['effect_fade_duration'], 0.05, 1.0, 19, "{:.2f}s"); self._create_single_slider(parent, "Zoom Duration (τ, s):", self.slider_vars['effect_zoom_duration'], 0.1, 2.0, 19, "{:.1f}s"); self._create_single_slider(parent, "Zoom Max Intensity (ψ, factor):", self.slider_vars['effect_zoom_impact'], 0.05, 1.0, 19, "{:.2f}"); self._create_single_slider(parent, "Zoom Visual Gain (ε):", self.slider_vars['effect_zoom_perceptual'], 0.0, 1.0, 20, "{:.1f}"); self._create_single_slider(parent, "Pan Duration (τ, s):", self.slider_vars['effect_pan_duration'], 0.1, 2.0, 19, "{:.1f}s"); self._create_single_slider(parent, "Pan Max Distance (ψ, % width):", self.slider_vars['effect_pan_impact'], 0.01, 0.5, 49, "{:.2f}"); self._create_single_slider(parent, "Pan Visual Gain (ε):", self.slider_vars['effect_pan_perceptual'], 0.0, 1.0, 20, "{:.1f}")
        # --- Populate RENDER SETTINGS Tab ---
        parent = self.render_tab_frame
        add_separator(parent, "--- Output Video Settings ---")
        self._create_single_slider(parent, "Render Width (pixels):", self.slider_vars['render_width'], 640, 3840, 320, "{:.0f}") # Example range/steps
        self._create_single_slider(parent, "Render Height (pixels):", self.slider_vars['render_height'], 360, 2160, 180, "{:.0f}") # Example range/steps
        self._create_single_slider(parent, "Render FPS:", self.slider_vars['render_fps'], 24, 60, 36, "{:.0f}")
        # Add more render settings sliders here if needed (e.g., CRF, preset, codecs)

    def _create_single_slider(self, parent, label_text, variable, from_val, to_val, steps, format_str="{:.2f}"):
        row = customtkinter.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", pady=4, padx=5)
        customtkinter.CTkLabel(row, text=label_text, width=300, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5, 5))
        val_lab = customtkinter.CTkLabel(row, text=format_str.format(variable.get()), width=70, anchor="e", font=self.button_font, text_color=self.gold_accent); val_lab.pack(side="right", padx=(5, 5))
        # Adjust slider creation for IntVars
        is_int = isinstance(variable, tkinter.IntVar)
        # Ensure number_of_steps is integer or None
        num_steps = int(steps) if steps is not None and steps > 0 else None
        # Lambda to update label text correctly for int/float
        command_lambda = (lambda v, lbl=val_lab, fmt=format_str: lbl.configure(text=fmt.format(int(round(float(v)))))) if is_int else \
                         (lambda v, lbl=val_lab, fmt=format_str: lbl.configure(text=fmt.format(float(v))))

        slider = customtkinter.CTkSlider(
            row, variable=variable, from_=from_val, to=to_val,
            number_of_steps=num_steps,
            command=command_lambda,
            progress_color=self.gold_accent, button_color=self.diamond_white,
            button_hover_color=self.gold_accent, fg_color=self.jewel_blue
        )
        slider.pack(side="left", fill="x", expand=True, padx=5)


    # --- File Handling Methods (Logging Enhanced) ---
    def _select_beat_track(self):
        if self.is_processing: return
        filetypes = (("Audio/Video files", "*.wav *.mp3 *.aac *.flac *.ogg *.mp4 *.mov *.avi *.mkv"),("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select Master Audio Track Source", filetypes=filetypes)
        if filepath:
            try:
                if os.path.isfile(filepath):
                    self.beat_track_path = filepath; self.beat_track_label.configure(text=os.path.basename(filepath))
                    logger.info(f"Master audio source selected: {filepath}")
                    self.status_label.configure(text=f"Master: {os.path.basename(filepath)}", text_color=self.diamond_white)
                else: self.beat_track_path = None; self.beat_track_label.configure(text="Selection is not a file."); logger.warning(f"Selected path is not a file: {filepath}"); tk_write(f"Invalid selection (not a file):\n{filepath}", parent=self, level="warning")
            except Exception as e: self.beat_track_path = None; self.beat_track_label.configure(text="Error checking path."); logger.error(f"Could not verify path '{filepath}': {e}", exc_info=True); tk_write(f"Error checking file:\n{e}", parent=self, level="error")

    def _handle_drop(self, event):
        if self.is_processing: return 'break'
        try:
            # Try splitting with shlex first (handles spaces in paths better), fallback to tk.splitlist
            try: raw_paths = shlex.split(event.data.strip('{}'))
            except ValueError: raw_paths = self.tk.splitlist(event.data.strip('{}')) # Fallback for simple lists
            except Exception as shlex_e: # Catch other potential shlex errors
                 logger.error(f"Shlex parsing failed for drop data: {shlex_e}. Raw data: {event.data}")
                 raw_paths = self.tk.splitlist(event.data.strip('{}')) # Fallback

            filepaths = [p.strip() for p in raw_paths if p.strip()] # Clean whitespace
            video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg", ".webm"); count = 0
            current_files_set = set(self.video_files); skipped_non_video = 0; skipped_duplicates = 0
            for fp in filepaths:
                fp_clean = fp.strip("'\" "); # Remove potential quotes
                try:
                    if fp_clean and os.path.isfile(fp_clean):
                        if fp_clean.lower().endswith(video_extensions):
                            if fp_clean not in current_files_set:
                                self.video_files.append(fp_clean); self.video_listbox.insert(END, os.path.basename(fp_clean)); current_files_set.add(fp_clean); count += 1; logger.debug(f"Added dropped video: {fp_clean}")
                            else: skipped_duplicates += 1; logger.debug(f"Skipped duplicate drop: {fp_clean}")
                        else: skipped_non_video += 1; logger.debug(f"Skipped drop (not video ext): {fp_clean}")
                    elif fp_clean: logger.debug(f"Skipped drop (not valid file path): {fp_clean}")
                except Exception as file_check_err: logger.warning(f"Error checking dropped path '{fp_clean}': {file_check_err}")
            status_parts = [];
            if count > 0: status_parts.append(f"Added {count} video(s)")
            if skipped_non_video > 0: status_parts.append(f"skipped {skipped_non_video} non-video")
            if skipped_duplicates > 0: status_parts.append(f"skipped {skipped_duplicates} duplicate(s)")
            if status_parts: self.status_label.configure(text=f"Drop: {', '.join(status_parts)}."); logger.info(f"Drop event: {', '.join(status_parts)}.")
            else: self.status_label.configure(text="Drop: No valid new videos found."); logger.info("Drop event: No valid new videos found.")
        except Exception as e: logger.error(f"Error handling drop: {e}\nRaw: {event.data}", exc_info=True); tk_write(f"Error processing dropped files:\n{e}", parent=self, level="warning")
        return event.action # Required by TkinterDnD

    def _add_videos_manual(self):
        if self.is_processing: return
        filetypes = (("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.mpg *.mpeg *.webm"), ("All files", "*.*"))
        filepaths = filedialog.askopenfilename(title="Select Source Video Ingredients", filetypes=filetypes, multiple=True)
        count = 0; current_files_set = set(self.video_files)
        if filepaths:
            for fp in filepaths:
                fp_clean = fp.strip("'\" ")
                try:
                    if fp_clean and os.path.isfile(fp_clean):
                         video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg", ".webm")
                         if fp_clean.lower().endswith(video_extensions):
                             if fp_clean not in current_files_set: self.video_files.append(fp_clean); self.video_listbox.insert(END, os.path.basename(fp_clean)); current_files_set.add(fp_clean); count += 1; logger.debug(f"Added selected file: {fp_clean}")
                             else: logger.debug(f"Skipped duplicate selected file: {fp_clean}")
                         else: logger.warning(f"Skipped non-video selected file: {fp_clean}")
                    elif fp_clean: logger.warning(f"Selected item not a valid file: {fp_clean}")
                except Exception as e: logger.warning(f"Could not add selected file '{fp_clean}': {e}")
            if count > 0: self.status_label.configure(text=f"Added {count} video ingredient(s)."); logger.info(f"Added {count} video(s) via dialog.")

    def _remove_selected_videos(self):
         if self.is_processing: return
         selected_indices = self.video_listbox.curselection()
         if not selected_indices: self.status_label.configure(text="Select videos to remove.", text_color="yellow"); return
         # Remove items from back to front to avoid index shifting issues
         indices_to_remove = sorted(list(selected_indices), reverse=True); removed_count = 0
         for i in indices_to_remove:
             if 0 <= i < len(self.video_files):
                 try: removed_path = self.video_files.pop(i); self.video_listbox.delete(i); removed_count += 1; logger.debug(f"Removed video: {removed_path}")
                 except Exception as e: logger.error(f"Error removing item at index {i}: {e}")
             else: logger.warning(f"Attempted to remove invalid listbox index {i}")
         if removed_count > 0: self.status_label.configure(text=f"Removed {removed_count} ingredient(s).", text_color=self.diamond_white); logger.info(f"Removed {removed_count} video(s).")

    def _clear_video_list(self):
        if self.is_processing or not self.video_files: return
        if messagebox.askyesno("Confirm Clear", "Discard all current video ingredients?", parent=self):
            self.video_files.clear(); self.video_listbox.delete(0, END)
            self.status_label.configure(text="Ingredient list cleared.", text_color=self.diamond_white)
            logger.info("Cleared all videos from the list.")

    # --- Configuration Management ---
    def _get_analysis_config(self) -> AnalysisConfig:
        logger.debug("Gathering analysis configuration from UI...")
        config_dict = {key: var.get() for key, var in self.slider_vars.items()}
        config_dict["sequencing_mode"] = self.mode_var.get()
        config_dict["resolution_height"] = 256 # Fixed analysis resolution
        config_dict["resolution_width"] = 256  # Fixed analysis resolution

        try:
            valid_keys = {f.name for f in AnalysisConfig.__dataclass_fields__.values()}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

            # Type conversions and validation
            for key, field_info in AnalysisConfig.__dataclass_fields__.items():
                if key in filtered_dict:
                    current_val = filtered_dict[key]
                    target_type = field_info.type
                    origin_type = getattr(target_type, '__origin__', None)
                    possible_types = getattr(target_type, '__args__', (target_type,)) if origin_type else (target_type,)

                    converted = False
                    for p_type in possible_types:
                        try:
                            # Handle specific type conversions more carefully
                            if p_type == int and not isinstance(current_val, int):
                                filtered_dict[key] = int(round(float(current_val)))
                                converted = True; break
                            elif p_type == float and not isinstance(current_val, float):
                                filtered_dict[key] = float(current_val)
                                converted = True; break
                            elif p_type == bool and not isinstance(current_val, bool):
                                # Handle 0/1 or True/False from BooleanVar/IntVar
                                filtered_dict[key] = bool(int(current_val)) if isinstance(current_val, (int, float)) else bool(current_val)
                                converted = True; break
                            elif isinstance(current_val, p_type): # Already correct type
                                converted = True; break
                        except (ValueError, TypeError) as conv_err:
                            logger.debug(f"Conversion to {p_type} failed for key '{key}', value '{current_val}': {conv_err}")
                            continue # Try next possible type

                    if not converted:
                         logger.warning(f"Invalid type or conversion failed for AnalysisConfig key '{key}': Value='{current_val}' (Type: {type(current_val)}), Expected='{target_type}'. Using default if available.")
                         # Remove the invalid entry so dataclass uses default
                         if key in filtered_dict: del filtered_dict[key]

            # Create dataclass instance
            cfg = AnalysisConfig(**filtered_dict)
            logger.debug(f"Generated AnalysisConfig: {cfg}")
            self.analysis_config = cfg # Store the generated config
            return cfg
        except Exception as e:
            logger.error(f"Failed to create AnalysisConfig: {e}", exc_info=True)
            tk_write("Error reading analysis config. Using defaults.", parent=self, level="error")
            self.analysis_config = AnalysisConfig() # Store default on error
            return self.analysis_config

    def _get_render_config(self) -> RenderConfig:
        logger.debug("Gathering render configuration...")
        analysis_cfg = self.analysis_config if self.analysis_config else AnalysisConfig() # Get defaults if not set
        effect_settings = { # Populate from sliders
            "cut": EffectParams(type="cut"),
            "fade": EffectParams(type="fade", tau=self.slider_vars['effect_fade_duration'].get(), psi=0.1, epsilon=0.2),
            "zoom": EffectParams(type="zoom", tau=self.slider_vars['effect_zoom_duration'].get(), psi=self.slider_vars['effect_zoom_impact'].get(), epsilon=self.slider_vars['effect_zoom_perceptual'].get()),
            "pan": EffectParams(type="pan", tau=self.slider_vars['effect_pan_duration'].get(), psi=self.slider_vars['effect_pan_impact'].get(), epsilon=self.slider_vars['effect_pan_perceptual'].get()),
        }
        try:
            cpu_cores = os.cpu_count() or 1; render_threads = max(1, min(cpu_cores // 2, 8)) # Conservative default
            # Get render targets from UI sliders (check existence first)
            target_w = self.slider_vars['render_width'].get() if 'render_width' in self.slider_vars else 1920
            target_h = self.slider_vars['render_height'].get() if 'render_height' in self.slider_vars else 1080
            target_fps = self.slider_vars['render_fps'].get() if 'render_fps' in self.slider_vars else 30

            # Validate render dimensions and FPS
            target_w = max(16, target_w) # Ensure minimum width
            target_h = max(16, target_h) # Ensure minimum height
            target_fps = max(1, target_fps) # Ensure minimum FPS

            cfg = RenderConfig(
                norm_max_velocity=analysis_cfg.norm_max_velocity, norm_max_acceleration=analysis_cfg.norm_max_acceleration,
                effect_settings=effect_settings,
                video_codec='libx264', preset='medium', crf=23, audio_codec='aac', audio_bitrate='192k',
                threads=render_threads,
                resolution_width=target_w, resolution_height=target_h, fps=target_fps
            )
            logger.debug(f"Generated RenderConfig: {cfg}")
            self.render_config = cfg # Store it
            return cfg
        except Exception as e:
            logger.error(f"Failed to create RenderConfig: {e}", exc_info=True)
            tk_write("Error reading render config. Using defaults.", parent=self, level="error"); self.render_config = RenderConfig(); return self.render_config

    def _set_ui_processing_state(self, processing: bool):
        """Disables/Enables UI elements during processing."""
        self.is_processing = processing
        state = "disabled" if processing else "normal"

        widgets_to_toggle = [ # List widgets that need state toggling
            self.beat_track_button, self.add_button, self.remove_button,
            self.clear_button, self.run_button, self.mode_selector # Use named widget
        ]

        # Toggle top-level controls
        for widget in widgets_to_toggle:
            if widget is not None and hasattr(widget, 'configure'):
                 try:
                     # Check if 'state' is a valid option for this widget type
                     # CTkSegmentedButton might not have 'state', handle gracefully
                     if isinstance(widget, (customtkinter.CTkButton, customtkinter.CTkCheckBox)):
                          widget.configure(state=state)
                     elif isinstance(widget, customtkinter.CTkSegmentedButton):
                          # SegmentedButton doesn't have a direct 'state' config like Button
                          # We might need to disable interaction differently if required,
                          # but often just disabling the 'Run' button is enough feedback.
                          # For now, we'll skip disabling it directly to avoid errors.
                          logger.debug(f"Skipping state toggle for CTkSegmentedButton: {widget}")
                          pass
                 except tkinter.TclError as tcl_err:
                      logger.warning(f"TclError configuring widget {widget} state: {tcl_err} (Widget type: {type(widget)})")
                 except Exception as config_err:
                      logger.warning(f"Error configuring widget {widget} state: {config_err}", exc_info=False)

        # Toggle elements within the tabs more safely
        try:
             for tab_name in ["Shared", "Greedy Heuristic", "Physics MC", "Render Settings"]:
                 tab_frame = self.tab_view.tab(tab_name)
                 if not (tab_frame and hasattr(tab_frame, 'winfo_children')): continue

                 # Iterate through all descendants of the tab's scrollable frame content
                 q = list(tab_frame.winfo_children())
                 while q:
                    widget = q.pop(0)
                    if widget is None: continue
                    # Add children to queue
                    if hasattr(widget, 'winfo_children'):
                        q.extend(widget.winfo_children())
                    # Check if widget supports state configuration
                    if hasattr(widget, 'configure') and isinstance(widget, (customtkinter.CTkSlider, customtkinter.CTkCheckBox, customtkinter.CTkRadioButton, customtkinter.CTkButton)):
                        try:
                            # Check if state is a valid option before setting it
                            if 'state' in widget.configure():
                                widget.configure(state=state)
                        except tkinter.TclError: # Catch errors if state is not applicable
                             logger.debug(f"Widget {widget} of type {type(widget)} does not support 'state' config.")
                        except Exception as inner_e:
                             logger.warning(f"Error configuring state for widget {widget}: {inner_e}")

        except Exception as e:
             logger.error(f"Error toggling config UI state in tabs: {e}", exc_info=True)


        # Update Run Button Text
        if self.run_button and hasattr(self.run_button, 'configure'):
            self.run_button.configure(text="Chef is Cooking..." if processing else "4. Compose Video Remix")
        self.update_idletasks() # Force UI update


    # --- Processing Workflow (Logging Enhanced) ---
    def _start_processing(self):
        if self.is_processing: logger.warning("Processing already ongoing."); return
        # Validation
        if not self.beat_track_path or not os.path.isfile(self.beat_track_path): tk_write("Chef needs a valid master audio track!", parent=self, level="warning"); return
        if not self.video_files: tk_write("Chef needs video ingredients!", parent=self, level="warning"); return
        # Validate video files exist before starting
        valid_video_files = [f for f in self.video_files if os.path.isfile(f)]; invalid_count = len(self.video_files) - len(valid_video_files)
        if invalid_count > 0:
             logger.warning(f"{invalid_count} invalid video path(s) removed from list.")
             self.video_files = valid_video_files
             # Update listbox visually
             self.video_listbox.delete(0, END); [self.video_listbox.insert(END, os.path.basename(vf)) for vf in self.video_files]
             tk_write(f"{invalid_count} invalid video path(s) were removed from the list.", parent=self, level="warning")
             if not self.video_files: tk_write("No valid video ingredients remaining!", parent=self, level="error"); return # Abort if no valid files left
        # Start
        logger.info("Starting processing workflow...")
        self.analysis_config = self._get_analysis_config(); self.render_config = self._get_render_config()
        if not self.analysis_config or not self.render_config: logger.error("Failed config gathering. Aborting."); return
        self._set_ui_processing_state(True); self.status_label.configure(text="Chef is prepping...", text_color=self.diamond_white)
        # Reset state variables
        self.master_audio_data = None; self.all_potential_clips = []; self.analysis_futures = []; self.futures_map = {}; self.total_tasks = 0; self.completed_tasks = 0
        self.shutdown_executor() # Ensure clean state before starting new pool
        logger.info("Starting master audio analysis thread..."); self.status_label.configure(text="Analyzing master audio...")
        # Ensure previous thread is finished before starting new one
        if self.processing_thread is not None and self.processing_thread.is_alive():
            logger.warning("Waiting for existing processing thread to finish...")
            self.processing_thread.join(timeout=5.0) # Wait up to 5 seconds
            if self.processing_thread.is_alive():
                 logger.error("Existing processing thread did not finish. Aborting new process start.")
                 self._set_ui_processing_state(False)
                 tk_write("Could not start new process, another is still running.", parent=self, level="error")
                 return

        self.processing_thread = threading.Thread(target=self._analyze_master_audio, name="AudioAnalysisThread", daemon=True); self.processing_thread.start()


    def _analyze_master_audio(self):
        try:
            audio_analyzer = BBZAudioUtils(); timestamp = time.strftime("%Y%m%d%H%M%S")
            # Use analysis_subdir for temporary audio files too
            temp_audio_for_master = os.path.join(self.analysis_subdir, f"temp_master_audio_{timestamp}.wav")
            audio_file_to_analyze = self.beat_track_path; needs_cleanup = False
            video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg', '.webm')

            if not os.path.isfile(audio_file_to_analyze):
                raise FileNotFoundError(f"Master source file invalid or not found: {audio_file_to_analyze}")

            if self.beat_track_path.lower().endswith(video_extensions):
                self.after(0, lambda: self.status_label.configure(text="Extracting audio from master..."))
                extracted_path = audio_analyzer.extract_audio(self.beat_track_path, temp_audio_for_master)
                if not extracted_path: raise RuntimeError(f"Failed audio extraction from master video: {self.beat_track_path}")
                audio_file_to_analyze = extracted_path; needs_cleanup = True; logger.info(f"Using extracted audio: {audio_file_to_analyze}")

            self.after(0, lambda: self.status_label.configure(text="Analyzing master audio features..."))
            logger.info(f"Starting Librosa analysis on: {os.path.basename(audio_file_to_analyze)}")

            if not self.analysis_config:
                 logger.error("Analysis config not available for master audio analysis. Aborting.")
                 raise RuntimeError("Analysis config missing.")

            self.master_audio_data = audio_analyzer.analyze_audio(audio_file_to_analyze, self.analysis_config)
            if not self.master_audio_data: raise RuntimeError("Audio analysis failed (returned None).")

            logger.info("Master audio analysis successful."); self.after(0, self._start_parallel_video_analysis) # Schedule next step on main thread

        except Exception as e:
            error_msg = f"Error processing master audio: {type(e).__name__}"
            logger.error(f"Error processing master audio: {e}", exc_info=True)
            self.after(0, lambda em=error_msg, ex=e: [ # Use lambda defaults to capture exception details
                self.status_label.configure(text=f"Error: {em}", text_color="orange"),
                tk_write(f"Failed master audio processing.\n\n{ex}", parent=self, level="error"),
                self._set_ui_processing_state(False) # Re-enable UI on error
            ])
        finally:
            # Cleanup temporary extracted audio file
            if 'needs_cleanup' in locals() and needs_cleanup and 'audio_file_to_analyze' in locals() and os.path.exists(audio_file_to_analyze):
                try: os.remove(audio_file_to_analyze); logger.debug("Cleaned up temp master audio.")
                except OSError as del_err: logger.warning(f"Failed to remove temp audio {audio_file_to_analyze}: {del_err}")

    def _start_parallel_video_analysis(self):
        """Schedules the parallel video analysis to run in a separate thread."""
        self.status_label.configure(text=f"Analyzing {len(self.video_files)} video ingredients...")
        # Ensure previous thread is finished before starting new one
        if self.processing_thread is not None and self.processing_thread.is_alive():
             logger.warning("Waiting for audio thread before starting video analysis pool...")
             self.processing_thread.join(timeout=5.0)
             if self.processing_thread.is_alive():
                  logger.error("Audio thread did not finish. Aborting video analysis.")
                  self.after(0, lambda: [self.status_label.configure(text="Error: Thread conflict.", text_color="red"), self._set_ui_processing_state(False)])
                  return

        # Start the pool management in a new thread
        self.processing_thread = threading.Thread(target=self._run_parallel_video_analysis_pool, name="VideoAnalysisPoolMgr", daemon=True)
        self.processing_thread.start()

    def _run_parallel_video_analysis_pool(self):
        """Manages the ProcessPoolExecutor for video analysis."""
        cpu_cores = os.cpu_count() or 1; max_workers = max(1, min(cpu_cores - 1, 6)) # Leave one core for main process/OS, max 6
        logger.info(f"Starting parallel video analysis using up to {max_workers} workers.")
        self.analysis_futures = []; self.futures_map = {}; self.total_tasks = len(self.video_files); self.completed_tasks = 0

        # Crucial checks before submitting jobs
        if not self.analysis_config or not self.master_audio_data:
            logger.critical("Analysis config or master audio data missing before submitting jobs. Aborting.")
            self.after(0, lambda: [
                self.status_label.configure(text="Error: Config missing.", text_color="red"),
                tk_write("Internal error: Configuration data unavailable.", parent=self, level="error"),
                self._set_ui_processing_state(False)
            ])
            return
        # Verify data is pickleable for multiprocessing
        try:
             import pickle; pickle.dumps(self.master_audio_data); pickle.dumps(self.analysis_config); logger.debug("Data appears pickleable.")
        except Exception as pickle_e:
             err_msg = f"Internal Error: Cannot send data to workers (pickling failed).\n\n{pickle_e}"; logger.critical(f"Pickle Error: {pickle_e}", exc_info=True)
             self.after(0, lambda: [tk_write(err_msg, parent=self, level="error"), self._set_ui_processing_state(False)])
             return

        # Create executor and submit tasks
        self.shutdown_executor(); # Ensure any old executor is gone
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        submitted_count = 0; submission_errors = 0
        for vid_path in self.video_files:
            if not os.path.exists(vid_path): logger.error(f"Skipping non-existent file: {vid_path}"); submission_errors += 1; continue
            try:
                # Pass analysis_subdir directly to the worker
                future = self.executor.submit(process_single_video, vid_path, self.master_audio_data, self.analysis_config, self.analysis_subdir)
                self.futures_map[future] = vid_path; self.analysis_futures.append(future); submitted_count += 1
            except Exception as submit_e: logger.error(f"ERROR submitting job for {os.path.basename(vid_path)}: {submit_e}", exc_info=True); submission_errors += 1

        self.total_tasks = submitted_count # Adjust based on actual submissions
        if not self.analysis_futures:
            final_msg = "Error: Failed to start any analysis jobs."
            if submission_errors > 0: final_msg += f" ({submission_errors} submission errors)"
            logger.error(f"Aborting parallel analysis: {final_msg}")
            self.after(0, lambda msg=final_msg: [
                self.status_label.configure(text=msg, text_color="red"),
                tk_write(msg, parent=self, level="error"),
                self._set_ui_processing_state(False)
            ])
            self.shutdown_executor() # Shutdown pool even if no jobs submitted
            return

        logger.info(f"Submitted {submitted_count} analysis jobs.");
        # Schedule the first status check from the main thread
        self.after(0, lambda: self.status_label.configure(text=f"Analyzing ingredients... 0/{self.total_tasks} (0.0%)"))
        self.after(1000, self._check_analysis_status) # Start polling after 1 second

    def _check_analysis_status(self):
        """Periodically checks the status of analysis futures."""
        if not self.is_processing or not hasattr(self, 'analysis_futures') or not self.analysis_futures:
             logger.debug("Stopping analysis check (not processing or no futures).")
             self.shutdown_executor()
             return

        try:
            # Check which futures are done without blocking
            done_futures = [f for f in self.analysis_futures if f.done()]
            self.completed_tasks = len(done_futures)

            # Update status label
            if self.total_tasks > 0:
                progress = (self.completed_tasks / self.total_tasks) * 100
                self.status_label.configure(text=f"Analyzing... {self.completed_tasks}/{self.total_tasks} ({progress:.1f}%)")
            else:
                self.status_label.configure(text="Waiting for analysis results...") # Should not happen if checks are right

            # Check if all tasks are completed
            if self.completed_tasks == self.total_tasks:
                logger.info("--- All Video Analyses Finished ---")
                self.shutdown_executor() # Shutdown pool now that tasks are done

                logger.info("Collecting analysis results...")
                self.all_potential_clips = []
                success_count = 0; fail_count = 0; failed_videos = []

                # Process results from futures
                for future in self.analysis_futures:
                    vid_path_for_log = self.futures_map.get(future, "Unknown Video")
                    try:
                        video_path, status, potential_clips_result = future.result(timeout=10) # Use timeout
                        logger.info(f"Result for {os.path.basename(video_path)}: {status}")
                        # Check status string and if result is a list
                        if "Analysis OK" in status and isinstance(potential_clips_result, list):
                            valid_clips = [clip for clip in potential_clips_result if isinstance(clip, ClipSegment)]
                            if valid_clips:
                                self.all_potential_clips.extend(valid_clips)
                                success_count += 1
                            else:
                                # Analysis was OK but yielded no valid clips (might be expected for some videos)
                                logger.info(f"Analysis OK for {os.path.basename(video_path)} but no usable clips generated.")
                                # Don't count as failure unless this is unexpected
                        else: # Status indicates failure or result is not a list
                            fail_count += 1
                            failed_videos.append(os.path.basename(video_path))
                            logger.error(f"Analysis failed or returned invalid result for {os.path.basename(video_path)}: Status '{status}'")

                    except concurrent.futures.TimeoutError:
                        logger.error(f"Timeout retrieving result for {os.path.basename(vid_path_for_log)}.")
                        fail_count += 1; failed_videos.append(os.path.basename(vid_path_for_log))
                    except Exception as e:
                        logger.error(f"Worker error retrieving result for {os.path.basename(vid_path_for_log)}: {type(e).__name__} - {e}", exc_info=False)
                        fail_count += 1; failed_videos.append(os.path.basename(vid_path_for_log))

                logger.info(f"Collected {len(self.all_potential_clips)} potential clips from {success_count} source(s).")

                if fail_count > 0:
                    fail_msg = f"{fail_count} video(s) failed analysis or timed out. Check logs."
                    if failed_videos: fail_msg += f"\nFailed: {', '.join(failed_videos[:5])}{'...' if len(failed_videos) > 5 else ''}"
                    tk_write(fail_msg, parent=self, level="warning")

                # Check if any usable clips were generated before proceeding
                if not self.all_potential_clips:
                    self.after(0, lambda: [
                        self.status_label.configure(text="Error: No usable clips found.", text_color="orange"),
                        tk_write("Analysis finished, but no usable clips identified across all videos.", parent=self, level="error"),
                        self._set_ui_processing_state(False) # Re-enable UI
                    ])
                    logger.error("Aborting sequence building: No potential clips identified.")
                    return

                # Proceed to Sequence Building on the main thread via 'after'
                self.after(0, self._schedule_sequence_build)

            else:
                # Not all tasks done, schedule the next check
                self.after(1000, self._check_analysis_status)

        except Exception as poll_err:
            logger.error(f"Error checking analysis status: {poll_err}", exc_info=True)
            self.after(0, lambda: [
                self.status_label.configure(text="Error checking status.", text_color="red"),
                self._set_ui_processing_state(False) # Re-enable UI on error
            ])
            self.shutdown_executor()

    def _schedule_sequence_build(self):
         """Schedules the sequence building and rendering task."""
         self.status_label.configure(text="Chef is composing the sequence...")
         logger.info("Starting sequence building thread...")
         # Ensure previous thread (pool manager) is finished
         if self.processing_thread is not None and self.processing_thread.is_alive():
             logger.warning("Waiting for analysis pool manager thread before starting sequence build...")
             self.processing_thread.join(timeout=5.0)
             if self.processing_thread.is_alive():
                  logger.error("Analysis pool manager thread did not finish. Aborting sequence build.")
                  self.after(0, lambda: [self.status_label.configure(text="Error: Thread conflict.", text_color="red"), self._set_ui_processing_state(False)])
                  return

         self.processing_thread = threading.Thread(target=self._build_final_sequence_and_video, name="SequenceBuildRenderThread", daemon=True)
         self.processing_thread.start()


    def _build_final_sequence_and_video(self):
        """Builds sequence and renders video (using MoviePy) in a background thread."""
        try:
            # --- Pre-checks ---
            if not self.analysis_config or not self.master_audio_data or not self.render_config:
                raise RuntimeError("Missing config or audio data for sequence building.")
            if not self.all_potential_clips:
                raise RuntimeError("No potential clips available for sequence building.")

            # --- Instantiate Builder ---
            selected_mode = self.analysis_config.sequencing_mode; builder: Optional[Union[SequenceBuilderGreedy, SequenceBuilderPhysicsMC]] = None
            logger.info(f"Instantiating sequence builder: {selected_mode}")
            self.after(0, lambda: self.status_label.configure(text=f"Composing ({selected_mode})..."))

            if selected_mode == "Physics Pareto MC":
                builder = SequenceBuilderPhysicsMC(self.all_potential_clips, self.master_audio_data, self.analysis_config)
                builder.effects = self.render_config.effect_settings # Pass RenderConfig effects
                logger.debug(f"Physics MC using effects from RenderConfig: {builder.effects}")
            else: # Default to Greedy
                 builder = SequenceBuilderGreedy(self.all_potential_clips, self.master_audio_data, self.analysis_config)

            # --- Build Sequence ---
            logger.info("Building sequence..."); final_sequence = builder.build_sequence()

            if not final_sequence:
                logger.error(f"Sequence building failed ({selected_mode}). No clips selected.")
                self.after(0, lambda: [
                     self.status_label.configure(text=f"Error: Failed sequence ({selected_mode}).", text_color="orange"),
                     tk_write(f"Could not create sequence using {selected_mode} mode. No clips were selected.", parent=self, level="error"),
                     self._set_ui_processing_state(False)])
                return

            logger.info(f"Sequence built ({len(final_sequence)} clips). Preparing render."); self.after(0, lambda: self.status_label.configure(text=f"Preparing final render..."))

            # --- Prepare for Render ---
            timestamp = time.strftime("%Y%m%d_%H%M%S"); mode_tag = "Greedy" if selected_mode == "Greedy Heuristic" else "PhysicsMC"
            output_video_path = os.path.join(self.render_subdir, f"videous_chef_{mode_tag}_{timestamp}.mp4")
            master_audio_path_for_render = self.beat_track_path; temp_audio_render = None; audio_util = BBZAudioUtils()
            video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg', '.webm')

            # Re-extract audio if master source was a video file
            if self.beat_track_path.lower().endswith(video_extensions):
                self.after(0, lambda: self.status_label.configure(text="Extracting audio for render..."))
                temp_audio_render = os.path.join(self.render_subdir, f"master_audio_render_{timestamp}.wav")
                extracted = audio_util.extract_audio(self.beat_track_path, temp_audio_render)
                if not extracted or not os.path.exists(extracted):
                    raise RuntimeError("Failed audio extraction for render.")
                master_audio_path_for_render = extracted
                logger.info(f"Using re-extracted audio for render: {master_audio_path_for_render}")
            elif not os.path.isfile(master_audio_path_for_render):
                raise FileNotFoundError(f"Master audio for render not found or is invalid: {master_audio_path_for_render}")

            # --- Build Video using MoviePy ---
            self.after(0, lambda: self.status_label.configure(text=f"Rendering final dish ({selected_mode})..."))
            logger.info(f"Starting final video render (MoviePy make_frame) to {output_video_path}...")

            # Call the external rendering function
            buildSequenceVideo(final_sequence, output_video_path, master_audio_path_for_render, self.render_config)

            # --- Success ---
            final_msg = f"Success ({mode_tag})! Saved:\n{os.path.basename(output_video_path)}"; final_color = "light green"
            logger.info(f"Video composition successful: {output_video_path}")
            self.after(0, lambda: self.status_label.configure(text=final_msg, text_color=final_color))
            tk_write(f"Video Remix Successful!\nMode: {selected_mode}\nOutput:\n{output_video_path}", parent=self, level="info")

        except Exception as e:
            error_type_name = type(e).__name__; error_message = str(e)
            final_msg = f"Error during compose/render: {error_type_name}"
            final_color = "orange"
            logger.critical(f"!!! FATAL ERROR during Sequence/Render !!!", exc_info=True)
            # Schedule UI update and message box on main thread
            self.after(0, lambda et=error_type_name, em=error_message: [
                 self.status_label.configure(text=final_msg, text_color=final_color),
                 tk_write(f"Chef problem during composition/render:\n\n{et}: {em}\n\nCheck log/console.", parent=self, level="error")])
        finally:
            # Cleanup temporary render audio
            if 'temp_audio_render' in locals() and temp_audio_render and os.path.exists(temp_audio_render):
                try: os.remove(temp_audio_render); logger.debug("Cleaned temp render audio.")
                except OSError as del_err: logger.warning(f"Failed to remove temp render audio: {del_err}")
            # Ensure UI is re-enabled regardless of success or failure
            self.after(0, self._set_ui_processing_state, False); logger.info("Build/Render thread finished.")


    def on_closing(self):
        logger.info("Shutdown requested via window close...")
        if self.is_processing:
            if messagebox.askyesno("Confirm Exit", "Processing ongoing.\nExiting might corrupt output or leave processes running.\n\nExit anyway?", parent=self):
                 logger.warning("Forcing shutdown during processing.")
                 self.shutdown_executor() # Request pool shutdown
                 # TODO: Add thread signaling if complex operations need graceful stop
                 self.destroy()
            else: logger.info("Shutdown cancelled by user."); return
        else:
             logger.info("Closing application normally.")
             self.shutdown_executor() # Ensure clean shutdown even if not processing
             self.destroy()
        logger.info("Application closing sequence complete.")


    def shutdown_executor(self):
         """Shuts down the ProcessPoolExecutor gracefully."""
         if hasattr(self, 'executor') and self.executor:
             # Check if executor is already shutting down or shut down
             # ProcessPoolExecutor doesn't have a public _shutdown flag reliably across versions
             # Instead, try a shutdown and catch potential errors if already shut down
             logger.info("Attempting to shut down process pool executor...")
             cancelled_count = 0
             if hasattr(self, 'analysis_futures'):
                 for f in self.analysis_futures:
                     if not f.done():
                          if f.cancel(): cancelled_count += 1
             logger.debug(f"Attempted to cancel {cancelled_count} pending future(s).")
             try:
                 # Use cancel_futures=True for Python 3.9+
                 if sys.version_info >= (3, 9):
                      self.executor.shutdown(wait=False, cancel_futures=True)
                 else:
                      self.executor.shutdown(wait=False) # Older versions just signal workers
                 logger.info("Executor shutdown initiated.")
             except Exception as e:
                  # Catch errors if shutdown is called multiple times or on an invalid state
                  logger.error(f"Error during executor shutdown (possibly already shut down): {e}", exc_info=False)
             finally:
                 self.executor = None # Clear the reference
         else: logger.debug("No active executor instance found to shut down.")


# ========================================================================
#                      REQUIREMENTS.TXT Block
# ========================================================================
"""
# requirements.txt for Videous Chef v4.7.3 (MoviePy + Fixes)

# Core UI & Analysis
customtkinter>=5.2.0,<6.0.0           # Check for latest stable 5.x
opencv-python>=4.6.0,<5.0.0           # Check MediaPipe/MoviePy compatibility
numpy>=1.21.0,<2.0.0                  # Avoid numpy 2.0 for now (major changes)
librosa>=0.9.0,<0.11.0                # Allows 0.9.x for 'k' arg in agglomerative, 0.10+ preferred but handled
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

# Video Reading & Rendering (MoviePy)
moviepy>=1.0.3                         # For video reading & rendering

# NOTE 1: Ensure FFmpeg (the executable) is installed separately and accessible in your system's PATH!
# MoviePy heavily relies on FFmpeg as its backend for writing files and audio operations.
# Download from: https://ffmpeg.org/download.html

# NOTE 2: On macOS with Apple Silicon (M1/M2/M3), ensure PyTorch is installed correctly
# for MPS support if desired (often requires specific build or nightly). Check PyTorch docs.
# Example: pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# NOTE 3: Consider creating a virtual environment (e.g., venv, conda) to manage these dependencies.
# Example setup:
# python3 -m venv venv_videous
# source venv_videous/bin/activate  # (or venv_videous\Scripts\activate on Windows)
# pip install -r requirements.txt
"""

# ========================================================================
#                       APPLICATION ENTRY POINT
# ========================================================================
if __name__ == "__main__":
    # --- Multiprocessing Setup (Crucial for PyInstaller/cx_Freeze) ---
    multiprocessing.freeze_support()
    try:
        # Force 'spawn' method on non-Windows if available, as 'fork' can cause issues with complex libs
        if sys.platform != 'win32':
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                if 'spawn' in multiprocessing.get_all_start_methods():
                    multiprocessing.set_start_method('spawn', force=True)
                    print(f"INFO: Set multiprocessing start method to 'spawn'.")
                else:
                     print(f"WARNING: 'spawn' start method not available. Using default: {multiprocessing.get_start_method()}.")
            else: print(f"INFO: Multiprocessing start method already 'spawn'.")
        else: print(f"INFO: Using default multiprocessing method 'spawn' on Windows.")
    except Exception as E:
         print(f"WARNING: Error setting multiprocessing method: {E}. Using default: {multiprocessing.get_start_method()}.")

    # --- Logging Setup ---
    print("--- Videous Chef v4.7.3 (MoviePy + Fixes) Starting ---") # Updated print version
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s [%(threadName)s]') # Added thread name
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger(); root_logger.setLevel(logging.DEBUG); root_logger.handlers.clear(); root_logger.addHandler(console_handler) # Clear existing handlers
    try: # File handler
        log_dir = "logs"; os.makedirs(log_dir, exist_ok=True); log_file = os.path.join(log_dir, f"videous_chef_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8'); file_handler.setFormatter(log_formatter); file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler); logger.info(f"Logging to console (INFO+) and file (DEBUG+): {log_file}")
    except Exception as log_setup_e: logger.error(f"Failed file logging: {log_setup_e}"); logger.info("Logging to console only.")

    try: # --- STARTUP CHECKS ---
        logger.info("Checking critical dependencies...")
        missing_deps = []; critical_deps_list = ["cv2", "numpy", "customtkinter", "moviepy", "torch", "mediapipe", "librosa", "soundfile", "scipy"]
        # Updated deps list based on imports
        deps = {"customtkinter": "customtkinter", "opencv-python": "cv2", "numpy": "numpy", "librosa": "librosa", "mediapipe": "mediapipe", "tkinterdnd2-universal": "tkinterdnd2", "soundfile": "soundfile", "scipy": "scipy", "torch": "torch", "torchaudio": "torchaudio", "torchvision": "torchvision", "timm": "timm", "matplotlib": "matplotlib", "tqdm": "tqdm", "moviepy": "moviepy"}
        critical_missing = False; all_found = True
        for pkg_name, mod_name in deps.items():
            try: __import__(mod_name); logger.debug(f"  [OK] {mod_name} ({pkg_name})")
            except ImportError as imp_err:
                logger.error(f"  [FAIL] {mod_name} ({pkg_name}): {imp_err}")
                missing_deps.append(pkg_name); all_found = False
                if mod_name in critical_deps_list: critical_missing = True; logger.critical(f"*** Critical dependency '{mod_name}' missing/failed import. ***")
            except Exception as other_err:
                logger.error(f"  [FAIL] {mod_name} ({pkg_name}) - Other Error: {other_err}", exc_info=False)
                missing_deps.append(f"{pkg_name} (Error:{type(other_err).__name__})"); all_found = False
                if mod_name in critical_deps_list: critical_missing = True; logger.critical(f"*** Critical dependency '{mod_name}' failed import with error. ***")

        if critical_missing:
             err_msg = f"Critical dependencies missing/failed: {', '.join([p for p, m in deps.items() if m in critical_deps_list and m not in sys.modules])}\nInstall via: pip install <package_name>\nExiting."; logger.critical(err_msg)
             root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showerror("Dependency Error", err_msg); root_err.destroy(); sys.exit(1)
        elif not all_found:
             install_cmd = f"pip install {' '.join(missing_deps)}"
             err_msg = f"Missing non-critical dependencies: {', '.join(missing_deps)}\nSome features might not work.\nTry: {install_cmd}"; logger.warning(err_msg)
             root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showwarning("Dependency Warning", err_msg); root_err.destroy()
        else: logger.info("All listed dependencies checked successfully.")

        # Check FFmpeg
        try:
            logger.debug("Checking for FFmpeg executable...");
            # Use shell=True for simpler cross-platform path handling, but be aware of security if command was dynamic
            result = subprocess.run("ffmpeg -version", shell=True, capture_output=True, text=True, check=False, timeout=5, encoding='utf-8')
            if result.returncode != 0 or "ffmpeg version" not in result.stdout.lower(): raise FileNotFoundError("FFmpeg basic check failed.")
            logger.info("FFmpeg executable found and seems operational.")
        except FileNotFoundError:
            err_msg = "FFmpeg executable not found or failed basic check.\nInstall FFmpeg and add to your system's PATH.\nMoviePy relies heavily on it.\nDownload: https://ffmpeg.org/download.html\nExiting."; logger.critical(err_msg)
            root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showerror("Dependency Error", err_msg); root_err.destroy(); sys.exit(1)
        except subprocess.TimeoutExpired: logger.warning("FFmpeg check timed out. Ensure it's working correctly."); root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showwarning("Dependency Warning", "FFmpeg check timed out."); root_err.destroy()
        except Exception as ffmpeg_e: logger.error(f"Error checking FFmpeg: {ffmpeg_e}", exc_info=True); root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showwarning("Dependency Warning", f"Error checking FFmpeg: {ffmpeg_e}"); root_err.destroy()

        # Check PyTorch Backend
        try:
            logger.info(f"PyTorch version: {torch.__version__}")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): logger.info("PyTorch MPS backend available (Apple Silicon).")
            elif torch.cuda.is_available(): logger.info(f"PyTorch CUDA backend available. Devices: {torch.cuda.device_count()}")
            else: logger.info("PyTorch CPU backend will be used.")
        except Exception as torch_check_e:
            logger.warning(f"Could not fully check PyTorch backend: {torch_check_e}")


        # Run App
        logger.info("Initializing Application UI...")
        app = VideousApp(); logger.info("Starting Tkinter main loop...")
        app.mainloop()

    except SystemExit as se: logger.warning(f"Application exited during startup (Code: {se.code}).")
    except Exception as e: # Catch unexpected startup errors
        logger.critical(f"!!! UNHANDLED STARTUP ERROR !!!", exc_info=True)
        try: # Try to show a graphical error message
            root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showerror("Startup Error", f"Application failed to start:\n\n{type(e).__name__}: {e}\n\nCheck log/console."); root_err.destroy()
        except Exception as msg_err:
            print(f"\n\n!!! CRITICAL STARTUP ERROR: {type(e).__name__}: {e} !!!")
            
            '''Okay, I'll explain the provided Python script, focusing on how it can be tailored for industry-grade music video editing, aiming for Taylor Swift-level quality. This will involve improvements to efficiency, features, and overall robustness. **1. Overview of the Script** The script represents a video editing tool that automatically creates remixes by analyzing video and audio, identifying key segments, and stitching them together. It uses several libraries: * **Core Libraries:** `tkinter` (UI), `cv2` (video processing), `numpy` (numerical operations), `moviepy` (video editing), `librosa` (audio analysis), `mediapipe` (pose and face detection), `torch` (MiDaS depth estimation). * **Analysis:** It extracts features like motion, audio energy, face size, and depth. It then scores these features to identify potential clips. * **Sequencing:** It uses either a "Greedy Heuristic" or "Physics Pareto MC" approach to select and arrange clips into a final sequence. * **Rendering:** The `moviepy` library is used to combine the selected video clips and audio into a final video file. **2. Areas for Improvement (Taylor Swift Level)** To reach a professional level, here's a breakdown of key areas and specific recommendations: * **A. Enhanced Visual Analysis** * **Object Detection/Segmentation:** *Industry Standard:* Integrate more sophisticated object detection (e.g., using YOLO, Detectron2) and segmentation. This would allow the script to identify specific objects (cars, instruments, people) and track them across shots. *Taylor Level:* Use this information to create dynamic edits that focus on key visual elements. Example: automatically cut to a close-up of a guitar during a solo, or highlight dancers. * **Scene Recognition:** *Industry Standard:* Implement scene recognition to categorize shots (e.g., concert stage, outdoor scene, indoor studio). *Taylor Level:* Use scene information to ensure visual coherence. If the song has a shift in tone, the video can shift from high energy concert footage to a more intimate, behind-the-scenes feel using different scenes. * **Color Grading Analysis:** Analyze the color palettes of different shots. *Taylor Level:* Correct and match the color grading from the input videos to provide a consistent visual feel of the video * **Advanced Motion Analysis:** Optical flow is good but limited. *Industry Standard:* Consider using motion tracking algorithms to track specific points in the video. *Taylor Level:* Stabilize shaky footage automatically, or create more complex motion-based transitions. * **B. More Sophisticated Audio Analysis** * **Beat Tracking and Musical Structure:** While `librosa` is used, the beat tracking and structure analysis can be improved. *Industry Standard:* Implement a Hidden Markov Model (HMM) or similar technique for more robust beat tracking, key detection, and chord recognition. *Taylor Level:* Use the key detection to automatically adjust color grading to match the song's key. Use chord changes to trigger specific cuts or effects. * **Vocal Detection and Isolation:** *Industry Standard:* Detect vocal segments and potentially isolate the vocals. *Taylor Level:* Ensure cuts align with vocal phrases, or create effects that sync with the vocals. * **C. Improved Sequencing and Editing Logic** * **Rule-Based Editing:** *Industry Standard:* Allow for rule-based editing. For example, the user could specify that certain types of shots (e.g., close-ups) should only be used during specific parts of the song. *Taylor Level:* Implement this via scripting or a more advanced UI. * **Motion Matching:** *Industry Standard:* Match the motion between shots. *Taylor Level:* Try to cut from one shot to another where the motion is similar to create a smoother, more visually appealing transition. This requires analyzing the motion vectors and finding good matches. * **Intelligent Transitions:** The current script seems to primarily use cuts. *Industry Standard:* Add a wider variety of transitions (fades, wipes, zooms, etc.). *Taylor Level:* Use AI to select the *best* transition based on the content of the shots. * **Rhythm-Aware Editing:** *Taylor Level:* Develop more sophisticated algorithms to ensure edits are perfectly synchronized with the music. This might involve micro-adjustments to clip timings. * **D. Enhanced Effects and Visuals** * **AI-Powered Effects:** *Taylor Level:* Integrate AI-powered effects like style transfer (to change the visual style of the video), or automated rotoscoping (to isolate objects). * **3D Integration:** *Taylor Level:* Integrate 3D elements or camera tracking to create more dynamic and visually interesting shots. * **E. Workflow and Performance** * **GPU Acceleration:** Ensure that all computationally intensive tasks (especially those involving OpenCV, Mediapipe, and PyTorch) are running on the GPU. * **Caching:** Implement more aggressive caching of intermediate results to avoid recomputation. * **Optimized Data Structures:** Profile the script to identify performance bottlenecks and optimize data structures accordingly. * **Asynchronous Operations:** Use asynchronous operations to improve UI responsiveness. * **F. User Interface and Control** * **Timeline View:** *Industry Standard:* Implement a timeline view similar to professional video editing software (Premiere Pro, Final Cut). * **Fine-Grained Control:** *Industry Standard:* Give the user fine-grained control over the editing process. Allow them to manually adjust cuts, transitions, and effects. * **Presets and Templates:** *Taylor Level:* Offer presets or templates that mimic the editing styles of famous music videos. * **G. Code Quality and Robustness** * **Comprehensive Error Handling:** Add more comprehensive error handling to gracefully handle unexpected situations. Log errors to a file for debugging. * **Unit Testing:** Implement unit tests to ensure the script is working correctly. * **Code Documentation:** Add detailed code documentation to make the script easier to understand and maintain. **3. Specific Code Improvements** I'll highlight some sections of the code and suggest concrete changes: * **a. Logging:** The logging is good, but could be expanded. * *Suggestion:* Add more debug-level logging to track the values of key variables during the analysis and sequencing stages. This helps with debugging. * **b. Data Classes:** The data classes (`AnalysisConfig`, `EffectParams`, `RenderConfig`) are excellent for organization. * *Suggestion:* Consider using `attrs` instead of `dataclasses`. `attrs` is another library for defining data classes that can sometimes offer more flexibility and performance. * **c. `get_midas_model()`:** This function is good for caching the MiDaS model. * *Suggestion:* Add error handling for cases where the model cannot be loaded (e.g., due to network issues). Provide a fallback mechanism (e.g., disable depth analysis). * **d. `BBZAudioUtils`:** * *Suggestion:* Add more sophisticated audio feature extraction (e.g., MFCCs, spectral centroid) for better mood and genre detection. * *Suggestion:* Use a more robust beat tracking algorithm. Libraries like `madmom` can provide better results than `librosa` in some cases. * **e. `BBZFaceUtils` and `calculate_flow_velocity`:** These functions are good starting points. * *Suggestion:* Consider using more advanced face recognition techniques or more robust optical flow algorithms (e.g., DIS optical flow in OpenCV). * **f. `VideousMain.analyzeVideo()`:** This is the core analysis function. * *Suggestion:* This function is very long. Break it down into smaller, more manageable functions. * *Suggestion:* Implement GPU acceleration for the OpenCV and Mediapipe operations. * *Suggestion:* Add caching to avoid recomputing features for the same frames. * **g. Sequence Builders:** The `SequenceBuilderGreedy` and `SequenceBuilderPhysicsMC` classes need more sophisticated logic. * *Suggestion:* Implement rule-based editing. * *Suggestion:* Add motion matching. * *Suggestion:* Implement more intelligent transitions. * **h. `buildSequenceVideo()`:** This function handles the final rendering. * *Suggestion:* Add more control over the encoding settings (e.g., bitrate, codec options). * *Suggestion:* Implement error handling for cases where the rendering fails. **4. Example: Implementing Rule-Based Editing** Here's a basic example of how you could start to implement rule-based editing. This is a high-level concept; actual implementation would be more complex. ```python class EditingRule: def __init__(self, shot_type=None, audio_segment=None, intensity=None): self.shot_type = shot_type # e.g., "close_up", "wide" self.audio_segment = audio_segment # e.g., "verse", "chorus" self.intensity = intensity # e.g. "High", "Low" def matches(self, clip, audio_segment_data): if self.shot_type and clip.get_shot_type() != self.shot_type: return False if self.audio_segment and audio_segment_data['segment_name'] != self.audio_segment: return False if self.intensity and clip.intensity_category != self.intensity: return False return True # Example rules rules = [ EditingRule(shot_type="close_up", audio_segment="verse"), EditingRule(shot_type="wide", audio_segment="chorus", intensity="High") ] def select_clip(clip_options, audio_segment_data, rules): for clip in clip_options: for rule in rules: if rule.matches(clip, audio_segment_data): return clip # Select the first matching clip return random.choice(clip_options) # If no rule matches, select a random clip ``` You would then integrate this `select_clip` function into your sequence building logic. **In Summary** To reach a Taylor Swift-level of music video editing, this script needs significant enhancements in visual and audio analysis, editing logic, effects, performance, UI, and code quality. Focus on implementing more sophisticated algorithms, leveraging AI-powered tools, and providing the user with fine-grained control over the editing process.'''