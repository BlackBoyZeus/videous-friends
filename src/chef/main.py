from __future__ import annotations
# -*- coding: utf-8 -*-
# ========================================================================
#            Videous Chef - Ensemble Edition v5.3 (Optimized & Fixed)
# ========================================================================
#                       IMPORTS (Essential Core Only)
# ========================================================================

#source /Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous\ CHEF/venv_videous_chef/bin/activate
#conda activate videous 
from tkinter import filedialog, Listbox, Scrollbar, END, MULTIPLE, Frame, messagebox, BooleanVar, IntVar
import customtkinter
import cv2
from math import exp, log
import numpy as np
import time
import os
import json
import threading
import concurrent.futures
import traceback
import sys
import shlex
import random
from collections import defaultdict, namedtuple, deque
from math import exp, log
import subprocess
import logging
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import List, Tuple, Dict, Any, Optional, Set, Union, Callable
import tracemalloc
import inspect
import hashlib # For audio caching hash
import pathlib 



# --- Core AI/ML/Audio Libs (Imported Early) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import mediapipe as mp
try:
    import soundfile # For robust audio loading
except ImportError:
    print("ERROR: soundfile library not found. Install with 'pip install soundfile'")
    sys.exit(1)
try:
    import torchaudio # Primary audio loading backend
    import torchaudio.transforms as T # For resampling
except ImportError:
    print("ERROR: torchaudio library not found. Install with 'pip install torchaudio'")
    sys.exit(1)

try:
    import customtkinter
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    
try:
    import moviepy
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# --- UI & Utilities ---
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    print("ERROR: tkinterdnd2-universal not found. Drag & Drop disabled. Install with 'pip install tkinterdnd2-universal'")
    # Fallback for basic operation without DnD
    DND_FILES = TkinterDnD = None
    
    class TkinterDnD: # Dummy class
        class DnDWrapper: pass
        _require = lambda x: None # Dummy method
try:
    import multiprocessing
except ImportError:
    print("WARNING: multiprocessing module not found, parallel processing disabled.")
    multiprocessing = None # Assign None to check later
    
try:
    import multiprocessing
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    
try:
    import tkinter
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try: from tqdm import tqdm
except ImportError: tqdm = None; print("INFO: tqdm not found, progress bars disabled.")
import matplotlib # For backend setting only
matplotlib.use('Agg') # Set backend before importing pyplot

# --- Global Flags for Optional Libs ---
TRANSFORMERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
AUDIOFLUX_AVAILABLE = False
DEMUCS_AVAILABLE = False
HUGGINGFACE_HUB_AVAILABLE = False
SKIMAGE_AVAILABLE = False
LIBROSA_AVAILABLE = False
TIMM_AVAILABLE = False # For MiDaS

# --- Attempt to set optional flags (Deferred Import Pattern) ---
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
    from transformers import pipeline, CLIPProcessor, CLIPModel, WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, AutoModelForAudioClassification, AutoProcessor
    print("INFO: `transformers` loaded successfully.")
except ImportError: print("INFO: `transformers` not found. Related features (LVRE, etc.) disabled.")
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    from sentence_transformers import SentenceTransformer
    print("INFO: `sentence-transformers` loaded successfully.")
except ImportError: print("INFO: `sentence-transformers` not found. LVRE semantic features disabled.")
try:
    import audioflux as af
    AUDIOFLUX_AVAILABLE = True
    print("INFO: `audioflux` loaded successfully. Using for advanced audio analysis.")
except ImportError: print("INFO: `audioflux` not found. Using Librosa/SciPy fallbacks for some audio analysis.")
try:
    # Check if demucs is importable - actual separation needs more setup
    # This basic check allows enabling the config flag
    import demucs.separate
    DEMUCS_AVAILABLE = True
    print("INFO: `demucs` found. Source separation (MRISE) enabled (if configured).")
except ImportError: print("INFO: `demucs` not found. Source separation for MRISE disabled.")
try:
    import huggingface_hub
    HUGGINGFACE_HUB_AVAILABLE = True
    from huggingface_hub import hf_hub_download
    print("INFO: `huggingface_hub` loaded successfully. Can auto-download models.")
except ImportError: print("INFO: `huggingface_hub` not found. Cannot auto-download models (SyncNet, etc.).")
try:
    import skimage.transform
    import skimage.metrics
    SKIMAGE_AVAILABLE = True
    from skimage.metrics import structural_similarity as ssim # Example import
    print("INFO: `scikit-image` loaded successfully. Required for SyncNet helpers.")
except ImportError: print("INFO: `scikit-image` not found. SyncNet PWRC scoring disabled.")
try:
    os.environ['LIBROSA_CORE'] = 'scipy' # Try to force SciPy backend for Librosa if used
    import librosa
    LIBROSA_AVAILABLE = True
    print("INFO: `librosa` loaded successfully. Used for fallbacks and Mel spectrogram.")
except ImportError: print("INFO: `librosa` not found. Fallbacks using it (beat, onset, Mel) are disabled.")
try:
    import timm
    TIMM_AVAILABLE = True
    print("INFO: `timm` library loaded successfully (required for MiDaS).")
except ImportError:
    print("INFO: `timm` library not found. MiDaS depth estimation disabled.")

# --- Other necessary core imports ---
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import VideoClip # Needed for make_frame render
except ImportError:
    print("ERROR: moviepy library not found. Install with 'pip install moviepy'")
    sys.exit(1)


import scipy.signal
import scipy.stats

# --- MediaPipe Solutions ---
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except AttributeError as mp_err:
    print(f"ERROR: Failed to load MediaPipe solutions: {mp_err}. Ensure mediapipe is installed correctly ('pip install mediapipe').")
    sys.exit(1)


# --- Constants ---
MIN_POTENTIAL_CLIP_DURATION_FRAMES = 5
TARGET_SR_TORCHAUDIO = 16000 # Preferred Sample Rate for analysis
LATENTSYNC_MAX_FRAMES = 60 # Limit frames processed by SyncNet per segment for performance
DEFAULT_SEQUENCE_HISTORY_LENGTH = 5 # For SAAPV predictability penalty
ENABLE_PROFILING = True # <<< Set to True to enable simple timing logs >>>

# Default normalization constants (can be overridden by config)
# Note: V_MAX/A_MAX/E_MAX removed as they are not directly used in v5.3 analysis config
# D_MAX_EXPECTED = 0.15 # Replaced by norm_max_depth_variance
DEFAULT_NORM_MAX_RMS = 0.5; DEFAULT_NORM_MAX_ONSET = 5.0; DEFAULT_NORM_MAX_CHROMA_VAR = 0.1
# These are Physics MC specific, moved to defaults in its section if needed.
# SIGMA_M_DEFAULT = 0.3; DEFAULT_REPETITION_PENALTY = 0.15
# DEFAULT_GREEDY_SOURCE_PENALTY = 0.1; DEFAULT_GREEDY_SHOT_PENALTY = 0.1

# --- Caches & Logging ---
_pytorch_model_cache = {}; _pytorch_processor_cache = {} # General cache for PyTorch models/processors
logger = logging.getLogger(__name__) # Get logger for the current module

# ========================================================================
#              <<< SYNC NET MODEL DEFINITION (EMBEDDED) >>>
# ========================================================================
# (VERIFY this matches the ByteDance/LatentSync-1.5/syncnet.pth architecture)
# (If ByteDance provides syncnet.py, replace this class with theirs)
class Conv3dRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(Conv3dRelu, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Conv2dRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class SyncNet(nn.Module):
    # Placeholder - Assuming the previous definition is correct.
    # Replace with the actual SyncNet definition from ByteDance/LatentSync if provided.
    def __init__(self):
        super(SyncNet, self).__init__()
        # --- Audio Stream ---
        self.audio_stream = nn.Sequential(
            Conv2dRelu(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dRelu(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            Conv2dRelu(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dRelu(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            Conv2dRelu(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dRelu(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dRelu(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            Conv2dRelu(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dRelu(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dRelu(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            # Final Conv 2D with smaller kernel, no padding
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)), # Adjust based on input size after maxpool
            nn.ReLU(inplace=True)
        )

        # --- Video Stream (Face Crops) ---
        self.video_stream = nn.Sequential(
            Conv3dRelu(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 0, 0)), # Adjusted padding
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            Conv3dRelu(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)), # Adjusted padding
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # Adjusted padding

            Conv3dRelu(256, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)), # Keep padding for 3x3
            Conv3dRelu(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            Conv3dRelu(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            # Final Conv 3D to reduce to 512 channels
            # Kernel size needs verification based on output shape of MaxPool3d
            # Example: Assuming (N, 512, 1, 6, 6) output -> kernel (1, 6, 6)
            nn.Conv3d(512, 512, kernel_size=(1, 6, 6), stride=(1, 1, 1), padding=(0, 0, 0)), # <<< VERIFY KERNEL SIZE >>>
            nn.ReLU(inplace=True)
        )

    def forward(self, audio_sequences, video_sequences):
        # Args documentation remains the same
        audio_embedding = self.audio_stream(audio_sequences) # Shape: (N, 512, 1, 1)
        video_embedding = self.video_stream(video_sequences) # Shape: (N, 512, 1, 1, 1)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1) # Flatten to (N, 512)
        video_embedding = video_embedding.view(video_embedding.size(0), -1) # Flatten to (N, 512)

        # L2 Normalize the embeddings
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        video_embedding = F.normalize(video_embedding, p=2, dim=1)

        return audio_embedding, video_embedding

# ========================================================================
#                       DATA CLASSES (Ensemble Ready)
# ========================================================================
from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    """Configuration for the analysis phase (Ensemble v5.4/5.5)."""
    # --- Shared Params ---
    min_sequence_clip_duration: float = 0.75
    max_sequence_clip_duration: float = 5.0
    min_potential_clip_duration_sec: float = 0.4  # Min duration for a potential clip during analysis
    resolution_height: int = 256  # Target analysis resolution
    resolution_width: int = 256  # Target analysis resolution
    save_analysis_data: bool = True  # Saves intermediate JSONs (incl. audio cache)
    cache_visual_features: bool = True  # Enable visual frame caching
    use_scene_detection: bool = True  # Enable PySceneDetect for scene change detection

    # --- Normalization Defaults ---
    norm_max_rms: float = 0.1  # Maximum RMS for normalization
    norm_max_onset: float = 0.1  # Maximum onset strength for normalization
    norm_max_pose_kinetic: float = 50.0  # Maximum pose kinetic energy for normalization
    norm_max_visual_flow: float = 50.0  # Maximum visual flow for normalization
    norm_max_depth_variance: float = 0.15  # Used by Physics MC only (V4 logic)
    norm_max_face_size: float = 1.0  # Used by Base Heuristic & Physics MC
    norm_max_jerk: float = 100000.0  # Used by Base Heuristic & Physics MC

    # --- Detection Thresholds ---
    min_face_confidence: float = 0.5
    min_pose_confidence: float = 0.5
    model_complexity: int = 1  # Pose model complexity (0, 1, 2)
    mouth_open_threshold: float = 0.05  # Base Heuristic face feature

    # --- Audio Analysis ---
    target_sr_audio: int = 44100  # Target sample rate for audio analysis
    use_dl_beat_tracker: bool = True  # Prioritize AudioFlux/DL model for beats if available
    hop_length_energy: int = 512  # Hop length for RMS energy calculation
    frame_length_energy: int = 1024  # Frame length for RMS energy calculation
    trend_window_sec: float = 3.0  # Window for long-term energy trend smoothing
    hop_length_mel: int = 160  # Hop length for Mel spectrogram

    # --- Sequencing Mode ---
    sequencing_mode: str = "Greedy Heuristic"  # or "Physics Pareto MC"

    # --- Feature Flags ---
    use_latent_sync: bool = True  # PWRC: Enable SyncNet Lip Sync Scoring
    use_lvre_features: bool = False  # LVRE: Enable Lyrical/Visual/Emotion Features
    use_saapv_adaptation: bool = True  # SAAPV: Enable Style Analysis & Pacing Adaptation
    use_mrise_sync: bool = True  # MRISE: Enable Micro-Rhythm Synchronization
    use_demucs_for_mrise: bool = False  # MRISE: Use Demucs for stems

    # --- SyncNet (PWRC) ---
    syncnet_repo_id: str = "ByteDance/LatentSync-1.5"  # SyncNet model repository
    syncnet_filename: str = "syncnet.pth"  # SyncNet model file
    syncnet_batch_size: int = 16  # Batch size for SyncNet scoring

    # --- LVRE Model Config ---
    whisper_model_name: str = "openai/whisper-tiny"  # Faster ASR model
    ser_model_name: str = "facebook/wav2vec2-large-robust-ft-emotion-msp-podcast"  # Speech Emotion Recognition
    text_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Sentence embedder
    vision_embed_model_name: str = "openai/clip-vit-base-patch32"  # CLIP model
    lvre_batch_size_text: int = 128
    lvre_batch_size_vision: int = 64
    lvre_batch_size_ser: int = 16

    # --- Scene Detection (SAAPV) ---
    scene_change_penalty: float = 0.1  # Penalty for cuts across scene boundaries

    # --- Render FPS ---
    render_fps: int = 30

    # === GREEDY HEURISTIC ENSEMBLE WEIGHTS ===
    # Base Heuristic (Original Concept)
    base_heuristic_weight: float = 0.1
    bh_audio_weight: float = 0.3
    bh_kinetic_weight: float = 0.25
    bh_sharpness_weight: float = 0.1  # Uses Jerk Proxy
    bh_camera_motion_weight: float = 0.05  # Uses Flow Velocity
    bh_face_size_weight: float = 0.1
    bh_percussive_weight: float = 0.05
    bh_depth_weight: float = 0.0  # Defaulting Depth to 0 for Base Heuristic

    # Performance Weights (PWRC)
    pwrc_weight: float = 0.3
    pwrc_lipsync_weight: float = 0.6  # Weight for SyncNet score within PWRC
    pwrc_pose_energy_weight: float = 0.4  # Weight for Pose energy vs Audio energy match

    # Energy Flow Weights (HEFM)
    hefm_weight: float = 0.2
    hefm_trend_match_weight: float = 1.0  # Weight for visual trend vs audio trend match

    # Lyrical/Visual/Emotion Weights (LVRE)
    lvre_weight: float = 0.15
    lvre_semantic_weight: float = 0.7  # Weight for lyric-visual semantic similarity
    lvre_emphasis_weight: float = 0.3  # Weight for vocal emphasis boost

    # Style/Pacing/Variety Weights (SAAPV)
    saapv_weight: float = 0.1  # Overall weight for SAAPV contributions
    saapv_predictability_weight: float = 0.5  # Weight of predictability penalty
    saapv_history_length: int = 10  # Number of past edits to consider for predictability
    saapv_variety_penalty_source: float = 0.1  # Penalty for same source
    saapv_variety_penalty_shot: float = 0.1  # Penalty for same shot
    saapv_variety_penalty_intensity: float = 0.10  # Penalty for similar intensity

    # Micro-Rhythm Weights (MRISE)
    mrise_weight: float = 0.15
    mrise_sync_weight: float = 1.0  # Weight of micro-beat sync bonus
    mrise_sync_tolerance_factor: float = 1.5  # Multiplier for base tolerance (e.g., 1 / fps)

    # Rhythm/Timing Weights (General)
    rhythm_beat_sync_weight: float = 0.1  # Bonus for cutting near a main beat
    rhythm_beat_boost_radius_sec: float = 0.1  # Window around beat for bonus

    # --- Greedy Candidate Selection ---
    candidate_pool_size: int = 15

    # === PHYSICS PARETO MC SPECIFIC ===
    score_threshold: float = 0.3  # Base heuristic threshold for V4 logic
    fit_weight_velocity: float = 0.3
    fit_weight_acceleration: float = 0.3
    fit_weight_mood: float = 0.4
    fit_sigmoid_steepness: float = 1.0
    objective_weight_rhythm: float = 1.0
    objective_weight_mood: float = 1.0
    objective_weight_continuity: float = 0.8
    objective_weight_variety: float = 0.7
    objective_weight_efficiency: float = 0.5
    mc_iterations: int = 500
    mood_similarity_variance: float = 0.3
    continuity_depth_weight: float = 0.5
    variety_repetition_penalty: float = 0.15

@dataclass
class EffectParams: # (Unchanged from v4.7.3)
    """Parameters for a specific effect type."""
    type: str = "cut"
    tau: float = 0.0 # Duration
    psi: float = 0.0 # Physical impact proxy
    epsilon: float = 0.0 # Perceptual gain

@dataclass
class RenderConfig:
    """Configuration for the rendering phase."""
    effect_settings: Dict[str, EffectParams] = field(default_factory=lambda: {
        "cut": EffectParams(type="cut"),
        "fade": EffectParams(type="fade", tau=0.2, psi=0.1, epsilon=0.2),
        "zoom": EffectParams(type="zoom", tau=0.5, psi=0.3, epsilon=0.4),
        "pan": EffectParams(type="pan", tau=0.5, psi=0.1, epsilon=0.3),
    })
    video_codec: str = 'libx264'
    preset: Optional[str] = 'medium'
    crf: Optional[int] = 23
    audio_codec: str = 'aac'
    audio_bitrate: str = '192k'
    threads: int = max(1, (os.cpu_count() or 2) // 2)
    resolution_width: int = 1920
    resolution_height: int = 1080
    fps: int = 30
    use_gfpgan_enhance: bool = True
    gfpgan_fidelity_weight: float = 0.5
    #gfpgan_model_path: Optional[str] = None  # Add this line
    # Or use your path:
    gfpgan_model_path: Optional[str] = "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/experiments/pretrained_models/GFPGANv1.3.pth"
# ========================================================================
#                       HELPER FUNCTIONS (Loaders, SyncNet Helpers, etc.)
# ========================================================================
def tk_write(tk_string1, parent=None, level="info"):
    """Shows message boxes and logs messages."""
    log_level_map = { "error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO, "debug": logging.DEBUG }
    log_level = log_level_map.get(level.lower(), logging.INFO)
    logger.log(log_level, f"Popup ({level}): {tk_string1}")
    try:
        use_messagebox = False
        # Check if parent is a valid Tk/Toplevel widget before showing messagebox
        if parent and isinstance(parent, (tkinter.Tk, tkinter.Toplevel)) and parent.winfo_exists():
            use_messagebox = True
        elif isinstance(parent, customtkinter.CTk) and parent.winfo_exists(): # Handle CTk main window
            use_messagebox = True
        else:
            logger.debug("tk_write called without valid parent or parent destroyed.")

        if use_messagebox:
            title = f"Videous Chef - {level.capitalize()}"
            if level == "error": messagebox.showerror(title, tk_string1, parent=parent)
            elif level == "warning": messagebox.showwarning(title, tk_string1, parent=parent)
            else: messagebox.showinfo(title, tk_string1, parent=parent)
    except Exception as e:
        logger.error(f"tk_write internal error: {e}", exc_info=True)
        # Fallback to console print if messagebox fails
        print(f"!! tk_write Error: {e}\n!! Level: {level}\n!! Message: {tk_string1}")

def sigmoid(x, k=1): # (Unchanged from v4.7.3)
    try:
        # Clip input to avoid large exponents leading to overflow
        x_clamped = np.clip(x * k, -700, 700) # Multiply by k before clipping
        return 1 / (1 + np.exp(-x_clamped))
    except OverflowError:
        logger.warning(f"Sigmoid overflow detected for input x={x}, k={k}.")
        # Return 0 for large negative inputs, 1 for large positive inputs
        return 0.0 if x < 0 else 1.0

def cosine_similarity(vec1, vec2): # (Unchanged from v4.7.3)
    if vec1 is None or vec2 is None: return 0.0
    vec1 = np.asarray(vec1, dtype=float); vec2 = np.asarray(vec2, dtype=float)
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0 # Avoid division by zero
    dot_product = np.dot(vec1, vec2); similarity = dot_product / (norm1 * norm2)
    # Clip similarity to [-1, 1] to handle potential floating point inaccuracies
    return np.clip(similarity, -1.0, 1.0)

def calculate_histogram_entropy(frame): # (Unchanged from v4.7.3)
    if frame is None or frame.size == 0: return 0.0
    try:
        # Ensure frame is grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2: gray = frame
        else: logger.warning(f"Invalid frame shape for entropy: {frame.shape}"); return 0.0

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_sum = hist.sum()
        if hist_sum <= 0: return 0.0 # Avoid division by zero if sum is zero

        # Normalize histogram to get probability distribution
        hist_norm = hist.ravel() / hist_sum
        # Filter out zero probabilities for entropy calculation
        hist_norm_nonzero = hist_norm[hist_norm > 0]
        if hist_norm_nonzero.size == 0: return 0.0 # Handle case where all probabilities are zero (e.g., blank image)

        # Calculate entropy using scipy.stats.entropy (base e)
        entropy = scipy.stats.entropy(hist_norm_nonzero)

        # Ensure finite result (e.g., avoid NaN if input was strange)
        return entropy if np.isfinite(entropy) else 0.0
    except cv2.error as cv_err: logger.warning(f"OpenCV error in histogram calc: {cv_err}"); return 0.0
    except Exception as e: logger.warning(f"Hist entropy calculation failed: {e}"); return 0.0

# UPDATED get_device (Adds MPS support)
def get_device() -> torch.device:
    """Gets the recommended device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): # <<< CHANGED >>>
        device = torch.device("mps")
        logger.debug("Using MPS (Apple Silicon GPU) device.")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU device.")
    return device

# --- Cached Model/Processor Loaders ---
# --- Cached Model/Processor Loaders (NEW for v5.4/5.5) ---
# --- Cached Model/Processor Loaders (NEW for v5.4/5.5) ---
def get_pytorch_model(cache_key: str, load_func: callable, *args, **kwargs) -> Optional[Tuple[torch.nn.Module, torch.device]]:
    """Loads a PyTorch model using cache, only loading if not present. Returns model and device."""
    global _pytorch_model_cache
    if cache_key in _pytorch_model_cache:
        model, device = _pytorch_model_cache[cache_key]
        logger.debug(f"Reusing cached PyTorch Model: {cache_key} on {device}")
        return model, device

    logger.info(f"Loading PyTorch Model: {cache_key}...")
    try:
        device = get_device()
        model = load_func(*args, **kwargs)
        if model is None: raise ValueError("Model loading function returned None")
        if hasattr(model, 'to') and callable(model.to): model.to(device)
        else: logger.warning(f"Model {cache_key} does not have .to() method.")
        if hasattr(model, 'eval') and callable(model.eval): model.eval()
        else: logger.warning(f"Model {cache_key} does not have .eval() method.")
        _pytorch_model_cache[cache_key] = (model, device)
        logger.info(f"PyTorch Model loaded successfully: {cache_key} to {device}")
        return model, device
    except ImportError as imp_err:
        logger.error(f"ImportError loading PyTorch Model {cache_key}: {imp_err}.")
        return None, None
    except Exception as e:
        logger.error(f"Failed load PyTorch Model {cache_key}: {e}", exc_info=True)
        # <<< CORRECTED BLOCK START >>>
        if cache_key in _pytorch_model_cache:
            del _pytorch_model_cache[cache_key] # Put deletion on its own line
        return None, None # Put return on its own line
        # <<< CORRECTED BLOCK END >>>

def get_pytorch_processor(cache_key: str, load_func: callable, *args, **kwargs) -> Optional[Any]:
    """Loads a PyTorch processor using cache, loading only if not present."""
    global _pytorch_processor_cache
    if cache_key in _pytorch_processor_cache:
        logger.debug(f"Reusing cached PyTorch Processor: {cache_key}")
        return _pytorch_processor_cache[cache_key]

    logger.info(f"Loading PyTorch Processor: {cache_key}...")
    try:
        processor = load_func(*args, **kwargs)
        if processor is None:
            raise ValueError("Processor loading function returned None")
        _pytorch_processor_cache[cache_key] = processor
        logger.info(f"PyTorch Processor loaded successfully: {cache_key}")
        return processor
    except ImportError as imp_err:
        logger.error(f"ImportError loading PyTorch Processor {cache_key}: {imp_err}.")
        return None
    except Exception as e:
        logger.error(f"Failed load PyTorch Processor {cache_key}: {e}", exc_info=True)
        if cache_key in _pytorch_processor_cache:
            del _pytorch_processor_cache[cache_key]
        return None 

def load_huggingface_pipeline_func(task: str, model_name: str, device: torch.device, **kwargs):
    """Actual loading function for HF pipeline."""
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found.")
    device_id = device.index if device.type == 'cuda' else (-1 if device.type == 'cpu' else 0)
    trust_code = kwargs.pop('trust_remote_code', False)
    return pipeline(task, model=model_name, device=device_id, trust_remote_code=trust_code, **kwargs)

def load_huggingface_model_func(model_name: str, model_class: type, **kwargs):
    """Actual loading function for HF model."""
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found.")
    trust_code = kwargs.pop('trust_remote_code', False)
    return model_class.from_pretrained(model_name, trust_remote_code=trust_code, **kwargs)

def load_huggingface_processor_func(model_name: str, processor_class: type, **kwargs):
    """Actual loading function for HF processor."""
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found.")
    trust_code = kwargs.pop('trust_remote_code', False)
    return processor_class.from_pretrained(model_name, trust_remote_code=trust_code, **kwargs)

def load_sentence_transformer_func(model_name: str):
    """Actual loading function for Sentence Transformer model."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE: raise ImportError("Sentence-Transformers library not found.")
    return SentenceTransformer(model_name)

def load_syncnet_model_from_hf_func(config: AnalysisConfig) -> Optional[SyncNet]:
    """Actual loading function for SyncNet model from Hugging Face Hub."""
    if not HUGGINGFACE_HUB_AVAILABLE: raise ImportError("huggingface_hub required.")
    if 'SyncNet' not in globals() or not inspect.isclass(globals()['SyncNet']): raise RuntimeError("SyncNet class missing.")
    repo_id = config.syncnet_repo_id; filename = config.syncnet_filename
    logger.info(f"Downloading/Loading SyncNet weights ({repo_id}/{filename})...")
    try:
        user_agent = {"library_name": "videous-chef", "library_version": "5.4"}; checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename, user_agent=user_agent, resume_download=True)
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        syncnet_model = SyncNet(); loaded_data = torch.load(checkpoint_path, map_location='cpu')
        state_dict = loaded_data.get('state_dict', loaded_data)
        if isinstance(state_dict, dict) and 'net' in state_dict and isinstance(state_dict['net'], dict): state_dict = state_dict['net']
        if not isinstance(state_dict, dict): raise ValueError(f"Unrecognized checkpoint format: {type(state_dict)}")
        adapted_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = syncnet_model.load_state_dict(adapted_state_dict, strict=False)
        if missing_keys: logger.warning(f"SyncNet MISSING keys: {missing_keys}")
        if unexpected_keys: logger.warning(f"SyncNet UNEXPECTED keys: {unexpected_keys}")
        logger.info(f"SyncNet weights loaded ({'strict' if not missing_keys and not unexpected_keys else 'non-strict'} match).")
        return syncnet_model
    except FileNotFoundError as fnf_err: logger.error(f"SyncNet FNF Error: {fnf_err}"); return None
    except Exception as e: logger.error(f"Failed load SyncNet weights: {e}", exc_info=True); return None


# --- MiDaS Loading Function (Defined for macOS fix) ---
from functools import lru_cache # Import if not already present
@lru_cache(maxsize=1) # Cache model after first load
def get_midas_model() -> Optional[Tuple[torch.nn.Module, Any, torch.device]]:
    """Loads the MiDaS model and transform using cached loader."""
    global TIMM_AVAILABLE
    if not TIMM_AVAILABLE:
        logger.error("MiDaS requires 'timm'. Install with: pip install timm")
        return None, None, None
    logger.info("Loading MiDaS model (intel-isl/MiDaS MiDaS_small)...")
    try:
        # <<< CHANGED: Use the generic cached loader >>>
        model, device = get_pytorch_model(
            "midas_small", # Cache key
            lambda: torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True) # Lambda for loading func
        )
        if model is None or device is None:
            raise RuntimeError("MiDaS model loading failed via get_pytorch_model.")

        # Load transform separately (doesn't need device/cache usually)
        transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms_hub.small_transform
        logger.info("MiDaS model and transform loaded.")
        return model, transform, device
    except Exception as e:
        logger.error(f"Failed to load MiDaS: {e}", exc_info=True)
        # Clear from cache if load failed
        if "midas_small" in _pytorch_model_cache:
            del _pytorch_model_cache["midas_small"]
        return None, None, None

# --- Mouth Crop Helper ---
# UPDATED extract_mouth_crop (uses skimage)
MOUTH_LM_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 61, 76, 77, 78, 80, 81, 82, 84, 85, 87, 88, 90, 91, 95, 146, 178, 180, 181, 267, 269, 270, 291, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 320, 321, 324, 375, 402, 404, 405, 409, 415]
def extract_mouth_crop(frame_bgr: np.ndarray, face_landmarks: Any, target_size=(112, 112)) -> Optional[np.ndarray]:
    """Extracts and resizes the mouth region from a frame given FaceMesh landmarks."""
    # <<< CHANGED: Requires scikit-image (SKIMAGE_AVAILABLE flag) >>>
    if face_landmarks is None or not SKIMAGE_AVAILABLE:
        if not SKIMAGE_AVAILABLE: logger.debug("Skipping mouth crop: scikit-image not available.")
        return None
    h, w = frame_bgr.shape[:2];
    if h == 0 or w == 0: return None
    try:
        mouth_points = []; lm = face_landmarks.landmark
        for idx in MOUTH_LM_INDICES:
            if 0 <= idx < len(lm): x = lm[idx].x * w; y = lm[idx].y * h;
            if np.isfinite(x) and np.isfinite(y): mouth_points.append([int(round(x)), int(round(y))])
        if len(mouth_points) < 4: logger.debug("Insufficient valid mouth landmarks."); return None
        mouth_points = np.array(mouth_points, dtype=np.int32); min_x, min_y = np.min(mouth_points, axis=0); max_x, max_y = np.max(mouth_points, axis=0); center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2; width_box = max(1, max_x - min_x); height_box = max(1, max_y - min_y); crop_size = int(max(width_box, height_box) * 1.6); crop_size = max(crop_size, 10); x1 = max(0, center_x - crop_size // 2); y1 = max(0, center_y - crop_size // 2); x2 = min(w, x1 + crop_size); y2 = min(h, y1 + crop_size); final_width = x2 - x1; final_height = y2 - y1;
        if final_width <= 0 or final_height <= 0: logger.debug(f"Initial mouth crop calc zero/neg size."); return None
        if final_width != crop_size and final_height == crop_size: x1 = max(0, x2 - crop_size)
        if final_height != crop_size and final_width == crop_size: y1 = max(0, y2 - crop_size)
        final_width = x2 - x1; final_height = y2 - y1;
        if final_width <=0 or final_height <= 0: logger.debug(f"Adjusted mouth crop zero/neg size."); return None
        mouth_crop = frame_bgr[y1:y2, x1:x2]
        # <<< CHANGED: Resize using scikit-image >>>
        resized_mouth = skimage.transform.resize(mouth_crop, target_size, anti_aliasing=True, mode='edge', preserve_range=True)
        resized_mouth_uint8 = np.clip(resized_mouth, 0, 255).astype(np.uint8)
        return resized_mouth_uint8
    except ImportError: logger.error("extract_mouth_crop called but scikit-image unavailable."); return None
    except Exception as e: logger.warning(f"Error extracting mouth crop: {e}", exc_info=False); return None

# --- Mel Spectrogram Helper ---
@lru_cache(maxsize=4)
def compute_mel_spectrogram(waveform_tensor: torch.Tensor, sr: int, hop_length: int = 160, n_mels: int = 80) -> Optional[np.ndarray]:
    """Compute Mel spectrogram using torchaudio."""
    # <<< CHANGED: Uses torchaudio.transforms.MelSpectrogram >>>
    if not TORCHAUDIO_AVAILABLE: logger.error("Torchaudio unavailable for Mel."); return None
    try:
        if waveform_tensor.dtype != torch.float32: waveform_tensor = waveform_tensor.float()
        if waveform_tensor.ndim == 1: waveform_tensor = waveform_tensor.unsqueeze(0)
        n_fft = 2048;
        if hop_length <= 0: hop_length = 160; logger.warning("Invalid hop_length, using 160.")
        if n_mels <= 0: n_mels = 80; logger.warning("Invalid n_mels, using 80.")
        mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=0.0, f_max=sr / 2.0).to(waveform_tensor.device)
        mel_spec_torch = mel_transform(waveform_tensor); mel_spec_log = torch.log(torch.clamp(mel_spec_torch, min=1e-9))
        mel_spec_np = mel_spec_log.squeeze(0).cpu().numpy()
        return mel_spec_np
    except Exception as e: logger.error(f"Failed compute Mel spectrogram: {e}", exc_info=True); return None
    
# --- SAAPV History Helper ---
def get_recent_history(history: deque, count: int) -> List:
    """Safely gets the last 'count' items from a deque."""
    return list(history) # Deque slicing is not standard, convert to list

# --- Pose/Visual Util Helpers ---
def calculate_flow_velocity(prev_gray, current_gray): # (Unchanged from v4.7.3)
    if prev_gray is None or current_gray is None or prev_gray.shape != current_gray.shape:
        return 0.0, None
    try:
        # Farneback parameters might need tuning based on resolution/content
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        if flow is None:
            return 0.0, None

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use nanmean to handle potential NaN values in flow
        avg_magnitude = np.nanmean(magnitude)
        # Ensure result is finite, default to 0 if not
        if not np.isfinite(avg_magnitude): avg_magnitude = 0.0

        # Apply empirical scaling factor (adjust if needed)
        return float(avg_magnitude * 10.0), flow
    except cv2.error as cv_err:
        logger.warning(f"OpenCV optical flow calculation error: {cv_err}")
        return 0.0, None
    except Exception as e:
        logger.warning(f"Unexpected error calculating flow velocity: {e}")
        return 0.0, None

def calculate_flow_acceleration(prev_flow, current_flow, dt): # (Unchanged from v4.7.3)
    if prev_flow is None or current_flow is None or prev_flow.shape != current_flow.shape or dt <= 1e-6:
        return 0.0
    try:
        # Calculate difference in flow vectors
        flow_diff = current_flow - prev_flow
        # Calculate magnitude of acceleration per pixel
        accel_magnitude_per_pixel, _ = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
        # Average magnitude, handling NaNs
        avg_accel_magnitude = np.nanmean(accel_magnitude_per_pixel)
        # Check for non-finite result
        if not np.isfinite(avg_accel_magnitude): avg_accel_magnitude = 0.0

        # Calculate acceleration (change in velocity over time)
        accel = avg_accel_magnitude / dt
        # Apply empirical scaling (adjust if needed)
        return float(accel * 10.0)
    except Exception as e:
        logger.warning(f"Error calculating flow acceleration: {e}")
        return 0.0

def calculate_kinetic_energy_proxy(landmarks_prev, landmarks_curr, dt): # (Unchanged from v4.7.3)
    if landmarks_prev is None or landmarks_curr is None or dt <= 1e-6: return 0.0
    if not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0

    lm_prev = landmarks_prev.landmark; lm_curr = landmarks_curr.landmark
    if len(lm_prev) != len(lm_curr): return 0.0

    total_sq_velocity = 0.0; num_valid = 0
    try:
        for i in range(len(lm_prev)):
            # Consider landmark visibility if available (MediaPipe specific)
            vis_prev = getattr(lm_prev[i], 'visibility', 1.0)
            vis_curr = getattr(lm_curr[i], 'visibility', 1.0)
            # Threshold visibility
            if vis_prev > 0.2 and vis_curr > 0.2:
                dx = lm_curr[i].x - lm_prev[i].x
                dy = lm_curr[i].y - lm_prev[i].y
                dz = lm_curr[i].z - lm_prev[i].z # Include depth change
                # Squared velocity for this landmark
                total_sq_velocity += (dx**2 + dy**2 + dz**2) / (dt**2)
                num_valid += 1
    except IndexError:
        logger.warning("Index error during kinetic energy calculation (landmark mismatch?).")
        return 0.0

    if num_valid == 0: return 0.0
    # Average squared velocity across valid landmarks
    avg_sq_velocity = total_sq_velocity / num_valid
    # Apply empirical scaling (adjust as needed)
    return float(avg_sq_velocity * 1000.0)

def calculate_movement_jerk_proxy(landmarks_prev_prev, landmarks_prev, landmarks_curr, dt): # (Unchanged from v4.7.3)
    if landmarks_prev_prev is None or landmarks_prev is None or landmarks_curr is None or dt <= 1e-6: return 0.0
    if not hasattr(landmarks_prev_prev, 'landmark') or not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0

    lm_pp = landmarks_prev_prev.landmark; lm_p = landmarks_prev.landmark; lm_c = landmarks_curr.landmark
    if len(lm_pp) != len(lm_p) or len(lm_p) != len(lm_c): return 0.0

    total_sq_accel_change = 0.0; num_valid = 0; dt_sq = dt * dt
    try:
        for i in range(len(lm_pp)):
            # Check visibility across all three frames
            vis_pp = getattr(lm_pp[i], 'visibility', 1.0)
            vis_p = getattr(lm_p[i], 'visibility', 1.0)
            vis_c = getattr(lm_c[i], 'visibility', 1.0)
            if vis_pp > 0.2 and vis_p > 0.2 and vis_c > 0.2:
                # Calculate acceleration components using finite difference
                ax = (lm_c[i].x - 2*lm_p[i].x + lm_pp[i].x) / dt_sq
                ay = (lm_c[i].y - 2*lm_p[i].y + lm_pp[i].y) / dt_sq
                az = (lm_c[i].z - 2*lm_p[i].z + lm_pp[i].z) / dt_sq
                # Use squared magnitude of acceleration as proxy for jerk intensity contribution
                # (Technically jerk is derivative of accel, but this captures accel changes)
                accel_magnitude_sq = ax**2 + ay**2 + az**2
                total_sq_accel_change += accel_magnitude_sq
                num_valid += 1
    except IndexError:
        logger.warning("Index error during jerk calculation (landmark mismatch?).")
        return 0.0

    if num_valid == 0: return 0.0
    avg_sq_accel_proxy = total_sq_accel_change / num_valid
    # Apply large scaling factor (jerk values can be large)
    return float(avg_sq_accel_proxy * 100000.0) # Adjust scaling as needed

class BBZPoseUtils: # (Unchanged from v4.7.3)
    def drawLandmarksOnImage(self, imageInput, poseProcessingInput):
        annotated_image = imageInput.copy()
        if poseProcessingInput and poseProcessingInput.pose_landmarks:
            try:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    poseProcessingInput.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            except Exception as e:
                logger.warning(f"Failed to draw pose landmarks: {e}")
        return annotated_image

# ========================================================================
#           AUDIO ANALYSIS UTILITIES (Major Refactor - Ensemble v5.3)
# ========================================================================
class BBZAudioUtils:
    def extract_audio(self, video_path: str, audio_output_path: str = "temp_audio.wav") -> Optional[str]:
        """Extracts audio using FFmpeg, ensuring output path exists."""
        logger.info(f"Extracting audio from '{os.path.basename(video_path)}' using FFmpeg...")
        start_time = time.time()
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            output_dir = os.path.dirname(audio_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            command = [
                "ffmpeg", "-i", shlex.quote(video_path), "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-y", shlex.quote(audio_output_path), "-hide_banner", "-loglevel", "error"
            ]

            logger.debug(f"Executing FFmpeg: {' '.join(command)}")
            result = subprocess.run(
                " ".join(command), shell=True, capture_output=True, text=True, check=False, encoding='utf-8'
            )

            if result.returncode != 0:
                logger.error(
                    f"FFmpeg failed (Code:{result.returncode}) extracting audio from "
                    f"'{os.path.basename(video_path)}'. Stderr: {result.stderr}"
                )
                return None
            elif not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
                logger.error(
                    f"FFmpeg ran but output audio file is missing or empty: {audio_output_path}. "
                    f"Stderr: {result.stderr}"
                )
                return None
            else:
                logger.info(
                    f"Audio extracted successfully to '{audio_output_path}' "
                    f"({time.time() - start_time:.2f}s)"
                )
                return audio_output_path
        except FileNotFoundError as fnf_err:
            logger.error(f"FFmpeg command failed: {fnf_err}. Is FFmpeg installed and in PATH?")
            return None
        except Exception as e:
            logger.error(f"FFmpeg audio extraction failed for '{video_path}': {e}", exc_info=True)
            return None

    def _detect_beats_downbeats(
        self, waveform_np: np.ndarray, sr: int, config: AnalysisConfig
    ) -> Tuple[List[float], List[float]]:
        """Detects beats/downbeats using audioFlux if available, else Librosa."""
        beats, downbeats = [], []
        method_used = "None"

        if config.use_dl_beat_tracker and AUDIOFLUX_AVAILABLE:
            logger.info("Using audioFlux for beat/downbeat detection.")
            method_used = "AudioFlux"
            try:
                hop_length_af = 256
                waveform_af = np.ascontiguousarray(waveform_np, dtype=np.float32)
                novelty_obj = af.Novelty(
                    num=waveform_af.shape[0] // hop_length_af,
                    radix2_exp=12,
                    samplate=sr,
                    novelty_type=af.NoveltyType.FLUX
                )
                novelty = novelty_obj.novelty(waveform_af)
                peak_picking_obj = af.PeakPicking(novelty, time_length=sr // hop_length_af)
                peak_picking_obj.pick(
                    thresh=1.5,
                    wait=int(0.1 * (sr / hop_length_af)),
                    pre_avg=int(0.05 * (sr / hop_length_af)),
                    post_avg=int(0.05 * (sr / hop_length_af))
                )
                peak_indexes = peak_picking_obj.get_peak_indexes()
                times = peak_indexes * hop_length_af / sr
                beats = sorted([float(t) for t in times if np.isfinite(t)])
                downbeats = beats[0::4] if len(beats) > 3 else beats[:1]
                if not beats:
                    logger.warning("AudioFlux beat detection returned no beats.")
            except Exception as af_err:
                logger.error(f"audioFlux beat detection failed: {af_err}", exc_info=False)
                beats, downbeats = [], []
                method_used = "AudioFlux_Failed"

        if not beats and LIBROSA_AVAILABLE:
            logger.warning("Using Librosa fallback for beat/downbeat detection.")
            method_used = "Librosa"
            try:
                tempo_val, beat_frames = librosa.beat.beat_track(
                    y=waveform_np, sr=sr, hop_length=512, units='frames'
                )
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512).tolist()
                downbeats = beat_times[0::4] if beat_times else []
                beats = beat_times
                if not beats:
                    logger.warning("Librosa beat detection returned no beats.")
            except Exception as librosa_beat_err:
                logger.error(f"Librosa beat fallback failed: {librosa_beat_err}")
                beats, downbeats = [], []
                method_used = "Librosa_Failed"

        if not beats:
            logger.error("No beat detection method succeeded or available.")
            method_used = "None"

        logger.debug(
            f"Beat detection method: {method_used}. Found {len(beats)} beats, "
            f"{len(downbeats)} downbeats."
        )
        return beats, downbeats

    def _detect_micro_beats(
        self, waveform_np: np.ndarray, sr: int, config: AnalysisConfig,
        stem_type: Optional[str] = None
    ) -> List[float]:
        """Detects micro-beats/onsets using audioFlux if available, else Librosa."""
        log_prefix = f"Micro-Beat ({stem_type or 'full mix'})"
        micro_beats = []
        method_used = "None"

        if AUDIOFLUX_AVAILABLE:
            logger.info(f"Using audioFlux for {log_prefix} detection.")
            method_used = "AudioFlux"
            try:
                hop_length_af_micro = 64
                waveform_af = np.ascontiguousarray(waveform_np, dtype=np.float32)
                novelty_type = af.NoveltyType.HFC
                novelty_obj = af.Novelty(
                    num=waveform_af.shape[0] // hop_length_af_micro,
                    radix2_exp=11,
                    samplate=sr,
                    novelty_type=novelty_type
                )
                novelty = novelty_obj.novelty(waveform_af)
                peak_picking_obj = af.PeakPicking(novelty, time_length=sr // hop_length_af_micro)
                peak_picking_obj.pick(
                    thresh=1.8,
                    wait=int(0.02 * (sr / hop_length_af_micro)),
                    pre_avg=int(0.01 * (sr / hop_length_af_micro)),
                    post_avg=int(0.01 * (sr / hop_length_af_micro))
                )
                peak_indexes = peak_picking_obj.get_peak_indexes()
                times = peak_indexes * hop_length_af_micro / sr
                micro_beats = sorted([float(t) for t in times if np.isfinite(t)])
                if not micro_beats:
                    logger.warning(f"AudioFlux {log_prefix} detection returned no micro-beats.")
            except Exception as af_err:
                logger.error(f"audioFlux {log_prefix} detection failed: {af_err}", exc_info=False)
                micro_beats = []
                method_used = "AudioFlux_Failed"

        if not micro_beats and LIBROSA_AVAILABLE:
            logger.warning(f"Using Librosa fallback for {log_prefix} detection.")
            method_used = "Librosa"
            try:
                onset_frames = librosa.onset.onset_detect(
                    y=waveform_np, sr=sr, hop_length=128, backtrack=False,
                    units='frames', delta=0.6, wait=2
                )
                micro_beats = librosa.frames_to_time(onset_frames, sr=sr, hop_length=128).tolist()
                if not micro_beats:
                    logger.warning(f"Librosa {log_prefix} detection returned no micro-beats.")
            except Exception as librosa_onset_err:
                logger.error(f"Librosa onset fallback failed: {librosa_onset_err}")
                micro_beats = []
                method_used = "Librosa_Failed"

        if not micro_beats:
            logger.error(f"No micro-beat detection method available/succeeded for {log_prefix}.")
            method_used = "None"

        logger.debug(
            f"Micro-beat detection method ({log_prefix}): {method_used}. "
            f"Found {len(micro_beats)} micro-beats."
        )
        return micro_beats

    def _segment_audio(
        self, waveform_tensor: torch.Tensor, sr: int, config: AnalysisConfig
    ) -> List[float]:
        """Segments audio using audioFlux novelty or fixed intervals."""
        duration = waveform_tensor.shape[-1] / sr
        waveform_np = waveform_tensor.squeeze(0).numpy().astype(np.float32)
        bound_times = []
        method_used = "None"

        if AUDIOFLUX_AVAILABLE:
            logger.info("Using audioFlux for audio segmentation.")
            method_used = "AudioFlux"
            try:
                hop_length_af_seg = 512
                waveform_af = np.ascontiguousarray(waveform_np, dtype=np.float32)
                novelty_obj = af.Novelty(
                    num=waveform_af.shape[0] // hop_length_af_seg,
                    radix2_exp=12,
                    samplate=sr,
                    novelty_type=af.NoveltyType.ENERGY
                )
                novelty = novelty_obj.novelty(waveform_af)
                win_len_smooth = max(11, int(sr * 2.5 / hop_length_af_seg) | 1)
                if len(novelty) > win_len_smooth:
                    novelty_smooth = scipy.signal.savgol_filter(
                        novelty, window_length=win_len_smooth, polyorder=3
                    )
                else:
                    logger.debug(f"Audio too short for smoothing. Using raw novelty.")
                    novelty_smooth = novelty
                peak_picking_obj = af.PeakPicking(novelty_smooth, time_length=sr // hop_length_af_seg)
                peak_picking_obj.pick(
                    thresh=1.2,
                    wait=int(1.0 * (sr / hop_length_af_seg)),
                    pre_avg=int(0.5 * (sr / hop_length_af_seg)),
                    post_avg=int(0.5 * (sr / hop_length_af_seg))
                )
                peak_indexes = peak_picking_obj.get_peak_indexes()
                bound_times_raw = peak_indexes * hop_length_af_seg / sr
                potential_bounds = sorted(
                    set([0.0] + [float(t) for t in bound_times_raw if np.isfinite(t)] + [duration])
                )
                final_bounds = [potential_bounds[0]]
                min_len_sec = max(0.5, config.min_sequence_clip_duration * 0.7)
                for t in potential_bounds[1:]:
                    if t - final_bounds[-1] >= min_len_sec:
                        final_bounds.append(t)
                if final_bounds[-1] < duration - 1e-3:
                    if duration - final_bounds[-1] >= min_len_sec / 2:
                        final_bounds.append(duration)
                    elif len(final_bounds) > 1:
                        final_bounds[-1] = duration
                bound_times = [float(b) for b in final_bounds]
                if len(bound_times) <= 2:
                    logger.warning(
                        f"AudioFlux segmentation resulted in only {len(bound_times)-1} segment(s)."
                    )
            except Exception as af_err:
                logger.error(f"audioFlux segmentation failed: {af_err}", exc_info=False)
                bound_times = []
                method_used = "AudioFlux_Failed"

        if not bound_times:
            logger.warning("Using fixed duration segmentation fallback.")
            method_used = "Fixed_Interval"
            avg_segment_dur = np.clip(
                (config.min_sequence_clip_duration + config.max_sequence_clip_duration) / 2,
                3.0, 10.0
            )
            num_segments = max(1, int(round(duration / avg_segment_dur)))
            bound_times = [float(t) for t in np.linspace(0, duration, num_segments + 1).tolist()]

        logger.debug(
            f"Segmentation method: {method_used}. Found {len(bound_times)-1} segments."
        )
        return bound_times

    def analyze_audio(self, audio_path: str, analysis_config: AnalysisConfig) -> Optional[Dict[str, Any]]:
        """Calculates enhanced audio features using Ensemble approach (v5.4/5.5)."""
        profiler_start_time = time.time()
        logger.info(f"Analyzing audio (Ensemble v5.4/5.5): {os.path.basename(audio_path)}")
        target_sr = analysis_config.target_sr_audio
        try:
            if not os.path.exists(audio_path): raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.debug(f"Loading audio with TorchAudio (Target SR: {target_sr})...")
            # <<< CHANGED: Primary loading using torchaudio >>>
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            if sr != target_sr: logger.debug(f"Resampling {sr}Hz->{target_sr}Hz"); resampler = T.Resample(sr, target_sr, dtype=waveform.dtype); waveform = resampler(waveform); sr = target_sr
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True) # Mono
            duration = waveform.shape[1] / sr
            if duration <= 0: raise ValueError("Audio zero duration.")
            waveform_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
            logger.debug(f"Audio loaded: Shape={waveform.shape}, SR={sr}, Dur={duration:.2f}s")

            # --- Demucs Placeholder --- (Same as before)
            stems = {}
            if analysis_config.use_mrise_sync and analysis_config.use_demucs_for_mrise:
                if DEMUCS_AVAILABLE:
                    logger.info("Demucs sep (Placeholder)...")
                    stems = {
                        'drums': waveform_np * 0.5,
                        'vocals': waveform_np * 0.5
                    }
                else:
                    logger.warning("Demucs enabled but unavailable.")

            # --- Rhythm Analysis --- (Same as before)
            logger.debug("Analyzing rhythm...")
            beat_times, downbeat_times = self._detect_beats_downbeats(waveform_np, sr, analysis_config)
            tempo = 120.0
            if LIBROSA_AVAILABLE:
                try:
                    tempo_val = librosa.beat.tempo(y=waveform_np, sr=sr, aggregate=np.median)[0]
                    # Check if tempo_val is finite and positive before converting
                    if np.isfinite(tempo_val) and tempo_val > 0:
                        tempo = float(tempo_val)
                    # else: tempo remains 120.0 (the default)
                except Exception as te:
                    logger.warning(f"Librosa tempo detection failed: {te}")
            # <<< CORRECTED BLOCK END >>>
            else:
                logger.warning("Librosa unavailable for tempo detection.")
            logger.debug(f"Detected Tempo: {tempo:.1f} BPM")

            # --- Micro-Rhythm (MRISE) --- (Same as before)
            micro_beat_times = [] # Initialize first
            if analysis_config.use_mrise_sync:
                logger.debug("Analyzing micro-rhythms (MRISE)...")
                target_stem = 'drums'
                audio_for_mrise = stems.get(target_stem, waveform_np) # Use drums if available, else full mix
                stem_name_log = target_stem if target_stem in stems else None
                micro_beat_times = self._detect_micro_beats(audio_for_mrise, sr, analysis_config, stem_name_log)
                
            # --- Energy & Trends (HEFM) ---
            logger.debug("Computing energy & trends (HEFM)...")
            hop_e = analysis_config.hop_length_energy; frame_e = analysis_config.frame_length_energy
            # <<< CHANGED: Use torchaudio RMS >>>
            rms_torch = torchaudio.functional.compute_rms(waveform, frame_length=frame_e, hop_length=hop_e).squeeze(0)
            rms_energy = rms_torch.cpu().numpy(); rms_times = np.linspace(0, duration, len(rms_energy), endpoint=False) + (hop_e / sr / 2.0)
            smooth_win_s = analysis_config.trend_window_sec; smooth_win_f = max(11, int(sr*smooth_win_s / hop_e) | 1)
            rms_long = scipy.signal.savgol_filter(rms_energy, smooth_win_f, 3) if len(rms_energy) > smooth_win_f else rms_energy.copy()
            time_step = hop_e / sr if sr > 0 else 1.0
            rms_deriv_short = np.gradient(rms_energy, time_step) if len(rms_energy) > 1 else np.zeros_like(rms_energy)
            rms_deriv_long = np.gradient(rms_long, time_step) if len(rms_long) > 1 else np.zeros_like(rms_long)

            # --- Mel Spectrogram (PWRC) ---
            mel_spec = None; mel_t = None
            if analysis_config.use_latent_sync: logger.debug("Computing Mel Spectrogram...");
                # <<< CHANGED: Pass tensor >>>
            mel_spec = compute_mel_spectrogram(waveform, sr, hop_length=analysis_config.hop_length_mel)
            if mel_spec is not None: mel_frames = mel_spec.shape[1]; mel_t = np.linspace(0, duration, mel_frames, endpoint=False) + (analysis_config.hop_length_mel / sr / 2.0)
            else: logger.error("Mel Spectrogram failed.")

            # --- Segmentation ---
            logger.debug("Segmenting audio..."); bound_times = self._segment_audio(waveform, sr, analysis_config)

            # --- Aggregate Segment Features --- (Logic mostly unchanged, ensures correct indices)
            seg_features = []; logger.debug(f"Aggregating features for {len(bound_times)-1} segments...")
            for i in range(len(bound_times) - 1):
                t_s, t_e = bound_times[i], bound_times[i+1]; seg_dur = t_e - t_s;
                if seg_dur <= 1e-6: continue
                rms_idx = np.where((rms_times >= t_s) & (rms_times < t_e))[0]
                if len(rms_idx) == 0: mid_t = (t_s + t_e) / 2.0; rms_idx = [np.argmin(np.abs(rms_times - mid_t))] if len(rms_times) > 0 else [];
                if len(rms_idx) == 0: logger.warning(f"No RMS frames for segment {i}. Skipping."); continue
                seg_rms=float(np.mean(rms_energy[rms_idx])); seg_rms_l=float(np.mean(rms_long[rms_idx])); seg_trend_s=float(np.mean(rms_deriv_short[rms_idx])); seg_trend_l=float(np.mean(rms_deriv_long[rms_idx]))
                b_i = np.clip(seg_rms / (analysis_config.norm_max_rms + 1e-6), 0.0, 1.0); onset_p = np.clip(seg_trend_s / (analysis_config.norm_max_onset + 1e-6), -1.0, 1.0); e_i = (onset_p + 1.0) / 2.0; arousal_p = np.clip(tempo / 180.0, 0.1, 1.0); valence_p = 0.5; m_i = [float(arousal_p), float(valence_p)]
                seg_features.append({'start': float(t_s), 'end': float(t_e), 'duration': float(seg_dur), 'rms_avg': seg_rms, 'rms_avg_long': seg_rms_l, 'trend_short': seg_trend_s, 'trend_long': seg_trend_l, 'b_i': float(b_i), 'e_i': float(e_i), 'm_i': m_i })

            # --- Raw Features Dict ---
            raw = { 'rms_energy': rms_energy, 'rms_times': rms_times, 'rms_energy_long': rms_long, 'rms_deriv_short': rms_deriv_short, 'rms_deriv_long': rms_deriv_long, 'original_audio_path': audio_path, 'mel_spectrogram': mel_spec, 'mel_times': mel_t, 'waveform_np_for_ser': waveform_np if analysis_config.use_lvre_features else None, 'perc_times': rms_times, 'percussive_ratio': np.zeros_like(rms_times)} # <<< Pass numpy arrays directly >>>

            # --- Result ---
            result = {'sr': sr, 'duration': float(duration), 'tempo': float(tempo), 'beat_times': [float(t) for t in beat_times], 'downbeat_times': [float(t) for t in downbeat_times], 'micro_beat_times': [float(t) for t in micro_beat_times], 'segment_boundaries': [float(t) for t in bound_times], 'segment_features': seg_features, 'raw_features': raw }
            logger.info(f"Audio analysis complete ({time.time() - profiler_start_time:.2f}s)")
            if ENABLE_PROFILING: logger.debug(f"PROFILING: Audio Analysis took {time.time() - profiler_start_time:.3f}s")
            return result
        except FileNotFoundError as fnf_err: logger.error(f"Audio analysis FNF: {fnf_err}"); return None
        except ValueError as val_err: logger.error(f"Audio analysis ValErr: {val_err}"); return None
        except RuntimeError as rt_err: logger.error(f"Audio analysis RuntimeErr: {rt_err}"); return None
        except Exception as e: logger.error(f"Unexpected audio analysis error: {e}", exc_info=True); return None

# ========================================================================
#                    FACE UTILITIES (Returns Landmarks via MP Results)
# ========================================================================
class BBZFaceUtils: # (Minor update: Focus on getting features, landmarks are in results)
    def __init__(self, static_mode=False, max_faces=1, min_detect_conf=0.5, min_track_conf=0.5):
        self.face_mesh = None; self._mp_face_mesh = mp.solutions.face_mesh
        try:
            self.face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=static_mode, max_num_faces=max_faces,
                refine_landmarks=True, # <<< CHANGED: Explicitly True >>>
                min_detection_confidence=min_detect_conf,
                min_tracking_confidence=min_track_conf)
            logger.info("FaceMesh initialized (Refine Landmarks: True).")
        except Exception as e: logger.error(f"Failed init FaceMesh: {e}"); self.face_mesh = None

    def process_frame(self, image_bgr):
        """Processes a BGR frame and returns MediaPipe FaceMesh results."""
        if self.face_mesh is None:
            # logger.debug("FaceMesh not initialized, skipping frame processing.")
            return None
        try:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image_bgr.flags.writeable = False
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            # Mark image writeable again before returning
            image_bgr.flags.writeable = True
            return results
        except Exception as e:
            logger.warning(f"FaceMesh process error on frame: {e}")
            # Ensure image flags are reset even on error
            if hasattr(image_bgr, 'flags'): image_bgr.flags.writeable = True
            return None

    def get_heuristic_face_features(self, results, h, w, mouth_open_threshold=0.05):
        is_open, size_ratio, center_x = False, 0.0, 0.5
        if results and results.multi_face_landmarks:
            try:
                face_landmarks = results.multi_face_landmarks[0]; lm = face_landmarks.landmark
                if 13 < len(lm) and 14 < len(lm): # Mouth open (using simple V4 points)
                    upper_y = lm[13].y * h; lower_y = lm[14].y * h; mouth_h = abs(lower_y - upper_y)
                    if 10 < len(lm) and 152 < len(lm): # Face height ref
                        forehead_y = lm[10].y * h; chin_y = lm[152].y * h; face_h = abs(chin_y - forehead_y)
                        if face_h > 1e-6: is_open = (mouth_h / face_h) > mouth_open_threshold
                # Face size (using all landmarks)
                all_x = [p.x * w for p in lm if np.isfinite(p.x)]; all_y = [p.y * h for p in lm if np.isfinite(p.y)]
                if all_x and all_y: min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y); face_w = max(1, max_x - min_x); face_h = max(1, max_y - min_y); face_diag = math.sqrt(face_w**2 + face_h**2); img_diag = math.sqrt(w**2 + h**2);
                if img_diag > 1e-6: size_ratio = np.clip(face_diag / img_diag, 0.0, 1.0)
                # Face center (using mean X)
                if all_x: center_x = np.clip(np.mean(all_x) / w, 0.0, 1.0)
            except IndexError: logger.warning("Index error accessing FaceMesh landmarks.")
            except Exception as e: logger.warning(f"Error extracting heuristic face features: {e}")
        return is_open, float(size_ratio), float(center_x)
    
    def close(self):
        """Releases MediaPipe FaceMesh resources."""
        if self.face_mesh:
            try:
                self.face_mesh.close()
                logger.info("FaceMesh resources released.")
            except Exception as e:
                logger.error(f"Error closing FaceMesh: {e}")
        self.face_mesh = None # Ensure reset

# ========================================================================
#         DYNAMIC SEGMENT IDENTIFIER (Unchanged - Now Optional/Legacy)
# ========================================================================
class DynamicSegmentIdentifier: # (Implementation unchanged from v4.7.3 - Used only if needed by a specific old heuristic)
    def __init__(self, analysis_config: AnalysisConfig, fps: float):
        self.fps = fps if fps > 0 else 30.0
        # Use score_threshold from V4 compatible section of config if this class is used
        self.score_threshold = getattr(analysis_config, 'score_threshold', 0.3)
        self.min_segment_len_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES, int(analysis_config.min_potential_clip_duration_sec * self.fps))
        logger.debug(f"Legacy Segment Identifier Init: FPS={self.fps:.2f}, Threshold={self.score_threshold:.2f}, MinLenFrames={self.min_segment_len_frames}")

    def find_potential_segments(self, frame_features_list):
        """Finds segments based on legacy boosted_score threshold."""
        potential_segments = []
        start_idx = -1; n = len(frame_features_list)
        if n == 0: return []
        for i, features in enumerate(frame_features_list):
            # Use 'boosted_score' which was the V4 heuristic score
            score = features.get('boosted_score', 0.0) if isinstance(features, dict) else 0.0
            is_candidate = score >= self.score_threshold
            if is_candidate and start_idx == -1: start_idx = i # Start segment
            elif not is_candidate and start_idx != -1: # End segment
                segment_len = i - start_idx
                if segment_len >= self.min_segment_len_frames:
                    potential_segments.append({'start_frame': start_idx, 'end_frame': i})
                start_idx = -1 # Reset start
        # Handle segment ending at the last frame
        if start_idx != -1:
            segment_len = n - start_idx
            if segment_len >= self.min_segment_len_frames:
                potential_segments.append({'start_frame': start_idx, 'end_frame': n})

        min_len_s = self.min_segment_len_frames / (self.fps if self.fps > 0 else 30.0)
        logger.info(f"Legacy Segment ID identified {len(potential_segments)} potential segments (BoostedScore >= {self.score_threshold:.2f}, MinLen={min_len_s:.2f}s)")
        return potential_segments


# ========================================================================
#                      IMAGE UTILITIES (Unchanged)
# ========================================================================
class BBZImageUtils: # (Unchanged from v4.7.3)
    def resizeTARGET(self, image, TARGET_HEIGHT=256, TARGET_WIDTH=256):
        """Resizes image to target dimensions using appropriate interpolation."""
        if image is None or image.size == 0:
            logger.warning("Attempted to resize an empty image.")
            return None
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            logger.warning(f"Attempted to resize image with zero dimension: {h}x{w}")
            return None

        if h != TARGET_HEIGHT or w != TARGET_WIDTH:
            # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
            interpolation = cv2.INTER_AREA if h > TARGET_HEIGHT or w > TARGET_WIDTH else cv2.INTER_LINEAR
            try:
                resized_image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=interpolation)
                return resized_image
            except cv2.error as cv_err:
                logger.warning(f"OpenCV resize failed: {cv_err}. Returning original image.")
                return image
            except Exception as e:
                logger.warning(f"Unexpected error during resize: {e}. Returning original image.")
                return image
        return image # Return original if already target size

# ========================================================================
#                       PREPROCESSING FUNCTION (LVRE - Batched, Lazy Load)
# ========================================================================
# ========================================================================
#           PREPROCESSING FUNCTION (LVRE - NEW for v5.4/5.5)
# ========================================================================
def preprocess_lyrics_and_visuals(master_audio_data: Dict, all_potential_clips: List['ClipSegment'], analysis_config: AnalysisConfig):
    """Runs ASR, SER, Text & Visual Embeddings if LVRE enabled, with lazy loading and batching."""
    if not analysis_config.use_lvre_features:
        logger.info("LVRE features disabled by config. Skipping preprocessing.")
        return
    if not TRANSFORMERS_AVAILABLE:
        logger.error("LVRE requires `transformers` library. Aborting LVRE preprocessing.")
        return

    logger.info("--- Starting LVRE Preprocessing (Batched, Lazy Load) ---")
    # <<< CORRECTED >>>
    if ENABLE_PROFILING:
        profiler_start_time = time.time()
    device = get_device()

    # --- 1. ASR (Whisper) ---
    timed_lyrics = master_audio_data.get('timed_lyrics')
    if timed_lyrics is None:
        logger.info("Running ASR (Whisper)...")
        asr_success = False
        try:
            audio_path = master_audio_data.get("raw_features", {}).get("original_audio_path")
            if not audio_path or not os.path.exists(audio_path):
                raise ValueError("Audio path missing/invalid for ASR.")

            whisper_pipeline, _ = get_pytorch_model(
                f"whisper_{analysis_config.whisper_model_name}",
                load_huggingface_pipeline_func,
                task="automatic-speech-recognition", model_name=analysis_config.whisper_model_name, device=device,
                chunk_length_s=30, stride_length_s=5, return_timestamps="word"
            )

            # <<< CORRECTED STRUCTURE >>>
            if whisper_pipeline:
                logger.debug(f"Running Whisper pipeline on {os.path.basename(audio_path)}...")
                asr_result = whisper_pipeline(audio_path)

                if isinstance(asr_result, dict) and 'chunks' in asr_result:
                    timed_lyrics = asr_result.get('chunks', [])
                    valid_count = 0
                    # <<< CORRECTED LOOP >>>
                    for chunk in timed_lyrics:
                        ts = chunk.get('timestamp')
                        if isinstance(ts, (tuple, list)) and len(ts) == 2 and isinstance(ts[0], (int, float)) and isinstance(ts[1], (int, float)):
                            valid_count += 1
                        else:
                            chunk['timestamp'] = None
                    master_audio_data['timed_lyrics'] = timed_lyrics
                    logger.info(f"ASR complete: {len(timed_lyrics)} word chunks found ({valid_count} with valid timestamps).")
                    asr_success = True
                else:
                    logger.warning(f"ASR result format unexpected: {type(asr_result)}. Keys: {asr_result.keys() if isinstance(asr_result, dict) else 'N/A'}")
                    master_audio_data['timed_lyrics'] = [] # Ensure list on format error
            else:
                raise RuntimeError("Whisper pipeline failed to load.")

        except Exception as asr_err:
            logger.error(f"ASR processing failed: {asr_err}", exc_info=True)
            master_audio_data['timed_lyrics'] = []
        # <<< CORRECTED >>>
        finally:
             if not asr_success:
                 master_audio_data['timed_lyrics'] = []
    else:
        logger.info(f"Skipping ASR (already present: {len(timed_lyrics)} words).")

    # --- 2. SER ---
    # Check if timed_lyrics exists and is not empty before checking content
    if timed_lyrics and not all('emotion_score' in w for w in timed_lyrics):
        logger.info("Running SER (Speech Emotion Recognition)...")
        ser_success = False
        try:
            ser_pipeline, _ = get_pytorch_model(
                f"ser_{analysis_config.ser_model_name}",
                load_huggingface_pipeline_func,
                task="audio-classification", model_name=analysis_config.ser_model_name, device=device, top_k=1
            )
            if ser_pipeline:
                sr_audio = master_audio_data.get('sr')
                waveform_np_full = master_audio_data.get("raw_features", {}).get("waveform_np_for_ser")
                if waveform_np_full is None: raise ValueError("Raw waveform missing for SER.")
                if not isinstance(waveform_np_full, np.ndarray): raise TypeError("Waveform not numpy array.")

                ser_batch_size = analysis_config.lvre_batch_size_ser
                pbar_ser = tqdm(total=len(timed_lyrics), desc="SER Processing", leave=False, disable=not TQDM_AVAILABLE)
                logger.info(f"Processing SER (Batch: {ser_batch_size})...")

                for i in range(0, len(timed_lyrics), ser_batch_size):
                    batch_lyrics = timed_lyrics[i:i+ser_batch_size] # Use correct variable name
                    batch_audio_snippets = []; valid_indices_in_batch = []

                    # <<< CORRECTED LOOP >>>
                    for k, word_info in enumerate(batch_lyrics): # Use correct variable name
                        ts = word_info.get('timestamp'); word_text = word_info.get('text', '').strip()
                        if isinstance(ts, (tuple, list)) and len(ts) == 2 and word_text:
                            start_t, end_t = ts
                            if isinstance(start_t, (int, float)) and isinstance(end_t, (int, float)) and end_t > start_t:
                                start_sample = int(start_t * sr_audio); end_sample = min(int(end_t * sr_audio), waveform_np_full.shape[-1])
                                start_sample = max(0, min(start_sample, end_sample - 1))
                                if start_sample < end_sample:
                                    batch_audio_snippets.append(waveform_np_full[start_sample:end_sample].astype(np.float32))
                                    valid_indices_in_batch.append(k)
                                else:
                                    word_info['emotion_score'] = 0.0 # Assign default if slice empty
                            else:
                                word_info['emotion_score'] = 0.0 # Assign default if times invalid
                        else:
                            word_info['emotion_score'] = 0.0 # Assign default if no timestamp/text

                    # <<< CORRECTED STRUCTURE/INDENTATION >>>
                    if batch_audio_snippets:
                        try:
                            batch_results = ser_pipeline(batch_audio_snippets, sampling_rate=sr_audio)
                            if isinstance(batch_results, list) and len(batch_results) == len(batch_audio_snippets):
                                for res_list, orig_idx in zip(batch_results, valid_indices_in_batch):
                                    top_result = res_list[0] if isinstance(res_list, list) and res_list else (res_list if isinstance(res_list, dict) else None)
                                    if isinstance(top_result, dict):
                                        score = top_result.get('score', 0.0)
                                        batch_lyrics[orig_idx]['emotion_score'] = float(score) # Use correct variable
                                    else:
                                        batch_lyrics[orig_idx]['emotion_score'] = 0.0 # Use correct variable
                            # <<< CORRECTED ELSE BLOCK >>>
                            else:
                                logger.warning(f"SER batch output format mismatch.")
                                for k in valid_indices_in_batch: # Use correct variable
                                     batch_lyrics[k]['emotion_score'] = 0.0 # Use correct variable

                        except Exception as ser_batch_err:
                            logger.error(f"SER batch {i//ser_batch_size} failed: {ser_batch_err}")
                            for k in valid_indices_in_batch: # Use correct variable
                                batch_lyrics[k]['emotion_score'] = 0.0 # Use correct variable

                    # Ensure all items in the original batch slice have the key
                    for k in range(len(batch_lyrics)): # Use correct variable
                         batch_lyrics[k].setdefault('emotion_score', 0.0) # Use correct variable
                    if pbar_ser: pbar_ser.update(len(batch_lyrics)) # Use correct variable

                if pbar_ser: pbar_ser.close()
                ser_success = True; logger.info("SER complete.")
            else:
                raise RuntimeError("SER pipeline failed load.")
        except Exception as ser_err:
            logger.error(f"SER failed: {ser_err}", exc_info=True)
        # <<< CORRECTED >>>
        finally:
            if not ser_success:
                for w in timed_lyrics: w.setdefault('emotion_score', 0.0)
    else:
        logger.info("Skipping SER (lyrics missing or scores present).")

    # --- 3. Embed Lyrics ---
    # Check if timed_lyrics exists and is not empty before checking content
    if timed_lyrics and not all('embedding' in w for w in timed_lyrics):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Cannot embed lyrics: sentence-transformers missing.")
        else:
            logger.info("Embedding lyrics...")
            text_embed_success = False
            try:
                text_model, text_dev = get_pytorch_model(
                    analysis_config.text_embed_model_name,
                    load_sentence_transformer_func, analysis_config.text_embed_model_name
                )
                if text_model:
                    texts = [w.get('text','').strip().lower() for w in timed_lyrics]
                    valid_i = [i for i, t in enumerate(texts) if t]
                    valid_t = [texts[i] for i in valid_i]
                    if valid_t:
                        logger.debug(f"Encoding {len(valid_t)} lyric words...")
                        text_batch = analysis_config.lvre_batch_size_text
                        embeds = text_model.encode(
                            valid_t, convert_to_numpy=True,
                            show_progress_bar=(TQDM_AVAILABLE),
                            batch_size=text_batch, device=text_dev
                        )
                        for idx, emb in zip(valid_i, embeds):
                            timed_lyrics[idx]['embedding'] = emb.tolist()
                    # Ensure all words have the key, assign None if needed
                    for i in range(len(timed_lyrics)):
                        timed_lyrics[i].setdefault('embedding', None)
                    text_embed_success = True; logger.info("Lyric embeddings generated.")
                else:
                    raise RuntimeError("Sentence Transformer failed load.")
            except Exception as text_err:
                logger.error(f"Text embedding failed: {text_err}", exc_info=True)
            # <<< CORRECTED >>>
            finally:
                if not text_embed_success:
                    for w in timed_lyrics: w.setdefault('embedding', None)
    else:
        logger.info("Skipping Text Embeddings (lyrics missing or embeds present).")

    # --- 4. Embed Visual Clips ---
    clips_need_embed = [c for c in all_potential_clips if not hasattr(c,'visual_embedding') or c.visual_embedding is None]
    if clips_need_embed:
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot embed visuals: transformers missing.")
        else:
            logger.info(f"Generating visual embeddings for {len(clips_need_embed)} clips...")
            vis_embed_success = False; clip_proc = None; clip_model = None; clip_dev = None
            readers: Dict[str, Optional[VideoFileClip]] = {}
            try:
                proc_key = f"clip_proc_{analysis_config.vision_embed_model_name}"
                model_key = f"clip_model_{analysis_config.vision_embed_model_name}"
                clip_proc = get_pytorch_processor(proc_key, load_huggingface_processor_func, analysis_config.vision_embed_model_name, CLIPProcessor)
                clip_model, clip_dev = get_pytorch_model(model_key, load_huggingface_model_func, analysis_config.vision_embed_model_name, CLIPModel)

                if clip_model and clip_proc and clip_dev:
                    kframes = []; clip_map = {id(c): i for i, c in enumerate(all_potential_clips)}
                    logger.debug("Extracting keyframes...")
                    pbar_kf = tqdm(total=len(clips_need_embed), desc="Keyframes", disable=not TQDM_AVAILABLE)
                    for clip in clips_need_embed:
                        path = str(clip.source_video_path)
                        if not path or not os.path.exists(path):
                            clip.visual_embedding = None
                            if pbar_kf: pbar_kf.update(1)
                            continue
                        reader = readers.get(path)
                        if reader is None and path not in readers:
                            try:
                                reader = VideoFileClip(path, audio=False)
                                readers[path] = reader
                            except Exception as re:
                                readers[path] = None; logger.error(f"Reader fail {os.path.basename(path)}: {re}"); reader = None
                        elif path in readers and reader is None: pass # Known failure

                        if reader:
                            try:
                                kf_time = np.clip(clip.start_time + clip.duration / 2.0, 0, reader.duration - 1e-6 if reader.duration else 0)
                                kf_rgb = reader.get_frame(kf_time)
                                if kf_rgb is not None and kf_rgb.size > 0:
                                    kframes.append((clip_map[id(clip)], kf_rgb))
                                else:
                                    clip.visual_embedding = None
                            except Exception as kfe:
                                clip.visual_embedding = None; logger.warning(f"Keyframe fail clip {clip.start_frame}: {kfe}")
                        else:
                            clip.visual_embedding = None
                        if pbar_kf: pbar_kf.update(1)
                    if pbar_kf: pbar_kf.close()

                    # Close readers
                    for r in readers.values():
                        if r and hasattr(r, 'close'):
                            try: r.close()
                            except: pass
                    readers.clear()

                    if kframes:
                        logger.debug(f"Embedding {len(kframes)} keyframes...")
                        vis_batch = analysis_config.lvre_batch_size_vision
                        pbar_emb = tqdm(total=len(kframes), desc="Vis Embeds", disable=not TQDM_AVAILABLE)
                        with torch.no_grad():
                            for i in range(0, len(kframes), vis_batch):
                                batch_dat = kframes[i:i+vis_batch]
                                batch_orig_idx = [item[0] for item in batch_dat]
                                batch_rgb = [item[1] for item in batch_dat]
                                try:
                                    inputs = clip_proc(images=batch_rgb, return_tensors="pt", padding=True).to(clip_dev)
                                    img_feat = clip_model.get_image_features(**inputs)
                                    img_feat = F.normalize(img_feat, p=2, dim=-1)
                                    embeds_np = img_feat.cpu().numpy()
                                    for j, orig_idx in enumerate(batch_orig_idx):
                                        if 0 <= orig_idx < len(all_potential_clips):
                                             all_potential_clips[orig_idx].visual_embedding = embeds_np[j].tolist()
                                except Exception as emb_err:
                                    logger.error(f"Vis embed batch {i//vis_batch} failed: {emb_err}")
                                    for orig_idx in batch_orig_idx:
                                        if 0 <= orig_idx < len(all_potential_clips):
                                             all_potential_clips[orig_idx].visual_embedding = None
                                finally:
                                     if pbar_emb: pbar_emb.update(len(batch_dat))
                        if pbar_emb: pbar_emb.close()
                        vis_embed_success = True; logger.info("Visual embedding finished.")
                    else:
                        logger.info("No keyframes for visual embedding.")
                        for clip in clips_need_embed: clip.visual_embedding = None
                else:
                    logger.error("CLIP load failed.")
                    for clip in clips_need_embed: clip.visual_embedding = None
            except Exception as vis_outer:
                logger.error(f"Outer visual embed err: {vis_outer}", exc_info=True)
                for clip in clips_need_embed: clip.visual_embedding = None
            # <<< CORRECTED >>>
            finally:
                if not vis_embed_success:
                    for clip in clips_need_embed: clip.visual_embedding = None
    else:
        logger.info("Skipping visual embeds (already present).")

    if ENABLE_PROFILING:
        logger.debug(f"PROFILING: LVRE Preprocessing took {time.time() - profiler_start_time:.3f}s")

# ========================================================================
#                       STYLE ANALYSIS FUNCTION (SAAPV)
# ========================================================================
def analyze_music_style(master_audio_data: Dict, analysis_config: AnalysisConfig) -> str:
    """Analyzes basic audio features to determine a broad musical style for pacing adaptation."""
    logger.info("Analyzing music style (SAAPV)...")
    try:
        tempo = master_audio_data.get('tempo', 120.0); rms_energy = master_audio_data.get('raw_features', {}).get('rms_energy');
        if rms_energy is not None and not isinstance(rms_energy, np.ndarray): rms_energy = np.asarray(rms_energy, dtype=np.float32)
        onset_times = master_audio_data.get('micro_beat_times') or master_audio_data.get('beat_times', [])
        duration = master_audio_data.get('duration', 0)
        if duration <= 0: return "Unknown"
        tempo_cat = "Slow" if tempo < 95 else ("Medium" if tempo < 135 else "Fast")
        dyn_cat = "Medium"; rms_var_norm = 0.0
        if rms_energy is not None and len(rms_energy) > 1: rms_var = np.var(rms_energy); rms_var_norm = np.clip(rms_var / 0.05, 0.0, 1.0); dyn_cat = "Low" if rms_var_norm < 0.3 else ("Medium" if rms_var_norm < 0.7 else "High")
        comp_cat = "Moderate"; rhythm_comp = 0.5
        if onset_times and len(onset_times) > 1:
            ioi = np.diff(onset_times); ioi = ioi[ioi > 0.01]
            if len(ioi) > 1: ioi_var = np.var(ioi); rhythm_comp = np.clip(1.0 - exp(-ioi_var * 50.0), 0.0, 1.0); comp_cat = "Simple" if rhythm_comp < 0.4 else ("Moderate" if rhythm_comp < 0.8 else "Complex")
            else: rhythm_comp = 0.0; comp_cat = "Simple"
        else: rhythm_comp = 0.0; comp_cat = "Simple"
        logger.debug(f"SAAPV Style: Tempo={tempo:.1f}({tempo_cat}), Dyn={rms_var_norm:.2f}({dyn_cat}), Rhythm={rhythm_comp:.2f}({comp_cat})")
        if tempo_cat == "Fast" and dyn_cat == "High" and comp_cat != "Simple": style = "High-Energy Complex"
        elif tempo_cat == "Fast" and dyn_cat != "Low": style = "Uptempo Pop/Electronic"
        elif tempo_cat == "Slow" and dyn_cat == "Low" and comp_cat == "Simple": style = "Ballad/Ambient"
        elif tempo_cat == "Slow": style = "Slow Groove/RnB/Chill"
        elif dyn_cat == "High": style = "Dynamic Rock/Orchestral"
        elif comp_cat == "Complex": style = "Complex Rhythm (Jazz/Prog)"
        else: style = "Mid-Tempo Pop/General"
        logger.info(f"Detected Music Style (SAAPV): {style}")
        return style
    except Exception as e: logger.error(f"Music style analysis error: {e}", exc_info=True); return "Unknown"

# ========================================================================
#      BATCH SYNC NET SCORING FUNCTION (NEW for v5.4/5.5)
# ========================================================================
# ========================================================================
#      BATCH SYNC NET SCORING FUNCTION (NEW for v5.4/5.5)
# ========================================================================
def batch_score_syncnet(clips_to_score: List[ClipSegment], syncnet_model: SyncNet, full_mel_spectrogram: np.ndarray, mel_times: np.ndarray, config: AnalysisConfig):
    """Scores multiple clips for lip-sync using SyncNet in batches."""
    # <<< CORRECTED BLOCK START >>>
    if not clips_to_score or syncnet_model is None or full_mel_spectrogram is None or mel_times is None:
        logger.warning("Skipping batch SyncNet scoring: Missing clips, model, or Mel data.")
        # Ensure scores are defaulted to None or 0.0 if scoring skipped
        for clip in clips_to_score:
            clip.latent_sync_score = None # Initialize/default score
        return # Exit the function
    if not SKIMAGE_AVAILABLE:
        logger.warning("Skipping batch SyncNet scoring: scikit-image not available.")
        for clip in clips_to_score:
            clip.latent_sync_score = None # Initialize/default score
        return # Exit the function
    # <<< CORRECTED BLOCK END >>>

    start_time = time.time()
    device = next(syncnet_model.parameters()).device
    syncnet_model.eval() # Ensure model is in eval mode

    all_video_batch_items = [] # Stores preprocessed video tensors (N, C, T=5, H, W)
    all_audio_batch_items = [] # Stores preprocessed audio tensors (N, 1, M, T_mel)
    batch_item_map: List[Tuple[int, int]] = [] # Maps batch item index back to (clip_idx, window_idx)

    logger.info(f"Preparing {len(clips_to_score)} clips for batched SyncNet scoring...")
    pbar_prep = tqdm(total=len(clips_to_score), desc="Prep SyncNet Batches", leave=False, disable=not TQDM_AVAILABLE)

    # Define Mel Window Parameters
    mel_hop_sec = config.hop_length_mel / config.target_sr_audio if config.target_sr_audio > 0 else 0.01
    mel_window_size_frames = int(round(0.2 / mel_hop_sec)) if mel_hop_sec > 0 else 20
    n_mels = full_mel_spectrogram.shape[0]
    mel_total_frames = full_mel_spectrogram.shape[1]

    # Prepare Batch Items
    for clip_idx, clip in enumerate(clips_to_score):
        clip.latent_sync_score = 0.0 # Initialize score (use 0.0 as default indicating no sync found yet)

        # Get Valid Mouth Crops
        mouth_crops = []; original_frame_indices = []
        for i, f in enumerate(clip.segment_frame_features):
            if isinstance(f, dict):
                 crop = f.get('mouth_crop')
                 if crop is not None and isinstance(crop, np.ndarray) and crop.shape == (112, 112, 3):
                     mouth_crops.append(crop)
                     original_frame_indices.append(clip.start_frame + i)

        # Subsample if Needed
        num_original_crops = len(mouth_crops)
        if num_original_crops > LATENTSYNC_MAX_FRAMES:
            subsample_indices = np.linspace(0, num_original_crops - 1, LATENTSYNC_MAX_FRAMES, dtype=int)
            mouth_crops = [mouth_crops[i] for i in subsample_indices]
            original_frame_indices = [original_frame_indices[i] for i in subsample_indices]

        if len(mouth_crops) < 5: # Need 5 frames for SyncNet window
            if pbar_prep: pbar_prep.update(1)
            continue # Skip this clip

        # Create 5-frame Windows
        num_windows = len(mouth_crops) - 4
        for win_idx in range(num_windows):
            # 1. Prepare Video Chunk
            video_chunk_bgr = mouth_crops[win_idx:win_idx+5]; video_chunk_processed = []
            for frame_bgr in video_chunk_bgr:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB); frame_float = frame_rgb.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1); video_chunk_processed.append(frame_tensor)
            video_batch_item = torch.stack(video_chunk_processed, dim=1) # (C, T=5, H, W)

            # 2. Prepare Audio Chunk (Mel Spectrogram)
            center_video_frame_idx_in_clip = win_idx + 2; center_original_frame_idx = original_frame_indices[center_video_frame_idx_in_clip]
            center_time = (center_original_frame_idx + 0.5) / clip.fps
            center_mel_idx = np.argmin(np.abs(mel_times - center_time))
            start_mel = max(0, center_mel_idx - mel_window_size_frames // 2); end_mel = min(mel_total_frames, start_mel + mel_window_size_frames); start_mel = max(0, end_mel - mel_window_size_frames)
            audio_chunk_mel = full_mel_spectrogram[:, start_mel:end_mel]
            current_len = audio_chunk_mel.shape[1]
            if current_len < mel_window_size_frames:
                padding_needed = mel_window_size_frames - current_len; padding = np.full((n_mels, padding_needed), np.min(full_mel_spectrogram), dtype=np.float32); audio_chunk_mel = np.concatenate((audio_chunk_mel, padding), axis=1)
            elif current_len > mel_window_size_frames:
                 audio_chunk_mel = audio_chunk_mel[:, :mel_window_size_frames]
            audio_batch_item = torch.from_numpy(audio_chunk_mel).unsqueeze(0) # (1, Mels, Time)

            # Add prepared items to batch lists
            all_video_batch_items.append(video_batch_item); all_audio_batch_items.append(audio_batch_item); batch_item_map.append((clip_idx, win_idx))

        if pbar_prep: pbar_prep.update(1)
    if pbar_prep: pbar_prep.close()

    if not all_video_batch_items:
        logger.warning("No valid windows found for SyncNet batch processing.")
        return # All clips were too short or had no crops

    # Run Inference in Batches
    num_batch_items = len(all_video_batch_items); batch_size = config.syncnet_batch_size; all_confidences = []
    logger.info(f"Running SyncNet inference on {num_batch_items} windows in batches of {batch_size}...")
    pbar_infer = tqdm(total=num_batch_items, desc="SyncNet Inference", leave=False, disable=not TQDM_AVAILABLE)
    try:
        with torch.no_grad():
            for i in range(0, num_batch_items, batch_size):
                video_batch = torch.stack(all_video_batch_items[i : i + batch_size]).to(device); audio_batch = torch.stack(all_audio_batch_items[i : i + batch_size]).to(device)
                audio_embed, video_embed = syncnet_model(audio_batch, video_batch); confidences = F.cosine_similarity(audio_embed, video_embed, dim=-1)
                all_confidences.extend(confidences.cpu().numpy().tolist());
                if pbar_infer: pbar_infer.update(video_batch.size(0))
    except Exception as infer_err:
         logger.error(f"SyncNet batch inference failed: {infer_err}", exc_info=True);
         if pbar_infer: pbar_infer.close(); return
    finally:
         if pbar_infer: pbar_infer.close()

    # Assign Scores Back to Clips
    logger.info("Assigning SyncNet scores back to clips...")
    clip_max_scores = defaultdict(lambda: 0.0) # Default score is 0.0
    if len(all_confidences) != len(batch_item_map):
        logger.error(f"Mismatch between SyncNet confidences ({len(all_confidences)}) and batch map ({len(batch_item_map)})."); return
    for i, conf in enumerate(all_confidences):
        clip_list_idx, _ = batch_item_map[i]; score = float(np.clip((conf + 1.0) / 2.0, 0.0, 1.0)); clip_max_scores[clip_list_idx] = max(clip_max_scores[clip_list_idx], score)
    for clip_list_idx, max_score in clip_max_scores.items():
        if 0 <= clip_list_idx < len(clips_to_score):
            clips_to_score[clip_list_idx].latent_sync_score = max_score

    logger.info(f"Batch SyncNet scoring finished ({time.time() - start_time:.2f}s).")

# ========================================================================
#               CLIP SEGMENT DATA STRUCTURE (Ensemble v5.3 Ready)
# ========================================================================
class ClipSegment:
    def __init__(self, source_video_path: str, start_frame: int, end_frame: int, fps: float,
                 all_frame_features: List[Dict], analysis_config: AnalysisConfig,
                 master_audio_data: Dict):
        self.source_video_path = source_video_path
        self.start_frame = start_frame; self.end_frame = end_frame # Exclusive index
        self.num_frames = max(0, end_frame - start_frame)
        self.fps = fps if fps > 0 else 30.0
        self.duration = self.num_frames / self.fps if self.fps > 0 else 0.0
        self.start_time = start_frame / self.fps if self.fps > 0 else 0.0
        self.end_time = end_frame / self.fps if self.fps > 0 else 0.0
        self.analysis_config = analysis_config # Store config dataclass used for analysis

        # Safely slice features for this segment
        if 0 <= start_frame < end_frame <= len(all_frame_features):
             self.segment_frame_features = all_frame_features[start_frame:end_frame]
        else:
             logger.warning(f"Invalid frame indices [{start_frame}-{end_frame}] for ClipSegment (Total frames: {len(all_frame_features)}) from '{os.path.basename(str(source_video_path))}'. Segment features will be empty.")
             self.segment_frame_features = []
             self.num_frames = 0; self.duration = 0.0 # Reset if slice failed

        # === Feature Aggregation (Called during init) ===
        # Initialize all features to defaults
        self._initialize_features()

        if self.segment_frame_features:
            self._aggregate_visual_features()
            self._assign_audio_segment_features(master_audio_data) # Find corresponding audio segment
        else:
            # Log warning only if duration > 0 but features are missing (should not happen if slicing logic is correct)
             if self.duration > 0: logger.warning(f"Cannot aggregate features for clip {start_frame}-{end_frame} due to empty segment_frame_features list.")

    def _initialize_features(self):
        """Initializes all aggregated features to default values."""
        # Heuristic / Basic Visuals
        self.avg_raw_score: float = 0.0; self.avg_boosted_score: float = 0.0; self.peak_boosted_score: float = 0.0
        self.avg_motion_heuristic: float = 0.0; self.avg_jerk_heuristic: float = 0.0; self.avg_camera_motion: float = 0.0
        self.face_presence_ratio: float = 0.0; self.avg_face_size: float = 0.0
        self.intensity_category: str = "Low"; self.dominant_contributor: str = "none"; self.contains_beat: bool = False
        self.musical_section_indices: Set[int] = set()
        self.avg_lip_activity: float = 0.0 # Placeholder

        # Physics / HEFM Visuals
        self.avg_visual_flow: float = 0.0 # Renamed v_k
        self.avg_visual_accel: float = 0.0 # Renamed a_j
        self.avg_depth_variance: float = 0.0 # Renamed d_r (now unnormalized)
        self.avg_visual_entropy: float = 0.0 # Renamed phi
        self.avg_pose_kinetic: float = 0.0 # Specific pose energy
        self.avg_visual_flow_trend: float = 0.0 # Calculated from frame trends
        self.avg_visual_pose_trend: float = 0.0 # Calculated from frame trends
        self.visual_mood_vector: List[float] = [0.0, 0.0] # Based on visual features (V4 physics mode)
        # V4 Physics Mode specific features (needed if using that builder)
        self.v_k: float = 0.0; self.a_j: float = 0.0; self.d_r: float = 0.0; self.phi: float = 0.0
        self.mood_vector: List[float] = [0.0, 0.0] # V4 Physics Mode

        # PWRC Feature
        self.latent_sync_score: Optional[float] = None # SyncNet score (calculated later)

        # LVRE Feature
        self.visual_embedding: Optional[List[float]] = None # CLIP embedding (calculated later)

        # Corresponding Audio Segment Features (copied for convenience)
        self.audio_segment_data: Dict = {} # Stores features of the matched audio segment

        # Sequence specific (set during building)
        self.sequence_start_time: float = 0.0
        self.sequence_end_time: float = 0.0
        self.chosen_duration: float = 0.0
        self.chosen_effect: Optional[EffectParams] = None
        self.subclip_start_time_in_source: float = 0.0
        self.subclip_end_time_in_source: float = 0.0


    def _aggregate_visual_features(self):
        """Aggregates frame-level features into segment-level features."""
        count = len(self.segment_frame_features)
        if count == 0: return # Should not happen if called correctly

        # --- Helper to safely get mean of a feature list ---
        def safe_mean(key, default=0.0, sub_dict=None):
            values = []
            for f in self.segment_frame_features:
                if not isinstance(f, dict): continue
                container = f.get(sub_dict) if sub_dict else f
                if isinstance(container, dict):
                    val = container.get(key)
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        values.append(val)
            return float(np.mean(values)) if values else default

        # --- Heuristic / Basic Aggregations ---
        self.avg_raw_score = safe_mean('raw_score')
        self.avg_boosted_score = safe_mean('boosted_score')
        # Use max for peak score
        peak_scores = [f.get('boosted_score', 0.0) for f in self.segment_frame_features if isinstance(f, dict) and np.isfinite(f.get('boosted_score', -np.inf))]
        self.peak_boosted_score = float(np.max(peak_scores)) if peak_scores else 0.0

        self.avg_motion_heuristic = safe_mean('kinetic_energy_proxy', sub_dict='pose_features')
        self.avg_jerk_heuristic = safe_mean('movement_jerk_proxy', sub_dict='pose_features')
        self.avg_camera_motion = safe_mean('flow_velocity') # Use flow velocity as camera motion proxy

        face_sizes = [f.get('pose_features', {}).get('face_size_ratio', 0.0) for f in self.segment_frame_features if isinstance(f, dict) and f.get('pose_features', {}).get('face_size_ratio', 0.0) > 1e-3]
        face_present_frames = len(face_sizes)
        self.face_presence_ratio = float(face_present_frames / count)
        self.avg_face_size = float(np.mean(face_sizes)) if face_sizes else 0.0

        self.contains_beat = any(f.get('is_beat_frame', False) for f in self.segment_frame_features if isinstance(f, dict))
        self.musical_section_indices = {f.get('musical_section_index', -1) for f in self.segment_frame_features if isinstance(f, dict) and f.get('musical_section_index', -1) != -1}
        # Placeholder for lip activity calculation (e.g., variance of mouth height)
        # self.avg_lip_activity = ...

        # --- Aggregate Physics/HEFM features ---
        self.avg_visual_flow = safe_mean('flow_velocity')
        self.avg_pose_kinetic = safe_mean('kinetic_energy_proxy', sub_dict='pose_features')
        self.avg_visual_accel = safe_mean('flow_acceleration')
        self.avg_depth_variance = safe_mean('depth_variance')
        self.avg_visual_entropy = safe_mean('histogram_entropy')

        # Calculate average trends from frame-level trends
        self.avg_visual_flow_trend = safe_mean('visual_flow_trend')
        self.avg_visual_pose_trend = safe_mean('visual_pose_trend')

        # --- Populate V4 Physics Mode features (for compatibility) ---
        # These are often normalized versions or direct copies
        self.v_k = np.clip(self.avg_visual_flow / (self.analysis_config.norm_max_visual_flow + 1e-6), 0.0, 1.0)
        self.a_j = np.clip(self.avg_visual_accel / (100.0 + 1e-6), 0.0, 1.0) # Needs A_MAX_EXPECTED if physics accel used
        self.d_r = np.clip(self.avg_depth_variance / (self.analysis_config.norm_max_depth_variance + 1e-6), 0.0, 1.0)
        self.phi = self.avg_visual_entropy # Already entropy

        # Determine visual mood vector (Example V4: Flow ~ Arousal, Inv Depth Var ~ Valence)
        norm_flow_v4 = self.v_k
        norm_depth_v4 = self.d_r
        # V4 used (Flow, 1-DepthVar) as mood
        self.mood_vector = [float(norm_flow_v4), float(1.0 - norm_depth_v4)]
        # Keep visual_mood_vector potentially separate if Ensemble uses different mapping
        self.visual_mood_vector = self.mood_vector.copy()


        # Determine dominant contributor & intensity (reuse V4 heuristic helpers)
        dominant_contribs = [f.get('dominant_contributor', 'none') for f in self.segment_frame_features if isinstance(f, dict)]
        if dominant_contribs:
            non_none = [c for c in dominant_contribs if c != 'none']
            self.dominant_contributor = max(set(non_none), key=non_none.count) if non_none else 'none'
        intensities = [f.get('intensity_category', 'Low') for f in self.segment_frame_features if isinstance(f, dict)]
        intensity_order = ['Low', 'Medium', 'High']
        if intensities:
            indices = [intensity_order.index(i) for i in intensities if i in intensity_order]
            highest_intensity = intensity_order[max(indices)] if indices else 'Low'
            self.intensity_category = highest_intensity


    def _assign_audio_segment_features(self, master_audio_data: Dict):
        """Finds the audio segment corresponding to the midpoint of this clip and copies its features."""
        if not master_audio_data: return # No audio data available
        mid_time = self.start_time + self.duration / 2.0
        audio_segments = master_audio_data.get('segment_features', [])
        matched_segment = None

        if not audio_segments: # Handle case with no segments found
            logger.warning("No audio segments available in master_audio_data to assign features.")
            self.audio_segment_data = {}
            return

        for seg in audio_segments:
            # Check if midpoint falls within the segment [start, end)
            if seg['start'] <= mid_time < seg['end']:
                matched_segment = seg
                break

        # Handle edge case: If midpoint is exactly at or beyond the end of the last segment
        if matched_segment is None and mid_time >= audio_segments[-1]['end'] - 1e-6:
            matched_segment = audio_segments[-1]

        if matched_segment:
            self.audio_segment_data = matched_segment.copy() # Store a copy
        else:
            # If no match found (e.g., gap in segments, unlikely with proper segmentation)
            logger.warning(f"Could not find matching audio segment for clip at midpoint {mid_time:.2f}s")
            self.audio_segment_data = {} # Assign empty dict

    # --- SyncNet Scoring (PWRC) ---
    def _calculate_and_set_latent_sync(self, syncnet_model: SyncNet, full_mel_spectrogram: np.ndarray, mel_times: np.ndarray):
        """Calculates lip-sync score using SyncNet if enabled and model available."""
        # --- Pre-checks ---
        if not self.analysis_config.use_latent_sync: self.latent_sync_score = None; return
        if syncnet_model is None: logger.warning("SyncNet model not loaded, cannot calculate score."); self.latent_sync_score = None; return
        if not SKIMAGE_AVAILABLE: logger.warning("scikit-image not available, cannot process mouth crops for SyncNet."); self.latent_sync_score = None; return
        if full_mel_spectrogram is None or mel_times is None: logger.warning(f"Cannot calculate latent sync for clip {self.start_frame}: Missing Mel data."); self.latent_sync_score = None; return
        if self.num_frames < 5: logger.debug(f"Clip {self.start_frame} too short ({self.num_frames} frames) for SyncNet."); self.latent_sync_score = 0.0; return

        start_time = time.time()
        syncnet_model_device = next(syncnet_model.parameters()).device # Get device model is on

        # 1. Get Mouth Crops for this segment
        # Extract mouth crops directly from frame features if stored
        mouth_crops = []
        original_frame_indices = [] # Keep track of which original frame the crop came from
        for i, f in enumerate(self.segment_frame_features):
            if isinstance(f, dict):
                 crop = f.get('mouth_crop')
                 if crop is not None and isinstance(crop, np.ndarray) and crop.shape == (112, 112, 3): # Basic validation
                     mouth_crops.append(crop)
                     original_frame_indices.append(self.start_frame + i) # Store absolute frame index

        if len(mouth_crops) < 5: # Need at least 5 frames for SyncNet standard input
            logger.debug(f"Insufficient valid mouth crops ({len(mouth_crops)} found) for SyncNet scoring in clip {self.start_frame}.")
            self.latent_sync_score = 0.0; return

        # 2. Subsample Frames if necessary (Limit frames processed for performance)
        num_original_crops = len(mouth_crops)
        if num_original_crops > LATENTSYNC_MAX_FRAMES:
            # Select frames evenly spaced across the segment
            subsample_indices = np.linspace(0, num_original_crops - 1, LATENTSYNC_MAX_FRAMES, dtype=int)
            mouth_crops = [mouth_crops[i] for i in subsample_indices]
            original_frame_indices = [original_frame_indices[i] for i in subsample_indices]
            logger.debug(f"SyncNet: Limited frames from {num_original_crops} to {len(mouth_crops)} for performance.")

        # 3. Get Corresponding Mel Spectrogram Slice Indices
        # Map original frame indices to timestamps (use midpoint time of frame)
        crop_timestamps = [(idx + 0.5) / self.fps for idx in original_frame_indices]

        # Find corresponding Mel indices (SyncNet expects audio centered around video frame)
        mel_spec_indices = []
        if len(mel_times) < 2: # Need at least two points to estimate hop
            logger.warning(f"Cannot determine Mel hop time for SyncNet in clip {self.start_frame}.")
            self.latent_sync_score = 0.0; return
        mel_hop_sec = mel_times[1] - mel_times[0] # Estimate hop duration
        # SyncNet typically uses 0.2s audio window (check model specifics)
        mel_window_size_frames = int(round(0.2 / mel_hop_sec)) if mel_hop_sec > 0 else 20

        for ts in crop_timestamps:
            # Find the Mel frame index closest to the video frame timestamp
            center_mel_idx = np.argmin(np.abs(mel_times - ts))
            mel_spec_indices.append(center_mel_idx)

        # 4. Prepare Batches for SyncNet
        # SyncNet expects batches of 5 video frames and corresponding audio chunks
        video_batches = []; audio_batches = []
        num_windows = len(mouth_crops) - 4 # Number of 5-frame windows we can create
        if num_windows <= 0:
             logger.debug(f"Not enough crops ({len(mouth_crops)}) to form a 5-frame window for SyncNet in clip {self.start_frame}.")
             self.latent_sync_score = 0.0; return

        for i in range(num_windows):
            # Video chunk (5 frames)
            video_chunk_bgr = mouth_crops[i:i+5] # List of (H, W, C) BGR uint8 numpy arrays
            # Preprocess video: Convert to RGB float, transpose to (C, H, W), stack, normalize? (Check SyncNet docs for norm)
            video_chunk_processed = []
            for frame in video_chunk_bgr:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_float = frame_rgb.astype(np.float32) / 255.0 # Normalize to [0, 1]
                frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1) # (C, H, W)
                # Apply normalization if required by the specific SyncNet implementation (e.g., ImageNet stats)
                # frame_tensor = torchvision.transforms.functional.normalize(frame_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                video_chunk_processed.append(frame_tensor)
            # Stack along time dimension: (C, T=5, H, W)
            video_batch_item = torch.stack(video_chunk_processed, dim=1)
            video_batches.append(video_batch_item)

            # Audio chunk (corresponding Mel slice centered around middle video frame of the 5)
            center_video_frame_index_in_window = i + 2 # Index 2 is the middle of 0,1,2,3,4
            center_mel_idx = mel_spec_indices[center_video_frame_index_in_window] # Get pre-calculated Mel index
            # Calculate start/end Mel indices for the window
            start_mel = max(0, center_mel_idx - mel_window_size_frames // 2)
            end_mel = min(full_mel_spectrogram.shape[1], start_mel + mel_window_size_frames)
            # Adjust start if end hit boundary to maintain window size
            start_mel = max(0, end_mel - mel_window_size_frames)

            # Extract Mel slice and ensure it has the correct length by padding/truncating
            audio_chunk_mel = full_mel_spectrogram[:, start_mel:end_mel]
            current_len = audio_chunk_mel.shape[1]
            if current_len < mel_window_size_frames:
                padding_needed = mel_window_size_frames - current_len
                # Pad with minimum value (or zeros) - common for log-Mel
                padding = np.full((full_mel_spectrogram.shape[0], padding_needed), np.min(full_mel_spectrogram), dtype=np.float32)
                audio_chunk_mel = np.concatenate((audio_chunk_mel, padding), axis=1)
            elif current_len > mel_window_size_frames:
                 audio_chunk_mel = audio_chunk_mel[:, :mel_window_size_frames] # Truncate

            # Convert to tensor, add channel dim: (1, n_mels, time_steps)
            audio_batch_item = torch.from_numpy(audio_chunk_mel).unsqueeze(0)
            audio_batches.append(audio_batch_item)

        if not video_batches or not audio_batches:
             logger.debug(f"No valid SyncNet batches created for clip {self.start_frame}.")
             self.latent_sync_score = 0.0; return

        # 5. Run SyncNet Inference in Batches
        all_scores = []
        batch_size = self.analysis_config.syncnet_batch_size
        try:
            with torch.no_grad(): # Ensure no gradients are computed
                for i in range(0, len(video_batches), batch_size):
                    # Prepare batch tensors on the correct device
                    video_batch = torch.stack(video_batches[i:i+batch_size]).to(syncnet_model_device) # (N, C, T, H, W)
                    audio_batch = torch.stack(audio_batches[i:i+batch_size]).to(syncnet_model_device) # (N, 1, Mels, Time)

                    # Run model
                    audio_embed, video_embed = syncnet_model(audio_batch, video_batch)
                    # Calculate cosine similarity between audio and video embeddings for the batch
                    cosine_sim = F.cosine_similarity(audio_embed, video_embed, dim=-1) # Shape: (N,)
                    # Store scores (move back to CPU if needed, then convert to list)
                    all_scores.extend(cosine_sim.cpu().numpy().tolist())
        except Exception as sync_err:
            logger.error(f"SyncNet inference failed for clip {self.start_frame}: {sync_err}", exc_info=True)
            self.latent_sync_score = 0.0; return # Set score to 0 on error

        # 6. Calculate Final Score
        if all_scores:
            # Average the cosine similarity scores (range -1 to 1)
            avg_score = np.mean(all_scores)
            # Map score to [0, 1] as a confidence metric: (score + 1) / 2
            self.latent_sync_score = float(np.clip((avg_score + 1.0) / 2.0, 0.0, 1.0))
            logger.debug(f"SyncNet score for clip {self.start_frame}: {self.latent_sync_score:.3f} (from {len(all_scores)} windows) ({time.time()-start_time:.3f}s)")
        else:
            logger.warning(f"SyncNet inference yielded no scores for clip {self.start_frame}.")
            self.latent_sync_score = 0.0

    # --- V4 Physics Mode Helpers (kept for compatibility) ---
    def clip_audio_fit(self, audio_segment_data, analysis_config):
        """ V4 Physics Mode: Calculates fit between clip visuals and audio segment. """
        if not audio_segment_data: return 0.0
        cfg = analysis_config # Config used during analysis
        # V4 fit calculation: sigmoid( w_v*|v_k-b_i| + w_a*|a_j-e_i| + w_m*(1-mood_sim) )
        b_i = audio_segment_data.get('b_i', 0.0) # Normalized audio energy
        e_i = audio_segment_data.get('e_i', 0.0) # Normalized audio onset proxy
        m_i_aud = np.asarray(audio_segment_data.get('m_i', [0.0, 0.0])) # Audio mood vector
        m_i_vid = np.asarray(self.mood_vector) # Visual mood vector (V4 version)

        # Calculate differences (absolute)
        diff_v = abs(self.v_k - b_i) # v_k is normalized flow velocity
        diff_a = abs(self.a_j - e_i) # a_j is normalized flow accel

        # Mood similarity (Gaussian similarity from V4)
        sigma_m_sq = cfg.mood_similarity_variance**2 * 2
        mood_dist_sq = np.sum((m_i_vid - m_i_aud)**2)
        mood_sim = exp(-mood_dist_sq / (sigma_m_sq + 1e-9))
        diff_m = 1.0 - mood_sim # Difference is 1 - similarity

        # Weighted sum of differences
        fit_arg = (cfg.fit_weight_velocity * diff_v +
                   cfg.fit_weight_acceleration * diff_a +
                   cfg.fit_weight_mood * diff_m)

        # Probability is inverse sigmoid of the weighted difference (lower diff -> higher prob)
        probability = 1.0 - sigmoid(fit_arg, cfg.fit_sigmoid_steepness)
        return float(probability)

    def get_feature_vector(self, analysis_config):
        """ V4 Physics Mode: Returns feature vector [v_k, a_j, d_r] for continuity calc. """
        # Uses the normalized V4 features calculated in _aggregate_visual_features
        return [float(self.v_k), float(self.a_j), float(self.d_r)]

    def get_shot_type(self): # (Unchanged from v4.7.3)
        """Categorizes shot type based on face presence and size (Heuristic)."""
        if self.face_presence_ratio < 0.1: return 'wide/no_face'
        if self.avg_face_size < 0.15: return 'medium_wide'
        if self.avg_face_size < 0.35: return 'medium'
        return 'close_up'

    def __repr__(self):
        source_basename = os.path.basename(str(self.source_video_path)) if self.source_video_path else "N/A"
        sync_score_str = f"{self.latent_sync_score:.2f}" if self.latent_sync_score is not None else "N/A"
        embed_str = "Yes" if self.visual_embedding is not None else "No"
        return (f"ClipSegment({source_basename} @ {self.fps:.1f}fps | "
                f"Frames:[{self.start_frame}-{self.end_frame}] | Time:[{self.start_time:.2f}s-{self.end_time:.2f}s] | Dur:{self.duration:.2f}s)\n"
                f"  Visual: Flow:{self.avg_visual_flow:.1f}(T:{self.avg_visual_flow_trend:+.2f}) | PoseKin:{self.avg_pose_kinetic:.1f}(T:{self.avg_visual_pose_trend:+.2f}) | FaceSz:{self.avg_face_size:.2f} ({self.face_presence_ratio*100:.0f}%)\n"
                f"  Audio Seg: RMS:{self.audio_segment_data.get('rms_avg', 0):.3f} | Trend:{self.audio_segment_data.get('trend_long', 0):+.4f}\n"
                f"  Ensemble: SyncNet:{sync_score_str} | VisEmbed:{embed_str} | Shot:{self.get_shot_type()} | Intensity:{self.intensity_category}")


# ========================================================================
#         MAIN ANALYSIS CLASS (VideousMain - Ensemble v5.3 Ready)
# ========================================================================
class VideousMain:
    def __init__(self):
        # Utilities are now stateless or manage their own state/models
        self.BBZImageUtils = BBZImageUtils()
        self.BBZPoseUtils = BBZPoseUtils()
        # MiDaS model/transform/device loaded via helper function get_midas_model() if needed
        self.midas_model = None
        self.midas_transform = None
        self.midas_device = None


    def _ensure_midas_loaded(self, analysis_config: AnalysisConfig):
        """Loads MiDaS model if needed by config and not already loaded."""
        # Determine if MiDaS is needed (Physics MC V4 logic or base heuristic depth)
        needs_midas = (analysis_config.sequencing_mode == 'Physics Pareto MC') or \
                      (analysis_config.base_heuristic_weight > 0 and analysis_config.bh_depth_weight > 0)

        if needs_midas and self.midas_model is None:
            logger.info("MiDaS model required by configuration, attempting load...")
            # Use the specific helper function for MiDaS
            model, transform, device = get_midas_model()
            if model and transform and device:
                 self.midas_model = model
                 self.midas_transform = transform
                 self.midas_device = device
                 logger.info("MiDaS model and transform loaded successfully.")
            else:
                 logger.warning("MiDaS failed to load. Depth features will be disabled.")
                 # Optionally disable features that rely on it
                 # analysis_config.bh_depth_weight = 0.0 # Example
        elif needs_midas:
            logger.debug("MiDaS model already loaded.")
        elif not needs_midas:
             logger.debug("MiDaS model not required by current configuration.")


    # --- Heuristic Score Calculation (Kept for base heuristic component) ---
    def _determine_dominant_contributor(self, norm_features_weighted): # (Unchanged)
        if not norm_features_weighted: return "unknown"
        max_val = -float('inf'); dominant_key = "none"
        for key, value in norm_features_weighted.items():
            if value > max_val: max_val = value; dominant_key = key
        # Map internal keys to display names
        key_map = {'audio_energy': 'Audio', 'kinetic_proxy': 'Motion', 'jerk_proxy': 'Jerk',
                   'camera_motion': 'CamMove', 'face_size': 'FaceSize',
                   'percussive': 'Percuss', 'depth_variance': 'DepthVar'}
        return key_map.get(dominant_key, dominant_key) if max_val > 1e-4 else "none"

    def _categorize_intensity(self, score, thresholds=(0.3, 0.7)): # (Unchanged)
        """Categorizes score into Low/Medium/High based on thresholds."""
        if score < thresholds[0]: return "Low"
        if score < thresholds[1]: return "Medium"
        return "High"

    def calculate_base_heuristic_score(self, frame_features: Dict, analysis_config: AnalysisConfig) -> Tuple[float, str, str]:
        """Calculates the *base* heuristic score component (V4 logic)."""
        # Use weights specific to the base heuristic component from config
        weights = {
            'audio_energy': analysis_config.bh_audio_weight,
            'kinetic_proxy': analysis_config.bh_kinetic_weight,
            'jerk_proxy': analysis_config.bh_sharpness_weight,
            'camera_motion': analysis_config.bh_camera_motion_weight,
            'face_size': analysis_config.bh_face_size_weight,
            'percussive': analysis_config.bh_percussive_weight,
            'depth_variance': analysis_config.bh_depth_weight
        }
        # Use normalization parameters from config
        norm_params = {
            'rms': analysis_config.norm_max_rms + 1e-6,
            'kinetic': analysis_config.norm_max_pose_kinetic + 1e-6,
            'jerk': analysis_config.norm_max_jerk + 1e-6,
            'cam_motion': analysis_config.norm_max_visual_flow + 1e-6, # Use flow norm
            'face_size': analysis_config.norm_max_face_size + 1e-6,
            'percussive_ratio': 1.0 + 1e-6, # Percussive ratio assumed [0,1]
            'depth_variance': analysis_config.norm_max_depth_variance + 1e-6
        }

        f = frame_features
        pose_f = f.get('pose_features', {})

        # Safely get features from the frame dictionary
        audio_energy = f.get('audio_energy', 0.0)
        kinetic_proxy = pose_f.get('kinetic_energy_proxy', 0.0)
        jerk_proxy = pose_f.get('movement_jerk_proxy', 0.0)
        # Use flow_velocity as the camera_motion proxy
        camera_motion = f.get('flow_velocity', 0.0)
        face_size_ratio = pose_f.get('face_size_ratio', 0.0)
        percussive_ratio = f.get('percussive_ratio', 0.0) # Needs to be calculated if used
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
        # Filter contributions with zero weight
        weighted_contrib = {k: v for k, v in contrib.items() if abs(weights.get(k, 0)) > 1e-6}

        # Calculate final score, dominant contributor, and intensity
        score = sum(weighted_contrib.values())
        final_score = np.clip(score, 0.0, 1.0)
        dominant = self._determine_dominant_contributor(weighted_contrib)
        intensity = self._categorize_intensity(final_score)
        return float(final_score), dominant, intensity

    def apply_beat_boost(self, frame_features_list: List[Dict], audio_data: Dict, video_fps: float, analysis_config: AnalysisConfig):
        """Applies boost to BASE heuristic score near beat times."""
        num_frames = len(frame_features_list)
        if num_frames == 0 or not audio_data or video_fps <= 0: return

        # Scale boost relative to base heuristic weight (makes it less impactful if base heuristic is down-weighted)
        # Example: If base weight is 0.1, boost is 0.05. If base weight is 0.5, boost is 0.25.
        # This assumes the 'boosted_score' is primarily the V4 score used for legacy segmentation.
        beat_boost_value = analysis_config.base_heuristic_weight * 0.5
        boost_radius_sec = analysis_config.rhythm_beat_boost_radius_sec
        boost_radius_frames = max(0, int(boost_radius_sec * video_fps))
        beat_times = audio_data.get('beat_times', [])

        if not beat_times or beat_boost_value <= 0: return # No boost if no beats or zero boost value

        # Find all frame indices within the boost radius of any beat
        boost_frame_indices = set()
        for t in beat_times:
            beat_frame_center = int(round(t * video_fps))
            for r in range(-boost_radius_frames, boost_radius_frames + 1):
                idx = beat_frame_center + r
                if 0 <= idx < num_frames: boost_frame_indices.add(idx)

        # Apply boost to the 'boosted_score' field (legacy V4 score)
        for i, features in enumerate(frame_features_list):
            if not isinstance(features, dict): continue
            is_beat = i in boost_frame_indices
            features['is_beat_frame'] = is_beat # Mark frame for potential use elsewhere
            boost = beat_boost_value if is_beat else 0.0
            raw_score = features.get('raw_score', 0.0) # Get the raw V4 score
            # Add boost, ensuring score doesn't exceed 1.0
            features['boosted_score'] = min(raw_score + boost, 1.0)

    def get_feature_at_time(self, times_array, values_array, target_time): # (Unchanged)
        """Linearly interpolates a feature value at a specific time."""
        if times_array is None or values_array is None or len(times_array) == 0 or len(times_array) != len(values_array):
            logger.debug(f"Interpolation skipped: Invalid input arrays for time {target_time:.3f}.")
            return 0.0
        if len(times_array) == 1: # Handle single data point
             return float(values_array[0])
        try:
            # Ensure times are sorted for np.interp
            if not np.all(np.diff(times_array) >= 0):
                sort_indices = np.argsort(times_array)
                times_array = times_array[sort_indices]
                values_array = values_array[sort_indices]

            # Interpolate, using boundary values for times outside the range
            interpolated_value = np.interp(target_time, times_array, values_array, left=values_array[0], right=values_array[-1])
            # Ensure finite result
            return float(interpolated_value) if np.isfinite(interpolated_value) else 0.0
        except Exception as e:
            logger.error(f"Interpolation error at time={target_time:.3f}: {e}")
            return 0.0 # Fallback on error


    # ============================================================ #
    #         analyzeVideo Method (Refactored for Ensemble v5.3)   #
    # ============================================================ #
    def analyzeVideo(self, videoPath: str, analysis_config: AnalysisConfig,
                     master_audio_data: Dict) -> Tuple[Optional[List[Dict]], Optional[List[ClipSegment]]]:
        """ Analyzes video frames for Ensemble features using MoviePy reader. """
        logger.info(f"--- Analyzing Video (Ensemble v5.3): {os.path.basename(videoPath)} ---")
        if ENABLE_PROFILING: profiler_start_time = time.time()

        # --- Initialization ---
        TARGET_HEIGHT = analysis_config.resolution_height
        TARGET_WIDTH = analysis_config.resolution_width
        all_frame_features: List[Dict] = []; potential_clips: List[ClipSegment] = []
        clip: Optional[VideoFileClip] = None
        pose_detector = None; face_detector_util = None; pose_context = None # For context manager
        prev_gray: Optional[np.ndarray] = None; prev_flow: Optional[np.ndarray] = None
        pose_results_buffer = deque([None, None, None], maxlen=3) # Use deque for efficient rolling buffer
        fps: float = 30.0 # Default FPS

        # --- Load Dependencies Based on Config ---
        self._ensure_midas_loaded(analysis_config) # Load MiDaS if needed

        # --- SyncNet Data Setup (if enabled) ---
        syncnet_model = None; syncnet_model_device = None
        full_mel_spectrogram = None; mel_times = None
        if analysis_config.use_latent_sync:
            full_mel_spectrogram = master_audio_data.get('raw_features', {}).get('mel_spectrogram')
            mel_times = master_audio_data.get('raw_features', {}).get('mel_times')
            if full_mel_spectrogram is None or mel_times is None:
                 logger.error(f"SyncNet enabled but Mel data missing for {os.path.basename(videoPath)}. Disabling SyncNet scoring for this video.")
                 # Create a temporary config with SyncNet disabled for this video
                 analysis_config = dataclass_replace(analysis_config, use_latent_sync=False)
            else:
                 # Lazy Load SyncNet Model via cached helper
                 model_cache_key = f"syncnet_{analysis_config.syncnet_repo_id}_{analysis_config.syncnet_filename}"
                 syncnet_model, syncnet_model_device = get_pytorch_model(
                     model_cache_key,
                     load_syncnet_model_from_hf_func, config=analysis_config # Pass actual loading function
                 )
                 if syncnet_model is None:
                      logger.error(f"Failed to load SyncNet model for {os.path.basename(videoPath)}. Disabling scoring.")
                      analysis_config = dataclass_replace(analysis_config, use_latent_sync=False)

        # --- Load Video Properties ---
        try:
            logger.debug(f"Loading video with MoviePy: {videoPath}")
            clip = VideoFileClip(videoPath, audio=False)
            fps = clip.reader.fps if hasattr(clip, 'reader') and clip.reader and clip.reader.fps > 0 else (clip.fps if clip.fps and clip.fps > 0 else 30.0)
            if fps <= 0 or not np.isfinite(fps):
                 fps = 30.0; logger.warning(f"Invalid FPS detected ({fps}), using default 30.")
            frame_time_diff = 1.0 / fps
            total_frames = int(clip.duration * fps) if clip.duration and clip.duration > 0 else 0
            logger.info(f"Video Properties: FPS={fps:.2f}, Frames~{total_frames}, Dur={clip.duration:.2f}s")
            if total_frames <= 0 or clip.duration <=0:
                raise ValueError(f"Video has zero duration or frames ({videoPath}).")

            # === Step 2: Setup MediaPipe & Audio Refs ===
            face_detector_util = BBZFaceUtils(static_mode=False, max_faces=1, min_detect_conf=analysis_config.min_face_confidence)
            if face_detector_util.face_mesh is None: logger.warning("FaceMesh failed to initialize. Face features disabled.")

            # Check if Pose is needed by any enabled feature
            pose_needed = analysis_config.use_latent_sync or \
                          (analysis_config.pwrc_weight > 0 and analysis_config.pwrc_pose_energy_weight > 0) or \
                          (analysis_config.hefm_weight > 0) or \
                          (analysis_config.base_heuristic_weight > 0 and (analysis_config.bh_kinetic_weight > 0 or analysis_config.bh_sharpness_weight > 0)) or \
                          analysis_config.sequencing_mode == "Physics Pareto MC"

            if pose_needed:
                logger.debug(f"Initializing Pose (Complexity: {analysis_config.model_complexity})...")
                try:
                    pose_context = mp_pose.Pose(static_image_mode=False, model_complexity=analysis_config.model_complexity,
                                                min_detection_confidence=analysis_config.min_pose_confidence, min_tracking_confidence=0.5)
                    pose_detector = pose_context.__enter__() # Enter context manager
                except Exception as pose_init_err:
                     logger.error(f"Failed to initialize MediaPipe Pose: {pose_init_err}. Pose features disabled.")
                     pose_detector = None; pose_context = None # Ensure cleanup doesn't fail

            # Get references to audio features (should be numpy arrays if loaded correctly)
            audio_raw_features = master_audio_data.get('raw_features', {})
            audio_rms_times = audio_raw_features.get('rms_times')
            audio_rms_energy = audio_raw_features.get('rms_energy')
            # V4 percussive ratio - ensure fallback if not present
            audio_perc_times = audio_raw_features.get('perc_times', audio_rms_times) # Fallback to RMS times
            audio_perc_ratio = audio_raw_features.get('percussive_ratio', np.zeros_like(audio_perc_times) if audio_perc_times is not None else []) # Fallback to zeros
            segment_boundaries = master_audio_data.get('segment_boundaries', [0, master_audio_data.get('duration', float('inf'))])
            if not segment_boundaries or len(segment_boundaries) < 2:
                segment_boundaries = [0, master_audio_data.get('duration', float('inf'))]

            # Ensure audio feature arrays are numpy arrays
            if audio_rms_times is not None and not isinstance(audio_rms_times, np.ndarray): audio_rms_times = np.asarray(audio_rms_times)
            if audio_rms_energy is not None and not isinstance(audio_rms_energy, np.ndarray): audio_rms_energy = np.asarray(audio_rms_energy)
            if audio_perc_times is not None and not isinstance(audio_perc_times, np.ndarray): audio_perc_times = np.asarray(audio_perc_times)
            if audio_perc_ratio is not None and not isinstance(audio_perc_ratio, np.ndarray): audio_perc_ratio = np.asarray(audio_perc_ratio)


            # === Step 3: Feature Extraction Loop ===
            logger.info("Processing frames & generating features (Ensemble v5.3)...")
            # Use MoviePy iterator
            frame_iterator = clip.iter_frames(fps=fps, dtype="uint8", logger=None)
            pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"Analyzing {os.path.basename(videoPath)}", unit="frame", dynamic_ncols=True, leave=False, disable=tqdm is None)

            # Store timeseries for trend calculation after loop
            frame_timestamps = []; frame_flow_velocities = []; frame_pose_kinetics = []

            for frame_idx, frame_rgb in enumerate(frame_iterator):
                if frame_rgb is None:
                    logger.warning(f"Received None frame at index {frame_idx} from MoviePy. Stopping analysis for this video.")
                    break
                timestamp = frame_idx / fps

                # --- Base Image Processing ---
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                image_resized_bgr = self.BBZImageUtils.resizeTARGET(frame_bgr, TARGET_HEIGHT, TARGET_WIDTH)
                if image_resized_bgr is None or image_resized_bgr.size == 0:
                    logger.warning(f"Frame {frame_idx}: Resize failed or resulted in empty image. Skipping frame.")
                    if pbar: pbar.update(1)
                    continue # Skip processing for this frame

                current_features = {'frame_index': frame_idx, 'timestamp': timestamp}
                pose_features_dict = {} # Store pose-related sub-features

                current_gray = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2GRAY)

                # --- HEFM/Physics: Visual Flow, Accel, Entropy ---
                flow_velocity, current_flow_field = calculate_flow_velocity(prev_gray, current_gray)
                flow_acceleration = calculate_flow_acceleration(prev_flow, current_flow_field, frame_time_diff)
                current_features['flow_velocity'] = flow_velocity
                current_features['flow_acceleration'] = flow_acceleration
                current_features['camera_motion'] = flow_velocity # Alias for base heuristic V4 compatibility
                current_features['histogram_entropy'] = calculate_histogram_entropy(current_gray)
                # Store for trend calculation later
                frame_timestamps.append(timestamp); frame_flow_velocities.append(flow_velocity)

                # --- HEFM/Physics: Depth Variance (MiDaS) ---
                depth_variance = 0.0
                if self.midas_model and self.midas_transform and self.midas_device:
                    try:
                        with torch.no_grad():
                            # Convert BGR resized to RGB for MiDaS
                            image_resized_rgb_midas = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB)
                            # Apply MiDaS transform and move to device
                            input_batch = self.midas_transform(image_resized_rgb_midas).to(self.midas_device)
                            # Run MiDaS model
                            prediction = self.midas_model(input_batch)
                            # Interpolate prediction to input size
                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1), size=image_resized_rgb_midas.shape[:2],
                                mode="bicubic", align_corners=False
                            ).squeeze()
                            # Get depth map and calculate normalized variance
                            depth_map = prediction.cpu().numpy()
                            depth_min = depth_map.min(); depth_max = depth_map.max()
                            if depth_max > depth_min + 1e-6: # Avoid division by zero
                                norm_depth = (depth_map - depth_min) / (depth_max - depth_min)
                                depth_variance = float(np.var(norm_depth))
                    except Exception as midas_e:
                        # Log only periodically or at debug level to avoid flooding
                        if frame_idx % 100 == 0: logger.debug(f"MiDaS error on frame {frame_idx}: {midas_e}")
                current_features['depth_variance'] = depth_variance

                # --- Face & Pose Processing ---
                face_results = face_detector_util.process_frame(image_resized_bgr) if face_detector_util else None
                current_pose_results = None
                if pose_detector:
                    try:
                        # MediaPipe Pose needs RGB
                        image_resized_rgb_pose = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB)
                        image_resized_rgb_pose.flags.writeable = False # Performance hint
                        current_pose_results = pose_detector.process(image_resized_rgb_pose)
                        image_resized_rgb_pose.flags.writeable = True
                    except Exception as pose_err:
                         if frame_idx % 100 == 0: logger.debug(f"Pose processing error on frame {frame_idx}: {pose_err}")

                # Update pose buffer (use deque for efficiency)
                pose_results_buffer.append(current_pose_results)
                lm_t2, lm_t1, lm_t = pose_results_buffer # Unpack the buffer (oldest to newest)

                # --- PWRC: Extract Mouth Crop ---
                mouth_crop_np = None
                if analysis_config.use_latent_sync and face_results and face_results.multi_face_landmarks:
                    # Ensure we use the first face if multiple detected (shouldn't happen with max_faces=1)
                    mouth_crop_np = extract_mouth_crop(image_resized_bgr, face_results.multi_face_landmarks[0])
                current_features['mouth_crop'] = mouth_crop_np # Store numpy array (or None)

                # --- HEFM/PWRC: Pose Kinetic Energy & Jerk ---
                kinetic = calculate_kinetic_energy_proxy(lm_t1, lm_t, frame_time_diff)
                jerk = calculate_movement_jerk_proxy(lm_t2, lm_t1, lm_t, frame_time_diff)
                pose_features_dict['kinetic_energy_proxy'] = kinetic
                pose_features_dict['movement_jerk_proxy'] = jerk
                frame_pose_kinetics.append(kinetic) # Store for trend calculation

                # --- Base Heuristic: Face Size & Mouth Open ---
                is_mouth_open, face_size_ratio, face_center_x = face_detector_util.get_heuristic_face_features(face_results, TARGET_HEIGHT, TARGET_WIDTH, analysis_config.mouth_open_threshold) if face_detector_util else (False, 0.0, 0.5)
                pose_features_dict['is_mouth_open'] = is_mouth_open
                pose_features_dict['face_size_ratio'] = face_size_ratio
                pose_features_dict['face_center_x'] = face_center_x
                current_features['pose_features'] = pose_features_dict

                # --- Align with Audio Features ---
                mid_frame_time = timestamp + (frame_time_diff / 2.0) # Time at middle of frame exposure
                # Interpolate audio features at this time
                current_features['audio_energy'] = self.get_feature_at_time(audio_rms_times, audio_rms_energy, mid_frame_time)
                current_features['percussive_ratio'] = self.get_feature_at_time(audio_perc_times, audio_perc_ratio, mid_frame_time)

                # Determine Musical Section Index based on segment boundaries
                section_idx = -1
                for i in range(len(segment_boundaries) - 1):
                    if segment_boundaries[i] <= mid_frame_time < segment_boundaries[i+1]:
                        section_idx = i
                        break
                # Handle case where time is exactly end boundary or slightly past last segment
                if section_idx == -1 and mid_frame_time >= segment_boundaries[-1] - 1e-6:
                     section_idx = len(segment_boundaries) - 2 # Assign to last segment index

                current_features['musical_section_index'] = section_idx

                # --- Calculate BASE heuristic score (V4 logic, used as one component in ensemble) ---
                raw_score, dominant, intensity = self.calculate_base_heuristic_score(current_features, analysis_config)
                current_features['raw_score'] = raw_score
                current_features['dominant_contributor'] = dominant
                current_features['intensity_category'] = intensity
                # Initialize boosted_score (V4 legacy score) with raw score, will be modified by beat boost
                current_features['boosted_score'] = raw_score
                current_features['is_beat_frame'] = False # Will be set later by apply_beat_boost

                # --- Store frame features ---
                all_frame_features.append(current_features)

                # --- Update previous states for next iteration ---
                prev_gray = current_gray.copy()
                prev_flow = current_flow_field.copy() if current_flow_field is not None else None

                if pbar: pbar.update(1)
            # --- End Frame Loop ---

            if pbar: pbar.close()

            # === Step 3.5: Calculate Visual Trends (Post-Loop) ===
            logger.debug("Calculating visual trends from frame features...")
            if len(frame_timestamps) > 1:
                 # Calculate gradients (trends) using numpy.gradient
                 # Need to ensure timestamps are monotonically increasing (should be from loop)
                 ts_diff = np.diff(frame_timestamps)
                 if np.any(ts_diff <= 0): # Check for non-increasing timestamps
                      logger.warning("Timestamps not monotonic, cannot reliably calculate trends.")
                      visual_flow_trend = np.zeros(len(frame_timestamps))
                      visual_pose_trend = np.zeros(len(frame_timestamps))
                 else:
                     visual_flow_trend = np.gradient(frame_flow_velocities, frame_timestamps)
                     visual_pose_trend = np.gradient(frame_pose_kinetics, frame_timestamps)

                 # Add trends back to the frame features dictionary
                 for i in range(len(all_frame_features)):
                      # Check index bounds before assigning
                      if i < len(visual_flow_trend): all_frame_features[i]['visual_flow_trend'] = float(visual_flow_trend[i])
                      else: all_frame_features[i]['visual_flow_trend'] = 0.0 # Assign 0 if gradient calculation failed for some frames
                      if i < len(visual_pose_trend): all_frame_features[i]['visual_pose_trend'] = float(visual_pose_trend[i])
                      else: all_frame_features[i]['visual_pose_trend'] = 0.0
            else: # Assign zero trend if only zero or one frame
                 for i in range(len(all_frame_features)):
                     all_frame_features[i]['visual_flow_trend'] = 0.0
                     all_frame_features[i]['visual_pose_trend'] = 0.0
            logger.debug("Visual trend calculation complete.")

        except ValueError as ve:
            logger.error(f"ValueError during analysis setup or loop for {videoPath}: {ve}")
            return None, None
        except Exception as e:
            logger.error(f"Error during video analysis loop for {videoPath}: {e}", exc_info=True)
            return None, None
        finally:
            # === Step 4: Cleanup Resources ===
            logger.debug(f"Cleaning up resources for {os.path.basename(videoPath)}...")
            if pose_context:
                try: pose_context.__exit__(None, None, None) # Manually exit context manager
                except Exception as pose_close_err: logger.error(f"Error closing Pose context: {pose_close_err}")
            if face_detector_util: face_detector_util.close()
            if clip:
                try: clip.close()
                except Exception as clip_close_err: logger.error(f"Error closing MoviePy clip: {clip_close_err}")
            logger.debug(f"Resource cleanup for {os.path.basename(videoPath)} finished.")

        # === Step 5: Post-processing & Clip Identification ===
        if not all_frame_features:
            logger.error(f"No features extracted for {videoPath}. Cannot create clips.")
            return None, None

        logger.debug("Applying V4 beat boost to base heuristic score...")
        self.apply_beat_boost(all_frame_features, master_audio_data, fps, analysis_config)

        potential_clips: List[ClipSegment] = []
        actual_total_frames = len(all_frame_features)

        # --- Create Potential Segments using Fixed/Overlapping Chunks ---
        # (DynamicSegmentIdentifier based on V4 heuristic score is less relevant now)
        logger.info("Creating potential segments using fixed/overlapping chunks...")
        min_clip_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES, int(analysis_config.min_potential_clip_duration_sec * fps))
        # Allow potential clips to be slightly longer than max sequence clip for flexibility
        max_clip_frames = max(min_clip_frames + 1, int(analysis_config.max_sequence_clip_duration * fps * 1.2))
        step_frames = max(1, int(0.5 * fps)) # Step ~0.5 seconds for overlap

        if actual_total_frames < min_clip_frames:
            logger.warning(f"Video too short ({actual_total_frames} frames) for min potential clip length ({min_clip_frames} frames). No clips generated.")
        else:
            for start_f in range(0, actual_total_frames - min_clip_frames + 1, step_frames):
                 end_f = min(start_f + max_clip_frames, actual_total_frames) # Clamp end frame
                 # Ensure the segment meets the minimum length *after* clamping
                 if (end_f - start_f) >= min_clip_frames:
                     try:
                         # Create ClipSegment instance
                         clip_seg = ClipSegment(videoPath, start_f, end_f, fps, all_frame_features, analysis_config, master_audio_data)
                         # Calculate SyncNet score here if enabled (needs model and mel data)
                         if analysis_config.use_latent_sync and syncnet_model and full_mel_spectrogram is not None and mel_times is not None:
                              clip_seg._calculate_and_set_latent_sync(syncnet_model, full_mel_spectrogram, mel_times)
                         potential_clips.append(clip_seg)
                     except Exception as clip_err:
                         logger.warning(f"Failed to create ClipSegment for frames {start_f}-{end_f}: {clip_err}", exc_info=False)

        end_time = time.time()
        analysis_duration = end_time - profiler_start_time if ENABLE_PROFILING else 0
        logger.info(f"--- Analysis & Clip Creation complete for {os.path.basename(videoPath)} ({analysis_duration:.2f}s) ---")
        logger.info(f"Created {len(potential_clips)} potential clips.")
        if ENABLE_PROFILING: logger.debug(f"PROFILING: Video Analysis ({os.path.basename(videoPath)}) took {analysis_duration:.3f}s")

        # Return frame features (for saving if needed) and potential clips list
        return all_frame_features, potential_clips

        # --- 5. Style/Pacing/Variety (SAAPV) Component ---
        saapv_penalty = 0.0
        # Predictability Penalty (based on variance in recent history)
        if cfg.use_saapv_adaptation: # Only apply if adaptation is on
             predictability_penalty_raw = self._calculate_predictability_penalty()
             saapv_penalty += predictability_penalty_raw * cfg.saapv_predictability_weight

        # Apply V4 variety penalties under SAAPV umbrella
        if self.last_clip_used:
             # Same Source Penalty
             if str(clip.source_video_path) == str(self.last_clip_used.source_video_path):
                  saapv_penalty += cfg.saapv_variety_penalty_source
             # Same Shot Type Penalty (ignore if wide/no face)
             shot_type = clip.get_shot_type(); last_shot_type = self.last_clip_used.get_shot_type()
             if shot_type == last_shot_type and shot_type != 'wide/no_face':
                  saapv_penalty += cfg.saapv_variety_penalty_shot
             # Same Intensity Category Penalty
             if clip.intensity_category == self.last_clip_used.intensity_category:
                  saapv_penalty += cfg.saapv_variety_penalty_intensity

        total_score -= saapv_penalty * cfg.saapv_weight # Subtract weighted penalties
        score_components['SAAPV_Penalty'] = -saapv_penalty * cfg.saapv_weight

        # --- 6. Micro-Rhythm (MRISE) Component ---
        mrise_bonus = 0.0
        if cfg.use_mrise_sync and self.micro_beat_times:
             # Calculate tolerance based on render FPS
             frame_dur = 1.0 / cfg.render_fps if cfg.render_fps > 0 else 1/30.0
             tolerance = frame_dur * cfg.mrise_sync_tolerance_factor
             # Check proximity of clip START time to nearest micro-beat
             start_time = clip.start_time
             nearest_micro_beat_diff = min([abs(start_time - mb) for mb in self.micro_beat_times])
             # Exponential decay bonus: Max bonus if diff=0, decays sharply
             mrise_bonus = exp(- (nearest_micro_beat_diff**2) / (2 * tolerance**2)) * cfg.mrise_sync_weight

        total_score += mrise_bonus * cfg.mrise_weight
        score_components['MRISE'] = mrise_bonus * cfg.mrise_weight

        # --- 7. Rhythm (Beat Sync) Component ---
        # Bonus for starting near a main beat (less strict than MRISE)
        beat_sync_bonus = 0.0
        if cfg.rhythm_beat_sync_weight > 0 and self.beat_times:
             nearest_beat_diff = min([abs(clip.start_time - bt) for bt in self.beat_times])
             if nearest_beat_diff <= cfg.rhythm_beat_boost_radius_sec:
                  beat_sync_bonus = cfg.rhythm_beat_sync_weight

        total_score += beat_sync_bonus
        score_components['BeatSync'] = beat_sync_bonus

        # --- Log Detailed Score Breakdown (Optional Debug) ---
        # if random.random() < 0.01: # Log occasionally
        #      log_msg = f"Clip {clip.start_frame} Score={total_score:.4f} | Breakdown: "
        #      log_msg += " ".join([f"{k}:{v:.3f}" for k, v in score_components.items()])
        #      logger.debug(log_msg)

        return total_score


    def _calculate_predictability_penalty(self) -> float:
        """Calculates penalty based on recent edit history variance (SAAPV). Higher variance -> Lower penalty."""
        history_len = len(self.sequence_history)
        # Need enough history to calculate variance meaningfully
        if history_len < max(3, self.analysis_config.saapv_history_length // 2):
            return 0.0

        recent_edits = get_recent_history(self.sequence_history, history_len) # Get current history

        # --- Duration Variance Penalty ---
        durations = [e['duration'] for e in recent_edits]
        duration_penalty = 0.0
        if len(durations) > 1:
            duration_std_dev = np.std(durations)
            avg_dur = np.mean(durations)
            # Normalize std dev relative to average duration
            norm_std_dev = duration_std_dev / (avg_dur + 1e-6)
            # Inverse relationship: higher variance -> lower penalty (less predictable)
            # Use sigmoid: penalty = sigmoid(- (norm_std_dev - threshold) * scale )
            # Example: Penalty increases if std dev is LOW (below 0.3 * avg_dur)
            duration_penalty = sigmoid(-(norm_std_dev - 0.3) * 5.0) # Adjust threshold (0.3) and scale (5.0)

        # --- Shot Type Repetition Penalty ---
        shot_types = [e['shot_type'] for e in recent_edits]
        shot_type_penalty = 0.0
        if len(shot_types) > 1:
             num_unique_shots = len(set(shot_types))
             max_possible_unique = len(shot_types)
             # Repetition ratio: 1 - (unique / total)
             repetition_ratio = 1.0 - (num_unique_shots / max_possible_unique)
             shot_type_penalty = repetition_ratio # Higher ratio = more repetition = higher penalty

        # --- Source Repetition Penalty ---
        source_paths = [e['source_path'] for e in recent_edits]
        source_penalty = 0.0
        if len(source_paths) > 1:
             num_unique_sources = len(set(source_paths))
             max_possible_sources = len(source_paths)
             source_repetition_ratio = 1.0 - (num_unique_sources / max_possible_sources)
             source_penalty = source_repetition_ratio

        # Combine penalties (e.g., weighted average or max)
        # Example: Average duration penalty and the max of shot/source repetition
        predictability_penalty = (duration_penalty + max(shot_type_penalty, source_penalty)) / 2.0

        # logger.debug(f"SAAPV Predictability Penalty: {predictability_penalty:.3f} (DurP: {duration_penalty:.2f}, ShotRepP: {shot_type_penalty:.2f}, SrcRepP: {source_penalty:.2f})")
        return np.clip(predictability_penalty, 0.0, 1.0) # Ensure penalty is [0, 1]


# ========================================================================
#          SEQUENCE BUILDER - PHYSICS PARETO MC (Basic V4 Implementation)
# ========================================================================
class SequenceBuilderPhysicsMC: # (Kept original logic - Needs update for full Ensemble integration)
    def __init__(self, all_potential_clips: List[ClipSegment], audio_data: Dict, analysis_config: AnalysisConfig):
        logger.warning("Physics Pareto MC mode is using the older V4 scoring logic. Ensemble features (SyncNet, LVRE, etc.) are NOT used in this mode.")
        self.all_clips = all_potential_clips
        self.audio_data = audio_data
        self.analysis_config = analysis_config # Use the passed config (could be adapted)
        self.target_duration = audio_data.get('duration', 0)
        self.beat_times = audio_data.get('beat_times', [])
        self.audio_segments = audio_data.get('segment_features', [])
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
        self.effects: Dict[str, EffectParams] = {} # Populated later from RenderConfig

    def get_audio_segment_at(self, time): # (Unchanged from v4.7.3)
        """Finds the audio segment active at a given time."""
        if not self.audio_segments: return None
        for seg in self.audio_segments:
            # Check if time falls within [start, end)
            if seg['start'] <= time < seg['end']:
                return seg
        # Handle time exactly at or slightly past the end of the last segment
        if time >= self.audio_segments[-1]['end'] - 1e-6:
            return self.audio_segments[-1]
        logger.debug(f"Time {time:.2f}s outside defined audio segments.")
        return None # Return None if time is before the first segment or in a gap

    def build_sequence(self) -> List[ClipSegment]: # (Unchanged logic from v4.7.3)
        logger.info(f"--- Composing Sequence (Physics Pareto MC Mode - V4 Logic - {self.mc_iterations} iterations) ---")
        if not self.all_clips or not self.audio_segments or self.target_duration <= 0 or not self.effects:
            logger.error("Physics MC pre-conditions not met (clips, audio segments, duration, or effects dictionary missing).")
            return []

        # Filter clips based on MIN duration only for initial eligibility (max is handled during build)
        eligible_clips = [c for c in self.all_clips if c.duration >= self.min_clip_duration]
        if not eligible_clips:
            logger.error(f"No clips meet minimum duration ({self.min_clip_duration:.2f}s) for Physics MC.")
            return []

        self.all_clips = eligible_clips # Use only eligible clips
        logger.info(f"Starting Physics MC with {len(self.all_clips)} eligible clips.")

        pareto_front: List[Tuple[List[Tuple[ClipSegment, float, EffectParams]], List[float]]] = []
        successful_sims = 0
        pbar_mc = tqdm(range(self.mc_iterations), desc="MC Simulations (V4 Logic)", leave=False, disable=tqdm is None)

        for i in pbar_mc:
            sim_seq_info = None; scores = None # Define outside try
            try:
                # Run one stochastic simulation to build a sequence candidate
                sim_seq_info = self._run_stochastic_build()
                if sim_seq_info:
                    successful_sims += 1
                    # Evaluate the candidate sequence against Pareto objectives
                    scores = self._evaluate_pareto(sim_seq_info)
                    # Check if scores are valid numbers before updating front
                    if all(np.isfinite(s) for s in scores):
                        self._update_pareto_front(pareto_front, (sim_seq_info, scores))
                    else:
                        logger.debug(f"MC iter {i}: Invalid scores generated: {scores}")
            except Exception as mc_err:
                # Log error but continue with other simulations
                logger.error(f"MC simulation iteration {i} failed: {mc_err}", exc_info=False)

        if pbar_mc: pbar_mc.close()
        logger.info(f"MC simulations complete: {successful_sims}/{self.mc_iterations} sims yielded sequences. Pareto front size: {len(pareto_front)}")

        if not pareto_front:
            logger.error("Monte Carlo simulation yielded no valid sequences on the Pareto front.")
            return []

        # Select the 'best' solution from the Pareto front based on weighted sum of objectives
        # Weights are defined in analysis_config
        obj_weights = [self.w_r, self.w_m, self.w_c, self.w_v, self.w_e]
        # Note: Objective scores are [NegRhythmOffset, Mood, Continuity, Variety, NegEfficiencyCost]
        # We want to maximize Mood, Continuity, Variety, and minimize RhythmOffset and EfficiencyCost
        # So we maximize: w_m*M + w_c*C + w_v*V - w_r*R - w_e*EC
        # which is equivalent to maximizing: w_m*M + w_c*C + w_v*V + w_r*(-R) + w_e*(-EC)
        best_solution = max(pareto_front, key=lambda item: sum(s * w for s, w in zip(item[1], obj_weights)))

        logger.info(f"Chosen sequence objectives (NegR, M, C, V, NegEC): {[f'{s:.3f}' for s in best_solution[1]]}")

        # Reconstruct the final sequence list of ClipSegments from the chosen solution info
        final_sequence_info = best_solution[0]
        final_sequence_segments: List[ClipSegment] = []
        current_t = 0.0
        for i, (clip, duration, effect) in enumerate(final_sequence_info):
            if not isinstance(clip, ClipSegment):
                logger.warning(f"Invalid item in chosen sequence info at index {i}. Skipping.")
                continue
            # Set sequence-specific timing and effect info on the clip object
            clip.sequence_start_time = current_t
            clip.sequence_end_time = current_t + duration
            clip.chosen_duration = duration
            clip.chosen_effect = effect
            # Determine subclip times within the source clip
            clip.subclip_start_time_in_source = clip.start_time # Start from beginning of potential clip
            clip.subclip_end_time_in_source = min(clip.start_time + duration, clip.end_time) # End after chosen duration or at clip end

            final_sequence_segments.append(clip)
            current_t += duration

        if not final_sequence_segments:
            logger.error("Physics MC failed to reconstruct final sequence from chosen Pareto solution.")
            return []

        logger.info("--- Sequence Composition Complete (Physics Pareto MC - V4 Logic) ---")
        logger.info(f"Final Duration: {current_t:.2f}s, Clips: {len(final_sequence_segments)}")
        return final_sequence_segments

    def _run_stochastic_build(self): # (Unchanged logic from v4.7.3)
        """Runs a single Monte Carlo simulation to build one sequence candidate."""
        sequence_info: List[Tuple[ClipSegment, float, EffectParams]] = [] # Stores (clip, duration, effect)
        current_time = 0.0
        # Create a list of indices into self.all_clips to track availability
        available_clip_indices = list(range(len(self.all_clips)))
        random.shuffle(available_clip_indices) # Shuffle for randomness
        last_clip_segment: Optional[ClipSegment] = None
        num_sources = len(set(c.source_video_path for c in self.all_clips))

        while current_time < self.target_duration and available_clip_indices:
            audio_seg = self.get_audio_segment_at(current_time)
            if not audio_seg: break # Stop if no audio segment found

            # --- Candidate Selection based on V4 Fit ---
            candidates_info = [] # List of (clip, list_index, probability)
            total_prob = 0.0
            for list_idx_pos, original_clip_index in enumerate(available_clip_indices):
                clip = self.all_clips[original_clip_index]
                # Calculate V4 fit probability
                prob = clip.clip_audio_fit(audio_seg, self.analysis_config)

                # Apply V4 repetition penalty
                if num_sources > 1 and last_clip_segment and str(clip.source_video_path) == str(last_clip_segment.source_video_path):
                    prob *= (1.0 - self.analysis_config.variety_repetition_penalty)

                if prob > 1e-5: # Only consider candidates with non-negligible probability
                    candidates_info.append((clip, list_idx_pos, prob))
                    total_prob += prob

            if not candidates_info: break # Stop if no suitable candidates found

            # --- Choose Clip Stochastically ---
            probabilities = [p / (total_prob + 1e-9) for _, _, p in candidates_info] # Normalize probabilities
            try:
                # Randomly choose based on calculated probabilities
                chosen_candidate_local_idx = random.choices(range(len(candidates_info)), weights=probabilities, k=1)[0]
            except ValueError: # Handle potential empty probabilities list or other issues
                if candidates_info: chosen_candidate_local_idx = random.randrange(len(candidates_info)) # Fallback to uniform random
                else: break # Should not happen if candidates_info check passed

            chosen_clip, chosen_list_idx_pos, _ = candidates_info[chosen_candidate_local_idx]

            # --- Determine Duration ---
            remaining_time = self.target_duration - current_time
            # Duration is limited by clip's duration, remaining time, and max config duration
            chosen_duration = min(chosen_clip.duration, remaining_time, self.max_clip_duration)
            # Ensure duration meets minimum, unless it's the very last clip
            chosen_duration = max(chosen_duration, self.min_clip_duration if remaining_time >= self.min_clip_duration else 0.01)
            chosen_duration = max(0.01, chosen_duration) # Final check for positive duration

            # --- Choose Effect Stochastically (V4 Logic) ---
            effect_options = list(self.effects.values())
            efficiencies = []
            for e in effect_options:
                denom = e.tau * e.psi; numer = e.epsilon
                # Calculate efficiency: Perceptual Gain / (Duration * Impact)
                eff = ((numer + 1e-9) / (denom + 1e-9)) if abs(denom) > 1e-9 else (0.0 if abs(numer) < 1e-9 else 1e9) # Handle zero denominator/numerator
                efficiencies.append(eff)

            # Boost efficiency of 'cut' transition
            cut_index = next((i for i, e in enumerate(effect_options) if e.type == "cut"), -1)
            if cut_index != -1: efficiencies[cut_index] = max(efficiencies[cut_index], 1.0) * 2.0 # Example boost

            # Choose effect based on efficiency probabilities
            positive_efficiencies = [max(0, eff) for eff in efficiencies]
            total_efficiency = sum(positive_efficiencies)
            chosen_effect = self.effects.get('cut', EffectParams(type='cut')) # Default to cut
            if total_efficiency > 1e-9 and effect_options:
                 effect_probs = [eff / total_efficiency for eff in positive_efficiencies]
                 # Renormalize probabilities if needed due to floating point issues
                 sum_probs = sum(effect_probs)
                 if abs(sum_probs - 1.0) > 1e-6: effect_probs = [p / (sum_probs + 1e-9) for p in effect_probs]
                 try:
                     chosen_effect = random.choices(effect_options, weights=effect_probs, k=1)[0]
                 except (ValueError, IndexError) as choice_err:
                      logger.debug(f"Effect choice failed: {choice_err}. Defaulting to cut.")
                      chosen_effect = self.effects.get('cut', EffectParams(type='cut'))

            # --- Add to Sequence and Update State ---
            sequence_info.append((chosen_clip, chosen_duration, chosen_effect))
            last_clip_segment = chosen_clip
            current_time += chosen_duration
            # Remove chosen clip index from available list
            available_clip_indices.pop(chosen_list_idx_pos)

        # Return the built sequence info if it meets minimum duration criteria
        final_sim_duration = sum(item[1] for item in sequence_info)
        return sequence_info if final_sim_duration >= self.min_clip_duration else None

    def _evaluate_pareto(self, seq_info): # (Unchanged V4 evaluation logic)
        """Evaluates a sequence candidate against the V4 Pareto objectives."""
        if not seq_info: return [-1e9] * 5 # Return poor scores if sequence is empty
        num_clips = len(seq_info); total_duration = sum(item[1] for item in seq_info)
        if total_duration <= 1e-6: return [-1e9] * 5 # Avoid division by zero

        # Config parameters needed for evaluation
        w_r, w_m, w_c, w_v, w_e = self.w_r, self.w_m, self.w_c, self.w_v, self.w_e
        sigma_m_sq = self.analysis_config.mood_similarity_variance**2 * 2
        kd = self.analysis_config.continuity_depth_weight
        lambda_penalty = self.analysis_config.variety_repetition_penalty
        num_sources = len(set(item[0].source_video_path for item in seq_info))

        # --- 1. Rhythm Score (Negative Squared Offset from Beats) ---
        r_score_sum = 0.0; num_trans_r = 0; current_t = 0.0
        for i, (_, duration, _) in enumerate(seq_info):
            trans_time = current_t + duration
            if i < num_clips - 1: # Evaluate transitions between clips
                nearest_b = self._nearest_beat_time(trans_time)
                if nearest_b is not None and self.beat_period > 1e-6:
                    # Calculate normalized squared offset from nearest beat
                    offset_norm = abs(trans_time - nearest_b) / self.beat_period
                    r_score_sum += offset_norm**2
                    num_trans_r += 1
            current_t = trans_time
        avg_sq_offset = (r_score_sum / num_trans_r) if num_trans_r > 0 else 1.0 # Default penalty if no transitions
        neg_r_score = -w_r * avg_sq_offset # Higher offset -> more negative score (worse)

        # --- 2. Mood Score (Average Mood Similarity) ---
        m_score_sum = 0.0; mood_calcs = 0; current_t = 0.0
        for clip, duration, _ in seq_info:
            mid_time = current_t + duration / 2.0
            audio_seg = self.get_audio_segment_at(mid_time)
            if audio_seg:
                # Use V4 mood vectors
                vid_mood = np.asarray(clip.mood_vector)
                aud_mood = np.asarray(audio_seg.get('m_i', [0.0, 0.0]))
                mood_dist_sq = np.sum((vid_mood - aud_mood)**2)
                # Gaussian similarity based on distance
                m_score_sum += exp(-mood_dist_sq / (sigma_m_sq + 1e-9))
                mood_calcs += 1
            current_t += duration
        m_score = w_m * (m_score_sum / mood_calcs if mood_calcs > 0 else 0.0) # Higher similarity -> higher score (better)

        # --- 3. Continuity Score (Average Transition Smoothness) ---
        c_score_sum = 0.0; num_trans_c = 0
        for i in range(num_clips - 1):
            clip1, _, effect = seq_info[i]
            clip2, _, _ = seq_info[i+1]
            # Use V4 feature vector [v_k, a_j, d_r]
            f1 = clip1.get_feature_vector(self.analysis_config)
            f2 = clip2.get_feature_vector(self.analysis_config)
            safe_kd = max(0.0, kd) # Ensure non-negative weight
            # Calculate squared Euclidean distance in feature space (weighted by depth)
            delta_e_sq = (f1[0]-f2[0])**2 + (f1[1]-f2[1])**2 + safe_kd*(f1[2]-f2[2])**2
            # Normalize distance
            max_delta_e_sq = 1**2 + 1**2 + safe_kd*(1**2) # Max possible squared distance
            delta_e_norm_sq = delta_e_sq / (max_delta_e_sq + 1e-9)
            # Continuity term: 1 - sqrt(normalized_distance)
            cont_term = (1.0 - np.sqrt(np.clip(delta_e_norm_sq, 0.0, 1.0)))
            # Add effect's perceptual gain
            c_score_sum += cont_term + effect.epsilon
            num_trans_c += 1
        c_score = w_c * (c_score_sum / num_trans_c if num_trans_c > 0 else 1.0) # Higher continuity -> higher score (better)

        # --- 4. Variety Score (Average Entropy minus Repetition Penalty) ---
        # Average visual entropy (phi) from clips
        valid_phis = [item[0].phi for item in seq_info if isinstance(item[0], ClipSegment) and item[0].phi is not None and np.isfinite(item[0].phi)]
        avg_phi = np.mean(valid_phis) if valid_phis else 0.0
        # Calculate repetition penalty
        repetition_count = 0; num_trans_v = 0
        if num_sources > 1: # Only penalize if multiple sources exist
            for i in range(num_clips - 1):
                p1 = str(seq_info[i][0].source_video_path)
                p2 = str(seq_info[i+1][0].source_video_path)
                if p1 and p2 and p1 == p2: repetition_count += 1 # Count same-source transitions
                num_trans_v += 1
        rep_term = lambda_penalty * (repetition_count / num_trans_v if num_trans_v > 0 else 0)
        # Normalize average entropy (optional, depends on range of phi)
        max_entropy = log(256) # Max possible entropy for 8-bit grayscale histogram
        avg_phi_norm = avg_phi / max_entropy if max_entropy > 0 else 0.0
        v_score = w_v * (avg_phi_norm - rep_term) # Higher entropy & lower repetition -> higher score (better)

        # --- 5. Efficiency Score (Negative Average Cost per Transition) ---
        ec_score_sum = 0.0; cost_calcs = 0
        for _, _, effect in seq_info:
             psi_tau = effect.psi * effect.tau # Cost component (Impact * Duration)
             epsilon = effect.epsilon # Benefit component (Perceptual Gain)
             # Cost = (Impact * Duration) / Perceptual Gain
             cost = (psi_tau + 1e-9) / (epsilon + 1e-9) if abs(epsilon) > 1e-9 else (psi_tau + 1e-9) / 1e-9 # Handle zero gain
             ec_score_sum += cost
             cost_calcs += 1
        avg_cost = (ec_score_sum / cost_calcs if cost_calcs > 0 else 0.0)
        neg_ec_score = -w_e * avg_cost # Lower cost -> less negative score (better)

        # Return the list of objective scores
        final_scores = [neg_r_score, m_score, c_score, v_score, neg_ec_score]
        # Ensure all scores are finite floats
        return [float(s) if np.isfinite(s) else -1e9 for s in final_scores]

    def _nearest_beat_time(self, time_sec): # (Unchanged from v4.7.3)
        """Finds the time of the beat closest to the given time."""
        if not self.beat_times: return None
        beat_times_arr = np.asarray(self.beat_times)
        if len(beat_times_arr) == 0: return None
        # Find index of the minimum absolute difference
        closest_beat_index = np.argmin(np.abs(beat_times_arr - time_sec))
        return float(beat_times_arr[closest_beat_index])

    def _update_pareto_front(self, front, new_solution): # (Unchanged logic from v4.7.3)
        """Updates the Pareto front with a new solution, removing dominated ones."""
        new_seq_info, new_scores = new_solution
        # Basic validation of scores
        if not isinstance(new_scores, list) or len(new_scores) != 5 or not all(isinstance(s, float) and np.isfinite(s) for s in new_scores):
            logger.debug(f"Skipping Pareto update due to invalid new scores: {new_scores}")
            return

        dominated_indices = set()
        is_dominated_by_existing = False

        # Iterate through existing front solutions
        indices_to_check = list(range(len(front))) # Create list of indices to iterate over safely
        for i in reversed(indices_to_check): # Iterate backwards for safe deletion
            # Check if index is still valid after potential deletions
            if i >= len(front): continue

            existing_seq_info, existing_scores = front[i]
            # Validate existing scores
            if not isinstance(existing_scores, list) or len(existing_scores) != 5 or not all(isinstance(s, float) and np.isfinite(s) for s in existing_scores):
                logger.warning(f"Removing solution with invalid scores from Pareto front at index {i}: {existing_scores}")
                del front[i]
                continue

            # Check for dominance relationship
            if self._dominates(new_scores, existing_scores):
                # New solution dominates this existing one
                dominated_indices.add(i)
            if self._dominates(existing_scores, new_scores):
                # New solution is dominated by this existing one
                is_dominated_by_existing = True
                break # No need to check further, new solution won't be added

        # Update the front based on dominance checks
        if not is_dominated_by_existing:
            # Remove all existing solutions dominated by the new one
            # Sort indices descending to avoid index shifting issues during deletion
            for i in sorted(list(dominated_indices), reverse=True):
                 if 0 <= i < len(front): del front[i] # Check index bounds again before deleting
            # Add the new non-dominated solution
            front.append(new_solution)

    def _dominates(self, scores1, scores2): # (Unchanged from v4.7.3)
        """Checks if solution with scores1 dominates solution with scores2."""
        # scores1 dominates scores2 if scores1 is at least as good in all objectives
        # and strictly better in at least one objective.
        if len(scores1) != len(scores2): raise ValueError("Score lists must have same length for dominance check.")
        better_in_at_least_one = False
        for s1, s2 in zip(scores1, scores2):
            if s1 < s2 - 1e-9: # s1 is significantly worse than s2 in one objective
                return False
            if s1 > s2 + 1e-9: # s1 is significantly better than s2 in at least one objective
                better_in_at_least_one = True
        # Must be better in at least one objective to dominate
        return better_in_at_least_one

# ========================================================================
#        MOVIEPY VIDEO BUILDING FUNCTION (Includes GFPGAN Hook)
# ========================================================================
def buildSequenceVideo(final_sequence: List[ClipSegment], output_video_path: str, master_audio_path: str, render_config: RenderConfig):
    """Builds the final video sequence using MoviePy's make_frame with optional GFPGAN enhancement."""
    logger.info(f"Rendering video to '{os.path.basename(output_video_path)}' with audio '{os.path.basename(master_audio_path)}' using MoviePy make_frame...")
    start_time = time.time()
    if ENABLE_PROFILING: tracemalloc.start(); start_memory = tracemalloc.get_traced_memory()[0]

    # --- Pre-checks ---
    if not final_sequence: logger.error("Cannot build: Empty final sequence provided."); raise ValueError("Empty sequence")
    if not master_audio_path or not os.path.exists(master_audio_path): logger.error(f"Cannot build: Master audio not found at '{master_audio_path}'"); raise FileNotFoundError("Master audio not found")
    if not output_video_path: logger.error("Cannot build: Output video path not specified."); raise ValueError("Output path required")

    width = render_config.resolution_width; height = render_config.resolution_height; fps = render_config.fps
    if not isinstance(fps, (int, float)) or fps <= 0: fps = 30; logger.warning(f"Invalid render FPS {render_config.fps}, using default 30.")

    # --- GFPGAN Setup (Conditional) ---
    gfpgan_enhancer = None
    if render_config.use_gfpgan_enhance:
        logger.info("GFPGAN enhancement enabled. Attempting to load model...")
        try:
            # Ensure necessary dependencies like realesrgan, facexlib are installed
            # This requires: pip install gfpgan>=1.3.8 realesrgan>=0.3.0 facexlib>=0.3.0

            # Find appropriate model path or use default (check GFPGAN docs/repo)
            # Example using a common pre-trained model name
            model_path = render_config.gfpgan_model_path # Use path from config
            if not model_path:
                 raise ValueError("GFPGAN model path not specified in RenderConfig.")
            # Add logic here to download the model if it doesn't exist (or handle error)
            if not os.path.exists(model_path):
                logger.warning(f"GFPGAN model not found at '{model_path}'. Attempting download placeholder...")
                # --- Placeholder for Download Logic ---
                # You might use requests, huggingface_hub, or basicsr.utils.download_util
                # Example conceptual download:
                # download_url = "URL_TO_GFPGAN_MODEL_PTH"
                # download_file(download_url, model_path)
                # --- End Placeholder ---
                # For now, just disable if model not found manually
                raise FileNotFoundError(f"GFPGAN model not found at {model_path}. Enhancement disabled. Please download manually or check path.")

            # Initialize GFPGANer (check docs for parameters)
            # upscale=1 means no upscaling, just restoration
            # bg_upsampler=None means background is not processed by another upscaler (like RealESRGAN)
            gfpgan_enhancer = GFPGANer(model_path=model_path, upscale=1, arch='clean',
                                       channel_multiplier=2, bg_upsampler=None)
            logger.info("GFPGAN Enhancer loaded successfully.")

        except ImportError:
            logger.error("GFPGAN library not found. Enhancement disabled. Install with 'pip install gfpgan'.")
            render_config = dataclass_replace(render_config, use_gfpgan_enhance=False) # Disable flag if import fails
        except FileNotFoundError as fnf_err:
            logger.error(f"{fnf_err}") # Log the specific file not found error
            render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
        except ValueError as val_err: # Handle missing path from config
            logger.error(f"{val_err}")
            render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
        except Exception as gfpgan_load_err:
            logger.error(f"Failed to load GFPGAN model: {gfpgan_load_err}", exc_info=True)
            render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)


    # --- Pre-load Source Video Clips ---
    source_clips_dict: Dict[str, Optional[VideoFileClip]] = {}
    logger.info("Pre-loading/preparing source video readers...")
    unique_source_paths = sorted(list(set(str(seg.source_video_path) for seg in final_sequence)))
    for source_path in unique_source_paths:
        if not source_path or not os.path.exists(source_path):
            logger.error(f"Source video not found: {source_path}. Frames from this source will be black.")
            source_clips_dict[source_path] = None; continue
        try:
            logger.debug(f"Preparing reader for: {os.path.basename(source_path)}")
            # Load without audio, specify target resolution for efficiency if needed,
            # but resizing in make_frame ensures final dimensions are correct.
            clip_obj = VideoFileClip(source_path, audio=False) # target_resolution=(height, width)
            # Ensure the reader is initialized (accessing duration usually does this)
            _ = clip_obj.duration
            if not hasattr(clip_obj, 'reader') or clip_obj.reader is None:
                 raise RuntimeError("MoviePy reader initialization failed.")
            source_clips_dict[source_path] = clip_obj
        except Exception as load_err:
            logger.error(f"Failed to load source video {os.path.basename(source_path)}: {load_err}")
            source_clips_dict[source_path] = None # Mark as failed

    # --- Define make_frame function ---
    def make_frame(t):
        """Generates the frame for time 't' of the final sequence."""
        active_segment = None
        final_source_time = -1.0 # Initialize for logging
        source_path_log = "N/A"
        # Find the clip segment active at time 't'
        for segment in final_sequence:
            # Add small tolerance to end time check
            if segment.sequence_start_time <= t < segment.sequence_end_time + 1e-6:
                active_segment = segment
                break

        # Handle case where t is exactly the end time of the last segment
        if active_segment is None and final_sequence and abs(t - final_sequence[-1].sequence_end_time) < 1e-6:
            active_segment = final_sequence[-1]

        if active_segment:
            source_path = str(active_segment.source_video_path)
            source_path_log = os.path.basename(source_path) # For logging
            source_clip = source_clips_dict.get(source_path)

            if source_clip and hasattr(source_clip, 'get_frame'):
                try:
                    # Calculate the corresponding time within the source clip's subclip
                    clip_time_in_seq = t - active_segment.sequence_start_time
                    source_time = active_segment.subclip_start_time_in_source + clip_time_in_seq

                    # Clamp source_time to the valid range of the subclip defined for this segment
                    subclip_start = active_segment.subclip_start_time_in_source
                    subclip_end = active_segment.subclip_end_time_in_source
                    source_dur = source_clip.duration if source_clip.duration else 0

                    # Clamp first to source duration, then to subclip range
                    final_source_time = np.clip(source_time, 0, source_dur - 1e-6 if source_dur > 0 else 0)
                    final_source_time = np.clip(final_source_time, subclip_start, subclip_end - 1e-6 if subclip_end > subclip_start else subclip_start)

                    # Get the frame (MoviePy returns RGB)
                    frame_rgb = source_clip.get_frame(final_source_time)
                    if frame_rgb is None: raise ValueError("get_frame returned None")

                    # Convert to BGR for OpenCV processing (like resizing, GFPGAN)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    # Ensure frame matches target render dimensions
                    if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
                        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

                    # --- Apply GFPGAN Enhancement (Optional) ---
                    if gfpgan_enhancer and render_config.use_gfpgan_enhance:
                        # Apply only if face is likely present (using pre-calculated ratio)
                        if active_segment.face_presence_ratio > 0.1: # Threshold can be adjusted
                            try:
                                # GFPGAN expects BGR, uint8 input
                                # enhance() returns: tuple(cropped_faces, restored_faces, restored_img)
                                _, _, restored_img = gfpgan_enhancer.enhance(
                                    frame_bgr, # Pass the BGR frame
                                    has_aligned=False, # Assume face needs detection/alignment by GFPGAN's facexlib
                                    only_center_face=False, # Process all detected faces
                                    paste_back=True, # Paste enhanced faces back onto original image
                                    weight=render_config.gfpgan_fidelity_weight # Control fidelity/realism trade-off
                                )
                                if restored_img is not None:
                                    frame_bgr = restored_img # Replace original BGR frame with enhanced one
                                else:
                                     logger.debug(f"GFPGAN enhancement returned None for frame at t={t:.2f}")
                            except Exception as gfpgan_err:
                                # Log warning periodically to avoid flooding
                                if int(t*10) % 50 == 0: # Log roughly every 5 seconds
                                     logger.warning(f"GFPGAN enhancement failed for frame at t={t:.2f}: {gfpgan_err}")

                    # --- Apply Other Effects (Placeholder) ---
                    # Example: Fade effect could modify alpha or blend frames here based on t

                    # Convert final frame back to RGB for MoviePy
                    final_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    return final_frame_rgb

                except Exception as frame_err:
                    logger.error(f"Error getting/processing frame at t={t:.3f} (Source: {source_path_log} @ {final_source_time:.3f}): {frame_err}", exc_info=False) # Less verbose logging
                    # Return black frame on error
                    return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # Source clip object is missing or invalid
                logger.warning(f"Source clip '{source_path_log}' invalid or reader failed at t={t:.3f}. Returning black frame.")
                return np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Time 't' is outside the calculated sequence range
            seq_end_time = final_sequence[-1].sequence_end_time if final_sequence else 0
            logger.warning(f"Time t={t:.4f}s outside sequence range [0, {seq_end_time:.4f}s]. Returning black frame.")
            return np.zeros((height, width, 3), dtype=np.uint8)

    # --- Create and Write Video ---
    master_audio = None; sequence_clip_mvp = None; temp_audio_filepath = None
    try:
        total_duration = final_sequence[-1].sequence_end_time if final_sequence else 0
        if total_duration <= 0: raise ValueError(f"Sequence has zero or negative duration ({total_duration}).")

        logger.info(f"Creating final VideoClip object (Duration: {total_duration:.2f}s, FPS: {fps})")
        sequence_clip_mvp = VideoClip(make_frame, duration=total_duration, ismask=False)

        logger.debug(f"Loading master audio: {master_audio_path}")
        master_audio = AudioFileClip(master_audio_path)
        logger.debug(f"Master audio duration: {master_audio.duration:.2f}s")

        # Adjust audio/video length mismatch
        if master_audio.duration > total_duration + 1e-3: # Audio longer than video
            logger.info(f"Audio duration ({master_audio.duration:.2f}s) > Video duration ({total_duration:.2f}s). Trimming audio.")
            master_audio = master_audio.subclip(0, total_duration)
        elif master_audio.duration < total_duration - 1e-3: # Video longer than audio
            logger.warning(f"Video duration ({total_duration:.2f}s) > Audio duration ({master_audio.duration:.2f}s). Trimming video to audio length.")
            total_duration = master_audio.duration
            sequence_clip_mvp = sequence_clip_mvp.set_duration(total_duration)

        if master_audio:
            sequence_clip_mvp = sequence_clip_mvp.set_audio(master_audio)
        else:
            logger.warning("No master audio loaded or audio duration is zero. Rendering silent video.")

        # Prepare temporary audio file path for MoviePy Muxing
        temp_audio_filename = f"temp-audio_{int(time.time())}_{random.randint(1000,9999)}.m4a" # Use m4a or aac
        temp_audio_dir = os.path.dirname(output_video_path) or "."
        os.makedirs(temp_audio_dir, exist_ok=True)
        temp_audio_filepath = os.path.join(temp_audio_dir, temp_audio_filename)

        # Prepare FFmpeg parameters for write_videofile
        ffmpeg_params_list = []
        if render_config.preset: ffmpeg_params_list.extend(["-preset", str(render_config.preset)])
        if render_config.crf is not None: ffmpeg_params_list.extend(["-crf", str(render_config.crf)])
        # Add other params like -tune, -profile:v if needed

        write_params = {
            "codec": render_config.video_codec,
            "audio_codec": render_config.audio_codec,
            "temp_audiofile": temp_audio_filepath,
            "remove_temp": True,
            "threads": render_config.threads,
            "preset": None, # Preset passed via ffmpeg_params
            "logger": 'bar' if tqdm is not None else None, # Use tqdm progress bar if available
            "write_logfile": False, # Disable MoviePy log file
            "audio_bitrate": render_config.audio_bitrate,
            "fps": fps,
            "ffmpeg_params": ffmpeg_params_list if ffmpeg_params_list else None
        }

        logger.info(f"Writing final video using MoviePy write_videofile...")
        logger.debug(f"Write params: Codec={write_params['codec']}, AudioCodec={write_params['audio_codec']}, Threads={write_params['threads']}, FPS={write_params['fps']}, Preset={render_config.preset}, CRF={render_config.crf}, Params={write_params['ffmpeg_params']}")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True) # Ensure output dir exists

        # Perform the render
        sequence_clip_mvp.write_videofile(output_video_path, **write_params)

        # --- Success Logging ---
        if ENABLE_PROFILING and tracemalloc.is_tracing():
             current_mem, peak_mem = tracemalloc.get_traced_memory()
             end_memory = current_mem; tracemalloc.stop()
             logger.info(f"Render Perf: Time: {time.time() - start_time:.2f}s, PyMem Δ: {(end_memory - start_memory) / 1024**2:.2f} MB, Peak: {peak_mem / 1024**2:.2f} MB")
        else: logger.info(f"Render took {time.time() - start_time:.2f} seconds.")
        logger.info(f"MoviePy rendering successful: '{output_video_path}'")

    except Exception as e:
        if ENABLE_PROFILING and tracemalloc.is_tracing(): tracemalloc.stop()
        logger.error(f"MoviePy rendering failed: {e}", exc_info=True)
        # Attempt to clean up potentially incomplete output file
        if os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
                logger.info(f"Removed potentially incomplete output file: {output_video_path}")
            except OSError as del_err:
                logger.warning(f"Could not remove failed output file {output_video_path}: {del_err}")

        # Attempt to clean up temporary audio file
        if temp_audio_filepath and os.path.exists(temp_audio_filepath):
             try:
                 os.remove(temp_audio_filepath)
                 logger.debug(f"Removed temp audio file {temp_audio_filepath} after error.")
             except OSError as del_err:
                 logger.warning(f"Could not remove temp audio file {temp_audio_filepath} after error: {del_err}")
        raise # Re-raise the exception to be caught by the calling thread
    finally:
        # --- Cleanup MoviePy Objects ---
        logger.debug("Cleaning up MoviePy objects...")
        if sequence_clip_mvp and hasattr(sequence_clip_mvp, 'close'):
            try: sequence_clip_mvp.close() # Use close() method if available
            except Exception as e: logger.debug(f"Minor error closing sequence_clip: {e}")
        if master_audio and hasattr(master_audio, 'close'):
            try: master_audio.close()
            except Exception as e: logger.debug(f"Minor error closing master_audio: {e}")
        # Close source clip readers
        for clip_key, source_clip_obj in source_clips_dict.items():
            if source_clip_obj and hasattr(source_clip_obj, 'close'):
                try: source_clip_obj.close()
                except Exception as e: logger.debug(f"Minor error closing source clip {clip_key}: {e}")
        source_clips_dict.clear() # Clear the dictionary
        # Force garbage collection maybe?
        import gc
        gc.collect()
        logger.debug("MoviePy clip cleanup attempt finished.")


# ========================================================================
#         WORKER FUNCTION (Returns None for Frame Features)
# ========================================================================
def process_single_video(video_path: str, master_audio_data: Dict, analysis_config: AnalysisConfig, output_dir: str) -> Tuple[str, str, List[ClipSegment]]:
    """Worker function: Analyzes video, returns potential clips, saves data optionally."""
    start_t = time.time(); pid = os.getpid(); thread_name = threading.current_thread().name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    worker_logger = logging.getLogger(f"Worker.{pid}.{thread_name}") # Worker specific logger
    if not worker_logger.hasHandlers(): # Add handler if logger was just created
        ch = logging.StreamHandler(sys.stdout); formatter = logging.Formatter(f'%(asctime)s - %(levelname)-8s - Worker {pid} - %(message)s')
        ch.setFormatter(formatter); worker_logger.addHandler(ch); worker_logger.setLevel(logging.INFO) # Set level for worker logger

    worker_logger.info(f"Starting Analysis: {base_name}")
    status = "Unknown Error"; potential_clips: List[ClipSegment] = []; frame_features: Optional[List[Dict]] = None

    try:
        analyzer = VideousMain() # Instantiate analyzer in worker
        frame_features, potential_clips_result = analyzer.analyzeVideo(video_path, analysis_config, master_audio_data)

        if potential_clips_result is None:
            status = "Analysis Failed (returned None)"
            potential_clips = []
        elif not potential_clips_result:
            status = "Analysis OK (0 potential clips)"
            potential_clips = []
        else:
            potential_clips = [clip for clip in potential_clips_result if isinstance(clip, ClipSegment)]
            status = f"Analysis OK ({len(potential_clips)} potential clips)"

            # Save analysis data if requested (saves only clip metadata by default now)
            if analysis_config.save_analysis_data:
                 # Pass None for frame_features to saveAnalysisData to optimize memory unless needed
                 save_frame_features = None # <<< Change to frame_features if full frame data saving is desired
                 if (save_frame_features or potential_clips): # Check if there's anything to save
                     try:
                         analyzer.saveAnalysisData(video_path, save_frame_features, potential_clips, output_dir, analysis_config)
                     except Exception as save_err:
                         worker_logger.error(f"Save analysis data failed for {base_name}: {save_err}")
                 else:
                     worker_logger.debug(f"Save analysis data requested for {base_name}, but nothing to save.")

    except Exception as e:
        status = f"Failed: {type(e).__name__}"
        worker_logger.error(f"!!! FATAL ERROR analyzing {base_name} in worker {pid} !!!", exc_info=True)
        potential_clips = []; frame_features = None # Ensure clear failure state
    finally:
        if 'analyzer' in locals(): del analyzer # Help GC
        # Frame features are potentially large, ensure they are cleared if not needed outside saving
        frame_features = None # Explicitly clear reference

    end_t = time.time()
    worker_logger.info(f"Finished Analysis {base_name} ({status}) in {end_t - start_t:.2f}s")
    # Return only essential results: path, status, and ClipSegment list
    return (video_path, status, potential_clips if potential_clips is not None else [])


# ========================================================================
#                      APP INTERFACE (Optimized Workflow v5.3)
# ========================================================================
# ========================================================================
#                      APP INTERFACE (Updated Workflow v5.4/5.5)
# ========================================================================
# ========================================================================
#                      APP INTERFACE (Updated Workflow v5.4/5.5)
# ========================================================================

ENABLE_PROFILING = False  # Assuming this is a global flag

# ========================================================================
#                      APP INTERFACE (Updated Workflow v5.4/5.5)
# ========================================================================

ENABLE_PROFILING = False  # Assuming this is a global flag

# ========================================================================
#                      APP INTERFACE (Updated Workflow v5.4/5.5)
# ========================================================================

ENABLE_PROFILING = False  # Assuming this is a global flag

# Ensure tkinter is imported (it should be near the top of your file)
import tkinter
# ========================================================================
#                      APP INTERFACE (Updated Workflow v5.4/5.5)
# ========================================================================

ENABLE_PROFILING = False  # Assuming this is a global flag

# Ensure tkinter is imported (it should be near the top of your file)
import tkinter

# ========================================================================
#                      APP INTERFACE (Updated Workflow v5.4/5.5)
# ========================================================================

ENABLE_PROFILING = False  # Assuming this is a global flag

# Ensure tkinter is imported
import tkinter

class VideousApp(customtkinter.CTk, TkinterDnD.DnDWrapper):
    """Main GUI application class for Videous Chef."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if TKINTERDND_AVAILABLE:
             try:
                 # self.TkdndVersion = TkinterDnD._require(self) # Optional check
                 pass # Rely on super().__init__() and DnDWrapper
             except Exception as dnd_init_err:
                  logger.error(f"TkinterDnD initialization failed: {dnd_init_err}")
        # else: pass

        # --- Standard Window Setup ---
        self.title("Videous Chef - Ensemble v5.4/5.5")
        self.geometry("950x850")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Core State ---
        self.video_files: List[str] = []
        self.beat_track_path: Optional[str] = None
        self.master_audio_data: Optional[Dict] = None
        self.all_potential_clips: List[ClipSegment] = []
        self.analysis_config: Optional[AnalysisConfig] = None
        self.adapted_analysis_config: Optional[AnalysisConfig] = None
        self.render_config: Optional[RenderConfig] = None
        self.is_processing = False
        # Process/Thread Management
        self.processing_thread = None
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.analysis_futures: List[concurrent.futures.Future] = []
        self.futures_map: Dict[concurrent.futures.Future, str] = {}
        self.total_tasks = 0
        self.completed_tasks = 0
        # Output Directories
        self.output_dir = "output_videous_chef_v5.4"
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.analysis_subdir = os.path.join(self.output_dir, "analysis_cache")
            os.makedirs(self.analysis_subdir, exist_ok=True)
            self.render_subdir = os.path.join(self.output_dir, "final_renders")
            os.makedirs(self.render_subdir, exist_ok=True)
        except OSError as e:
            logger.critical(f"Failed create output dirs: {e}")
            messagebox.showerror("Fatal Error", f"Could not create output directories in '{self.output_dir}':\n{e}\nCheck permissions.", parent=self)
            sys.exit(1)


        # --- Define Tkinter Variables (Still define them, even if sliders aren't created yet) ---
        logger.debug("Defining Tkinter variables...")
        self.slider_vars = {}
        default_analysis_cfg = AnalysisConfig()
        default_render_cfg = RenderConfig()
        for field_info in AnalysisConfig.__dataclass_fields__.values():
            key = field_info.name
            default = getattr(default_analysis_cfg, key)
            var = None
            try:
                # Use isinstance to determine the type based on the default value
                if isinstance(default, bool):
                    var = tkinter.BooleanVar(value=default)
                elif isinstance(default, int):
                    var = tkinter.IntVar(value=default)
                elif isinstance(default, float):
                    var = tkinter.DoubleVar(value=default)
                elif isinstance(default, str):
                    var = tkinter.StringVar(value=default)
                else:
                    logger.warning(f"Unsupported type for {key}: {type(default)}. Skipping variable creation.")
                if var is not None:
                    self.slider_vars[key] = var
                    logger.debug(f"Created Tk var for {key}: {type(var).__name__}")
            except Exception as e_tkvar:
                logger.error(f"Failed create Tk var for AnalysisConfig key '{key}': {e_tkvar}")
        try:  # Render vars
            self.slider_vars['render_width'] = tkinter.IntVar(value=default_render_cfg.resolution_width)
            self.slider_vars['render_height'] = tkinter.IntVar(value=default_render_cfg.resolution_height)
            self.slider_vars['render_fps'] = tkinter.IntVar(value=default_render_cfg.fps)
            fade_effect_default = default_render_cfg.effect_settings.get('fade', EffectParams())
            self.slider_vars['effect_fade_duration'] = tkinter.DoubleVar(value=fade_effect_default.tau)
            self.slider_vars['use_gfpgan_enhance'] = tkinter.BooleanVar(value=default_render_cfg.use_gfpgan_enhance)
            self.slider_vars['gfpgan_fidelity_weight'] = tkinter.DoubleVar(value=default_render_cfg.gfpgan_fidelity_weight)
            self.slider_vars['gfpgan_model_path'] = tkinter.StringVar(value=default_render_cfg.gfpgan_model_path or "")
        except Exception as e_render_tkvar:
            logger.error(f"Failed create Tk vars for RenderConfig: {e_render_tkvar}")
        if 'sequencing_mode' not in self.slider_vars:
            self.slider_vars['sequencing_mode'] = tkinter.StringVar(value="Greedy Heuristic")
        logger.debug(f"Defined {len(self.slider_vars)} Tkinter variables.")
        # --- End Tkinter Variable Definitions ---


        # --- UI Setup ---
        try:  # Fonts
            self.header_font = customtkinter.CTkFont(family="Garamond", size=28, weight="bold")
            self.label_font = customtkinter.CTkFont(family="Garamond", size=14)
            self.button_font = customtkinter.CTkFont(family="Garamond", size=12)
            self.dropdown_font = customtkinter.CTkFont(family="Garamond", size=12)
            self.small_font = customtkinter.CTkFont(family="Garamond", size=10)
            self.mode_font = customtkinter.CTkFont(family="Garamond", size=13, weight="bold")
            self.tab_font = customtkinter.CTkFont(family="Garamond", size=14, weight="bold")
            self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")
        except Exception as font_e:  # Fallback fonts
            logger.warning(f"Garamond font not found: {font_e}. Using defaults.")
            # Assign default fonts
            self.header_font = customtkinter.CTkFont(size=28, weight="bold")
            self.label_font = customtkinter.CTkFont(size=14)
            self.button_font = customtkinter.CTkFont(size=12)
            self.dropdown_font = customtkinter.CTkFont(size=12)
            self.small_font = customtkinter.CTkFont(size=10)
            self.mode_font = customtkinter.CTkFont(size=13, weight="bold")
            self.tab_font = customtkinter.CTkFont(size=14, weight="bold")
            self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")

        self._setup_luxury_theme()
        self._build_ui() # Build the UI structure
        logger.info("VideousApp initialized (v5.4/5.5).")

    def _setup_luxury_theme(self):
        # (Keep original code)
        self.diamond_white = "#F5F6F5"
        self.deep_black = "#1C2526"
        self.gold_accent = "#D4AF37"
        self.jewel_blue = "#2A4B7C"
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        self.configure(bg=self.deep_black)

    def _button_styles(self, border_width=1):
        # (Keep original code)
        return {
            "corner_radius": 8,
            "fg_color": self.jewel_blue,
            "hover_color": self.gold_accent,
            "border_color": self.diamond_white,
            "border_width": border_width,
            "text_color": self.diamond_white
        }

    def _radio_styles(self):
        # (Keep original code)
        return {
            "border_color": self.diamond_white,
            "fg_color": self.jewel_blue,
            "hover_color": self.gold_accent,
            "text_color": self.diamond_white
        }

    def _build_ui(self):
        # (Keep original code, BUT comment out the slider creation and DnD bindings)
        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(1, weight=1)
        customtkinter.CTkLabel(self, text="Videous Chef - Ensemble v5.4/5.5", font=self.header_font, text_color=self.gold_accent).grid(row=0, column=0, columnspan=2, pady=(15, 10), sticky="ew")
        config_outer_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15)
        config_outer_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nsew")
        config_outer_frame.grid_rowconfigure(0, weight=1)
        config_outer_frame.grid_columnconfigure(0, weight=1)
        self.tab_view = customtkinter.CTkTabview(
            config_outer_frame,
            fg_color=self.deep_black,
            segmented_button_fg_color=self.deep_black,
            segmented_button_selected_color=self.jewel_blue,
            segmented_button_selected_hover_color=self.gold_accent,
            segmented_button_unselected_color="#333",
            segmented_button_unselected_hover_color="#555",
            text_color=self.diamond_white,
            border_color=self.diamond_white,
            border_width=2
        )
        self.tab_view.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.tab_view.add("Shared")
        self.tab_view.add("Ensemble Greedy")
        self.tab_view.add("Physics MC (V4)")
        self.tab_view.add("Render Settings")
        self.tab_view._segmented_button.configure(font=self.tab_font)
        # Create frames, but don't populate with sliders yet
        self.shared_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Shared"), fg_color="transparent")
        self.shared_tab_frame.pack(expand=True, fill="both")
        self.ensemble_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Ensemble Greedy"), fg_color="transparent")
        self.ensemble_tab_frame.pack(expand=True, fill="both")
        self.physics_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Physics MC (V4)"), fg_color="transparent")
        self.physics_tab_frame.pack(expand=True, fill="both")
        self.render_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Render Settings"), fg_color="transparent")
        self.render_tab_frame.pack(expand=True, fill="both")

        # --- DISABLED CALL TO SLIDER CREATION ---
        self._create_config_sliders()
        #logger.info("Skipping slider creation for debugging.")
        # --- END DISABLED CALL ---

        # Right Column: Files
        files_outer_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        files_outer_frame.grid(row=1, column=1, padx=(5, 10), pady=5, sticky="nsew")
        files_outer_frame.grid_rowconfigure(1, weight=1)
        files_outer_frame.grid_columnconfigure(0, weight=1)
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
        video_files_frame = customtkinter.CTkFrame(files_outer_frame, fg_color=self.deep_black, border_color=self.diamond_white, border_width=2, corner_radius=10)
        video_files_frame.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")
        video_files_frame.grid_rowconfigure(1, weight=1)
        video_files_frame.grid_columnconfigure(0, weight=1)
        customtkinter.CTkLabel(video_files_frame, text="3. Source Videos (The Ingredients)", anchor="w", font=self.label_font, text_color=self.diamond_white).grid(row=0, column=0, columnspan=2, pady=(5, 2), padx=10, sticky="ew")
        list_frame = Frame(video_files_frame, bg=self.deep_black, highlightbackground=self.diamond_white, highlightthickness=1)
        list_frame.grid(row=1, column=0, columnspan=2, pady=5, padx=10, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        self.video_listbox = Listbox(
            list_frame,
            selectmode="multiple",
            bg=self.deep_black,
            fg=self.diamond_white,
            borderwidth=0,
            highlightthickness=0,
            font=("Garamond", 12),
            selectbackground=self.jewel_blue,
            selectforeground=self.gold_accent
        )
        self.video_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = Scrollbar(list_frame, orient="vertical", command=self.video_listbox.yview, background=self.deep_black, troughcolor=self.jewel_blue)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.video_listbox.configure(yscrollcommand=scrollbar.set)

        # --- DISABLED DRAG AND DROP BINDING ---
        if TKINTERDND_AVAILABLE:
            # try:
            #    if hasattr(self.video_listbox, 'drop_target_register'):
            #         # self.video_listbox.drop_target_register(DND_FILES) # COMMENTED OUT
            #         # self.video_listbox.dnd_bind('<<Drop>>', self._handle_drop) # COMMENTED OUT
            #         logger.info("DnD binding temporarily disabled for debugging.")
            #    else: logger.warning("Listbox does not support drop_target_register.")
            # except Exception as dnd_bind_err: logger.error(f"Failed bind DnD: {dnd_bind_err}")
            logger.warning("Drag-and-drop binding disabled for debugging.") # Log that it's off
        else:
            logger.warning("Drag-and-drop library (tkinterdnd2) not available.")
        # --- END DISABLED DRAG AND DROP ---

        list_button_frame = customtkinter.CTkFrame(video_files_frame, fg_color="transparent")
        list_button_frame.grid(row=2, column=0, columnspan=2, pady=(5, 10), padx=10, sticky="ew")
        list_button_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.add_button = customtkinter.CTkButton(list_button_frame, text="Add", width=70, font=self.button_font, command=self._add_videos_manual, **self._button_styles())
        self.add_button.grid(row=0, column=0, padx=5, sticky="ew")
        self.remove_button = customtkinter.CTkButton(list_button_frame, text="Remove", width=70, font=self.button_font, command=self._remove_selected_videos, **self._button_styles())
        self.remove_button.grid(row=0, column=1, padx=5, sticky="ew")
        self.clear_button = customtkinter.CTkButton(list_button_frame, text="Clear", width=70, font=self.button_font, command=self._clear_video_list, **self._button_styles())
        self.clear_button.grid(row=0, column=2, padx=5, sticky="ew")
        # Bottom Frame: Mode Select & Action
        self.bottom_control_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15)
        self.bottom_control_frame.grid(row=2, column=0, columnspan=2, pady=(5, 5), padx=10, sticky="ew")
        self.bottom_control_frame.grid_columnconfigure(0, weight=1)
        self.bottom_control_frame.grid_columnconfigure(1, weight=1)
        mode_inner_frame = customtkinter.CTkFrame(self.bottom_control_frame, fg_color="transparent")
        mode_inner_frame.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="ew")
        customtkinter.CTkLabel(mode_inner_frame, text="Sequencing Mode:", font=self.mode_font, text_color=self.gold_accent).pack(side="left", padx=(5, 10))

        self.mode_var = self.slider_vars.get('sequencing_mode')
        if not isinstance(self.mode_var, tkinter.StringVar):
            logger.error("Sequencing mode Tkinter variable invalid. Recreating.")
            self.mode_var = tkinter.StringVar(value="Greedy Heuristic")

        self.mode_selector = customtkinter.CTkSegmentedButton(
            mode_inner_frame,
            values=["Greedy Heuristic", "Physics Pareto MC"],
            variable=self.mode_var,
            font=self.button_font,
            selected_color=self.jewel_blue,
            selected_hover_color=self.gold_accent,
            unselected_color="#333",
            unselected_hover_color="#555",
            text_color=self.diamond_white
            # command=self._mode_changed # Keep commented out
        )
        self.mode_selector.pack(side="left", expand=True, fill="x")
        self.run_button = customtkinter.CTkButton(
            self.bottom_control_frame,
            text="4. Compose Video Remix",
            height=45,
            font=customtkinter.CTkFont(family="Garamond", size=16, weight="bold"),
            command=self._start_processing,
            **self._button_styles(border_width=2)
        )
        self.run_button.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="e")
        # Status Label & Footer
        self.status_label = customtkinter.CTkLabel(
            self,
            text="Ready for Chef's command (Ensemble v5.4/5.5).",
            anchor="w",
            font=self.button_font,
            text_color=self.diamond_white,
            wraplength=900
        )
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(0, 5), padx=20, sticky="ew")
        customtkinter.CTkLabel(self, text="Videous Chef v5.4/5.5 - Ensemble Edition", font=self.small_font, text_color=self.gold_accent).grid(row=4, column=1, pady=5, padx=10, sticky="se")

    def _create_config_sliders(self):
        print("Starting _create_config_sliders")
        for key in self.slider_vars:
            print(f"Creating widget for: {key}")
            # Example widget creation (adjust based on your actual code)
            slider = customtkinter.CTkSlider(master=self.shared_tab_frame, variable=self.slider_vars[key])
            slider.grid(row=some_row, column=0, sticky="ew")
            print(f"Added slider for {key} to {slider.master}")
        print("Finished _create_config_sliders")

    
        logger.info("UI sliders and controls created.") # Log moved to end

    def _create_single_slider(self, parent, label_text, variable, from_val, to_val, steps, format_str="{:.2f}"):
        # (Keep original code with safety checks)
        row = customtkinter.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4, padx=5)
        customtkinter.CTkLabel(row, text=label_text, width=300, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5, 5))
        if variable is None:
            customtkinter.CTkLabel(row, text="Var Error", font=self.small_font, text_color="orange").pack(side="left", fill="x", expand=True, padx=5)
            return
        current_val = 0.0
        try: current_val = variable.get()
        except Exception as e: logger.warning(f"Could not get initial value for {label_text} slider: {e}")
        val_lab = customtkinter.CTkLabel(row, text=format_str.format(current_val), width=70, anchor="e", font=self.button_font, text_color=self.gold_accent)
        val_lab.pack(side="right", padx=(5, 5))
        is_int = isinstance(variable, tkinter.IntVar)
        num_steps = int(steps) if steps is not None and steps > 0 else None
        def safe_update_label(v, lbl=val_lab, fmt=format_str, var=variable):
            try: lbl.configure(text=fmt.format(int(round(float(v))))) if is_int else lbl.configure(text=fmt.format(float(v)))
            except tkinter.TclError: pass
            except Exception as le: logger.warning(f"Error updating slider label {label_text}: {le}")
        cmd_lambda = safe_update_label
        slider = customtkinter.CTkSlider(row, variable=variable, from_=from_val, to=to_val, number_of_steps=num_steps, command=cmd_lambda, progress_color=self.gold_accent, button_color=self.diamond_white, button_hover_color=self.gold_accent, fg_color=self.jewel_blue)
        slider.pack(side="left", fill="x", expand=True, padx=5)

    # --- Keep ALL other methods unchanged ---
    def _mode_changed(self, value): logger.info(f"Sequencing mode changed to: {value}"); self.status_label.configure(text=f"Mode set to: {value}. Ready.", text_color=self.diamond_white); # ... (rest of original) ...
    def _select_beat_track(self): pass # Keep original
    def _handle_drop(self, event): pass # Keep original
    def _add_videos_manual(self): pass # Keep original
    def _remove_selected_videos(self): pass # Keep original
    def _clear_video_list(self): pass # Keep original
    def _get_analysis_config(self) -> AnalysisConfig: pass # Keep updated version from previous answer
    def _get_render_config(self) -> RenderConfig: pass # Keep updated version from previous answer
    def _set_ui_processing_state(self, processing: bool): pass # Keep original
    def _start_processing(self): pass # Keep original
    def _generate_audio_cache_path(self, audio_path: str) -> str: pass # Keep original
    def _analyze_master_audio(self): pass # Keep original
    def _start_parallel_video_analysis(self): pass # Keep original
    def _run_parallel_video_analysis_pool(self): pass # Keep original
    def _check_analysis_status(self): pass # Keep original
    def _schedule_sequence_build(self): pass # Keep original
    def _run_post_analysis_workflow(self): pass # Keep original
    def on_closing(self): pass # Keep original
    def shutdown_executor(self): pass # Keep original

# === End of VideousApp Class ===

# ========================================================================
#                      REQUIREMENTS.TXT Block
# ========================================================================
"""
# requirements.txt for Videous Chef - Ensemble Edition v5.3 (Optimized & Fixed)

# Core UI & Analysis
customtkinter>=5.2.0,<6.0.0
opencv-python>=4.6.0,<5.0.0
numpy>=1.21.0,<2.0.0
scipy>=1.8.0
matplotlib>=3.5.0
tkinterdnd2-universal>=2.1.0 # For Drag & Drop
tqdm>=4.60.0                 # For progress bars

# Video Reading & Rendering
moviepy>=1.0.3

# Audio Core & Fallbacks
soundfile>=0.11.0,<0.13.0     # Robust audio loading fallback
torchaudio>=0.12.0             # Primary audio backend (check version for MPS/CUDA support)
librosa>=0.9.0,<0.11.0         # For Mel spec helper & audio analysis fallbacks

# Ensemble - AI / ML Features
torch>=1.12.0                  # Base PyTorch (check version for MPS/CUDA support)
torchvision>=0.13.0            # Often needed with PyTorch
mediapipe>=0.10.0,<0.11.0      # For Face Mesh, Pose
transformers[torch]>=4.25.0    # For CLIP, Whisper, SER models
sentence-transformers>=2.2.0   # For text embeddings
huggingface_hub>=0.10.0        # For downloading models (like SyncNet)
scikit-image>=0.19.0           # For SyncNet mouth crop helper

# MiDaS Dependency (Needed for Depth features in Base Heuristic / Physics MC)
timm>=0.6.0

# Optional Ensemble Features (Install Manually If Desired)
# audioflux>=0.1.8       # (Faster audio analysis - check GitHub for install)
# demucs>=4.0.0          # (Source separation for MRISE - check GitHub for install)

# Optional - Face Restoration (Install Manually If Desired)
# gfpgan>=1.3.8
# basicsr>=1.4.2         # Dependency for GFPGAN
# facexlib>=0.3.0        # Dependency for GFPGAN
# realesrgan>=0.3.0      # Optional dependency for GFPGAN

# --- Notes ---
# 1. FFmpeg: Ensure FFmpeg executable is installed and accessible in your system's PATH.
# 2. GPU: Highly recommended (NVIDIA CUDA or Apple Silicon MPS). Verify PyTorch compatibility.
# 3. Environment: Use a Python virtual environment (e.g., venv, conda).
# 4. SyncNet Model: Verify the embedded SyncNet class definition matches the weights used (ByteDance/LatentSync-1.5/syncnet.pth).
# 5. GFPGAN Model: If using GFPGAN, manually download the .pth model file (e.g., GFPGANv1.4.pth) and place it in 'experiments/pretrained_models'.
"""
# ========================================================================
#                       APPLICATION ENTRY POINT
# ========================================================================
# Dummy definitions for optional dependencies flags if not defined elsewhere
TRANSFORMERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
AUDIOFLUX_AVAILABLE = False
DEMUCS_AVAILABLE = False
HUGGINGFACE_HUB_AVAILABLE = False
SKIMAGE_AVAILABLE = False
LIBROSA_AVAILABLE = False
TIMM_AVAILABLE = False
PYSCENEDETECT_AVAILABLE = True
GFPGAN_AVAILABLE = True
MOVIEPY_AVAILABLE = False
TKINTER_AVAILABLE = False
CTK_AVAILABLE = False
TORCH_AVAILABLE = False
TORCHAUDIO_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
SOUNDFILE_AVAILABLE = False
MULTIPROCESSING_AVAILABLE = False
TKINTERDND_AVAILABLE = False  # Added for tkinterdnd2 (drag-and-drop)

# Import statements assumed to be at the top of the script
import sys
import os
import time
import logging
import subprocess
import tkinter
from tkinter import messagebox

try:
    import multiprocessing
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    pass

try:
    import moviepy
    MOVIEPY_AVAILABLE = True
except ImportError:
    pass

try:
    import tkinter
    TKINTER_AVAILABLE = True
except ImportError:
    pass

try:
    import customtkinter
    CTK_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    pass

try:
    import mediapipe
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:
    pass

try:
    import tkinterdnd2
    TKINTERDND_AVAILABLE = True
except ImportError:
    pass

if __name__ == "__main__":
    # --- Multiprocessing Setup ---
    if MULTIPROCESSING_AVAILABLE:
        multiprocessing.freeze_support()  # For PyInstaller/cx_Freeze
        try:
            default_method = multiprocessing.get_start_method(allow_none=True)
            # Force spawn on non-Windows if available, otherwise use default
            if sys.platform != 'win32':  # Non-Windows
                if 'spawn' in multiprocessing.get_all_start_methods():
                    multiprocessing.set_start_method('spawn', force=True)
                    print("INFO: Set MP start method to 'spawn'.")
                else:
                    print(f"WARNING: 'spawn' MP method unavailable. Using default: {default_method}.")
            else:  # Windows (spawn is default, but good practice to check/set)
                print(f"INFO: Using MP start method '{default_method}' on Windows.")
                if 'spawn' not in multiprocessing.get_all_start_methods():
                    print(f"WARNING: 'spawn' MP method unexpectedly unavailable on Windows?")
        except Exception as mp_setup_e:
            print(f"WARNING: MP start method setup error: {mp_setup_e}. Using default.")
    else:
        print("WARNING: Multiprocessing module not loaded, parallel disabled.")

    # --- Logging Setup ---
    print("--- Videous Chef - Ensemble Edition v5.4/5.5 ---")
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s [%(threadName)s]')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = None
    try:
        log_dir = "logs_v5.4"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"videous_chef_ensemble_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)
        log_setup_message = f"Logging to console (INFO+) and file (DEBUG+): {log_file}"
    except Exception as log_setup_e:
        log_setup_message = f"Failed file logging: {log_setup_e}. Console only."
        file_handler = None
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    logger = root_logger
    logger.info(log_setup_message)

    # --- Re-check flags after imports ---
    try:
        import transformers
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        pass
    try:
        import sentence_transformers
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        pass
    try:
        import audioflux
        AUDIOFLUX_AVAILABLE = True
    except ImportError:
        pass
    try:
        import demucs
        DEMUCS_AVAILABLE = True
    except ImportError:
        pass
    try:
        import huggingface_hub
        HUGGINGFACE_HUB_AVAILABLE = True
    except ImportError:
        pass
    try:
        import skimage
        SKIMAGE_AVAILABLE = True
    except ImportError:
        pass
    try:
        import librosa
        LIBROSA_AVAILABLE = True
    except ImportError:
        pass
    try:
        import timm
        TIMM_AVAILABLE = True
    except ImportError:
        pass
    try:
        import pyscenedetect
        PYSCENEDETECT_AVAILABLE = True
    except ImportError:
        pass
    try:
        from gfpgan import GFPGANer
        GFPGAN_AVAILABLE = True
    except ImportError:
        pass

    try:
        # --- STARTUP CHECKS ---
        logger.info("Checking essential dependencies...")
        essential_deps = {"numpy": "numpy", "opencv-python": "cv2"}
        missing_critical = []
        if not TKINTER_AVAILABLE:
            missing_critical.append("tkinter")
        if not CTK_AVAILABLE:
            missing_critical.append("customtkinter")
        if not MOVIEPY_AVAILABLE:
            missing_critical.append("moviepy")
        if not TORCH_AVAILABLE:
            missing_critical.append("torch")
        if not TORCHAUDIO_AVAILABLE:
            missing_critical.append("torchaudio")
        if not MEDIAPIPE_AVAILABLE:
            missing_critical.append("mediapipe")
        if not SOUNDFILE_AVAILABLE:
            missing_critical.append("soundfile")
        for pkg, mod in essential_deps.items():
            try:
                __import__(mod)
                logger.debug(f"  [OK] {mod}")
            except ImportError:
                logger.critical(f"  [FAIL] {mod} ({pkg})")
                missing_critical.append(pkg)
        if missing_critical:
            err_msg = f"Critical missing: {', '.join(missing_critical)}\nInstall requirements.\nExiting."
            try:
                root = tkinter.Tk()
                root.withdraw()
                messagebox.showerror("Error", err_msg)
                root.destroy()
            except:
                print(f"\nFATAL:\n{err_msg}\n")
            sys.exit(1)
        logger.info("Essential dependencies OK.")

        # FFmpeg Check
        try:
            ffmpeg_cmd = "ffmpeg"
            result = subprocess.run([ffmpeg_cmd, "-version"], capture_output=True, text=True, check=False, timeout=5, encoding='utf-8')
            if result.returncode != 0 or "ffmpeg version" not in result.stdout.lower():
                raise FileNotFoundError(f"FFmpeg check failed (cmd: '{ffmpeg_cmd}').")
            logger.info(f"FFmpeg check OK (via '{ffmpeg_cmd}').")
        except FileNotFoundError as fnf_err:
            err_msg = f"CRITICAL: FFmpeg not found/failed.\nDetails: {fnf_err}\nInstall FFmpeg & ensure PATH.\nExiting."
            logger.critical(err_msg)
            try:
                root = tkinter.Tk()
                root.withdraw()
                messagebox.showerror("Error", err_msg)
                root.destroy()
            except:
                print(f"\nFATAL:\n{err_msg}\n")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg check timed out.")
        except Exception as ff_e:
            logger.error(f"FFmpeg check error: {ff_e}")

        # PyTorch Backend Check
        try:
            logger.info(f"PyTorch version: {torch.__version__}")
            # get_device() assumed to be defined elsewhere
        except Exception as pt_e:
            logger.warning(f"PyTorch check error: {pt_e}")

        # --- Log Optional Dependency Status ---
        logger.info("Optional dependencies status:")
        logger.info(f"  transformers:          {'OK' if TRANSFORMERS_AVAILABLE else 'MISSING (LVRE disabled)'}")
        logger.info(f"  sentence-transformers: {'OK' if SENTENCE_TRANSFORMERS_AVAILABLE else 'MISSING (LVRE text embed disabled)'}")
        logger.info(f"  audioflux:             {'OK' if AUDIOFLUX_AVAILABLE else 'MISSING (Using Librosa fallback)'}")
        logger.info(f"  demucs:                {'OK' if DEMUCS_AVAILABLE else 'MISSING (MRISE sep disabled)'}")
        logger.info(f"  huggingface_hub:       {'OK' if HUGGINGFACE_HUB_AVAILABLE else 'MISSING (No auto-download)'}")
        logger.info(f"  scikit-image:          {'OK' if SKIMAGE_AVAILABLE else 'MISSING (SyncNet disabled)'}")
        logger.info(f"  librosa:               {'OK' if LIBROSA_AVAILABLE else 'MISSING (Audio fallbacks disabled)'}")
        logger.info(f"  timm:                  {'OK' if TIMM_AVAILABLE else 'MISSING (MiDaS disabled)'}")
        logger.info(f"  pyscenedetect:         {'OK' if PYSCENEDETECT_AVAILABLE else 'MISSING (Scene detect disabled)'}")
        logger.info(f"  gfpgan:                {'OK' if GFPGAN_AVAILABLE else 'MISSING (Face enhance disabled)'}")
        logger.info(f"  tkinterdnd2:           {'OK' if TKINTERDND_AVAILABLE else 'MISSING (Drag-and-drop disabled)'}")

        # --- Run App ---
        logger.info("Initializing Application UI (Ensemble v5.4/5.5)...")
        app = VideousApp()  # Assumed to be defined elsewhere
        logger.info("Starting Tkinter main loop...")
        app.mainloop()

    except SystemExit as se:
        logger.warning(f"App exited during startup/shutdown (Code: {se.code}).")
    except Exception as e:
        logger.critical(f"!!! UNHANDLED STARTUP ERROR !!!", exc_info=True)
        try:
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showerror("Startup Error", f"App failed start:\n{type(e).__name__}: {e}\nCheck logs.")
            root.destroy()
        except Exception as msg_err:
            print(f"\n!!! CRITICAL STARTUP ERROR: {e} !!!\nGUI message err: {msg_err}")
    logger.info("--- Videous Chef session ended ---")