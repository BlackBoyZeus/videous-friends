
# -*- coding: utf-8 -*-
# ========================================================================
#          Videous Chef - Ensemble Edition v5.4 (Integrated)
# ========================================================================
#                       IMPORTS (Ensemble Ready)
# ========================================================================

# --- Core Python & UI ---
import tkinter
from tkinter import filedialog, Listbox, Scrollbar, END, MULTIPLE, Frame, messagebox, BooleanVar, IntVar, PhotoImage
import customtkinter
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
import tracemalloc # For memory profiling (optional)
import inspect # For checking class definitions
import hashlib # For file hashing (caching)
import time
import math
import pathlib # For path handling
from functools import lru_cache # For caching models like MiDaS
from huggingface_hub import hf_hub_download

# --- Core Analysis Libs ---
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import scipy.signal
import scipy.stats

# Optional imports with flags
try:
    import pyscenedetect
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False
    
# --- Audio Processing (Primary & Fallback) ---
try:
    import soundfile # For robust audio loading fallback
    SOUNDFILE_AVAILABLE = True
except ImportError:
    print("ERROR: soundfile library not found. Install with 'pip install soundfile'")
    SOUNDFILE_AVAILABLE = False; # sys.exit(1) # Soft fail for now

try:
    import torchaudio # Primary audio loading backend
    import torchaudio.transforms as T # For resampling, transforms
    TORCHAUDIO_AVAILABLE = True
    print("INFO: `torchaudio` loaded successfully.")
except ImportError:
    print("ERROR: torchaudio library not found. Install with 'pip install torchaudio'")
    TORCHAUDIO_AVAILABLE = False; sys.exit(1) # Critical dependency

try:
    # Attempt import first, flag set later based on actual function calls
    import librosa
    # Force SciPy backend if Librosa is used
    os.environ['LIBROSA_CORE'] = 'scipy'
    LIBROSA_AVAILABLE = True
    print("INFO: `librosa` loaded. Used for Mel spec & fallbacks.")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("INFO: `librosa` not found. Mel spectrogram (SyncNet) & audio fallbacks disabled.")

# --- Video Processing ---
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import VideoClip # Needed for make_frame render
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("ERROR: moviepy library not found. Install with 'pip install moviepy'")
    MOVIEPY_AVAILABLE = False; sys.exit(1) # Critical dependency

# --- UI & Utilities ---
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKINTERDND2_AVAILABLE = True
except ImportError:
    print("ERROR: tkinterdnd2-universal not found. Drag & Drop disabled. Install with 'pip install tkinterdnd2-universal'")
    class TkinterDnD: # Dummy class for graceful failure
        class DnDWrapper: pass
        _require = lambda x: None # Dummy method
    TKINTERDND2_AVAILABLE = False
try:
    import multiprocessing
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    print("WARNING: multiprocessing module not found, parallel processing disabled.")
    MULTIPROCESSING_AVAILABLE = False; multiprocessing = None
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False; tqdm = None; print("INFO: tqdm not found, progress bars disabled.")
import matplotlib # For backend setting only
try:
    matplotlib.use('Agg') # Set backend before importing pyplot
except Exception as mpl_err:
     print(f"Warning: Could not set Matplotlib backend 'Agg': {mpl_err}")


# --- Optional Ensemble Feature Libs ---
TRANSFORMERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
AUDIOFLUX_AVAILABLE = False
DEMUCS_AVAILABLE = False
HUGGINGFACE_HUB_AVAILABLE = False
SKIMAGE_AVAILABLE = False
TIMM_AVAILABLE = False
PYSCENEDETECT_AVAILABLE = False
GFPGAN_AVAILABLE = False
LIBROSA_AVAILABLE = False
TIMM_AVAILABLE = False

# Update existing optional flags if not already set correctly
TRANSFORMERS_AVAILABLE = 'transformers' in sys.modules
SENTENCE_TRANSFORMERS_AVAILABLE = 'sentence_transformers' in sys.modules
AUDIOFLUX_AVAILABLE = 'audioflux' in sys.modules
DEMUCS_AVAILABLE = 'demucs' in sys.modules
HUGGINGFACE_HUB_AVAILABLE = 'huggingface_hub' in sys.modules
SKIMAGE_AVAILABLE = 'skimage' in sys.modules
LIBROSA_AVAILABLE = 'librosa' in sys.modules
TIMM_AVAILABLE = 'timm' in sys.modules

# --- Attempt to load optional libraries and set flags ---
try:
    import transformers
    from transformers import pipeline, CLIPProcessor, CLIPModel, WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, AutoModelForAudioClassification, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
    print("INFO: `transformers` loaded successfully (for LVRE: CLIP, Whisper, SER).")
except ImportError: print("INFO: `transformers` not found. LVRE features disabled.")
try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("INFO: `sentence-transformers` loaded successfully (for LVRE: Text Embeddings).")
except ImportError: print("INFO: `sentence-transformers` not found. LVRE semantic features disabled.")
try:
    import audioflux as af
    AUDIOFLUX_AVAILABLE = True
    print("INFO: `audioflux` loaded successfully (for advanced audio analysis: Beats, Onsets, Segments).")
except ImportError: print("INFO: `audioflux` not found. Using Librosa/SciPy fallbacks for some audio analysis.")
try:
    import demucs.separate # Check if importable
    DEMUCS_AVAILABLE = True
    print("INFO: `demucs` found. Source separation (MRISE) can be enabled (if configured).")
except ImportError: print("INFO: `demucs` not found. Source separation for MRISE disabled.")
try:
    import huggingface_hub
    from huggingface_hub import hf_hub_download
    HUGGINGFACE_HUB_AVAILABLE = True
    print("INFO: `huggingface_hub` loaded successfully (for auto-downloading models like SyncNet).")
except ImportError: print("INFO: `huggingface_hub` not found. Cannot auto-download models.")
try:
    import skimage.transform
    import skimage.metrics
    from skimage.metrics import structural_similarity # Example import
    SKIMAGE_AVAILABLE = True
    print("INFO: `scikit-image` loaded successfully (required for SyncNet mouth crop helper).")
except ImportError: print("INFO: `scikit-image` not found. SyncNet PWRC scoring disabled.")
try:
    import timm
    TIMM_AVAILABLE = True
    print("INFO: `timm` library loaded successfully (required for MiDaS).")
except ImportError: print("INFO: `timm` library not found. MiDaS depth estimation disabled.")
try:
    import scenedetect
    # Conditional import of specific modules
    if 'scenedetect' in sys.modules:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector # Updated import path
        # from scenedetect.scene_manager import save_images # If saving scene thumbnails
        PYSCENEDETECT_AVAILABLE = True
        print("INFO: `PySceneDetect` loaded successfully. Scene detection enabled (if configured).")
    else:
         raise ImportError # Force failure if base import succeeded but specifics failed
except ImportError: print("INFO: `PySceneDetect` not found or failed to import submodules. Scene detection features disabled.")
try:
    # Check GFPGAN dependencies carefully
    import gfpgan
    from gfpgan import GFPGANer
    # Optionally check other dependencies like basicsr, facexlib here if needed
    GFPGAN_AVAILABLE = True
    print("INFO: `GFPGANer` loaded successfully. Face enhancement enabled (if configured).")
except ImportError: print("INFO: `gfpgan` or its dependencies not found. Face enhancement disabled. Install with 'pip install gfpgan'.")


# --- MediaPipe Solutions ---
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("ERROR: Failed to import MediaPipe solutions. Ensure mediapipe is installed ('pip install mediapipe').")
    MEDIAPIPE_AVAILABLE = False; # sys.exit(1) # Soft fail for now
except AttributeError as mp_err:
    print(f"ERROR: Failed to load MediaPipe solutions: {mp_err}. Ensure mediapipe is installed correctly ('pip install mediapipe').")
    MEDIAPIPE_AVAILABLE = False; sys.exit(1) # Critical failure


# --- Constants ---
MIN_POTENTIAL_CLIP_DURATION_FRAMES = 5 # Absolute minimum frames for any processing step
TARGET_SR_TORCHAUDIO = 16000 # Preferred Sample Rate for analysis
LATENTSYNC_MAX_FRAMES = 60 # Limit frames processed by SyncNet per segment for performance
DEFAULT_SEQUENCE_HISTORY_LENGTH = 5 # For SAAPV predictability penalty
ENABLE_PROFILING = True # <<< Set to True to enable simple timing logs >>>
CACHE_VERSION = "v5.4" # Increment this if cache format changes significantly

# Default normalization constants (can be overridden by config)
DEFAULT_NORM_MAX_RMS = 0.5; DEFAULT_NORM_MAX_ONSET = 5.0; DEFAULT_NORM_MAX_CHROMA_VAR = 0.1
DEFAULT_NORM_MAX_POSE_KINETIC = 50.0; DEFAULT_NORM_MAX_VISUAL_FLOW = 50.0; DEFAULT_NORM_MAX_FACE_SIZE = 1.0
DEFAULT_NORM_MAX_DEPTH_VARIANCE = 0.15 # V4 compatibility
DEFAULT_NORM_MAX_JERK = 100000.0 # V4 compatibility

# --- Caches & Logging ---
_pytorch_model_cache = {}; _pytorch_processor_cache = {} # General cache for PyTorch models/processors
logger = logging.getLogger(__name__) # Get logger for the current module (configured in __main__)

# ========================================================================
#              <<< SYNC NET MODEL DEFINITION (EMBEDDED) >>>
# ========================================================================
# (VERIFY this matches the ByteDance/LatentSync-1.5/syncnet.pth architecture)
class Conv3dRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(Conv3dRelu, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.conv(x))

class Conv2dRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.conv(x))

class SyncNet(nn.Module):
    # Note: This architecture needs careful verification against the specific
    # syncnet.pth file from ByteDance/LatentSync-1.5. It might differ.
    def __init__(self):
        super(SyncNet, self).__init__()
        # --- Audio Stream (from common architectures, needs verification) ---
        self.audio_stream = nn.Sequential(
            Conv2dRelu(1, 64, kernel_size=3, stride=1, padding=1),
            Conv2dRelu(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)), # (N, 64, 80, T) -> (N, 64, 80, T/2)
            Conv2dRelu(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2dRelu(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)), # (N, 128, 40, T/4) -> (N, 128, 19, T/8) ? Check padding
            Conv2dRelu(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2dRelu(256, 256, kernel_size=3, stride=1, padding=1),
            Conv2dRelu(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)), # (N, 256, 9, T/16) -> (N, 256, 4, T/32) ?
            Conv2dRelu(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2dRelu(512, 512, kernel_size=3, stride=1, padding=1),
            # Assuming input Mel is 80 bins, output dim after pools needs verification
            # Let's assume pool -> (N, 512, 4, T') -> Final Conv(kernel=4) -> (N, 512, 1, T'')
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(1, 1), padding=0), # Needs verification
            nn.ReLU(inplace=True),
            # Global average pooling or adaptive pooling might be used here in some variants
            nn.AdaptiveAvgPool2d((1, 1)) # -> (N, 512, 1, 1)
        )
        # --- Video Stream (Face Crops - based on common SyncNet, needs verification) ---
        self.video_stream = nn.Sequential(
            # Input (N, 3, 5, H, W) = (N, 3, 5, 112, 112) assumed for LatentSync
            Conv3dRelu(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)), # T=5->1, H/W=112->56
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)), # T=1, H/W=56->28 ? Needs padding check
            Conv3dRelu(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)), # T=1, H/W=28->14
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)), # T=1, H/W=14->7 ?
            Conv3dRelu(256, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)), # T=1, H/W=7->7
            Conv3dRelu(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)), # T=1, H/W=7->7
            # Final Conv to match feature size - needs verification based on output shape
            # Assume (N, 512, 1, 7, 7) -> Final Conv kernel=(1, 7, 7) -> (N, 512, 1, 1, 1)
            nn.Conv3d(512, 512, kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, audio_sequences, video_sequences):
        """
        Args:
            audio_sequences: (N, 1, n_mels, T_audio) - Batch of Mel spectrogram chunks
            video_sequences: (N, C, T_video, H, W) - Batch of face crop sequences (e.g., N, 3, 5, 112, 112)
        Returns:
            audio_embedding: (N, embed_dim)
            video_embedding: (N, embed_dim)
        """
        audio_embedding = self.audio_stream(audio_sequences) # Shape: (N, 512, 1, 1)
        video_embedding = self.video_stream(video_sequences) # Shape: (N, 512, 1, 1, 1)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1) # Flatten to (N, 512)
        video_embedding = video_embedding.view(video_embedding.size(0), -1) # Flatten to (N, 512)

        # L2 Normalize the embeddings (standard practice for contrastive loss / similarity)
        audio_embedding_norm = F.normalize(audio_embedding, p=2, dim=1)
        video_embedding_norm = F.normalize(video_embedding, p=2, dim=1)

        # Original LatentSync might return a confidence score directly after distance calc.
        # Here we return the embeddings, the caller calculates similarity/distance.
        # If the loaded model calculates confidence internally, adapt this forward pass.
        # For now, assume embeddings are returned.
        return audio_embedding_norm, video_embedding_norm

# ========================================================================
#                       DATA CLASSES (Ensemble v5.4)
# ========================================================================
@dataclass
class AnalysisConfig:
    """Configuration for the analysis phase (Ensemble v5.4)."""
    # --- Core Params ---
    min_sequence_clip_duration: float = 0.75
    max_sequence_clip_duration: float = 5.0
    min_potential_clip_duration_sec: float = 0.4 # Min duration for a *potential* clip during analysis
    resolution_height: int = 256 # Target analysis resolution
    resolution_width: int = 256 # Target analysis resolution
    save_analysis_data: bool = True # Saves persistent audio cache (JSON)
    cache_visual_features: bool = True # Saves persistent visual feature cache (NPZ)
    use_scene_detection: bool = True    # New: Enable PySceneDetect

    # --- Normalization Defaults ---
    norm_max_rms: float = DEFAULT_NORM_MAX_RMS
    norm_max_onset: float = DEFAULT_NORM_MAX_ONSET
    norm_max_pose_kinetic: float = DEFAULT_NORM_MAX_POSE_KINETIC
    norm_max_visual_flow: float = DEFAULT_NORM_MAX_VISUAL_FLOW
    norm_max_face_size: float = DEFAULT_NORM_MAX_FACE_SIZE
    # Physics MC / V4 compatibility
    norm_max_depth_variance: float = DEFAULT_NORM_MAX_DEPTH_VARIANCE
    norm_max_jerk: float = DEFAULT_NORM_MAX_JERK

    # --- Detection Thresholds ---
    min_face_confidence: float = 0.5
    min_pose_confidence: float = 0.5
    model_complexity: int = 1 # Pose model complexity (0, 1, 2)
    mouth_open_threshold: float = 0.05 # Base Heuristic V4 face feature

    # --- Audio Analysis ---
    target_sr_audio: int = TARGET_SR_TORCHAUDIO
    use_dl_beat_tracker: bool = True # Prioritize AudioFlux/DL model for beats if available

    # --- Sequencing Mode ---
    sequencing_mode: str = "Greedy Heuristic" # or "Physics Pareto MC"

    # --- Feature Flags (Ensemble Components) ---
    use_latent_sync: bool = True # <<< PWRC: Enable SyncNet Lip Sync Scoring (Requires SyncNet model)
    use_lvre_features: bool = True # <<< LVRE: Enable Lyrical/Visual/Emotion Features (Needs Transformers, SBERT)
    use_saapv_adaptation: bool = True # <<< SAAPV: Enable Style Analysis & Pacing Adaptation
    use_mrise_sync: bool = True # <<< MRISE: Enable Micro-Rhythm Synchronization (Needs AudioFlux/Librosa)
    use_demucs_for_mrise: bool = False # <<< MRISE: Use Demucs for stems (Requires Demucs install)
    use_scene_detection: bool = True # <<< Enable Scene Detection (Requires PySceneDetect)

    # --- SyncNet (PWRC) ---
    syncnet_repo_id: str = "ByteDance/LatentSync-1.5" # <<< VERIFY REPO ID >>>
    syncnet_filename: str = "syncnet.pth" # <<< VERIFY FILENAME >>>
    syncnet_batch_size: int = 16

    # --- LVRE Model Config ---
    whisper_model_name: str = "openai/whisper-tiny" # Faster ASR
    ser_model_name: str = "facebook/wav2vec2-large-robust-ft-emotion-msp-podcast" # Example SER
    text_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2" # Efficient sentence embedder
    vision_embed_model_name: str = "openai/clip-vit-base-patch32" # CLIP model
    lvre_batch_size_text: int = 128
    lvre_batch_size_vision: int = 64
    lvre_batch_size_ser: int = 16

    # --- Render FPS (passed for convenience) ---
    render_fps: int = 30 # Default, will be overridden by RenderConfig

    # --- Scene Detection ---
    scene_detection_threshold: float = 27.0 # PySceneDetect ContentDetector threshold

    # === GREEDY HEURISTIC ENSEMBLE WEIGHTS ===
    # Base Heuristic (V4 Logic - Less Emphasis)
    base_heuristic_weight: float = 0.05 # Reduced weight
    bh_audio_weight: float = 0.3; bh_kinetic_weight: float = 0.25; bh_sharpness_weight: float = 0.1; bh_camera_motion_weight: float = 0.05; bh_face_size_weight: float = 0.1; bh_percussive_weight: float = 0.05; bh_depth_weight: float = 0.0 # Depth off by default

    # Performance Weights (PWRC)
    pwrc_weight: float = 0.30
    pwrc_lipsync_weight: float = 0.7 # Weight for SyncNet score within PWRC
    pwrc_pose_energy_weight: float = 0.3 # Weight for Pose energy vs Audio energy match

    # Energy Flow Weights (HEFM)
    hefm_weight: float = 0.20
    hefm_trend_match_weight: float = 1.0 # Weight for visual trend vs audio trend match

    # Lyrical/Visual/Emotion Weights (LVRE)
    lvre_weight: float = 0.15
    lvre_semantic_weight: float = 0.6 # Weight for lyric-visual semantic similarity
    lvre_emphasis_weight: float = 0.4 # Weight for vocal emphasis boost

    # Style/Pacing/Variety Weights (SAAPV)
    saapv_weight: float = 0.15 # Overall weight for SAAPV contributions
    saapv_predictability_weight: float = 0.4 # Weight of the predictability penalty
    saapv_history_length: int = DEFAULT_SEQUENCE_HISTORY_LENGTH
    saapv_variety_penalty_source: float = 0.15 # Increased slightly
    saapv_variety_penalty_shot: float = 0.15   # Increased slightly
    saapv_variety_penalty_intensity: float = 0.10
    scene_change_penalty: float = 0.20 # Penalty for cutting across detected scenes

    # Micro-Rhythm Weights (MRISE)
    mrise_weight: float = 0.15
    mrise_sync_weight: float = 1.0 # Weight of the micro-beat sync bonus
    mrise_sync_tolerance_factor: float = 1.5 # Multiplier for base tolerance (e.g., 1.5 * frame_duration)

    # Rhythm/Timing Weights (General)
    rhythm_beat_sync_weight: float = 0.1 # Bonus for cutting near a main beat
    rhythm_beat_boost_radius_sec: float = 0.1 # Window around beat for bonus

    # --- Greedy Candidate Selection ---
    candidate_pool_size: int = 15

    # === PHYSICS PARETO MC SPECIFIC (V4 Logic) ===
    score_threshold: float = 0.3 # Legacy segment ID threshold
    fit_weight_velocity: float = 0.3; fit_weight_acceleration: float = 0.3; fit_weight_mood: float = 0.4; fit_sigmoid_steepness: float = 1.0
    objective_weight_rhythm: float = 1.0; objective_weight_mood: float = 1.0; objective_weight_continuity: float = 0.8; objective_weight_variety: float = 0.7; objective_weight_efficiency: float = 0.5
    mc_iterations: int = 500; mood_similarity_variance: float = 0.3; continuity_depth_weight: float = 0.5; variety_repetition_penalty: float = 0.15

@dataclass
class EffectParams: # (Unchanged from v4.7.3)
    type: str = "cut"
    tau: float = 0.0 # Duration
    psi: float = 0.0 # Physical impact proxy
    epsilon: float = 0.0 # Perceptual gain

@dataclass
class RenderConfig:
    """Configuration for the rendering phase."""
    effect_settings: Dict[str, EffectParams] = field(default_factory=lambda: {
        "cut": EffectParams(type="cut"), "fade": EffectParams(type="fade", tau=0.2, psi=0.1, epsilon=0.2),
        "zoom": EffectParams(type="zoom", tau=0.5, psi=0.3, epsilon=0.4), "pan": EffectParams(type="pan", tau=0.5, psi=0.1, epsilon=0.3),
    })
    video_codec: str = 'libx264'; preset: Optional[str] = 'medium'; crf: Optional[int] = 23
    audio_codec: str = 'aac'; audio_bitrate: str = '192k'
    threads: int = max(1, (os.cpu_count() or 2) // 2)
    resolution_width: int = 1920; resolution_height: int = 1080; fps: int = 30
    # --- Optional GFPGAN Enhancement ---
    use_gfpgan_enhance: bool = False # <<< Enable/disable GFPGAN
    gfpgan_model_path: str = "experiments/pretrained_models/GFPGANv1.4.pth" # <<< Default path, adjust or use UI input >>>
    gfpgan_fidelity_weight: float = 0.5 # <<< Control fidelity (0=Realism, 1=Quality)

# ========================================================================
#                       HELPER FUNCTIONS (Loaders, Utils, etc.)
# ========================================================================
def tk_write(tk_string1, parent=None, level="info"):
    """Shows message boxes and logs messages."""
    log_level_map = {"error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO, "debug": logging.DEBUG}
    log_level = log_level_map.get(level.lower(), logging.INFO)
    logger.log(log_level, f"Popup ({level}): {tk_string1}")
    try:
        use_messagebox = False
        # Check if parent is a valid Tk/Toplevel widget before showing messagebox
        if parent and isinstance(parent, (tkinter.Tk, tkinter.Toplevel)) and parent.winfo_exists(): use_messagebox = True
        elif isinstance(parent, customtkinter.CTk) and parent.winfo_exists(): use_messagebox = True # Handle CTk main window
        else: logger.debug("tk_write called without valid parent or parent destroyed.")
        if use_messagebox:
            title = f"Videous Chef - {level.capitalize()}";
            if level == "error": messagebox.showerror(title, tk_string1, parent=parent)
            elif level == "warning": messagebox.showwarning(title, tk_string1, parent=parent)
            else: messagebox.showinfo(title, tk_string1, parent=parent)
    except Exception as e: logger.error(f"tk_write internal error: {e}", exc_info=True); print(f"!! tk_write Error: {e}\n!! Level: {level}\n!! Message: {tk_string1}")

def sigmoid(x, k=1):
    try: x_clamped = np.clip(x * k, -700, 700); return 1 / (1 + np.exp(-x_clamped))
    except OverflowError: logger.warning(f"Sigmoid overflow: x={x}, k={k}."); return 0.0 if x < 0 else 1.0

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    vec1 = np.asarray(vec1, dtype=float); vec2 = np.asarray(vec2, dtype=float)
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    dot_product = np.dot(vec1, vec2); similarity = dot_product / (norm1 * norm2); return np.clip(similarity, -1.0, 1.0)

def calculate_histogram_entropy(frame):
    if frame is None or frame.size == 0: return 0.0
    try:
        if len(frame.shape) == 3 and frame.shape[2] == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2: gray = frame
        else: logger.warning(f"Invalid frame shape for entropy: {frame.shape}"); return 0.0
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]); hist_sum = hist.sum()
        if hist_sum <= 0: return 0.0
        hist_norm = hist.ravel() / hist_sum; hist_norm_nonzero = hist_norm[hist_norm > 0]
        if hist_norm_nonzero.size == 0: return 0.0
        entropy = scipy.stats.entropy(hist_norm_nonzero); return entropy if np.isfinite(entropy) else 0.0
    except cv2.error as cv_err: logger.warning(f"OpenCV error in histogram calc: {cv_err}"); return 0.0
    except Exception as e: logger.warning(f"Hist entropy calculation failed: {e}"); return 0.0

def get_device() -> torch.device:
    """Gets the recommended device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda"); logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps"); logger.info("Using MPS (Apple Silicon GPU) device.")
    else:
        device = torch.device("cpu"); logger.info("Using CPU device.")
    return device

# --- Cached Model/Processor Loaders ---
def get_pytorch_model(cache_key: str, load_func: callable, *args, **kwargs) -> Optional[Tuple[torch.nn.Module, torch.device]]:
    """Loads a PyTorch model using cache. Returns model and device."""
    global _pytorch_model_cache
    if cache_key in _pytorch_model_cache:
        model, device = _pytorch_model_cache[cache_key]; logger.debug(f"Reusing cached PyTorch Model: {cache_key} on {device}"); return model, device
    logger.info(f"Loading PyTorch Model: {cache_key}...");
    try:
        device = get_device(); model = load_func(*args, **kwargs);
        if model is None: raise ValueError("Model loading function returned None")
        if hasattr(model, 'to') and callable(model.to): model.to(device)
        else: logger.warning(f"Model {cache_key} lacks .to() method, cannot move to {device}.")
        if hasattr(model, 'eval') and callable(model.eval): model.eval()
        else: logger.warning(f"Model {cache_key} lacks .eval() method.")
        _pytorch_model_cache[cache_key] = (model, device); logger.info(f"PyTorch Model loaded: {cache_key} to {device}"); return model, device
    except ImportError as imp_err: logger.error(f"ImportError loading Model {cache_key}: {imp_err}. Libs missing?"); return None, None
    except Exception as e: logger.error(f"Failed load Model {cache_key}: {e}", exc_info=True); if cache_key in _pytorch_model_cache: del _pytorch_model_cache[cache_key]; return None, None

def get_pytorch_processor(cache_key: str, load_func: callable, *args, **kwargs) -> Optional[Any]:
    """Loads a PyTorch processor using cache."""
    global _pytorch_processor_cache
    if cache_key in _pytorch_processor_cache: logger.debug(f"Reusing cached PyTorch Processor: {cache_key}"); return _pytorch_processor_cache[cache_key]
    logger.info(f"Loading PyTorch Processor: {cache_key}...");
    try:
        processor = load_func(*args, **kwargs);
        if processor is None: raise ValueError("Processor loading function returned None")
        _pytorch_processor_cache[cache_key] = processor; logger.info(f"PyTorch Processor loaded: {cache_key}"); return processor
    except ImportError as imp_err: logger.error(f"ImportError loading Processor {cache_key}: {imp_err}. Libs missing?"); return None
    except Exception as e: logger.error(f"Failed load Processor {cache_key}: {e}", exc_info=True); if cache_key in _pytorch_processor_cache: del _pytorch_processor_cache[cache_key]; return None

# --- Specific Model Loading Functions ---
def load_huggingface_pipeline_func(task: str, model_name: str, device: torch.device, **kwargs):
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found."); device_id = device.index if device.type == 'cuda' else (-1 if device.type == 'cpu' else 0); trust_code = kwargs.pop('trust_remote_code', False); return pipeline(task, model=model_name, device=device_id, trust_remote_code=trust_code, **kwargs)
def load_huggingface_model_func(model_name: str, model_class: type, **kwargs):
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found."); trust_code = kwargs.pop('trust_remote_code', False); return model_class.from_pretrained(model_name, trust_remote_code=trust_code, **kwargs)
def load_huggingface_processor_func(model_name: str, processor_class: type, **kwargs):
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found."); trust_code = kwargs.pop('trust_remote_code', False); return processor_class.from_pretrained(model_name, trust_remote_code=trust_code, **kwargs)
def load_sentence_transformer_func(model_name: str):
    if not SENTENCE_TRANSFORMERS_AVAILABLE: raise ImportError("Sentence-Transformers library not found."); return SentenceTransformer(model_name)

def load_syncnet_model_from_hf_func(config: AnalysisConfig) -> Optional[SyncNet]:
    """Loads SyncNet model from Hugging Face Hub."""
    if not HUGGINGFACE_HUB_AVAILABLE: raise ImportError("huggingface_hub required for SyncNet download.")
    if 'SyncNet' not in globals() or not inspect.isclass(globals()['SyncNet']): raise RuntimeError("SyncNet class definition missing.")
    repo_id = config.syncnet_repo_id; filename = config.syncnet_filename; logger.info(f"Attempting SyncNet load ({repo_id}/{filename})...")
    try:
        user_agent = {"library_name": "videous-chef", "library_version": CACHE_VERSION}; checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename, user_agent=user_agent, resume_download=True)
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"SyncNet checkpoint not found after download: {checkpoint_path}")
        syncnet_model = SyncNet(); loaded_data = torch.load(checkpoint_path, map_location='cpu'); state_dict = loaded_data.get('state_dict', loaded_data);
        if isinstance(state_dict, dict) and 'net' in state_dict and isinstance(state_dict['net'], dict): state_dict = state_dict['net'] # Handle nested 'net' key
        adapted_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()} # Remove 'module.' prefix
        missing_keys, unexpected_keys = syncnet_model.load_state_dict(adapted_state_dict, strict=False) # Load non-strictly
        if missing_keys: logger.warning(f"SyncNet loaded with MISSING keys: {missing_keys}")
        if unexpected_keys: logger.warning(f"SyncNet loaded with UNEXPECTED keys: {unexpected_keys}")
        logger.info("SyncNet weights loaded successfully (non-strict).")
        return syncnet_model
    except FileNotFoundError as fnf_err: logger.error(f"SyncNet Download/Load FNF Error: {fnf_err}"); return None
    except Exception as e: logger.error(f"Failed to load SyncNet weights: {e}", exc_info=True); return None

@lru_cache(maxsize=1) # Cache MiDaS model after first load
def get_midas_model() -> Optional[Tuple[torch.nn.Module, Any, torch.device]]:
    """Loads the MiDaS model and transform using generic loader."""
    if not TIMM_AVAILABLE: logger.error("MiDaS requires 'timm'. Install with: pip install timm"); return None, None, None
    logger.info("Loading MiDaS model (intel-isl/MiDaS MiDaS_small)...")
    try:
        model, device = get_pytorch_model("midas_small", lambda: torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True))
        if model is None: return None, None, None
        transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True); transform = transforms_hub.small_transform; logger.info("MiDaS model and transform loaded."); return model, transform, device
    except Exception as e: logger.error(f"Failed to load MiDaS: {e}", exc_info=True); if "midas_small" in _pytorch_model_cache: del _pytorch_model_cache["midas_small"]; return None, None, None

# --- Mouth Crop Helper (Using scikit-image) ---
MOUTH_LM_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 61, 76, 77, 78, 80, 81, 82, 84, 85, 87, 88, 90, 91, 95, 146, 178, 180, 181, 267, 269, 270, 291, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 320, 321, 324, 375, 402, 404, 405, 409, 415]
def extract_mouth_crop(frame_bgr: np.ndarray, face_landmarks: Any, target_size=(112, 112)) -> Optional[np.ndarray]:
    """Extracts and resizes the mouth region using FaceMesh landmarks and scikit-image."""
    if face_landmarks is None or not SKIMAGE_AVAILABLE:
        if not SKIMAGE_AVAILABLE: logger.debug("Skipping mouth crop: scikit-image not available.")
        return None
    h, w = frame_bgr.shape[:2];
    if h == 0 or w == 0: return None
    try:
        lm = face_landmarks.landmark; mouth_points = []
        for idx in MOUTH_LM_INDICES:
            if 0 <= idx < len(lm):
                 x = lm[idx].x * w; y = lm[idx].y * h;
                 if np.isfinite(x) and np.isfinite(y): mouth_points.append([int(round(x)), int(round(y))])
        if len(mouth_points) < 4: logger.debug("Insufficient valid mouth landmarks."); return None
        mouth_points = np.array(mouth_points, dtype=np.int32); min_x, min_y = np.min(mouth_points, axis=0); max_x, max_y = np.max(mouth_points, axis=0)
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2; width_box = max(1, max_x - min_x); height_box = max(1, max_y - min_y); crop_size = max(int(max(width_box, height_box) * 1.6), 10)
        x1 = max(0, center_x - crop_size // 2); y1 = max(0, center_y - crop_size // 2); x2 = min(w, x1 + crop_size); y2 = min(h, y1 + crop_size);
        if (x2 - x1) != crop_size and (y2 - y1) == crop_size: x1 = max(0, x2 - crop_size)
        if (y2 - y1) != crop_size and (x2 - x1) == crop_size: y1 = max(0, y2 - crop_size)
        final_width = x2 - x1; final_height = y2 - y1
        if final_width <=0 or final_height <= 0: logger.debug(f"Mouth crop invalid size: W={final_width}, H={final_height}"); return None
        mouth_crop = frame_bgr[y1:y2, x1:x2]
        # Use skimage.transform.resize
        resized_mouth = skimage.transform.resize(mouth_crop, target_size, anti_aliasing=True, mode='edge', preserve_range=False)
        resized_mouth_uint8 = (resized_mouth * 255).astype(np.uint8)
        return resized_mouth_uint8 # Return BGR uint8 crop
    except Exception as e: logger.warning(f"Error extracting mouth crop: {e}", exc_info=False); return None

# --- Mel Spectrogram Helper (Using Librosa) ---
@lru_cache(maxsize=4) # Cache a few recent Mel specs
def compute_mel_spectrogram(audio_np: np.ndarray, sr: int, n_mels: int = 80) -> Optional[np.ndarray]:
    """Computes the Mel spectrogram using Librosa (required for SyncNet)."""
    if not LIBROSA_AVAILABLE: logger.error("Librosa required for Mel Spectrogram (SyncNet dependency)."); return None
    try:
        n_fft=512; win_length=400; hop_length=160 # Common SyncNet params @ 16kHz
        audio_np_float32 = audio_np.astype(np.float32)
        mel_spec = librosa.feature.melspectrogram(y=audio_np_float32, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, fmin=55, fmax=7600)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        logger.debug(f"Computed Mel Spectrogram: Shape {log_mel_spec.shape} (Mels, TimeSteps)")
        return log_mel_spec.astype(np.float32)
    except Exception as e: logger.error(f"Error computing Mel Spectrogram: {e}", exc_info=True); return None

# --- SAAPV History Helper ---
def get_recent_history(history: deque, count: int) -> List: return list(history)

# --- Pose/Visual Util Helpers (Unchanged logic, minor logging/error handling tweaks) ---
def calculate_flow_velocity(prev_gray, current_gray):
    if prev_gray is None or current_gray is None or prev_gray.shape != current_gray.shape: return 0.0, None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if flow is None: return 0.0, None
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.nanmean(magnitude); avg_magnitude = avg_magnitude if np.isfinite(avg_magnitude) else 0.0
        return float(avg_magnitude * 10.0), flow
    except cv2.error as cv_err: logger.warning(f"OpenCV optical flow error: {cv_err}"); return 0.0, None
    except Exception as e: logger.warning(f"Flow velocity error: {e}"); return 0.0, None

def calculate_flow_acceleration(prev_flow, current_flow, dt):
    if prev_flow is None or current_flow is None or prev_flow.shape != current_flow.shape or dt <= 1e-6: return 0.0
    try:
        flow_diff = current_flow - prev_flow; accel_magnitude_per_pixel, _ = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
        avg_accel_magnitude = np.nanmean(accel_magnitude_per_pixel); avg_accel_magnitude = avg_accel_magnitude if np.isfinite(avg_accel_magnitude) else 0.0
        accel = avg_accel_magnitude / dt; return float(accel * 10.0)
    except Exception as e: logger.warning(f"Flow acceleration error: {e}"); return 0.0

def calculate_kinetic_energy_proxy(landmarks_prev, landmarks_curr, dt):
    if landmarks_prev is None or landmarks_curr is None or dt <= 1e-6 or not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0
    lm_prev = landmarks_prev.landmark; lm_curr = landmarks_curr.landmark;
    if len(lm_prev) != len(lm_curr): return 0.0
    total_sq_velocity = 0.0; num_valid = 0
    try:
        for i in range(len(lm_prev)):
            vis_prev = getattr(lm_prev[i], 'visibility', 1.0); vis_curr = getattr(lm_curr[i], 'visibility', 1.0)
            if vis_prev > 0.2 and vis_curr > 0.2:
                dx = lm_curr[i].x - lm_prev[i].x; dy = lm_curr[i].y - lm_prev[i].y; dz = lm_curr[i].z - lm_prev[i].z
                total_sq_velocity += (dx**2 + dy**2 + dz**2) / (dt**2); num_valid += 1
    except IndexError: logger.warning("Kinetic energy calc index error."); return 0.0
    if num_valid == 0: return 0.0
    avg_sq_velocity = total_sq_velocity / num_valid; return float(avg_sq_velocity * 1000.0)

def calculate_movement_jerk_proxy(landmarks_prev_prev, landmarks_prev, landmarks_curr, dt):
    if landmarks_prev_prev is None or landmarks_prev is None or landmarks_curr is None or dt <= 1e-6 or not hasattr(landmarks_prev_prev, 'landmark') or not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0
    lm_pp = landmarks_prev_prev.landmark; lm_p = landmarks_prev.landmark; lm_c = landmarks_curr.landmark
    if len(lm_pp) != len(lm_p) or len(lm_p) != len(lm_c): return 0.0
    total_sq_accel_change = 0.0; num_valid = 0; dt_sq = dt * dt
    try:
        for i in range(len(lm_pp)):
            vis_pp = getattr(lm_pp[i], 'visibility', 1.0); vis_p = getattr(lm_p[i], 'visibility', 1.0); vis_c = getattr(lm_c[i], 'visibility', 1.0)
            if vis_pp > 0.2 and vis_p > 0.2 and vis_c > 0.2:
                ax = (lm_c[i].x - 2*lm_p[i].x + lm_pp[i].x) / dt_sq; ay = (lm_c[i].y - 2*lm_p[i].y + lm_pp[i].y) / dt_sq; az = (lm_c[i].z - 2*lm_p[i].z + lm_pp[i].z) / dt_sq
                accel_magnitude_sq = ax**2 + ay**2 + az**2; total_sq_accel_change += accel_magnitude_sq; num_valid += 1
    except IndexError: logger.warning("Jerk calc index error."); return 0.0
    if num_valid == 0: return 0.0
    avg_sq_accel_proxy = total_sq_accel_change / num_valid; return float(avg_sq_accel_proxy * 100000.0)

class BBZPoseUtils: # (Unchanged from v4.7.3)
    def drawLandmarksOnImage(self, imageInput, poseProcessingInput):
        annotated_image = imageInput.copy();
        if poseProcessingInput and poseProcessingInput.pose_landmarks:
            try: mp_drawing.draw_landmarks(annotated_image, poseProcessingInput.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            except Exception as e: logger.warning(f"Failed draw pose landmarks: {e}")
        return annotated_image

# ========================================================================
#           AUDIO ANALYSIS UTILITIES (Major Refactor - Ensemble v5.4)
# ========================================================================
class BBZAudioUtils:
    def extract_audio(self, video_path, audio_output_path="temp_audio.wav"):
        logger.info(f"Extracting audio from '{os.path.basename(video_path)}' via FFmpeg...")
        start_time = time.time();
        try:
            if not os.path.exists(video_path): raise FileNotFoundError(f"Video file not found: {video_path}")
            output_dir = os.path.dirname(audio_output_path);
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            command = ["ffmpeg", "-i", shlex.quote(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-y", shlex.quote(audio_output_path), "-hide_banner", "-loglevel", "error"]
            logger.debug(f"FFmpeg command: {' '.join(command)}")
            result = subprocess.run(" ".join(command), shell=True, capture_output=True, text=True, check=False, encoding='utf-8')
            if result.returncode != 0 or not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
                logger.error(f"FFmpeg failed (Code:{result.returncode}) extracting audio. Stderr: {result.stderr}"); return None
            else: logger.info(f"Audio extracted: '{audio_output_path}' ({time.time() - start_time:.2f}s)"); return audio_output_path
        except FileNotFoundError as fnf_err: logger.error(f"FFmpeg command failed: {fnf_err}. Is FFmpeg installed/in PATH?"); return None
        except Exception as e: logger.error(f"FFmpeg extraction failed for '{video_path}': {e}", exc_info=True); return None

    def _detect_beats_downbeats(self, waveform_np: np.ndarray, sr: int, config: AnalysisConfig) -> Tuple[List[float], List[float]]:
        beats, downbeats = [], []; method_used = "None"
        if config.use_dl_beat_tracker and AUDIOFLUX_AVAILABLE:
            logger.info("Using audioFlux for beat/downbeat detection."); method_used = "AudioFlux"
            try:
                hop_length_af = 256; waveform_af = np.ascontiguousarray(waveform_np, dtype=np.float32); novelty_obj = af.Novelty(num=waveform_af.shape[0] // hop_length_af, radix2_exp=12, samplate=sr, novelty_type=af.NoveltyType.FLUX); novelty = novelty_obj.novelty(waveform_af); peak_picking_obj = af.PeakPicking(novelty, time_length=sr // hop_length_af)
                peak_picking_obj.pick(thresh=1.5, wait=int(0.1 * (sr / hop_length_af)), pre_avg=int(0.05 * (sr / hop_length_af)), post_avg=int(0.05 * (sr / hop_length_af))); peak_indexes = peak_picking_obj.get_peak_indexes(); times = peak_indexes * hop_length_af / sr; beats = sorted([float(t) for t in times if np.isfinite(t)]); downbeats = beats[0::4] if len(beats) > 3 else beats[:1]
                if not beats: logger.warning("AudioFlux beat detection returned no beats.")
            except Exception as af_err: logger.error(f"audioFlux beat detection failed: {af_err}", exc_info=False); beats, downbeats = [], []; method_used = "AudioFlux_Failed"
        if not beats and LIBROSA_AVAILABLE:
            logger.warning("Using Librosa fallback for beat/downbeat detection."); method_used = "Librosa"
            try: tempo_val, beat_frames = librosa.beat.beat_track(y=waveform_np, sr=sr, hop_length=512, units='frames'); beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512).tolist(); downbeats = beat_times[0::4] if beat_times else [];
                if not beat_times: logger.warning("Librosa beat detection returned no beats."); else: beats = beat_times
            except Exception as librosa_beat_err: logger.error(f"Librosa beat fallback failed: {librosa_beat_err}"); beats, downbeats = [], []; method_used = "Librosa_Failed"
        if not beats: logger.error("No beat detection method succeeded or available."); method_used = "None"
        logger.debug(f"Beat detection method: {method_used}. Found {len(beats)} beats, {len(downbeats)} downbeats."); return beats, downbeats

    def _detect_micro_beats(self, waveform_np: np.ndarray, sr: int, config: AnalysisConfig, stem_type: Optional[str] = None) -> List[float]:
        log_prefix = f"Micro-Beat ({stem_type or 'full mix'})"; micro_beats = []; method_used = "None"
        if AUDIOFLUX_AVAILABLE:
            logger.info(f"Using audioFlux for {log_prefix} detection."); method_used = "AudioFlux"
            try:
                hop_length_af_micro = 64; waveform_af = np.ascontiguousarray(waveform_np, dtype=np.float32); novelty_type = af.NoveltyType.HFC; novelty_obj = af.Novelty(num=waveform_af.shape[0] // hop_length_af_micro, radix2_exp=11, samplate=sr, novelty_type=novelty_type); novelty = novelty_obj.novelty(waveform_af); peak_picking_obj = af.PeakPicking(novelty, time_length=sr // hop_length_af_micro)
                peak_picking_obj.pick(thresh=1.8, wait=int(0.02 * (sr / hop_length_af_micro)), pre_avg=int(0.01 * (sr / hop_length_af_micro)), post_avg=int(0.01 * (sr / hop_length_af_micro))); peak_indexes = peak_picking_obj.get_peak_indexes(); times = peak_indexes * hop_length_af_micro / sr; micro_beats = sorted([float(t) for t in times if np.isfinite(t)])
                if not micro_beats: logger.warning(f"AudioFlux {log_prefix} detection returned no micro-beats.")
            except Exception as af_err: logger.error(f"audioFlux {log_prefix} detection failed: {af_err}", exc_info=False); micro_beats = []; method_used = "AudioFlux_Failed"
        if not micro_beats and LIBROSA_AVAILABLE:
            logger.warning(f"Using Librosa fallback for {log_prefix} detection."); method_used = "Librosa"
            try: onset_frames = librosa.onset.onset_detect(y=waveform_np, sr=sr, hop_length=128, backtrack=False, units='frames', delta=0.6, wait=2); micro_beats = librosa.frames_to_time(onset_frames, sr=sr, hop_length=128).tolist()
                if not micro_beats: logger.warning(f"Librosa {log_prefix} detection returned no micro-beats.")
            except Exception as librosa_onset_err: logger.error(f"Librosa onset fallback failed: {librosa_onset_err}"); micro_beats = []; method_used = "Librosa_Failed"
        if not micro_beats: logger.error(f"No micro-beat detection available/succeeded for {log_prefix}."); method_used = "None"
        logger.debug(f"Micro-beat detection ({log_prefix}): {method_used}. Found {len(micro_beats)} micro-beats."); return micro_beats

    def _segment_audio(self, waveform_tensor: torch.Tensor, sr: int, config: AnalysisConfig) -> List[float]:
        duration = waveform_tensor.shape[-1] / sr; waveform_np = waveform_tensor.squeeze(0).numpy().astype(np.float32); bound_times = []; method_used = "None"
        if AUDIOFLUX_AVAILABLE:
            logger.info("Using audioFlux for audio segmentation."); method_used = "AudioFlux"
            try:
                hop_length_af_seg = 512; waveform_af = np.ascontiguousarray(waveform_np, dtype=np.float32); novelty_obj = af.Novelty(num=waveform_af.shape[0] // hop_length_af_seg, radix2_exp=12, samplate=sr, novelty_type=af.NoveltyType.ENERGY); novelty = novelty_obj.novelty(waveform_af); win_len_smooth = max(11, int(sr * 2.5 / hop_length_af_seg) | 1); novelty_smooth = scipy.signal.savgol_filter(novelty, window_length=win_len_smooth, polyorder=3) if len(novelty) > win_len_smooth else novelty; peak_picking_obj = af.PeakPicking(novelty_smooth, time_length=sr // hop_length_af_seg)
                peak_picking_obj.pick(thresh=1.2, wait=int(1.0 * (sr / hop_length_af_seg)), pre_avg=int(0.5 * (sr / hop_length_af_seg)), post_avg=int(0.5 * (sr / hop_length_af_seg))); peak_indexes = peak_picking_obj.get_peak_indexes(); bound_times_raw = peak_indexes * hop_length_af_seg / sr; potential_bounds = sorted(list(set([0.0] + [float(t) for t in bound_times_raw if np.isfinite(t)] + [duration]))); final_bounds = [potential_bounds[0]]; min_len_sec = max(0.5, config.min_sequence_clip_duration * 0.7)
                for t in potential_bounds[1:]:
                    if t - final_bounds[-1] >= min_len_sec: final_bounds.append(t)
                if final_bounds[-1] < duration - 1e-3:
                    if duration - final_bounds[-1] >= min_len_sec / 2: final_bounds.append(duration)
                    elif len(final_bounds) > 1: final_bounds[-1] = duration
                if len(final_bounds) < 2: final_bounds = [0.0, duration] # Failsafe
                bound_times = [float(b) for b in final_bounds];
                if len(bound_times) <= 2: logger.warning(f"AudioFlux segmentation resulted in {len(bound_times)-1} segment(s).")
            except Exception as af_err: logger.error(f"audioFlux segmentation failed: {af_err}", exc_info=False); bound_times = []; method_used = "AudioFlux_Failed"
        if not bound_times:
            logger.warning("Using fixed duration segmentation fallback."); method_used = "Fixed_Interval"; avg_segment_dur = np.clip((config.min_sequence_clip_duration + config.max_sequence_clip_duration) / 2, 3.0, 10.0); num_segments = max(1, int(round(duration / avg_segment_dur))); bound_times = [float(t) for t in np.linspace(0, duration, num_segments + 1).tolist()]
        logger.debug(f"Segmentation method: {method_used}. Found {len(bound_times)-1} segments."); return bound_times

    def analyze_audio(self, audio_path: str, analysis_config: AnalysisConfig) -> Optional[Dict[str, Any]]:
        """Calculates enhanced audio features using Ensemble approach (v5.4)."""
        if ENABLE_PROFILING: profiler_start_time = time.time()
        logger.info(f"Analyzing audio (Ensemble v5.4): {os.path.basename(audio_path)}")
        target_sr_torch = analysis_config.target_sr_audio;
        try:
            if not os.path.exists(audio_path): raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.debug(f"Loading audio (Target SR: {target_sr_torch})...")
            try: waveform, sr = torchaudio.load(audio_path, normalize=True)
            except Exception as load_err:
                logger.warning(f"TorchAudio failed ({load_err}). Trying SoundFile fallback.")
                if not SOUNDFILE_AVAILABLE: raise RuntimeError("SoundFile library not available for fallback.") from load_err
                try: waveform_np_sf, sr = soundfile.read(audio_path, dtype='float32');
                    if waveform_np_sf.ndim > 1: waveform_np_sf = np.mean(waveform_np_sf, axis=1);
                    waveform = torch.from_numpy(waveform_np_sf).unsqueeze(0)
                except Exception as sf_err: raise RuntimeError(f"Failed to load audio file {audio_path} with both TorchAudio and SoundFile: {sf_err}") from sf_err
            if sr != target_sr_torch: logger.debug(f"Resampling {sr} Hz -> {target_sr_torch} Hz..."); waveform = T.Resample(sr, target_sr_torch)(waveform); sr = target_sr_torch
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True) # Ensure mono
            duration = waveform.shape[1] / sr;
            if duration <= 0: raise ValueError("Audio has zero or negative duration.")
            waveform_np = waveform.squeeze(0).numpy().astype(np.float32); logger.debug(f"Audio loaded: Shape={waveform.shape}, SR={sr}, Duration={duration:.2f}s")
            # --- Optional Demucs Placeholder ---
            stems = {}; # !!! Actual Demucs Implementation Needed Here if use_demucs_for_mrise=True !!!
            if analysis_config.use_mrise_sync and analysis_config.use_demucs_for_mrise:
                if DEMUCS_AVAILABLE: logger.warning("Demucs integration NOT IMPLEMENTED - using full mix for MRISE."); # stems = run_demucs(...)
                else: logger.warning("Demucs enabled but library not found. Using full mix.")
            # --- Rhythm Analysis ---
            logger.debug("Analyzing rhythm (Beats/Downbeats/Tempo)..."); beat_times, downbeat_times = self._detect_beats_downbeats(waveform_np, sr, analysis_config); tempo = 120.0
            if LIBROSA_AVAILABLE:
                try: tempo_val = librosa.beat.tempo(y=waveform_np, sr=sr, aggregate=np.median)[0]; tempo = float(tempo_val) if np.isfinite(tempo_val) else 120.0
                except Exception as tempo_err: logger.warning(f"Librosa tempo detection failed: {tempo_err}")
            else: logger.warning("Librosa unavailable for tempo, using default 120 BPM."); logger.debug(f"Tempo: {tempo:.1f} BPM")
            # --- Micro-Rhythm (MRISE) ---
            micro_beat_times = [];
            if analysis_config.use_mrise_sync:
                logger.debug("Analyzing micro-rhythms (MRISE)..."); target_stem_mrise = 'drums'; audio_for_mrise = stems.get(target_stem_mrise, waveform_np) if (analysis_config.use_demucs_for_mrise and stems) else waveform_np; stem_name_log = target_stem_mrise if (analysis_config.use_demucs_for_mrise and stems and target_stem_mrise in stems) else None; micro_beat_times = self._detect_micro_beats(audio_for_mrise, sr, analysis_config, stem_name_log)
            # --- Energy & Trend (HEFM) ---
            logger.debug("Computing energy features and trends (HEFM)..."); hop_length_energy = 512; frame_length_energy = hop_length_energy * 2; wav_padded = torch.nn.functional.pad(waveform, (frame_length_energy // 2, frame_length_energy // 2), mode='reflect'); frames = wav_padded.unfold(1, frame_length_energy, hop_length_energy); rms_energy_torch = torch.sqrt(torch.mean(frames**2, dim=-1)).squeeze(0); rms_energy = rms_energy_torch.cpu().numpy(); rms_times = np.linspace(0, duration, len(rms_energy), endpoint=False); smoothing_window_len_sec = 3.0; smoothing_window_len_frames = max(11, int(sr * smoothing_window_len_sec / hop_length_energy) | 1)
            rms_energy_long = scipy.signal.savgol_filter(rms_energy, smoothing_window_len_frames, 3) if len(rms_energy) > smoothing_window_len_frames else rms_energy.copy()
            rms_deriv_short = np.gradient(rms_energy, rms_times) if len(rms_energy) > 1 else np.zeros_like(rms_energy)
            rms_deriv_long = np.gradient(rms_energy_long, rms_times) if len(rms_energy_long) > 1 else np.zeros_like(rms_energy_long)
            # --- Mel Spectrogram (PWRC) ---
            full_mel_spectrogram = None; mel_times = None
            if analysis_config.use_latent_sync:
                logger.debug("Computing Mel Spectrogram for SyncNet (PWRC)..."); full_mel_spectrogram = compute_mel_spectrogram(waveform_np, sr)
                if full_mel_spectrogram is not None and LIBROSA_AVAILABLE: hop_length_mel = 160; mel_times = librosa.frames_to_time(np.arange(full_mel_spectrogram.shape[1]), sr=sr, hop_length=hop_length_mel)
                elif full_mel_spectrogram is not None: logger.warning("Librosa unavailable for Mel times, using linspace."); mel_times = np.linspace(0, duration, full_mel_spectrogram.shape[1], dtype=np.float32)
                else: logger.error("Mel Spectrogram computation failed. SyncNet scoring disabled.")
            # --- Structure Analysis (Segmentation) ---
            logger.debug("Analyzing musical structure (Segmentation)..."); bound_times = self._segment_audio(waveform, sr, analysis_config)
            # --- Aggregate Features per Segment ---
            segment_features = []; logger.debug(f"Aggregating features for {len(bound_times)-1} segments...")
            for i in range(len(bound_times) - 1):
                t_start, t_end = bound_times[i], bound_times[i+1]; seg_duration = t_end - t_start;
                if seg_duration <= 1e-6: continue
                rms_indices = np.where((rms_times >= t_start) & (rms_times < t_end))[0]
                if len(rms_indices) == 0:
                    if len(rms_times) > 0: nearest_idx = np.argmin(np.abs(rms_times - (t_start + t_end) / 2.0)); rms_indices = [nearest_idx]
                    else: logger.warning(f"No RMS frames for segment {i} [{t_start:.2f}-{t_end:.2f}]. Skipping."); continue
                seg_rms = float(np.mean(rms_energy[rms_indices])) if len(rms_indices) > 0 else 0.0; seg_rms_long = float(np.mean(rms_energy_long[rms_indices])) if len(rms_indices) > 0 else 0.0; seg_trend_short = float(np.mean(rms_deriv_short[rms_indices])) if len(rms_indices) > 0 else 0.0; seg_trend_long = float(np.mean(rms_deriv_long[rms_indices])) if len(rms_indices) > 0 else 0.0
                b_i = np.clip(seg_rms / (analysis_config.norm_max_rms + 1e-6), 0.0, 1.0); onset_proxy = np.clip(seg_trend_short / (analysis_config.norm_max_onset + 1e-6), -1.0, 1.0); e_i = (onset_proxy + 1.0) / 2.0; arousal_proxy = np.clip(tempo / 180.0, 0.1, 1.0); valence_proxy = 0.5; m_i = [float(arousal_proxy), float(valence_proxy)]
                seg_data = {'start': float(t_start), 'end': float(t_end), 'duration': float(seg_duration), 'rms_avg': seg_rms, 'rms_avg_long': seg_rms_long, 'trend_short': seg_trend_short, 'trend_long': seg_trend_long, 'b_i': float(b_i), 'e_i': float(e_i), 'm_i': m_i}
                segment_features.append(seg_data)
            # --- Prepare Return Data ---
            raw_features_dict = {
                 'rms_energy': rms_energy, 'rms_times': rms_times, 'rms_energy_long': rms_energy_long, 'rms_deriv_short': rms_deriv_short, 'rms_deriv_long': rms_deriv_long,
                 'original_audio_path': audio_path, # Store original path for later use (ASR/SER)
                 'mel_spectrogram': full_mel_spectrogram, # Keep as numpy array initially
                 'mel_times': mel_times,                 # Keep as numpy array initially
                 'perc_times': [], 'percussive_ratio': [], # Placeholder for V4 percussive feature
                 'waveform_np_for_ser': waveform_np.copy() # Store waveform for SER if needed later
            }
            result_data = {
                "sr": sr, "duration": float(duration), "tempo": float(tempo), "beat_times": [float(t) for t in beat_times], "downbeat_times": [float(t) for t in downbeat_times], "micro_beat_times": [float(t) for t in micro_beat_times], "segment_boundaries": [float(t) for t in bound_times], "segment_features": segment_features, # Already basic types
                "raw_features": raw_features_dict, # Contains numpy arrays/lists
            }
            logger.info(f"Audio analysis complete ({time.time() - profiler_start_time:.2f}s)");
            if ENABLE_PROFILING: logger.debug(f"PROFILING: Audio Analysis took {time.time() - profiler_start_time:.3f}s");
            return result_data
        except FileNotFoundError as fnf_err: logger.error(f"Audio analysis failed - File Not Found: {fnf_err}"); return None
        except ValueError as val_err: logger.error(f"Audio analysis failed - Value Error: {val_err}"); return None
        except RuntimeError as rt_err: logger.error(f"Audio analysis failed - Runtime Error: {rt_err}"); return None
        except Exception as e: logger.error(f"Unexpected error during audio analysis: {e}", exc_info=True); return None


# ========================================================================
#                    FACE UTILITIES (Returns Landmarks via MP Results)
# ========================================================================
class BBZFaceUtils: # (Focus on getting MP results, refine landmarks=True)
    def __init__(self, static_mode=False, max_faces=1, min_detect_conf=0.5, min_track_conf=0.5):
        self.face_mesh = None; self._mp_face_mesh = mp.solutions.face_mesh if MEDIAPIPE_AVAILABLE else None;
        if not self._mp_face_mesh: logger.error("MediaPipe FaceMesh solution not available."); return
        try: self.face_mesh = self._mp_face_mesh.FaceMesh(static_image_mode=static_mode, max_num_faces=max_faces, refine_landmarks=True, min_detection_confidence=min_detect_conf, min_tracking_confidence=min_track_conf); logger.info("FaceMesh initialized (Refine Landmarks: True).")
        except Exception as e: logger.error(f"Failed to initialize FaceMesh: {e}"); self.face_mesh = None
    def process_frame(self, image_bgr):
        if self.face_mesh is None: return None
        try: image_bgr.flags.writeable = False; image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB); results = self.face_mesh.process(image_rgb); image_bgr.flags.writeable = True; return results
        except Exception as e: logger.warning(f"FaceMesh process error: {e}"); if hasattr(image_bgr, 'flags'): image_bgr.flags.writeable = True; return None
    def get_heuristic_face_features(self, results, h, w, mouth_open_threshold=0.05):
        is_open, size_ratio, center_x = False, 0.0, 0.5;
        if results and results.multi_face_landmarks:
            try:
                face_landmarks = results.multi_face_landmarks[0]; lm = face_landmarks.landmark
                if 13 < len(lm) and 14 < len(lm) and 10 < len(lm) and 152 < len(lm):
                    upper_lip_y = lm[13].y * h; lower_lip_y = lm[14].y * h; mouth_height = abs(lower_lip_y - upper_lip_y); forehead_y = lm[10].y * h; chin_y = lm[152].y * h; face_height = abs(chin_y - forehead_y);
                    if face_height > 1e-6: mouth_open_ratio = mouth_height / face_height; is_open = mouth_open_ratio > mouth_open_threshold
                all_x = [p.x * w for p in lm]; all_y = [p.y * h for p in lm]; min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y); face_box_w = max(1, max_x - min_x); face_box_h = max(1, max_y - min_y); face_diagonal = math.sqrt(face_box_w**2 + face_box_h**2); image_diagonal = math.sqrt(w**2 + h**2);
                if image_diagonal > 1e-6: size_ratio = np.clip(face_diagonal / image_diagonal, 0.0, 1.0)
                if 234 < len(lm) and 454 < len(lm): left_cheek_x = lm[234].x; right_cheek_x = lm[454].x; center_x = np.clip((left_cheek_x + right_cheek_x) / 2.0, 0.0, 1.0)
                else: center_x = np.clip(np.mean([p.x for p in lm]), 0.0, 1.0)
            except IndexError: logger.warning("FaceMesh landmark index error.")
            except Exception as e: logger.warning(f"Heuristic face feature error: {e}")
        return is_open, float(size_ratio), float(center_x)
    def close(self):
        if self.face_mesh:
            try: self.face_mesh.close(); logger.info("FaceMesh resources released.")
            except Exception as e: logger.error(f"Error closing FaceMesh: {e}")
        self.face_mesh = None

# ========================================================================
#         DYNAMIC SEGMENT IDENTIFIER (V4 Logic - Optional/Legacy)
# ========================================================================
class DynamicSegmentIdentifier:
    def __init__(self, analysis_config: AnalysisConfig, fps: float):
        self.fps = fps if fps > 0 else 30.0; self.score_threshold = getattr(analysis_config, 'score_threshold', 0.3); self.min_segment_len_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES, int(analysis_config.min_potential_clip_duration_sec * self.fps)); logger.debug(f"Legacy Segment ID Init: FPS={self.fps:.2f}, Threshold={self.score_threshold:.2f}, MinLenFrames={self.min_segment_len_frames}")
    def find_potential_segments(self, frame_features_list):
        potential_segments = []; start_idx = -1; n = len(frame_features_list);
        if n == 0: return []
        for i, features in enumerate(frame_features_list):
            score = features.get('boosted_score', 0.0) if isinstance(features, dict) else 0.0; is_candidate = score >= self.score_threshold
            if is_candidate and start_idx == -1: start_idx = i
            elif not is_candidate and start_idx != -1: segment_len = i - start_idx;
                if segment_len >= self.min_segment_len_frames: potential_segments.append({'start_frame': start_idx, 'end_frame': i}); start_idx = -1
        if start_idx != -1: segment_len = n - start_idx;
            if segment_len >= self.min_segment_len_frames: potential_segments.append({'start_frame': start_idx, 'end_frame': n})
        min_len_s = self.min_segment_len_frames / (self.fps if self.fps > 0 else 30.0); logger.info(f"Legacy Segment ID identified {len(potential_segments)} segments (BoostedScore >= {self.score_threshold:.2f}, MinLen={min_len_s:.2f}s)"); return potential_segments

# ========================================================================
#                      IMAGE UTILITIES (Unchanged)
# ========================================================================
class BBZImageUtils:
    def resizeTARGET(self, image, TARGET_HEIGHT=256, TARGET_WIDTH=256):
        if image is None or image.size == 0: logger.warning("Resize empty image."); return None
        h, w = image.shape[:2];
        if h == 0 or w == 0: logger.warning(f"Resize zero dim image: {h}x{w}"); return None
        if h != TARGET_HEIGHT or w != TARGET_WIDTH:
            interpolation = cv2.INTER_AREA if h > TARGET_HEIGHT or w > TARGET_WIDTH else cv2.INTER_LINEAR
            try: return cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=interpolation)
            except cv2.error as cv_err: logger.warning(f"OpenCV resize failed: {cv_err}. Ret original."); return image
            except Exception as e: logger.warning(f"Resize error: {e}. Ret original."); return image
        return image

# ========================================================================
#         BATCH SYNC NET SCORING (PWRC - Called after video analysis)
# ========================================================================
def batch_score_syncnet(clips_to_score: List['ClipSegment'], syncnet_model: SyncNet,
                        full_mel_spectrogram: np.ndarray, mel_times: np.ndarray, config: AnalysisConfig):
    """Calculates SyncNet scores for multiple ClipSegments in batches."""
    # --- Pre-checks ---
    if not config.use_latent_sync: logger.info("SyncNet scoring skipped (disabled in config)."); return
    if not clips_to_score: logger.info("SyncNet scoring skipped (no clips provided)."); return
    if syncnet_model is None: logger.error("SyncNet model not loaded, cannot score."); return
    if not SKIMAGE_AVAILABLE: logger.error("scikit-image not available, cannot process mouth crops for SyncNet."); return
    if full_mel_spectrogram is None or mel_times is None: logger.error("SyncNet scoring skipped: Missing Mel data."); return
    if len(mel_times) < 2: logger.error("SyncNet scoring skipped: Cannot determine Mel hop time."); return

    logger.info(f"--- Starting Batch SyncNet Scoring (PWRC) for {len(clips_to_score)} clips ---")
    if ENABLE_PROFILING: start_time = time.time()
    syncnet_model_device = next(syncnet_model.parameters()).device # Get device model is on
    syncnet_batch_size = config.syncnet_batch_size
    target_crop_h, target_crop_w = 112, 112 # Expected by SyncNet

    # --- 1. Prepare all batch items (video & audio chunks) from all clips ---
    all_video_batch_items = [] # List of (1, C, 5, H, W) tensors
    all_audio_batch_items = [] # List of (1, 1, Mels, T_audio) tensors
    clip_chunk_map = [] # Stores (clip_index_in_list, chunk_index_within_clip) for score aggregation

    mel_hop_sec = mel_times[1] - mel_times[0] # Estimate hop duration
    mel_window_size_frames = int(round(0.2 / mel_hop_sec)) if mel_hop_sec > 0 else 20 # SyncNet audio window ~0.2s

    pbar_prep = tqdm(total=len(clips_to_score), desc="Prepare SyncNet Batches", leave=False, disable=not TQDM_AVAILABLE)
    num_skipped_clips = 0

    for clip_idx, clip in enumerate(clips_to_score):
        # Check if clip is long enough and has frame features
        if clip.num_frames < 5 or not clip.segment_frame_features:
             clip.latent_sync_score = 0.0 # Set default score for short/invalid clips
             num_skipped_clips += 1
             if pbar_prep: pbar_prep.update(1); continue

        # Extract mouth crops and original frame indices
        mouth_crops = []; original_frame_indices = []
        for i, f in enumerate(clip.segment_frame_features):
            if isinstance(f, dict):
                 crop = f.get('mouth_crop') # This should be the BGR uint8 112x112 array
                 if crop is not None and isinstance(crop, np.ndarray) and crop.shape == (target_crop_h, target_crop_w, 3):
                     mouth_crops.append(crop)
                     original_frame_indices.append(clip.start_frame + i)
                 # else: logger.debug(f"Frame {clip.start_frame+i} in clip {clip_idx} missing valid mouth crop.")

        if len(mouth_crops) < 5:
            clip.latent_sync_score = 0.0 # Not enough crops for a window
            num_skipped_clips += 1
            if pbar_prep: pbar_prep.update(1); continue

        # Subsample frames if necessary
        num_original_crops = len(mouth_crops)
        if num_original_crops > LATENTSYNC_MAX_FRAMES:
            subsample_indices = np.linspace(0, num_original_crops - 1, LATENTSYNC_MAX_FRAMES, dtype=int)
            mouth_crops = [mouth_crops[i] for i in subsample_indices]
            original_frame_indices = [original_frame_indices[i] for i in subsample_indices]

        # Map frame indices to Mel indices
        crop_timestamps = [(idx + 0.5) / clip.fps for idx in original_frame_indices]
        mel_spec_indices = [np.argmin(np.abs(mel_times - ts)) for ts in crop_timestamps]

        # Iterate through 5-frame sliding windows for this clip
        num_windows = len(mouth_crops) - 4
        clip_chunk_scores = [] # Temporarily store scores for this clip's chunks
        for i in range(num_windows):
            # --- Video Chunk ---
            video_chunk_bgr = mouth_crops[i:i+5]
            video_chunk_processed = []
            for frame in video_chunk_bgr: # frame is (H, W, C) BGR uint8
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_float = frame_rgb.astype(np.float32) / 255.0 # Normalize to [0, 1]
                frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1) # (C, H, W)
                # Optional: Apply torchvision normalization if SyncNet expects it
                # frame_tensor = torchvision.transforms.functional.normalize(frame_tensor, mean=[...], std=[...])
                video_chunk_processed.append(frame_tensor)
            video_batch_item = torch.stack(video_chunk_processed, dim=1).unsqueeze(0) # (1, C, T=5, H, W)
            all_video_batch_items.append(video_batch_item)

            # --- Audio Chunk ---
            center_video_frame_index_in_window = i + 2
            center_mel_idx = mel_spec_indices[center_video_frame_index_in_window]
            start_mel = max(0, center_mel_idx - mel_window_size_frames // 2)
            end_mel = min(full_mel_spectrogram.shape[1], start_mel + mel_window_size_frames)
            start_mel = max(0, end_mel - mel_window_size_frames) # Adjust start if end hit boundary
            audio_chunk_mel = full_mel_spectrogram[:, start_mel:end_mel]
            current_len = audio_chunk_mel.shape[1]
            if current_len < mel_window_size_frames: # Pad if too short
                 padding = np.full((full_mel_spectrogram.shape[0], mel_window_size_frames - current_len), np.min(full_mel_spectrogram), dtype=np.float32)
                 audio_chunk_mel = np.concatenate((audio_chunk_mel, padding), axis=1)
            elif current_len > mel_window_size_frames: # Truncate if too long
                 audio_chunk_mel = audio_chunk_mel[:, :mel_window_size_frames]
            audio_batch_item = torch.from_numpy(audio_chunk_mel).unsqueeze(0).unsqueeze(0) # (1, 1, Mels, T_audio)
            all_audio_batch_items.append(audio_batch_item)

            # Store mapping
            clip_chunk_map.append((clip_idx, i)) # Store (original_clip_index, chunk_index_in_clip)

        if pbar_prep: pbar_prep.update(1)

    if pbar_prep: pbar_prep.close()
    if num_skipped_clips > 0: logger.info(f"Skipped SyncNet scoring for {num_skipped_clips} clips (too short or missing crops).")
    if not all_video_batch_items: logger.warning("No valid SyncNet batch items prepared. Skipping inference."); return

    # --- 2. Run Inference in Batches ---
    logger.info(f"Running SyncNet inference on {len(all_video_batch_items)} windows (Batch Size: {syncnet_batch_size})...")
    all_confidences = [] # Will store scores for each 5-frame window
    pbar_infer = tqdm(total=len(all_video_batch_items), desc="SyncNet Inference", leave=False, disable=not TQDM_AVAILABLE)
    try:
        with torch.no_grad():
            for i in range(0, len(all_video_batch_items), syncnet_batch_size):
                video_batch = torch.cat(all_video_batch_items[i:i+syncnet_batch_size], dim=0).to(syncnet_model_device) # (N, C, T, H, W)
                audio_batch = torch.cat(all_audio_batch_items[i:i+syncnet_batch_size], dim=0).to(syncnet_model_device) # (N, 1, Mels, T_audio)

                # --- SyncNet Forward Pass ---
                audio_embed, video_embed = syncnet_model(audio_batch, video_batch)
                # Calculate cosine similarity between audio and video embeddings for the batch
                cosine_sim = F.cosine_similarity(audio_embed, video_embed, dim=-1) # Shape: (N,)
                # Convert similarity [-1, 1] to confidence [0, 1]
                batch_confidences = (cosine_sim + 1.0) / 2.0
                all_confidences.extend(batch_confidences.cpu().numpy().tolist())
                if pbar_infer: pbar_infer.update(video_batch.shape[0])

    except Exception as infer_err:
         logger.error(f"SyncNet batch inference failed: {infer_err}", exc_info=True)
         # Assign default score of 0.0 to all clips that were supposed to be scored
         for clip in clips_to_score: clip.latent_sync_score = 0.0
         if pbar_infer: pbar_infer.close();
         return # Abort scoring on inference error

    if pbar_infer: pbar_infer.close()

    # --- 3. Aggregate scores per clip ---
    scores_per_clip = defaultdict(list)
    if len(all_confidences) != len(clip_chunk_map):
        logger.error(f"SyncNet score/map mismatch! Scores: {len(all_confidences)}, Map: {len(clip_chunk_map)}. Assigning default scores.")
        for clip in clips_to_score: clip.latent_sync_score = 0.0
        return

    for conf, (clip_idx, chunk_idx) in zip(all_confidences, clip_chunk_map):
        scores_per_clip[clip_idx].append(conf)

    # --- 4. Assign final score to ClipSegment objects ---
    for clip_idx, scores in scores_per_clip.items():
        if 0 <= clip_idx < len(clips_to_score):
            # Use average confidence score for the clip
            avg_score = float(np.mean(scores)) if scores else 0.0
            clips_to_score[clip_idx].latent_sync_score = np.clip(avg_score, 0.0, 1.0)
            # logger.debug(f"Clip {clip_idx} SyncNet Score: {clips_to_score[clip_idx].latent_sync_score:.3f} (from {len(scores)} windows)")
        else: logger.warning(f"Invalid clip index {clip_idx} found during SyncNet score aggregation.")

    # Ensure clips that were skipped or had no windows processed have a default score
    for clip in clips_to_score:
        if clip.latent_sync_score is None: clip.latent_sync_score = 0.0

    if ENABLE_PROFILING: logger.debug(f"PROFILING: Batch SyncNet Scoring took {time.time() - start_time:.3f}s")
    logger.info("Batch SyncNet scoring finished.")

# ========================================================================
#                       LVRE PREPROCESSING FUNCTION (Batched, Lazy Load)
# ========================================================================
def preprocess_lyrics_and_visuals(master_audio_data: Dict, all_potential_clips: List['ClipSegment'], analysis_config: AnalysisConfig):
    """Runs ASR, SER, Text & Visual Embeddings if LVRE enabled."""
    if not analysis_config.use_lvre_features: logger.info("LVRE features disabled. Skipping preprocessing."); return
    if not TRANSFORMERS_AVAILABLE: logger.error("LVRE requires `transformers` library. Aborting LVRE preprocessing."); return
    logger.info("--- Starting LVRE Preprocessing (Batched, Lazy Load) ---")
    if ENABLE_PROFILING: profiler_start_time = time.time()
    device = get_device()

    # --- 1. ASR (Whisper) ---
    timed_lyrics = master_audio_data.get('timed_lyrics') # Check cache first
    if timed_lyrics is None: # Run ASR if not cached
        logger.info("Running ASR (Whisper)..."); asr_success = False
        try:
            audio_path = master_audio_data.get("raw_features", {}).get("original_audio_path")
            if not audio_path or not os.path.exists(audio_path): raise ValueError(f"Original audio path missing for ASR: '{audio_path}'")
            whisper_pipeline, _ = get_pytorch_model(f"whisper_{analysis_config.whisper_model_name}", load_huggingface_pipeline_func, task="automatic-speech-recognition", model_name=analysis_config.whisper_model_name, device=device, chunk_length_s=30, stride_length_s=5, return_timestamps="word")
            if whisper_pipeline:
                logger.debug(f"Running Whisper pipeline on {os.path.basename(audio_path)}...")
                asr_result = whisper_pipeline(audio_path)
                if isinstance(asr_result, dict) and 'chunks' in asr_result:
                    timed_lyrics = asr_result.get('chunks', [])
                    valid_count = 0
                    for chunk in timed_lyrics:
                         ts = chunk.get('timestamp');
                         if isinstance(ts, (tuple, list)) and len(ts) == 2 and isinstance(ts[0], (int, float)) and isinstance(ts[1], (int, float)): valid_count += 1
                         else: logger.debug(f"ASR chunk ts format invalid: {ts}. Word: '{chunk.get('text')}'"); chunk['timestamp'] = None
                    master_audio_data['timed_lyrics'] = timed_lyrics # Store in master data
                    logger.info(f"ASR complete: {len(timed_lyrics)} word chunks ({valid_count} valid timestamps)."); asr_success = True
                else: logger.warning(f"ASR result format unexpected: {type(asr_result)}. Keys: {asr_result.keys() if isinstance(asr_result, dict) else 'N/A'}"); master_audio_data['timed_lyrics'] = []
            else: raise RuntimeError("Whisper pipeline failed to load.")
        except Exception as asr_err: logger.error(f"ASR processing failed: {asr_err}", exc_info=True); master_audio_data['timed_lyrics'] = []
        finally:
             if not asr_success: master_audio_data['timed_lyrics'] = [] # Ensure list on failure
    else: logger.info(f"Skipping ASR (timed_lyrics already present: {len(timed_lyrics)} words).")

    # Refresh local variable after potential update
    timed_lyrics = master_audio_data.get('timed_lyrics', [])

    # --- 2. SER ---
    run_ser = timed_lyrics and not all('emotion_score' in w for w in timed_lyrics)
    if run_ser:
        logger.info("Running SER (Speech Emotion Recognition)..."); ser_success = False
        try:
            ser_pipeline, _ = get_pytorch_model(f"ser_{analysis_config.ser_model_name}", load_huggingface_pipeline_func, task="audio-classification", model_name=analysis_config.ser_model_name, device=device, top_k=1)
            if ser_pipeline:
                sr_audio = master_audio_data.get('sr')
                waveform_np_full = master_audio_data.get("raw_features", {}).get("waveform_np_for_ser")
                if waveform_np_full is None: raise ValueError("Audio waveform missing for SER.")
                ser_batch_size = analysis_config.lvre_batch_size_ser; pbar_ser = tqdm(total=len(timed_lyrics), desc="SER Processing", leave=False, disable=not TQDM_AVAILABLE)
                for i in range(0, len(timed_lyrics), ser_batch_size):
                    batch_lyrics = timed_lyrics[i:i+ser_batch_size]; batch_audio_snippets = []; valid_indices_in_batch = []
                    for k, word_info in enumerate(batch_lyrics):
                        ts = word_info.get('timestamp'); word_text = word_info.get('text', '').strip()
                        if isinstance(ts, (tuple, list)) and len(ts) == 2 and word_text and isinstance(ts[0], (int, float)) and isinstance(ts[1], (int, float)) and ts[1] > ts[0]:
                             start_t, end_t = ts; start_sample = int(start_t * sr_audio); end_sample = min(int(end_t * sr_audio), waveform_np_full.shape[-1]); start_sample = max(0, min(start_sample, end_sample - 1))
                             if start_sample < end_sample: batch_audio_snippets.append(waveform_np_full[start_sample:end_sample]); valid_indices_in_batch.append(k)
                             else: word_info['emotion_score'] = 0.0
                        else: word_info['emotion_score'] = 0.0
                    if batch_audio_snippets:
                        try:
                             snippets_float32 = [s.astype(np.float32) for s in batch_audio_snippets]; batch_results = ser_pipeline(snippets_float32, sampling_rate=sr_audio)
                             if isinstance(batch_results, list) and len(batch_results) == len(batch_audio_snippets):
                                 for res_list, orig_idx in zip(batch_results, valid_indices_in_batch):
                                     top_result = res_list[0] if isinstance(res_list, list) and res_list else (res_list if isinstance(res_list, dict) else None)
                                     if isinstance(top_result, dict): score = top_result.get('score', 0.0); batch_lyrics[orig_idx]['emotion_score'] = float(score)
                                     else: batch_lyrics[orig_idx]['emotion_score'] = 0.0
                             else: logger.warning(f"SER batch output format mismatch."); [batch_lyrics[k].setdefault('emotion_score', 0.0) for k in valid_indices_in_batch]
                        except Exception as ser_batch_err: logger.error(f"SER batch {i//ser_batch_size} failed: {ser_batch_err}"); [batch_lyrics[k].setdefault('emotion_score', 0.0) for k in valid_indices_in_batch]
                    for k in range(len(batch_lyrics)): batch_lyrics[k].setdefault('emotion_score', 0.0) # Ensure key exists
                    if pbar_ser: pbar_ser.update(len(batch_lyrics))
                if pbar_ser: pbar_ser.close()
                # Optionally clear waveform from memory if large
                # if "waveform_np_for_ser" in master_audio_data.get("raw_features", {}): del master_audio_data["raw_features"]["waveform_np_for_ser"]
                ser_success = True; logger.info("SER complete.")
            else: raise RuntimeError("SER pipeline failed to load.")
        except Exception as ser_err: logger.error(f"SER processing failed: {ser_err}", exc_info=True); [w.setdefault('emotion_score', 0.0) for w in timed_lyrics]
        finally:
             if not ser_success: [w.setdefault('emotion_score', 0.0) for w in timed_lyrics] # Default on any failure
    else: logger.info("Skipping SER (timed_lyrics missing or scores already present).")

    # --- 3. Embed Lyrics ---
    run_text_embed = timed_lyrics and not all('embedding' in w for w in timed_lyrics)
    if run_text_embed:
        if not SENTENCE_TRANSFORMERS_AVAILABLE: logger.error("Cannot embed lyrics: sentence-transformers not found.")
        else:
            logger.info("Embedding lyrics..."); text_embed_success = False
            try:
                text_model, text_model_device = get_pytorch_model(analysis_config.text_embed_model_name, load_sentence_transformer_func, analysis_config.text_embed_model_name)
                if text_model:
                    all_texts = [word_info.get('text', '').strip().lower() for word_info in timed_lyrics]; valid_indices = [i for i, t in enumerate(all_texts) if t]; valid_texts = [all_texts[i] for i in valid_indices]
                    if valid_texts:
                        logger.debug(f"Encoding {len(valid_texts)} non-empty lyric words...")
                        text_batch_size = analysis_config.lvre_batch_size_text
                        embeddings = text_model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=TQDM_AVAILABLE, batch_size=text_batch_size, device=text_model_device)
                        for valid_idx, emb in zip(valid_indices, embeddings): timed_lyrics[valid_idx]['embedding'] = emb.tolist() # Store as list
                    for i in range(len(timed_lyrics)): timed_lyrics[i].setdefault('embedding', None) # Ensure key exists
                    text_embed_success = True; logger.info("Lyrical embeddings generated.")
                else: raise RuntimeError("Sentence Transformer model failed to load.")
            except Exception as text_emb_err: logger.error(f"Text embedding failed: {text_emb_err}", exc_info=True); [w.setdefault('embedding', None) for w in timed_lyrics]
            finally:
                 if not text_embed_success: [w.setdefault('embedding', None) for w in timed_lyrics] # Default on failure
    else: logger.info("Skipping Text Embeddings (timed_lyrics missing or embeddings present).")

    # --- 4. Embed Visual Clips ---
    clips_needing_embed = [c for c in all_potential_clips if not hasattr(c, 'visual_embedding') or c.visual_embedding is None]
    if clips_needing_embed:
        if not TRANSFORMERS_AVAILABLE: logger.error("Cannot embed visuals: transformers (CLIP) not found.")
        else:
            logger.info(f"Generating visual embeddings for {len(clips_needing_embed)} clips..."); vis_embed_success = False; source_readers: Dict[str, Optional[VideoFileClip]] = {}
            try:
                processor_cache_key = f"clip_proc_{analysis_config.vision_embed_model_name}"; model_cache_key = f"clip_model_{analysis_config.vision_embed_model_name}"; clip_processor = get_pytorch_processor(processor_cache_key, load_huggingface_processor_func, analysis_config.vision_embed_model_name, CLIPProcessor); clip_model, clip_model_device = get_pytorch_model(model_cache_key, load_huggingface_model_func, analysis_config.vision_embed_model_name, CLIPModel)
                if clip_model and clip_processor and clip_model_device:
                    keyframes_to_process = []; clip_indices_map = {id(c): i for i, c in enumerate(all_potential_clips)}
                    logger.debug("Extracting keyframes for visual embedding..."); pbar_kf = tqdm(total=len(clips_needing_embed), desc="Extract Keyframes", leave=False, disable=not TQDM_AVAILABLE)
                    for clip in clips_needing_embed:
                        source_path = str(clip.source_video_path);
                        if not source_path or not os.path.exists(source_path): logger.warning(f"Skip keyframe clip {clip.start_frame}: Invalid path '{source_path}'"); clip.visual_embedding = None; if pbar_kf: pbar_kf.update(1); continue
                        reader = source_readers.get(source_path)
                        if reader is None and source_path not in source_readers: # Open once per path
                            try: reader = VideoFileClip(source_path, audio=False); _ = reader.duration; source_readers[source_path] = reader
                            except Exception as read_err: source_readers[source_path] = None; logger.error(f"MoviePy failed open {os.path.basename(source_path)}: {read_err}"); reader = None
                        elif source_path in source_readers and reader is None: pass # Known failure
                        if reader:
                            try: keyframe_time = np.clip(clip.start_time + clip.duration / 2.0, 0, reader.duration - 1e-6 if reader.duration else 0); keyframe_rgb = reader.get_frame(keyframe_time);
                                if keyframe_rgb is not None and keyframe_rgb.size > 0: keyframes_to_process.append((clip_indices_map[id(clip)], keyframe_rgb))
                                else: logger.warning(f"Empty keyframe clip {clip.start_frame} @ {keyframe_time:.2f}s."); clip.visual_embedding = None
                            except Exception as kf_err: logger.warning(f"Keyframe extract failed clip {clip.start_frame}: {kf_err}"); clip.visual_embedding = None
                        else: clip.visual_embedding = None
                        if pbar_kf: pbar_kf.update(1)
                    if pbar_kf: pbar_kf.close()
                    logger.debug("Closing video readers for LVRE keyframes...");
                    for path, reader in source_readers.items():
                        if reader and hasattr(reader, 'close'):
                            try: reader.close()
                            except Exception as close_err: logger.warning(f"Minor error closing reader for {os.path.basename(path)}: {close_err}")
                    source_readers.clear()
                    if keyframes_to_process:
                        logger.debug(f"Embedding {len(keyframes_to_process)} keyframes using CLIP..."); vis_batch_size = analysis_config.lvre_batch_size_vision; pbar_emb = tqdm(total=len(keyframes_to_process), desc="Visual Embeddings", leave=False, disable=not TQDM_AVAILABLE)
                        with torch.no_grad():
                            for i in range(0, len(keyframes_to_process), vis_batch_size):
                                batch_data = keyframes_to_process[i:i+vis_batch_size]; batch_original_indices = [item[0] for item in batch_data]; batch_frames_rgb = [item[1] for item in batch_data]
                                try:
                                    inputs = clip_processor(images=batch_frames_rgb, return_tensors="pt", padding=True).to(clip_model_device); image_features = clip_model.get_image_features(**inputs); image_features = F.normalize(image_features, p=2, dim=-1); embeddings_np = image_features.cpu().numpy()
                                    for j, original_clip_index in enumerate(batch_original_indices):
                                        if 0 <= original_clip_index < len(all_potential_clips): all_potential_clips[original_clip_index].visual_embedding = embeddings_np[j].tolist() # Store as list
                                        else: logger.error(f"Invalid clip index {original_clip_index} in visual embed.")
                                except Exception as emb_err: logger.error(f"Visual embed batch {i//vis_batch_size} failed: {emb_err}"); [all_potential_clips[original_clip_index].setdefault('visual_embedding', None) for original_clip_index in batch_original_indices if 0 <= original_clip_index < len(all_potential_clips)]
                                finally:
                                     if pbar_emb: pbar_emb.update(len(batch_data))
                        if pbar_emb: pbar_emb.close()
                        vis_embed_success = True; logger.info("Visual embedding generation finished.")
                    else: logger.info("No valid keyframes for visual embedding."); [clip.setdefault('visual_embedding', None) for clip in clips_needing_embed]
                else: logger.error("CLIP model/processor failed load."); [clip.setdefault('visual_embedding', None) for clip in clips_needing_embed]
            except Exception as vis_err_outer: logger.error(f"Outer visual embed error: {vis_err_outer}", exc_info=True); [clip.setdefault('visual_embedding', None) for clip in clips_needing_embed]
            finally:
                if not vis_embed_success: [clip.setdefault('visual_embedding', None) for clip in clips_needing_embed] # Default on failure
    else: logger.info("Skipping visual embeddings (all clips have them or none needed).")
    if ENABLE_PROFILING: logger.debug(f"PROFILING: LVRE Preprocessing took {time.time() - profiler_start_time:.3f}s")

# ========================================================================
#                       STYLE ANALYSIS FUNCTION (SAAPV)
# ========================================================================
def analyze_music_style(master_audio_data: Dict, analysis_config: AnalysisConfig) -> str:
    """Analyzes basic audio features to determine a broad musical style."""
    logger.info("Analyzing music style (SAAPV)...")
    try:
        tempo = master_audio_data.get('tempo', 120.0); rms_energy = master_audio_data.get('raw_features', {}).get('rms_energy'); onset_times = master_audio_data.get('micro_beat_times', master_audio_data.get('beat_times', [])); duration = master_audio_data.get('duration', 0)
        if rms_energy is not None and not isinstance(rms_energy, np.ndarray): rms_energy = np.asarray(rms_energy, dtype=np.float32)
        if duration <= 0: return "Unknown"
        tempo_category = "Slow" if tempo < 95 else ("Medium" if tempo < 135 else "Fast")
        dynamic_range_category = "Medium"; rms_var_norm = 0.0
        if rms_energy is not None and len(rms_energy) > 1: rms_variance = np.var(rms_energy); rms_var_norm = np.clip(rms_variance / 0.05, 0.0, 1.0); dynamic_range_category = "Low" if rms_var_norm < 0.3 else ("Medium" if rms_var_norm < 0.7 else "High")
        complexity_category = "Moderate"; rhythmic_complexity = 0.5
        if onset_times and len(onset_times) > 1:
            inter_onset_intervals = np.diff(onset_times); inter_onset_intervals = inter_onset_intervals[inter_onset_intervals > 0.01]
            if len(inter_onset_intervals) > 1: ioi_variance = np.var(inter_onset_intervals); rhythmic_complexity = np.clip(1.0 - exp(-ioi_variance * 50), 0.0, 1.0); complexity_category = "Simple" if rhythmic_complexity < 0.4 else ("Moderate" if rhythmic_complexity < 0.8 else "Complex")
            else: rhythmic_complexity = 0.0; complexity_category="Simple"
        else: rhythmic_complexity = 0.0; complexity_category="Simple"
        logger.debug(f"SAAPV Style Features: Tempo={tempo:.1f} ({tempo_category}), DynRange={rms_var_norm:.2f} ({dynamic_range_category}), RhythmCmplx={rhythmic_complexity:.2f} ({complexity_category})")
        # --- Style Classification Logic ---
        if tempo_category == "Fast" and dynamic_range_category == "High" and complexity_category != "Simple": style = "High-Energy Complex (Dance/Rock/Metal)"
        elif tempo_category == "Fast" and dynamic_range_category != "Low": style = "Uptempo Pop/Electronic"
        elif tempo_category == "Slow" and dynamic_range_category == "Low" and complexity_category == "Simple": style = "Ballad/Ambient"
        elif tempo_category == "Slow": style = "Slow Groove/RnB/Chill"
        elif dynamic_range_category == "High": style = "Dynamic Rock/Orchestral"
        elif complexity_category == "Complex": style = "Complex Rhythm (Jazz/Prog)"
        else: style = "Mid-Tempo Pop/General"
        logger.info(f"Detected Music Style (SAAPV): {style}"); return style
    except Exception as e: logger.error(f"Error during music style analysis: {e}", exc_info=True); return "Unknown"

# ========================================================================
#        SCENE DETECTION HELPER (using PySceneDetect - Optional)
# ========================================================================
def detect_scenes(video_path: str, config: AnalysisConfig) -> Optional[List[Tuple[float, float]]]:
    """Detects scene boundaries using PySceneDetect if enabled."""
    if not PYSCENEDETECT_AVAILABLE or not config.use_scene_detection:
        logger.info("Scene detection skipped (disabled or PySceneDetect not found).")
        return None
    logger.info(f"Detecting scenes in {os.path.basename(video_path)}...")
    try:
        video = open_video(video_path) # Uses FFmpeg backend by default
        scene_manager = SceneManager()
        # Add ContentDetector with specified threshold
        scene_manager.add_detector(ContentDetector(threshold=config.scene_detection_threshold))
        # Set downscale factor for faster processing (e.g., 2 means process at half resolution)
        # downscale_factor = 2 # Adjust as needed - Let PySceneDetect handle it unless performance issues
        scene_manager.detect_scenes(video=video, show_progress=TQDM_AVAILABLE) # Process frames
        scene_list_tc = scene_manager.get_scene_list() # List of tuples (start_timecode, end_timecode)

        if not scene_list_tc:
            logger.warning("PySceneDetect found no scene boundaries.")
            # Get video duration safely
            video_duration_sec = 0.0
            if hasattr(video, 'duration') and video.duration:
                 try: video_duration_sec = video.duration.get_seconds()
                 except: pass # Ignore errors getting duration
            return [(0.0, video_duration_sec)] # Return one scene covering the whole video

        # Convert Timecodes to seconds
        scene_list_sec = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list_tc]
        logger.info(f"Scene detection complete: Found {len(scene_list_sec)} scenes.")
        # Release video object properly
        if hasattr(video, 'release'): video.release()
        elif hasattr(video, 'close'): video.close() # Some backends might use close
        return scene_list_sec

    except Exception as e:
        logger.error(f"PySceneDetect failed for {os.path.basename(video_path)}: {e}", exc_info=True)
        # Attempt to release video object even on error
        if 'video' in locals() and video:
            if hasattr(video, 'release'): try: video.release()
            except: pass
            elif hasattr(video, 'close'): try: video.close()
            except: pass
        return None # Return None on failure

# ========================================================================
#               CLIP SEGMENT DATA STRUCTURE (Ensemble v5.4 Ready)
# ========================================================================
class ClipSegment:
    def __init__(self, source_video_path: str, start_frame: int, end_frame: int, fps: float,
                 segment_frame_features: List[Dict], # Pass only the slice relevant to this segment
                 analysis_config: AnalysisConfig, master_audio_data: Dict):
        self.source_video_path = source_video_path
        self.start_frame = start_frame; self.end_frame = end_frame # Exclusive index
        self.num_frames = max(0, end_frame - start_frame)
        self.fps = fps if fps > 0 else 30.0
        self.duration = self.num_frames / self.fps if self.fps > 0 else 0.0
        self.start_time = start_frame / self.fps if self.fps > 0 else 0.0
        self.end_time = end_frame / self.fps if self.fps > 0 else 0.0
        self.analysis_config = analysis_config # Store config dataclass used for analysis
        self.segment_frame_features = segment_frame_features # Store the provided slice

        # === Initialize & Aggregate Features ===
        self._initialize_features()
        if self.segment_frame_features:
            self._aggregate_visual_features() # Includes scene aggregation now
            self._assign_audio_segment_features(master_audio_data)
        else:
             if self.duration > 0: logger.warning(f"Cannot aggregate features for clip {start_frame}-{end_frame}: Empty segment features.")

    def _initialize_features(self):
        """Initializes all aggregated features to default values."""
        # Heuristic / Basic Visuals
        self.avg_raw_score: float = 0.0; self.avg_boosted_score: float = 0.0; self.peak_boosted_score: float = 0.0; self.avg_motion_heuristic: float = 0.0; self.avg_jerk_heuristic: float = 0.0; self.avg_camera_motion: float = 0.0; self.face_presence_ratio: float = 0.0; self.avg_face_size: float = 0.0; self.intensity_category: str = "Low"; self.dominant_contributor: str = "none"; self.contains_beat: bool = False; self.musical_section_indices: Set[int] = set(); self.avg_lip_activity: float = 0.0
        # Physics / HEFM Visuals
        self.avg_visual_flow: float = 0.0; self.avg_visual_accel: float = 0.0; self.avg_depth_variance: float = 0.0; self.avg_visual_entropy: float = 0.0; self.avg_pose_kinetic: float = 0.0; self.avg_visual_flow_trend: float = 0.0; self.avg_visual_pose_trend: float = 0.0; self.visual_mood_vector: List[float] = [0.0, 0.0]
        # Scene Detection Info
        self.scene_indices: Set[int] = set(); self.dominant_scene_index: int = -1
        # V4 Physics Mode compatibility
        self.v_k: float = 0.0; self.a_j: float = 0.0; self.d_r: float = 0.0; self.phi: float = 0.0; self.mood_vector: List[float] = [0.0, 0.0]
        # Ensemble Scores / Embeddings (Set Externally)
        self.latent_sync_score: Optional[float] = None # PWRC (Batch scored)
        self.visual_embedding: Optional[List[float]] = None # LVRE (Preprocessed)
        # Audio Segment Features
        self.audio_segment_data: Dict = {}
        # Sequence Info (Set during building)
        self.sequence_start_time: float = 0.0; self.sequence_end_time: float = 0.0; self.chosen_duration: float = 0.0; self.chosen_effect: Optional[EffectParams] = None; self.subclip_start_time_in_source: float = 0.0; self.subclip_end_time_in_source: float = 0.0

    def _aggregate_visual_features(self):
        """Aggregates frame-level features into segment-level features, including scene info."""
        count = len(self.segment_frame_features);
        if count == 0: return
        def safe_mean(key, default=0.0, sub_dict=None):
            values = [f.get(sub_dict, {}).get(key) for f in self.segment_frame_features if isinstance(f, dict) and isinstance(f.get(sub_dict) if sub_dict else f, dict) and isinstance(f.get(sub_dict, {}).get(key), (int, float)) and np.isfinite(f.get(sub_dict, {}).get(key))]
            return float(np.mean(values)) if values else default
        # Heuristic / Basic Aggregations
        self.avg_raw_score = safe_mean('raw_score'); self.avg_boosted_score = safe_mean('boosted_score'); peak_scores = [f.get('boosted_score', -np.inf) for f in self.segment_frame_features if isinstance(f, dict) and np.isfinite(f.get('boosted_score', -np.inf))]; self.peak_boosted_score = float(np.max(peak_scores)) if peak_scores else 0.0; self.avg_motion_heuristic = safe_mean('kinetic_energy_proxy', sub_dict='pose_features'); self.avg_jerk_heuristic = safe_mean('movement_jerk_proxy', sub_dict='pose_features'); self.avg_camera_motion = safe_mean('flow_velocity'); face_sizes = [f.get('pose_features', {}).get('face_size_ratio', 0.0) for f in self.segment_frame_features if isinstance(f, dict) and f.get('pose_features', {}).get('face_size_ratio', 0.0) > 1e-3]; self.face_presence_ratio = float(len(face_sizes) / count); self.avg_face_size = float(np.mean(face_sizes)) if face_sizes else 0.0; self.contains_beat = any(f.get('is_beat_frame', False) for f in self.segment_frame_features if isinstance(f, dict)); self.musical_section_indices = {f.get('musical_section_index', -1) for f in self.segment_frame_features if isinstance(f, dict) and f.get('musical_section_index', -1) != -1}
        # Aggregate Physics/HEFM features
        self.avg_visual_flow = safe_mean('flow_velocity'); self.avg_pose_kinetic = safe_mean('kinetic_energy_proxy', sub_dict='pose_features'); self.avg_visual_accel = safe_mean('flow_acceleration'); self.avg_depth_variance = safe_mean('depth_variance'); self.avg_visual_entropy = safe_mean('histogram_entropy'); self.avg_visual_flow_trend = safe_mean('visual_flow_trend'); self.avg_visual_pose_trend = safe_mean('visual_pose_trend')
        # Aggregate Scene Indices
        scene_counts = defaultdict(int)
        for f in self.segment_frame_features:
             scene_idx = f.get('scene_index', -1)
             if scene_idx != -1: self.scene_indices.add(scene_idx); scene_counts[scene_idx] += 1
        if scene_counts: self.dominant_scene_index = max(scene_counts, key=scene_counts.get)
        # Populate V4 Physics Mode features (for compatibility)
        cfg = self.analysis_config; self.v_k = np.clip(self.avg_visual_flow / (cfg.norm_max_visual_flow + 1e-6), 0.0, 1.0); self.a_j = np.clip(self.avg_visual_accel / (DEFAULT_NORM_MAX_JERK / 1000 + 1e-6), 0.0, 1.0); self.d_r = np.clip(self.avg_depth_variance / (cfg.norm_max_depth_variance + 1e-6), 0.0, 1.0); self.phi = self.avg_visual_entropy / (log(256) + 1e-6) if self.avg_visual_entropy > 0 else 0.0; self.mood_vector = [float(self.v_k), float(1.0 - self.d_r)]; self.visual_mood_vector = self.mood_vector.copy()
        # Determine dominant contributor & intensity (reuse V4 heuristic helpers)
        dominant_contribs = [f.get('dominant_contributor', 'none') for f in self.segment_frame_features if isinstance(f, dict)]; non_none = [c for c in dominant_contribs if c != 'none']; self.dominant_contributor = max(set(non_none), key=non_none.count) if non_none else 'none'; intensities = [f.get('intensity_category', 'Low') for f in self.segment_frame_features if isinstance(f, dict)]; intensity_order = ['Low', 'Medium', 'High']; indices = [intensity_order.index(i) for i in intensities if i in intensity_order]; self.intensity_category = intensity_order[max(indices)] if indices else 'Low'

    def _assign_audio_segment_features(self, master_audio_data: Dict):
        """Finds the audio segment corresponding to the clip midpoint."""
        if not master_audio_data: self.audio_segment_data = {}; return
        mid_time = self.start_time + self.duration / 2.0; audio_segments = master_audio_data.get('segment_features', []); matched_segment = None;
        if not audio_segments: logger.warning("No audio segments available."); self.audio_segment_data = {}; return
        for seg in audio_segments:
            if seg['start'] <= mid_time < seg['end']: matched_segment = seg; break
        if matched_segment is None and mid_time >= audio_segments[-1]['end'] - 1e-6: matched_segment = audio_segments[-1]
        if matched_segment: self.audio_segment_data = matched_segment.copy()
        else: logger.warning(f"No matching audio segment for clip at midpoint {mid_time:.2f}s"); self.audio_segment_data = {}

    # --- V4 Physics Mode Helpers (kept for compatibility) ---
    def clip_audio_fit(self, audio_segment_data, analysis_config):
        if not audio_segment_data: return 0.0; cfg = analysis_config; b_i = audio_segment_data.get('b_i', 0.0); e_i = audio_segment_data.get('e_i', 0.0); m_i_aud = np.asarray(audio_segment_data.get('m_i', [0.0, 0.0])); m_i_vid = np.asarray(self.mood_vector); diff_v = abs(self.v_k - b_i); diff_a = abs(self.a_j - e_i); sigma_m_sq = cfg.mood_similarity_variance**2 * 2; mood_dist_sq = np.sum((m_i_vid - m_i_aud)**2); mood_sim = exp(-mood_dist_sq / (sigma_m_sq + 1e-9)); diff_m = 1.0 - mood_sim; fit_arg = (cfg.fit_weight_velocity * diff_v + cfg.fit_weight_acceleration * diff_a + cfg.fit_weight_mood * diff_m); probability = 1.0 - sigmoid(fit_arg, cfg.fit_sigmoid_steepness); return float(probability)
    def get_feature_vector(self, analysis_config): return [float(self.v_k), float(self.a_j), float(self.d_r)]
    def get_shot_type(self):
        if self.face_presence_ratio < 0.1: return 'wide/no_face'
        if self.avg_face_size < 0.15: return 'medium_wide'
        if self.avg_face_size < 0.35: return 'medium'
        return 'close_up'

    def __repr__(self):
        source_basename = os.path.basename(str(self.source_video_path)) if self.source_video_path else "N/A"; sync_score_str = f"{self.latent_sync_score:.2f}" if self.latent_sync_score is not None else "N/A"; embed_str = "Yes" if self.visual_embedding is not None else "No"
        scene_str = f"Scenes:{sorted(list(self.scene_indices))}(Dom:{self.dominant_scene_index})" if self.scene_indices else "N/A"
        return (f"ClipSegment({source_basename} @ {self.fps:.1f}fps | Frames:[{self.start_frame}-{self.end_frame}] | Time:[{self.start_time:.2f}s-{self.end_time:.2f}s] | Dur:{self.duration:.2f}s)\n"
                f"  Visual: Flow:{self.avg_visual_flow:.1f}(T:{self.avg_visual_flow_trend:+.2f}) | PoseKin:{self.avg_pose_kinetic:.1f}(T:{self.avg_visual_pose_trend:+.2f}) | FaceSz:{self.avg_face_size:.2f} ({self.face_presence_ratio*100:.0f}%)\n"
                f"  Audio Seg: RMS:{self.audio_segment_data.get('rms_avg', 0):.3f} | Trend:{self.audio_segment_data.get('trend_long', 0):+.4f}\n"
                f"  Scene Info: {scene_str}\n"
                f"  Ensemble: SyncNet:{sync_score_str} | VisEmbed:{embed_str} | Shot:{self.get_shot_type()} | Intensity:{self.intensity_category}")

# ========================================================================
#      VIDEO ANALYSIS MAIN CLASS (VideousMain - Ensemble v5.4 Ready)
# ========================================================================
# --- Cache Helper ---
def _generate_frame_cache_path(video_path: str, config: AnalysisConfig, cache_dir: str) -> str:
    """Generates a consistent cache path for visual features."""
    try:
        hasher = hashlib.sha256()
        # Hash based on video path and relevant config settings
        hasher.update(str(video_path).encode())
        hasher.update(str(config.resolution_width).encode())
        hasher.update(str(config.resolution_height).encode())
        hasher.update(str(config.model_complexity).encode()) # Pose complexity affects features
        hasher.update(str(config.use_latent_sync).encode()) # Affects whether mouth crops are stored
        hasher.update(str(config.use_scene_detection).encode()) # Scene detection affects features
        hasher.update(str(config.scene_detection_threshold).encode()) # Include threshold if scenes used
        hasher.update(CACHE_VERSION.encode()) # Include cache version
        config_hash = hasher.hexdigest()[:16]
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        cache_file = f"visual_features_{base_name}_{config_hash}.npz"
        return os.path.join(cache_dir, cache_file)
    except Exception as e:
        logger.warning(f"Could not generate cache path hash for {video_path}: {e}")
        # Fallback to simpler name (less robust to config changes)
        return os.path.join(cache_dir, f"visual_features_{os.path.basename(video_path)}.npz")

class VideousMain:
    def __init__(self):
        self.BBZImageUtils = BBZImageUtils(); self.BBZPoseUtils = BBZPoseUtils()
        self.midas_model = None; self.midas_transform = None; self.midas_device = None # Lazy loaded

    def _ensure_midas_loaded(self, analysis_config: AnalysisConfig):
        needs_midas = (analysis_config.sequencing_mode == 'Physics Pareto MC') or (analysis_config.base_heuristic_weight > 0 and analysis_config.bh_depth_weight > 0)
        if needs_midas and self.midas_model is None:
            logger.info("MiDaS model required, attempting load...")
            model, transform, device = get_midas_model()
            if model and transform and device: self.midas_model = model; self.midas_transform = transform; self.midas_device = device; logger.info("MiDaS loaded.")
            else: logger.warning("MiDaS failed to load. Depth features disabled.")
        elif needs_midas: logger.debug("MiDaS already loaded.")
        else: logger.debug("MiDaS not required.")

    def _determine_dominant_contributor(self, norm_features_weighted):
        if not norm_features_weighted: return "unknown"; max_val = -float('inf'); dominant_key = "none"
        for key, value in norm_features_weighted.items():
            if value > max_val: max_val = value; dominant_key = key
        key_map = {'audio_energy': 'Audio', 'kinetic_proxy': 'Motion', 'jerk_proxy': 'Jerk', 'camera_motion': 'CamMove', 'face_size': 'FaceSize', 'percussive': 'Percuss', 'depth_variance': 'DepthVar'}
        return key_map.get(dominant_key, dominant_key) if max_val > 1e-4 else "none"

    def _categorize_intensity(self, score, thresholds=(0.3, 0.7)):
        if score < thresholds[0]: return "Low";
        if score < thresholds[1]: return "Medium";
        return "High"

    def calculate_base_heuristic_score(self, frame_features: Dict, analysis_config: AnalysisConfig) -> Tuple[float, str, str]:
        """Calculates the *base* heuristic score component (V4 logic)."""
        weights = {'audio_energy': analysis_config.bh_audio_weight, 'kinetic_proxy': analysis_config.bh_kinetic_weight, 'jerk_proxy': analysis_config.bh_sharpness_weight, 'camera_motion': analysis_config.bh_camera_motion_weight, 'face_size': analysis_config.bh_face_size_weight, 'percussive': analysis_config.bh_percussive_weight, 'depth_variance': analysis_config.bh_depth_weight}
        norm_params = {'rms': analysis_config.norm_max_rms + 1e-6, 'kinetic': analysis_config.norm_max_pose_kinetic + 1e-6, 'jerk': analysis_config.norm_max_jerk + 1e-6, 'cam_motion': analysis_config.norm_max_visual_flow + 1e-6, 'face_size': analysis_config.norm_max_face_size + 1e-6, 'percussive_ratio': 1.0 + 1e-6, 'depth_variance': analysis_config.norm_max_depth_variance + 1e-6}
        f = frame_features; pose_f = f.get('pose_features', {}); audio_energy = f.get('audio_energy', 0.0); kinetic_proxy = pose_f.get('kinetic_energy_proxy', 0.0); jerk_proxy = pose_f.get('movement_jerk_proxy', 0.0); camera_motion = f.get('flow_velocity', 0.0); face_size_ratio = pose_f.get('face_size_ratio', 0.0); percussive_ratio = f.get('percussive_ratio', 0.0); depth_variance = f.get('depth_variance', 0.0)
        norm_audio = np.clip(audio_energy / norm_params['rms'], 0.0, 1.0); norm_kinetic = np.clip(kinetic_proxy / norm_params['kinetic'], 0.0, 1.0); norm_jerk = np.clip(jerk_proxy / norm_params['jerk'], 0.0, 1.0); norm_cam_motion = np.clip(camera_motion / norm_params['cam_motion'], 0.0, 1.0); norm_face_size = np.clip(face_size_ratio / norm_params['face_size'], 0.0, 1.0); norm_percussive = np.clip(percussive_ratio / norm_params['percussive_ratio'], 0.0, 1.0); norm_depth_var = np.clip(depth_variance / norm_params['depth_variance'], 0.0, 1.0)
        contrib = {'audio_energy': norm_audio * weights['audio_energy'], 'kinetic_proxy': norm_kinetic * weights['kinetic_proxy'], 'jerk_proxy': norm_jerk * weights['jerk_proxy'], 'camera_motion': norm_cam_motion * weights['camera_motion'], 'face_size': norm_face_size * weights['face_size'], 'percussive': norm_percussive * weights['percussive'], 'depth_variance': norm_depth_var * weights['depth_variance']}
        weighted_contrib = {k: v for k, v in contrib.items() if abs(weights.get(k, 0)) > 1e-6}; score = sum(weighted_contrib.values()); final_score = np.clip(score, 0.0, 1.0); dominant = self._determine_dominant_contributor(weighted_contrib); intensity = self._categorize_intensity(final_score); return float(final_score), dominant, intensity

    def apply_beat_boost(self, frame_features_list: List[Dict], audio_data: Dict, video_fps: float, analysis_config: AnalysisConfig):
        """Applies boost to BASE heuristic score near beat times."""
        num_frames = len(frame_features_list);
        if num_frames == 0 or not audio_data or video_fps <= 0: return
        beat_boost_value = analysis_config.base_heuristic_weight * 0.5; boost_radius_sec = analysis_config.rhythm_beat_boost_radius_sec; boost_radius_frames = max(0, int(boost_radius_sec * video_fps)); beat_times = audio_data.get('beat_times', [])
        if not beat_times or beat_boost_value <= 0: return
        boost_frame_indices = set();
        for t in beat_times: beat_frame_center = int(round(t * video_fps));
            for r in range(-boost_radius_frames, boost_radius_frames + 1): idx = beat_frame_center + r;
                if 0 <= idx < num_frames: boost_frame_indices.add(idx)
        for i, features in enumerate(frame_features_list):
            if not isinstance(features, dict): continue; is_beat = i in boost_frame_indices; features['is_beat_frame'] = is_beat; boost = beat_boost_value if is_beat else 0.0; raw_score = features.get('raw_score', 0.0); features['boosted_score'] = min(raw_score + boost, 1.0)

    def get_feature_at_time(self, times_array, values_array, target_time):
        if times_array is None or values_array is None or len(times_array) == 0 or len(times_array) != len(values_array): logger.debug(f"Interpolation skipped for time {target_time:.3f}."); return 0.0
        if len(times_array) == 1: return float(values_array[0])
        try:
            if not np.all(np.diff(times_array) >= 0): sort_indices = np.argsort(times_array); times_array = times_array[sort_indices]; values_array = values_array[sort_indices]
            interpolated_value = np.interp(target_time, times_array, values_array, left=values_array[0], right=values_array[-1]); return float(interpolated_value) if np.isfinite(interpolated_value) else 0.0
        except Exception as e: logger.error(f"Interpolation error at time={target_time:.3f}: {e}"); return 0.0

    # ============================================================ #
    #         analyzeVideo Method (Refactored for Ensemble v5.4)   #
    # ============================================================ #
    def analyzeVideo(self, videoPath: str, analysis_config: AnalysisConfig,
                     master_audio_data: Dict, cache_dir: str) -> Tuple[Optional[List[Dict]], Optional[List[ClipSegment]]]:
        """ Analyzes video frames, using caching, generating Ensemble features. """
        logger.info(f"--- Analyzing Video (Ensemble v5.4): {os.path.basename(videoPath)} ---")
        if ENABLE_PROFILING: profiler_start_time = time.time()

        # --- Initialization ---
        TARGET_HEIGHT = analysis_config.resolution_height; TARGET_WIDTH = analysis_config.resolution_width
        all_frame_features: List[Dict] = []; potential_clips: List[ClipSegment] = []
        clip_reader: Optional[VideoFileClip] = None; pose_detector = None; face_detector_util = None; pose_context = None
        prev_gray: Optional[np.ndarray] = None; prev_flow: Optional[np.ndarray] = None
        pose_results_buffer = deque([None, None, None], maxlen=3); fps: float = 30.0; total_frames = 0
        loaded_from_cache = False

        # --- Load MiDaS (if needed) ---
        self._ensure_midas_loaded(analysis_config)

        # --- Check Visual Feature Cache ---
        frame_cache_path = None
        if analysis_config.cache_visual_features:
             frame_cache_path = _generate_frame_cache_path(videoPath, analysis_config, cache_dir)
             if os.path.exists(frame_cache_path):
                 logger.info(f"Found frame feature cache: {os.path.basename(frame_cache_path)}")
                 try:
                     with np.load(frame_cache_path, allow_pickle=True) as data:
                         # Basic validation of expected keys (add more as needed)
                         required_keys = ['timestamps', 'flow_velocity', 'fps'] # fps added
                         if not all(key in data for key in required_keys):
                             raise ValueError(f"Cache file missing required keys (e.g., {required_keys}).")

                         logger.info(f"Loading frame features from cache: {os.path.basename(frame_cache_path)}")
                         cached_timestamps = data['timestamps']
                         num_cached_frames = len(cached_timestamps)
                         temp_all_frame_features = [{} for _ in range(num_cached_frames)]

                         # Iterate through expected keys and populate features
                         feature_keys_map = { # Map cache key to feature dict structure
                             'timestamps': ('timestamp', None),
                             'flow_velocity': ('flow_velocity', None),
                             'flow_acceleration': ('flow_acceleration', None),
                             'histogram_entropy': ('histogram_entropy', None),
                             'depth_variance': ('depth_variance', None),
                             'visual_flow_trend': ('visual_flow_trend', None),
                             'visual_pose_trend': ('visual_pose_trend', None),
                             'scene_index': ('scene_index', None),
                             'raw_score': ('raw_score', None),
                             'boosted_score': ('boosted_score', None),
                             'is_beat_frame': ('is_beat_frame', None),
                             'audio_energy': ('audio_energy', None),
                             'percussive_ratio': ('percussive_ratio', None),
                             'musical_section_index': ('musical_section_index', None),
                             'kinetic_energy_proxy': ('kinetic_energy_proxy', 'pose_features'),
                             'movement_jerk_proxy': ('movement_jerk_proxy', 'pose_features'),
                             'is_mouth_open': ('is_mouth_open', 'pose_features'),
                             'face_size_ratio': ('face_size_ratio', 'pose_features'),
                             'face_center_x': ('face_center_x', 'pose_features'),
                         }

                         for cache_key, (feature_key, sub_dict_key) in feature_keys_map.items():
                             if cache_key in data:
                                 cached_array = data[cache_key]
                                 if len(cached_array) == num_cached_frames:
                                     for i in range(num_cached_frames):
                                         value = cached_array[i]
                                         if isinstance(value, np.generic): value = value.item()
                                         if isinstance(value, np.bool_): value = bool(value)
                                         if sub_dict_key:
                                             if sub_dict_key not in temp_all_frame_features[i]: temp_all_frame_features[i][sub_dict_key] = {}
                                             temp_all_frame_features[i][sub_dict_key][feature_key] = value
                                         else: temp_all_frame_features[i][feature_key] = value
                                 else: logger.warning(f"Cache length mismatch '{cache_key}' ({len(cached_array)} vs {num_cached_frames}). Skipping.")

                         if analysis_config.use_latent_sync and 'mouth_crops' in data:
                              mouth_crops_cached = data['mouth_crops']
                              if len(mouth_crops_cached) == num_cached_frames:
                                   for i in range(num_cached_frames): temp_all_frame_features[i]['mouth_crop'] = mouth_crops_cached[i]
                              else: logger.warning(f"Cache mouth crop length mismatch ({len(mouth_crops_cached)} vs {num_cached_frames}). Skipping.")

                         fps = float(data['fps'][0]) # Load FPS from cache
                         total_frames = num_cached_frames
                         all_frame_features = temp_all_frame_features
                         loaded_from_cache = True
                         logger.info(f"Successfully loaded {num_cached_frames} frames from cache.")

                 except Exception as cache_err:
                     logger.error(f"Failed load frame cache '{os.path.basename(frame_cache_path)}': {cache_err}. Re-analyzing.", exc_info=True)
                     loaded_from_cache = False; all_frame_features = []
                     if os.path.exists(frame_cache_path): try: os.remove(frame_cache_path); logger.info("Removed corrupted cache.")
                     except OSError as del_err: logger.warning(f"Failed remove cache: {del_err}")

        # --- Run Full Analysis if Not Loaded from Cache ---
        if not loaded_from_cache:
            logger.info("No valid cache found or cache disabled. Running full frame analysis...")
            try:
                # --- Load Video Properties ---
                logger.debug(f"Loading video with MoviePy: {videoPath}")
                clip_reader = VideoFileClip(videoPath, audio=False); _ = clip_reader.duration # Initialize reader
                fps = clip_reader.fps if hasattr(clip_reader, 'fps') and clip_reader.fps > 0 else 30.0
                if fps <= 0 or not np.isfinite(fps): fps = 30.0; logger.warning(f"Invalid FPS ({fps}), using default 30.")
                frame_time_diff = 1.0 / fps
                total_frames = int(clip_reader.duration * fps) if clip_reader.duration and clip_reader.duration > 0 else 0
                logger.info(f"Video Properties: FPS={fps:.2f}, Frames~{total_frames}, Dur={clip_reader.duration:.2f}s")
                if total_frames <= 0 or clip_reader.duration <=0: raise ValueError(f"Video has zero duration or frames ({videoPath}).")

                # === Setup MediaPipe & Audio Refs ===
                face_detector_util = BBZFaceUtils(static_mode=False, max_faces=1, min_detect_conf=analysis_config.min_face_confidence) if MEDIAPIPE_AVAILABLE else None
                if face_detector_util is None: logger.warning("Face detection disabled (MediaPipe missing or failed).")
                pose_needed = analysis_config.use_latent_sync or \
                              (analysis_config.pwrc_weight > 0 and analysis_config.pwrc_pose_energy_weight > 0) or \
                              (analysis_config.hefm_weight > 0) or \
                              (analysis_config.base_heuristic_weight > 0 and (analysis_config.bh_kinetic_weight > 0 or analysis_config.bh_sharpness_weight > 0)) or \
                              analysis_config.sequencing_mode == "Physics Pareto MC"
                if pose_needed and MEDIAPIPE_AVAILABLE:
                    logger.debug(f"Initializing Pose (Complexity: {analysis_config.model_complexity})...");
                    try: pose_context = mp_pose.Pose(static_image_mode=False, model_complexity=analysis_config.model_complexity, min_detection_confidence=analysis_config.min_pose_confidence, min_tracking_confidence=0.5); pose_detector = pose_context.__enter__()
                    except Exception as pose_init_err: logger.error(f"Pose init failed: {pose_init_err}. Pose features disabled."); pose_detector = None; pose_context = None
                elif pose_needed and not MEDIAPIPE_AVAILABLE: logger.warning("Pose features needed but MediaPipe unavailable.")

                # Get audio feature arrays (ensure numpy)
                audio_raw_features = master_audio_data.get('raw_features', {}); audio_rms_times = audio_raw_features.get('rms_times'); audio_rms_energy = audio_raw_features.get('rms_energy'); audio_perc_times = audio_raw_features.get('perc_times', audio_rms_times); audio_perc_ratio = audio_raw_features.get('percussive_ratio', np.zeros_like(audio_perc_times) if audio_perc_times is not None else []); segment_boundaries = master_audio_data.get('segment_boundaries', [0, master_audio_data.get('duration', float('inf'))])
                if not segment_boundaries or len(segment_boundaries) < 2: segment_boundaries = [0, master_audio_data.get('duration', float('inf'))]
                for arr_key in ['audio_rms_times', 'audio_rms_energy', 'audio_perc_times', 'audio_perc_ratio']:
                     local_vars = locals();
                     if local_vars.get(arr_key) is not None and not isinstance(local_vars[arr_key], np.ndarray): local_vars[arr_key] = np.asarray(local_vars[arr_key])

                # --- Detect Scenes Upfront ---
                scene_boundaries_sec = detect_scenes(videoPath, analysis_config)

                # === Feature Extraction Loop ===
                logger.info("Processing frames & generating features (Ensemble v5.4)...")
                frame_iterator = clip_reader.iter_frames(fps=fps, dtype="uint8", logger=None)
                pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"Analyzing {os.path.basename(videoPath)}", unit="frame", dynamic_ncols=True, leave=False, disable=not TQDM_AVAILABLE)
                frame_timestamps = []; frame_flow_velocities = []; frame_pose_kinetics = []

                for frame_idx, frame_rgb in enumerate(frame_iterator):
                    if frame_rgb is None: logger.warning(f"None frame @ index {frame_idx}. Stopping."); break
                    timestamp = frame_idx / fps;
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR); image_resized_bgr = self.BBZImageUtils.resizeTARGET(frame_bgr, TARGET_HEIGHT, TARGET_WIDTH)
                    if image_resized_bgr is None or image_resized_bgr.size == 0: logger.warning(f"Frame {frame_idx} resize failed."); if pbar: pbar.update(1); continue
                    current_features = {'frame_index': frame_idx, 'timestamp': timestamp}; pose_features_dict = {}
                    current_gray = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2GRAY)
                    # --- Visual Flow, Accel, Entropy (HEFM/Physics) ---
                    flow_velocity, current_flow_field = calculate_flow_velocity(prev_gray, current_gray); flow_acceleration = calculate_flow_acceleration(prev_flow, current_flow_field, frame_time_diff); current_features['flow_velocity'] = flow_velocity; current_features['flow_acceleration'] = flow_acceleration; current_features['camera_motion'] = flow_velocity; current_features['histogram_entropy'] = calculate_histogram_entropy(current_gray); frame_timestamps.append(timestamp); frame_flow_velocities.append(flow_velocity)
                    # --- Depth Variance (MiDaS - HEFM/Physics) ---
                    depth_variance = 0.0;
                    if self.midas_model and self.midas_transform and self.midas_device:
                        try:
                            with torch.no_grad():
                                image_resized_rgb_midas = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB); input_batch = self.midas_transform(image_resized_rgb_midas).to(self.midas_device); prediction = self.midas_model(input_batch); prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=image_resized_rgb_midas.shape[:2], mode="bicubic", align_corners=False).squeeze(); depth_map = prediction.cpu().numpy(); depth_min = depth_map.min(); depth_max = depth_map.max();
                                if depth_max > depth_min + 1e-6: norm_depth = (depth_map - depth_min) / (depth_max - depth_min); depth_variance = float(np.var(norm_depth))
                        except Exception as midas_e: if frame_idx % 100 == 0: logger.debug(f"MiDaS error frame {frame_idx}: {midas_e}")
                    current_features['depth_variance'] = depth_variance
                    # --- Face & Pose Processing ---
                    face_results = face_detector_util.process_frame(image_resized_bgr) if face_detector_util else None; current_pose_results = None
                    if pose_detector:
                        try: image_resized_rgb_pose = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB); image_resized_rgb_pose.flags.writeable = False; current_pose_results = pose_detector.process(image_resized_rgb_pose); image_resized_rgb_pose.flags.writeable = True
                        except Exception as pose_err: if frame_idx % 100 == 0: logger.debug(f"Pose error frame {frame_idx}: {pose_err}")
                    pose_results_buffer.append(current_pose_results); lm_t2, lm_t1, lm_t = pose_results_buffer
                    # --- Mouth Crop (PWRC) ---
                    mouth_crop_np = None;
                    if analysis_config.use_latent_sync and face_results and face_results.multi_face_landmarks: mouth_crop_np = extract_mouth_crop(image_resized_bgr, face_results.multi_face_landmarks[0])
                    current_features['mouth_crop'] = mouth_crop_np # Store numpy array (or None)
                    # --- Pose Kinetic & Jerk (HEFM/PWRC/V4) ---
                    kinetic = calculate_kinetic_energy_proxy(lm_t1, lm_t, frame_time_diff); jerk = calculate_movement_jerk_proxy(lm_t2, lm_t1, lm_t, frame_time_diff); pose_features_dict['kinetic_energy_proxy'] = kinetic; pose_features_dict['movement_jerk_proxy'] = jerk; frame_pose_kinetics.append(kinetic)
                    # --- Face Size & Mouth Open (V4 Heuristic) ---
                    is_mouth_open, face_size_ratio, face_center_x = face_detector_util.get_heuristic_face_features(face_results, TARGET_HEIGHT, TARGET_WIDTH, analysis_config.mouth_open_threshold) if face_detector_util else (False, 0.0, 0.5); pose_features_dict['is_mouth_open'] = is_mouth_open; pose_features_dict['face_size_ratio'] = face_size_ratio; pose_features_dict['face_center_x'] = face_center_x; current_features['pose_features'] = pose_features_dict
                    # --- Assign Scene Index ---
                    scene_idx = -1
                    if scene_boundaries_sec:
                        for idx, (start_sec, end_sec) in enumerate(scene_boundaries_sec):
                            # Use >= start and < end for non-overlapping check
                            if start_sec <= timestamp < end_sec: scene_idx = idx; break
                        # If exactly at or after last scene end time, assign to last scene index
                        if scene_idx == -1 and timestamp >= scene_boundaries_sec[-1][1] - 1e-6:
                            scene_idx = len(scene_boundaries_sec) - 1
                    current_features['scene_index'] = scene_idx
                    # --- Align with Audio Features ---
                    mid_frame_time = timestamp + (frame_time_diff / 2.0); current_features['audio_energy'] = self.get_feature_at_time(audio_rms_times, audio_rms_energy, mid_frame_time); current_features['percussive_ratio'] = self.get_feature_at_time(audio_perc_times, audio_perc_ratio, mid_frame_time)
                    # --- Musical Section Index ---
                    section_idx = -1
                    for i in range(len(segment_boundaries) - 1):
                        if segment_boundaries[i] <= mid_frame_time < segment_boundaries[i+1]: section_idx = i; break
                    if section_idx == -1 and mid_frame_time >= segment_boundaries[-1] - 1e-6: section_idx = len(segment_boundaries) - 2
                    current_features['musical_section_index'] = section_idx
                    # --- Calculate Base Heuristic Score ---
                    raw_score, dominant, intensity = self.calculate_base_heuristic_score(current_features, analysis_config); current_features['raw_score'] = raw_score; current_features['dominant_contributor'] = dominant; current_features['intensity_category'] = intensity; current_features['boosted_score'] = raw_score; current_features['is_beat_frame'] = False
                    # --- Store frame features ---
                    all_frame_features.append(current_features)
                    # --- Update previous states ---
                    prev_gray = current_gray.copy(); prev_flow = current_flow_field.copy() if current_flow_field is not None else None
                    if pbar: pbar.update(1)
                # --- End Frame Loop ---
                if pbar: pbar.close()

                # === Calculate Visual Trends (Post-Loop) ===
                logger.debug("Calculating visual trends...")
                if len(frame_timestamps) > 1:
                     ts_diff = np.diff(frame_timestamps);
                     if np.any(ts_diff <= 0): logger.warning("Timestamps not monotonic, visual trends set to 0."); visual_flow_trend = np.zeros(len(frame_timestamps)); visual_pose_trend = np.zeros(len(frame_timestamps));
                     else: visual_flow_trend = np.gradient(frame_flow_velocities, frame_timestamps); visual_pose_trend = np.gradient(frame_pose_kinetics, frame_timestamps)
                     for i in range(len(all_frame_features)): all_frame_features[i]['visual_flow_trend'] = float(visual_flow_trend[i]) if i < len(visual_flow_trend) else 0.0; all_frame_features[i]['visual_pose_trend'] = float(visual_pose_trend[i]) if i < len(visual_pose_trend) else 0.0
                else:
                     for i in range(len(all_frame_features)): all_frame_features[i]['visual_flow_trend'] = 0.0; all_frame_features[i]['visual_pose_trend'] = 0.0
                logger.debug("Visual trend calculation complete.")

                # === Save features to cache ===
                if analysis_config.cache_visual_features and frame_cache_path:
                    logger.info(f"Saving frame features to cache: {os.path.basename(frame_cache_path)}")
                    try:
                        cache_data = {}
                        all_keys = set().union(*(f.keys() for f in all_frame_features))
                        sub_keys = set().union(*(set((k, sk) for sk in v.keys()) for f in all_frame_features for k, v in f.items() if isinstance(v, dict)))

                        # Define expected data types for robust conversion
                        expected_types = { 'timestamp': float, 'flow_velocity': float, 'flow_acceleration': float,
                                           'histogram_entropy': float, 'depth_variance': float, 'visual_flow_trend': float,
                                           'visual_pose_trend': float, 'scene_index': int, 'raw_score': float,
                                           'boosted_score': float, 'is_beat_frame': bool, 'audio_energy': float,
                                           'percussive_ratio': float, 'musical_section_index': int,
                                           'kinetic_energy_proxy': float, 'movement_jerk_proxy': float,
                                           'is_mouth_open': bool, 'face_size_ratio': float, 'face_center_x': float }

                        # Store non-dict features
                        for key in all_keys:
                            if key not in ['pose_features', 'mouth_crop']:
                                default_val = 0.0 if expected_types.get(key, float) in [float, int] else False
                                feature_list = [f.get(key, default_val) for f in all_frame_features]
                                try: cache_data[key] = np.array(feature_list, dtype=expected_types.get(key, np.float32))
                                except TypeError as type_err: logger.warning(f"Type mismatch caching '{key}': {type_err}. Skipping.")
                                except ValueError as val_err: logger.warning(f"Value error caching '{key}': {val_err}. Skipping.")

                        # Store sub-dict features
                        for parent_key, sub_key in sub_keys:
                            cache_key_name = f"{parent_key}_{sub_key}"
                            expected_type = expected_types.get(sub_key, float) # Get type from sub_key
                            default_val = 0.0 if expected_type in [float, int] else False
                            feature_list = [f.get(parent_key, {}).get(sub_key, default_val) for f in all_frame_features]
                            try: cache_data[cache_key_name] = np.array(feature_list, dtype=expected_types.get(sub_key, np.float32))
                            except TypeError as type_err: logger.warning(f"Type mismatch caching '{cache_key_name}': {type_err}. Skipping.")
                            except ValueError as val_err: logger.warning(f"Value error caching '{cache_key_name}': {val_err}. Skipping.")

                        if analysis_config.use_latent_sync:
                            mouth_crops_list = [f.get('mouth_crop') for f in all_frame_features]
                            cache_data['mouth_crops'] = np.array(mouth_crops_list, dtype=object)

                        cache_data['fps'] = np.array([fps]) # Store FPS used for analysis

                        np.savez_compressed(frame_cache_path, **cache_data)
                        logger.info(f"Saved visual features to cache: {os.path.basename(frame_cache_path)}")
                    except Exception as cache_save_err:
                        logger.error(f"Failed to save frame feature cache: {cache_save_err}", exc_info=True)
                        if os.path.exists(frame_cache_path): try: os.remove(frame_cache_path)
                        except OSError: pass

            except ValueError as ve: logger.error(f"ValueError during analysis loop for {videoPath}: {ve}"); return None, None
            except Exception as e: logger.error(f"Error during video analysis loop for {videoPath}: {e}", exc_info=True); return None, None
            finally:
                logger.debug(f"Cleaning up resources for {os.path.basename(videoPath)}...")
                if pose_context: try: pose_context.__exit__(None, None, None)
                except Exception as pose_close_err: logger.error(f"Pose close error: {pose_close_err}")
                if face_detector_util: face_detector_util.close()
                if clip_reader: try: clip_reader.close()
                except Exception as clip_close_err: logger.error(f"MoviePy close error: {clip_close_err}")
                logger.debug(f"Resource cleanup finished for {os.path.basename(videoPath)}.")
        # --- End of Analysis Block (Cache or Full Run) ---

        # === Post-processing & Clip Identification ===
        if not all_frame_features: logger.error(f"No features extracted or loaded for {videoPath}."); return None, None

        if not loaded_from_cache: # Apply beat boost only if features were calculated, not loaded
             logger.debug("Applying V4 beat boost to base heuristic score...")
             self.apply_beat_boost(all_frame_features, master_audio_data, fps, analysis_config)

        potential_clips: List[ClipSegment] = []
        actual_total_frames = len(all_frame_features)

        # --- Create Potential Segments (Fixed/Overlapping Chunks) ---
        logger.info("Creating potential ClipSegment instances...")
        min_clip_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES, int(analysis_config.min_potential_clip_duration_sec * fps))
        max_clip_frames = max(min_clip_frames + 1, int(analysis_config.max_sequence_clip_duration * fps * 1.2))
        step_frames = max(1, int(0.5 * fps)) # Step ~0.5 seconds

        if actual_total_frames < min_clip_frames:
            logger.warning(f"Video too short ({actual_total_frames} frames) for min potential clip length ({min_clip_frames} frames). No clips generated.")
        else:
            for start_f in range(0, actual_total_frames - min_clip_frames + 1, step_frames):
                 end_f = min(start_f + max_clip_frames, actual_total_frames)
                 if (end_f - start_f) >= min_clip_frames:
                     try:
                         # Pass the relevant *slice* of features to ClipSegment
                         segment_features_slice = all_frame_features[start_f:end_f]
                         clip_seg = ClipSegment(videoPath, start_f, end_f, fps, segment_features_slice, analysis_config, master_audio_data)
                         potential_clips.append(clip_seg)
                     except Exception as clip_err: logger.warning(f"Failed create ClipSegment {start_f}-{end_f}: {clip_err}", exc_info=False)

        analysis_duration = time.time() - profiler_start_time if ENABLE_PROFILING else 0
        logger.info(f"--- Analysis & Clip Creation complete for {os.path.basename(videoPath)} ({analysis_duration:.2f}s) ---")
        logger.info(f"Created {len(potential_clips)} potential clips.")
        if ENABLE_PROFILING: logger.debug(f"PROFILING: Video Analysis ({os.path.basename(videoPath)}) took {analysis_duration:.3f}s")

        # Return None for features (handled internally/cached), and list of ClipSegments
        return None, potential_clips

    def saveAnalysisData(self, video_path: str, potential_clips: List[ClipSegment], output_dir: str, analysis_config: AnalysisConfig):
        """Saves essential ClipSegment metadata (no frame features)."""
        # Note: Visual frame features are saved in .npz in analyzeVideo if caching enabled.
        # Audio features are saved in JSON cache in _analyze_master_audio.
        # This function now only saves the list of potential clips' metadata if needed for debugging/review.
        if not analysis_config.save_analysis_data or not potential_clips:
            logger.debug("Skipping clip metadata saving (disabled or no clips).")
            return

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_subdir = output_dir # analysis_subdir
        clips_path = os.path.join(out_subdir, f"{base_name}_potential_clips_metadata.json")
        logger.info(f"Saving potential clip metadata for {base_name}...")

        def sanitize_for_json(value):
            """Converts numpy types, sets, paths etc. for JSON."""
            if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)): return float(value)
            elif isinstance(value, (np.complex_, np.complex64, np.complex128)): return {'real': float(value.real), 'imag': float(value.imag)}
            elif isinstance(value, (np.ndarray,)): return None # Exclude arrays from metadata dump
            elif isinstance(value, (np.bool_)): return bool(value)
            elif isinstance(value, (np.void)): return None
            elif isinstance(value, set): return sorted([sanitize_for_json(item) for item in value])
            elif isinstance(value, dict): return {str(k): sanitize_for_json(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)): return [sanitize_for_json(item) for item in value]
            elif isinstance(value, (int, float, str, bool)) or value is None: return value
            elif isinstance(value, pathlib.Path): return str(value)
            else:
                try: json.dumps(value); return value # Check if already serializable
                except TypeError: return f"<{type(value).__name__}_NotSerializable>" # Placeholder

        try:
            os.makedirs(out_subdir, exist_ok=True)
            clips_data = []
            serializable_attrs = [ # Reduced list for metadata file
                 'source_video_path', 'start_frame', 'end_frame', 'fps', 'duration', 'start_time', 'end_time',
                 'face_presence_ratio', 'avg_face_size', 'intensity_category', 'dominant_contributor',
                 'avg_visual_flow', 'avg_pose_kinetic', 'avg_visual_flow_trend', 'avg_visual_pose_trend',
                 'dominant_scene_index', 'latent_sync_score', 'visual_embedding_present', # Save presence flag
                 'audio_segment_data' # Save aggregated audio features
            ]
            for clip in potential_clips:
                 clip_dict = {}
                 for attr in serializable_attrs:
                     if hasattr(clip, attr):
                         val = getattr(clip, attr)
                         if attr == 'source_video_path': clip_dict[attr] = os.path.basename(str(val)) if val else "N/A"
                         elif attr == 'visual_embedding_present': clip_dict[attr] = clip.visual_embedding is not None # Derived flag
                         else: sanitized_val = sanitize_for_json(val); clip_dict[attr] = sanitized_val
                     else: pass # Attribute missing is fine here
                 clips_data.append(clip_dict)

            with open(clips_path, 'w', encoding='utf-8') as f: json.dump(clips_data, f, indent=2)
            logger.info(f"Saved clip metadata to {os.path.basename(clips_path)}")

        except Exception as e: logger.error(f"ERROR saving clip metadata for {base_name}: {e}", exc_info=True)


# ========================================================================
#          SEQUENCE BUILDERS (Greedy Ensemble v5.4 - Includes Scene Penalty)
# ========================================================================
class SequenceBuilderGreedy:
    def __init__(self, all_potential_clips: List[ClipSegment], master_audio_data: Dict, analysis_config: AnalysisConfig):
        self.all_clips = all_potential_clips
        self.master_audio_data = master_audio_data
        self.analysis_config = analysis_config # Use the (potentially adapted) config
        self.target_duration = master_audio_data.get('duration', 0)
        self.segment_boundaries = master_audio_data.get('segment_boundaries', [0, self.target_duration])
        if len(self.segment_boundaries) < 2: self.segment_boundaries = [0, self.target_duration]
        self.beat_times = master_audio_data.get('beat_times', [])
        self.micro_beat_times = master_audio_data.get('micro_beat_times', [])
        self.timed_lyrics = master_audio_data.get('timed_lyrics', []) # Assumes LVRE preproc ran if needed
        # State
        self.final_sequence: List[ClipSegment] = []; self.current_time = 0.0; self.last_clip_used: Optional[ClipSegment] = None; self.sequence_history = deque(maxlen=analysis_config.saapv_history_length)

    def build_sequence(self) -> List[ClipSegment]:
        logger.info("--- Composing Sequence (Greedy Ensemble Mode v5.4) ---")
        if not self.all_clips or self.target_duration <= 0: logger.error(f"Greedy build cannot start: Clips={len(self.all_clips)}, TargetDur={self.target_duration}"); return []
        min_dur_cfg = self.analysis_config.min_sequence_clip_duration; max_dur_cfg = self.analysis_config.max_sequence_clip_duration
        eligible_clips = [c for c in self.all_clips if min_dur_cfg <= c.duration <= max_dur_cfg * 1.5] # Allow slightly longer source
        if not eligible_clips: logger.error(f"No clips match sequence duration constraints [{min_dur_cfg:.2f}s - {max_dur_cfg:.2f}s]."); return []
        logger.info(f"Starting Greedy Ensemble: {len(eligible_clips)} eligible clips (Target: {self.target_duration:.2f}s)."); available_clips = eligible_clips.copy(); pbar_seq = tqdm(total=self.target_duration, desc="Building Sequence", unit="s", disable=not TQDM_AVAILABLE)
        while self.current_time < self.target_duration - 0.05 and available_clips:
            pool_size = min(self.analysis_config.candidate_pool_size * 5, len(available_clips)); candidate_pool = random.sample(available_clips, pool_size)
            evaluated_candidates = [];
            for candidate in candidate_pool: score = self._calculate_selection_score(candidate);
                if score > -float('inf'): evaluated_candidates.append((candidate, score))
            if not evaluated_candidates: # Fallback: check all remaining
                 logger.debug(f"No suitable candidates in pool at {self.current_time:.2f}s. Checking full list..."); evaluated_candidates = []
                 for candidate in available_clips: score = self._calculate_selection_score(candidate);
                     if score > -float('inf'): evaluated_candidates.append((candidate, score))
                 if not evaluated_candidates: logger.error(f"No suitable candidates found in entire list at {self.current_time:.2f}s. Ending build."); break
            evaluated_candidates.sort(key=lambda item: item[1], reverse=True); best_candidate, max_selection_score = evaluated_candidates[0]
            chosen_duration = np.clip(best_candidate.duration, min_dur_cfg, max_dur_cfg); chosen_duration = min(chosen_duration, max(0.01, self.target_duration - self.current_time)); chosen_duration = max(0.01, chosen_duration)
            best_candidate.sequence_start_time = self.current_time; best_candidate.sequence_end_time = self.current_time + chosen_duration; best_candidate.chosen_duration = chosen_duration; best_candidate.subclip_start_time_in_source = best_candidate.start_time; best_candidate.subclip_end_time_in_source = best_candidate.start_time + chosen_duration; best_candidate.chosen_effect = EffectParams(type="cut")
            self.final_sequence.append(best_candidate); clip_basename = os.path.basename(str(best_candidate.source_video_path)) if best_candidate.source_video_path else "N/A"; logger.debug(f"  [T={self.current_time:.2f}s] + Added Clip: {clip_basename} [{best_candidate.start_frame}-{best_candidate.end_frame}] Score:{max_selection_score:.3f} -> Dur:{chosen_duration:.2f}s")
            self.sequence_history.append({'duration': chosen_duration, 'shot_type': best_candidate.get_shot_type(), 'intensity': best_candidate.intensity_category, 'source_path': str(best_candidate.source_video_path), 'scene_index': best_candidate.dominant_scene_index})
            if pbar_seq: pbar_seq.update(chosen_duration); self.current_time += chosen_duration; self.last_clip_used = best_candidate; available_clips.remove(best_candidate)
        if pbar_seq: pbar_seq.close(); final_duration = self.final_sequence[-1].sequence_end_time if self.final_sequence else 0
        logger.info(f"--- Sequence Composition Complete (Greedy Ensemble) ---"); logger.info(f"Final Duration: {final_duration:.2f}s (Target: {self.target_duration:.2f}s), Clips: {len(self.final_sequence)}"); return self.final_sequence

    def _calculate_selection_score(self, clip: ClipSegment) -> float:
        cfg = self.analysis_config; total_score = 0.0; score_components = {}
        # --- 1. Base Heuristic (V4 Logic) ---
        base_score_val = clip.avg_boosted_score * cfg.base_heuristic_weight; total_score += base_score_val; score_components['Base'] = base_score_val
        # --- 2. PWRC (Performance) ---
        pwrc_score = 0.0
        if cfg.use_latent_sync and clip.latent_sync_score is not None: pwrc_score += clip.latent_sync_score * cfg.pwrc_lipsync_weight
        audio_rms_norm = np.clip(clip.audio_segment_data.get('rms_avg', 0) / (cfg.norm_max_rms + 1e-6), 0.0, 1.0); pose_kinetic_norm = np.clip(clip.avg_pose_kinetic / (cfg.norm_max_pose_kinetic + 1e-6), 0.0, 1.0); energy_diff_sq = (audio_rms_norm - pose_kinetic_norm)**2; energy_match_bonus = exp(-energy_diff_sq / (2 * 0.1**2)); pwrc_score += energy_match_bonus * cfg.pwrc_pose_energy_weight; total_score += pwrc_score * cfg.pwrc_weight; score_components['PWRC'] = pwrc_score * cfg.pwrc_weight
        # --- 3. HEFM (Energy Flow) ---
        hefm_score = 0.0; audio_trend_long = clip.audio_segment_data.get('trend_long', 0.0); vis_flow_trend = clip.avg_visual_flow_trend; vis_pose_trend = clip.avg_visual_pose_trend; flow_trend_match = 1.0 if np.sign(audio_trend_long) * np.sign(vis_flow_trend) >= 0 else 0.0; pose_trend_match = 1.0 if np.sign(audio_trend_long) * np.sign(vis_pose_trend) >= 0 else 0.0; trend_match_bonus = (flow_trend_match + pose_trend_match) / 2.0; hefm_score += trend_match_bonus * cfg.hefm_trend_match_weight; total_score += hefm_score * cfg.hefm_weight; score_components['HEFM'] = hefm_score * cfg.hefm_weight
        # --- 4. LVRE (Lyrics/Visuals/Emotion) ---
        lvre_score = 0.0;
        if cfg.use_lvre_features and self.timed_lyrics and clip.visual_embedding is not None:
            overlapping_lyrics = []; clip_mid_time = clip.start_time + clip.duration / 2.0
            for word_info in self.timed_lyrics:
                ts = word_info.get('timestamp');
                if word_info.get('embedding') is not None and isinstance(ts, (tuple, list)) and len(ts) == 2:
                    start_t, end_t = ts;
                    if isinstance(start_t, (int, float)) and isinstance(end_t, (int, float)) and end_t > start_t:
                        if max(clip.start_time, start_t) < min(clip.end_time, end_t): relevance = 1.0 / (abs(clip_mid_time - ((start_t + end_t) / 2.0)) + 0.1); word_info['relevance'] = relevance; overlapping_lyrics.append(word_info)
            if overlapping_lyrics:
                 best_lyric = max(overlapping_lyrics, key=lambda w: w.get('relevance', 0)); lyric_embed = best_lyric.get('embedding'); emotion_score = best_lyric.get('emotion_score', 0.0); semantic_sim = cosine_similarity(clip.visual_embedding, lyric_embed); semantic_bonus = (semantic_sim + 1.0) / 2.0; lvre_score += semantic_bonus * cfg.lvre_semantic_weight; emphasis_bonus = emotion_score; lvre_score += emphasis_bonus * cfg.lvre_emphasis_weight
        total_score += lvre_score * cfg.lvre_weight; score_components['LVRE'] = lvre_score * cfg.lvre_weight
        # --- 5. SAAPV (Style/Pacing/Variety/Scene) ---
        saapv_penalty = 0.0
        if cfg.use_saapv_adaptation: predictability_penalty_raw = self._calculate_predictability_penalty(); saapv_penalty += predictability_penalty_raw * cfg.saapv_predictability_weight
        if self.last_clip_used:
             if str(clip.source_video_path) == str(self.last_clip_used.source_video_path): saapv_penalty += cfg.saapv_variety_penalty_source
             shot_type = clip.get_shot_type(); last_shot_type = self.last_clip_used.get_shot_type();
             if shot_type == last_shot_type and shot_type != 'wide/no_face': saapv_penalty += cfg.saapv_variety_penalty_shot
             if clip.intensity_category == self.last_clip_used.intensity_category: saapv_penalty += cfg.saapv_variety_penalty_intensity
             # Scene Change Penalty
             if cfg.use_scene_detection and clip.dominant_scene_index != -1 and self.last_clip_used.dominant_scene_index != -1 and clip.dominant_scene_index != self.last_clip_used.dominant_scene_index: saapv_penalty += cfg.scene_change_penalty
        total_score -= saapv_penalty * cfg.saapv_weight; score_components['SAAPV_Penalty'] = -saapv_penalty * cfg.saapv_weight
        # --- 6. MRISE (Micro-Rhythm) ---
        mrise_bonus = 0.0
        if cfg.use_mrise_sync and self.micro_beat_times:
             frame_dur = 1.0 / cfg.render_fps if cfg.render_fps > 0 else 1/30.0; tolerance = frame_dur * cfg.mrise_sync_tolerance_factor; start_time = clip.start_time; nearest_micro_beat_diff = min([abs(start_time - mb) for mb in self.micro_beat_times]); mrise_bonus = exp(- (nearest_micro_beat_diff**2) / (2 * tolerance**2)) * cfg.mrise_sync_weight
        total_score += mrise_bonus * cfg.mrise_weight; score_components['MRISE'] = mrise_bonus * cfg.mrise_weight
        # --- 7. Rhythm (Beat Sync) ---
        beat_sync_bonus = 0.0
        if cfg.rhythm_beat_sync_weight > 0 and self.beat_times: nearest_beat_diff = min([abs(clip.start_time - bt) for bt in self.beat_times]);
             if nearest_beat_diff <= cfg.rhythm_beat_boost_radius_sec: beat_sync_bonus = cfg.rhythm_beat_sync_weight
        total_score += beat_sync_bonus; score_components['BeatSync'] = beat_sync_bonus
        # --- Log Breakdown (Optional Debug) ---
        # if random.random() < 0.01: logger.debug(f"Clip {clip.start_frame} Score={total_score:.4f} | Breakdown: {' '.join([f'{k}:{v:.3f}' for k, v in score_components.items()])}")
        return total_score

    def _calculate_predictability_penalty(self) -> float:
        history_len = len(self.sequence_history);
        if history_len < max(3, self.analysis_config.saapv_history_length // 2): return 0.0
        recent_edits = get_recent_history(self.sequence_history, history_len)
        durations = [e['duration'] for e in recent_edits]; duration_penalty = 0.0
        if len(durations) > 1: duration_std_dev = np.std(durations); avg_dur = np.mean(durations); norm_std_dev = duration_std_dev / (avg_dur + 1e-6); duration_penalty = sigmoid(-(norm_std_dev - 0.3) * 5.0)
        shot_types = [e['shot_type'] for e in recent_edits]; shot_type_penalty = 0.0
        if len(shot_types) > 1: num_unique_shots = len(set(shot_types)); repetition_ratio = 1.0 - (num_unique_shots / len(shot_types)); shot_type_penalty = repetition_ratio
        source_paths = [e['source_path'] for e in recent_edits]; source_penalty = 0.0
        if len(source_paths) > 1: num_unique_sources = len(set(source_paths)); source_repetition_ratio = 1.0 - (num_unique_sources / len(source_paths)); source_penalty = source_repetition_ratio
        predictability_penalty = (duration_penalty + max(shot_type_penalty, source_penalty)) / 2.0; return np.clip(predictability_penalty, 0.0, 1.0)

# ========================================================================
#          SEQUENCE BUILDER - PHYSICS PARETO MC (V4 Logic - Kept for Compatibility)
# ========================================================================
class SequenceBuilderPhysicsMC: # (Unchanged V4 logic)
    def __init__(self, all_potential_clips: List[ClipSegment], audio_data: Dict, analysis_config: AnalysisConfig):
        logger.warning("Physics Pareto MC mode uses V4 scoring logic. Ensemble features (SyncNet, LVRE, etc.) are NOT used."); self.all_clips = all_potential_clips; self.audio_data = audio_data; self.analysis_config = analysis_config; self.target_duration = audio_data.get('duration', 0); self.beat_times = audio_data.get('beat_times', []); self.audio_segments = audio_data.get('segment_features', []); self.mc_iterations = analysis_config.mc_iterations; self.min_clip_duration = analysis_config.min_sequence_clip_duration; self.max_clip_duration = analysis_config.max_sequence_clip_duration; self.w_r = analysis_config.objective_weight_rhythm; self.w_m = analysis_config.objective_weight_mood; self.w_c = analysis_config.objective_weight_continuity; self.w_v = analysis_config.objective_weight_variety; self.w_e = analysis_config.objective_weight_efficiency; self.tempo = audio_data.get('tempo', 120.0); self.beat_period = 60.0 / self.tempo if self.tempo > 0 else 0.5; self.effects: Dict[str, EffectParams] = {}
    def get_audio_segment_at(self, time):
        if not self.audio_segments: return None;
        for seg in self.audio_segments:
            if seg['start'] <= time < seg['end']: return seg
        if time >= self.audio_segments[-1]['end'] - 1e-6: return self.audio_segments[-1];
        logger.debug(f"Time {time:.2f}s outside audio segments."); return None
    def build_sequence(self) -> List[ClipSegment]:
        logger.info(f"--- Composing Sequence (Physics Pareto MC - V4 Logic - {self.mc_iterations} iters) ---");
        if not self.all_clips or not self.audio_segments or self.target_duration <= 0 or not self.effects: logger.error("Physics MC pre-conditions unmet."); return []
        eligible_clips = [c for c in self.all_clips if c.duration >= self.min_clip_duration];
        if not eligible_clips: logger.error(f"No clips meet min duration ({self.min_clip_duration:.2f}s) for Physics MC."); return []
        self.all_clips = eligible_clips; logger.info(f"Starting Physics MC: {len(self.all_clips)} eligible clips."); pareto_front: List[Tuple[List[Tuple[ClipSegment, float, EffectParams]], List[float]]] = []; successful_sims = 0; pbar_mc = tqdm(range(self.mc_iterations), desc="MC Sims (V4)", leave=False, disable=not TQDM_AVAILABLE)
        for i in pbar_mc:
            sim_seq_info = None; scores = None;
            try: sim_seq_info = self._run_stochastic_build();
                if sim_seq_info: successful_sims += 1; scores = self._evaluate_pareto(sim_seq_info);
                    if all(np.isfinite(s) for s in scores): self._update_pareto_front(pareto_front, (sim_seq_info, scores))
                    else: logger.debug(f"MC iter {i}: Invalid scores: {scores}")
            except Exception as mc_err: logger.error(f"MC sim iter {i} failed: {mc_err}", exc_info=False)
        if pbar_mc: pbar_mc.close(); logger.info(f"MC sims complete: {successful_sims}/{self.mc_iterations}. Pareto front size: {len(pareto_front)}")
        if not pareto_front: logger.error("Monte Carlo yielded no valid sequences."); return []
        obj_weights = [self.w_r, self.w_m, self.w_c, self.w_v, self.w_e]; best_solution = max(pareto_front, key=lambda item: sum(s * w for s, w in zip(item[1], obj_weights)))
        logger.info(f"Chosen seq objectives (NegR, M, C, V, NegEC): {[f'{s:.3f}' for s in best_solution[1]]}")
        final_sequence_info = best_solution[0]; final_sequence_segments: List[ClipSegment] = []; current_t = 0.0
        for i, (clip, duration, effect) in enumerate(final_sequence_info):
            if not isinstance(clip, ClipSegment): logger.warning(f"Invalid item index {i}. Skipping."); continue
            clip.sequence_start_time = current_t; clip.sequence_end_time = current_t + duration; clip.chosen_duration = duration; clip.chosen_effect = effect; clip.subclip_start_time_in_source = clip.start_time; clip.subclip_end_time_in_source = min(clip.start_time + duration, clip.end_time); final_sequence_segments.append(clip); current_t += duration
        if not final_sequence_segments: logger.error("Physics MC failed to reconstruct final sequence."); return []
        logger.info("--- Sequence Composition Complete (Physics Pareto MC - V4 Logic) ---"); logger.info(f"Final Duration: {current_t:.2f}s, Clips: {len(final_sequence_segments)}"); return final_sequence_segments
    def _run_stochastic_build(self):
        sequence_info: List[Tuple[ClipSegment, float, EffectParams]] = []; current_time = 0.0; available_clip_indices = list(range(len(self.all_clips))); random.shuffle(available_clip_indices); last_clip_segment: Optional[ClipSegment] = None; num_sources = len(set(c.source_video_path for c in self.all_clips))
        while current_time < self.target_duration and available_clip_indices:
            audio_seg = self.get_audio_segment_at(current_time);
            if not audio_seg: break
            candidates_info = []; total_prob = 0.0
            for list_idx_pos, original_clip_index in enumerate(available_clip_indices):
                clip = self.all_clips[original_clip_index]; prob = clip.clip_audio_fit(audio_seg, self.analysis_config);
                if num_sources > 1 and last_clip_segment and str(clip.source_video_path) == str(last_clip_segment.source_video_path): prob *= (1.0 - self.analysis_config.variety_repetition_penalty)
                if prob > 1e-5: candidates_info.append((clip, list_idx_pos, prob)); total_prob += prob
            if not candidates_info: break
            probabilities = [p / (total_prob + 1e-9) for _, _, p in candidates_info]
            try: chosen_candidate_local_idx = random.choices(range(len(candidates_info)), weights=probabilities, k=1)[0]
            except ValueError: chosen_candidate_local_idx = random.randrange(len(candidates_info)) if candidates_info else -1;
            if chosen_candidate_local_idx == -1: break;
            chosen_clip, chosen_list_idx_pos, _ = candidates_info[chosen_candidate_local_idx]
            remaining_time = self.target_duration - current_time; chosen_duration = min(chosen_clip.duration, remaining_time, self.max_clip_duration); chosen_duration = max(chosen_duration, self.min_clip_duration if remaining_time >= self.min_clip_duration else 0.01); chosen_duration = max(0.01, chosen_duration)
            effect_options = list(self.effects.values()); efficiencies = []
            for e in effect_options: denom = e.tau * e.psi; numer = e.epsilon; eff = ((numer + 1e-9) / (denom + 1e-9)) if abs(denom) > 1e-9 else (0.0 if abs(numer) < 1e-9 else 1e9); efficiencies.append(eff)
            cut_index = next((i for i, e in enumerate(effect_options) if e.type == "cut"), -1);
            if cut_index != -1: efficiencies[cut_index] = max(efficiencies[cut_index], 1.0) * 2.0
            positive_efficiencies = [max(0, eff) for eff in efficiencies]; total_efficiency = sum(positive_efficiencies); chosen_effect = self.effects.get('cut', EffectParams(type='cut'))
            if total_efficiency > 1e-9 and effect_options:
                 effect_probs = [eff / total_efficiency for eff in positive_efficiencies]; sum_probs = sum(effect_probs);
                 if abs(sum_probs - 1.0) > 1e-6: effect_probs = [p / (sum_probs + 1e-9) for p in effect_probs]
                 try: chosen_effect = random.choices(effect_options, weights=effect_probs, k=1)[0]
                 except (ValueError, IndexError) as choice_err: logger.debug(f"Effect choice failed: {choice_err}. Defaulting cut."); chosen_effect = self.effects.get('cut', EffectParams(type='cut'))
            sequence_info.append((chosen_clip, chosen_duration, chosen_effect)); last_clip_segment = chosen_clip; current_time += chosen_duration; available_clip_indices.pop(chosen_list_idx_pos)
        final_sim_duration = sum(item[1] for item in sequence_info); return sequence_info if final_sim_duration >= self.min_clip_duration else None
    def _evaluate_pareto(self, seq_info):
        if not seq_info: return [-1e9] * 5; num_clips = len(seq_info); total_duration = sum(item[1] for item in seq_info);
        if total_duration <= 1e-6: return [-1e9] * 5
        w_r, w_m, w_c, w_v, w_e = self.w_r, self.w_m, self.w_c, self.w_v, self.w_e; sigma_m_sq = self.analysis_config.mood_similarity_variance**2 * 2; kd = self.analysis_config.continuity_depth_weight; lambda_penalty = self.analysis_config.variety_repetition_penalty; num_sources = len(set(item[0].source_video_path for item in seq_info))
        r_score_sum = 0.0; num_trans_r = 0; current_t = 0.0
        for i, (_, duration, _) in enumerate(seq_info):
            trans_time = current_t + duration;
            if i < num_clips - 1: nearest_b = self._nearest_beat_time(trans_time);
                if nearest_b is not None and self.beat_period > 1e-6: offset_norm = abs(trans_time - nearest_b) / self.beat_period; r_score_sum += offset_norm**2; num_trans_r += 1
            current_t = trans_time
        avg_sq_offset = (r_score_sum / num_trans_r) if num_trans_r > 0 else 1.0; neg_r_score = -w_r * avg_sq_offset
        m_score_sum = 0.0; mood_calcs = 0; current_t = 0.0
        for clip, duration, _ in seq_info:
            mid_time = current_t + duration / 2.0; audio_seg = self.get_audio_segment_at(mid_time);
            if audio_seg: vid_mood = np.asarray(clip.mood_vector); aud_mood = np.asarray(audio_seg.get('m_i', [0.0, 0.0])); mood_dist_sq = np.sum((vid_mood - aud_mood)**2); m_score_sum += exp(-mood_dist_sq / (sigma_m_sq + 1e-9)); mood_calcs += 1
            current_t += duration
        m_score = w_m * (m_score_sum / mood_calcs if mood_calcs > 0 else 0.0)
        c_score_sum = 0.0; num_trans_c = 0
        for i in range(num_clips - 1):
            clip1, _, effect = seq_info[i]; clip2, _, _ = seq_info[i+1]; f1 = clip1.get_feature_vector(self.analysis_config); f2 = clip2.get_feature_vector(self.analysis_config); safe_kd = max(0.0, kd); delta_e_sq = (f1[0]-f2[0])**2 + (f1[1]-f2[1])**2 + safe_kd*(f1[2]-f2[2])**2; max_delta_e_sq = 1**2 + 1**2 + safe_kd*(1**2); delta_e_norm_sq = delta_e_sq / (max_delta_e_sq + 1e-9); cont_term = (1.0 - np.sqrt(np.clip(delta_e_norm_sq, 0.0, 1.0))); c_score_sum += cont_term + effect.epsilon; num_trans_c += 1
        c_score = w_c * (c_score_sum / num_trans_c if num_trans_c > 0 else 1.0)
        valid_phis = [item[0].phi for item in seq_info if isinstance(item[0], ClipSegment) and item[0].phi is not None and np.isfinite(item[0].phi)]; avg_phi = np.mean(valid_phis) if valid_phis else 0.0; repetition_count = 0; num_trans_v = 0
        if num_sources > 1:
            for i in range(num_clips - 1): p1 = str(seq_info[i][0].source_video_path); p2 = str(seq_info[i+1][0].source_video_path);
                if p1 and p2 and p1 == p2: repetition_count += 1; num_trans_v += 1
        rep_term = lambda_penalty * (repetition_count / num_trans_v if num_trans_v > 0 else 0); max_entropy = log(256); avg_phi_norm = avg_phi / max_entropy if max_entropy > 0 else 0.0; v_score = w_v * (avg_phi_norm - rep_term)
        ec_score_sum = 0.0; cost_calcs = 0
        for _, _, effect in seq_info: psi_tau = effect.psi * effect.tau; epsilon = effect.epsilon; cost = (psi_tau + 1e-9) / (epsilon + 1e-9) if abs(epsilon) > 1e-9 else (psi_tau + 1e-9) / 1e-9; ec_score_sum += cost; cost_calcs += 1
        avg_cost = (ec_score_sum / cost_calcs if cost_calcs > 0 else 0.0); neg_ec_score = -w_e * avg_cost
        final_scores = [neg_r_score, m_score, c_score, v_score, neg_ec_score]; return [float(s) if np.isfinite(s) else -1e9 for s in final_scores]
    def _nearest_beat_time(self, time_sec):
        if not self.beat_times: return None; beat_times_arr = np.asarray(self.beat_times);
        if len(beat_times_arr) == 0: return None; closest_beat_index = np.argmin(np.abs(beat_times_arr - time_sec)); return float(beat_times_arr[closest_beat_index])
    def _update_pareto_front(self, front, new_solution):
        new_seq_info, new_scores = new_solution;
        if not isinstance(new_scores, list) or len(new_scores) != 5 or not all(isinstance(s, float) and np.isfinite(s) for s in new_scores): logger.debug(f"Skip Pareto update: invalid scores {new_scores}"); return
        dominated_indices = set(); is_dominated_by_existing = False; indices_to_check = list(range(len(front)))
        for i in reversed(indices_to_check):
            if i >= len(front): continue; existing_seq_info, existing_scores = front[i];
            if not isinstance(existing_scores, list) or len(existing_scores) != 5 or not all(isinstance(s, float) and np.isfinite(s) for s in existing_scores): logger.warning(f"Removing invalid solution index {i}: {existing_scores}"); del front[i]; continue
            if self._dominates(new_scores, existing_scores): dominated_indices.add(i)
            if self._dominates(existing_scores, new_scores): is_dominated_by_existing = True; break
        if not is_dominated_by_existing:
            for i in sorted(list(dominated_indices), reverse=True):
                 if 0 <= i < len(front): del front[i]
            front.append(new_solution)
    def _dominates(self, scores1, scores2):
        if len(scores1) != len(scores2): raise ValueError("Score lists length mismatch."); better_in_at_least_one = False
        for s1, s2 in zip(scores1, scores2):
            if s1 < s2 - 1e-9: return False;
            if s1 > s2 + 1e-9: better_in_at_least_one = True
        return better_in_at_least_one

# ========================================================================
#        MOVIEPY VIDEO BUILDING FUNCTION (Includes GFPGAN Hook)
# ========================================================================
def buildSequenceVideo(final_sequence: List[ClipSegment], output_video_path: str, master_audio_path: str, render_config: RenderConfig):
    """Builds the final video sequence using MoviePy's make_frame with optional GFPGAN."""
    logger.info(f"Rendering video '{os.path.basename(output_video_path)}' with audio '{os.path.basename(master_audio_path)}'...")
    start_time = time.time()
    if ENABLE_PROFILING: tracemalloc.start(); start_memory = tracemalloc.get_traced_memory()[0]
    if not final_sequence: logger.error("Cannot build: Empty final sequence."); raise ValueError("Empty sequence")
    if not master_audio_path or not os.path.exists(master_audio_path): logger.error(f"Master audio not found: '{master_audio_path}'"); raise FileNotFoundError("Master audio")
    if not output_video_path: logger.error("Output path not specified."); raise ValueError("Output path required")
    width = render_config.resolution_width; height = render_config.resolution_height; fps = render_config.fps
    if not isinstance(fps, (int, float)) or fps <= 0: fps = 30; logger.warning(f"Invalid render FPS {render_config.fps}, using 30.")

    # --- GFPGAN Setup (Conditional) ---
    gfpgan_enhancer = None
    if render_config.use_gfpgan_enhance and GFPGAN_AVAILABLE:
        logger.info("GFPGAN enabled. Loading model...");
        try:
            model_path = str(render_config.gfpgan_model_path) # Ensure path is string
            if not model_path: raise ValueError("GFPGAN model path not specified.")
            if not os.path.exists(model_path):
                 # Add simple download placeholder or better error
                 err_msg = f"GFPGAN model not found at '{model_path}'. Enhancement disabled. Download required model (e.g., GFPGANv1.4.pth) and place it correctly, or update the path in Render Settings."
                 logger.error(err_msg)
                 # Disable enhancement if model missing
                 render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
            else:
                # Initialize GFPGANer
                gfpgan_enhancer = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
                logger.info("GFPGAN Enhancer loaded.")
        except ImportError as imp_err: logger.error(f"GFPGAN library/dependency missing: {imp_err}. Enhancement disabled."); render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
        except FileNotFoundError as fnf_err: logger.error(f"{fnf_err}"); render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
        except ValueError as val_err: logger.error(f"{val_err}"); render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
        except Exception as gfpgan_load_err: logger.error(f"Failed to load GFPGAN model: {gfpgan_load_err}", exc_info=True); render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
    elif render_config.use_gfpgan_enhance and not GFPGAN_AVAILABLE:
        logger.warning("GFPGAN enabled in config but library not found. Enhancement disabled.")
        render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)

    # --- Pre-load Source Video Clips ---
    source_clips_dict: Dict[str, Optional[VideoFileClip]] = {}; logger.info("Pre-loading source video readers...")
    unique_source_paths = sorted(list(set(str(seg.source_video_path) for seg in final_sequence)))
    for source_path in unique_source_paths:
        if not source_path or not os.path.exists(source_path): logger.error(f"Source video missing: {source_path}. Frames black."); source_clips_dict[source_path] = None; continue
        try: logger.debug(f"Preparing reader: {os.path.basename(source_path)}"); clip_obj = VideoFileClip(source_path, audio=False); _ = clip_obj.duration;
            if not hasattr(clip_obj, 'reader') or clip_obj.reader is None: raise RuntimeError("Reader init failed."); source_clips_dict[source_path] = clip_obj
        except Exception as load_err: logger.error(f"Failed load source {os.path.basename(source_path)}: {load_err}"); source_clips_dict[source_path] = None

    # --- Define make_frame function ---
    def make_frame(t):
        active_segment = None; final_source_time = -1.0; source_path_log = "N/A"
        for segment in final_sequence:
            if segment.sequence_start_time <= t < segment.sequence_end_time + 1e-6: active_segment = segment; break
        if active_segment is None and final_sequence and abs(t - final_sequence[-1].sequence_end_time) < 1e-6: active_segment = final_sequence[-1]
        if active_segment:
            source_path = str(active_segment.source_video_path); source_path_log = os.path.basename(source_path); source_clip = source_clips_dict.get(source_path)
            if source_clip and hasattr(source_clip, 'get_frame'):
                try:
                    clip_time_in_seq = t - active_segment.sequence_start_time; source_time = active_segment.subclip_start_time_in_source + clip_time_in_seq; subclip_start = active_segment.subclip_start_time_in_source; subclip_end = active_segment.subclip_end_time_in_source; source_dur = source_clip.duration if source_clip.duration else 0; final_source_time = np.clip(source_time, 0, source_dur - 1e-6 if source_dur > 0 else 0); final_source_time = np.clip(final_source_time, subclip_start, subclip_end - 1e-6 if subclip_end > subclip_start else subclip_start)
                    frame_rgb = source_clip.get_frame(final_source_time);
                    if frame_rgb is None: raise ValueError("get_frame returned None")
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width: frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
                    # --- Apply GFPGAN Enhancement ---
                    if gfpgan_enhancer and render_config.use_gfpgan_enhance and active_segment.face_presence_ratio > 0.1: # Check ratio
                        try: _, _, restored_img = gfpgan_enhancer.enhance(frame_bgr, has_aligned=False, only_center_face=False, paste_back=True, weight=render_config.gfpgan_fidelity_weight);
                            if restored_img is not None: frame_bgr = restored_img
                            else: logger.debug(f"GFPGAN returned None @ t={t:.2f}")
                        except Exception as gfpgan_err: if int(t*10) % 50 == 0: logger.warning(f"GFPGAN enhance failed @ t={t:.2f}: {gfpgan_err}") # Log periodically
                    # --- Convert final frame back to RGB ---
                    final_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB); return final_frame_rgb
                except Exception as frame_err: logger.error(f"Error frame @ t={t:.3f} (Src: {source_path_log} @ {final_source_time:.3f}): {frame_err}", exc_info=False); return np.zeros((height, width, 3), dtype=np.uint8)
            else: logger.warning(f"Source '{source_path_log}' invalid @ t={t:.3f}. Black frame."); return np.zeros((height, width, 3), dtype=np.uint8)
        else: seq_end_time = final_sequence[-1].sequence_end_time if final_sequence else 0; logger.warning(f"Time t={t:.4f}s outside sequence [0, {seq_end_time:.4f}s]. Black frame."); return np.zeros((height, width, 3), dtype=np.uint8)

    # --- Create and Write Video ---
    master_audio = None; sequence_clip_mvp = None; temp_audio_filepath = None
    try:
        total_duration = final_sequence[-1].sequence_end_time if final_sequence else 0;
        if total_duration <= 0: raise ValueError(f"Sequence duration <= 0 ({total_duration}).")
        logger.info(f"Creating VideoClip (Dur: {total_duration:.2f}s, FPS: {fps})"); sequence_clip_mvp = VideoClip(make_frame, duration=total_duration, ismask=False); logger.debug(f"Loading master audio: {master_audio_path}"); master_audio = AudioFileClip(master_audio_path); logger.debug(f"Audio duration: {master_audio.duration:.2f}s")
        if master_audio.duration > total_duration + 1e-3: logger.info(f"Trimming audio ({master_audio.duration:.2f}s -> {total_duration:.2f}s)."); master_audio = master_audio.subclip(0, total_duration)
        elif master_audio.duration < total_duration - 1e-3: logger.warning(f"Trimming video ({total_duration:.2f}s -> {master_audio.duration:.2f}s) to audio length."); total_duration = master_audio.duration; sequence_clip_mvp = sequence_clip_mvp.set_duration(total_duration)
        if master_audio: sequence_clip_mvp = sequence_clip_mvp.set_audio(master_audio)
        else: logger.warning("No audio. Rendering silent.")
        temp_audio_filename = f"temp-audio_{int(time.time())}_{random.randint(1000,9999)}.m4a"; temp_audio_dir = os.path.dirname(output_video_path) or "."; os.makedirs(temp_audio_dir, exist_ok=True); temp_audio_filepath = os.path.join(temp_audio_dir, temp_audio_filename)
        ffmpeg_params_list = [];
        if render_config.preset: ffmpeg_params_list.extend(["-preset", str(render_config.preset)])
        if render_config.crf is not None: ffmpeg_params_list.extend(["-crf", str(render_config.crf)])
        write_params = {"codec": render_config.video_codec, "audio_codec": render_config.audio_codec, "temp_audiofile": temp_audio_filepath, "remove_temp": True, "threads": render_config.threads, "preset": None, "logger": 'bar' if TQDM_AVAILABLE else None, "write_logfile": False, "audio_bitrate": render_config.audio_bitrate, "fps": fps, "ffmpeg_params": ffmpeg_params_list if ffmpeg_params_list else None}
        logger.info(f"Writing final video..."); logger.debug(f"Write params: {write_params}")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True); sequence_clip_mvp.write_videofile(output_video_path, **write_params)
        if ENABLE_PROFILING and tracemalloc.is_tracing(): current_mem, peak_mem = tracemalloc.get_traced_memory(); end_memory = current_mem; tracemalloc.stop(); logger.info(f"Render Perf: Time:{time.time()-start_time:.2f}s, PyMem Δ:{(end_memory - start_memory)/1024**2:.2f}MB, Peak:{peak_mem/1024**2:.2f}MB")
        else: logger.info(f"Render took {time.time() - start_time:.2f} seconds.")
        logger.info(f"MoviePy rendering successful: '{output_video_path}'")
    except Exception as e:
        if ENABLE_PROFILING and tracemalloc.is_tracing(): tracemalloc.stop()
        logger.error(f"MoviePy rendering failed: {e}", exc_info=True)
        if os.path.exists(output_video_path):
            try: os.remove(output_video_path); logger.info(f"Removed incomplete output: {output_video_path}")
            except OSError as del_err: logger.warning(f"Could not remove failed output {output_video_path}: {del_err}")
        if temp_audio_filepath and os.path.exists(temp_audio_filepath):
             try: os.remove(temp_audio_filepath); logger.debug("Removed temp audio file after error.")
             except OSError as del_err: logger.warning(f"Could not remove temp audio file {temp_audio_filepath}: {del_err}")
        raise
    finally:
        logger.debug("Cleaning up MoviePy objects...");
        if sequence_clip_mvp and hasattr(sequence_clip_mvp, 'close'): try: sequence_clip_mvp.close()
        except Exception as e: logger.debug(f"Minor error close sequence_clip: {e}")
        if master_audio and hasattr(master_audio, 'close'): try: master_audio.close()
        except Exception as e: logger.debug(f"Minor error close master_audio: {e}")
        for clip_key, source_clip_obj in source_clips_dict.items():
            if source_clip_obj and hasattr(source_clip_obj, 'close'): try: source_clip_obj.close()
            except Exception as e: logger.debug(f"Minor error close source {clip_key}: {e}")
        source_clips_dict.clear(); import gc; gc.collect(); logger.debug("MoviePy clip cleanup attempt finished.")

# ========================================================================
#         WORKER FUNCTION (Adjusted for new analyzeVideo return & caching)
# ========================================================================
def process_single_video(video_path: str, master_audio_data: Dict, analysis_config: AnalysisConfig, output_dir: str, cache_dir: str) -> Tuple[str, str, List[ClipSegment]]:
    """Worker: Analyzes video using VideousMain, returns potential clips."""
    start_t = time.time(); pid = os.getpid(); thread_name = threading.current_thread().name; base_name = os.path.splitext(os.path.basename(video_path))[0]
    worker_logger = logging.getLogger(f"Worker.{pid}.{thread_name}"); # Assumes logger configured in main
    worker_logger.info(f"Starting Analysis: {base_name}")
    status = "Unknown Error"; potential_clips: List[ClipSegment] = []
    try:
        analyzer = VideousMain()
        # analyzeVideo now returns (None, List[ClipSegment] or None) and takes cache_dir
        _, potential_clips_result = analyzer.analyzeVideo(video_path, analysis_config, master_audio_data, cache_dir)

        if potential_clips_result is None: status = "Analysis Failed (returned None)"; potential_clips = []
        elif not potential_clips_result: status = "Analysis OK (0 potential clips)"; potential_clips = []
        else:
            potential_clips = [clip for clip in potential_clips_result if isinstance(clip, ClipSegment)]
            status = f"Analysis OK ({len(potential_clips)} potential clips)"
            # Metadata saving (optional, if needed beyond visual cache)
            if analysis_config.save_analysis_data and potential_clips: # Check if metadata saving specifically enabled
                 try: analyzer.saveAnalysisData(video_path, potential_clips, output_dir, analysis_config) # output_dir is analysis_subdir
                 except Exception as save_err: worker_logger.error(f"Save clip metadata failed for {base_name}: {save_err}")

    except Exception as e: status = f"Failed: {type(e).__name__}"; worker_logger.error(f"!!! FATAL WORKER ERROR analyzing {base_name} !!!", exc_info=True); potential_clips = []
    finally:
        if 'analyzer' in locals(): del analyzer # Help GC
    end_t = time.time(); worker_logger.info(f"Finished Analysis {base_name} ({status}) in {end_t - start_t:.2f}s")
    return (video_path, status, potential_clips if potential_clips is not None else [])

# ========================================================================
#                      APP INTERFACE (Updated Workflow v5.4)
# ========================================================================
class VideousApp(customtkinter.CTk, TkinterDnD.DnDWrapper if TKINTERDND2_AVAILABLE else object):
    """Main GUI application class for Videous Chef Ensemble v5.4."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if TKINTERDND2_AVAILABLE: self.TkdndVersion = TkinterDnD._require(self)
        self.title(f"Videous Chef - Ensemble v{CACHE_VERSION}"); self.geometry("950x850"); self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Core State
        self.video_files: List[str] = []; self.beat_track_path: Optional[str] = None; self.master_audio_data: Optional[Dict] = None; self.all_potential_clips: List[ClipSegment] = []; self.analysis_config: Optional[AnalysisConfig] = None; self.adapted_analysis_config: Optional[AnalysisConfig] = None; self.render_config: Optional[RenderConfig] = None; self.is_processing = False
        # Process/Thread Management
        self.processing_thread = None; self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None; self.analysis_futures: List[concurrent.futures.Future] = []; self.futures_map: Dict[concurrent.futures.Future, str] = {}; self.total_tasks = 0; self.completed_tasks = 0
        # Output Directories
        self.output_dir = f"output_videous_chef_{CACHE_VERSION}";
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.analysis_subdir = os.path.join(self.output_dir, "analysis_data"); os.makedirs(self.analysis_subdir, exist_ok=True)
            self.render_subdir = os.path.join(self.output_dir, "final_renders"); os.makedirs(self.render_subdir, exist_ok=True)
            self.audio_cache_dir = os.path.join(self.analysis_subdir, "audio_cache"); os.makedirs(self.audio_cache_dir, exist_ok=True)
            self.visual_cache_dir = os.path.join(self.analysis_subdir, "visual_cache"); os.makedirs(self.visual_cache_dir, exist_ok=True)
        except OSError as e: logger.critical(f"Failed create output dirs: {e}"); messagebox.showerror("Fatal Error", f"Could not create output directories in '{self.output_dir}'. Check permissions.", parent=self); sys.exit(1)
        # --- UI Setup ---
        try: # Fonts
            self.header_font = customtkinter.CTkFont(family="Garamond", size=28, weight="bold"); self.label_font = customtkinter.CTkFont(family="Garamond", size=14); self.button_font = customtkinter.CTkFont(family="Garamond", size=12); self.dropdown_font = customtkinter.CTkFont(family="Garamond", size=12); self.small_font = customtkinter.CTkFont(family="Garamond", size=10); self.mode_font = customtkinter.CTkFont(family="Garamond", size=13, weight="bold"); self.tab_font = customtkinter.CTkFont(family="Garamond", size=14, weight="bold"); self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")
        except Exception as font_e: logger.warning(f"Font load error: {font_e}. Using defaults."); self.header_font = customtkinter.CTkFont(size=28, weight="bold"); self.label_font = customtkinter.CTkFont(size=14); self.button_font = customtkinter.CTkFont(size=12); self.dropdown_font = customtkinter.CTkFont(size=12); self.small_font = customtkinter.CTkFont(size=10); self.mode_font = customtkinter.CTkFont(size=13, weight="bold"); self.tab_font = customtkinter.CTkFont(size=14, weight="bold"); self.separator_font = customtkinter.CTkFont(size=13, weight="bold", slant="italic")
        self._setup_luxury_theme(); self._build_ui();
        logger.info(f"VideousApp initialized (v{CACHE_VERSION}).")

    def _setup_luxury_theme(self): self.diamond_white = "#F5F6F5"; self.deep_black = "#1C2526"; self.gold_accent = "#D4AF37"; self.jewel_blue = "#2A4B7C"; customtkinter.set_appearance_mode("dark"); customtkinter.set_default_color_theme("blue"); self.configure(bg=self.deep_black)
    def _button_styles(self, border_width=1): return { "corner_radius": 8, "fg_color": self.jewel_blue, "hover_color": self.gold_accent, "border_color": self.diamond_white, "border_width": border_width, "text_color": self.diamond_white }
    def _radio_styles(self): return { "border_color": self.diamond_white, "fg_color": self.jewel_blue, "hover_color": self.gold_accent, "text_color": self.diamond_white }

    def _build_ui(self): # Layout mostly unchanged, calls updated _create_config_sliders
        self.grid_columnconfigure(0, weight=4); self.grid_columnconfigure(1, weight=3); self.grid_rowconfigure(1, weight=1);
        customtkinter.CTkLabel(self, text=f"Videous Chef - Ensemble v{CACHE_VERSION}", font=self.header_font, text_color=self.gold_accent).grid(row=0, column=0, columnspan=2, pady=(15, 10), sticky="ew")
        config_outer_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15); config_outer_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nsew"); config_outer_frame.grid_rowconfigure(0, weight=1); config_outer_frame.grid_columnconfigure(0, weight=1)
        self.tab_view = customtkinter.CTkTabview(config_outer_frame, fg_color=self.deep_black, segmented_button_fg_color=self.deep_black, segmented_button_selected_color=self.jewel_blue, segmented_button_selected_hover_color=self.gold_accent, segmented_button_unselected_color="#333", segmented_button_unselected_hover_color="#555", text_color=self.diamond_white, border_color=self.diamond_white, border_width=2); self.tab_view.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.tab_view.add("Shared"); self.tab_view.add("Ensemble Greedy"); self.tab_view.add("Physics MC (V4)"); self.tab_view.add("Render Settings")
        self.tab_view._segmented_button.configure(font=self.tab_font)
        self.shared_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Shared"), fg_color="transparent"); self.shared_tab_frame.pack(expand=True, fill="both")
        self.ensemble_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Ensemble Greedy"), fg_color="transparent"); self.ensemble_tab_frame.pack(expand=True, fill="both")
        self.physics_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Physics MC (V4)"), fg_color="transparent"); self.physics_tab_frame.pack(expand=True, fill="both")
        self.render_tab_frame = customtkinter.CTkScrollableFrame(self.tab_view.tab("Render Settings"), fg_color="transparent"); self.render_tab_frame.pack(expand=True, fill="both")
        self._create_config_sliders() # Creates sliders and controls
        files_outer_frame = customtkinter.CTkFrame(self, fg_color="transparent"); files_outer_frame.grid(row=1, column=1, padx=(5, 10), pady=5, sticky="nsew"); files_outer_frame.grid_rowconfigure(1, weight=1); files_outer_frame.grid_columnconfigure(0, weight=1)
        controls_frame = customtkinter.CTkFrame(files_outer_frame, fg_color=self.deep_black, border_color=self.diamond_white, border_width=2, corner_radius=10); controls_frame.grid(row=0, column=0, padx=0, pady=(0, 5), sticky="new")
        beat_track_frame = customtkinter.CTkFrame(controls_frame, fg_color="transparent"); beat_track_frame.pack(pady=(10, 10), padx=10, fill="x")
        customtkinter.CTkLabel(beat_track_frame, text="2. Master Audio Track (The Base)", anchor="w", font=self.label_font, text_color=self.diamond_white).pack(pady=(0, 2), anchor="w")
        beat_btn_frame = customtkinter.CTkFrame(beat_track_frame, fg_color="transparent"); beat_btn_frame.pack(fill="x")
        self.beat_track_button = customtkinter.CTkButton(beat_btn_frame, text="Select Audio/Video", font=self.button_font, command=self._select_beat_track, **self._button_styles()); self.beat_track_button.pack(side="left", padx=(0, 10))
        self.beat_track_label = customtkinter.CTkLabel(beat_btn_frame, text="No master track selected.", anchor="w", wraplength=300, font=self.small_font, text_color=self.diamond_white); self.beat_track_label.pack(side="left", fill="x", expand=True)
        video_files_frame = customtkinter.CTkFrame(files_outer_frame, fg_color=self.deep_black, border_color=self.diamond_white, border_width=2, corner_radius=10); video_files_frame.grid(row=1, column=0, padx=0, pady=5, sticky="nsew"); video_files_frame.grid_rowconfigure(1, weight=1); video_files_frame.grid_columnconfigure(0, weight=1)
        customtkinter.CTkLabel(video_files_frame, text="3. Source Videos (The Ingredients)", anchor="w", font=self.label_font, text_color=self.diamond_white).grid(row=0, column=0, columnspan=2, pady=(5, 2), padx=10, sticky="ew")
        list_frame = Frame(video_files_frame, bg=self.deep_black, highlightbackground=self.diamond_white, highlightthickness=1); list_frame.grid(row=1, column=0, columnspan=2, pady=5, padx=10, sticky="nsew"); list_frame.grid_rowconfigure(0, weight=1); list_frame.grid_columnconfigure(0, weight=1)
        self.video_listbox = Listbox(list_frame, selectmode=MULTIPLE, bg=self.deep_black, fg=self.diamond_white, borderwidth=0, highlightthickness=0, font=("Garamond", 12), selectbackground=self.jewel_blue, selectforeground=self.gold_accent); self.video_listbox.grid(row=0, column=0, sticky="nsew"); scrollbar = Scrollbar(list_frame, orient="vertical", command=self.video_listbox.yview, background=self.deep_black, troughcolor=self.jewel_blue); scrollbar.grid(row=0, column=1, sticky="ns"); self.video_listbox.configure(yscrollcommand=scrollbar.set)
        if TKINTERDND2_AVAILABLE: self.video_listbox.drop_target_register(DND_FILES); self.video_listbox.dnd_bind('<<Drop>>', self._handle_drop)
        list_button_frame = customtkinter.CTkFrame(video_files_frame, fg_color="transparent"); list_button_frame.grid(row=2, column=0, columnspan=2, pady=(5, 10), padx=10, sticky="ew"); list_button_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.add_button = customtkinter.CTkButton(list_button_frame, text="Add", width=70, font=self.button_font, command=self._add_videos_manual, **self._button_styles()); self.add_button.grid(row=0, column=0, padx=5, sticky="ew"); self.remove_button = customtkinter.CTkButton(list_button_frame, text="Remove", width=70, font=self.button_font, command=self._remove_selected_videos, **self._button_styles()); self.remove_button.grid(row=0, column=1, padx=5, sticky="ew"); self.clear_button = customtkinter.CTkButton(list_button_frame, text="Clear", width=70, font=self.button_font, command=self._clear_video_list, **self._button_styles()); self.clear_button.grid(row=0, column=2, padx=5, sticky="ew")
        self.bottom_control_frame = customtkinter.CTkFrame(self, fg_color=self.deep_black, border_color=self.diamond_white, border_width=3, corner_radius=15); self.bottom_control_frame.grid(row=2, column=0, columnspan=2, pady=(5, 5), padx=10, sticky="ew"); self.bottom_control_frame.grid_columnconfigure(0, weight=1); self.bottom_control_frame.grid_columnconfigure(1, weight=1)
        mode_inner_frame = customtkinter.CTkFrame(self.bottom_control_frame, fg_color="transparent"); mode_inner_frame.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="ew")
        customtkinter.CTkLabel(mode_inner_frame, text="Sequencing Mode:", font=self.mode_font, text_color=self.gold_accent).pack(side="left", padx=(5, 10))
        self.mode_var = tkinter.StringVar(value="Greedy Heuristic"); self.mode_selector = customtkinter.CTkSegmentedButton(mode_inner_frame, values=["Greedy Heuristic", "Physics Pareto MC"], variable=self.mode_var, font=self.button_font, selected_color=self.jewel_blue, selected_hover_color=self.gold_accent, unselected_color="#333", unselected_hover_color="#555", text_color=self.diamond_white, command=self._mode_changed); self.mode_selector.pack(side="left", expand=True, fill="x")
        self.run_button = customtkinter.CTkButton(self.bottom_control_frame, text="4. Compose Video Remix", height=45, font=customtkinter.CTkFont(family="Garamond", size=16, weight="bold"), command=self._start_processing, **self._button_styles(border_width=2)); self.run_button.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="e")
        self.status_label = customtkinter.CTkLabel(self, text=f"Ready (Ensemble v{CACHE_VERSION}).", anchor="w", font=self.button_font, text_color=self.diamond_white, wraplength=900); self.status_label.grid(row=3, column=0, columnspan=2, pady=(0, 5), padx=20, sticky="ew"); customtkinter.CTkLabel(self, text=f"Videous Chef v{CACHE_VERSION} - Ensemble Edition", font=self.small_font, text_color=self.gold_accent).grid(row=4, column=1, pady=5, padx=10, sticky="se")

    def _mode_changed(self, value): logger.info(f"Mode changed: {value}"); self.status_label.configure(text=f"Mode: {value}. Ready.", text_color=self.diamond_white)

    def _create_config_sliders(self):
        """Creates sliders and controls, including new flags and weights."""
        default_analysis_cfg = AnalysisConfig(); default_render_cfg = RenderConfig(); self.slider_vars = {}
        # Analysis Config Vars
        for field_info in AnalysisConfig.__dataclass_fields__.values():
            key = field_info.name; default_value = getattr(default_analysis_cfg, key)
            if field_info.type == bool: self.slider_vars[key] = BooleanVar(value=default_value)
            elif field_info.type == int: self.slider_vars[key] = IntVar(value=default_value)
            elif field_info.type == float: self.slider_vars[key] = tkinter.DoubleVar(value=default_value)
            elif field_info.type == str: self.slider_vars[key] = tkinter.StringVar(value=default_value)
        # Render Config UI Vars
        self.slider_vars['render_width'] = IntVar(value=default_render_cfg.resolution_width); self.slider_vars['render_height'] = IntVar(value=default_render_cfg.resolution_height); self.slider_vars['render_fps'] = IntVar(value=default_render_cfg.fps); self.slider_vars['effect_fade_duration'] = tkinter.DoubleVar(value=default_render_cfg.effect_settings.get('fade', EffectParams()).tau); self.slider_vars['use_gfpgan_enhance'] = BooleanVar(value=default_render_cfg.use_gfpgan_enhance); self.slider_vars['gfpgan_fidelity_weight'] = tkinter.DoubleVar(value=default_render_cfg.gfpgan_fidelity_weight); self.slider_vars['gfpgan_model_path'] = tkinter.StringVar(value=default_render_cfg.gfpgan_model_path)

        def add_separator(parent, text): customtkinter.CTkFrame(parent, height=2, fg_color=self.diamond_white).pack(fill="x", padx=5, pady=(15, 2)); customtkinter.CTkLabel(parent, text=text, font=self.separator_font, text_color=self.gold_accent, anchor="w").pack(fill="x", padx=5, pady=(0, 5))
        def add_checkbox(parent, label, key):
             if key not in self.slider_vars: logger.warning(f"Checkbox key '{key}' not found."); return; frame = customtkinter.CTkFrame(parent, fg_color="transparent"); frame.pack(fill="x", pady=4, padx=5); customtkinter.CTkCheckBox(frame, text=label, variable=self.slider_vars[key], font=self.label_font, text_color=self.diamond_white, hover_color=self.gold_accent, fg_color=self.jewel_blue, border_color=self.diamond_white).pack(side="left", padx=5)
        def add_slider(parent, label, key, from_val, to_val, steps, fmt="{:.2f}"):
             if key not in self.slider_vars: logger.warning(f"Slider key '{key}' not found."); return; self._create_single_slider(parent, label, self.slider_vars[key], from_val, to_val, steps, fmt)

        # --- SHARED Tab ---
        parent = self.shared_tab_frame; add_separator(parent, "--- Sequencing Constraints ---"); add_slider(parent, "Min Clip Length (s):", 'min_sequence_clip_duration', 0.2, 3.0, 28, "{:.2f}s"); add_slider(parent, "Max Clip Length (s):", 'max_sequence_clip_duration', 1.0, 10.0, 90, "{:.1f}s"); add_slider(parent, "Min Potential Clip (Analysis, s):", 'min_potential_clip_duration_sec', 0.2, 2.0, 18, "{:.1f}s")
        add_separator(parent, "--- Analysis: Detection & Caching ---"); add_slider(parent, "Min Face Certainty:", 'min_face_confidence', 0.1, 0.9, 16, "{:.2f}"); add_slider(parent, "Min Pose Certainty:", 'min_pose_confidence', 0.1, 0.9, 16, "{:.2f}")
        comp_frame_shared = customtkinter.CTkFrame(parent, fg_color="transparent"); comp_frame_shared.pack(fill="x", pady=5, padx=5); customtkinter.CTkLabel(comp_frame_shared, text="Pose Model Quality:", width=190, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5, 5)); radio_fr_shared = customtkinter.CTkFrame(comp_frame_shared, fg_color="transparent"); radio_fr_shared.pack(side="left", padx=5); customtkinter.CTkRadioButton(radio_fr_shared, text="Fast(0)", variable=self.slider_vars['model_complexity'], value=0, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3); customtkinter.CTkRadioButton(radio_fr_shared, text="Balanced(1)", variable=self.slider_vars['model_complexity'], value=1, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3); customtkinter.CTkRadioButton(radio_fr_shared, text="Accurate(2)", variable=self.slider_vars['model_complexity'], value=2, font=self.button_font, **self._radio_styles()).pack(side="left", padx=3);
        add_checkbox(parent, "Cache Audio Analysis (JSON)", 'save_analysis_data'); add_checkbox(parent, "Cache Visual Features (NPZ)", 'cache_visual_features')
        add_separator(parent, "--- Advanced: Normalization Calibration ---"); add_slider(parent, "Max Audio Loudness (RMS):", 'norm_max_rms', 0.1, 1.0, 18, "{:.1f}"); add_slider(parent, "Max Audio Attack (Onset Proxy):", 'norm_max_onset', 1.0, 50.0, 49, "{:.0f}"); add_slider(parent, "Max Pose Motion (Kinetic):", 'norm_max_pose_kinetic', 10.0, 200.0, 38, "{:.0f}"); add_slider(parent, "Max Visual Flow (Optical):", 'norm_max_visual_flow', 10.0, 200.0, 38, "{:.0f}"); add_slider(parent, "Max Face Size Ratio:", 'norm_max_face_size', 0.1, 1.0, 18, "{:.1f}")

        # --- ENSEMBLE GREEDY Tab ---
        parent = self.ensemble_tab_frame; add_separator(parent, "--- Feature Flags (Require Dependencies!) ---"); add_checkbox(parent, "Use SyncNet Lip-Sync (PWRC)", 'use_latent_sync'); add_checkbox(parent, "Use Lyrics/Emotion/CLIP (LVRE)", 'use_lvre_features'); add_checkbox(parent, "Use Music Style Adaptation (SAAPV)", 'use_saapv_adaptation'); add_checkbox(parent, "Use Micro-Rhythm Sync (MRISE)", 'use_mrise_sync'); add_checkbox(parent, "Use Scene Detection (SAAPV)", 'use_scene_detection'); add_checkbox(parent, "Use Demucs Stems (MRISE)", 'use_demucs_for_mrise'); add_checkbox(parent, "Prioritize AudioFlux (Beats/Onsets)", 'use_dl_beat_tracker')
        add_separator(parent, "--- Ensemble Component Weights ---"); add_slider(parent, "Weight: Base Heuristic (V4)", 'base_heuristic_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "Weight: Performance (PWRC)", 'pwrc_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "Weight: Energy Flow (HEFM)", 'hefm_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "Weight: Lyrics/Visuals (LVRE)", 'lvre_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "Weight: Style/Variety (SAAPV)", 'saapv_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "Weight: Micro-Rhythm (MRISE)", 'mrise_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "Weight: Beat Sync Bonus", 'rhythm_beat_sync_weight', 0.0, 0.5, 25, "{:.2f}")
        add_separator(parent, "--- PWRC Detail ---"); add_slider(parent, "PWRC: Lip Sync Importance", 'pwrc_lipsync_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "PWRC: Pose Energy Match", 'pwrc_pose_energy_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "PWRC: SyncNet Batch Size", 'syncnet_batch_size', 4, 64, 60, "{:.0f}")
        add_separator(parent, "--- HEFM Detail ---"); add_slider(parent, "HEFM: Trend Match Importance", 'hefm_trend_match_weight', 0.0, 1.0, 20, "{:.2f}")
        add_separator(parent, "--- LVRE Detail ---"); add_slider(parent, "LVRE: Semantic Match", 'lvre_semantic_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "LVRE: Vocal Emphasis Boost", 'lvre_emphasis_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "LVRE: Text Batch Size", 'lvre_batch_size_text', 16, 256, 240, "{:.0f}"); add_slider(parent, "LVRE: Vision Batch Size", 'lvre_batch_size_vision', 8, 128, 120, "{:.0f}"); add_slider(parent, "LVRE: SER Batch Size", 'lvre_batch_size_ser', 4, 64, 60, "{:.0f}")
        add_separator(parent, "--- SAAPV Detail ---"); add_slider(parent, "SAAPV: Predictability Penalty", 'saapv_predictability_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "SAAPV: History Length (Edits)", 'saapv_history_length', 2, 10, 8, "{:.0f}"); add_slider(parent, "SAAPV: Same Source Penalty", 'saapv_variety_penalty_source', 0.0, 0.5, 25, "{:.2f}"); add_slider(parent, "SAAPV: Same Shot Type Penalty", 'saapv_variety_penalty_shot', 0.0, 0.5, 25, "{:.2f}"); add_slider(parent, "SAAPV: Same Intensity Penalty", 'saapv_variety_penalty_intensity', 0.0, 0.5, 25, "{:.2f}"); add_slider(parent, "SAAPV: Scene Change Penalty", 'scene_change_penalty', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "SAAPV: Scene Detect Threshold", 'scene_detection_threshold', 10.0, 50.0, 40, "{:.1f}")
        add_separator(parent, "--- MRISE Detail ---"); add_slider(parent, "MRISE: Sync Bonus Strength", 'mrise_sync_weight', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "MRISE: Sync Tolerance (x FrameDur)", 'mrise_sync_tolerance_factor', 0.5, 5.0, 45, "{:.1f}")
        add_separator(parent, "--- Candidate Selection ---"); add_slider(parent, "Candidate Pool Size:", 'candidate_pool_size', 5, 50, 45, "{:.0f}")

        # --- PHYSICS MC (V4 Logic) Tab ---
        parent = self.physics_tab_frame; add_separator(parent, "--- V4 Clip-Audio Fit Weights ---"); add_slider(parent, "V4 Match: Motion Speed (w_v):", 'fit_weight_velocity', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "V4 Match: Motion Accel (w_a):", 'fit_weight_acceleration', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "V4 Match: Mood (w_m):", 'fit_weight_mood', 0.0, 1.0, 20, "{:.2f}"); add_slider(parent, "V4 Fit Sensitivity (k):", 'fit_sigmoid_steepness', 0.1, 5.0, 49, "{:.1f}")
        add_separator(parent, "--- V4 Pareto Objective Priorities ---"); add_slider(parent, "V4 Prio: Rhythm (w_r):", 'objective_weight_rhythm', 0.0, 2.0, 20, "{:.1f}"); add_slider(parent, "V4 Prio: Mood (w_m):", 'objective_weight_mood', 0.0, 2.0, 20, "{:.1f}"); add_slider(parent, "V4 Prio: Continuity (w_c):", 'objective_weight_continuity', 0.0, 2.0, 20, "{:.1f}"); add_slider(parent, "V4 Prio: Variety (w_v):", 'objective_weight_variety', 0.0, 2.0, 20, "{:.1f}"); add_slider(parent, "V4 Prio: Efficiency (w_e):", 'objective_weight_efficiency', 0.0, 2.0, 20, "{:.1f}")
        add_separator(parent, "--- V4 Sequence Evaluation ---"); add_slider(parent, "V4 MC Iterations:", 'mc_iterations', 100, 5000, 490, "{:d}"); add_slider(parent, "V4 Mood Tolerance (σm):", 'mood_similarity_variance', 0.05, 0.5, 18, "{:.2f}"); add_slider(parent, "V4 Continuity Depth Wt (kd):", 'continuity_depth_weight', 0.0, 1.0, 20, "{:.1f}"); add_slider(parent, "V4 Repetition Penalty (λ):", 'variety_repetition_penalty', 0.0, 1.0, 20, "{:.1f}")
        add_separator(parent, "--- V4 Effect Tuning ---"); add_slider(parent, "Fade Duration (τ, s):", 'effect_fade_duration', 0.05, 1.0, 19, "{:.2f}s")

        # --- RENDER SETTINGS Tab ---
        parent = self.render_tab_frame; add_separator(parent, "--- Output Video Settings ---"); add_slider(parent, "Render Width (pixels):", 'render_width', 640, 3840, 320, "{:.0f}"); add_slider(parent, "Render Height (pixels):", 'render_height', 360, 2160, 180, "{:.0f}"); add_slider(parent, "Render FPS:", 'render_fps', 24, 60, 36, "{:.0f}")
        add_separator(parent, "--- Face Enhancement (Optional) ---"); add_checkbox(parent, "Use GFPGAN Face Enhancement (Needs gfpgan + model)", 'use_gfpgan_enhance'); add_slider(parent, "GFPGAN Fidelity (0=Realism, 1=Quality):", 'gfpgan_fidelity_weight', 0.0, 1.0, 20, "{:.2f}")
        gfpgan_path_frame = customtkinter.CTkFrame(parent, fg_color="transparent"); gfpgan_path_frame.pack(fill="x", pady=4, padx=5); customtkinter.CTkLabel(gfpgan_path_frame, text="GFPGAN Model Path:", width=180, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5,5)); gfpgan_path_entry = customtkinter.CTkEntry(gfpgan_path_frame, textvariable=self.slider_vars['gfpgan_model_path'], font=self.small_font, width=300); gfpgan_path_entry.pack(side="left", fill="x", expand=True, padx=5)
        # End of _create_config_sliders content for Render tab

    def _create_single_slider(self, parent, label_text, variable, from_val, to_val, steps, format_str="{:.2f}"):
        """Helper to create a labeled slider row."""
        row = customtkinter.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", pady=4, padx=5)
        customtkinter.CTkLabel(row, text=label_text, width=300, anchor="w", font=self.label_font, text_color=self.diamond_white).pack(side="left", padx=(5, 5))
        val_lab = customtkinter.CTkLabel(row, text=format_str.format(variable.get()), width=70, anchor="e", font=self.button_font, text_color=self.gold_accent); val_lab.pack(side="right", padx=(5, 5))
        is_int = isinstance(variable, tkinter.IntVar); num_steps = int(steps) if steps is not None and steps > 0 else None
        cmd_lambda = (lambda v, lbl=val_lab, fmt=format_str: lbl.configure(text=fmt.format(int(round(float(v)))))) if is_int else (lambda v, lbl=val_lab, fmt=format_str: lbl.configure(text=fmt.format(float(v))))
        slider = customtkinter.CTkSlider(row, variable=variable, from_=from_val, to=to_val, number_of_steps=num_steps, command=cmd_lambda, progress_color=self.gold_accent, button_color=self.diamond_white, button_hover_color=self.gold_accent, fg_color=self.jewel_blue); slider.pack(side="left", fill="x", expand=True, padx=5)

    # --- File Handling Methods ---
    def _select_beat_track(self):
        if self.is_processing: return; filetypes = (("Audio/Video files", "*.wav *.mp3 *.aac *.flac *.ogg *.mp4 *.mov *.avi *.mkv"),("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select Master Audio Track Source", filetypes=filetypes)
        if filepath:
            try:
                if os.path.isfile(filepath): self.beat_track_path = filepath; self.beat_track_label.configure(text=os.path.basename(filepath)); logger.info(f"Master selected: {filepath}"); self.status_label.configure(text=f"Master: {os.path.basename(filepath)}", text_color=self.diamond_white)
                else: self.beat_track_path = None; self.beat_track_label.configure(text="Selection invalid."); logger.warning(f"Selected path not file: {filepath}"); tk_write(f"Invalid selection:\n{filepath}", parent=self, level="warning")
            except Exception as e: self.beat_track_path = None; self.beat_track_label.configure(text="Error checking."); logger.error(f"Path error '{filepath}': {e}"); tk_write(f"Error checking file:\n{e}", parent=self, level="error")

    def _handle_drop(self, event):
        if self.is_processing or not TKINTERDND2_AVAILABLE: return 'break'
        try:
            try: raw_paths = shlex.split(event.data.strip('{}')) # Try shlex first
            except Exception: raw_paths = self.tk.splitlist(event.data.strip('{}')) # Fallback
            filepaths = [p.strip() for p in raw_paths if p.strip()]; vid_ext = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg", ".webm"); count = 0; curr_set = set(self.video_files); skip_nonvid = 0; skip_dup = 0
            for fp in filepaths:
                fp_clean = fp.strip("'\" ");
                try:
                    if fp_clean and os.path.isfile(fp_clean):
                        if fp_clean.lower().endswith(vid_ext):
                            if fp_clean not in curr_set: self.video_files.append(fp_clean); self.video_listbox.insert(END, os.path.basename(fp_clean)); curr_set.add(fp_clean); count += 1;
                            else: skip_dup += 1;
                        else: skip_nonvid += 1;
                except Exception as file_err: logger.warning(f"Check drop path err '{fp_clean}': {file_err}")
            parts = [];
            if count > 0: parts.append(f"Added {count} video(s)")
            if skip_nonvid > 0: parts.append(f"skipped {skip_nonvid} non-video")
            if skip_dup > 0: parts.append(f"skipped {skip_dup} duplicate(s)")
            if parts: self.status_label.configure(text=f"Drop: {', '.join(parts)}."); logger.info(f"Drop: {', '.join(parts)}.")
            else: self.status_label.configure(text="Drop: No valid videos."); logger.info("Drop: No valid videos.")
        except Exception as e: logger.error(f"Drop error: {e}\nRaw: {event.data}", exc_info=True); tk_write(f"Drop error:\n{e}", parent=self, level="warning")
        return event.action

    def _add_videos_manual(self):
        if self.is_processing: return; filetypes = (("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.mpg *.mpeg *.webm"), ("All files", "*.*")); filepaths = filedialog.askopenfilename(title="Select Source Videos", filetypes=filetypes, multiple=True); count = 0; curr_set = set(self.video_files)
        if filepaths:
            for fp in filepaths:
                fp_clean = fp.strip("'\" ")
                try:
                    if fp_clean and os.path.isfile(fp_clean):
                         vid_ext = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg", ".webm")
                         if fp_clean.lower().endswith(vid_ext):
                             if fp_clean not in curr_set: self.video_files.append(fp_clean); self.video_listbox.insert(END, os.path.basename(fp_clean)); curr_set.add(fp_clean); count += 1;
                         else: logger.warning(f"Skipped non-video: {fp_clean}")
                    elif fp_clean: logger.warning(f"Selected invalid file: {fp_clean}")
                except Exception as e: logger.warning(f"Could not add file '{fp_clean}': {e}")
            if count > 0: self.status_label.configure(text=f"Added {count} video(s)."); logger.info(f"Added {count} video(s).")

    def _remove_selected_videos(self):
         if self.is_processing: return; indices = self.video_listbox.curselection()
         if not indices: self.status_label.configure(text="Select videos to remove.", text_color="yellow"); return
         removed_count = 0
         for i in sorted(list(indices), reverse=True):
             if 0 <= i < len(self.video_files):
                 try: self.video_files.pop(i); self.video_listbox.delete(i); removed_count += 1;
                 except Exception as e: logger.error(f"Remove error index {i}: {e}")
         if removed_count > 0: self.status_label.configure(text=f"Removed {removed_count} video(s).", text_color=self.diamond_white); logger.info(f"Removed {removed_count} video(s).")

    def _clear_video_list(self):
        if self.is_processing or not self.video_files: return
        if messagebox.askyesno("Confirm Clear", "Clear all source videos?", parent=self): self.video_files.clear(); self.video_listbox.delete(0, END); self.status_label.configure(text="Video list cleared.", text_color=self.diamond_white); logger.info("Cleared video list.")

    # --- Configuration Management ---
    def _get_analysis_config(self) -> AnalysisConfig:
        logger.debug("Gathering analysis configuration from UI...")
        config_dict = {}
        for key, var in self.slider_vars.items():
             if key in AnalysisConfig.__dataclass_fields__:
                 try: config_dict[key] = var.get()
                 except Exception as e: logger.warning(f"Could not get UI value for '{key}': {e}. Skipping.")
        config_dict["sequencing_mode"] = self.mode_var.get();
        try: config_dict["render_fps"] = self.slider_vars['render_fps'].get()
        except Exception: logger.warning("Could not get render_fps from UI."); config_dict["render_fps"] = 30
        try: cfg = AnalysisConfig(**config_dict); logger.debug(f"Generated AnalysisConfig: {cfg}"); self.analysis_config = cfg; return cfg
        except Exception as e: logger.error(f"Failed create AnalysisConfig: {e}", exc_info=True); tk_write("Error reading analysis config. Using defaults.", parent=self, level="error"); self.analysis_config = AnalysisConfig(); self.analysis_config.render_fps = config_dict.get('render_fps', 30); return self.analysis_config

    def _get_render_config(self) -> RenderConfig:
        logger.debug("Gathering render configuration..."); effect_settings = {"cut": EffectParams(type="cut"), "fade": EffectParams(type="fade", tau=self.slider_vars['effect_fade_duration'].get(), psi=0.1, epsilon=0.2), } # Add others
        try:
            cpu_cores = os.cpu_count() or 1; render_threads = max(1, min(cpu_cores - 1, 8)); target_w = max(16, self.slider_vars['render_width'].get()); target_h = max(16, self.slider_vars['render_height'].get()); target_fps = max(1, self.slider_vars['render_fps'].get()); use_gfpgan = self.slider_vars['use_gfpgan_enhance'].get(); gfpgan_weight = self.slider_vars['gfpgan_fidelity_weight'].get(); gfpgan_path = self.slider_vars['gfpgan_model_path'].get()
            cfg = RenderConfig(effect_settings=effect_settings, video_codec='libx264', preset='medium', crf=23, audio_codec='aac', audio_bitrate='192k', threads=render_threads, resolution_width=target_w, resolution_height=target_h, fps=target_fps, use_gfpgan_enhance=use_gfpgan, gfpgan_fidelity_weight=float(gfpgan_weight), gfpgan_model_path=str(gfpgan_path))
            logger.debug(f"Generated RenderConfig: {cfg}"); self.render_config = cfg; return cfg
        except Exception as e: logger.error(f"Failed create RenderConfig: {e}", exc_info=True); tk_write("Error reading render config. Using defaults.", parent=self, level="error"); self.render_config = RenderConfig(); return self.render_config

    def _set_ui_processing_state(self, processing: bool):
        """Disables/Enables UI elements during processing."""
        self.is_processing = processing; state = "disabled" if processing else "normal"
        widgets_to_toggle = [self.beat_track_button, self.add_button, self.remove_button, self.clear_button, self.run_button]
        for widget in widgets_to_toggle:
            if widget is not None and hasattr(widget, 'configure') and 'state' in widget.configure():
                try: widget.configure(state=state)
                except Exception as config_err: logger.warning(f"Error config widget {widget} state: {config_err}", exc_info=False)
        try:
            for tab_name in self.tab_view._name_list:
                 tab_frame_container = self.tab_view.tab(tab_name)
                 if tab_frame_container and hasattr(tab_frame_container, 'winfo_children') and tab_frame_container.winfo_children():
                     tab_frame = tab_frame_container.winfo_children()[0];
                     if tab_frame and hasattr(tab_frame, 'winfo_children'):
                         q = list(tab_frame.winfo_children())
                         while q:
                            widget = q.pop(0);
                            if widget is None: continue
                            if hasattr(widget, 'winfo_children'): q.extend(widget.winfo_children())
                            if hasattr(widget, 'configure') and isinstance(widget, (customtkinter.CTkSlider, customtkinter.CTkCheckBox, customtkinter.CTkRadioButton, customtkinter.CTkButton, customtkinter.CTkEntry)):
                                try:
                                    if 'state' in widget.configure(): widget.configure(state=state)
                                except Exception: pass
        except Exception as e: logger.error(f"Error toggling tab UI state: {e}", exc_info=True)
        if self.run_button and hasattr(self.run_button, 'configure'): self.run_button.configure(text="Chef is Cooking..." if processing else "4. Compose Video Remix")
        self.update_idletasks()

    # --- Processing Workflow (v5.4) ---
    def _start_processing(self):
        if self.is_processing: logger.warning("Processing ongoing."); return
        if not self.beat_track_path or not os.path.isfile(self.beat_track_path): tk_write("Select master audio track.", parent=self, level="warning"); return
        if not self.video_files: tk_write("Add source videos.", parent=self, level="warning"); return
        valid_videos = [f for f in self.video_files if os.path.isfile(f)]
        if len(valid_videos) != len(self.video_files): tk_write(f"Removed {len(self.video_files) - len(valid_videos)} invalid paths.", parent=self, level="warning"); self.video_files = valid_videos; self.video_listbox.delete(0, END); [self.video_listbox.insert(END, os.path.basename(vf)) for vf in self.video_files];
        if not self.video_files: tk_write("No valid videos remain.", parent=self, level="error"); return
        self.analysis_config = self._get_analysis_config(); self.render_config = self._get_render_config()
        if not self.analysis_config or not self.render_config: logger.error("Config gathering failed."); return
        self.analysis_config.render_fps = self.render_config.fps # Sync FPS
        self._set_ui_processing_state(True); self.status_label.configure(text="Chef is prepping...", text_color=self.diamond_white); self.master_audio_data = None; self.all_potential_clips = []; self.adapted_analysis_config = None; self.analysis_futures = []; self.futures_map = {}; self.total_tasks = 0; self.completed_tasks = 0
        self.shutdown_executor()
        logger.info("Starting master audio analysis thread...")
        if self.processing_thread and self.processing_thread.is_alive(): logger.warning("Waiting for existing thread..."); self.processing_thread.join(5.0);
        if self.processing_thread and self.processing_thread.is_alive(): logger.error("Existing thread stuck."); self._set_ui_processing_state(False); return
        self.processing_thread = threading.Thread(target=self._analyze_master_audio, name="AudioAnalysisThread", daemon=True); self.processing_thread.start()

    def _generate_audio_cache_path(self, audio_path: str) -> str:
        """Generates consistent cache path based on audio file hash and cache version."""
        try:
            hasher = hashlib.sha256();
            hasher.update(CACHE_VERSION.encode()) # Include version in hash
            with open(audio_path, 'rb') as f:
                while True: chunk = f.read(65536);
                if not chunk: break; hasher.update(chunk)
            file_hash = hasher.hexdigest()[:16]; base_name = os.path.splitext(os.path.basename(audio_path))[0]; cache_file = f"audio_analysis_{base_name}_{file_hash}.json"; return os.path.join(self.audio_cache_dir, cache_file)
        except Exception as e: logger.warning(f"Hash gen failed for {audio_path}: {e}"); return os.path.join(self.audio_cache_dir, f"audio_analysis_{os.path.basename(audio_path)}.json") # Fallback

    def _analyze_master_audio(self):
        """Analyzes master audio, using persistent cache if available."""
        if ENABLE_PROFILING: profiler_start_time = time.time()
        temp_audio_for_master = None; needs_cleanup = False
        try:
            if not self.analysis_config: raise RuntimeError("Analysis config missing.")
            audio_analyzer = BBZAudioUtils(); timestamp = time.strftime("%Y%m%d%H%M%S"); temp_audio_for_master = os.path.join(self.analysis_subdir, f"temp_master_audio_{timestamp}.wav"); audio_file_to_analyze = self.beat_track_path; video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg', '.webm')
            if not os.path.isfile(audio_file_to_analyze): raise FileNotFoundError(f"Master file not found: {audio_file_to_analyze}")
            if self.beat_track_path.lower().endswith(video_extensions):
                self.after(0, lambda: self.status_label.configure(text="Extracting audio from master..."))
                extracted_path = audio_analyzer.extract_audio(self.beat_track_path, temp_audio_for_master);
                if not extracted_path: raise RuntimeError(f"Audio extraction failed: {self.beat_track_path}");
                audio_file_to_analyze = extracted_path; needs_cleanup = True
            else: needs_cleanup = False
            cache_path = self._generate_audio_cache_path(audio_file_to_analyze); use_cache = self.analysis_config.save_analysis_data # Reuse flag for audio cache
            if use_cache and os.path.exists(cache_path):
                logger.info(f"Found audio cache: {os.path.basename(cache_path)}"); self.after(0, lambda: self.status_label.configure(text="Loading cached audio..."))
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
                    if 'raw_features' in loaded_data: # Convert necessary arrays back to numpy
                        raw = loaded_data['raw_features']
                        for key in ['mel_spectrogram', 'mel_times', 'rms_energy', 'rms_times', 'rms_energy_long', 'rms_deriv_short', 'rms_deriv_long', 'perc_times', 'percussive_ratio', 'waveform_np_for_ser']:
                             if key in raw and raw[key] is not None:
                                 try: raw[key] = np.asarray(raw[key], dtype=np.float32)
                                 except Exception as np_err: logger.warning(f"Failed numpy cache conversion '{key}': {np_err}"); raw[key] = None
                    if 'sr' in loaded_data and 'duration' in loaded_data and 'raw_features' in loaded_data:
                        logger.info("Loaded cached audio data."); self.master_audio_data = loaded_data
                    else: raise ValueError("Cache missing essential keys.")
                except Exception as cache_load_err:
                    logger.error(f"Failed load audio cache {os.path.basename(cache_path)}: {cache_load_err}. Re-analyzing."); self.master_audio_data = None;
                    if os.path.exists(cache_path): try: os.remove(cache_path); logger.info("Removed corrupted cache.")
                    except OSError as del_err: logger.warning(f"Failed remove cache: {del_err}")
            if self.master_audio_data is None: # Run analysis if not cached/loaded
                self.after(0, lambda: self.status_label.configure(text="Analyzing master audio...")); logger.info(f"Running full audio analysis: {os.path.basename(audio_file_to_analyze)}"); self.master_audio_data = audio_analyzer.analyze_audio(audio_file_to_analyze, self.analysis_config)
                if self.master_audio_data and use_cache: # Save to cache if successful
                    logger.info(f"Saving audio analysis to cache: {os.path.basename(cache_path)}")
                    try:
                        data_to_save = json.loads(json.dumps(self.master_audio_data, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else f"<unserializable:{type(o).__name__}>")) # Convert numpy arrays for JSON
                        with open(cache_path, 'w', encoding='utf-8') as f: json.dump(data_to_save, f, indent=1)
                    except Exception as cache_save_err: logger.error(f"Failed save audio cache: {cache_save_err}")
            if not self.master_audio_data: raise RuntimeError("Audio analysis failed.")
            logger.info("Master audio analysis successful."); self.after(0, self._start_parallel_video_analysis)
        except Exception as e: error_msg = f"Error master audio: {type(e).__name__}"; logger.error(f"Error master audio: {e}", exc_info=True); self.after(0, lambda em=error_msg, ex=str(e): [self.status_label.configure(text=f"Error: {em}", text_color="orange"), tk_write(f"Failed master audio.\n\n{ex}", parent=self, level="error"), self._set_ui_processing_state(False)])
        finally:
            if needs_cleanup and temp_audio_for_master and os.path.exists(temp_audio_for_master): try: os.remove(temp_audio_for_master); logger.debug("Cleaned temp master audio.")
            except OSError as del_err: logger.warning(f"Failed remove temp audio: {del_err}")
            if ENABLE_PROFILING: logger.debug(f"PROFILING: Master Audio took {time.time() - profiler_start_time:.3f}s")

    def _start_parallel_video_analysis(self):
        self.status_label.configure(text=f"Analyzing {len(self.video_files)} videos (using cache)...");
        if self.processing_thread is not None and self.processing_thread.is_alive(): logger.warning("Waiting for audio thread..."); self.processing_thread.join(timeout=5.0);
        if self.processing_thread and self.processing_thread.is_alive(): logger.error("Audio thread stuck."); self.after(0, lambda: [self.status_label.configure(text="Error: Thread conflict.", text_color="red"), self._set_ui_processing_state(False)]); return
        self.processing_thread = threading.Thread(target=self._run_parallel_video_analysis_pool, name="VideoAnalysisPoolMgr", daemon=True); self.processing_thread.start()

    def _run_parallel_video_analysis_pool(self):
        if not MULTIPROCESSING_AVAILABLE: logger.error("Multiprocessing unavailable."); self.after(0, lambda: [tk_write("Multiprocessing Error", "Parallel processing unavailable.", level="error"), self._set_ui_processing_state(False)]); return
        cpu_cores = os.cpu_count() or 1; max_workers = max(1, min(cpu_cores - 1, 6)); logger.info(f"Starting parallel video analysis ({max_workers} workers)."); self.analysis_futures = []; self.futures_map = {}; self.total_tasks = len(self.video_files); self.completed_tasks = 0
        if not self.analysis_config or not self.master_audio_data: logger.critical("Config/master audio missing for video pool."); self.after(0, lambda: [tk_write("Internal error: Config unavailable.", level="error"), self._set_ui_processing_state(False)]); return
        self.shutdown_executor(); self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers); submitted_count = 0; submission_errors = 0
        for vid_path in self.video_files:
            if not os.path.exists(vid_path): logger.error(f"Skipping non-existent: {vid_path}"); submission_errors += 1; self.total_tasks -= 1; continue
            try: # Pass visual_cache_dir to worker
                future = self.executor.submit(process_single_video, vid_path, self.master_audio_data, self.analysis_config, self.analysis_subdir, self.visual_cache_dir);
                self.futures_map[future] = vid_path; self.analysis_futures.append(future); submitted_count += 1
            except Exception as submit_e: logger.error(f"ERROR submitting job {os.path.basename(vid_path)}: {submit_e}"); submission_errors += 1; self.total_tasks -= 1
        if not self.analysis_futures: final_msg = "Error: Failed to start any analysis jobs." + (f" ({submission_errors} submission errors)" if submission_errors else ""); logger.error(f"Aborting: {final_msg}"); self.after(0, lambda msg=final_msg: [tk_write(msg, level="error"), self._set_ui_processing_state(False)]); self.shutdown_executor(); return
        logger.info(f"Submitted {submitted_count} analysis jobs. Waiting..."); self.after(0, lambda: self.status_label.configure(text=f"Analyzing... 0/{self.total_tasks} (0.0%)"))
        self.after(1000, self._check_analysis_status) # Start polling

    def _check_analysis_status(self):
        if not self.is_processing or not hasattr(self, 'analysis_futures') or not self.analysis_futures: logger.debug("Analysis check stopping."); self.shutdown_executor(); return
        try:
            done_futures = [f for f in self.analysis_futures if f.done()]; self.completed_tasks = len(done_futures)
            if self.total_tasks > 0: progress = (self.completed_tasks / self.total_tasks) * 100; self.after(0, lambda ct=self.completed_tasks, tt=self.total_tasks, p=progress: self.status_label.configure(text=f"Analyzing... {ct}/{tt} ({p:.1f}%)"))
            else: self.after(0, lambda: self.status_label.configure(text="Waiting... (No tasks?)"))
            if self.completed_tasks == self.total_tasks:
                logger.info("--- All Video Analyses Finished ---"); self.shutdown_executor(); logger.info("Collecting analysis results..."); self.all_potential_clips = []; success_count = 0; fail_count = 0; failed_videos = []
                for future in self.analysis_futures:
                    vid_path_log = self.futures_map.get(future, "Unknown");
                    try: video_path_result, status, potential_clips_result = future.result(timeout=60); logger.info(f"Result for {os.path.basename(vid_path_log)}: {status}")
                        if "Analysis OK" in status and isinstance(potential_clips_result, list):
                            valid_clips = [clip for clip in potential_clips_result if isinstance(clip, ClipSegment)]
                            if valid_clips: self.all_potential_clips.extend(valid_clips); success_count += 1
                            else: logger.info(f"Analysis OK {os.path.basename(vid_path_log)} but 0 usable clips.")
                        else: fail_count += 1; failed_videos.append(os.path.basename(vid_path_log)); logger.error(f"Analysis failed/invalid data for {os.path.basename(vid_path_log)}: Status '{status}'")
                    except concurrent.futures.TimeoutError: logger.error(f"Timeout result for {os.path.basename(vid_path_log)}. Worker stuck?"); fail_count += 1; failed_videos.append(os.path.basename(vid_path_log))
                    except Exception as e: logger.error(f"Error result for {os.path.basename(vid_path_log)}: {type(e).__name__}", exc_info=True); fail_count += 1; failed_videos.append(os.path.basename(vid_path_log))
                logger.info(f"Collected {len(self.all_potential_clips)} potential clips from {success_count} source(s).")
                if fail_count > 0: fail_msg = f"{fail_count} video(s) failed analysis."; display_limit = 5; fail_msg += f"\nFailed sources (partial): {', '.join(failed_videos[:display_limit])}{('...' if len(failed_videos) > display_limit else '')}"; self.after(0, lambda msg=fail_msg: tk_write(msg, parent=self, level="warning"))
                if not self.all_potential_clips: logger.error("Aborting build: No potential clips identified."); self.after(0, lambda: [self.status_label.configure(text="Error: No usable clips found.", text_color="orange"), tk_write("Analysis finished, but no usable clips found.\nCheck logs.", parent=self, level="error"), self._set_ui_processing_state(False)]); return
                self.after(0, self._schedule_post_analysis_workflow) # Schedule next step
            else: self.after(1000, self._check_analysis_status) # Poll again
        except Exception as poll_err: logger.error(f"Error checking analysis status: {poll_err}", exc_info=True); self.after(0, lambda: [self.status_label.configure(text="Error checking status.", text_color="red"), self._set_ui_processing_state(False)]); self.shutdown_executor()

    def _schedule_post_analysis_workflow(self):
         """Schedules the post-analysis workflow (SyncNet, LVRE, Build, Render) in a new thread."""
         self.status_label.configure(text="Chef is post-processing..."); logger.info("Starting post-analysis workflow thread...")
         if self.processing_thread is not None and self.processing_thread.is_alive(): logger.warning("Waiting for analysis pool manager thread..."); self.processing_thread.join(timeout=5.0);
         if self.processing_thread and self.processing_thread.is_alive(): logger.error("Analysis pool mgr stuck. Abort."); self.after(0, lambda: [self.status_label.configure(text="Error: Thread conflict.", text_color="red"), self._set_ui_processing_state(False)]); return
         self.processing_thread = threading.Thread(target=self._run_post_analysis_workflow, name="PostAnalysisWorkflowThread", daemon=True); self.processing_thread.start()

    def _run_post_analysis_workflow(self):
        """Runs steps after parallel analysis: Batch SyncNet, LVRE, Style, Build, Render."""
        if ENABLE_PROFILING: profiler_start_time_post = time.time()
        final_msg = "Process failed (Unknown)."; final_color = "red"; final_tk_level = "error"; output_video_path = None; selected_mode = "N/A"; temp_audio_render = None
        try:
            if not self.analysis_config or not self.master_audio_data or not self.render_config: raise RuntimeError("Missing config/data for post-analysis.")
            if not self.all_potential_clips: raise RuntimeError("No potential clips for post-analysis.")

            # --- 1. Batch SyncNet Scoring (PWRC) ---
            if self.analysis_config.use_latent_sync:
                logger.info("Starting Batch SyncNet scoring..."); self.after(0, lambda: self.status_label.configure(text="Scoring Lip Sync (Batch)..."))
                syncnet_model, _ = get_pytorch_model(f"syncnet_{self.analysis_config.syncnet_repo_id}_{self.analysis_config.syncnet_filename}", load_syncnet_model_from_hf_func, config=self.analysis_config)
                full_mel = self.master_audio_data.get('raw_features', {}).get('mel_spectrogram')
                mel_t = self.master_audio_data.get('raw_features', {}).get('mel_times')
                if syncnet_model and full_mel is not None and mel_t is not None:
                     try: batch_score_syncnet(self.all_potential_clips, syncnet_model, full_mel, mel_t, self.analysis_config)
                     except Exception as sync_batch_err: logger.error(f"Batch SyncNet scoring failed: {sync_batch_err}", exc_info=True); self.after(0, lambda: tk_write(f"Warning: Batch SyncNet scoring failed:\n{sync_batch_err}", parent=self, level="warning"))
                else: logger.error("SyncNet model/Mel data missing for batch scoring. Scores set to 0.")
                    # Set default scores if model/data missing
                    for clip in self.all_potential_clips: clip.latent_sync_score = 0.0

            # --- 2. LVRE Preprocessing ---
            if self.analysis_config.use_lvre_features:
                logger.info("Starting LVRE preprocessing..."); self.after(0, lambda: self.status_label.configure(text="Preprocessing Lyrics/Visuals..."))
                try: preprocess_lyrics_and_visuals(self.master_audio_data, self.all_potential_clips, self.analysis_config)
                except Exception as preproc_err: logger.error(f"LVRE preprocessing failed: {preproc_err}", exc_info=True); self.after(0, lambda: tk_write(f"Warning: LVRE preprocessing failed:\n{preproc_err}", parent=self, level="warning"))

            # --- 3. SAAPV Style Analysis & Config Adaptation ---
            self.adapted_analysis_config = self.analysis_config
            if self.analysis_config.use_saapv_adaptation:
                logger.info("Analyzing music style (SAAPV)..."); self.after(0, lambda: self.status_label.configure(text="Analyzing Style..."))
                try:
                    detected_style = analyze_music_style(self.master_audio_data, self.analysis_config); logger.info(f"Adapting parameters for style: {detected_style}...")
                    changes = {}; # Example adaptation rules
                    if "High-Energy" in detected_style or "Fast" in detected_style: changes = {'min_sequence_clip_duration': max(0.3, self.analysis_config.min_sequence_clip_duration * 0.75), 'max_sequence_clip_duration': max(1.0, self.analysis_config.max_sequence_clip_duration * 0.85), 'saapv_predictability_weight': self.analysis_config.saapv_predictability_weight * 0.8, 'mrise_weight': self.analysis_config.mrise_weight * 1.1, 'rhythm_beat_sync_weight': self.analysis_config.rhythm_beat_sync_weight * 1.1 }
                    elif "Ballad" in detected_style or "Slow" in detected_style: changes = {'min_sequence_clip_duration': min(1.8, self.analysis_config.min_sequence_clip_duration * 1.2), 'max_sequence_clip_duration': min(15.0, self.analysis_config.max_sequence_clip_duration * 1.1), 'saapv_predictability_weight': self.analysis_config.saapv_predictability_weight * 1.2, 'rhythm_beat_sync_weight': self.analysis_config.rhythm_beat_sync_weight * 0.5, 'hefm_weight': self.analysis_config.hefm_weight * 1.1 }
                    if changes: self.adapted_analysis_config = dataclass_replace(self.analysis_config, **changes); logger.info(f"Using Adapted Config for Builder. Changes: {changes}")
                    else: logger.info(f"No specific style adaptations for: {detected_style}.")
                except Exception as style_err: logger.error(f"Style analysis/adaptation failed: {style_err}", exc_info=True); self.adapted_analysis_config = self.analysis_config; self.after(0, lambda: tk_write(f"Warning: Style analysis failed:\n{style_err}", parent=self, level="warning"))
            else: logger.info("Skipping SAAPV adaptation."); self.adapted_analysis_config = self.analysis_config

            # --- 4. Instantiate & Build Sequence ---
            selected_mode = self.adapted_analysis_config.sequencing_mode; builder = None; logger.info(f"Instantiating builder: {selected_mode}"); self.after(0, lambda sm=selected_mode: self.status_label.configure(text=f"Composing sequence ({sm})...")); build_start = time.time()
            if selected_mode == "Physics Pareto MC": builder = SequenceBuilderPhysicsMC(self.all_potential_clips, self.master_audio_data, self.adapted_analysis_config); builder.effects = self.render_config.effect_settings
            else: builder = SequenceBuilderGreedy(self.all_potential_clips, self.master_audio_data, self.adapted_analysis_config)
            final_sequence = builder.build_sequence(); build_duration = time.time() - build_start; logger.info(f"Sequence building ({selected_mode}) finished in {build_duration:.2f}s.");
            if ENABLE_PROFILING: logger.debug(f"PROFILING: Seq Build took {build_duration:.3f}s")
            if not final_sequence: raise RuntimeError(f"Builder ({selected_mode}) resulted in empty sequence.")
            logger.info(f"Generated sequence: {len(final_sequence)} clips.")

            # --- 5. Prepare Render ---
            timestamp = time.strftime("%Y%m%d_%H%M%S"); mode_tag = "GreedyEns" if selected_mode == "Greedy Heuristic" else "PhysicsMCV4"; output_video_path = os.path.join(self.render_subdir, f"videous_chef_{mode_tag}_{timestamp}.mp4"); master_audio_path_for_render = self.beat_track_path; audio_util = BBZAudioUtils(); video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg', '.webm')
            if self.beat_track_path.lower().endswith(video_extensions):
                logger.info("Extracting audio from master video for render..."); self.after(0, lambda: self.status_label.configure(text="Extracting render audio..."))
                temp_audio_render = os.path.join(self.render_subdir, f"master_audio_render_{timestamp}.wav"); extracted = audio_util.extract_audio(self.beat_track_path, temp_audio_render);
                if not extracted or not os.path.exists(temp_audio_render): raise RuntimeError("Failed audio extraction for render.");
                master_audio_path_for_render = temp_audio_render; logger.info(f"Using extracted audio: {os.path.basename(master_audio_path_for_render)}")
            if not os.path.isfile(master_audio_path_for_render): raise FileNotFoundError(f"Render audio invalid: {master_audio_path_for_render}")

            # --- 6. Render Video ---
            self.after(0, lambda sm=selected_mode: self.status_label.configure(text=f"Rendering final video ({sm})..."))
            buildSequenceVideo(final_sequence, output_video_path, master_audio_path_for_render, self.render_config)
            final_msg = f"Success ({mode_tag})! Saved:\n{os.path.basename(output_video_path)}"; final_color = "light green"; final_tk_level = "info"; logger.info(f"Video composition successful: {output_video_path}")

        except Exception as e: logger.critical(f"!!! FATAL ERROR during Post-Analysis ({selected_mode}) !!!", exc_info=True); error_type_name = type(e).__name__; error_message = str(e); final_msg = f"Error: {error_type_name}"; final_color = "orange"; final_tk_level = "error"; self.after(0, lambda et=error_type_name, em=error_message: tk_write(f"Chef Error:\n\n{et}: {em}\n\nCheck logs.", parent=self, level="error"))
        finally:
            if temp_audio_render and os.path.exists(temp_audio_render): try: os.remove(temp_audio_render); logger.debug("Cleaned temp render audio.")
            except OSError as del_err: logger.warning(f"Failed remove temp audio: {del_err}")
            self.after(0, lambda msg=final_msg, color=final_color: self.status_label.configure(text=msg, text_color=color))
            if final_tk_level == "info" and output_video_path and os.path.exists(output_video_path): self.after(100, lambda path=output_video_path, mode=selected_mode : tk_write(f"Success!\nMode: {mode}\nOutput:\n{os.path.basename(path)}", parent=self, level="info"))
            elif final_tk_level == "error" and output_video_path and os.path.exists(output_video_path): self.after(100, lambda path=output_video_path : tk_write(f"Rendering failed. Output file exists:\n{os.path.basename(path)}\n(May be incomplete)", parent=self, level="warning"))
            self.after(0, self._set_ui_processing_state, False);
            if ENABLE_PROFILING: logger.debug(f"PROFILING: Post-Analysis Workflow took {time.time() - profiler_start_time_post:.3f}s")
            logger.info("Post-Analysis Workflow thread finished.")

    # --- Window Closing & Executor Shutdown ---
    def on_closing(self):
        logger.info("Shutdown requested...")
        if self.is_processing:
            if messagebox.askyesno("Confirm Exit", "Processing ongoing.\nStop and exit?", parent=self):
                 logger.warning("Forcing shutdown."); self.is_processing = False; self.shutdown_executor(); self.destroy()
            else: logger.info("Shutdown cancelled."); return
        else: logger.info("Closing application."); self.shutdown_executor(); self.destroy()
        logger.info("App closing sequence initiated.")

    def shutdown_executor(self):
        if hasattr(self, 'executor') and self.executor:
             logger.info("Shutting down process pool executor..."); cancelled_count = 0
             if hasattr(self, 'analysis_futures'):
                 for f in self.analysis_futures:
                     if not f.done():
                          if f.cancel(): cancelled_count += 1
             logger.debug(f"Attempted cancel {cancelled_count} pending future(s).")
             try:
                 if sys.version_info >= (3, 9): self.executor.shutdown(wait=True, cancel_futures=True)
                 else: self.executor.shutdown(wait=True)
                 logger.info("Executor shutdown complete.")
             except Exception as e: logger.error(f"Executor shutdown error: {e}", exc_info=False)
             finally: self.executor = None; self.analysis_futures = []; self.futures_map = {}
        else: logger.debug("No active executor.")

# ========================================================================
#                      REQUIREMENTS.TXT Block
# ========================================================================
"""
# requirements.txt for Videous Chef - Ensemble Edition v5.4

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
torchaudio>=0.12.0             # Primary audio backend (check version/build for MPS/CUDA)
librosa>=0.9.0,<0.11.0         # For Mel spec helper & audio analysis fallbacks

# Ensemble - AI / ML Features
torch>=1.12.0                  # Base PyTorch (check version/build for MPS/CUDA)
torchvision>=0.13.0            # Often needed with PyTorch
mediapipe>=0.10.0,<0.11.0      # For Face Mesh, Pose
transformers[torch]>=4.25.0    # For CLIP, Whisper, SER models
sentence-transformers>=2.2.0   # For text embeddings
huggingface_hub>=0.10.0        # For downloading models (like SyncNet)
scikit-image>=0.19.0           # For SyncNet mouth crop helper

# MiDaS Dependency (Needed for Depth features)
timm>=0.6.0

# Optional Ensemble Features (Install Manually If Desired)
# audioflux>=0.1.8       # (Faster audio analysis - check GitHub/PyPI for install)
# demucs>=4.0.0          # (Source separation for MRISE - check GitHub for install)
# PySceneDetect>=0.6     # (Scene detection - pip install scenedetect[opencv])

# Optional - Face Restoration (Install Manually If Desired)
# gfpgan>=1.3.8
# basicsr>=1.4.2         # Dependency for GFPGAN
# facexlib>=0.3.0        # Dependency for GFPGAN

# --- Notes ---
# 1. FFmpeg: Ensure FFmpeg executable is installed and in system PATH.
# 2. GPU: Highly recommended (NVIDIA CUDA or Apple Silicon MPS). Verify PyTorch compatibility.
# 3. Environment: Use a Python virtual environment (e.g., venv, conda).
# 4. SyncNet Model: Verify embedded SyncNet definition matches weights used (ByteDance/LatentSync-1.5).
# 5. GFPGAN Model: If using GFPGAN, manually download .pth model (e.g., GFPGANv1.4.pth) and place in specified path or update path in UI.
"""

# ========================================================================
#                       APPLICATION ENTRY POINT
# ========================================================================
# Ensure global flags are checked/set based on imports
# (Flags defined near top imports section)
TKINTER_AVAILABLE = 'tkinter' in sys.modules
CTK_AVAILABLE = 'customtkinter' in sys.modules
MOVIEPY_AVAILABLE = 'moviepy' in sys.modules
TORCH_AVAILABLE = 'torch' in sys.modules

if __name__ == "__main__":
    # --- Multiprocessing Setup ---
    if MULTIPROCESSING_AVAILABLE:
        multiprocessing.freeze_support()
        try:
            default_method = multiprocessing.get_start_method(allow_none=True)
            if sys.platform != 'win32' and default_method != 'spawn':
                if 'spawn' in multiprocessing.get_all_start_methods():
                    multiprocessing.set_start_method('spawn', force=True); print("INFO: Set MP start method to 'spawn'.")
                else: print(f"WARNING: 'spawn' MP method unavailable. Using default: {default_method}.")
            elif sys.platform == 'win32': print(f"INFO: Using default Windows MP start method: '{default_method}'.") # Spawn is default
            else: print(f"INFO: Using MP start method '{default_method}'.")
        except Exception as mp_setup_e: print(f"WARNING: MP start method setup error: {mp_setup_e}. Using default.")
    else: print("WARNING: Multiprocessing disabled.")

    # --- Logging Setup ---
    print(f"--- Videous Chef - Ensemble Edition v{CACHE_VERSION} ---")
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s [%(threadName)s]')
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); console_handler.setLevel(logging.INFO)
    file_handler = None
    try: log_dir = f"logs_{CACHE_VERSION}"; os.makedirs(log_dir, exist_ok=True); log_file = os.path.join(log_dir, f"videous_chef_ensemble_{time.strftime('%Y%m%d_%H%M%S')}.log"); file_handler = logging.FileHandler(log_file, encoding='utf-8'); file_handler.setFormatter(log_formatter); file_handler.setLevel(logging.DEBUG); log_setup_message = f"Logging to console (INFO+) and file (DEBUG+): {log_file}"
    except Exception as log_setup_e: log_setup_message = f"Failed file logging: {log_setup_e}. Console only."; file_handler = None
    root_logger = logging.getLogger(); root_logger.setLevel(logging.DEBUG); root_logger.handlers.clear(); root_logger.addHandler(console_handler);
    if file_handler: root_logger.addHandler(file_handler);
    logger.info(log_setup_message) # Log setup status

    try:
        # --- STARTUP CHECKS ---
        logger.info("Checking essential dependencies...")
        # Assume essential flags are defined/checked correctly near imports
        essential_available = all([TKINTER_AVAILABLE, CTK_AVAILABLE, MOVIEPY_AVAILABLE, TORCH_AVAILABLE, TORCHAUDIO_AVAILABLE, MEDIAPIPE_AVAILABLE]) # Add mediapipe check
        if not essential_available: logger.critical("One or more critical dependencies missing! Check errors above."); sys.exit(1)
        # Check FFmpeg
        try:
            ffmpeg_cmd = "ffmpeg"; result = subprocess.run([ffmpeg_cmd, "-version"], shell=False, capture_output=True, text=True, check=False, timeout=5, encoding='utf-8')
            if result.returncode != 0 or "ffmpeg version" not in result.stdout.lower(): raise FileNotFoundError(f"FFmpeg check failed.")
            logger.info(f"FFmpeg check successful.")
        except Exception as ffmpeg_e: err_msg = f"CRITICAL: FFmpeg not found or failed check: {ffmpeg_e}\nInstall FFmpeg and ensure it's in PATH.\nExiting."; logger.critical(err_msg); root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showerror("Dependency Error", err_msg); root_err.destroy(); sys.exit(1)
        # Check PyTorch Backend
        try: logger.info(f"PyTorch version: {torch.__version__}"); get_device()
        except Exception as torch_check_e: logger.warning(f"PyTorch backend check issue: {torch_check_e}")
        # --- Log Optional Dependency Status ---
        logger.info("Optional dependencies status:"); [logger.info(f"  {lib}: {'Available' if flag else 'NOT FOUND'}") for lib, flag in [("transformers", TRANSFORMERS_AVAILABLE), ("sentence-transformers", SENTENCE_TRANSFORMERS_AVAILABLE), ("audioflux", AUDIOFLUX_AVAILABLE), ("demucs", DEMUCS_AVAILABLE), ("huggingface_hub", HUGGINGFACE_HUB_AVAILABLE), ("scikit-image", SKIMAGE_AVAILABLE), ("librosa", LIBROSA_AVAILABLE), ("timm", TIMM_AVAILABLE), ("PySceneDetect", PYSCENEDETECT_AVAILABLE), ("gfpgan", GFPGAN_AVAILABLE)]]

        # --- Run App ---
        logger.info(f"Initializing Application UI (Ensemble v{CACHE_VERSION})...")
        app = VideousApp()
        logger.info("Starting Tkinter main loop...")
        app.mainloop()

    except SystemExit as se: logger.warning(f"Application exited (Code: {se.code}).")
    except Exception as e: logger.critical(f"!!! UNHANDLED STARTUP ERROR !!!", exc_info=True);
        try: root_err = tkinter.Tk(); root_err.withdraw(); messagebox.showerror("Startup Error", f"Application failed:\n\n{type(e).__name__}: {e}\n\nCheck logs."); root_err.destroy()
        except Exception as msg_err: print(f"\n\n!!! CRITICAL STARTUP ERROR: {type(e).__name__}: {e} !!!\nCheck logs.\nGUI Message Error: {msg_err}")

    logger.info(f"--- Videous Chef v{CACHE_VERSION} session ended ---")