```python
from __future__ import annotations
# -*- coding: utf-8 -*-
# ========================================================================
#            Videous Chef - Ensemble Edition v5.3 (Optimized & Fixed)
# ========================================================================
#                       IMPORTS (Essential Core Only)
# ========================================================================

#source /Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/venv_videous_chef/bin/activate
#conda activate videous 
# Removed: from tkinter import filedialog, Listbox, Scrollbar, END, MULTIPLE, Frame, messagebox, BooleanVar, IntVar
# Removed: import customtkinter
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
# from math import exp, log # Already imported
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
# torchaudio already imported

# CTK_AVAILABLE will be false as customtkinter import is removed
CTK_AVAILABLE = False
    
try:
    import moviepy
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# tkinterdnd2 removed
DND_FILES = None
TkinterDnD = None
    
try:
    import multiprocessing
    MULTIPROCESSING_AVAILABLE = True # Keep this for actual multiprocessing
except ImportError:
    print("WARNING: multiprocessing module not found, parallel processing disabled.")
    multiprocessing = None 
    MULTIPROCESSING_AVAILABLE = False
    
# TKINTER_AVAILABLE will be false as tkinter specific UI imports are removed.
# The main `import tkinter` might be needed by matplotlib implicitly or for very basic things.
# For strict GUI removal, if not used by non-GUI parts, it would also go.
# For now, let's assume matplotlib handles its backend without explicit Tkinter windowing.
TKINTER_AVAILABLE = False 
try:
    import tkinter # Keep for now, matplotlib might need it, or __main__ error reporting fallback
    TKINTER_AVAILABLE = True # This flag is more about GUI capability
except ImportError:
    pass


try: from tqdm import tqdm
except ImportError: tqdm = None; print("INFO: tqdm not found, progress bars disabled.")
TQDM_AVAILABLE = tqdm is not None

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
GFPGAN_AVAILABLE = False # Moved definition higher

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
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
    print("INFO: `gfpgan` loaded successfully for optional rendering enhancement.")
except ImportError:
    print("INFO: `gfpgan` not found. Face enhancement in rendering disabled.")


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

DEFAULT_NORM_MAX_RMS = 0.5; DEFAULT_NORM_MAX_ONSET = 5.0; DEFAULT_NORM_MAX_CHROMA_VAR = 0.1

# --- Caches & Logging ---
_pytorch_model_cache = {}; _pytorch_processor_cache = {} # General cache for PyTorch models/processors
logger = logging.getLogger(__name__) # Get logger for the current module

# ========================================================================
#              <<< SYNC NET MODEL DEFINITION (EMBEDDED) >>>
# ========================================================================
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
    def __init__(self):
        super(SyncNet, self).__init__()
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

            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)), 
            nn.ReLU(inplace=True)
        )

        self.video_stream = nn.Sequential(
            Conv3dRelu(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 0, 0)), 
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            Conv3dRelu(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)), 
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), 

            Conv3dRelu(256, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)), 
            Conv3dRelu(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            Conv3dRelu(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(512, 512, kernel_size=(1, 6, 6), stride=(1, 1, 1), padding=(0, 0, 0)), 
            nn.ReLU(inplace=True)
        )

    def forward(self, audio_sequences, video_sequences):
        audio_embedding = self.audio_stream(audio_sequences) 
        video_embedding = self.video_stream(video_sequences) 

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1) 
        video_embedding = video_embedding.view(video_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        video_embedding = F.normalize(video_embedding, p=2, dim=1)

        return audio_embedding, video_embedding

# ========================================================================
#                       DATA CLASSES (Ensemble Ready)
# ========================================================================
# from dataclasses import dataclass # Already imported

@dataclass
class AnalysisConfig:
    min_sequence_clip_duration: float = 0.75
    max_sequence_clip_duration: float = 5.0
    min_potential_clip_duration_sec: float = 0.4
    resolution_height: int = 256
    resolution_width: int = 256
    save_analysis_data: bool = True
    cache_visual_features: bool = True
    use_scene_detection: bool = True
    norm_max_rms: float = 0.1
    norm_max_onset: float = 0.1
    norm_max_pose_kinetic: float = 50.0
    norm_max_visual_flow: float = 50.0
    norm_max_depth_variance: float = 0.15
    norm_max_face_size: float = 1.0
    norm_max_jerk: float = 100000.0
    min_face_confidence: float = 0.5
    min_pose_confidence: float = 0.5
    model_complexity: int = 1
    mouth_open_threshold: float = 0.05
    target_sr_audio: int = 44100
    use_dl_beat_tracker: bool = True
    hop_length_energy: int = 512
    frame_length_energy: int = 1024
    trend_window_sec: float = 3.0
    hop_length_mel: int = 160
    sequencing_mode: str = "Greedy Heuristic"
    use_latent_sync: bool = True
    use_lvre_features: bool = False
    use_saapv_adaptation: bool = True
    use_mrise_sync: bool = True
    use_demucs_for_mrise: bool = False
    syncnet_repo_id: str = "ByteDance/LatentSync-1.5"
    syncnet_filename: str = "syncnet.pth"
    syncnet_batch_size: int = 16
    whisper_model_name: str = "openai/whisper-tiny"
    ser_model_name: str = "facebook/wav2vec2-large-robust-ft-emotion-msp-podcast"
    text_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vision_embed_model_name: str = "openai/clip-vit-base-patch32"
    lvre_batch_size_text: int = 128
    lvre_batch_size_vision: int = 64
    lvre_batch_size_ser: int = 16
    scene_change_penalty: float = 0.1
    render_fps: int = 30
    base_heuristic_weight: float = 0.1
    bh_audio_weight: float = 0.3
    bh_kinetic_weight: float = 0.25
    bh_sharpness_weight: float = 0.1
    bh_camera_motion_weight: float = 0.05
    bh_face_size_weight: float = 0.1
    bh_percussive_weight: float = 0.05
    bh_depth_weight: float = 0.0
    pwrc_weight: float = 0.3
    pwrc_lipsync_weight: float = 0.6
    pwrc_pose_energy_weight: float = 0.4
    hefm_weight: float = 0.2
    hefm_trend_match_weight: float = 1.0
    lvre_weight: float = 0.15
    lvre_semantic_weight: float = 0.7
    lvre_emphasis_weight: float = 0.3
    saapv_weight: float = 0.1
    saapv_predictability_weight: float = 0.5
    saapv_history_length: int = 10
    saapv_variety_penalty_source: float = 0.1
    saapv_variety_penalty_shot: float = 0.1
    saapv_variety_penalty_intensity: float = 0.10
    mrise_weight: float = 0.15
    mrise_sync_weight: float = 1.0
    mrise_sync_tolerance_factor: float = 1.5
    rhythm_beat_sync_weight: float = 0.1
    rhythm_beat_boost_radius_sec: float = 0.1
    candidate_pool_size: int = 15
    score_threshold: float = 0.3
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
class EffectParams:
    type: str = "cut"
    tau: float = 0.0
    psi: float = 0.0
    epsilon: float = 0.0

@dataclass
class RenderConfig:
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
    gfpgan_model_path: Optional[str] = "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/Videous CHEF/experiments/pretrained_models/GFPGANv1.3.pth"

# ========================================================================
#                       HELPER FUNCTIONS (Loaders, SyncNet Helpers, etc.)
# ========================================================================
# tk_write function removed as it's GUI-specific. Callers should use logger directly.

def sigmoid(x, k=1):
    try:
        x_clamped = np.clip(x * k, -700, 700)
        return 1 / (1 + np.exp(-x_clamped))
    except OverflowError:
        logger.warning(f"Sigmoid overflow detected for input x={x}, k={k}.")
        return 0.0 if x < 0 else 1.0

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    vec1 = np.asarray(vec1, dtype=float); vec2 = np.asarray(vec2, dtype=float)
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    dot_product = np.dot(vec1, vec2); similarity = dot_product / (norm1 * norm2)
    return np.clip(similarity, -1.0, 1.0)

def calculate_histogram_entropy(frame):
    if frame is None or frame.size == 0: return 0.0
    try:
        if len(frame.shape) == 3 and frame.shape[2] == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2: gray = frame
        else: logger.warning(f"Invalid frame shape for entropy: {frame.shape}"); return 0.0
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_sum = hist.sum()
        if hist_sum <= 0: return 0.0
        hist_norm = hist.ravel() / hist_sum
        hist_norm_nonzero = hist_norm[hist_norm > 0]
        if hist_norm_nonzero.size == 0: return 0.0
        entropy = scipy.stats.entropy(hist_norm_nonzero)
        return entropy if np.isfinite(entropy) else 0.0
    except cv2.error as cv_err: logger.warning(f"OpenCV error in histogram calc: {cv_err}"); return 0.0
    except Exception as e: logger.warning(f"Hist entropy calculation failed: {e}"); return 0.0

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.debug("Using MPS (Apple Silicon GPU) device.")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU device.")
    return device

def get_pytorch_model(cache_key: str, load_func: callable, *args, **kwargs) -> Optional[Tuple[torch.nn.Module, torch.device]]:
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
        if cache_key in _pytorch_model_cache:
            del _pytorch_model_cache[cache_key]
        return None, None

def get_pytorch_processor(cache_key: str, load_func: callable, *args, **kwargs) -> Optional[Any]:
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
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found.")
    device_id = device.index if device.type == 'cuda' else (-1 if device.type == 'cpu' else 0) # MPS uses device=0 or device='mps'
    if device.type == 'mps': device_id = 'mps' # Explicitly for transformers pipeline
    trust_code = kwargs.pop('trust_remote_code', False)
    return pipeline(task, model=model_name, device=device_id, trust_remote_code=trust_code, **kwargs)


def load_huggingface_model_func(model_name: str, model_class: type, **kwargs):
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found.")
    trust_code = kwargs.pop('trust_remote_code', False)
    return model_class.from_pretrained(model_name, trust_remote_code=trust_code, **kwargs)

def load_huggingface_processor_func(model_name: str, processor_class: type, **kwargs):
    if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers library not found.")
    trust_code = kwargs.pop('trust_remote_code', False)
    return processor_class.from_pretrained(model_name, trust_remote_code=trust_code, **kwargs)

def load_sentence_transformer_func(model_name: str):
    if not SENTENCE_TRANSFORMERS_AVAILABLE: raise ImportError("Sentence-Transformers library not found.")
    return SentenceTransformer(model_name)

def load_syncnet_model_from_hf_func(config: AnalysisConfig) -> Optional[SyncNet]:
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

from functools import lru_cache
@lru_cache(maxsize=1)
def get_midas_model() -> Optional[Tuple[torch.nn.Module, Any, torch.device]]:
    global TIMM_AVAILABLE
    if not TIMM_AVAILABLE:
        logger.error("MiDaS requires 'timm'. Install with: pip install timm")
        return None, None, None
    logger.info("Loading MiDaS model (intel-isl/MiDaS MiDaS_small)...")
    try:
        model, device = get_pytorch_model(
            "midas_small",
            lambda: torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        )
        if model is None or device is None:
            raise RuntimeError("MiDaS model loading failed via get_pytorch_model.")
        transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms_hub.small_transform
        logger.info("MiDaS model and transform loaded.")
        return model, transform, device
    except Exception as e:
        logger.error(f"Failed to load MiDaS: {e}", exc_info=True)
        if "midas_small" in _pytorch_model_cache:
            del _pytorch_model_cache["midas_small"]
        return None, None, None

MOUTH_LM_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 61, 76, 77, 78, 80, 81, 82, 84, 85, 87, 88, 90, 91, 95, 146, 178, 180, 181, 267, 269, 270, 291, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 320, 321, 324, 375, 402, 404, 405, 409, 415]
def extract_mouth_crop(frame_bgr: np.ndarray, face_landmarks: Any, target_size=(112, 112)) -> Optional[np.ndarray]:
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
        resized_mouth = skimage.transform.resize(mouth_crop, target_size, anti_aliasing=True, mode='edge', preserve_range=True)
        resized_mouth_uint8 = np.clip(resized_mouth, 0, 255).astype(np.uint8)
        return resized_mouth_uint8
    except ImportError: logger.error("extract_mouth_crop called but scikit-image unavailable."); return None
    except Exception as e: logger.warning(f"Error extracting mouth crop: {e}", exc_info=False); return None

@lru_cache(maxsize=4)
def compute_mel_spectrogram(waveform_tensor: torch.Tensor, sr: int, hop_length: int = 160, n_mels: int = 80) -> Optional[np.ndarray]:
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
    
def get_recent_history(history: deque, count: int) -> List:
    return list(history)

def calculate_flow_velocity(prev_gray, current_gray):
    if prev_gray is None or current_gray is None or prev_gray.shape != current_gray.shape:
        return 0.0, None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        if flow is None:
            return 0.0, None
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.nanmean(magnitude)
        if not np.isfinite(avg_magnitude): avg_magnitude = 0.0
        return float(avg_magnitude * 10.0), flow
    except cv2.error as cv_err:
        logger.warning(f"OpenCV optical flow calculation error: {cv_err}")
        return 0.0, None
    except Exception as e:
        logger.warning(f"Unexpected error calculating flow velocity: {e}")
        return 0.0, None

def calculate_flow_acceleration(prev_flow, current_flow, dt):
    if prev_flow is None or current_flow is None or prev_flow.shape != current_flow.shape or dt <= 1e-6:
        return 0.0
    try:
        flow_diff = current_flow - prev_flow
        accel_magnitude_per_pixel, _ = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
        avg_accel_magnitude = np.nanmean(accel_magnitude_per_pixel)
        if not np.isfinite(avg_accel_magnitude): avg_accel_magnitude = 0.0
        accel = avg_accel_magnitude / dt
        return float(accel * 10.0)
    except Exception as e:
        logger.warning(f"Error calculating flow acceleration: {e}")
        return 0.0

def calculate_kinetic_energy_proxy(landmarks_prev, landmarks_curr, dt):
    if landmarks_prev is None or landmarks_curr is None or dt <= 1e-6: return 0.0
    if not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0
    lm_prev = landmarks_prev.landmark; lm_curr = landmarks_curr.landmark
    if len(lm_prev) != len(lm_curr): return 0.0
    total_sq_velocity = 0.0; num_valid = 0
    try:
        for i in range(len(lm_prev)):
            vis_prev = getattr(lm_prev[i], 'visibility', 1.0)
            vis_curr = getattr(lm_curr[i], 'visibility', 1.0)
            if vis_prev > 0.2 and vis_curr > 0.2:
                dx = lm_curr[i].x - lm_prev[i].x
                dy = lm_curr[i].y - lm_prev[i].y
                dz = lm_curr[i].z - lm_prev[i].z
                total_sq_velocity += (dx**2 + dy**2 + dz**2) / (dt**2)
                num_valid += 1
    except IndexError:
        logger.warning("Index error during kinetic energy calculation (landmark mismatch?).")
        return 0.0
    if num_valid == 0: return 0.0
    avg_sq_velocity = total_sq_velocity / num_valid
    return float(avg_sq_velocity * 1000.0)

def calculate_movement_jerk_proxy(landmarks_prev_prev, landmarks_prev, landmarks_curr, dt):
    if landmarks_prev_prev is None or landmarks_prev is None or landmarks_curr is None or dt <= 1e-6: return 0.0
    if not hasattr(landmarks_prev_prev, 'landmark') or not hasattr(landmarks_prev, 'landmark') or not hasattr(landmarks_curr, 'landmark'): return 0.0
    lm_pp = landmarks_prev_prev.landmark; lm_p = landmarks_prev.landmark; lm_c = landmarks_curr.landmark
    if len(lm_pp) != len(lm_p) or len(lm_p) != len(lm_c): return 0.0
    total_sq_accel_change = 0.0; num_valid = 0; dt_sq = dt * dt
    try:
        for i in range(len(lm_pp)):
            vis_pp = getattr(lm_pp[i], 'visibility', 1.0)
            vis_p = getattr(lm_p[i], 'visibility', 1.0)
            vis_c = getattr(lm_c[i], 'visibility', 1.0)
            if vis_pp > 0.2 and vis_p > 0.2 and vis_c > 0.2:
                ax = (lm_c[i].x - 2*lm_p[i].x + lm_pp[i].x) / dt_sq
                ay = (lm_c[i].y - 2*lm_p[i].y + lm_pp[i].y) / dt_sq
                az = (lm_c[i].z - 2*lm_p[i].z + lm_pp[i].z) / dt_sq
                accel_magnitude_sq = ax**2 + ay**2 + az**2
                total_sq_accel_change += accel_magnitude_sq
                num_valid += 1
    except IndexError:
        logger.warning("Index error during jerk calculation (landmark mismatch?).")
        return 0.0
    if num_valid == 0: return 0.0
    avg_sq_accel_proxy = total_sq_accel_change / num_valid
    return float(avg_sq_accel_proxy * 100000.0)

class BBZPoseUtils:
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
        profiler_start_time = time.time()
        logger.info(f"Analyzing audio (Ensemble v5.4/5.5): {os.path.basename(audio_path)}")
        target_sr = analysis_config.target_sr_audio
        try:
            if not os.path.exists(audio_path): raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.debug(f"Loading audio with TorchAudio (Target SR: {target_sr})...")
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            if sr != target_sr: logger.debug(f"Resampling {sr}Hz->{target_sr}Hz"); resampler = T.Resample(sr, target_sr, dtype=waveform.dtype); waveform = resampler(waveform); sr = target_sr
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            duration = waveform.shape[1] / sr
            if duration <= 0: raise ValueError("Audio zero duration.")
            waveform_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
            logger.debug(f"Audio loaded: Shape={waveform.shape}, SR={sr}, Dur={duration:.2f}s")
            stems = {}
            if analysis_config.use_mrise_sync and analysis_config.use_demucs_for_mrise:
                if DEMUCS_AVAILABLE:
                    logger.info("Demucs sep (Placeholder)...")
                    stems = {'drums': waveform_np * 0.5, 'vocals': waveform_np * 0.5}
                else:
                    logger.warning("Demucs enabled but unavailable.")
            logger.debug("Analyzing rhythm...")
            beat_times, downbeat_times = self._detect_beats_downbeats(waveform_np, sr, analysis_config)
            tempo = 120.0
            if LIBROSA_AVAILABLE:
                try:
                    tempo_val = librosa.beat.tempo(y=waveform_np, sr=sr, aggregate=np.median)[0]
                    if np.isfinite(tempo_val) and tempo_val > 0:
                        tempo = float(tempo_val)
                except Exception as te:
                    logger.warning(f"Librosa tempo detection failed: {te}")
            else:
                logger.warning("Librosa unavailable for tempo detection.")
            logger.debug(f"Detected Tempo: {tempo:.1f} BPM")
            micro_beat_times = []
            if analysis_config.use_mrise_sync:
                logger.debug("Analyzing micro-rhythms (MRISE)...")
                target_stem = 'drums'
                audio_for_mrise = stems.get(target_stem, waveform_np)
                stem_name_log = target_stem if target_stem in stems else None
                micro_beat_times = self._detect_micro_beats(audio_for_mrise, sr, analysis_config, stem_name_log)
            logger.debug("Computing energy & trends (HEFM)...")
            hop_e = analysis_config.hop_length_energy; frame_e = analysis_config.frame_length_energy
            rms_torch = torchaudio.functional.compute_rms(waveform, frame_length=frame_e, hop_length=hop_e).squeeze(0)
            rms_energy = rms_torch.cpu().numpy(); rms_times = np.linspace(0, duration, len(rms_energy), endpoint=False) + (hop_e / sr / 2.0)
            smooth_win_s = analysis_config.trend_window_sec; smooth_win_f = max(11, int(sr*smooth_win_s / hop_e) | 1)
            rms_long = scipy.signal.savgol_filter(rms_energy, smooth_win_f, 3) if len(rms_energy) > smooth_win_f else rms_energy.copy()
            time_step = hop_e / sr if sr > 0 else 1.0
            rms_deriv_short = np.gradient(rms_energy, time_step) if len(rms_energy) > 1 else np.zeros_like(rms_energy)
            rms_deriv_long = np.gradient(rms_long, time_step) if len(rms_long) > 1 else np.zeros_like(rms_long)
            mel_spec = None; mel_t = None
            if analysis_config.use_latent_sync: logger.debug("Computing Mel Spectrogram...");
            mel_spec = compute_mel_spectrogram(waveform, sr, hop_length=analysis_config.hop_length_mel)
            if mel_spec is not None: mel_frames = mel_spec.shape[1]; mel_t = np.linspace(0, duration, mel_frames, endpoint=False) + (analysis_config.hop_length_mel / sr / 2.0)
            else: logger.error("Mel Spectrogram failed.")
            logger.debug("Segmenting audio..."); bound_times = self._segment_audio(waveform, sr, analysis_config)
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
            raw = { 'rms_energy': rms_energy, 'rms_times': rms_times, 'rms_energy_long': rms_long, 'rms_deriv_short': rms_deriv_short, 'rms_deriv_long': rms_deriv_long, 'original_audio_path': audio_path, 'mel_spectrogram': mel_spec, 'mel_times': mel_t, 'waveform_np_for_ser': waveform_np if analysis_config.use_lvre_features else None, 'perc_times': rms_times, 'percussive_ratio': np.zeros_like(rms_times)}
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
class BBZFaceUtils:
    def __init__(self, static_mode=False, max_faces=1, min_detect_conf=0.5, min_track_conf=0.5):
        self.face_mesh = None; self._mp_face_mesh = mp.solutions.face_mesh
        try:
            self.face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=static_mode, max_num_faces=max_faces,
                refine_landmarks=True, 
                min_detection_confidence=min_detect_conf,
                min_tracking_confidence=min_track_conf)
            logger.info("FaceMesh initialized (Refine Landmarks: True).")
        except Exception as e: logger.error(f"Failed init FaceMesh: {e}"); self.face_mesh = None

    def process_frame(self, image_bgr):
        if self.face_mesh is None:
            return None
        try:
            image_bgr.flags.writeable = False
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            image_bgr.flags.writeable = True
            return results
        except Exception as e:
            logger.warning(f"FaceMesh process error on frame: {e}")
            if hasattr(image_bgr, 'flags'): image_bgr.flags.writeable = True
            return None

    def get_heuristic_face_features(self, results, h, w, mouth_open_threshold=0.05):
        is_open, size_ratio, center_x = False, 0.0, 0.5
        if results and results.multi_face_landmarks:
            try:
                face_landmarks = results.multi_face_landmarks[0]; lm = face_landmarks.landmark
                if 13 < len(lm) and 14 < len(lm): 
                    upper_y = lm[13].y * h; lower_y = lm[14].y * h; mouth_h = abs(lower_y - upper_y)
                    if 10 < len(lm) and 152 < len(lm): 
                        forehead_y = lm[10].y * h; chin_y = lm[152].y * h; face_h = abs(chin_y - forehead_y)
                        if face_h > 1e-6: is_open = (mouth_h / face_h) > mouth_open_threshold
                all_x = [p.x * w for p in lm if np.isfinite(p.x)]; all_y = [p.y * h for p in lm if np.isfinite(p.y)]
                if all_x and all_y: min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y); face_w = max(1, max_x - min_x); face_h = max(1, max_y - min_y); face_diag = np.sqrt(face_w**2 + face_h**2); img_diag = np.sqrt(w**2 + h**2); # Used numpy sqrt
                if img_diag > 1e-6: size_ratio = np.clip(face_diag / img_diag, 0.0, 1.0)
                if all_x: center_x = np.clip(np.mean(all_x) / w, 0.0, 1.0)
            except IndexError: logger.warning("Index error accessing FaceMesh landmarks.")
            except Exception as e: logger.warning(f"Error extracting heuristic face features: {e}")
        return is_open, float(size_ratio), float(center_x)
    
    def close(self):
        if self.face_mesh:
            try:
                self.face_mesh.close()
                logger.info("FaceMesh resources released.")
            except Exception as e:
                logger.error(f"Error closing FaceMesh: {e}")
        self.face_mesh = None

# ========================================================================
#         DYNAMIC SEGMENT IDENTIFIER (Unchanged - Now Optional/Legacy)
# ========================================================================
class DynamicSegmentIdentifier:
    def __init__(self, analysis_config: AnalysisConfig, fps: float):
        self.fps = fps if fps > 0 else 30.0
        self.score_threshold = getattr(analysis_config, 'score_threshold', 0.3)
        self.min_segment_len_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES, int(analysis_config.min_potential_clip_duration_sec * self.fps))
        logger.debug(f"Legacy Segment Identifier Init: FPS={self.fps:.2f}, Threshold={self.score_threshold:.2f}, MinLenFrames={self.min_segment_len_frames}")

    def find_potential_segments(self, frame_features_list):
        potential_segments = []
        start_idx = -1; n = len(frame_features_list)
        if n == 0: return []
        for i, features in enumerate(frame_features_list):
            score = features.get('boosted_score', 0.0) if isinstance(features, dict) else 0.0
            is_candidate = score >= self.score_threshold
            if is_candidate and start_idx == -1: start_idx = i
            elif not is_candidate and start_idx != -1:
                segment_len = i - start_idx
                if segment_len >= self.min_segment_len_frames:
                    potential_segments.append({'start_frame': start_idx, 'end_frame': i})
                start_idx = -1
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
class BBZImageUtils:
    def resizeTARGET(self, image, TARGET_HEIGHT=256, TARGET_WIDTH=256):
        if image is None or image.size == 0:
            logger.warning("Attempted to resize an empty image.")
            return None
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            logger.warning(f"Attempted to resize image with zero dimension: {h}x{w}")
            return None
        if h != TARGET_HEIGHT or w != TARGET_WIDTH:
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
        return image

# ========================================================================
#           PREPROCESSING FUNCTION (LVRE - NEW for v5.4/5.5)
# ========================================================================
def preprocess_lyrics_and_visuals(master_audio_data: Dict, all_potential_clips: List['ClipSegment'], analysis_config: AnalysisConfig):
    if not analysis_config.use_lvre_features:
        logger.info("LVRE features disabled by config. Skipping preprocessing.")
        return
    if not TRANSFORMERS_AVAILABLE:
        logger.error("LVRE requires `transformers` library. Aborting LVRE preprocessing.")
        return
    logger.info("--- Starting LVRE Preprocessing (Batched, Lazy Load) ---")
    if ENABLE_PROFILING:
        profiler_start_time = time.time()
    device = get_device()
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
            if whisper_pipeline:
                logger.debug(f"Running Whisper pipeline on {os.path.basename(audio_path)}...")
                asr_result = whisper_pipeline(audio_path)
                if isinstance(asr_result, dict) and 'chunks' in asr_result:
                    timed_lyrics = asr_result.get('chunks', [])
                    valid_count = 0
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
                    master_audio_data['timed_lyrics'] = []
            else:
                raise RuntimeError("Whisper pipeline failed to load.")
        except Exception as asr_err:
            logger.error(f"ASR processing failed: {asr_err}", exc_info=True)
            master_audio_data['timed_lyrics'] = []
        finally:
             if not asr_success:
                 master_audio_data['timed_lyrics'] = []
    else:
        logger.info(f"Skipping ASR (already present: {len(timed_lyrics)} words).")

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
                    batch_lyrics_slice = timed_lyrics[i:i+ser_batch_size]
                    batch_audio_snippets = []; valid_indices_in_batch = []
                    for k, word_info in enumerate(batch_lyrics_slice):
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
                                    word_info['emotion_score'] = 0.0
                            else:
                                word_info['emotion_score'] = 0.0
                        else:
                            word_info['emotion_score'] = 0.0
                    if batch_audio_snippets:
                        try:
                            batch_results = ser_pipeline(batch_audio_snippets, sampling_rate=sr_audio)
                            if isinstance(batch_results, list) and len(batch_results) == len(batch_audio_snippets):
                                for res_list, orig_idx in zip(batch_results, valid_indices_in_batch):
                                    top_result = res_list[0] if isinstance(res_list, list) and res_list else (res_list if isinstance(res_list, dict) else None)
                                    if isinstance(top_result, dict):
                                        score = top_result.get('score', 0.0)
                                        batch_lyrics_slice[orig_idx]['emotion_score'] = float(score)
                                    else:
                                        batch_lyrics_slice[orig_idx]['emotion_score'] = 0.0
                            else:
                                logger.warning(f"SER batch output format mismatch.")
                                for k_idx in valid_indices_in_batch:
                                     batch_lyrics_slice[k_idx]['emotion_score'] = 0.0
                        except Exception as ser_batch_err:
                            logger.error(f"SER batch {i//ser_batch_size} failed: {ser_batch_err}")
                            for k_idx in valid_indices_in_batch:
                                batch_lyrics_slice[k_idx]['emotion_score'] = 0.0
                    for k_item in batch_lyrics_slice:
                         k_item.setdefault('emotion_score', 0.0)
                    if pbar_ser: pbar_ser.update(len(batch_lyrics_slice))
                if pbar_ser: pbar_ser.close()
                ser_success = True; logger.info("SER complete.")
            else:
                raise RuntimeError("SER pipeline failed load.")
        except Exception as ser_err:
            logger.error(f"SER failed: {ser_err}", exc_info=True)
        finally:
            if not ser_success:
                for w_item in timed_lyrics: w_item.setdefault('emotion_score', 0.0)
    else:
        logger.info("Skipping SER (lyrics missing or scores present).")

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
                    valid_i = [i_idx for i_idx, t_val in enumerate(texts) if t_val]
                    valid_t = [texts[i_idx] for i_idx in valid_i]
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
                    for i_idx in range(len(timed_lyrics)):
                        timed_lyrics[i_idx].setdefault('embedding', None)
                    text_embed_success = True; logger.info("Lyric embeddings generated.")
                else:
                    raise RuntimeError("Sentence Transformer failed load.")
            except Exception as text_err:
                logger.error(f"Text embedding failed: {text_err}", exc_info=True)
            finally:
                if not text_embed_success:
                    for w_item in timed_lyrics: w_item.setdefault('embedding', None)
    else:
        logger.info("Skipping Text Embeddings (lyrics missing or embeds present).")

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
                    for clip_item in clips_need_embed:
                        path = str(clip_item.source_video_path)
                        if not path or not os.path.exists(path):
                            clip_item.visual_embedding = None
                            if pbar_kf: pbar_kf.update(1)
                            continue
                        reader = readers.get(path)
                        if reader is None and path not in readers:
                            try:
                                reader = VideoFileClip(path, audio=False)
                                readers[path] = reader
                            except Exception as re:
                                readers[path] = None; logger.error(f"Reader fail {os.path.basename(path)}: {re}"); reader = None
                        elif path in readers and reader is None: pass 
                        if reader:
                            try:
                                kf_time = np.clip(clip_item.start_time + clip_item.duration / 2.0, 0, reader.duration - 1e-6 if reader.duration else 0)
                                kf_rgb = reader.get_frame(kf_time)
                                if kf_rgb is not None and kf_rgb.size > 0:
                                    kframes.append((clip_map[id(clip_item)], kf_rgb))
                                else:
                                    clip_item.visual_embedding = None
                            except Exception as kfe:
                                clip_item.visual_embedding = None; logger.warning(f"Keyframe fail clip {clip_item.start_frame}: {kfe}")
                        else:
                            clip_item.visual_embedding = None
                        if pbar_kf: pbar_kf.update(1)
                    if pbar_kf: pbar_kf.close()
                    for r_item in readers.values():
                        if r_item and hasattr(r_item, 'close'):
                            try: r_item.close()
                            except: pass
                    readers.clear()
                    if kframes:
                        logger.debug(f"Embedding {len(kframes)} keyframes...")
                        vis_batch = analysis_config.lvre_batch_size_vision
                        pbar_emb = tqdm(total=len(kframes), desc="Vis Embeds", disable=not TQDM_AVAILABLE)
                        with torch.no_grad():
                            for i_idx in range(0, len(kframes), vis_batch):
                                batch_dat = kframes[i_idx:i_idx+vis_batch]
                                batch_orig_idx = [item[0] for item in batch_dat]
                                batch_rgb = [item[1] for item in batch_dat]
                                try:
                                    inputs = clip_proc(images=batch_rgb, return_tensors="pt", padding=True).to(clip_dev)
                                    img_feat = clip_model.get_image_features(**inputs)
                                    img_feat = F.normalize(img_feat, p=2, dim=-1)
                                    embeds_np = img_feat.cpu().numpy()
                                    for j_idx, orig_idx in enumerate(batch_orig_idx):
                                        if 0 <= orig_idx < len(all_potential_clips):
                                             all_potential_clips[orig_idx].visual_embedding = embeds_np[j_idx].tolist()
                                except Exception as emb_err:
                                    logger.error(f"Vis embed batch {i_idx//vis_batch} failed: {emb_err}")
                                    for orig_idx in batch_orig_idx:
                                        if 0 <= orig_idx < len(all_potential_clips):
                                             all_potential_clips[orig_idx].visual_embedding = None
                                finally:
                                     if pbar_emb: pbar_emb.update(len(batch_dat))
                        if pbar_emb: pbar_emb.close()
                        vis_embed_success = True; logger.info("Visual embedding finished.")
                    else:
                        logger.info("No keyframes for visual embedding.")
                        for clip_item in clips_need_embed: clip_item.visual_embedding = None
                else:
                    logger.error("CLIP load failed.")
                    for clip_item in clips_need_embed: clip_item.visual_embedding = None
            except Exception as vis_outer:
                logger.error(f"Outer visual embed err: {vis_outer}", exc_info=True)
                for clip_item in clips_need_embed: clip_item.visual_embedding = None
            finally:
                if not vis_embed_success:
                    for clip_item in clips_need_embed: clip_item.visual_embedding = None
    else:
        logger.info("Skipping visual embeds (already present).")
    if ENABLE_PROFILING:
        logger.debug(f"PROFILING: LVRE Preprocessing took {time.time() - profiler_start_time:.3f}s")

# ========================================================================
#                       STYLE ANALYSIS FUNCTION (SAAPV)
# ========================================================================
def analyze_music_style(master_audio_data: Dict, analysis_config: AnalysisConfig) -> str:
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
def batch_score_syncnet(clips_to_score: List[ClipSegment], syncnet_model: SyncNet, full_mel_spectrogram: np.ndarray, mel_times: np.ndarray, config: AnalysisConfig):
    if not clips_to_score or syncnet_model is None or full_mel_spectrogram is None or mel_times is None:
        logger.warning("Skipping batch SyncNet scoring: Missing clips, model, or Mel data.")
        for clip_item in clips_to_score:
            clip_item.latent_sync_score = None 
        return 
    if not SKIMAGE_AVAILABLE:
        logger.warning("Skipping batch SyncNet scoring: scikit-image not available.")
        for clip_item in clips_to_score:
            clip_item.latent_sync_score = None 
        return 
    start_time = time.time()
    device = next(syncnet_model.parameters()).device
    syncnet_model.eval() 
    all_video_batch_items = [] 
    all_audio_batch_items = [] 
    batch_item_map: List[Tuple[int, int]] = [] 
    logger.info(f"Preparing {len(clips_to_score)} clips for batched SyncNet scoring...")
    pbar_prep = tqdm(total=len(clips_to_score), desc="Prep SyncNet Batches", leave=False, disable=not TQDM_AVAILABLE)
    mel_hop_sec = config.hop_length_mel / config.target_sr_audio if config.target_sr_audio > 0 else 0.01
    mel_window_size_frames = int(round(0.2 / mel_hop_sec)) if mel_hop_sec > 0 else 20
    n_mels = full_mel_spectrogram.shape[0]
    mel_total_frames = full_mel_spectrogram.shape[1]
    for clip_idx, clip_item in enumerate(clips_to_score):
        clip_item.latent_sync_score = 0.0 
        mouth_crops = []; original_frame_indices = []
        for i_idx, f_item in enumerate(clip_item.segment_frame_features):
            if isinstance(f_item, dict):
                 crop = f_item.get('mouth_crop')
                 if crop is not None and isinstance(crop, np.ndarray) and crop.shape == (112, 112, 3):
                     mouth_crops.append(crop)
                     original_frame_indices.append(clip_item.start_frame + i_idx)
        num_original_crops = len(mouth_crops)
        if num_original_crops > LATENTSYNC_MAX_FRAMES:
            subsample_indices = np.linspace(0, num_original_crops - 1, LATENTSYNC_MAX_FRAMES, dtype=int)
            mouth_crops = [mouth_crops[i_idx] for i_idx in subsample_indices]
            original_frame_indices = [original_frame_indices[i_idx] for i_idx in subsample_indices]
        if len(mouth_crops) < 5: 
            if pbar_prep: pbar_prep.update(1)
            continue 
        num_windows = len(mouth_crops) - 4
        for win_idx in range(num_windows):
            video_chunk_bgr = mouth_crops[win_idx:win_idx+5]; video_chunk_processed = []
            for frame_bgr in video_chunk_bgr:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB); frame_float = frame_rgb.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1); video_chunk_processed.append(frame_tensor)
            video_batch_item = torch.stack(video_chunk_processed, dim=1) 
            center_video_frame_idx_in_clip = win_idx + 2; center_original_frame_idx = original_frame_indices[center_video_frame_idx_in_clip]
            center_time = (center_original_frame_idx + 0.5) / clip_item.fps
            center_mel_idx = np.argmin(np.abs(mel_times - center_time))
            start_mel = max(0, center_mel_idx - mel_window_size_frames // 2); end_mel = min(mel_total_frames, start_mel + mel_window_size_frames); start_mel = max(0, end_mel - mel_window_size_frames)
            audio_chunk_mel = full_mel_spectrogram[:, start_mel:end_mel]
            current_len = audio_chunk_mel.shape[1]
            if current_len < mel_window_size_frames:
                padding_needed = mel_window_size_frames - current_len; padding = np.full((n_mels, padding_needed), np.min(full_mel_spectrogram), dtype=np.float32); audio_chunk_mel = np.concatenate((audio_chunk_mel, padding), axis=1)
            elif current_len > mel_window_size_frames:
                 audio_chunk_mel = audio_chunk_mel[:, :mel_window_size_frames]
            audio_batch_item = torch.from_numpy(audio_chunk_mel).unsqueeze(0) 
            all_video_batch_items.append(video_batch_item); all_audio_batch_items.append(audio_batch_item); batch_item_map.append((clip_idx, win_idx))
        if pbar_prep: pbar_prep.update(1)
    if pbar_prep: pbar_prep.close()
    if not all_video_batch_items:
        logger.warning("No valid windows found for SyncNet batch processing.")
        return 
    num_batch_items = len(all_video_batch_items); batch_size = config.syncnet_batch_size; all_confidences = []
    logger.info(f"Running SyncNet inference on {num_batch_items} windows in batches of {batch_size}...")
    pbar_infer = tqdm(total=num_batch_items, desc="SyncNet Inference", leave=False, disable=not TQDM_AVAILABLE)
    try:
        with torch.no_grad():
            for i_idx in range(0, num_batch_items, batch_size):
                video_batch = torch.stack(all_video_batch_items[i_idx : i_idx + batch_size]).to(device); audio_batch = torch.stack(all_audio_batch_items[i_idx : i_idx + batch_size]).to(device)
                audio_embed, video_embed = syncnet_model(audio_batch, video_batch); confidences = F.cosine_similarity(audio_embed, video_embed, dim=-1)
                all_confidences.extend(confidences.cpu().numpy().tolist());
                if pbar_infer: pbar_infer.update(video_batch.size(0))
    except Exception as infer_err:
         logger.error(f"SyncNet batch inference failed: {infer_err}", exc_info=True);
         if pbar_infer: pbar_infer.close(); return
    finally:
         if pbar_infer: pbar_infer.close()
    logger.info("Assigning SyncNet scores back to clips...")
    clip_max_scores = defaultdict(lambda: 0.0) 
    if len(all_confidences) != len(batch_item_map):
        logger.error(f"Mismatch between SyncNet confidences ({len(all_confidences)}) and batch map ({len(batch_item_map)})."); return
    for i_idx, conf in enumerate(all_confidences):
        clip_list_idx, _ = batch_item_map[i_idx]; score = float(np.clip((conf + 1.0) / 2.0, 0.0, 1.0)); clip_max_scores[clip_list_idx] = max(clip_max_scores[clip_list_idx], score)
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
        self.start_frame = start_frame; self.end_frame = end_frame 
        self.num_frames = max(0, end_frame - start_frame)
        self.fps = fps if fps > 0 else 30.0
        self.duration = self.num_frames / self.fps if self.fps > 0 else 0.0
        self.start_time = start_frame / self.fps if self.fps > 0 else 0.0
        self.end_time = end_frame / self.fps if self.fps > 0 else 0.0
        self.analysis_config = analysis_config 
        if 0 <= start_frame < end_frame <= len(all_frame_features):
             self.segment_frame_features = all_frame_features[start_frame:end_frame]
        else:
             logger.warning(f"Invalid frame indices [{start_frame}-{end_frame}] for ClipSegment (Total frames: {len(all_frame_features)}) from '{os.path.basename(str(source_video_path))}'. Segment features will be empty.")
             self.segment_frame_features = []
             self.num_frames = 0; self.duration = 0.0 
        self._initialize_features()
        if self.segment_frame_features:
            self._aggregate_visual_features()
            self._assign_audio_segment_features(master_audio_data) 
        else:
             if self.duration > 0: logger.warning(f"Cannot aggregate features for clip {start_frame}-{end_frame} due to empty segment_frame_features list.")

    def _initialize_features(self):
        self.avg_raw_score: float = 0.0; self.avg_boosted_score: float = 0.0; self.peak_boosted_score: float = 0.0
        self.avg_motion_heuristic: float = 0.0; self.avg_jerk_heuristic: float = 0.0; self.avg_camera_motion: float = 0.0
        self.face_presence_ratio: float = 0.0; self.avg_face_size: float = 0.0
        self.intensity_category: str = "Low"; self.dominant_contributor: str = "none"; self.contains_beat: bool = False
        self.musical_section_indices: Set[int] = set()
        self.avg_lip_activity: float = 0.0 
        self.avg_visual_flow: float = 0.0 
        self.avg_visual_accel: float = 0.0 
        self.avg_depth_variance: float = 0.0 
        self.avg_visual_entropy: float = 0.0 
        self.avg_pose_kinetic: float = 0.0 
        self.avg_visual_flow_trend: float = 0.0 
        self.avg_visual_pose_trend: float = 0.0 
        self.visual_mood_vector: List[float] = [0.0, 0.0] 
        self.v_k: float = 0.0; self.a_j: float = 0.0; self.d_r: float = 0.0; self.phi: float = 0.0
        self.mood_vector: List[float] = [0.0, 0.0] 
        self.latent_sync_score: Optional[float] = None 
        self.visual_embedding: Optional[List[float]] = None 
        self.audio_segment_data: Dict = {} 
        self.sequence_start_time: float = 0.0
        self.sequence_end_time: float = 0.0
        self.chosen_duration: float = 0.0
        self.chosen_effect: Optional[EffectParams] = None
        self.subclip_start_time_in_source: float = 0.0
        self.subclip_end_time_in_source: float = 0.0

    def _aggregate_visual_features(self):
        count = len(self.segment_frame_features)
        if count == 0: return 
        def safe_mean(key, default=0.0, sub_dict=None):
            values = []
            for f_item in self.segment_frame_features:
                if not isinstance(f_item, dict): continue
                container = f_item.get(sub_dict) if sub_dict else f_item
                if isinstance(container, dict):
                    val = container.get(key)
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        values.append(val)
            return float(np.mean(values)) if values else default
        self.avg_raw_score = safe_mean('raw_score')
        self.avg_boosted_score = safe_mean('boosted_score')
        peak_scores = [f_item.get('boosted_score', 0.0) for f_item in self.segment_frame_features if isinstance(f_item, dict) and np.isfinite(f_item.get('boosted_score', -np.inf))]
        self.peak_boosted_score = float(np.max(peak_scores)) if peak_scores else 0.0
        self.avg_motion_heuristic = safe_mean('kinetic_energy_proxy', sub_dict='pose_features')
        self.avg_jerk_heuristic = safe_mean('movement_jerk_proxy', sub_dict='pose_features')
        self.avg_camera_motion = safe_mean('flow_velocity') 
        face_sizes = [f_item.get('pose_features', {}).get('face_size_ratio', 0.0) for f_item in self.segment_frame_features if isinstance(f_item, dict) and f_item.get('pose_features', {}).get('face_size_ratio', 0.0) > 1e-3]
        face_present_frames = len(face_sizes)
        self.face_presence_ratio = float(face_present_frames / count) if count > 0 else 0.0
        self.avg_face_size = float(np.mean(face_sizes)) if face_sizes else 0.0
        self.contains_beat = any(f_item.get('is_beat_frame', False) for f_item in self.segment_frame_features if isinstance(f_item, dict))
        self.musical_section_indices = {f_item.get('musical_section_index', -1) for f_item in self.segment_frame_features if isinstance(f_item, dict) and f_item.get('musical_section_index', -1) != -1}
        self.avg_visual_flow = safe_mean('flow_velocity')
        self.avg_pose_kinetic = safe_mean('kinetic_energy_proxy', sub_dict='pose_features')
        self.avg_visual_accel = safe_mean('flow_acceleration')
        self.avg_depth_variance = safe_mean('depth_variance')
        self.avg_visual_entropy = safe_mean('histogram_entropy')
        self.avg_visual_flow_trend = safe_mean('visual_flow_trend')
        self.avg_visual_pose_trend = safe_mean('visual_pose_trend')
        self.v_k = np.clip(self.avg_visual_flow / (self.analysis_config.norm_max_visual_flow + 1e-6), 0.0, 1.0)
        self.a_j = np.clip(self.avg_visual_accel / (100.0 + 1e-6), 0.0, 1.0) 
        self.d_r = np.clip(self.avg_depth_variance / (self.analysis_config.norm_max_depth_variance + 1e-6), 0.0, 1.0)
        self.phi = self.avg_visual_entropy 
        norm_flow_v4 = self.v_k
        norm_depth_v4 = self.d_r
        self.mood_vector = [float(norm_flow_v4), float(1.0 - norm_depth_v4)]
        self.visual_mood_vector = self.mood_vector.copy()
        dominant_contribs = [f_item.get('dominant_contributor', 'none') for f_item in self.segment_frame_features if isinstance(f_item, dict)]
        if dominant_contribs:
            non_none = [c_item for c_item in dominant_contribs if c_item != 'none']
            self.dominant_contributor = max(set(non_none), key=non_none.count) if non_none else 'none'
        intensities = [f_item.get('intensity_category', 'Low') for f_item in self.segment_frame_features if isinstance(f_item, dict)]
        intensity_order = ['Low', 'Medium', 'High']
        if intensities:
            indices = [intensity_order.index(i_val) for i_val in intensities if i_val in intensity_order]
            highest_intensity = intensity_order[max(indices)] if indices else 'Low'
            self.intensity_category = highest_intensity

    def _assign_audio_segment_features(self, master_audio_data: Dict):
        if not master_audio_data: return 
        mid_time = self.start_time + self.duration / 2.0
        audio_segments = master_audio_data.get('segment_features', [])
        matched_segment = None
        if not audio_segments: 
            logger.warning("No audio segments available in master_audio_data to assign features.")
            self.audio_segment_data = {}
            return
        for seg in audio_segments:
            if seg['start'] <= mid_time < seg['end']:
                matched_segment = seg
                break
        if matched_segment is None and mid_time >= audio_segments[-1]['end'] - 1e-6:
            matched_segment = audio_segments[-1]
        if matched_segment:
            self.audio_segment_data = matched_segment.copy() 
        else:
            logger.warning(f"Could not find matching audio segment for clip at midpoint {mid_time:.2f}s")
            self.audio_segment_data = {} 

    def _calculate_and_set_latent_sync(self, syncnet_model: SyncNet, full_mel_spectrogram: np.ndarray, mel_times: np.ndarray):
        if not self.analysis_config.use_latent_sync: self.latent_sync_score = None; return
        if syncnet_model is None: logger.warning("SyncNet model not loaded, cannot calculate score."); self.latent_sync_score = None; return
        if not SKIMAGE_AVAILABLE: logger.warning("scikit-image not available, cannot process mouth crops for SyncNet."); self.latent_sync_score = None; return
        if full_mel_spectrogram is None or mel_times is None: logger.warning(f"Cannot calculate latent sync for clip {self.start_frame}: Missing Mel data."); self.latent_sync_score = None; return
        if self.num_frames < 5: logger.debug(f"Clip {self.start_frame} too short ({self.num_frames} frames) for SyncNet."); self.latent_sync_score = 0.0; return
        start_time_sync = time.time() # Renamed variable
        syncnet_model_device = next(syncnet_model.parameters()).device
        mouth_crops = []
        original_frame_indices = [] 
        for i_idx, f_item in enumerate(self.segment_frame_features):
            if isinstance(f_item, dict):
                 crop = f_item.get('mouth_crop')
                 if crop is not None and isinstance(crop, np.ndarray) and crop.shape == (112, 112, 3): 
                     mouth_crops.append(crop)
                     original_frame_indices.append(self.start_frame + i_idx) 
        if len(mouth_crops) < 5: 
            logger.debug(f"Insufficient valid mouth crops ({len(mouth_crops)} found) for SyncNet scoring in clip {self.start_frame}.")
            self.latent_sync_score = 0.0; return
        num_original_crops = len(mouth_crops)
        if num_original_crops > LATENTSYNC_MAX_FRAMES:
            subsample_indices = np.linspace(0, num_original_crops - 1, LATENTSYNC_MAX_FRAMES, dtype=int)
            mouth_crops = [mouth_crops[i_idx] for i_idx in subsample_indices]
            original_frame_indices = [original_frame_indices[i_idx] for i_idx in subsample_indices]
            logger.debug(f"SyncNet: Limited frames from {num_original_crops} to {len(mouth_crops)} for performance.")
        crop_timestamps = [(idx + 0.5) / self.fps for idx in original_frame_indices]
        mel_spec_indices = []
        if len(mel_times) < 2: 
            logger.warning(f"Cannot determine Mel hop time for SyncNet in clip {self.start_frame}.")
            self.latent_sync_score = 0.0; return
        mel_hop_sec = mel_times[1] - mel_times[0] 
        mel_window_size_frames = int(round(0.2 / mel_hop_sec)) if mel_hop_sec > 0 else 20
        for ts in crop_timestamps:
            center_mel_idx = np.argmin(np.abs(mel_times - ts))
            mel_spec_indices.append(center_mel_idx)
        video_batches = []; audio_batches = []
        num_windows = len(mouth_crops) - 4 
        if num_windows <= 0:
             logger.debug(f"Not enough crops ({len(mouth_crops)}) to form a 5-frame window for SyncNet in clip {self.start_frame}.")
             self.latent_sync_score = 0.0; return
        for i_idx in range(num_windows):
            video_chunk_bgr = mouth_crops[i_idx:i_idx+5] 
            video_chunk_processed = []
            for frame in video_chunk_bgr:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_float = frame_rgb.astype(np.float32) / 255.0 
                frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1) 
                video_chunk_processed.append(frame_tensor)
            video_batch_item = torch.stack(video_chunk_processed, dim=1)
            video_batches.append(video_batch_item)
            center_video_frame_index_in_window = i_idx + 2 
            center_mel_idx = mel_spec_indices[center_video_frame_index_in_window] 
            start_mel = max(0, center_mel_idx - mel_window_size_frames // 2)
            end_mel = min(full_mel_spectrogram.shape[1], start_mel + mel_window_size_frames)
            start_mel = max(0, end_mel - mel_window_size_frames)
            audio_chunk_mel = full_mel_spectrogram[:, start_mel:end_mel]
            current_len = audio_chunk_mel.shape[1]
            if current_len < mel_window_size_frames:
                padding_needed = mel_window_size_frames - current_len
                padding = np.full((full_mel_spectrogram.shape[0], padding_needed), np.min(full_mel_spectrogram), dtype=np.float32)
                audio_chunk_mel = np.concatenate((audio_chunk_mel, padding), axis=1)
            elif current_len > mel_window_size_frames:
                 audio_chunk_mel = audio_chunk_mel[:, :mel_window_size_frames] 
            audio_batch_item = torch.from_numpy(audio_chunk_mel).unsqueeze(0)
            audio_batches.append(audio_batch_item)
        if not video_batches or not audio_batches:
             logger.debug(f"No valid SyncNet batches created for clip {self.start_frame}.")
             self.latent_sync_score = 0.0; return
        all_scores = []
        batch_size = self.analysis_config.syncnet_batch_size
        try:
            with torch.no_grad(): 
                for i_idx in range(0, len(video_batches), batch_size):
                    video_batch = torch.stack(video_batches[i_idx:i_idx+batch_size]).to(syncnet_model_device) 
                    audio_batch = torch.stack(audio_batches[i_idx:i_idx+batch_size]).to(syncnet_model_device) 
                    audio_embed, video_embed = syncnet_model(audio_batch, video_batch)
                    cosine_sim = F.cosine_similarity(audio_embed, video_embed, dim=-1) 
                    all_scores.extend(cosine_sim.cpu().numpy().tolist())
        except Exception as sync_err:
            logger.error(f"SyncNet inference failed for clip {self.start_frame}: {sync_err}", exc_info=True)
            self.latent_sync_score = 0.0; return 
        if all_scores:
            avg_score = np.mean(all_scores)
            self.latent_sync_score = float(np.clip((avg_score + 1.0) / 2.0, 0.0, 1.0))
            logger.debug(f"SyncNet score for clip {self.start_frame}: {self.latent_sync_score:.3f} (from {len(all_scores)} windows) ({time.time()-start_time_sync:.3f}s)")
        else:
            logger.warning(f"SyncNet inference yielded no scores for clip {self.start_frame}.")
            self.latent_sync_score = 0.0

    def clip_audio_fit(self, audio_segment_data, analysis_config):
        if not audio_segment_data: return 0.0
        cfg = analysis_config 
        b_i = audio_segment_data.get('b_i', 0.0) 
        e_i = audio_segment_data.get('e_i', 0.0) 
        m_i_aud = np.asarray(audio_segment_data.get('m_i', [0.0, 0.0])) 
        m_i_vid = np.asarray(self.mood_vector) 
        diff_v = abs(self.v_k - b_i) 
        diff_a = abs(self.a_j - e_i) 
        sigma_m_sq = cfg.mood_similarity_variance**2 * 2
        mood_dist_sq = np.sum((m_i_vid - m_i_aud)**2)
        mood_sim = exp(-mood_dist_sq / (sigma_m_sq + 1e-9))
        diff_m = 1.0 - mood_sim 
        fit_arg = (cfg.fit_weight_velocity * diff_v +
                   cfg.fit_weight_acceleration * diff_a +
                   cfg.fit_weight_mood * diff_m)
        probability = 1.0 - sigmoid(fit_arg, cfg.fit_sigmoid_steepness)
        return float(probability)

    def get_feature_vector(self, analysis_config):
        return [float(self.v_k), float(self.a_j), float(self.d_r)]

    def get_shot_type(self):
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
        self.BBZImageUtils = BBZImageUtils()
        self.BBZPoseUtils = BBZPoseUtils()
        self.midas_model = None
        self.midas_transform = None
        self.midas_device = None

    def _ensure_midas_loaded(self, analysis_config: AnalysisConfig):
        needs_midas = (analysis_config.sequencing_mode == 'Physics Pareto MC') or \
                      (analysis_config.base_heuristic_weight > 0 and analysis_config.bh_depth_weight > 0)
        if needs_midas and self.midas_model is None:
            logger.info("MiDaS model required by configuration, attempting load...")
            model, transform, device = get_midas_model()
            if model and transform and device:
                 self.midas_model = model
                 self.midas_transform = transform
                 self.midas_device = device
                 logger.info("MiDaS model and transform loaded successfully.")
            else:
                 logger.warning("MiDaS failed to load. Depth features will be disabled.")
        elif needs_midas:
            logger.debug("MiDaS model already loaded.")
        elif not needs_midas:
             logger.debug("MiDaS model not required by current configuration.")

    def _determine_dominant_contributor(self, norm_features_weighted):
        if not norm_features_weighted: return "unknown"
        max_val = -float('inf'); dominant_key = "none"
        for key, value in norm_features_weighted.items():
            if value > max_val: max_val = value; dominant_key = key
        key_map = {'audio_energy': 'Audio', 'kinetic_proxy': 'Motion', 'jerk_proxy': 'Jerk',
                   'camera_motion': 'CamMove', 'face_size': 'FaceSize',
                   'percussive': 'Percuss', 'depth_variance': 'DepthVar'}
        return key_map.get(dominant_key, dominant_key) if max_val > 1e-4 else "none"

    def _categorize_intensity(self, score, thresholds=(0.3, 0.7)):
        if score < thresholds[0]: return "Low"
        if score < thresholds[1]: return "Medium"
        return "High"

    def calculate_base_heuristic_score(self, frame_features: Dict, analysis_config: AnalysisConfig) -> Tuple[float, str, str]:
        weights = {
            'audio_energy': analysis_config.bh_audio_weight,
            'kinetic_proxy': analysis_config.bh_kinetic_weight,
            'jerk_proxy': analysis_config.bh_sharpness_weight,
            'camera_motion': analysis_config.bh_camera_motion_weight,
            'face_size': analysis_config.bh_face_size_weight,
            'percussive': analysis_config.bh_percussive_weight,
            'depth_variance': analysis_config.bh_depth_weight
        }
        norm_params = {
            'rms': analysis_config.norm_max_rms + 1e-6,
            'kinetic': analysis_config.norm_max_pose_kinetic + 1e-6,
            'jerk': analysis_config.norm_max_jerk + 1e-6,
            'cam_motion': analysis_config.norm_max_visual_flow + 1e-6, 
            'face_size': analysis_config.norm_max_face_size + 1e-6,
            'percussive_ratio': 1.0 + 1e-6, 
            'depth_variance': analysis_config.norm_max_depth_variance + 1e-6
        }
        f = frame_features
        pose_f = f.get('pose_features', {})
        audio_energy = f.get('audio_energy', 0.0)
        kinetic_proxy = pose_f.get('kinetic_energy_proxy', 0.0)
        jerk_proxy = pose_f.get('movement_jerk_proxy', 0.0)
        camera_motion = f.get('flow_velocity', 0.0)
        face_size_ratio = pose_f.get('face_size_ratio', 0.0)
        percussive_ratio = f.get('percussive_ratio', 0.0) 
        depth_variance = f.get('depth_variance', 0.0)
        norm_audio = np.clip(audio_energy / norm_params['rms'], 0.0, 1.0)
        norm_kinetic = np.clip(kinetic_proxy / norm_params['kinetic'], 0.0, 1.0)
        norm_jerk = np.clip(jerk_proxy / norm_params['jerk'], 0.0, 1.0)
        norm_cam_motion = np.clip(camera_motion / norm_params['cam_motion'], 0.0, 1.0)
        norm_face_size = np.clip(face_size_ratio / norm_params['face_size'], 0.0, 1.0)
        norm_percussive = np.clip(percussive_ratio / norm_params['percussive_ratio'], 0.0, 1.0)
        norm_depth_var = np.clip(depth_variance / norm_params['depth_variance'], 0.0, 1.0)
        contrib = {
            'audio_energy': norm_audio * weights['audio_energy'],
            'kinetic_proxy': norm_kinetic * weights['kinetic_proxy'],
            'jerk_proxy': norm_jerk * weights['jerk_proxy'],
            'camera_motion': norm_cam_motion * weights['camera_motion'],
            'face_size': norm_face_size * weights['face_size'],
            'percussive': norm_percussive * weights['percussive'],
            'depth_variance': norm_depth_var * weights['depth_variance']
        }
        weighted_contrib = {k: v for k, v in contrib.items() if abs(weights.get(k, 0)) > 1e-6}
        score = sum(weighted_contrib.values())
        final_score = np.clip(score, 0.0, 1.0)
        dominant = self._determine_dominant_contributor(weighted_contrib)
        intensity = self._categorize_intensity(final_score)
        return float(final_score), dominant, intensity

    def apply_beat_boost(self, frame_features_list: List[Dict], audio_data: Dict, video_fps: float, analysis_config: AnalysisConfig):
        num_frames = len(frame_features_list)
        if num_frames == 0 or not audio_data or video_fps <= 0: return
        beat_boost_value = analysis_config.base_heuristic_weight * 0.5
        boost_radius_sec = analysis_config.rhythm_beat_boost_radius_sec
        boost_radius_frames = max(0, int(boost_radius_sec * video_fps))
        beat_times = audio_data.get('beat_times', [])
        if not beat_times or beat_boost_value <= 0: return 
        boost_frame_indices = set()
        for t in beat_times:
            beat_frame_center = int(round(t * video_fps))
            for r in range(-boost_radius_frames, boost_radius_frames + 1):
                idx = beat_frame_center + r
                if 0 <= idx < num_frames: boost_frame_indices.add(idx)
        for i, features in enumerate(frame_features_list):
            if not isinstance(features, dict): continue
            is_beat = i in boost_frame_indices
            features['is_beat_frame'] = is_beat 
            boost = beat_boost_value if is_beat else 0.0
            raw_score = features.get('raw_score', 0.0) 
            features['boosted_score'] = min(raw_score + boost, 1.0)

    def get_feature_at_time(self, times_array, values_array, target_time):
        if times_array is None or values_array is None or len(times_array) == 0 or len(times_array) != len(values_array):
            logger.debug(f"Interpolation skipped: Invalid input arrays for time {target_time:.3f}.")
            return 0.0
        if len(times_array) == 1: 
             return float(values_array[0])
        try:
            if not np.all(np.diff(times_array) >= 0):
                sort_indices = np.argsort(times_array)
                times_array = times_array[sort_indices]
                values_array = values_array[sort_indices]
            interpolated_value = np.interp(target_time, times_array, values_array, left=values_array[0], right=values_array[-1])
            return float(interpolated_value) if np.isfinite(interpolated_value) else 0.0
        except Exception as e:
            logger.error(f"Interpolation error at time={target_time:.3f}: {e}")
            return 0.0 

    def analyzeVideo(self, videoPath: str, analysis_config: AnalysisConfig,
                     master_audio_data: Dict) -> Tuple[Optional[List[Dict]], Optional[List[ClipSegment]]]:
        logger.info(f"--- Analyzing Video (Ensemble v5.3): {os.path.basename(videoPath)} ---")
        if ENABLE_PROFILING: profiler_start_time = time.time()
        TARGET_HEIGHT = analysis_config.resolution_height
        TARGET_WIDTH = analysis_config.resolution_width
        all_frame_features: List[Dict] = []; potential_clips: List[ClipSegment] = []
        clip: Optional[VideoFileClip] = None
        pose_detector = None; face_detector_util = None; pose_context = None 
        prev_gray: Optional[np.ndarray] = None; prev_flow: Optional[np.ndarray] = None
        pose_results_buffer = deque([None, None, None], maxlen=3) 
        fps: float = 30.0 
        self._ensure_midas_loaded(analysis_config) 
        syncnet_model = None; syncnet_model_device = None
        full_mel_spectrogram = None; mel_times = None
        if analysis_config.use_latent_sync:
            full_mel_spectrogram = master_audio_data.get('raw_features', {}).get('mel_spectrogram')
            mel_times = master_audio_data.get('raw_features', {}).get('mel_times')
            if full_mel_spectrogram is None or mel_times is None:
                 logger.error(f"SyncNet enabled but Mel data missing for {os.path.basename(videoPath)}. Disabling SyncNet scoring for this video.")
                 analysis_config = dataclass_replace(analysis_config, use_latent_sync=False)
            else:
                 model_cache_key = f"syncnet_{analysis_config.syncnet_repo_id}_{analysis_config.syncnet_filename}"
                 syncnet_model, syncnet_model_device = get_pytorch_model(
                     model_cache_key,
                     load_syncnet_model_from_hf_func, config=analysis_config 
                 )
                 if syncnet_model is None:
                      logger.error(f"Failed to load SyncNet model for {os.path.basename(videoPath)}. Disabling scoring.")
                      analysis_config = dataclass_replace(analysis_config, use_latent_sync=False)
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
            face_detector_util = BBZFaceUtils(static_mode=False, max_faces=1, min_detect_conf=analysis_config.min_face_confidence)
            if face_detector_util.face_mesh is None: logger.warning("FaceMesh failed to initialize. Face features disabled.")
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
                    pose_detector = pose_context.__enter__() 
                except Exception as pose_init_err:
                     logger.error(f"Failed to initialize MediaPipe Pose: {pose_init_err}. Pose features disabled.")
                     pose_detector = None; pose_context = None 
            audio_raw_features = master_audio_data.get('raw_features', {})
            audio_rms_times = audio_raw_features.get('rms_times')
            audio_rms_energy = audio_raw_features.get('rms_energy')
            audio_perc_times = audio_raw_features.get('perc_times', audio_rms_times) 
            audio_perc_ratio = audio_raw_features.get('percussive_ratio', np.zeros_like(audio_perc_times) if audio_perc_times is not None else []) 
            segment_boundaries = master_audio_data.get('segment_boundaries', [0, master_audio_data.get('duration', float('inf'))])
            if not segment_boundaries or len(segment_boundaries) < 2:
                segment_boundaries = [0, master_audio_data.get('duration', float('inf'))]
            if audio_rms_times is not None and not isinstance(audio_rms_times, np.ndarray): audio_rms_times = np.asarray(audio_rms_times)
            if audio_rms_energy is not None and not isinstance(audio_rms_energy, np.ndarray): audio_rms_energy = np.asarray(audio_rms_energy)
            if audio_perc_times is not None and not isinstance(audio_perc_times, np.ndarray): audio_perc_times = np.asarray(audio_perc_times)
            if audio_perc_ratio is not None and not isinstance(audio_perc_ratio, np.ndarray): audio_perc_ratio = np.asarray(audio_perc_ratio)
            logger.info("Processing frames & generating features (Ensemble v5.3)...")
            frame_iterator = clip.iter_frames(fps=fps, dtype="uint8", logger=None)
            pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"Analyzing {os.path.basename(videoPath)}", unit="frame", dynamic_ncols=True, leave=False, disable=tqdm is None)
            frame_timestamps = []; frame_flow_velocities = []; frame_pose_kinetics = []
            for frame_idx, frame_rgb in enumerate(frame_iterator):
                if frame_rgb is None:
                    logger.warning(f"Received None frame at index {frame_idx} from MoviePy. Stopping analysis for this video.")
                    break
                timestamp = frame_idx / fps
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                image_resized_bgr = self.BBZImageUtils.resizeTARGET(frame_bgr, TARGET_HEIGHT, TARGET_WIDTH)
                if image_resized_bgr is None or image_resized_bgr.size == 0:
                    logger.warning(f"Frame {frame_idx}: Resize failed or resulted in empty image. Skipping frame.")
                    if pbar: pbar.update(1)
                    continue 
                current_features = {'frame_index': frame_idx, 'timestamp': timestamp}
                pose_features_dict = {} 
                current_gray = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2GRAY)
                flow_velocity, current_flow_field = calculate_flow_velocity(prev_gray, current_gray)
                flow_acceleration = calculate_flow_acceleration(prev_flow, current_flow_field, frame_time_diff)
                current_features['flow_velocity'] = flow_velocity
                current_features['flow_acceleration'] = flow_acceleration
                current_features['camera_motion'] = flow_velocity 
                current_features['histogram_entropy'] = calculate_histogram_entropy(current_gray)
                frame_timestamps.append(timestamp); frame_flow_velocities.append(flow_velocity)
                depth_variance = 0.0
                if self.midas_model and self.midas_transform and self.midas_device:
                    try:
                        with torch.no_grad():
                            image_resized_rgb_midas = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB)
                            input_batch = self.midas_transform(image_resized_rgb_midas).to(self.midas_device)
                            prediction = self.midas_model(input_batch)
                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1), size=image_resized_rgb_midas.shape[:2],
                                mode="bicubic", align_corners=False
                            ).squeeze()
                            depth_map = prediction.cpu().numpy()
                            depth_min = depth_map.min(); depth_max = depth_map.max()
                            if depth_max > depth_min + 1e-6: 
                                norm_depth = (depth_map - depth_min) / (depth_max - depth_min)
                                depth_variance = float(np.var(norm_depth))
                    except Exception as midas_e:
                        if frame_idx % 100 == 0: logger.debug(f"MiDaS error on frame {frame_idx}: {midas_e}")
                current_features['depth_variance'] = depth_variance
                face_results = face_detector_util.process_frame(image_resized_bgr) if face_detector_util else None
                current_pose_results = None
                if pose_detector:
                    try:
                        image_resized_rgb_pose = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB)
                        image_resized_rgb_pose.flags.writeable = False 
                        current_pose_results = pose_detector.process(image_resized_rgb_pose)
                        image_resized_rgb_pose.flags.writeable = True
                    except Exception as pose_err:
                         if frame_idx % 100 == 0: logger.debug(f"Pose processing error on frame {frame_idx}: {pose_err}")
                pose_results_buffer.append(current_pose_results)
                lm_t2, lm_t1, lm_t = pose_results_buffer 
                mouth_crop_np = None
                if analysis_config.use_latent_sync and face_results and face_results.multi_face_landmarks:
                    mouth_crop_np = extract_mouth_crop(image_resized_bgr, face_results.multi_face_landmarks[0])
                current_features['mouth_crop'] = mouth_crop_np 
                kinetic = calculate_kinetic_energy_proxy(lm_t1, lm_t, frame_time_diff)
                jerk = calculate_movement_jerk_proxy(lm_t2, lm_t1, lm_t, frame_time_diff)
                pose_features_dict['kinetic_energy_proxy'] = kinetic
                pose_features_dict['movement_jerk_proxy'] = jerk
                frame_pose_kinetics.append(kinetic) 
                is_mouth_open, face_size_ratio, face_center_x = face_detector_util.get_heuristic_face_features(face_results, TARGET_HEIGHT, TARGET_WIDTH, analysis_config.mouth_open_threshold) if face_detector_util else (False, 0.0, 0.5)
                pose_features_dict['is_mouth_open'] = is_mouth_open
                pose_features_dict['face_size_ratio'] = face_size_ratio
                pose_features_dict['face_center_x'] = face_center_x
                current_features['pose_features'] = pose_features_dict
                mid_frame_time = timestamp + (frame_time_diff / 2.0) 
                current_features['audio_energy'] = self.get_feature_at_time(audio_rms_times, audio_rms_energy, mid_frame_time)
                current_features['percussive_ratio'] = self.get_feature_at_time(audio_perc_times, audio_perc_ratio, mid_frame_time)
                section_idx = -1
                for i in range(len(segment_boundaries) - 1):
                    if segment_boundaries[i] <= mid_frame_time < segment_boundaries[i+1]:
                        section_idx = i
                        break
                if section_idx == -1 and mid_frame_time >= segment_boundaries[-1] - 1e-6:
                     section_idx = len(segment_boundaries) - 2 
                current_features['musical_section_index'] = section_idx
                raw_score, dominant, intensity = self.calculate_base_heuristic_score(current_features, analysis_config)
                current_features['raw_score'] = raw_score
                current_features['dominant_contributor'] = dominant
                current_features['intensity_category'] = intensity
                current_features['boosted_score'] = raw_score
                current_features['is_beat_frame'] = False 
                all_frame_features.append(current_features)
                prev_gray = current_gray.copy()
                prev_flow = current_flow_field.copy() if current_flow_field is not None else None
                if pbar: pbar.update(1)
            if pbar: pbar.close()
            logger.debug("Calculating visual trends from frame features...")
            if len(frame_timestamps) > 1:
                 ts_diff = np.diff(frame_timestamps)
                 if np.any(ts_diff <= 0): 
                      logger.warning("Timestamps not monotonic, cannot reliably calculate trends.")
                      visual_flow_trend = np.zeros(len(frame_timestamps))
                      visual_pose_trend = np.zeros(len(frame_timestamps))
                 else:
                     visual_flow_trend = np.gradient(frame_flow_velocities, frame_timestamps)
                     visual_pose_trend = np.gradient(frame_pose_kinetics, frame_timestamps)
                 for i in range(len(all_frame_features)):
                      if i < len(visual_flow_trend): all_frame_features[i]['visual_flow_trend'] = float(visual_flow_trend[i])
                      else: all_frame_features[i]['visual_flow_trend'] = 0.0 
                      if i < len(visual_pose_trend): all_frame_features[i]['visual_pose_trend'] = float(visual_pose_trend[i])
                      else: all_frame_features[i]['visual_pose_trend'] = 0.0
            else: 
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
            logger.debug(f"Cleaning up resources for {os.path.basename(videoPath)}...")
            if pose_context:
                try: pose_context.__exit__(None, None, None) 
                except Exception as pose_close_err: logger.error(f"Error closing Pose context: {pose_close_err}")
            if face_detector_util: face_detector_util.close()
            if clip:
                try: clip.close()
                except Exception as clip_close_err: logger.error(f"Error closing MoviePy clip: {clip_close_err}")
            logger.debug(f"Resource cleanup for {os.path.basename(videoPath)} finished.")
        if not all_frame_features:
            logger.error(f"No features extracted for {videoPath}. Cannot create clips.")
            return None, None
        logger.debug("Applying V4 beat boost to base heuristic score...")
        self.apply_beat_boost(all_frame_features, master_audio_data, fps, analysis_config)
        potential_clips: List[ClipSegment] = []
        actual_total_frames = len(all_frame_features)
        logger.info("Creating potential segments using fixed/overlapping chunks...")
        min_clip_frames = max(MIN_POTENTIAL_CLIP_DURATION_FRAMES, int(analysis_config.min_potential_clip_duration_sec * fps))
        max_clip_frames = max(min_clip_frames + 1, int(analysis_config.max_sequence_clip_duration * fps * 1.2))
        step_frames = max(1, int(0.5 * fps)) 
        if actual_total_frames < min_clip_frames:
            logger.warning(f"Video too short ({actual_total_frames} frames) for min potential clip length ({min_clip_frames} frames). No clips generated.")
        else:
            for start_f in range(0, actual_total_frames - min_clip_frames + 1, step_frames):
                 end_f = min(start_f + max_clip_frames, actual_total_frames) 
                 if (end_f - start_f) >= min_clip_frames:
                     try:
                         clip_seg = ClipSegment(videoPath, start_f, end_f, fps, all_frame_features, analysis_config, master_audio_data)
                         if analysis_config.use_latent_sync and syncnet_model and full_mel_spectrogram is not None and mel_times is not None:
                              clip_seg._calculate_and_set_latent_sync(syncnet_model, full_mel_spectrogram, mel_times)
                         potential_clips.append(clip_seg)
                     except Exception as clip_err:
                         logger.warning(f"Failed to create ClipSegment for frames {start_f}-{end_f}: {clip_err}", exc_info=False)
        end_time_analysis = time.time() # Renamed variable
        analysis_duration = end_time_analysis - profiler_start_time if ENABLE_PROFILING else 0
        logger.info(f"--- Analysis & Clip Creation complete for {os.path.basename(videoPath)} ({analysis_duration:.2f}s) ---")
        logger.info(f"Created {len(potential_clips)} potential clips.")
        if ENABLE_PROFILING: logger.debug(f"PROFILING: Video Analysis ({os.path.basename(videoPath)}) took {analysis_duration:.3f}s")
        return all_frame_features, potential_clips

    # Method for saving analysis data, if it were part of VideousMain
    # def saveAnalysisData(self, video_path, frame_features, potential_clips, output_dir, analysis_config):
    #     # Example implementation (not part of original prompt, but for completeness if process_single_video calls it)
    #     base_name = os.path.splitext(os.path.basename(video_path))[0]
    #     output_path = os.path.join(output_dir, f"{base_name}_analysis.json")
    #     data_to_save = {
    #         "video_path": video_path,
    #         "analysis_config": dataclasses.asdict(analysis_config),
    #         # "frame_features": frame_features, # Potentially very large
    #         "potential_clips_metadata": [
    #             {
    #                 "source_video_path": clip.source_video_path,
    #                 "start_frame": clip.start_frame,
    #                 "end_frame": clip.end_frame,
    #                 "duration": clip.duration,
    #                 "avg_boosted_score": clip.avg_boosted_score, # Example field
    #                 "latent_sync_score": clip.latent_sync_score
    #             } for clip in potential_clips
    #         ]
    #     }
    #     try:
    #         with open(output_path, 'w', encoding='utf-8') as f:
    #             json.dump(data_to_save, f, indent=2, cls=NumpyEncoder) # NumpyEncoder would be needed for numpy types
    #         logger.info(f"Analysis data saved to {output_path}")
    #     except Exception as e:
    #         logger.error(f"Failed to save analysis data for {video_path}: {e}")

# ========================================================================
#          SEQUENCE BUILDER - PHYSICS PARETO MC (Basic V4 Implementation)
# ========================================================================
class SequenceBuilderPhysicsMC:
    def __init__(self, all_potential_clips: List[ClipSegment], audio_data: Dict, analysis_config: AnalysisConfig):
        logger.warning("Physics Pareto MC mode is using the older V4 scoring logic. Ensemble features (SyncNet, LVRE, etc.) are NOT used in this mode.")
        self.all_clips = all_potential_clips
        self.audio_data = audio_data
        self.analysis_config = analysis_config 
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
        self.effects: Dict[str, EffectParams] = {} 

    def get_audio_segment_at(self, time_val): 
        if not self.audio_segments: return None
        for seg in self.audio_segments:
            if seg['start'] <= time_val < seg['end']:
                return seg
        if time_val >= self.audio_segments[-1]['end'] - 1e-6:
            return self.audio_segments[-1]
        logger.debug(f"Time {time_val:.2f}s outside defined audio segments.")
        return None 

    def build_sequence(self) -> List[ClipSegment]: 
        logger.info(f"--- Composing Sequence (Physics Pareto MC Mode - V4 Logic - {self.mc_iterations} iterations) ---")
        if not self.all_clips or not self.audio_segments or self.target_duration <= 0 or not self.effects:
            logger.error("Physics MC pre-conditions not met (clips, audio segments, duration, or effects dictionary missing).")
            return []
        eligible_clips = [c for c in self.all_clips if c.duration >= self.min_clip_duration]
        if not eligible_clips:
            logger.error(f"No clips meet minimum duration ({self.min_clip_duration:.2f}s) for Physics MC.")
            return []
        self.all_clips = eligible_clips 
        logger.info(f"Starting Physics MC with {len(self.all_clips)} eligible clips.")
        pareto_front: List[Tuple[List[Tuple[ClipSegment, float, EffectParams]], List[float]]] = []
        successful_sims = 0
        pbar_mc = tqdm(range(self.mc_iterations), desc="MC Simulations (V4 Logic)", leave=False, disable=tqdm is None)
        for i in pbar_mc:
            sim_seq_info = None; scores = None 
            try:
                sim_seq_info = self._run_stochastic_build()
                if sim_seq_info:
                    successful_sims += 1
                    scores = self._evaluate_pareto(sim_seq_info)
                    if all(np.isfinite(s) for s in scores):
                        self._update_pareto_front(pareto_front, (sim_seq_info, scores))
                    else:
                        logger.debug(f"MC iter {i}: Invalid scores generated: {scores}")
            except Exception as mc_err:
                logger.error(f"MC simulation iteration {i} failed: {mc_err}", exc_info=False)
        if pbar_mc: pbar_mc.close()
        logger.info(f"MC simulations complete: {successful_sims}/{self.mc_iterations} sims yielded sequences. Pareto front size: {len(pareto_front)}")
        if not pareto_front:
            logger.error("Monte Carlo simulation yielded no valid sequences on the Pareto front.")
            return []
        obj_weights = [self.w_r, self.w_m, self.w_c, self.w_v, self.w_e]
        best_solution = max(pareto_front, key=lambda item: sum(s * w for s, w in zip(item[1], obj_weights)))
        logger.info(f"Chosen sequence objectives (NegR, M, C, V, NegEC): {[f'{s:.3f}' for s in best_solution[1]]}")
        final_sequence_info = best_solution[0]
        final_sequence_segments: List[ClipSegment] = []
        current_t = 0.0
        for i, (clip_item, duration, effect) in enumerate(final_sequence_info):
            if not isinstance(clip_item, ClipSegment):
                logger.warning(f"Invalid item in chosen sequence info at index {i}. Skipping.")
                continue
            clip_item.sequence_start_time = current_t
            clip_item.sequence_end_time = current_t + duration
            clip_item.chosen_duration = duration
            clip_item.chosen_effect = effect
            clip_item.subclip_start_time_in_source = clip_item.start_time 
            clip_item.subclip_end_time_in_source = min(clip_item.start_time + duration, clip_item.end_time) 
            final_sequence_segments.append(clip_item)
            current_t += duration
        if not final_sequence_segments:
            logger.error("Physics MC failed to reconstruct final sequence from chosen Pareto solution.")
            return []
        logger.info("--- Sequence Composition Complete (Physics Pareto MC - V4 Logic) ---")
        logger.info(f"Final Duration: {current_t:.2f}s, Clips: {len(final_sequence_segments)}")
        return final_sequence_segments

    def _run_stochastic_build(self):
        sequence_info: List[Tuple[ClipSegment, float, EffectParams]] = [] 
        current_time = 0.0
        available_clip_indices = list(range(len(self.all_clips)))
        random.shuffle(available_clip_indices) 
        last_clip_segment: Optional[ClipSegment] = None
        num_sources = len(set(c.source_video_path for c in self.all_clips))
        while current_time < self.target_duration and available_clip_indices:
            audio_seg = self.get_audio_segment_at(current_time)
            if not audio_seg: break 
            candidates_info = [] 
            total_prob = 0.0
            for list_idx_pos, original_clip_index in enumerate(available_clip_indices):
                clip_item = self.all_clips[original_clip_index]
                prob = clip_item.clip_audio_fit(audio_seg, self.analysis_config)
                if num_sources > 1 and last_clip_segment and str(clip_item.source_video_path) == str(last_clip_segment.source_video_path):
                    prob *= (1.0 - self.analysis_config.variety_repetition_penalty)
                if prob > 1e-5: 
                    candidates_info.append((clip_item, list_idx_pos, prob))
                    total_prob += prob
            if not candidates_info: break 
            probabilities = [p / (total_prob + 1e-9) for _, _, p in candidates_info] 
            try:
                chosen_candidate_local_idx = random.choices(range(len(candidates_info)), weights=probabilities, k=1)[0]
            except ValueError: 
                if candidates_info: chosen_candidate_local_idx = random.randrange(len(candidates_info)) 
                else: break 
            chosen_clip, chosen_list_idx_pos, _ = candidates_info[chosen_candidate_local_idx]
            remaining_time = self.target_duration - current_time
            chosen_duration = min(chosen_clip.duration, remaining_time, self.max_clip_duration)
            chosen_duration = max(chosen_duration, self.min_clip_duration if remaining_time >= self.min_clip_duration else 0.01)
            chosen_duration = max(0.01, chosen_duration) 
            effect_options = list(self.effects.values())
            efficiencies = []
            for e_item in effect_options:
                denom = e_item.tau * e_item.psi; numer = e_item.epsilon
                eff = ((numer + 1e-9) / (denom + 1e-9)) if abs(denom) > 1e-9 else (0.0 if abs(numer) < 1e-9 else 1e9) 
                efficiencies.append(eff)
            cut_index = next((i_idx for i_idx, e_item in enumerate(effect_options) if e_item.type == "cut"), -1)
            if cut_index != -1: efficiencies[cut_index] = max(efficiencies[cut_index], 1.0) * 2.0 
            positive_efficiencies = [max(0, eff) for eff in efficiencies]
            total_efficiency = sum(positive_efficiencies)
            chosen_effect = self.effects.get('cut', EffectParams(type='cut')) 
            if total_efficiency > 1e-9 and effect_options:
                 effect_probs = [eff / total_efficiency for eff in positive_efficiencies]
                 sum_probs = sum(effect_probs)
                 if abs(sum_probs - 1.0) > 1e-6: effect_probs = [p / (sum_probs + 1e-9) for p in effect_probs]
                 try:
                     chosen_effect = random.choices(effect_options, weights=effect_probs, k=1)[0]
                 except (ValueError, IndexError) as choice_err:
                      logger.debug(f"Effect choice failed: {choice_err}. Defaulting to cut.")
                      chosen_effect = self.effects.get('cut', EffectParams(type='cut'))
            sequence_info.append((chosen_clip, chosen_duration, chosen_effect))
            last_clip_segment = chosen_clip
            current_time += chosen_duration
            available_clip_indices.pop(chosen_list_idx_pos)
        final_sim_duration = sum(item[1] for item in sequence_info)
        return sequence_info if final_sim_duration >= self.min_clip_duration else None

    def _evaluate_pareto(self, seq_info):
        if not seq_info: return [-1e9] * 5 
        num_clips = len(seq_info); total_duration = sum(item[1] for item in seq_info)
        if total_duration <= 1e-6: return [-1e9] * 5 
        w_r, w_m, w_c, w_v, w_e = self.w_r, self.w_m, self.w_c, self.w_v, self.w_e
        sigma_m_sq = self.analysis_config.mood_similarity_variance**2 * 2
        kd = self.analysis_config.continuity_depth_weight
        lambda_penalty = self.analysis_config.variety_repetition_penalty
        num_sources = len(set(item[0].source_video_path for item in seq_info))
        r_score_sum = 0.0; num_trans_r = 0; current_t = 0.0
        for i, (_, duration, _) in enumerate(seq_info):
            trans_time = current_t + duration
            if i < num_clips - 1: 
                nearest_b = self._nearest_beat_time(trans_time)
                if nearest_b is not None and self.beat_period > 1e-6:
                    offset_norm = abs(trans_time - nearest_b) / self.beat_period
                    r_score_sum += offset_norm**2
                    num_trans_r += 1
            current_t = trans_time
        avg_sq_offset = (r_score_sum / num_trans_r) if num_trans_r > 0 else 1.0 
        neg_r_score = -w_r * avg_sq_offset 
        m_score_sum = 0.0; mood_calcs = 0; current_t = 0.0
        for clip_item, duration, _ in seq_info:
            mid_time = current_t + duration / 2.0
            audio_seg = self.get_audio_segment_at(mid_time)
            if audio_seg:
                vid_mood = np.asarray(clip_item.mood_vector)
                aud_mood = np.asarray(audio_seg.get('m_i', [0.0, 0.0]))
                mood_dist_sq = np.sum((vid_mood - aud_mood)**2)
                m_score_sum += exp(-mood_dist_sq / (sigma_m_sq + 1e-9))
                mood_calcs += 1
            current_t += duration
        m_score = w_m * (m_score_sum / mood_calcs if mood_calcs > 0 else 0.0) 
        c_score_sum = 0.0; num_trans_c = 0
        for i in range(num_clips - 1):
            clip1, _, effect = seq_info[i]
            clip2, _, _ = seq_info[i+1]
            f1 = clip1.get_feature_vector(self.analysis_config)
            f2 = clip2.get_feature_vector(self.analysis_config)
            safe_kd = max(0.0, kd) 
            delta_e_sq = (f1[0]-f2[0])**2 + (f1[1]-f2[1])**2 + safe_kd*(f1[2]-f2[2])**2
            max_delta_e_sq = 1**2 + 1**2 + safe_kd*(1**2) 
            delta_e_norm_sq = delta_e_sq / (max_delta_e_sq + 1e-9)
            cont_term = (1.0 - np.sqrt(np.clip(delta_e_norm_sq, 0.0, 1.0)))
            c_score_sum += cont_term + effect.epsilon
            num_trans_c += 1
        c_score = w_c * (c_score_sum / num_trans_c if num_trans_c > 0 else 1.0) 
        valid_phis = [item[0].phi for item in seq_info if isinstance(item[0], ClipSegment) and item[0].phi is not None and np.isfinite(item[0].phi)]
        avg_phi = np.mean(valid_phis) if valid_phis else 0.0
        repetition_count = 0; num_trans_v = 0
        if num_sources > 1: 
            for i in range(num_clips - 1):
                p1 = str(seq_info[i][0].source_video_path)
                p2 = str(seq_info[i+1][0].source_video_path)
                if p1 and p2 and p1 == p2: repetition_count += 1 
                num_trans_v += 1
        rep_term = lambda_penalty * (repetition_count / num_trans_v if num_trans_v > 0 else 0)
        max_entropy = log(256) 
        avg_phi_norm = avg_phi / max_entropy if max_entropy > 0 else 0.0
        v_score = w_v * (avg_phi_norm - rep_term) 
        ec_score_sum = 0.0; cost_calcs = 0
        for _, _, effect in seq_info:
             psi_tau = effect.psi * effect.tau 
             epsilon = effect.epsilon 
             cost = (psi_tau + 1e-9) / (epsilon + 1e-9) if abs(epsilon) > 1e-9 else (psi_tau + 1e-9) / 1e-9 
             ec_score_sum += cost
             cost_calcs += 1
        avg_cost = (ec_score_sum / cost_calcs if cost_calcs > 0 else 0.0)
        neg_ec_score = -w_e * avg_cost 
        final_scores = [neg_r_score, m_score, c_score, v_score, neg_ec_score]
        return [float(s) if np.isfinite(s) else -1e9 for s in final_scores]

    def _nearest_beat_time(self, time_sec):
        if not self.beat_times: return None
        beat_times_arr = np.asarray(self.beat_times)
        if len(beat_times_arr) == 0: return None
        closest_beat_index = np.argmin(np.abs(beat_times_arr - time_sec))
        return float(beat_times_arr[closest_beat_index])

    def _update_pareto_front(self, front, new_solution):
        new_seq_info, new_scores = new_solution
        if not isinstance(new_scores, list) or len(new_scores) != 5 or not all(isinstance(s, float) and np.isfinite(s) for s in new_scores):
            logger.debug(f"Skipping Pareto update due to invalid new scores: {new_scores}")
            return
        dominated_indices = set()
        is_dominated_by_existing = False
        indices_to_check = list(range(len(front))) 
        for i in reversed(indices_to_check): 
            if i >= len(front): continue
            existing_seq_info, existing_scores = front[i]
            if not isinstance(existing_scores, list) or len(existing_scores) != 5 or not all(isinstance(s, float) and np.isfinite(s) for s in existing_scores):
                logger.warning(f"Removing solution with invalid scores from Pareto front at index {i}: {existing_scores}")
                del front[i]
                continue
            if self._dominates(new_scores, existing_scores):
                dominated_indices.add(i)
            if self._dominates(existing_scores, new_scores):
                is_dominated_by_existing = True
                break 
        if not is_dominated_by_existing:
            for i in sorted(list(dominated_indices), reverse=True):
                 if 0 <= i < len(front): del front[i] 
            front.append(new_solution)

    def _dominates(self, scores1, scores2):
        if len(scores1) != len(scores2): raise ValueError("Score lists must have same length for dominance check.")
        better_in_at_least_one = False
        for s1, s2 in zip(scores1, scores2):
            if s1 < s2 - 1e-9: 
                return False
            if s1 > s2 + 1e-9: 
                better_in_at_least_one = True
        return better_in_at_least_one

# ========================================================================
#        MOVIEPY VIDEO BUILDING FUNCTION (Includes GFPGAN Hook)
# ========================================================================
def buildSequenceVideo(final_sequence: List[ClipSegment], output_video_path: str, master_audio_path: str, render_config: RenderConfig):
    logger.info(f"Rendering video to '{os.path.basename(output_video_path)}' with audio '{os.path.basename(master_audio_path)}' using MoviePy make_frame...")
    start_time_render = time.time() # Renamed variable
    if ENABLE_PROFILING: tracemalloc.start(); start_memory = tracemalloc.get_traced_memory()[0]
    if not final_sequence: logger.error("Cannot build: Empty final sequence provided."); raise ValueError("Empty sequence")
    if not master_audio_path or not os.path.exists(master_audio_path): logger.error(f"Cannot build: Master audio not found at '{master_audio_path}'"); raise FileNotFoundError("Master audio not found")
    if not output_video_path: logger.error("Cannot build: Output video path not specified."); raise ValueError("Output path required")
    width = render_config.resolution_width; height = render_config.resolution_height; fps = render_config.fps
    if not isinstance(fps, (int, float)) or fps <= 0: fps = 30; logger.warning(f"Invalid render FPS {render_config.fps}, using default 30.")
    gfpgan_enhancer = None
    if render_config.use_gfpgan_enhance and GFPGAN_AVAILABLE: # Check GFPGAN_AVAILABLE
        logger.info("GFPGAN enhancement enabled. Attempting to load model...")
        try:
            model_path = render_config.gfpgan_model_path 
            if not model_path:
                 raise ValueError("GFPGAN model path not specified in RenderConfig.")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GFPGAN model not found at {model_path}. Enhancement disabled. Please download manually or check path.")
            gfpgan_enhancer = GFPGANer(model_path=model_path, upscale=1, arch='clean',
                                       channel_multiplier=2, bg_upsampler=None)
            logger.info("GFPGAN Enhancer loaded successfully.")
        except FileNotFoundError as fnf_err:
            logger.error(f"{fnf_err}") 
            render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
        except ValueError as val_err: 
            logger.error(f"{val_err}")
            render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
        except Exception as gfpgan_load_err:
            logger.error(f"Failed to load GFPGAN model: {gfpgan_load_err}", exc_info=True)
            render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)
    elif render_config.use_gfpgan_enhance and not GFPGAN_AVAILABLE:
        logger.warning("GFPGAN enhancement enabled in config, but GFPGAN library is not available. Disabling.")
        render_config = dataclass_replace(render_config, use_gfpgan_enhance=False)

    source_clips_dict: Dict[str, Optional[VideoFileClip]] = {}
    logger.info("Pre-loading/preparing source video readers...")
    unique_source_paths = sorted(list(set(str(seg.source_video_path) for seg in final_sequence)))
    for source_path in unique_source_paths:
        if not source_path or not os.path.exists(source_path):
            logger.error(f"Source video not found: {source_path}. Frames from this source will be black.")
            source_clips_dict[source_path] = None; continue
        try:
            logger.debug(f"Preparing reader for: {os.path.basename(source_path)}")
            clip_obj = VideoFileClip(source_path, audio=False) 
            _ = clip_obj.duration
            if not hasattr(clip_obj, 'reader') or clip_obj.reader is None:
                 raise RuntimeError("MoviePy reader initialization failed.")
            source_clips_dict[source_path] = clip_obj
        except Exception as load_err:
            logger.error(f"Failed to load source video {os.path.basename(source_path)}: {load_err}")
            source_clips_dict[source_path] = None 
    def make_frame(t):
        active_segment = None
        final_source_time = -1.0 
        source_path_log = "N/A"
        for segment in final_sequence:
            if segment.sequence_start_time <= t < segment.sequence_end_time + 1e-6:
                active_segment = segment
                break
        if active_segment is None and final_sequence and abs(t - final_sequence[-1].sequence_end_time) < 1e-6:
            active_segment = final_sequence[-1]
        if active_segment:
            source_path = str(active_segment.source_video_path)
            source_path_log = os.path.basename(source_path) 
            source_clip = source_clips_dict.get(source_path)
            if source_clip and hasattr(source_clip, 'get_frame'):
                try:
                    clip_time_in_seq = t - active_segment.sequence_start_time
                    source_time = active_segment.subclip_start_time_in_source + clip_time_in_seq
                    subclip_start = active_segment.subclip_start_time_in_source
                    subclip_end = active_segment.subclip_end_time_in_source
                    source_dur = source_clip.duration if source_clip.duration else 0
                    final_source_time = np.clip(source_time, 0, source_dur - 1e-6 if source_dur > 0 else 0)
                    final_source_time = np.clip(final_source_time, subclip_start, subclip_end - 1e-6 if subclip_end > subclip_start else subclip_start)
                    frame_rgb = source_clip.get_frame(final_source_time)
                    if frame_rgb is None: raise ValueError("get_frame returned None")
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
                        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
                    if gfpgan_enhancer and render_config.use_gfpgan_enhance:
                        if active_segment.face_presence_ratio > 0.1: 
                            try:
                                _, _, restored_img = gfpgan_enhancer.enhance(
                                    frame_bgr, 
                                    has_aligned=False, 
                                    only_center_face=False, 
                                    paste_back=True, 
                                    weight=render_config.gfpgan_fidelity_weight 
                                )
                                if restored_img is not None:
                                    frame_bgr = restored_img 
                                else:
                                     logger.debug(f"GFPGAN enhancement returned None for frame at t={t:.2f}")
                            except Exception as gfpgan_err:
                                if int(t*10) % 50 == 0: 
                                     logger.warning(f"GFPGAN enhancement failed for frame at t={t:.2f}: {gfpgan_err}")
                    final_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    return final_frame_rgb
                except Exception as frame_err:
                    logger.error(f"Error getting/processing frame at t={t:.3f} (Source: {source_path_log} @ {final_source_time:.3f}): {frame_err}", exc_info=False) 
                    return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                logger.warning(f"Source clip '{source_path_log}' invalid or reader failed at t={t:.3f}. Returning black frame.")
                return np.zeros((height, width, 3), dtype=np.uint8)
        else:
            seq_end_time = final_sequence[-1].sequence_end_time if final_sequence else 0
            logger.warning(f"Time t={t:.4f}s outside sequence range [0, {seq_end_time:.4f}s]. Returning black frame.")
            return np.zeros((height, width, 3), dtype=np.uint8)
    master_audio = None; sequence_clip_mvp = None; temp_audio_filepath = None
    try:
        total_duration = final_sequence[-1].sequence_end_time if final_sequence else 0
        if total_duration <= 0: raise ValueError(f"Sequence has zero or negative duration ({total_duration}).")
        logger.info(f"Creating final VideoClip object (Duration: {total_duration:.2f}s, FPS: {fps})")
        sequence_clip_mvp = VideoClip(make_frame, duration=total_duration, ismask=False)
        logger.debug(f"Loading master audio: {master_audio_path}")
        master_audio = AudioFileClip(master_audio_path)
        logger.debug(f"Master audio duration: {master_audio.duration:.2f}s")
        if master_audio.duration > total_duration + 1e-3: 
            logger.info(f"Audio duration ({master_audio.duration:.2f}s) > Video duration ({total_duration:.2f}s). Trimming audio.")
            master_audio = master_audio.subclip(0, total_duration)
        elif master_audio.duration < total_duration - 1e-3: 
            logger.warning(f"Video duration ({total_duration:.2f}s) > Audio duration ({master_audio.duration:.2f}s). Trimming video to audio length.")
            total_duration = master_audio.duration
            sequence_clip_mvp = sequence_clip_mvp.set_duration(total_duration)
        if master_audio:
            sequence_clip_mvp = sequence_clip_mvp.set_audio(master_audio)
        else:
            logger.warning("No master audio loaded or audio duration is zero. Rendering silent video.")
        temp_audio_filename = f"temp-audio_{int(time.time())}_{random.randint(1000,9999)}.m4a" 
        temp_audio_dir = os.path.dirname(output_video_path) or "."
        os.makedirs(temp_audio_dir, exist_ok=True)
        temp_audio_filepath = os.path.join(temp_audio_dir, temp_audio_filename)
        ffmpeg_params_list = []
        if render_config.preset: ffmpeg_params_list.extend(["-preset", str(render_config.preset)])
        if render_config.crf is not None: ffmpeg_params_list.extend(["-crf", str(render_config.crf)])
        write_params = {
            "codec": render_config.video_codec,
            "audio_codec": render_config.audio_codec,
            "temp_audiofile": temp_audio_filepath,
            "remove_temp": True,
            "threads": render_config.threads,
            "preset": None, 
            "logger": 'bar' if tqdm is not None else None, 
            "write_logfile": False, 
            "audio_bitrate": render_config.audio_bitrate,
            "fps": fps,
            "ffmpeg_params": ffmpeg_params_list if ffmpeg_params_list else None
        }
        logger.info(f"Writing final video using MoviePy write_videofile...")
        logger.debug(f"Write params: Codec={write_params['codec']}, AudioCodec={write_params['audio_codec']}, Threads={write_params['threads']}, FPS={write_params['fps']}, Preset={render_config.preset}, CRF={render_config.crf}, Params={write_params['ffmpeg_params']}")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True) 
        sequence_clip_mvp.write_videofile(output_video_path, **write_params)
        if ENABLE_PROFILING and tracemalloc.is_tracing():
             current_mem, peak_mem = tracemalloc.get_traced_memory()
             end_memory = current_mem; tracemalloc.stop()
             logger.info(f"Render Perf: Time: {time.time() - start_time_render:.2f}s, PyMem Δ: {(end_memory - start_memory) / 1024**2:.2f} MB, Peak: {peak_mem / 1024**2:.2f} MB")
        else: logger.info(f"Render took {time.time() - start_time_render:.2f} seconds.")
        logger.info(f"MoviePy rendering successful: '{output_video_path}'")
    except Exception as e:
        if ENABLE_PROFILING and tracemalloc.is_tracing(): tracemalloc.stop()
        logger.error(f"MoviePy rendering failed: {e}", exc_info=True)
        if os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
                logger.info(f"Removed potentially incomplete output file: {output_video_path}")
            except OSError as del_err:
                logger.warning(f"Could not remove failed output file {output_video_path}: {del_err}")
        if temp_audio_filepath and os.path.exists(temp_audio_filepath):
             try:
                 os.remove(temp_audio_filepath)
                 logger.debug(f"Removed temp audio file {temp_audio_filepath} after error.")
             except OSError as del_err:
                 logger.warning(f"Could not remove temp audio file {temp_audio_filepath} after error: {del_err}")
        raise 
    finally:
        logger.debug("Cleaning up MoviePy objects...")
        if sequence_clip_mvp and hasattr(sequence_clip_mvp, 'close'):
            try: sequence_clip_mvp.close() 
            except Exception as e: logger.debug(f"Minor error closing sequence_clip: {e}")
        if master_audio and hasattr(master_audio, 'close'):
            try: master_audio.close()
            except Exception as e: logger.debug(f"Minor error closing master_audio: {e}")
        for clip_key, source_clip_obj in source_clips_dict.items():
            if source_clip_obj and hasattr(source_clip_obj, 'close'):
                try: source_clip_obj.close()
                except Exception as e: logger.debug(f"Minor error closing source clip {clip_key}: {e}")
        source_clips_dict.clear() 
        import gc
        gc.collect()
        logger.debug("MoviePy clip cleanup attempt finished.")

# ========================================================================
#         WORKER FUNCTION (Returns None for Frame Features)
# ========================================================================
def process_single_video(video_path: str, master_audio_data: Dict, analysis_config: AnalysisConfig, output_dir: str) -> Tuple[str, str, List[ClipSegment]]:
    start_t = time.time(); pid = os.getpid(); thread_name = threading.current_thread().name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    worker_logger = logging.getLogger(f"Worker.{pid}.{thread_name}") 
    if not worker_logger.hasHandlers(): 
        ch = logging.StreamHandler(sys.stdout); formatter = logging.Formatter(f'%(asctime)s - %(levelname)-8s - Worker {pid} - %(message)s')
        ch.setFormatter(formatter); worker_logger.addHandler(ch); worker_logger.setLevel(logging.INFO) 
    worker_logger.info(f"Starting Analysis: {base_name}")
    status = "Unknown Error"; potential_clips: List[ClipSegment] = []; frame_features: Optional[List[Dict]] = None
    try:
        analyzer = VideousMain() 
        frame_features, potential_clips_result = analyzer.analyzeVideo(video_path, analysis_config, master_audio_data)
        if potential_clips_result is None:
            status = "Analysis Failed (returned None)"
            potential_clips = []
        elif not potential_clips_result:
            status = "Analysis OK (0 potential clips)"
            potential_clips = []
        else:
            potential_clips = [clip_item for clip_item in potential_clips_result if isinstance(clip_item, ClipSegment)]
            status = f"Analysis OK ({len(potential_clips)} potential clips)"
            if analysis_config.save_analysis_data:
                 save_frame_features = None 
                 if (save_frame_features or potential_clips): 
                     try:
                         # The method saveAnalysisData is not defined in VideousMain in the provided script.
                         # Commenting out the call. If this method is intended to exist, it should be added to VideousMain.
                         # analyzer.saveAnalysisData(video_path, save_frame_features, potential_clips, output_dir, analysis_config)
                         logger.info(f"Note: Call to analyzer.saveAnalysisData skipped as method is not defined in provided VideousMain for {base_name}.")
                         pass
                     except Exception as save_err:
                         worker_logger.error(f"Save analysis data failed for {base_name}: {save_err}")
                 else:
                     worker_logger.debug(f"Save analysis data requested for {base_name}, but nothing to save.")
    except Exception as e:
        status = f"Failed: {type(e).__name__}"
        worker_logger.error(f"!!! FATAL ERROR analyzing {base_name} in worker {pid} !!!", exc_info=True)
        potential_clips = []; frame_features = None 
    finally:
        if 'analyzer' in locals(): del analyzer 
        frame_features = None 
    end_t = time.time()
    worker_logger.info(f"Finished Analysis {base_name} ({status}) in {end_t - start_t:.2f}s")
    return (video_path, status, potential_clips if potential_clips is not None else [])

# ========================================================================
#                      APP INTERFACE (GUI Removed)
# ========================================================================
# VideousApp class has been removed.
# The following is the application entry point, adapted for non-GUI execution.
# ========================================================================
"""
# requirements.txt for Videous Chef - Ensemble Edition v5.3 (Optimized & Fixed)

# Core UI & Analysis
# customtkinter>=5.2.0,<6.0.0 # REMOVED (GUI)
opencv-python>=4.6.0,<5.0.0
numpy>=1.21.0,<2.0.0
scipy>=1.8.0
matplotlib>=3.5.0
# tkinterdnd2-universal>=2.1.0 # REMOVED (GUI)
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
# Global flags for optional libraries are defined near the top of the script.
# PySceneDetect is not used in the provided script, so PYSCENEDETECT_AVAILABLE is not strictly needed here.
PYSCENEDETECT_AVAILABLE = False 
try:
    import pyscenedetect
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    pass


if __name__ == "__main__":
    if MULTIPROCESSING_AVAILABLE:
        multiprocessing.freeze_support()
        try:
            default_method = multiprocessing.get_start_method(allow_none=True)
            if sys.platform != 'win32':
                if 'spawn' in multiprocessing.get_all_start_methods():
                    multiprocessing.set_start_method('spawn', force=True)
                    print("INFO: Set MP start method to 'spawn'.")
                else:
                    print(f"WARNING: 'spawn' MP method unavailable. Using default: {default_method}.")
            else:
                print(f"INFO: Using MP start method '{default_method}' on Windows.")
                if 'spawn' not in multiprocessing.get_all_start_methods():
                    print(f"WARNING: 'spawn' MP method unexpectedly unavailable on Windows?")
        except Exception as mp_setup_e:
            print(f"WARNING: MP start method setup error: {mp_setup_e}. Using default.")
    else:
        print("WARNING: Multiprocessing module not loaded, parallel disabled.")

    print("--- Videous Chef - Ensemble Edition v5.4/5.5 (API Mode) ---")
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s [%(threadName)s]')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = None
    try:
        log_dir = "logs_v5.4_api" # Different log dir for API mode
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"videous_chef_api_{time.strftime('%Y%m%d_%H%M%S')}.log")
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
    logger = root_logger # Ensure global logger is this one
    logger.info(log_setup_message)

    try:
        logger.info("Checking essential non-GUI dependencies...")
        essential_deps = {"numpy": "numpy", "opencv-python": "cv2"}
        missing_critical = []
        # GUI specific checks removed: TKINTER_AVAILABLE, CTK_AVAILABLE
        if not MOVIEPY_AVAILABLE: missing_critical.append("moviepy")
        if not TORCH_AVAILABLE: missing_critical.append("torch") # TORCH_AVAILABLE set by try-except import
        if not TORCHAUDIO_AVAILABLE: missing_critical.append("torchaudio") # TORCHAUDIO_AVAILABLE set by try-except import
        if not MEDIAPIPE_AVAILABLE: missing_critical.append("mediapipe") # MEDIAPIPE_AVAILABLE set by try-except import
        if not SOUNDFILE_AVAILABLE: missing_critical.append("soundfile") # SOUNDFILE_AVAILABLE set by try-except import

        for pkg, mod_name in essential_deps.items():
            try:
                __import__(mod_name)
                logger.debug(f"  [OK] {mod_name}")
            except ImportError:
                logger.critical(f"  [FAIL] {mod_name} ({pkg})")
                missing_critical.append(pkg)
        if missing_critical:
            err_msg = f"Critical non-GUI libraries missing: {', '.join(missing_critical)}\nPlease install requirements.\nExiting."
            logger.critical(err_msg)
            sys.exit(1)
        logger.info("Essential non-GUI dependencies OK.")

        try:
            ffmpeg_cmd = "ffmpeg"
            result = subprocess.run([ffmpeg_cmd, "-version"], capture_output=True, text=True, check=False, timeout=5, encoding='utf-8')
            if result.returncode != 0 or "ffmpeg version" not in result.stdout.lower():
                raise FileNotFoundError(f"FFmpeg check failed (cmd: '{ffmpeg_cmd}'). Output: {result.stdout} {result.stderr}")
            logger.info(f"FFmpeg check OK (via '{ffmpeg_cmd}').")
        except FileNotFoundError as fnf_err:
            err_msg = f"CRITICAL: FFmpeg not found or failed during version check.\nDetails: {fnf_err}\nPlease install FFmpeg and ensure it's in your system's PATH.\nExiting."
            logger.critical(err_msg)
            sys.exit(1)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg check timed out. Assuming it's available if this is not a fresh install.")
        except Exception as ff_e:
            logger.error(f"FFmpeg check encountered an error: {ff_e}. Proceeding, but FFmpeg is crucial.")

        try:
            if TORCH_AVAILABLE:
                logger.info(f"PyTorch version: {torch.__version__}")
                # device = get_device() # get_device logs its own choice
                # logger.info(f"PyTorch using device: {device}")
            else:
                logger.warning("PyTorch not available, cannot check version or device.")
        except Exception as pt_e:
            logger.warning(f"PyTorch check error: {pt_e}")

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
        # tkinterdnd2 is GUI specific, so its check is removed

        logger.info("Application core initialized in API mode.")
        logger.info("To use the API, import necessary classes (e.g., VideousMain, AnalysisConfig, RenderConfig) and functions.")
        logger.info("Example: ")
        logger.info("  config = AnalysisConfig()")
        logger.info("  audio_utils = BBZAudioUtils()")
        logger.info("  master_audio_data = audio_utils.analyze_audio('path/to/audio.wav', config)")
        logger.info("  # ... and so on for video analysis and rendering.")

    except SystemExit as se:
        logger.warning(f"Application exited during startup (Code: {se.code}).")
    except Exception as e:
        logger.critical(f"!!! UNHANDLED STARTUP ERROR (API Mode) !!!", exc_info=True)
    logger.info("--- Videous Chef (API Mode) session ended ---")
