
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

try:
	import colour
	_colour_available = True
except ImportError:
	colour = None
	_colour_available = False

try:
	from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
	_sam_available = True
except ImportError:
	_sam_available = False

try:
	from ultralytics import SAM
	_sam2_available = True
except ImportError:
	_sam2_available = False

import argparse

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[
		logging.FileHandler(f"logs/video_effects_{time.strftime('%Y%m%d_%H%M%S')}.log"),
		logging.StreamHandler()
	]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
DEFAULT_FPS = 30.0
CONFIG_FILE = "config.json"
GOLD_TINT_COLOR = (210, 165, 30)  # BGR format for OpenCV
TRAIL_START_ALPHA, TRAIL_END_ALPHA = 1.0, 0.3
TRAIL_RADIUS = 2
TINT_STRENGTH = 0.2
SEGMENTATION_THRESHOLD = 0.5
MASK_BLUR_KERNEL = (21, 21)
RVM_DEFAULT_BG_COLOR = (0, 120, 0) # Green screen BGR
SAM_MASK_ALPHA = 0.3

# --- Utility Functions ---

def validate_frame(frame: np.ndarray) -> np.ndarray:
	"""Ensures the frame is valid and in BGR format."""
	if frame is None or frame.size == 0:
		h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
		logger.warning(f"Invalid frame detected, returning default {w}x{h} black frame")
		return np.zeros((h, w, 3), dtype=np.uint8)
	if frame.ndim != 3 or frame.shape[2] != 3:
		logger.warning("Invalid frame format, attempting to convert to BGR")
		if frame.ndim == 2: # Grayscale
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		elif frame.shape[2] == 4: # BGRA
			frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
		elif frame.shape[2] == 1: # Single channel treated as gray
			 frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		else:
			 logger.error(f"Unsupported frame shape {frame.shape}, returning black frame")
			 return np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
	return frame

def resize_frame(frame: np.ndarray, target_dims: Tuple[int, int]) -> np.ndarray:
	"""Resizes a frame to target dimensions (height, width)."""
	if frame is None or frame.size == 0:
		raise ValueError("Cannot resize empty frame")
	# OpenCV resize takes (width, height)
	return cv2.resize(
		frame,
		(target_dims[1], target_dims[0]),
		interpolation=cv2.INTER_LANCZOS4
	)

# --- Model Loading Functions ---

def load_rvm_model(device: str, model_name: str = 'resnet50', pretrained: bool = True) -> Optional[torch.nn.Module]:
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
			'PeterL1n/RobustVideoMatting',
			model_name,
			pretrained=pretrained,
			# force_reload=False  # Avoid force reload unless necessary
		)
		target_device = "cpu" # Default to CPU
		if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			target_device = "mps"
		elif device == "cuda" and torch.cuda.is_available():
			target_device = "cuda"
		elif device != "cpu":
			 logger.warning(f"Requested device {device} unavailable or unsupported, falling back to CPU")

		model = model.to(device=target_device).eval()
		logger.info(f"RVM model '{model_name}' loaded to {target_device}")
		return model
	except Exception as e:
		logger.error(f"Failed to load RVM model '{model_name}': {e}", exc_info=True)
		return None

def load_sam_mask_generator(
	device: str,
	model_type: str = 'vit_h',
	checkpoint_path: Optional[str] = None
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

		target_device = "cpu" # Default to CPU
		if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			target_device = "mps"
		elif device == "cuda" and torch.cuda.is_available():
			target_device = "cuda"
		elif device != "cpu":
			 logger.warning(f"Requested device {device} unavailable or unsupported, falling back to CPU for SAM v1")

		sam.to(device=target_device)
		logger.info(f"SAM v1 model moved to {target_device}")

		generator = SamAutomaticMaskGenerator(
			model=sam,
			points_per_side=32,
			pred_iou_thresh=0.88,
			stability_score_thresh=0.95,
			crop_n_layers=1,
			crop_n_points_downscale_factor=2,
			min_mask_region_area=100 # Example: Filter small masks
		)
		logger.info(f"SAM v1 generator initialized on {target_device}")
		return generator
	except Exception as e:
		logger.error(f"Failed to load SAM v1 model: {e}", exc_info=True)
		return None

def load_sam2_video_predictor(
	device: str,
	checkpoint_path: Optional[str] = None
) -> Optional[Any]:
	"""Loads the Segment Anything (SAM v2 / ultralytics SAM) Predictor."""
	if not _sam2_available:
		logger.warning("ultralytics library unavailable, cannot load SAM 2")
		return None
	if not checkpoint_path or not os.path.exists(checkpoint_path):
		logger.warning(f"SAM 2 checkpoint invalid or missing: {checkpoint_path}")
		return None
	try:
		logger.info(f"Loading SAM 2 model from {checkpoint_path}")
		model = SAM(checkpoint_path)

		target_device = "cpu" # Default to CPU
		if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			target_device = "mps"
		elif device == "cuda" and torch.cuda.is_available():
			target_device = "cuda"
		elif device != "cpu":
			 logger.warning(f"Requested device {device} unavailable or unsupported, falling back to CPU for SAM 2")

		# Ultralytics models typically handle device setting internally or via predict args
		# model.to(device=target_device) # Might not be needed or correct API
		logger.info(f"SAM 2 model initialized (will use device: {target_device} during prediction)")
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
		requires_sam: bool = False,
		requires_sam2: bool = False,
		requires_colour: bool = False
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
		if self.requires_mediapipe and not _mediapipe_available:
			logger.warning(f"Mediapipe required but unavailable for {self.name}")
			return False
		if self.requires_torch and not torch.__version__:
			logger.warning(f"Torch required but unavailable for {self.name}")
			return False
		if self.requires_sam and not _sam_available:
			logger.warning(f"segment_anything required but unavailable for {self.name}")
			return False
		if self.requires_sam2 and not _sam2_available:
			 logger.warning(f"ultralytics required but unavailable for {self.name}")
			 return False
		if self.requires_colour and not _colour_available:
			 logger.warning(f"colour-science required but unavailable for {self.name}")
			 return False
		return True

	def __call__(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Applies the effect if enabled and dependencies met."""
		if not self.enabled:
			return frame
		if not self.check_dependencies():
			# Optionally add a visual indicator that the effect is unavailable
			cv2.putText(frame, f"{self.name.upper()} UNAVAILABLE", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			return frame
		try:
			return self.func(frame=frame, **kwargs)
		except Exception as e:
			logger.error(f"Error during effect '{self.name}': {e}", exc_info=True)
			# Optionally add visual error indicator
			cv2.putText(frame, f"{self.name.upper()} ERROR", (10, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			return frame # Return original frame on error

	def reset(self):
		"""Resets the effect state (e.g., enables it)."""
		self.enabled = True
		# If the wrapped function is an object with a reset method, call it
		if hasattr(self.func, 'reset') and callable(self.func.reset):
			self.func.reset()

class LUTColorGradeEffect:
	"""Applies Look-Up Table (LUT) color grading."""
	def __init__(self, lut_files: List[Tuple[str, str]]):
		self.lut_files = lut_files
		self.current_idx = 0
		self.luts: List[Tuple[str, Any]] = [] # Store loaded LUT objects
		self.lut_intensity = 1.0
		self._load_luts()

	def _load_luts(self):
		"""Loads LUT files using the colour-science library."""
		if not _colour_available:
			logger.warning("colour-science library unavailable, LUTs disabled")
			return
		self.luts = [] # Clear previous LUTs
		for lut_name, lut_path in self.lut_files:
			try:
				lut = colour.io.read_LUT(lut_path)
				# Check if it's a usable LUT type (e.g., LUT3D)
				if isinstance(lut, (colour.LUT3D, colour.LUT1D)): # Add other types if needed
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
		self.current_idx = 0 # Reset index after loading

	def process(self, frame: np.ndarray, runner: Any = None) -> np.ndarray:
		"""Applies the currently selected LUT to the frame."""
		start_time = time.time()
		frame = validate_frame(frame) # Ensure valid input

		if not self.luts or not _colour_available:
			cv2.putText(frame, "LUT Unavailable", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			return frame

		try:
			if not (0 <= self.current_idx < len(self.luts)):
				 logger.warning(f"Invalid LUT index {self.current_idx}, resetting to 0.")
				 self.current_idx = 0
				 if not self.luts: return frame # Still no LUTs

			lut_name, lut = self.luts[self.current_idx]

			# Convert frame to RGB float [0, 1] for colour-science
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

			# Apply the LUT
			frame_transformed = lut.apply(frame_rgb)

			# Convert back to BGR uint8 [0, 255]
			frame_out = (frame_transformed * 255.0).clip(0, 255).astype(np.uint8)
			frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)

			# Apply intensity blending
			intensity = runner.lut_intensity if runner else self.lut_intensity
			if intensity < 1.0:
				frame_out = cv2.addWeighted(frame, 1.0 - intensity, frame_out, intensity, 0.0)

			# Add overlay text
			cv2.putText(frame_out, f"LUT: {lut_name[:25]}{'...' if len(lut_name)>25 else ''}", (10, 70),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			logger.debug(f"Applied LUT: {lut_name} with intensity {intensity:.2f}")

		except Exception as e:
			logger.warning(f"LUT application failed for '{self.luts[self.current_idx][0]}': {e}", exc_info=True)
			cv2.putText(frame, "LUT Error", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			frame_out = frame # Return original on error

		# Log processing time if needed
		# logger.debug(f"LUT processing time: {time.time() - start_time:.4f}s")
		return frame_out

	def reset(self):
		"""Resets LUT index and intensity."""
		self.current_idx = 0
		self.lut_intensity = 1.0
		logger.info("LUT effect reset.")

	def cycle_lut(self):
		"""Cycles to the next available LUT."""
		if not self.luts:
			return
		self.current_idx = (self.current_idx + 1) % len(self.luts)
		logger.info(f"Switched LUT to index {self.current_idx}: {self.luts[self.current_idx][0]}")


# --- Main Effect Runner Class ---

class PixelSensingEffectRunner:
	"""Manages video capture, effects processing, and display."""
	def __init__(self, config_path: Optional[str] = None):
		# --- Initialization ---
		self.config = self._load_config(config_path)
		self._apply_config() # Set attributes from config

		# --- Mediapipe Setup ---
		self.face_mesh = None
		self.pose = None
		self.hands = None
		if _mediapipe_available:
			self._initialize_mediapipe()

		# --- Video I/O ---
		self.cap = None
		self.out = None
		self.frame_width = DEFAULT_WIDTH
		self.frame_height = DEFAULT_HEIGHT
		self.fps = DEFAULT_FPS

		# --- State Variables ---
		self.prev_gray = None
		self.prev_landmarks_flow = None # For optical flow tracking
		self.hue_offset = 0 # For hue rotation effect
		self.frame_buffer = deque(maxlen=10) # Small buffer for motion blur etc.
		self.frame_times = deque(maxlen=100) # For FPS calculation
		self.effect_times = {} # For performance monitoring
		self.error_count = 0
		self.current_effect = "none"
		self.brightness_factor = 0.0 # Range -1.0 to 1.0
		self.contrast_factor = 1.0 # Range 0.0 to 2.0+
		self.window_name = "PixelSense FX"

		# --- Effect Specific States ---
		# Goldenaura
		self.goldenaura_variant = 0
		self.goldenaura_variant_names = ["Original", "Enhanced"]
		self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
		self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
		self.face_history = deque(maxlen=self.trail_length) # Use config value
		self.pose_history = deque(maxlen=self.trail_length) # Use config value
		# Lightning
		self.lightning_style_index = 0
		self.lightning_style_names = ["Random", "Movement", "Branching"]
		self.lightning_styles = [ # Store functions directly
			self._apply_lightning_style_random,
			self._apply_lightning_style_movement,
			self._apply_lightning_style_branching
		]
		# Gestures
		self.gesture_state = {"last_gesture": None, "last_time": 0.0, "gesture_count": 0}
		# LUT
		self.lut_color_grade_effect = None # Initialize later
		self.lut_intensity = 1.0 # Controlled via trackbar
		# RVM
		self.rvm_rec = [None] * 4 # RVM recurrent state
		self.rvm_display_mode = 0 # 0: Composite, 1: Alpha, 2: Foreground
		self.current_rvm_model_idx = 0
		self.rvm_model_names = ["mobilenetv3", "resnet50"] # Available RVM models
		self.rvm_models = {} # Loaded RVM models
		self.rvm_background = None # Loaded background image
		self.rvm_available = torch.__version__ is not None
		self.rvm_downsample_ratio = 0.25 # RVM performance tuning
		self.RVM_DISPLAY_MODES = ["Composite", "Alpha", "Foreground"]
		# SAM
		self.sam_available = _sam_available
		self.sam2_available = _sam2_available
		self.use_sam2_runtime = False # Toggle between SAM v1 and SAM v2
		self.sam_model_v1 = None # SAM v1 mask generator
		self.sam_model_v2 = None # SAM v2 predictor
		self.sam_masks_cache = None # Cache masks for display

		# --- Optical Flow Parameters ---
		self.LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
		self.LK_PARAMS_GOLD = dict(winSize=(15, 15), maxLevel=2, criteria=(
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

		# --- Load LUTs ---
		self._load_luts()

		# --- Initialize Effects ---
		self._initialize_effects() # Define the self.effects dictionary

	def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
		"""Loads configuration from a JSON file or uses defaults."""
		default_config = {
			"input_source": "0",
			"output_path": "output.mp4",
			"lut_dir": "luts",
			"rvm_background_path": None,
			"sam_checkpoint_path": None, # SAM v1 checkpoint
			"sam2_checkpoint_path": None, # SAM v2 checkpoint
			"mode": "live", # "live" or "record"
			"display": True,
			"device": "cpu", # "cpu", "cuda", "mps"
			"trail_length": 30,
			"glow_intensity": 0.5,
			"width": DEFAULT_WIDTH,
			"height": DEFAULT_HEIGHT,
			"fps": DEFAULT_FPS
		}
		config = default_config.copy()
		actual_config_path = config_path or CONFIG_FILE
		if os.path.exists(actual_config_path):
			try:
				with open(actual_config_path, 'r') as f:
					loaded_config = json.load(f)
					config.update(loaded_config) # Update defaults with loaded values
				logger.info(f"Loaded config from {actual_config_path}")
			except json.JSONDecodeError as e:
				logger.warning(f"Failed to decode config file {actual_config_path}: {e}. Using defaults.")
			except Exception as e:
				logger.warning(f"Failed to load config from {actual_config_path}: {e}. Using defaults.")
		else:
			logger.info(f"Config file {actual_config_path} not found. Using default settings.")
		return config

	def _apply_config(self):
		"""Applies loaded configuration settings to the runner's attributes."""
		self.input_source = self.config["input_source"]
		try: # Convert camera index to int if possible
			self.input_source = int(self.input_source)
		except (ValueError, TypeError):
			pass # Keep as string if it's a file path
		self.output_path = self.config["output_path"]
		self.lut_dir = self.config["lut_dir"]
		self._init_rvm_bg_path = self.config["rvm_background_path"] # Path to load later
		self.sam_checkpoint_path = self.config["sam_checkpoint_path"]
		self.sam2_checkpoint_path = self.config["sam2_checkpoint_path"]
		self.mode = self.config["mode"]
		self.display = self.config["display"]
		self.device = self.config["device"]
		# Validate device choice
		if self.device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
			logger.warning("MPS device requested but not available or supported. Falling back to CPU.")
			self.device = "cpu"
		elif self.device == "cuda" and not torch.cuda.is_available():
			logger.warning("CUDA device requested but not available. Falling back to CPU.")
			self.device = "cpu"
		logger.info(f"Selected device: {self.device}")

		self.trail_length = max(1, int(self.config["trail_length"]))
		self.glow_intensity = max(0.0, min(1.0, float(self.config["glow_intensity"])))

		# Update defaults if specified in config
		DEFAULT_WIDTH = int(self.config.get("width", DEFAULT_WIDTH))
		DEFAULT_HEIGHT = int(self.config.get("height", DEFAULT_HEIGHT))
		DEFAULT_FPS = float(self.config.get("fps", DEFAULT_FPS))

		# Re-initialize deques with new trail length
		self.face_history = deque(maxlen=self.trail_length)
		self.pose_history = deque(maxlen=self.trail_length)

	def _initialize_mediapipe(self):
		"""Initializes MediaPipe components if available."""
		logger.info("Initializing Mediapipe modules...")
		try:
			# Use new Task API if available, otherwise fallback
			# For simplicity, using the legacy solutions API here
			self.face_mesh = mp.solutions.face_mesh.FaceMesh(
				max_num_faces=2,
				refine_landmarks=True, # Get more landmarks like iris
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5
			)
			self.pose = mp.solutions.pose.Pose(
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5,
				model_complexity=1, # 0, 1, 2 - balance speed/accuracy
				enable_segmentation=True # Needed for glow effects
			)
			self.hands = mp.solutions.hands.Hands(
				max_num_hands=2,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5,
				model_complexity=1 # 0 or 1
			)
			logger.info("Mediapipe modules initialized successfully.")
		except Exception as e:
			logger.warning(f"Mediapipe initialization failed: {e}", exc_info=True)
			_mediapipe_available = False # Mark as unavailable
			self.face_mesh = self.pose = self.hands = None

	def _load_luts(self):
		"""Loads LUT files and initializes the LUT effect object."""
		self.luts = []
		if os.path.isdir(self.lut_dir):
			for root, _, files in os.walk(self.lut_dir):
				for file in sorted(files):
					if file.lower().endswith('.cube'):
						lut_path = os.path.join(root, file)
						lut_name = os.path.splitext(file)[0]
						self.luts.append((lut_name, lut_path))
			logger.info(f"Found {len(self.luts)} LUT files in '{self.lut_dir}' and subdirectories.")
		else:
			 logger.warning(f"LUT directory '{self.lut_dir}' not found.")

		if self.luts and _colour_available:
			self.lut_color_grade_effect = LUTColorGradeEffect(self.luts)
		else:
			self.lut_color_grade_effect = None
			if not _colour_available:
				 logger.warning("colour-science library not installed. LUT effect disabled.")
			elif not self.luts:
				 logger.warning("No LUT files found. LUT effect disabled.")

	def _initialize_effects(self):
		"""Defines the dictionary of available effects using FrameEffectWrapper."""
		self.effects: Dict[str, FrameEffectWrapper] = {
			"none": FrameEffectWrapper(lambda frame, **kw: frame, "none"),
			"led_base": FrameEffectWrapper(self._apply_led_base_style, "led_base", requires_mediapipe=True),
			"led_enhanced": FrameEffectWrapper(self._apply_led_enhanced_style, "led_enhanced", requires_mediapipe=True),
			"led_hue_rotate": FrameEffectWrapper(self._apply_led_hue_rotate_style, "led_hue_rotate", requires_mediapipe=True),
			"goldenaura": FrameEffectWrapper(self._apply_goldenaura_style, "goldenaura", requires_mediapipe=True),
			"motion_blur": FrameEffectWrapper(self._apply_motion_blur_style, "motion_blur"),
			"advanced_color_grade": FrameEffectWrapper(self._apply_advanced_color_grade_style, "advanced_color_grade"),
			"chromatic_aberration": FrameEffectWrapper(self._apply_chromatic_aberration_style, "chromatic_aberration"),
			"lightning": FrameEffectWrapper(self._apply_lightning_cycle, "lightning", requires_mediapipe=True), # Depends on movement style
			"lut_color_grade": FrameEffectWrapper(
				# Pass the process method of the instance if it exists
				self.lut_color_grade_effect.process if self.lut_color_grade_effect else lambda frame, **kw: frame,
				"lut_color_grade",
				requires_colour=True # Dependency check
			),
			"rvm_composite": FrameEffectWrapper(self._apply_rvm_composite_style, "rvm_composite", requires_torch=True),
			"sam_segmentation": FrameEffectWrapper(
				self._apply_sam_segmentation_style,
				"sam_segmentation",
				requires_torch=True, # Both SAM versions need torch
				requires_sam=True, # Require base SAM lib
				# No specific require for sam2 here, handled internally
			),
			"neon_glow": FrameEffectWrapper(self._apply_neon_glow_style, "neon_glow", requires_mediapipe=True),
			"particle_trail": FrameEffectWrapper(self._apply_particle_trail_style, "particle_trail", requires_mediapipe=True)
		}
		logger.info(f"Initialized {len(self.effects)} effects: {list(self.effects.keys())}")

	# --- Effect Implementation Methods ---
	# (These methods are now defined at the class level)

	def _apply_led_base_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Applies basic LED trail effect based on face landmarks."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			# Dependency check (already done by wrapper, but good practice)
			if not self.face_mesh or not self.frame_width or not self.frame_height:
				logger.debug("LED base: Missing dependencies or dimensions")
				return frame

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# Process with Mediapipe (on RGB image)
			results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

			landmarks = []
			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					for lm in face_landmarks.landmark:
						# Denormalize coordinates
						x = int(lm.x * self.frame_width)
						y = int(lm.y * self.frame_height)
						landmarks.append((x, y))

			if not landmarks:
				logger.debug("LED base: No faces detected")
				self.prev_gray = gray # Still update prev_gray
				self.prev_landmarks_flow = None # Reset flow points
				return frame

			# Optical Flow calculation
			if self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks):
				curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
				prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)

				# Ensure points are within valid range before calcOpticalFlowPyrLK
				valid_indices = np.where((prev_pts[:,0,0] >= 0) & (prev_pts[:,0,0] < self.frame_width) &
										 (prev_pts[:,0,1] >= 0) & (prev_pts[:,0,1] < self.frame_height))[0]

				if len(valid_indices) > 0:
					prev_pts_valid = prev_pts[valid_indices]
					curr_pts_valid = curr_pts[valid_indices]

					# Calculate flow only for valid points
					flow, status, err = cv2.calcOpticalFlowPyrLK(
						self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS
					)

					# Draw trails for points where flow was calculated successfully
					good_new = flow[status == 1]
					good_old = prev_pts_valid[status == 1]

					if good_new.shape[0] > 0:
						frame = self._draw_flow_trails_simple(
							frame, good_new, good_old,
							line_color=(255, 105, 180), # Pink line
							dot_color=(173, 216, 230), # Light blue dot
							line_thickness=1, dot_radius=2
						)
						# Pass current landmarks for glow calculation consistency
						frame = self._draw_motion_glow_separate(frame, landmarks, good_new, good_old)
				else:
					 logger.debug("LED base: No valid previous points for optical flow.")


			# Update for next frame
			self.prev_landmarks_flow = landmarks
			self.prev_gray = gray

		except Exception as e:
			logger.warning(f"LED base style failed: {e}", exc_info=True)
			self.prev_gray = None # Reset state on error
			self.prev_landmarks_flow = None
			self.error_count += 1
			cv2.putText(frame, "LED Base Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		self.effect_times['led_base'] = self.effect_times.get('led_base', []) + [time.time() - start_time]
		return frame

	def _apply_led_enhanced_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Applies enhanced LED trail effect with distortion."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			if not self.face_mesh or not self.frame_width or not self.frame_height:
				logger.debug("LED enhanced: Missing dependencies or dimensions")
				return frame

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

			landmarks = []
			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					 for lm in face_landmarks.landmark:
						x = int(lm.x * self.frame_width)
						y = int(lm.y * self.frame_height)
						landmarks.append((x, y))

			if not landmarks:
				logger.debug("LED enhanced: No faces detected")
				self.prev_gray = gray
				self.prev_landmarks_flow = None
				return frame # Apply distortion even without face? Maybe not.

			# Apply distortion *before* drawing trails
			frame_distorted = self._apply_distortion(frame)
			gray_distorted = cv2.cvtColor(frame_distorted, cv2.COLOR_BGR2GRAY) # Use distorted gray? Or original? Let's use original gray for flow calc

			if self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks):
				curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
				prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)

				# --- Optical Flow calculation (same validation as in led_base) ---
				valid_indices = np.where((prev_pts[:,0,0] >= 0) & (prev_pts[:,0,0] < self.frame_width) &
										 (prev_pts[:,0,1] >= 0) & (prev_pts[:,0,1] < self.frame_height))[0]

				if len(valid_indices) > 0:
					prev_pts_valid = prev_pts[valid_indices]
					curr_pts_valid = curr_pts[valid_indices] # Corresponding current points

					# Use original gray for flow stability, draw on distorted frame
					flow, status, err = cv2.calcOpticalFlowPyrLK(
						self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS
					)

					good_new = flow[status == 1]
					good_old = prev_pts_valid[status == 1]

					if good_new.shape[0] > 0:
						# Draw on the distorted frame
						frame_distorted = self._draw_flow_trails_simple(
							frame_distorted, good_new, good_old,
							line_color=(255, 20, 147), # Deep pink
							dot_color=(135, 206, 250), # Light sky blue
							line_thickness=2, dot_radius=3
						)
						# Apply glow on distorted frame
						frame_distorted = self._draw_motion_glow_separate(frame_distorted, landmarks, good_new, good_old)
				else:
					 logger.debug("LED enhanced: No valid previous points for optical flow.")
			else:
				 logger.debug("LED enhanced: No previous frame/landmarks for optical flow.")


			# Update for next frame (using original gray and landmarks)
			self.prev_landmarks_flow = landmarks
			self.prev_gray = gray
			frame = frame_distorted # Return the modified frame

		except Exception as e:
			logger.warning(f"LED enhanced style failed: {e}", exc_info=True)
			self.prev_gray = None
			self.prev_landmarks_flow = None
			self.error_count += 1
			cv2.putText(frame, "LED Enhanced Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# Return original frame on error? Or distorted? Let's return original.
			frame = validate_frame(frame) # Ensure it's valid if we skip distortion

		self.effect_times['led_enhanced'] = self.effect_times.get('led_enhanced', []) + [time.time() - start_time]
		return frame

	def _apply_led_hue_rotate_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Applies LED trails with a rotating hue shift."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			# Apply hue rotation first
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			# Ensure hue is handled correctly with uint8 wrap-around
			hue = hsv[:, :, 0].astype(np.int16) # Use signed int for addition
			hue = (hue + self.hue_offset) % 180 # Modulo 180 for OpenCV hue
			hsv[:, :, 0] = hue.astype(np.uint8) # Convert back to uint8
			self.hue_offset = (self.hue_offset + 2) % 180 # Slower rotation
			frame_hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

			# Now apply LED trails on the hue-shifted frame
			if not self.face_mesh or not self.frame_width or not self.frame_height:
				logger.debug("LED hue rotate: Missing dependencies or dimensions")
				return frame_hue # Return hue-shifted frame even without trails

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Use original gray for flow
			results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

			landmarks = []
			if results.multi_face_landmarks:
				 for face_landmarks in results.multi_face_landmarks:
					 for lm in face_landmarks.landmark:
						x = int(lm.x * self.frame_width)
						y = int(lm.y * self.frame_height)
						landmarks.append((x, y))

			if not landmarks:
				logger.debug("LED hue rotate: No faces detected")
				self.prev_gray = gray
				self.prev_landmarks_flow = None
				return frame_hue # Return hue-shifted frame

			if self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks):
				curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
				prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)

				# --- Optical Flow calculation (same validation as in led_base) ---
				valid_indices = np.where((prev_pts[:,0,0] >= 0) & (prev_pts[:,0,0] < self.frame_width) &
										 (prev_pts[:,0,1] >= 0) & (prev_pts[:,0,1] < self.frame_height))[0]

				if len(valid_indices) > 0:
					prev_pts_valid = prev_pts[valid_indices]
					curr_pts_valid = curr_pts[valid_indices] # Corresponding current points

					# Use original gray for flow, draw on hue-shifted frame
					flow, status, err = cv2.calcOpticalFlowPyrLK(
						self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS
					)

					good_new = flow[status == 1]
					good_old = prev_pts_valid[status == 1]

					if good_new.shape[0] > 0:
						 # Draw on the hue-rotated frame
						frame_hue = self._draw_flow_trails_simple(
							frame_hue, good_new, good_old,
							line_color=(255, 69, 0), # OrangeRed
							dot_color=(255, 215, 0), # Gold
							line_thickness=1, dot_radius=2
						)
						# Apply glow on hue-rotated frame
						frame_hue = self._draw_motion_glow_separate(frame_hue, landmarks, good_new, good_old)
				else:
					 logger.debug("LED hue rotate: No valid previous points for optical flow.")
			else:
				 logger.debug("LED hue rotate: No previous frame/landmarks for optical flow.")

			# Update for next frame
			self.prev_landmarks_flow = landmarks
			self.prev_gray = gray
			frame = frame_hue # Return the modified frame

		except Exception as e:
			logger.warning(f"LED hue rotate style failed: {e}", exc_info=True)
			self.prev_gray = None
			self.prev_landmarks_flow = None
			self.error_count += 1
			cv2.putText(frame, "LED Hue Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# Return original frame on error? Let's return original.
			frame = validate_frame(frame)

		self.effect_times['led_hue_rotate'] = self.effect_times.get('led_hue_rotate', []) + [time.time() - start_time]
		return frame

	def _apply_goldenaura_original(self, frame: np.ndarray, frame_time: float) -> np.ndarray:
		"""Applies the original Golden Aura effect with trails and tint."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			if not self.face_mesh or not self.pose or not self.frame_width or not self.frame_height:
				logger.debug("Goldenaura original: Missing dependencies or dimensions")
				return frame

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results_face = self.face_mesh.process(frame_rgb)
			results_pose = self.pose.process(frame_rgb)

			h, w = frame.shape[:2]
			frame_out = frame.copy() # Work on a copy

			face_landmarks_current = []
			pose_landmarks_current = []

			# Get current landmarks
			if results_face.multi_face_landmarks:
				for face_landmarks in results_face.multi_face_landmarks:
					for lm in face_landmarks.landmark:
						face_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
			if results_pose.pose_landmarks:
				for lm in results_pose.pose_landmarks.landmark:
					# Filter landmarks to visible/relevant ones if needed
					if lm.visibility > 0.3: # Example filter
						 pose_landmarks_current.append((int(lm.x * w), int(lm.y * h)))

			# Optical Flow and Trail History Update
			# Need to handle cases where the number of detected points changes
			if self.prev_gray is not None:
				# Face
				if face_landmarks_current and self.prev_face_pts_gold.shape[0] > 0:
					curr_pts_face = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					# Match points (simple nearest neighbor or more complex matching if needed)
					# For simplicity, assume order is consistent if count is same, else reset
					if curr_pts_face.shape[0] == self.prev_face_pts_gold.shape[0]:
						flow_face, status_face, _ = cv2.calcOpticalFlowPyrLK(
							self.prev_gray, gray, self.prev_face_pts_gold, curr_pts_face, # Use prev and curr hints
							 **self.LK_PARAMS_GOLD
						)
						# Store *successful* flow points
						good_new = flow_face[status_face == 1]
						good_old = self.prev_face_pts_gold[status_face == 1]
						if good_new.shape[0] > 0:
							self.face_history.append((good_new, good_old))
						# Update prev points only with the successfully tracked ones for next frame
						self.prev_face_pts_gold = good_new.reshape(-1, 1, 2) if good_new.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)

					else: # Number of points changed, reset history for face
						self.face_history.clear()
						self.prev_face_pts_gold = curr_pts_face # Start fresh
				elif face_landmarks_current: # First detection or reappearance
					 self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					 self.face_history.clear()
				else: # No face detected now
					 self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
					 self.face_history.clear()

				# Pose (similar logic)
				if pose_landmarks_current and self.prev_pose_pts_gold.shape[0] > 0:
					curr_pts_pose = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					if curr_pts_pose.shape[0] == self.prev_pose_pts_gold.shape[0]:
						flow_pose, status_pose, _ = cv2.calcOpticalFlowPyrLK(
							self.prev_gray, gray, self.prev_pose_pts_gold, curr_pts_pose,
							**self.LK_PARAMS_GOLD
						)
						good_new_pose = flow_pose[status_pose == 1]
						good_old_pose = self.prev_pose_pts_gold[status_pose == 1]
						if good_new_pose.shape[0] > 0:
							self.pose_history.append((good_new_pose, good_old_pose))
						self.prev_pose_pts_gold = good_new_pose.reshape(-1, 1, 2) if good_new_pose.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
					else:
						self.pose_history.clear()
						self.prev_pose_pts_gold = curr_pts_pose
				elif pose_landmarks_current:
					 self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					 self.pose_history.clear()
				else:
					 self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
					 self.pose_history.clear()

			# No need for manual pop, deque handles maxlen

			# Draw Trails (iterate backwards for fading effect)
			overlay = np.zeros_like(frame_out, dtype=np.uint8) # Draw trails on transparent overlay
			num_face_segments = len(self.face_history)
			for idx, (flow, prev_pts) in enumerate(reversed(self.face_history)):
				alpha = self.TRAIL_END_ALPHA + (self.TRAIL_START_ALPHA - self.TRAIL_END_ALPHA) * (idx / max(num_face_segments - 1, 1))
				trail_color = tuple(int(c * alpha) for c in self.GOLD_TINT_COLOR)
				overlay = self._draw_flow_trails_simple(
					overlay, flow, prev_pts,
					line_color=trail_color, dot_color=trail_color,
					line_thickness=1, dot_radius=TRAIL_RADIUS
				)

			num_pose_segments = len(self.pose_history)
			for idx, (flow, prev_pts) in enumerate(reversed(self.pose_history)):
				alpha = self.TRAIL_END_ALPHA + (self.TRAIL_START_ALPHA - self.TRAIL_END_ALPHA) * (idx / max(num_pose_segments - 1, 1))
				trail_color = tuple(int(c * alpha) for c in self.GOLD_TINT_COLOR)
				overlay = self._draw_flow_trails_simple(
					overlay, flow, prev_pts,
					line_color=trail_color, dot_color=trail_color,
					line_thickness=1, dot_radius=TRAIL_RADIUS # Thicker for pose?
				)

			# Blend trails onto the frame copy
			frame_out = cv2.add(frame_out, overlay)


			# Apply Golden Tint
			tint_layer = np.full_like(frame_out, self.GOLD_TINT_COLOR, dtype=np.uint8)
			frame_out = cv2.addWeighted(frame_out, 1.0 - TINT_STRENGTH, tint_layer, TINT_STRENGTH, 0.0)

			# Apply Glow based on Pose Segmentation
			if results_pose.segmentation_mask is not None:
				try:
					# Condition mask: floating point, 0-1 range.
					condition = (results_pose.segmentation_mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
					# Blur the mask
					mask_blurred = cv2.GaussianBlur(condition, MASK_BLUR_KERNEL, 0)
					# Normalize mask to 0-1 float for blending
					mask_alpha = mask_blurred.astype(np.float32) / 255.0 * self.glow_intensity
					# Ensure mask_alpha has 3 channels for broadcasting
					mask_alpha_3c = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)

					# Create glow layer (can be frame tinted gold, or just gold color)
					glow_color_layer = np.full_like(frame_out, self.GOLD_TINT_COLOR, dtype=np.float32)

					# Blend: frame * (1-alpha) + glow * alpha
					frame_float = frame_out.astype(np.float32)
					frame_float = frame_float * (1.0 - mask_alpha_3c) + glow_color_layer * mask_alpha_3c
					frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)

				except Exception as seg_e:
					 logger.warning(f"Goldenaura segmentation/glow failed: {seg_e}")


			# Update previous gray frame
			self.prev_gray = gray

		except Exception as e:
			logger.warning(f"Goldenaura original failed: {e}", exc_info=True)
			self.prev_gray = None # Reset state
			self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
			self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
			self.face_history.clear()
			self.pose_history.clear()
			self.error_count += 1
			cv2.putText(frame, "Goldenaura Orig. Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			frame_out = frame # Return original on error

		self.effect_times['goldenaura'] = self.effect_times.get('goldenaura', []) + [time.time() - start_time]
		return frame_out

	def _apply_goldenaura_enhanced(self, frame: np.ndarray, frame_time: float) -> np.ndarray:
		"""Applies enhanced Golden Aura with thicker trails and dynamic glow."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			# --- Similar setup as original ---
			if not self.face_mesh or not self.pose or not self.frame_width or not self.frame_height:
				logger.debug("Goldenaura enhanced: Missing dependencies or dimensions")
				return frame

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results_face = self.face_mesh.process(frame_rgb)
			results_pose = self.pose.process(frame_rgb)

			h, w = frame.shape[:2]
			frame_out = frame.copy()

			face_landmarks_current = []
			pose_landmarks_current = []
			if results_face.multi_face_landmarks:
				for face_landmarks in results_face.multi_face_landmarks:
					 for lm in face_landmarks.landmark:
						face_landmarks_current.append((int(lm.x * w), int(lm.y * h)))
			if results_pose.pose_landmarks:
				for lm in results_pose.pose_landmarks.landmark:
					 if lm.visibility > 0.3:
						 pose_landmarks_current.append((int(lm.x * w), int(lm.y * h)))

			# --- Optical Flow and Trail History Update (Identical logic to original) ---
			if self.prev_gray is not None:
				# Face
				if face_landmarks_current and self.prev_face_pts_gold.shape[0] > 0:
					curr_pts_face = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					if curr_pts_face.shape[0] == self.prev_face_pts_gold.shape[0]:
						flow_face, status_face, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_face_pts_gold, curr_pts_face, **self.LK_PARAMS_GOLD)
						good_new = flow_face[status_face == 1]
						good_old = self.prev_face_pts_gold[status_face == 1]
						if good_new.shape[0] > 0: self.face_history.append((good_new, good_old))
						self.prev_face_pts_gold = good_new.reshape(-1, 1, 2) if good_new.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
					else:
						self.face_history.clear()
						self.prev_face_pts_gold = curr_pts_face
				elif face_landmarks_current:
					 self.prev_face_pts_gold = np.array(face_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					 self.face_history.clear()
				else:
					 self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
					 self.face_history.clear()
				# Pose
				if pose_landmarks_current and self.prev_pose_pts_gold.shape[0] > 0:
					curr_pts_pose = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					if curr_pts_pose.shape[0] == self.prev_pose_pts_gold.shape[0]:
						flow_pose, status_pose, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pose_pts_gold, curr_pts_pose, **self.LK_PARAMS_GOLD)
						good_new_pose = flow_pose[status_pose == 1]
						good_old_pose = self.prev_pose_pts_gold[status_pose == 1]
						if good_new_pose.shape[0] > 0: self.pose_history.append((good_new_pose, good_old_pose))
						self.prev_pose_pts_gold = good_new_pose.reshape(-1, 1, 2) if good_new_pose.shape[0] > 0 else np.empty((0, 1, 2), dtype=np.float32)
					else:
						self.pose_history.clear()
						self.prev_pose_pts_gold = curr_pts_pose
				elif pose_landmarks_current:
					 self.prev_pose_pts_gold = np.array(pose_landmarks_current, dtype=np.float32).reshape(-1, 1, 2)
					 self.pose_history.clear()
				else:
					 self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
					 self.pose_history.clear()

			# --- Enhanced Drawing ---
			# Dynamic glow factor based on time
			dynamic_glow_factor = 0.6 + 0.4 * math.sin(frame_time * 1.5) # Adjust frequency/amplitude
			# Enhanced color (slightly brighter/more saturated gold maybe?)
			# BGR: (Blue, Green, Red)
			enhanced_tint_color_base = np.array([30, 180, 220]) # Brighter Gold BGR
			enhanced_tint_color = tuple(np.clip(enhanced_tint_color_base * dynamic_glow_factor, 0, 255).astype(int))

			overlay = np.zeros_like(frame_out, dtype=np.uint8)
			num_face_segments = len(self.face_history)
			for idx, (flow, prev_pts) in enumerate(reversed(self.face_history)):
				alpha = self.TRAIL_END_ALPHA + (self.TRAIL_START_ALPHA - self.TRAIL_END_ALPHA) * (idx / max(num_face_segments - 1, 1))
				trail_color = tuple(int(c * alpha) for c in enhanced_tint_color)
				overlay = self._draw_flow_trails_simple(
					overlay, flow, prev_pts,
					line_color=trail_color, dot_color=trail_color,
					line_thickness=2, # Thicker lines
					dot_radius=3 # Larger dots
				)

			num_pose_segments = len(self.pose_history)
			for idx, (flow, prev_pts) in enumerate(reversed(self.pose_history)):
				 alpha = self.TRAIL_END_ALPHA + (self.TRAIL_START_ALPHA - self.TRAIL_END_ALPHA) * (idx / max(num_pose_segments - 1, 1))
				 trail_color = tuple(int(c * alpha) for c in enhanced_tint_color)
				 overlay = self._draw_flow_trails_simple(
					overlay, flow, prev_pts,
					line_color=trail_color, dot_color=trail_color,
					line_thickness=3, # Even thicker for pose?
					dot_radius=4
				 )

			frame_out = cv2.add(frame_out, overlay)

			# Apply Enhanced Tint (slightly stronger)
			tint_strength_enhanced = min(1.0, TINT_STRENGTH * 1.2)
			tint_layer = np.full_like(frame_out, enhanced_tint_color, dtype=np.uint8)
			frame_out = cv2.addWeighted(frame_out, 1.0 - tint_strength_enhanced, tint_layer, tint_strength_enhanced, 0.0)


			# Apply Dynamic Glow
			if results_pose.segmentation_mask is not None:
				try:
					condition = (results_pose.segmentation_mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
					mask_blurred = cv2.GaussianBlur(condition, (31, 31), 0) # Wider blur for enhanced glow
					mask_alpha = mask_blurred.astype(np.float32) / 255.0
					mask_alpha = mask_alpha * self.glow_intensity * dynamic_glow_factor # Modulate by time
					mask_alpha_3c = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)

					glow_color_layer = np.full_like(frame_out, enhanced_tint_color, dtype=np.float32)

					frame_float = frame_out.astype(np.float32)
					frame_float = frame_float * (1.0 - mask_alpha_3c) + glow_color_layer * mask_alpha_3c
					frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
				except Exception as seg_e:
					 logger.warning(f"Goldenaura enhanced segmentation/glow failed: {seg_e}")

			# Update previous gray frame
			self.prev_gray = gray

		except Exception as e:
			logger.warning(f"Goldenaura enhanced failed: {e}", exc_info=True)
			self.prev_gray = None
			self.prev_face_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
			self.prev_pose_pts_gold = np.empty((0, 1, 2), dtype=np.float32)
			self.face_history.clear()
			self.pose_history.clear()
			self.error_count += 1
			cv2.putText(frame, "Goldenaura Enh. Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			frame_out = frame

		self.effect_times['goldenaura'] = self.effect_times.get('goldenaura', []) + [time.time() - start_time]
		return frame_out

	def _apply_goldenaura_style(self, frame: np.ndarray, frame_time: float = 0.0, **kwargs) -> np.ndarray:
		"""Applies the selected Golden Aura variant."""
		if self.goldenaura_variant == 0:
			return self._apply_goldenaura_original(frame, frame_time)
		else:
			return self._apply_goldenaura_enhanced(frame, frame_time)

	def _apply_lightning_style_random(self, frame: np.ndarray) -> np.ndarray:
		"""Draws random lightning bolts."""
		start_time = time.time()
		frame = validate_frame(frame)
		h, w = frame.shape[:2]
		# Draw on an overlay first
		overlay = np.zeros_like(frame, dtype=np.float32)
		lightning_color_float = np.array([255, 220, 200], dtype=np.float32) # Light blue/white BGR

		try:
			num_bolts = random.randint(1, 4)
			for _ in range(num_bolts):
				# Start point (often off-screen or near edge)
				edge = random.choice(['top', 'bottom', 'left', 'right'])
				if edge == 'top':
					x0, y0 = random.randint(0, w - 1), random.randint(-20, 0)
				elif edge == 'bottom':
					x0, y0 = random.randint(0, w - 1), random.randint(h, h + 20)
				elif edge == 'left':
					 x0, y0 = random.randint(-20, 0), random.randint(0, h - 1)
				else: # right
					 x0, y0 = random.randint(w, w + 20), random.randint(0, h - 1)

				# End point (somewhere within the frame, or another edge)
				x1, y1 = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)

				# Draw segmented line
				num_segments = random.randint(5, 15)
				px, py = x0, y0
				for i in range(num_segments):
					segment_len_factor = (num_segments - i) / num_segments # Longer segments first
					target_x = x0 + (x1 - x0) * (i + 1) / num_segments
					target_y = y0 + (y1 - y0) * (i + 1) / num_segments

					# Add random deviation (more deviation for longer segments)
					deviation = random.uniform(20, 60) * segment_len_factor
					angle_dev = random.uniform(-0.5, 0.5) # Radians
					nx = int(target_x + random.uniform(-deviation, deviation))
					ny = int(target_y + random.uniform(-deviation, deviation))

					thickness = random.randint(1, 3)
					cv2.line(overlay, (px, py), (nx, ny), lightning_color_float, thickness, cv2.LINE_AA)
					px, py = nx, ny # Start next segment from end of this one

			# Apply blur/glow to the overlay
			overlay_blurred = cv2.GaussianBlur(overlay, (7, 7), 0) # Smaller kernel for sharper bolts
			overlay_glow = cv2.GaussianBlur(overlay, (21, 21), 0) # Wider kernel for glow

			# Combine overlays and blend with frame (additively?)
			# Use addWeighted for controlled blending, avoid oversaturation
			frame_float = frame.astype(np.float32)
			# Add blurred bolts
			frame_float = cv2.addWeighted(frame_float, 1.0, overlay_blurred, 0.8, 0.0)
			# Add glow
			frame_float = cv2.addWeighted(frame_float, 1.0, overlay_glow, 0.4, 0.0)

			frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)

		except Exception as e:
			logger.warning(f"Random lightning failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame # Return original on error

		# No separate timing for sub-styles, use main effect time
		return frame_out

	def _apply_lightning_style_movement(self, frame: np.ndarray) -> np.ndarray:
		"""Draws lightning based on facial landmark movement."""
		start_time = time.time()
		frame = validate_frame(frame)
		overlay = np.zeros_like(frame, dtype=np.float32) # Use float overlay
		lightning_color_float = np.array([255, 220, 200], dtype=np.float32) # Light blue BGR

		try:
			# --- Face tracking and Optical Flow (similar to LED styles) ---
			if not self.face_mesh or not self.frame_width or not self.frame_height:
				logger.debug("Movement lightning: Missing dependencies")
				return frame

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

			landmarks = []
			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					 for lm in face_landmarks.landmark:
						x, y = int(lm.x * self.frame_width), int(lm.y * self.frame_height)
						landmarks.append((x, y))

			if not landmarks:
				logger.debug("Movement lightning: No faces detected")
				self.prev_gray = gray
				self.prev_landmarks_flow = None
				return frame

			h, w = frame.shape[:2]
			if self.prev_gray is not None and self.prev_landmarks_flow is not None and len(self.prev_landmarks_flow) == len(landmarks):
				curr_pts = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
				prev_pts = np.array(self.prev_landmarks_flow, dtype=np.float32).reshape(-1, 1, 2)

				# --- Optical Flow calculation (same validation) ---
				valid_indices = np.where((prev_pts[:,0,0] >= 0) & (prev_pts[:,0,0] < w) &
										 (prev_pts[:,0,1] >= 0) & (prev_pts[:,0,1] < h))[0]

				if len(valid_indices) > 0:
					prev_pts_valid = prev_pts[valid_indices]
					# Calculate flow
					flow, status, err = cv2.calcOpticalFlowPyrLK(
						self.prev_gray, gray, prev_pts_valid, None, **self.LK_PARAMS
					)
					good_new = flow[status == 1]
					good_old = prev_pts_valid[status == 1]

					# Draw lightning for points with significant movement
					if good_new.shape[0] > 0:
						movement_vectors = good_new - good_old
						magnitudes = np.linalg.norm(movement_vectors, axis=1)
						# Define a threshold for significant movement
						movement_threshold = 5.0 # Pixels per frame

						for i in range(len(good_new)):
							if magnitudes[i] > movement_threshold:
								xn, yn = good_new[i].ravel()
								xo, yo = good_old[i].ravel()
								thickness = random.randint(1, 2)
								# Draw a slightly jittered line for lightning effect
								jitter_x = random.uniform(-3, 3)
								jitter_y = random.uniform(-3, 3)
								cv2.line(overlay, (int(xo), int(yo)), (int(xn + jitter_x), int(yn + jitter_y)),
										 lightning_color_float, thickness, cv2.LINE_AA)
				else:
					 logger.debug("Movement lightning: No valid previous points.")
			else:
				 logger.debug("Movement lightning: No previous frame/landmarks.")

			# Apply blur/glow and blend (similar to random style)
			overlay_blurred = cv2.GaussianBlur(overlay, (5, 5), 0)
			overlay_glow = cv2.GaussianBlur(overlay, (15, 15), 0)

			frame_float = frame.astype(np.float32)
			frame_float = cv2.addWeighted(frame_float, 1.0, overlay_blurred, 0.9, 0.0) # Stronger bolts
			frame_float = cv2.addWeighted(frame_float, 1.0, overlay_glow, 0.5, 0.0) # Moderate glow
			frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)

			# Update state for next frame
			self.prev_landmarks_flow = landmarks
			self.prev_gray = gray

		except Exception as e:
			logger.warning(f"Movement lightning failed: {e}", exc_info=True)
			self.prev_gray = None
			self.prev_landmarks_flow = None
			self.error_count += 1
			frame_out = frame

		return frame_out

	def _apply_lightning_style_branching(self, frame: np.ndarray) -> np.ndarray:
		"""Draws branching lightning bolts."""
		start_time = time.time()
		frame = validate_frame(frame)
		h, w = frame.shape[:2]
		overlay = np.zeros_like(frame, dtype=np.float32)
		lightning_color_float = np.array([255, 230, 210], dtype=np.float32) # Slightly different blue

		try:
			num_bolts = random.randint(1, 3)
			for _ in range(num_bolts):
				# Start/End points (similar to random)
				edge = random.choice(['top', 'bottom', 'left', 'right'])
				if edge == 'top': x0, y0 = random.randint(0, w - 1), random.randint(-20, 0)
				elif edge == 'bottom': x0, y0 = random.randint(0, w - 1), random.randint(h, h + 20)
				elif edge == 'left': x0, y0 = random.randint(-20, 0), random.randint(0, h - 1)
				else: x0, y0 = random.randint(w, w + 20), random.randint(0, h - 1)
				x1, y1 = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)

				# Recursive function to draw branches
				def draw_branch(x_start, y_start, x_end, y_end, thickness, depth):
					if depth <= 0:
						return
					dx, dy = x_end - x_start, y_end - y_start
					length = math.sqrt(dx*dx + dy*dy)
					if length < 10: # Min length for branch segment
						 return

					# Draw main segment
					cv2.line(overlay, (int(x_start), int(y_start)), (int(x_end), int(y_end)),
							 lightning_color_float * (depth / 5.0), # Fades slightly
							 max(1, thickness), cv2.LINE_AA)

					# Chance to branch
					if random.random() < 0.4 and depth > 1: # Branch more likely further down
						branch_point_factor = random.uniform(0.4, 0.8)
						bx = x_start + dx * branch_point_factor
						by = y_start + dy * branch_point_factor

						# Branch direction (angle relative to main segment)
						main_angle = math.atan2(dy, dx)
						branch_angle_offset = random.uniform(0.4, 0.8) * random.choice([-1, 1]) # Radians
						branch_angle = main_angle + branch_angle_offset
						branch_length = length * random.uniform(0.4, 0.7) # Shorter branches

						branch_end_x = bx + branch_length * math.cos(branch_angle)
						branch_end_y = by + branch_length * math.sin(branch_angle)

						# Recurse for the branch
						draw_branch(bx, by, branch_end_x, branch_end_y, max(1, thickness - 1), depth - 1)

					# Continue main branch (slightly deviated)
					deviation = length * 0.1
					next_x = x_end + random.uniform(-deviation, deviation)
					next_y = y_end + random.uniform(-deviation, deviation)
					# Recurse (technically tail recursion, but conceptually simpler)
					# This is not quite right - needs proper segmentation like random style
					# Let's adapt the segmented approach from random style instead

				# --- Segmented Branching ---
				num_segments = random.randint(8, 20)
				px, py = x0, y0
				current_angle = math.atan2(y1 - y0, x1 - x0)
				base_thickness = random.randint(2, 4)

				for i in range(num_segments):
					segment_len = math.sqrt((x1-x0)**2 + (y1-y0)**2) / num_segments * random.uniform(0.8, 1.2)
					# Deviate angle slightly
					current_angle += random.uniform(-0.3, 0.3)
					# Calculate next point
					nx = px + segment_len * math.cos(current_angle)
					ny = py + segment_len * math.sin(current_angle)
					# Calculate thickness (tapering)
					thickness = max(1, int(base_thickness * ((num_segments - i) / num_segments)))

					# Draw segment
					cv2.line(overlay, (int(px), int(py)), (int(nx), int(ny)),
							 lightning_color_float, thickness, cv2.LINE_AA)

					# Chance to branch from the end of this segment
					if random.random() < 0.25 and i < num_segments - 2: # Don't branch from last segments
						branch_angle_offset = random.uniform(0.5, 1.0) * random.choice([-1, 1])
						branch_angle = current_angle + branch_angle_offset
						branch_length = segment_len * random.uniform(1.5, 3.0) # Branches can be longer
						branch_end_x = nx + branch_length * math.cos(branch_angle)
						branch_end_y = ny + branch_length * math.sin(branch_angle)
						branch_thickness = max(1, thickness - 1)
						# Draw simple branch line (could make this recursive too)
						cv2.line(overlay, (int(nx), int(ny)), (int(branch_end_x), int(branch_end_y)),
								 lightning_color_float, branch_thickness, cv2.LINE_AA)

					px, py = nx, ny # Move to next point


			# Apply blur/glow and blend
			overlay_blurred = cv2.GaussianBlur(overlay, (3, 3), 0) # Sharper bolts
			overlay_glow = cv2.GaussianBlur(overlay, (25, 25), 0) # Wider glow

			frame_float = frame.astype(np.float32)
			frame_float = cv2.addWeighted(frame_float, 1.0, overlay_blurred, 0.8, 0.0)
			frame_float = cv2.addWeighted(frame_float, 1.0, overlay_glow, 0.6, 0.0) # Stronger glow
			frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)

		except Exception as e:
			logger.warning(f"Branching lightning failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame

		return frame_out

	def _apply_lightning_cycle(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Cycles through the different lightning styles."""
		start_time = time.time()
		frame = validate_frame(frame)
		if not self.lightning_styles:
			logger.warning("No lightning styles available")
			return frame
		try:
			# Get the current style function
			style_function = self.lightning_styles[self.lightning_style_index]
			frame_out = style_function(frame) # Call the selected style function
			# Add text overlay indicating style
			style_name = self.lightning_style_names[self.lightning_style_index]
			cv2.putText(frame_out, f"Lightning: {style_name}", (10, 90),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		except Exception as e:
			logger.warning(f"Lightning cycle failed for style index {self.lightning_style_index}: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame
			cv2.putText(frame_out, "Lightning Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		self.effect_times['lightning'] = self.effect_times.get('lightning', []) + [time.time() - start_time]
		return frame_out

	def _apply_neon_glow_style(self, frame: np.ndarray, frame_time: float, **kwargs) -> np.ndarray:
		"""Applies a neon glow effect based on pose landmarks and segmentation."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			if not self.pose or not self.frame_width or not self.frame_height:
				logger.debug("Neon glow: Missing dependencies or dimensions")
				return frame

			frame_out = frame.copy()
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results_pose = self.pose.process(frame_rgb)

			h, w = frame.shape[:2]
			overlay = np.zeros_like(frame_out, dtype=np.uint8) # Draw glow on overlay

			if results_pose.pose_landmarks:
				landmarks = []
				connections = mp.solutions.pose.POSE_CONNECTIONS

				# Get landmark coordinates
				for id, lm in enumerate(results_pose.pose_landmarks.landmark):
					if lm.visibility > 0.3: # Filter less visible landmarks
						landmarks.append((int(lm.x * w), int(lm.y * h)))
					else:
						landmarks.append(None) # Placeholder for missing landmarks

				# Dynamic neon color
				# Cycle through BGR components for a rainbow effect
				b = int(127 + 127 * math.sin(frame_time * 1.0))
				g = int(127 + 127 * math.sin(frame_time * 1.0 + 2 * math.pi / 3))
				r = int(127 + 127 * math.sin(frame_time * 1.0 + 4 * math.pi / 3))
				neon_color = (b, g, r)

				# Draw connections (lines)
				line_thickness = 3
				if connections:
					for connection in connections:
						start_idx, end_idx = connection
						if start_idx < len(landmarks) and end_idx < len(landmarks):
							 start_pt = landmarks[start_idx]
							 end_pt = landmarks[end_idx]
							 if start_pt and end_pt: # Both landmarks visible
								 cv2.line(overlay, start_pt, end_pt, neon_color, line_thickness, cv2.LINE_AA)

				# Draw landmarks (circles) - Optional, can make it look cluttered
				# dot_radius = 5
				# for point in landmarks:
				#    if point:
				#         cv2.circle(overlay, point, dot_radius, neon_color, -1, cv2.LINE_AA)

				# Apply blur to the overlay to create the glow effect
				glow_overlay = cv2.GaussianBlur(overlay, (15, 15), 0) # Inner glow
				wider_glow_overlay = cv2.GaussianBlur(overlay, (31, 31), 0) # Outer glow

				# Additive blending for glow
				frame_out = cv2.addWeighted(frame_out, 1.0, glow_overlay, 0.8, 0)
				frame_out = cv2.addWeighted(frame_out, 1.0, wider_glow_overlay, 0.4, 0)


			# Optional: Enhance with segmentation mask glow (similar to Goldenaura)
			if results_pose.segmentation_mask is not None:
				try:
					condition = (results_pose.segmentation_mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
					mask_blurred = cv2.GaussianBlur(condition, (31, 31), 0)
					mask_alpha = mask_blurred.astype(np.float32) / 255.0 * self.glow_intensity * 0.5 # Less intense than lines
					mask_alpha_3c = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)

					# Use the dynamic neon color for the segmentation glow too
					seg_glow_layer = np.full_like(frame_out, neon_color, dtype=np.float32)

					frame_float = frame_out.astype(np.float32)
					frame_float = frame_float * (1.0 - mask_alpha_3c) + seg_glow_layer * mask_alpha_3c
					frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)
				except Exception as seg_e:
					 logger.warning(f"Neon glow segmentation effect failed: {seg_e}")


		except Exception as e:
			logger.warning(f"Neon glow failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame # Return original on error
			cv2.putText(frame_out, "Neon Glow Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		self.effect_times['neon_glow'] = self.effect_times.get('neon_glow', []) + [time.time() - start_time]
		return frame_out

	def _apply_particle_trail_style(self, frame: np.ndarray, frame_time: float, **kwargs) -> np.ndarray:
		"""Applies a particle trail effect following hand landmarks."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			if not self.hands or not self.frame_width or not self.frame_height:
				logger.debug("Particle trail: Missing dependencies or dimensions")
				return frame

			frame_out = frame.copy()
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results_hands = self.hands.process(frame_rgb)

			h, w = frame.shape[:2]

			if results_hands.multi_hand_landmarks:
				for hand_landmarks in results_hands.multi_hand_landmarks:
					# Use a prominent landmark, e.g., index finger tip or wrist
					# landmark_to_track = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
					landmark_to_track = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

					x = int(landmark_to_track.x * w)
					y = int(landmark_to_track.y * h)

					if 0 <= x < w and 0 <= y < h:
						num_particles = 15 # Number of particles per hand per frame
						max_offset = 25 # Max distance from the tracked point
						min_size, max_size = 1, 4 # Particle size range

						for _ in range(num_particles):
							# Calculate offset with falloff (particles closer to center are more likely)
							offset_dist = random.gauss(0, max_offset / 2.5) # Gaussian distribution for distance
							offset_angle = random.uniform(0, 2 * math.pi)
							offset_x = int(x + offset_dist * math.cos(offset_angle))
							offset_y = int(y + offset_dist * math.sin(offset_angle))

							# Ensure particle is within bounds
							if 0 <= offset_x < w and 0 <= offset_y < h:
								# Random color
								particle_color = (
									random.randint(100, 255),
									random.randint(100, 255),
									random.randint(100, 255)
								)
								particle_size = random.randint(min_size, max_size)
								# Use alpha blending for softer particles (optional, slower)
								# overlay = frame_out.copy()
								# cv2.circle(overlay, (offset_x, offset_y), particle_size, particle_color, -1, cv2.LINE_AA)
								# alpha = random.uniform(0.5, 0.9)
								# frame_out = cv2.addWeighted(frame_out, 1.0 - alpha, overlay, alpha, 0)

								# Direct drawing (faster)
								cv2.circle(frame_out, (offset_x, offset_y), particle_size, particle_color, -1, cv2.LINE_AA)

		except Exception as e:
			logger.warning(f"Particle trail failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame # Return original on error
			cv2.putText(frame_out, "Particle Trail Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		self.effect_times['particle_trail'] = self.effect_times.get('particle_trail', []) + [time.time() - start_time]
		return frame_out

	def _apply_motion_blur_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Applies a simple directional motion blur."""
		start_time = time.time()
		frame = validate_frame(frame)
		frame_out = frame # Default to original if buffer is empty

		try:
			# Add current frame to buffer
			self.frame_buffer.append(frame.astype(np.float32)) # Store as float for averaging

			if len(self.frame_buffer) > 1:
				# Average frames in the buffer
				avg_frame = np.mean(np.array(self.frame_buffer), axis=0)
				frame_out = np.clip(avg_frame, 0, 255).astype(np.uint8)

			# Alternative: Kernel-based blur (less realistic for camera motion)
			# kernel_size = 15
			# kernel = np.zeros((kernel_size, kernel_size))
			# kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size) # Horizontal blur
			# kernel = kernel / kernel_size
			# frame_out = cv2.filter2D(frame, -1, kernel)

		except Exception as e:
			logger.warning(f"Motion blur failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame # Return original on error
			cv2.putText(frame_out, "Motion Blur Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		self.effect_times['motion_blur'] = self.effect_times.get('motion_blur', []) + [time.time() - start_time]
		return frame_out

	def _apply_advanced_color_grade_style(self, frame: np.ndarray, frame_time: float, **kwargs) -> np.ndarray:
		"""Applies dynamic contrast, brightness, and vignette."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			frame_float = frame.astype(np.float32) / 255.0

			# 1. Dynamic Contrast and Brightness
			contrast = 1.1 + 0.15 * math.sin(frame_time * 0.8) # Oscillating contrast
			brightness = 0.05 * math.cos(frame_time * 0.8) # Oscillating brightness
			# Apply contrast first, then brightness (order matters)
			# Contrast adjusts range around mid-gray (0.5)
			frame_float = 0.5 + contrast * (frame_float - 0.5)
			# Apply brightness (simple addition)
			frame_float += brightness

			# 2. Vignette Effect
			h, w = frame.shape[:2]
			center_x, center_y = w / 2, h / 2
			# Use diagonal distance for normalization
			max_dist = math.sqrt(center_x**2 + center_y**2)
			# Create vignette mask (precompute if size is constant)
			vignette_strength = 0.6 # How dark the edges get
			vignette_falloff = 2.0 # How quickly the vignette fades (higher is faster)

			rows, cols = np.indices((h, w))
			dist = np.sqrt((cols - center_x)**2 + (rows - center_y)**2) / max_dist
			vignette_mask = 1.0 - vignette_strength * (dist ** vignette_falloff)
			vignette_mask = np.clip(vignette_mask, 0.0, 1.0) # Ensure mask is [0, 1]

			# Apply vignette mask (multiply channel-wise)
			frame_float *= vignette_mask[:, :, np.newaxis] # Add channel dim for broadcasting

			# Clip and convert back to uint8
			frame_out = np.clip(frame_float * 255.0, 0, 255).astype(np.uint8)

		except Exception as e:
			logger.warning(f"Advanced color grade failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame # Return original on error
			cv2.putText(frame_out, "Adv. Grade Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		self.effect_times['advanced_color_grade'] = self.effect_times.get('advanced_color_grade', []) + [time.time() - start_time]
		return frame_out

	def _apply_chromatic_aberration_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Applies a simple chromatic aberration effect by shifting color channels."""
		start_time = time.time()
		frame = validate_frame(frame)
		try:
			b, g, r = cv2.split(frame)
			shift = 3 # Pixel shift amount
			h, w = frame.shape[:2]

			# Create shifted channels (initialize with zeros or edge pixels)
			b_shifted = np.zeros_like(b)
			g_shifted = g.copy() # Keep green centered (usually dominant)
			r_shifted = np.zeros_like(r)

			# Shift Blue channel (e.g., left)
			b_shifted[:, shift:] = b[:, :-shift]
			# Fill the gap (e.g., with edge pixels)
			b_shifted[:, :shift] = b[:, :shift]

			# Shift Red channel (e.g., right)
			r_shifted[:, :-shift] = r[:, shift:]
			# Fill the gap
			r_shifted[:, -shift:] = r[:, -shift:]

			# Merge the shifted channels
			frame_out = cv2.merge((b_shifted, g_shifted, r_shifted))

		except Exception as e:
			logger.warning(f"Chromatic aberration failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame # Return original on error
			cv2.putText(frame_out, "Aberration Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		self.effect_times['chromatic_aberration'] = self.effect_times.get('chromatic_aberration', []) + [time.time() - start_time]
		return frame_out

	def _apply_rvm_composite_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Performs background replacement using Robust Video Matting."""
		start_time = time.time()
		frame = validate_frame(frame)
		frame_out = frame.copy() # Work on a copy

		# Check availability (already done by wrapper)
		model_name = self.rvm_model_names[self.current_rvm_model_idx % len(self.rvm_model_names)]
		model = self.rvm_models.get(model_name)

		if not self.rvm_available or not model:
			cv2.putText(frame_out, f"RVM ({model_name}) Unavailable", (10, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			self.effect_times['rvm_composite'] = self.effect_times.get('rvm_composite', []) + [time.time() - start_time]
			return frame_out

		try:
			# 1. Preprocess Frame
			src = self._preprocess_frame_rvm(frame)
			if src is None:
				cv2.putText(frame_out, "RVM Preprocess Failed", (10, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				self.effect_times['rvm_composite'] = self.effect_times.get('rvm_composite', []) + [time.time() - start_time]
				return frame_out

			# 2. Inference
			with torch.no_grad():
				# Ensure recurrent state is on the correct device
				rec = [r.to(self.device) if r is not None else None for r in self.rvm_rec]
				# Run model
				fgr, pha, *rec = model(src, *rec, downsample_ratio=self.rvm_downsample_ratio)
				# Store recurrent state back (potentially move back to CPU if needed elsewhere)
				self.rvm_rec = [r.cpu() if r is not None else None for r in rec] # Example: move back

			# 3. Postprocess Output
			fgr_np, pha_np = self._postprocess_output_rvm(fgr, pha)
			if fgr_np is None or pha_np is None:
				cv2.putText(frame_out, "RVM Postprocess Failed", (10, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				self.effect_times['rvm_composite'] = self.effect_times.get('rvm_composite', []) + [time.time() - start_time]
				return frame_out

			 # Ensure fgr/pha match frame dimensions if downsample_ratio was used
			if fgr_np.shape[:2] != frame.shape[:2]:
				 fgr_np = resize_frame(fgr_np, (frame.shape[0], frame.shape[1]))
				 pha_np = cv2.resize(pha_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)


			# 4. Composite based on display mode
			mode = self.RVM_DISPLAY_MODES[self.rvm_display_mode]
			if mode == "Alpha":
				# Convert single channel alpha to 3-channel grayscale for display
				frame_out = cv2.cvtColor(pha_np, cv2.COLOR_GRAY2BGR)
			elif mode == "Foreground":
				# Create black background
				black_bg = np.zeros_like(frame_out)
				# Blend foreground onto black using alpha
				alpha_f = pha_np.astype(np.float32) / 255.0
				alpha_3c = cv2.cvtColor(alpha_f, cv2.COLOR_GRAY2BGR)
				frame_out = (alpha_3c * fgr_np + (1.0 - alpha_3c) * black_bg).astype(np.uint8)
			else: # Composite mode
				# Load or create background if needed
				if self.rvm_background is None:
					self._load_rvm_background(self._init_rvm_bg_path)
				# Ensure background matches frame size
				bg = self.rvm_background
				if bg.shape[:2] != frame.shape[:2]:
					bg = resize_frame(bg, (frame.shape[0], frame.shape[1]))
					self.rvm_background = bg # Cache resized version

				# Composite: alpha * foreground + (1 - alpha) * background
				alpha_f = pha_np.astype(np.float32) / 255.0
				alpha_3c = cv2.cvtColor(alpha_f, cv2.COLOR_GRAY2BGR)
				frame_out = (alpha_3c * fgr_np + (1.0 - alpha_3c) * bg).astype(np.uint8)

			# Add overlay text
			cv2.putText(frame_out, f"RVM: {model_name} ({mode})", (10, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			logger.debug(f"RVM composite applied: {mode}")

		except Exception as e:
			logger.warning(f"RVM composite failed: {e}", exc_info=True)
			self.error_count += 1
			cv2.putText(frame_out, "RVM Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# frame_out remains the original frame copy

		self.effect_times['rvm_composite'] = self.effect_times.get('rvm_composite', []) + [time.time() - start_time]
		return frame_out

	def _draw_sam_masks(self, frame: np.ndarray, masks: Any) -> np.ndarray:
		"""Draws segmentation masks from SAM with random colors."""
		start_time = time.time()
		frame = validate_frame(frame)
		overlay = frame.copy() # Draw on a copy

		if masks is None:
			logger.debug("No SAM masks to draw")
			return frame

		try:
			h, w = frame.shape[:2]
			num_masks_drawn = 0
			masks_to_process = []

			# --- Standardize Mask Input ---
			if isinstance(masks, list) and len(masks) > 0:
				# SAM v1 format: list of dicts {'segmentation': np.ndarray, 'area': int, ...}
				if isinstance(masks[0], dict) and 'segmentation' in masks[0]:
					# Sort by area (optional, draw largest first)
					masks_to_process = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
					masks_to_process = [m['segmentation'] for m in masks_to_process] # Extract boolean arrays
					logger.debug(f"Processing {len(masks_to_process)} SAM v1 masks")
				# Potentially other list formats?
				else:
					 logger.warning(f"Unsupported list content type for SAM masks: {type(masks[0])}")
					 return frame
			elif _sam2_available and isinstance(masks, SAM.Results) and masks.masks is not None:
				 # SAM v2 (ultralytics) format: Results object with masks attribute (often tensor)
				 sam2_masks = masks.masks.data # Get the tensor data (e.g., [N, H, W])
				 if isinstance(sam2_masks, torch.Tensor):
					  # Convert tensor masks to numpy boolean arrays
					  masks_to_process = [m.cpu().numpy().astype(bool) for m in sam2_masks]
					  logger.debug(f"Processing {len(masks_to_process)} SAM v2 tensor masks")
				 else:
					  logger.warning(f"Unexpected SAM v2 mask data type: {type(sam2_masks)}")
					  return frame
			elif isinstance(masks, torch.Tensor): # Handle raw tensor input
				 if masks.ndim == 3: # Assume [N, H, W]
					  masks_to_process = [m.cpu().numpy().astype(bool) for m in masks]
					  logger.debug(f"Processing {len(masks_to_process)} raw tensor masks")
				 elif masks.ndim == 2: # Assume single mask [H, W]
					  masks_to_process = [masks.cpu().numpy().astype(bool)]
					  logger.debug("Processing 1 raw tensor mask")
				 else:
					  logger.warning(f"Unsupported raw tensor shape for SAM masks: {masks.shape}")
					  return frame
			elif isinstance(masks, np.ndarray): # Handle raw numpy array
				 if masks.ndim == 3: # Assume [N, H, W]
					 masks_to_process = [m.astype(bool) for m in masks]
					 logger.debug(f"Processing {len(masks_to_process)} raw numpy masks")
				 elif masks.ndim == 2: # Assume single mask [H, W]
					 masks_to_process = [masks.astype(bool)]
					 logger.debug("Processing 1 raw numpy mask")
				 else:
					 logger.warning(f"Unsupported raw numpy shape for SAM masks: {masks.shape}")
					 return frame
			else:
				logger.warning(f"Unsupported SAM mask input type: {type(masks)}")
				return frame

			if not masks_to_process:
				logger.debug("No valid SAM masks found after processing input.")
				return frame

			# --- Draw Masks ---
			for i, mask_data in enumerate(masks_to_process):
				if mask_data is None or mask_data.size == 0:
					logger.debug(f"Skipping empty mask at index {i}")
					continue

				try:
					# Ensure mask is boolean
					mask_bool = mask_data.astype(bool)

					# Ensure mask dimensions match frame
					if mask_bool.shape != (h, w):
						logger.debug(f"Resizing mask {i} from {mask_bool.shape} to {(h, w)}")
						mask_bool = cv2.resize(
							mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
						).astype(bool)

					if not np.any(mask_bool): # Skip empty masks after resize
						 logger.debug(f"Skipping empty mask {i} after resize")
						 continue

					# Generate random color
					color = [random.randint(64, 200) for _ in range(3)] # BGR

					# Apply color to the mask area on the overlay
					overlay[mask_bool] = color
					num_masks_drawn += 1

				except Exception as draw_e:
					logger.warning(f"Failed to process/draw SAM mask {i}: {draw_e}")
					continue # Skip problematic mask

			# Blend the colored overlay with the original frame copy
			if num_masks_drawn > 0:
				 frame_out = cv2.addWeighted(frame, 1.0 - self.SAM_MASK_ALPHA, overlay, self.SAM_MASK_ALPHA, 0)
				 logger.debug(f"Drew {num_masks_drawn} SAM masks")
			else:
				 logger.debug("No SAM masks were drawn.")
				 frame_out = frame # Return original if nothing drawn


		except Exception as e:
			logger.warning(f"SAM mask drawing failed: {e}", exc_info=True)
			self.error_count += 1
			frame_out = frame # Return original on error
			cv2.putText(frame_out, "SAM Draw Error", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Store time taken for drawing itself if needed
		# self.effect_times['sam_draw'] = self.effect_times.get('sam_draw', []) + [time.time() - start_time]
		return frame_out

	def _apply_sam_segmentation_style(self, frame: np.ndarray, **kwargs) -> np.ndarray:
		"""Generates and draws segmentation masks using SAM."""
		start_time = time.time()
		frame = validate_frame(frame)
		frame_out = frame.copy()

		# Determine which SAM model to use
		sam_model_to_use = None
		sam_version_name = "N/A"
		if self.use_sam2_runtime and self.sam_model_v2 and self.sam2_available:
			sam_model_to_use = self.sam_model_v2
			sam_version_name = "SAM v2 (Ultralytics)"
		elif self.sam_model_v1 and self.sam_available:
			sam_model_to_use = self.sam_model_v1
			sam_version_name = "SAM v1 (AutoMask)"
		else:
			# Neither model is available or selected correctly
			 cv2.putText(frame_out, "SAM Unavailable", (10, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			 logger.warning("Neither SAM v1 nor SAM v2 model available for segmentation.")
			 self.effect_times['sam_segmentation'] = self.effect_times.get('sam_segmentation', []) + [time.time() - start_time]
			 return frame_out


		logger.debug(f"Using {sam_version_name}")
		try:
			# Convert frame to RGB for SAM models
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			masks = None
			inference_start_time = time.time()

			# Run inference with the chosen model
			if self.use_sam2_runtime and self.sam_model_v2:
				# SAM v2 (Ultralytics) inference
				with torch.no_grad():
					# Specify device if needed by the predict call
					results = sam_model_to_use.predict(rgb_frame, device=self.device, verbose=False) # Add verbose=False
				masks = results # Pass the whole Results object to _draw_sam_masks
				logger.debug(f"SAM v2 Inference time: {time.time() - inference_start_time:.4f}s")

			elif not self.use_sam2_runtime and self.sam_model_v1:
				# SAM v1 (Automatic Mask Generator) inference
				# Needs to run on the device the model was loaded onto
				# No explicit torch.no_grad() needed if model is in eval mode
				masks = sam_model_to_use.generate(rgb_frame) # Returns list of dicts
				logger.debug(f"SAM v1 Inference time: {time.time() - inference_start_time:.4f}s")

			# Draw the generated masks
			if masks is not None:
				 frame_out = self._draw_sam_masks(frame, masks) # Pass original frame for blending base
				 cv2.putText(frame_out, f"{sam_version_name} Seg.", (10, 30),
							 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			else:
				 cv2.putText(frame_out, f"{sam_version_name} No Masks", (10, 50),
							 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange warning

		except Exception as e:
			logger.warning(f"SAM segmentation failed ({sam_version_name}): {e}", exc_info=True)
			self.error_count += 1
			cv2.putText(frame_out, "SAM Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# frame_out remains the original frame copy

		self.effect_times['sam_segmentation'] = self.effect_times.get('sam_segmentation', []) + [time.time() - start_time]
		return frame_out

	# --- Helper Methods ---

	def _draw_flow_trails_simple(self,
								 frame: np.ndarray,
								 flow_points: np.ndarray, # Points at current frame (xn, yn)
								 prev_points: np.ndarray, # Points at previous frame (xo, yo)
								 line_color: Tuple[int, int, int],
								 dot_color: Tuple[int, int, int],
								 line_thickness: int = 1,
								 dot_radius: int = 1) -> np.ndarray:
		"""Draws simple lines and dots for optical flow trails."""
		# Assumes flow_points and prev_points correspond row-wise
		# Frame can be the original or an overlay
		try:
			if flow_points.shape[0] != prev_points.shape[0]:
				 logger.warning("Mismatched flow/prev points in _draw_flow_trails_simple")
				 return frame

			h, w = frame.shape[:2]
			for i in range(flow_points.shape[0]):
				xn, yn = flow_points[i].ravel()
				xo, yo = prev_points[i].ravel()

				# Convert to int for drawing
				pt1 = (int(xo), int(yo))
				pt2 = (int(xn), int(yn))

				# Basic bounds check (optional, cv2 handles clipping)
				# if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:

				# Draw line segment
				cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
				# Draw dot at the current position
				cv2.circle(frame, pt2, dot_radius, dot_color, -1, cv2.LINE_AA)

		except Exception as e:
			logger.warning(f"Draw flow trails failed: {e}", exc_info=True)
			# Don't increment global error count for drawing helpers usually

		return frame

	def _draw_motion_glow_separate(self,
								   frame: np.ndarray,
								   landmarks: List[Tuple[int, int]], # Current landmarks (unused here, but passed for consistency)
								   flow_points: np.ndarray,
								   prev_points: np.ndarray) -> np.ndarray:
		"""Draws a glow effect based on motion vectors."""
		frame = validate_frame(frame)
		try:
			# Create a separate mask for the glow
			glow_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

			if flow_points.shape[0] != prev_points.shape[0]:
				 logger.warning("Mismatched flow/prev points in _draw_motion_glow_separate")
				 return frame

			h, w = frame.shape[:2]
			# Draw the motion lines onto the mask
			for i in range(flow_points.shape[0]):
				xn, yn = flow_points[i].ravel()
				xo, yo = prev_points[i].ravel()
				pt1 = (int(xo), int(yo))
				pt2 = (int(xn), int(yn))
				# Thicker line for better blur effect
				cv2.line(glow_mask, pt1, pt2, 255, 4, cv2.LINE_AA) # White line on black mask

			# Blur the mask significantly to create the glow
			glow_mask_blurred = cv2.GaussianBlur(glow_mask, (25, 25), 0) # Adjust kernel size for glow spread

			# Convert mask to float [0, 1] and scale by glow intensity
			glow_alpha = glow_mask_blurred.astype(np.float32) / 255.0 * self.glow_intensity
			# Make it 3-channel for blending
			glow_alpha_3c = cv2.cvtColor(glow_alpha, cv2.COLOR_GRAY2BGR)

			# Define glow color (e.g., white or a specific color)
			glow_color_layer = np.full_like(frame, (255, 255, 255), dtype=np.float32) # White glow

			# Blend using the blurred mask as alpha
			frame_float = frame.astype(np.float32)
			frame_float = frame_float * (1.0 - glow_alpha_3c) + glow_color_layer * glow_alpha_3c
			frame = np.clip(frame_float, 0, 255).astype(np.uint8)

		except Exception as e:
			logger.warning(f"Draw motion glow failed: {e}", exc_info=True)

		return frame

	def _apply_distortion(self, frame: np.ndarray) -> np.ndarray:
		"""Applies a barrel/pincushion distortion effect (example)."""
		frame = validate_frame(frame)
		try:
			h, w = frame.shape[:2]
			center_x, center_y = w / 2, h / 2
			# Distortion parameters (k1, k2, p1, p2 usually from calibration)
			# For a simple effect, let's just use k1 (radial distortion)
			k1 = 0.0000001 # Positive for barrel, negative for pincushion - adjust magnitude
			k2 = 0.0 # Radial distortion coeff 2
			p1 = 0.0 # Tangential distortion coeff 1
			p2 = 0.0 # Tangential distortion coeff 2

			# Assume identity camera matrix for effect purposes
			cam_matrix = np.array([[w, 0, center_x], [0, h, center_y], [0, 0, 1]], dtype=np.float32)
			dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)

			# Undistort (or distort if coeffs are negative)
			# Use undistort for applying the effect based on coeffs
			frame_distorted = cv2.undistort(frame, cam_matrix, dist_coeffs)

			return frame_distorted

		except Exception as e:
			logger.warning(f"Distortion effect failed: {e}", exc_info=True)
			return frame # Return original on error

	def _detect_gestures(self, frame: np.ndarray) -> Optional[str]:
		"""Detects simple hand gestures using Mediapipe Hands."""
		start_time = time.time()
		if not self.hands or not self.frame_width or not self.frame_height:
			# logger.debug("Gesture detection: Missing dependencies or dimensions")
			return None

		gesture = None
		try:
			# Process with Hands model
			results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

			if results.multi_hand_landmarks:
				 # Analyze the first detected hand for simplicity
				 hand_landmarks = results.multi_hand_landmarks[0]
				 # (Could also check results.multi_handedness for left/right)
				 lm = hand_landmarks.landmark

				 # --- Simple Gesture Logic ---
				 # Check for "Open Hand" (e.g., 4 or 5 fingers extended)
				 # Compare tip y-coord with pip (proximal interphalangeal) y-coord
				 thumb_tip = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]
				 index_tip = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
				 middle_tip = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
				 ring_tip = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
				 pinky_tip = lm[mp.solutions.hands.HandLandmark.PINKY_TIP]

				 index_pip = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
				 middle_pip = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
				 ring_pip = lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP]
				 pinky_pip = lm[mp.solutions.hands.HandLandmark.PINKY_PIP]
				 thumb_ip = lm[mp.solutions.hands.HandLandmark.THUMB_IP] # Intermediate phalange

				 fingers_extended = 0
				 # Check fingers (relative to PIP joints, lower y means higher up)
				 if index_tip.y < index_pip.y: fingers_extended += 1
				 if middle_tip.y < middle_pip.y: fingers_extended += 1
				 if ring_tip.y < ring_pip.y: fingers_extended += 1
				 if pinky_tip.y < pinky_pip.y: fingers_extended += 1
				 # Check thumb (relative to IP joint and maybe MCP x-coord)
				 thumb_mcp = lm[mp.solutions.hands.HandLandmark.THUMB_MCP]
				 if thumb_tip.y < thumb_ip.y and thumb_tip.x < thumb_mcp.x: # Basic thumb out check (for right hand)
					 fingers_extended += 1


				 # Check for "Fist" (e.g., 0 or 1 finger extended)
				 # Tips are below PIPs
				 fingers_closed = 0
				 if index_tip.y > index_pip.y: fingers_closed += 1
				 if middle_tip.y > middle_pip.y: fingers_closed += 1
				 if ring_tip.y > ring_pip.y: fingers_closed += 1
				 if pinky_tip.y > pinky_pip.y: fingers_closed += 1
				 # Thumb might be tucked in
				 if thumb_tip.y > thumb_ip.y and thumb_tip.x > index_pip.x: # Basic thumb in check
					  fingers_closed +=1


				 # --- Assign Gesture String ---
				 if fingers_extended >= 4:
					 gesture = "open_hand"
					 logger.debug("Detected gesture: Open hand")
				 elif fingers_closed >= 4:
					 gesture = "fist"
					 logger.debug("Detected gesture: Fist")
				 # Add more gestures (e.g., pointing, peace sign) here

		except Exception as e:
			logger.warning(f"Gesture detection failed: {e}", exc_info=True)
			gesture = None # Ensure None on error

		# self.effect_times['gesture_detect'] = self.effect_times.get('gesture_detect', []) + [time.time() - start_time]
		return gesture

	def _load_rvm_background(self, bg_path: Optional[str]) -> None:
		"""Loads the background image for RVM or creates a default one."""
		if self.rvm_background is not None and bg_path == self._init_rvm_bg_path:
			# Avoid reloading the same background
			# Ensure it's resized if frame dimensions changed since load
			if self.rvm_background.shape[:2] != (self.frame_height, self.frame_width):
				 logger.info("Resizing cached RVM background due to frame dimension change.")
				 self.rvm_background = resize_frame(self.rvm_background, (self.frame_height, self.frame_width))
			return

		loaded_bg = None
		if bg_path and os.path.exists(bg_path):
			try:
				loaded_bg = cv2.imread(bg_path)
				if loaded_bg is not None:
					loaded_bg = validate_frame(loaded_bg) # Ensure BGR
					logger.info(f"Loaded RVM background from {bg_path}")
				else:
					logger.warning(f"cv2.imread failed for RVM background: {bg_path}")
			except Exception as e:
				logger.warning(f"Failed to load RVM background image '{bg_path}': {e}")
				loaded_bg = None # Fallback to default

		if loaded_bg is None:
			# Create default green screen background matching current frame size
			h, w = self.frame_height, self.frame_width
			self.rvm_background = np.full((h, w, 3), RVM_DEFAULT_BG_COLOR, dtype=np.uint8)
			logger.info(f"Using default {RVM_DEFAULT_BG_COLOR} RVM background ({w}x{h})")
		else:
			# Resize loaded background to match current frame size
			self.rvm_background = resize_frame(loaded_bg, (self.frame_height, self.frame_width))

		self._init_rvm_bg_path = bg_path # Store path used for loading

	def _preprocess_frame_rvm(self, frame: np.ndarray) -> Optional[torch.Tensor]:
		"""Prepares a frame for RVM model input."""
		try:
			frame = validate_frame(frame)
			# Convert BGR uint8 [0, 255] to RGB float32 [0, 1]
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# Convert to tensor, add batch dimension, move to device
			transform = T.ToTensor()
			src = transform(frame_rgb).unsqueeze(0).to(self.device)
			return src
		except Exception as e:
			logger.warning(f"RVM preprocess failed: {e}", exc_info=True)
			return None

	def _postprocess_output_rvm(self,
								fgr: torch.Tensor,
								pha: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
		"""Converts RVM model output tensors to NumPy arrays."""
		try:
			# Move tensors to CPU, remove batch dim, permute channels, convert to numpy
			fgr_np = fgr.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
			pha_np = pha.cpu().detach().squeeze(0).squeeze(0).numpy() # Alpha is single channel

			# Convert from [0, 1] float to [0, 255] uint8
			fgr_out = np.clip(fgr_np * 255.0, 0, 255).astype(np.uint8)
			pha_out = np.clip(pha_np * 255.0, 0, 255).astype(np.uint8)

			return fgr_out, pha_out
		except Exception as e:
			logger.warning(f"RVM postprocess failed: {e}", exc_info=True)
			return None, None

	# --- Video Capture and Processing Loop ---

	def _initialize_capture(self) -> bool:
		"""Initializes video capture and output writer."""
		logger.info(f"Initializing capture from: {self.input_source}")
		try:
			self.cap = cv2.VideoCapture(self.input_source)
			if not self.cap.isOpened():
				logger.error(f"Failed to open input source: {self.input_source}")
				return False

			# Try setting desired resolution and FPS (might not be supported by all cameras)
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
			self.cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)

			# Read actual dimensions and FPS
			self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.fps = self.cap.get(cv2.CAP_PROP_FPS)
			if self.fps <= 0:
				logger.warning(f"Capture source reported FPS <= 0 ({self.fps}). Using default: {DEFAULT_FPS}")
				self.fps = DEFAULT_FPS

			logger.info(f"Capture source opened: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS (requested {DEFAULT_WIDTH}x{DEFAULT_HEIGHT} @ {DEFAULT_FPS:.2f})")

			# Initial read to ensure source is working
			ret, frame = self.cap.read()
			if not ret or frame is None:
				logger.error("Failed to read initial frame from capture source.")
				self.cap.release()
				return False
			logger.info("Initial frame read successfully.")

			# Initialize Video Writer if in record mode
			if self.mode == "record":
				# Ensure output directory exists
				os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
				# Use appropriate fourcc codec (mp4v for .mp4, avc1 might need specific ffmpeg builds)
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				self.out = cv2.VideoWriter(
					self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
				)
				if not self.out.isOpened():
					logger.error(f"Failed to initialize video writer for: {self.output_path}")
					self.cap.release()
					return False
				logger.info(f"Recording enabled. Outputting to {self.output_path}")

			# Initialize Display Window and Trackbars if display is enabled
			if self.display:
				cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE) # Or WINDOW_NORMAL for resizable
				# Brightness Trackbar (-100 to 100, mapped to -1.0 to 1.0)
				cv2.createTrackbar(
					"Brightness", self.window_name, 100, 200,
					lambda x: setattr(self, "brightness_factor", (x - 100) / 100.0)
				)
				# Contrast Trackbar (0 to 200, mapped to 0.0 to 2.0)
				cv2.createTrackbar(
					"Contrast", self.window_name, 100, 200,
					lambda x: setattr(self, "contrast_factor", x / 100.0)
				)
				 # LUT Intensity Trackbar (0 to 100, mapped to 0.0 to 1.0)
				cv2.createTrackbar(
					"LUT Intensity", self.window_name, 100, 100,
					lambda x: setattr(self, "lut_intensity", x / 100.0)
				)
				logger.info("Display window initialized with trackbars.")

			return True

		except Exception as e:
			logger.error(f"Capture initialization failed: {e}", exc_info=True)
			if self.cap and self.cap.isOpened(): self.cap.release()
			if self.out and self.out.isOpened(): self.out.release()
			return False

	def _cleanup(self) -> None:
		"""Releases resources like capture, writer, and windows."""
		logger.info("Starting cleanup...")
		try:
			if self.cap and self.cap.isOpened():
				self.cap.release()
				logger.info("Video capture released.")
			if self.out and self.out.isOpened():
				self.out.release()
				logger.info("Video writer released.")
			# Close Mediapipe resources
			if hasattr(self.face_mesh, 'close') and callable(self.face_mesh.close): self.face_mesh.close()
			if hasattr(self.pose, 'close') and callable(self.pose.close): self.pose.close()
			if hasattr(self.hands, 'close') and callable(self.hands.close): self.hands.close()
			logger.info("Mediapipe resources closed (if initialized).")

			cv2.destroyAllWindows()
			logger.info("OpenCV windows destroyed.")
			logger.info("Cleanup completed.")
		except Exception as e:
			logger.warning(f"Error during cleanup: {e}", exc_info=True)

	@staticmethod
	def list_cameras() -> List[Tuple[int, str]]:
		"""Tries to identify available camera indices and basic info."""
		available_cameras = []
		logger.info("Scanning for available cameras (indices 0-9)...")
		for index in range(10): # Check indices 0 through 9
			cap_test = cv2.VideoCapture(index)
			if cap_test.isOpened():
				ret, _ = cap_test.read()
				if ret:
					# Try to get backend name (may vary)
					backend = cap_test.getBackendName()
					name = f"Camera {index} ({backend})"
					available_cameras.append((index, name))
					logger.info(f"  Found: {name}")
				else:
					 logger.debug(f"  Index {index} opened but failed to read frame.")
				cap_test.release()
			# else: logger.debug(f"  Index {index} could not be opened.")
		if not available_cameras:
			 logger.warning("No cameras detected.")
		return available_cameras

	def _handle_gestures(self, gesture: Optional[str], frame_time: float) -> None:
		"""Handles recognized gestures to control effects."""
		if not gesture:
			return

		current_time = time.time() # Use system time for cooldown, not frame_time
		gesture_cooldown = 1.5 # Seconds between recognizing the same gesture action

		# Check cooldown for the *same* gesture
		if (self.gesture_state["last_gesture"] == gesture and
				current_time - self.gesture_state["last_time"] < gesture_cooldown):
			return # Gesture still on cooldown

		logger.info(f"Gesture '{gesture}' detected (Cooldown passed).")
		self.gesture_state["last_gesture"] = gesture
		self.gesture_state["last_time"] = current_time
		self.gesture_state["gesture_count"] += 1

		# --- Gesture Actions ---
		if gesture == "open_hand":
			# Cycle to next effect
			effect_keys = list(self.effects.keys())
			current_idx = effect_keys.index(self.current_effect)
			next_idx = (current_idx + 1) % len(effect_keys)
			self.current_effect = effect_keys[next_idx]
			logger.info(f"Gesture 'open_hand': Switched to effect -> {self.current_effect}")

		elif gesture == "fist":
			# Cycle variant/sub-mode of the current effect
			if self.current_effect == "goldenaura":
				self.goldenaura_variant = (self.goldenaura_variant + 1) % len(self.goldenaura_variant_names)
				variant_name = self.goldenaura_variant_names[self.goldenaura_variant]
				logger.info(f"Gesture 'fist': Switched Goldenaura variant -> {variant_name}")
			elif self.current_effect == "lightning":
				self.lightning_style_index = (self.lightning_style_index + 1) % len(self.lightning_styles)
				style_name = self.lightning_style_names[self.lightning_style_index]
				logger.info(f"Gesture 'fist': Switched Lightning style -> {style_name}")
			elif self.current_effect == "lut_color_grade" and self.lut_color_grade_effect:
				self.lut_color_grade_effect.cycle_lut() # Use the method in the effect class
				# Log message is handled within cycle_lut
			elif self.current_effect == "rvm_composite":
				self.rvm_display_mode = (self.rvm_display_mode + 1) % len(self.RVM_DISPLAY_MODES)
				mode_name = self.RVM_DISPLAY_MODES[self.rvm_display_mode]
				logger.info(f"Gesture 'fist': Switched RVM display mode -> {mode_name}")
				# Optionally cycle RVM model too? Maybe too complex for one gesture.
				# if self.rvm_display_mode == 0: # Cycle model when returning to Composite
				#     self.current_rvm_model_idx = (self.current_rvm_model_idx + 1) % len(self.rvm_model_names)
				#     model_name = self.rvm_model_names[self.current_rvm_model_idx]
				#     logger.info(f"Gesture 'fist': Switched RVM model -> {model_name}")
			elif self.current_effect == "sam_segmentation":
				# Toggle between SAM v1 and v2 if both are available
				can_switch_sam = self.sam_model_v1 is not None and self.sam_model_v2 is not None
				if can_switch_sam:
					self.use_sam2_runtime = not self.use_sam2_runtime
					runtime_name = "SAM v2" if self.use_sam2_runtime else "SAM v1"
					logger.info(f"Gesture 'fist': Switched SAM runtime -> {runtime_name}")
				else:
					 logger.info("Gesture 'fist': Only one SAM version available, cannot switch.")
			else:
				 logger.info(f"Gesture 'fist': No specific action for effect '{self.current_effect}'.")
		# Add more gestures and actions here

	def run(self) -> None:
		"""Main processing loop for video effects."""
		main_start_time = time.time()

		# --- Initialization ---
		if not self._initialize_capture():
			logger.error("Failed to initialize capture. Exiting.")
			return

		# --- Load Heavy Models (after capture init confirms dimensions) ---
		if self.rvm_available:
			for model_name in self.rvm_model_names:
				# Load only if not already loaded or if device changed?
				if model_name not in self.rvm_models:
					 model = load_rvm_model(self.device, model_name, pretrained=True)
					 if model: self.rvm_models[model_name] = model

		if self.sam_available and self.sam_checkpoint_path and not self.sam_model_v1:
			self.sam_model_v1 = load_sam_mask_generator(
				self.device, checkpoint_path=self.sam_checkpoint_path
			)
		if self.sam2_available and self.sam2_checkpoint_path and not self.sam_model_v2:
			 # SAM2 loading uses ultralytics API
			 self.sam_model_v2 = load_sam2_video_predictor(
				 self.device, checkpoint_path=self.sam2_checkpoint_path
			 )

		frame_count = 0
		try:
			# --- Main Loop ---
			while True: # Loop indefinitely until 'q' or error
				loop_start_time = time.time()

				# 1. Read Frame
				ret, frame = self.cap.read()
				if not ret or frame is None:
					if isinstance(self.input_source, str): # Video file ended
						 logger.info("End of video file reached.")
					else: # Camera error
						 logger.error("Failed to read frame from capture source.")
					break # Exit loop

				# 2. Validate and Resize (if necessary - usually not needed if capture matches)
				frame = validate_frame(frame)
				if frame.shape[0] != self.frame_height or frame.shape[1] != self.frame_width:
					logger.warning(f"Frame dimensions {frame.shape[:2]} differ from expected ({self.frame_height}, {self.frame_width}). Resizing.")
					frame = resize_frame(frame, (self.frame_height, self.frame_width))


				# 3. Detect Gestures (optional, can be slow)
				# gesture = self._detect_gestures(frame)
				# self._handle_gestures(gesture, loop_start_time) # Pass time for potential timing logic

				# 4. Apply Current Effect
				if self.current_effect not in self.effects:
					logger.warning(f"Current effect '{self.current_effect}' not found. Resetting to 'none'.")
					self.current_effect = "none"

				effect_wrapper = self.effects[self.current_effect]
				frame_out = effect_wrapper(frame=frame.copy(), frame_time=loop_start_time, runner=self) # Pass runner for access to state

				frame_out = validate_frame(frame_out) # Ensure effect output is valid

				# 5. Apply Global Adjustments (Brightness/Contrast)
				# Apply adjustments only if they are non-default
				if self.brightness_factor != 0.0 or self.contrast_factor != 1.0:
					# Use convertScaleAbs for efficiency, but clip manually for precision
					# beta = brightness * 127.5 (scale -1 to 1 -> -127.5 to 127.5 approx)
					beta = self.brightness_factor * 100 # Adjust scaling factor as needed
					frame_out = cv2.convertScaleAbs(frame_out, alpha=self.contrast_factor, beta=beta)
					# Manual clipping might be safer if convertScaleAbs behaves unexpectedly
					# frame_float = frame_out.astype(np.float32)
					# frame_float = self.contrast_factor * frame_float + self.brightness_factor * 255
					# frame_out = np.clip(frame_float, 0, 255).astype(np.uint8)


				# 6. Add General Info Overlay (FPS, Effect Name)
				current_time = time.time()
				self.frame_times.append(current_time - loop_start_time)
				if len(self.frame_times) > 1:
					avg_time = sum(self.frame_times) / len(self.frame_times)
					avg_fps = 1.0 / avg_time if avg_time > 0 else 0
					fps_text = f"FPS: {avg_fps:.1f}"
				else:
					fps_text = "FPS: N/A"

				effect_text = f"Effect: {self.current_effect}"
				cv2.putText(frame_out, fps_text, (10, self.frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
				cv2.putText(frame_out, effect_text, (10, self.frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


				# 7. Write to Output Video (if recording)
				if self.out:
					try:
						self.out.write(frame_out)
					except Exception as write_e:
						 logger.error(f"Failed to write frame {frame_count} to video file: {write_e}")
						 # Decide whether to continue or stop recording

				# 8. Display Frame (if enabled)
				if self.display:
					try:
						cv2.imshow(self.window_name, frame_out)
					except Exception as display_e:
						# Handle potential window closed errors etc.
						logger.warning(f"cv2.imshow failed: {display_e}. Display might be closed.")
						self.display = False # Stop trying to display


				frame_count += 1
				# Optional: Log performance periodically
				# if frame_count % 100 == 0:
				#     logger.info(f"Processed {frame_count} frames. {fps_text}")


				# 9. Handle User Input (Keyboard)
				key = cv2.waitKey(1) & 0xFF # Wait key is crucial for display updates

				if key == ord('q'):
					logger.info("User pressed 'q'. Exiting.")
					break
				elif key == ord('e'):
					# Cycle effects forward
					effect_keys = list(self.effects.keys())
					current_idx = effect_keys.index(self.current_effect)
					next_idx = (current_idx + 1) % len(effect_keys)
					self.current_effect = effect_keys[next_idx]
					logger.info(f"Key 'e': Switched to effect -> {self.current_effect}")
					# Reset effect-specific states if needed when switching
					self.rvm_rec = [None] * 4 # Reset RVM state
				elif key == ord('w'):
					 # Cycle effects backward
					effect_keys = list(self.effects.keys())
					current_idx = effect_keys.index(self.current_effect)
					prev_idx = (current_idx - 1 + len(effect_keys)) % len(effect_keys)
					self.current_effect = effect_keys[prev_idx]
					logger.info(f"Key 'w': Switched to effect -> {self.current_effect}")
					self.rvm_rec = [None] * 4 # Reset RVM state

				elif key == ord('v'): # Cycle Variant/Sub-mode
					if self.current_effect == "goldenaura":
						self.goldenaura_variant = (self.goldenaura_variant + 1) % len(self.goldenaura_variant_names)
						logger.info(f"Key 'v': Switched Goldenaura variant -> {self.goldenaura_variant_names[self.goldenaura_variant]}")
					elif self.current_effect == "lightning":
						self.lightning_style_index = (self.lightning_style_index + 1) % len(self.lightning_styles)
						logger.info(f"Key 'v': Switched Lightning style -> {self.lightning_style_names[self.lightning_style_index]}")
					elif self.current_effect == "lut_color_grade" and self.lut_color_grade_effect:
						self.lut_color_grade_effect.cycle_lut()
					elif self.current_effect == "rvm_composite":
						self.rvm_display_mode = (self.rvm_display_mode + 1) % len(self.RVM_DISPLAY_MODES)
						logger.info(f"Key 'v': Switched RVM display mode -> {self.RVM_DISPLAY_MODES[self.rvm_display_mode]}")
					elif self.current_effect == "sam_segmentation":
						 can_switch_sam = self.sam_model_v1 is not None and self.sam_model_v2 is not None
						 if can_switch_sam:
							 self.use_sam2_runtime = not self.use_sam2_runtime
							 logger.info(f"Key 'v': Switched SAM runtime -> {'SAM v2' if self.use_sam2_runtime else 'SAM v1'}")
						 else: logger.info("Key 'v': Cannot switch SAM runtime (only one version loaded).")
					else:
						 logger.info(f"Key 'v': No variant action for effect '{self.current_effect}'.")

				elif key == ord('m'): # Cycle Model (e.g., for RVM)
					if self.current_effect == "rvm_composite" and len(self.rvm_models) > 1:
						 self.current_rvm_model_idx = (self.current_rvm_model_idx + 1) % len(self.rvm_model_names)
						 model_name = self.rvm_model_names[self.current_rvm_model_idx]
						 # Check if the switched-to model is actually loaded
						 if model_name not in self.rvm_models:
							  logger.warning(f"RVM model {model_name} not loaded, cannot switch.")
							  # Revert index or try next? Revert is safer.
							  self.current_rvm_model_idx = (self.current_rvm_model_idx - 1 + len(self.rvm_model_names)) % len(self.rvm_model_names)
						 else:
							 logger.info(f"Key 'm': Switched RVM model -> {model_name}")
							 self.rvm_rec = [None] * 4 # Reset state when switching model

				elif key == ord('r'): # Reset state
					logger.info("Key 'r': Resetting effects and settings...")
					for effect in self.effects.values():
						effect.reset() # Call reset on wrapper (which might call reset on effect obj)
					# Reset runner-level states
					self.current_effect = "none"
					self.brightness_factor = 0.0
					self.contrast_factor = 1.0
					self.lut_intensity = 1.0
					self.prev_gray = None
					self.prev_landmarks_flow = None
					self.face_history.clear()
					self.pose_history.clear()
					self.frame_buffer.clear()
					self.rvm_rec = [None] * 4
					# Reset trackbars if display is active
					if self.display:
						try:
							 cv2.setTrackbarPos("Brightness", self.window_name, 100)
							 cv2.setTrackbarPos("Contrast", self.window_name, 100)
							 cv2.setTrackbarPos("LUT Intensity", self.window_name, 100)
						except cv2.error:
							 logger.warning("Could not reset trackbars (window might be closed).")
					logger.info("Reset complete.")


		except KeyboardInterrupt:
			logger.info("KeyboardInterrupt received. Exiting.")
		except Exception as e:
			logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
		finally:
			# --- Cleanup ---
			self._cleanup()

			# --- Final Stats ---
			total_time = time.time() - main_start_time
			if frame_count > 0 and len(self.frame_times) > 0:
				avg_loop_time = sum(self.frame_times) / len(self.frame_times)
				avg_fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
				logger.info(f"--- Processing Summary ---")
				logger.info(f"Total frames processed: {frame_count}")
				logger.info(f"Total run time: {total_time:.2f} seconds")
				logger.info(f"Average FPS: {avg_fps:.2f}")
				logger.info(f"Total errors encountered: {self.error_count}")
				# Log average time per effect if collected
				# for name, times in self.effect_times.items():
				#     if times: logger.info(f"Avg time for '{name}': {sum(times)/len(times):.4f}s")
			else:
				logger.info("No frames processed.")


# --- Entry Point ---

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PixelSensing Video Effects Runner")
	# --- Add Arguments corresponding to config keys ---
	parser.add_argument("--config", type=str, default=CONFIG_FILE, help="Path to config JSON file")
	parser.add_argument("--input-source", type=str, help="Override config: Camera index or video file path")
	parser.add_argument("--output-path", type=str, help="Override config: Output video file path (for record mode)")
	parser.add_argument("--lut-dir", type=str, help="Override config: Directory containing LUT files (.cube)")
	parser.add_argument("--rvm-background-path", type=str, help="Override config: Path to RVM background image")
	parser.add_argument("--sam-checkpoint-path", type=str, help="Override config: Path to SAM v1 checkpoint")
	parser.add_argument("--sam2-checkpoint-path", type=str, help="Override config: Path to SAM 2 (ultralytics) checkpoint")
	parser.add_argument("--mode", type=str, choices=["live", "record"], help="Override config: Operation mode")
	parser.add_argument("--display", action=argparse.BooleanOptionalAction, help="Override config: Display video output (--display / --no-display)")
	parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Override config: Device for ML models")
	parser.add_argument("--trail-length", type=int, help="Override config: Trail length for effects like Goldenaura")
	parser.add_argument("--glow-intensity", type=float, help="Override config: Glow intensity (0.0-1.0)")
	parser.add_argument("--list-cameras", action="store_true", help="List available camera devices and exit")

	args = parser.parse_args()

	if args.list_cameras:
		PixelSensingEffectRunner.list_cameras()
		exit()

	# --- Create or Update Config based on Args ---
	# Load existing config first
	runner = PixelSensingEffectRunner(config_path=args.config) # Loads config internally
	config_to_save = runner.config.copy() # Get the loaded/default config

	# Override config with any provided command-line arguments
	overridden_keys = []
	for key, value in vars(args).items():
		if value is not None and key in config_to_save and key != "config" and key != "list_cameras":
			config_to_save[key] = value
			overridden_keys.append(key)

	if overridden_keys:
		 logger.info(f"Overriding config values with command line arguments for: {', '.join(overridden_keys)}")
		 # Re-apply the potentially modified config to the runner instance
		 runner.config = config_to_save
		 runner._apply_config()
		 # Reload LUTs and re-init effects if relevant keys changed (e.g., lut_dir, device)
		 if "lut_dir" in overridden_keys:
			 runner._load_luts()
		 # Re-initialize effects might be needed if device changed wrapper requirements
		 runner._initialize_effects()


	# Save the final config being used (optional)
	# try:
	#     with open(args.config, 'w') as f:
	#         json.dump(config_to_save, f, indent=4)
	#     logger.info(f"Saved current configuration to {args.config}")
	# except Exception as e:
	#     logger.warning(f"Could not save configuration to {args.config}: {e}")

	# --- Run the Application ---
	runner.run()

	'''
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
'''