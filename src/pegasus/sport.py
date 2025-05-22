# =============================================================================
# StarLeague Highlight Engine - v3.8.7
# Autonomous Multi-Activity Detection, Advanced Components (Placeholders for ML),
# Dynamic HMM Smoothing, Motion/Audio Cues, Tracking, Chunked Processing & Writing.
# License Check: DISABLED
# Syntax Fix: Corrected MemoryMonitor.check_memory method structure.
# Includes all previous fixes and improved video synthesis pipeline.
# Updated: Fixed 'remove_temp' error in EnhancedFFMPEGWriter initialization.
# New Fixes (v3.8.6):
# - Fixed audio extraction failure ('NoneType' object has no attribute 'stdout')
# - Fixed FFMPEG_VideoWriter initialization error (unexpected keyword argument 'audio_codec')
# - Fixed GUI not appearing and NameError for 'run_b' in launch_user_interface
# - Added FFmpeg availability check and improved error handling
# - Added focus_force to ensure GUI visibility
# - Set FFMPEG path explicitly in MoviePy config
# - Added checks for audio_chunk and audio_path in process_highlight_generation
# - Ensured GUI window is focused and visible
# New Features and Fixes (v3.8.7):
# - Added OpenCV-based fallback pipeline to handle FFMPEG failures
# - Fixed "expected str, bytes or os.PathLike object, not NoneType" error with robust path validation
# - Added pipeline selection toggle (FFMPEG/OpenCV) in UI and config
# - Enhanced debug logging for better issue diagnosis
# - Added progress tracking with percentage completion in logs and UI
# - Added email notifications for process completion/failure
# - Added temporary file validation and recovery
# - Improved GUI stability with threaded processing
# - Added disk space monitoring and memory optimization
# New Fixes (v3.8.8 - Applied in this Update):
# - Fixed audio extraction issues by separating video and audio writing in process_highlight_generation
# - Fixed Tkinter TclError by preventing GUI updates after window closure
# - Improved error handling for FFMPEG writer initialization
# - Added cleanup for temporary files in process_highlight_generation
# - Enhanced logging for audio processing steps
# =============================================================================

import tkinter
from tkinter import filedialog
import customtkinter
import os
import cv2
import math
import numpy as np
import time
import traceback
import sys
import gc  # Garbage Collection
import tempfile
import subprocess  # For FFMPEG check and kill
import logging  # Using Python's logging
import threading  # For thread-safe operations and UI threading
from collections import OrderedDict  # For tracker
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import configparser  # For configuration file support
import smtplib  # For email notifications
from email.mime.text import MIMEText  # For email notifications

# --- Set FFMPEG Path Explicitly ---
import moviepy.config as mp_config
mp_config.FFMPEG_BINARY = '/opt/homebrew/bin/ffmpeg'  # Adjust this path if necessary

# --- Load Configuration ---
config = configparser.ConfigParser()
config.read('config.ini')
USE_OPENCV = config.getboolean('Settings', 'use_opencv', fallback=False)

# --- Dependency Checks & Imports ---
INSTALL_INSTRUCTIONS = {
    "librosa": "pip install librosa soundfile",
    "moviepy": "pip install moviepy",
    "mediapipe": "pip install mediapipe",
    "numpy": "pip install numpy",
    "opencv-python": "pip install opencv-python",
    "customtkinter": "pip install customtkinter",
    "psutil": "pip install psutil"
}
essential_packages = ["cv2", "numpy", "customtkinter", "moviepy.editor", "mediapipe", "psutil"]
audio_packages = ["librosa", "soundfile"]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StarLeagueEngine")

def check_import(pkg_name):
    try:
        __import__(pkg_name.split('.')[0])
        logger.info(f"{pkg_name} found.")
        return True
    except ImportError:
        base_pkg = pkg_name.split('.')[0]
        logger.error(f"Lib '{base_pkg}' (for {pkg_name}) not found.")
        inst = INSTALL_INSTRUCTIONS.get(base_pkg, f"pip install {base_pkg}")
        logger.error(f" Install: {inst}")
        return False
    except Exception as e:
        logger.error(f"ERROR check import {pkg_name}: {e}")
        return False

all_deps_ok = all(check_import(pkg) for pkg in essential_packages)
audio_deps_ok = all(check_import(pkg) for pkg in audio_packages)

# --- New: Check FFmpeg Availability ---
def check_ffmpeg_availability():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run([mp_config.FFMPEG_BINARY, '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"FFmpeg found: {result.stdout.splitlines()[0]}")
            return True
        else:
            logger.error("FFmpeg not found or failed to execute.")
            return False
    except FileNotFoundError:
        logger.error("FFmpeg not found in system PATH.")
        return False
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {e}")
        return False

ffmpeg_available = check_ffmpeg_availability()

def display_user_alert(alert_text):
    try:
        if tkinter._default_root is not None and hasattr(tkinter._default_root, 'winfo_exists') and tkinter._default_root.winfo_exists():
            tkinter._default_root.destroy()
        tkinter._default_root = None
        root = tkinter.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        popup = tkinter.Toplevel(root)
        popup.title("StarLeague Alert")
        popup.geometry("450x180")
        popup.resizable(False, False)
        popup.attributes('-topmost', True)
        root.update_idletasks()
        ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = 450, 180
        x, y = (ws // 2) - (w // 2), (hs // 2) - (h // 2)
        popup.geometry(f'{w}x{h}+{x}+{y}')
        popup.protocol("WM_DELETE_WINDOW", root.destroy)
        frame = tkinter.Frame(popup, padx=20, pady=15)
        frame.pack(expand=True, fill='both')
        msg = tkinter.Label(frame, text=str(alert_text), wraplength=400, justify='left')
        msg.pack(pady=(0, 15), expand=True)
        btn = tkinter.Button(frame, text="OK", command=root.destroy, width=10, relief=tkinter.GROOVE)
        btn.pack(pady=5)
        popup.lift()
        popup.focus_force()
        root.mainloop()
    except Exception as e:
        print(f"ERROR popup: {e}\nMsg: {alert_text}")

if not all_deps_ok:
    logger.critical("Essential libs missing.")
    display_user_alert("CRITICAL ERROR: Essential libraries missing.\nCheck console log.")
    exit()

if not ffmpeg_available:
    logger.critical("FFmpeg is required but not found.")
    display_user_alert("CRITICAL ERROR: FFmpeg is required but not found.\nPlease install FFmpeg and ensure it's in your system PATH.")
    exit()

import cv2
import numpy as np
import customtkinter
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import mediapipe as mp
import psutil

if audio_deps_ok:
    import librosa
    import soundfile as sf
else:
    librosa, sf = None, None
    logger.warning("Audio libs not found.")

logger.info(f"Mediapipe version: {getattr(mp, '__version__', 'Unknown')}")
try:
    if hasattr(cv2, 'dnn_superres'):
        logger.info("OpenCV DNN SuperRes available.")
    else:
        logger.warning("OpenCV DNN SuperRes unavailable.")
except AttributeError:
    logger.warning("OpenCV DNN check failed.")

# ======================================== Filesystem Utility ========================================
def create_directory_if_needed(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created dir: {dir_path}")
    except Exception as e:
        logger.error(f"ERROR creating dir '{dir_path}': {e}")

def validate_path(path: Optional[str], path_type: str = "input") -> str:
    """Validate a file path and ensure it is usable."""
    if path is None:
        logger.error(f"{path_type} path is None")
        raise ValueError(f"{path_type} path cannot be None")
    if not isinstance(path, (str, bytes, os.PathLike)):
        logger.error(f"{path_type} path must be str, bytes, or os.PathLike, got {type(path)}")
        raise TypeError(f"{path_type} path must be str, bytes, or os.PathLike, got {type(path)}")
    if path_type == "input" and not os.path.exists(path):
        logger.error(f"{path_type} file does not exist: {path}")
        raise FileNotFoundError(f"{path_type} file does not exist: {path}")
    if path_type == "output":
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
    return path

# ======================================== Memory Monitor Component ==================================
class MemoryMonitor:
    """Monitors memory usage during video processing operations"""
    def __init__(self, threshold_percent: float = 80.0, check_interval: float = 1.0):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self.last_check = 0
        self.peak_usage = 0

    def check_memory(self, force: bool = False) -> Tuple[float, bool]:
        """Checks current memory usage"""
        current_time = time.time()
        if not force and (current_time - self.last_check) < self.check_interval:
            return 0.0, False  # Return float for consistency

        self.last_check = current_time
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0, False
        self.peak_usage = max(self.peak_usage, usage_percent)
        is_critical = usage_percent > self.threshold_percent
        if is_critical:
            logger.warning(f"Memory usage critical: {usage_percent:.1f}% (threshold: {self.threshold_percent:.1f}%)")
        return usage_percent, is_critical

    def force_cleanup(self) -> float:
        """Force garbage collection and memory cleanup"""
        collected = gc.collect()
        logger.debug(f"GC collected {collected} objects")
        usage, _ = self.check_memory(force=True)
        return usage

    def get_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        mem = psutil.virtual_memory()
        return {
            "current_percent": mem.percent,
            "peak_percent": self.peak_usage,
            "available_gb": mem.available / (1024 ** 3)
        }

# ======================================== Enhanced FFMPEG Writer Component ==========================
class EnhancedFFMPEGWriter:
    """Enhanced version of FFMPEG_VideoWriter with buffering and retry mechanisms"""
    def __init__(
        self,
        filename: str,
        size: Tuple[int, int],
        fps: float,
        codec: str = 'libx264',
        audiofile: Optional[str] = None,
        audio_codec: str = 'aac',  # Will be passed via ffmpeg_params
        audio_bitrate: str = '128k',
        preset: Optional[str] = None,
        bitrate: Optional[str] = None,
        withmask: bool = False,
        threads: int = 1,
        ffmpeg_params: Optional[List[str]] = None,
        logger_level: str = 'bar',
        buffer_size: int = 64,
        retry_count: int = 3
    ):
        self.filename = filename
        self.size = size
        self.fps = fps
        self.codec = codec
        self.audiofile = audiofile
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.preset = preset
        self.bitrate = bitrate
        self.withmask = withmask
        self.threads = threads
        self.ffmpeg_params = ffmpeg_params or []
        self.logger_level = logger_level
        self.buffer_size = buffer_size
        self.retry_count = retry_count
        self.frame_buffer = []
        self.frames_written = 0
        self.proc = None
        self.ffmpeg_writer = None
        self.logger = logger  # Use the global logger
        self._initialize_writer()

    def _initialize_writer(self):
        """Initialize the FFMPEG writer with error handling"""
        try:
            self.filename = validate_path(self.filename, "output")
            if self.audiofile:
                self.audiofile = validate_path(self.audiofile, "input audio")
            output_dir = os.path.dirname(os.path.abspath(self.filename))
            os.makedirs(output_dir, exist_ok=True)
            ffmpeg_additional_params = self.ffmpeg_params.copy()
            if self.bitrate:
                ffmpeg_additional_params.extend(['-b:v', self.bitrate])
            if self.audio_codec:
                ffmpeg_additional_params.extend(['-c:a', self.audio_codec])
            if self.audio_bitrate:
                ffmpeg_additional_params.extend(['-b:a', self.audio_bitrate])
            self.logger.debug(f"FFMPEG_VideoWriter init: filename={self.filename}, size={self.size}, fps={self.fps}, "
                             f"codec={self.codec}, audiofile={self.audiofile}, threads={self.threads}, "
                             f"ffmpeg_params={ffmpeg_additional_params}")
            self.ffmpeg_writer = FFMPEG_VideoWriter(
                filename=self.filename,
                size=self.size,
                fps=self.fps,
                codec=self.codec,
                audiofile=self.audiofile,
                preset=None,
                bitrate=None,
                withmask=self.withmask,
                logfile=None,
                threads=self.threads,
                ffmpeg_params=ffmpeg_additional_params
            )
            self.proc = self.ffmpeg_writer.proc
            self.logger.info(f"Initialized EnhancedFFMPEGWriter for {os.path.basename(self.filename)}")
        except Exception as e:
            self.logger.error(f"Error initializing FFMPEG writer: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def write_frame(self, img_array: np.ndarray):
        """Write a frame to the video with buffering"""
        self.frame_buffer.append(img_array.copy())
        if len(self.frame_buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush the frame buffer with retries"""
        if not self.frame_buffer:
            return
        retry = 0
        success = False
        while not success and retry <= self.retry_count:
            try:
                if retry > 0:
                    self.logger.warning(f"Retry {retry}/{self.retry_count} writing frames")
                if self.proc is None or self.proc.poll() is not None:
                    self.logger.info("Reinitializing writer")
                    self._initialize_writer()
                for frame in self.frame_buffer:
                    self.ffmpeg_writer.write_frame(frame)
                self.frames_written += len(self.frame_buffer)
                self.frame_buffer = []
                gc.collect()
                success = True
            except BrokenPipeError as e:
                self.logger.error(f"Broken pipe writing frames: {e}")
                retry += 1
                time.sleep(0.5)
                gc.collect()
            except Exception as e:
                self.logger.error(f"Error writing frames: {e}")
                self.logger.debug(traceback.format_exc())
                retry += 1
                time.sleep(0.5)
                gc.collect()
        if not success:
            self.logger.error(f"Failed writing frames after {self.retry_count} retries")
            raise IOError("Failed to write frames")

    def close(self):
        """Close the writer with proper resource cleanup"""
        try:
            if self.frame_buffer:
                self._flush_buffer()
            if self.ffmpeg_writer is not None:
                self.ffmpeg_writer.close()
            self.logger.info(f"Closed FFMPEG writer after writing {self.frames_written} frames")
        except Exception as e:
            self.logger.error(f"Error closing FFMPEG writer: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.logger.warning("FFMPEG terminate timed out, killing.")
                        self.proc.kill()
                except Exception as term_err:
                    self.logger.warning(f"Error during FFMPEG terminate/wait: {term_err}")
                    try:
                        self.logger.warning("Attempting to forcefully kill FFMPEG process.")
                        self.proc.kill()
                    except Exception as kill_err:
                        self.logger.error(f"Failed to kill FFMPEG process: {kill_err}")
            self.proc = None
            self.ffmpeg_writer = None
            gc.collect()

def write_video_opencv(output_path: str, video_frames: List[np.ndarray], fps: float, audio_path: Optional[str] = None) -> bool:
    """
    Write a video using OpenCV as a fallback to FFMPEG.
    
    Args:
        output_path (str): Path to save the output video
        video_frames (List[np.ndarray]): List of video frames as numpy arrays
        fps (float): Frames per second
        audio_path (Optional[str]): Path to the audio file
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Using OpenCV to write video: {output_path}")
    if not video_frames:
        logger.error("No video frames provided for OpenCV writer")
        return False

    # Get frame dimensions
    height, width, _ = video_frames[0].shape
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")

    # Initialize OpenCV video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.error("Failed to initialize OpenCV video writer")
            return False

        # Write frames
        for frame in video_frames:
            out.write(frame)
        out.release()
        logger.info(f"Wrote temporary video: {temp_video_path}")

        # Combine with audio using moviepy if audio is provided
        if audio_path and os.path.exists(audio_path):
            try:
                video_clip = VideoFileClip(temp_video_path)
                audio_clip = AudioFileClip(audio_path)
                video_clip = video_clip.set_audio(audio_clip)
                video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                logger.info(f"Successfully wrote final video with audio: {output_path}")
            except Exception as e:
                logger.error(f"Failed to combine video and audio with moviepy: {str(e)}")
                return False
            finally:
                video_clip.close()
                audio_clip.close()
        else:
            # If no audio, rename the temp video to the final output
            os.rename(temp_video_path, output_path)
            logger.info(f"Successfully wrote video without audio: {output_path}")

        return True
    except Exception as e:
        logger.error(f"OpenCV video writing failed: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.info(f"Cleaned up temporary video: {temp_video_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp video {temp_video_path}: {e}")

def send_email(subject: str, body: str, to_email: str, from_email: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str) -> None:
    """Send an email notification."""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())
        logger.info(f"Email sent to {to_email}: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")

# ======================================== Chunked Video Processor Component ========================
class ChunkedVideoProcessor:
    """Processes videos in chunks with robust error handling"""
    def __init__(
        self,
        chunk_duration: float = 10.0,
        max_retries: int = 3,
        memory_threshold: float = 80.0,
        temp_dir: Optional[str] = None,
        threads: Optional[int] = None,
        logger_name: str = "ChunkedVideoProcessor"
    ):
        self.chunk_duration = chunk_duration
        self.max_retries = max_retries
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.threads = threads or max(1, os.cpu_count() // 2)
        self.memory_monitor = MemoryMonitor(memory_threshold)
        self.logger = logging.getLogger(logger_name)
        self.lock = threading.Lock()  # Added for thread-safe temporary file creation
        # Verify temp directory permissions
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            test_file = os.path.join(self.temp_dir, "test_write.txt")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            self.logger.info(f"Temporary directory {self.temp_dir} is writable.")
        except Exception as e:
            self.logger.error(f"Cannot write to temp directory {self.temp_dir}: {e}")
            raise RuntimeError(f"Temp directory {self.temp_dir} is not writable.")
        # Check disk space
        try:
            stat = os.statvfs(self.temp_dir)
            free_space = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)  # Free space in GB
            if free_space < 1.0:
                self.logger.warning(f"Low disk space in temp directory {self.temp_dir}: {free_space:.2f} GB available")
        except Exception as e:
            self.logger.warning(f"Could not check disk space for {self.temp_dir}: {e}")
        self.temp_files = []

    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Failed remove temp {temp_file}: {e}")
        self.temp_files = []

    def _create_temp_filename(self, prefix: str = "chunk_", suffix: str = ".mp4") -> str:
        """Create a temporary filename with thread safety"""
        with self.lock:
            temp_file = os.path.join(
                self.temp_dir,
                f"{prefix}{int(time.time()*1000)}_{os.getpid()}_{np.random.randint(1000,9999)}{suffix}"
            )
            self.temp_files.append(temp_file)
            return temp_file

    def process_highlight_generation(
        self,
        input_path: str,
        output_path: str,
        intervals: List[Tuple[float, float]],
        output_params: Optional[Dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> bool:
        """Generate highlights from specified intervals using chunked processing"""
        if not os.path.exists(input_path):
            self.logger.error(f"Input missing: {input_path}")
            return False
        if not intervals:
            self.logger.error("No intervals provided")
            return False
        output_params = output_params or {
            'codec': 'libx264',
            'audio_codec': 'aac',
            'remove_temp': True,
            'threads': self.threads,
            'logger': 'bar'
        }
        acceptable_params = {
            'codec', 'audio_codec', 'audio_bitrate', 'preset',
            'bitrate', 'withmask', 'threads', 'ffmpeg_params'
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        processed_clips_paths = []
        total_intervals = len(intervals)
        main_clip_obj = None
        try:
            main_clip_obj = VideoFileClip(input_path)
            duration = main_clip_obj.duration
            fps = main_clip_obj.fps
            size = main_clip_obj.size
            has_audio = main_clip_obj.audio is not None
            self.logger.info(f"Input video: duration={duration:.2f}s, fps={fps}, size={size}, has_audio={has_audio}")
            for i, (start, end) in enumerate(intervals):
                start = max(0, start)
                end = min(end, duration)
                if end <= start + 0.01:
                    self.logger.warning(f"Skipping tiny interval {i+1}")
                    continue
                chunk_success = False
                retries = 0
                writer = None
                video_frames = []
                temp_output_path = self._create_temp_filename(f"highlight_{i+1}_")
                temp_audiofile_path = self._create_temp_filename(f"h_audio_{i+1}_", ".wav") if has_audio else None
                audio_chunk = None
                try:
                    if retries > 0:
                        self.logger.warning(f"Retry {retries}/{self.max_retries} interval {i+1}")
                    _, mem_critical = self.memory_monitor.check_memory()
                    if mem_critical:
                        self.memory_monitor.force_cleanup()
                    self.logger.info(f"Processing interval {i+1}/{total_intervals} ({start:.2f}s - {end:.2f}s) -> {os.path.basename(temp_output_path)}")

                    # Collect frames for processing
                    with main_clip_obj.subclip(start, end) as video_subclip:
                        for frame in video_subclip.iter_frames(fps=fps, dtype="uint8"):
                            video_frames.append(frame)

                    # Write video first (without audio)
                    self.logger.info(f"Processing interval with video: {temp_output_path}")
                    if not USE_OPENCV:
                        try:
                            writer_kwargs = {k: v for k, v in output_params.items() if k in acceptable_params}
                            writer = EnhancedFFMPEGWriter(
                                temp_output_path,
                                size,
                                fps,
                                audiofile=None,  # Write video without audio first
                                **writer_kwargs
                            )
                            for frame in video_frames:
                                writer.write_frame(frame)
                            writer.close()
                            writer = None
                        except Exception as e:
                            self.logger.warning(f"FFMPEG failed for interval {i+1}: {e}. Falling back to OpenCV.")
                            if writer:
                                writer.close()
                            # Fallback to OpenCV
                            success = write_video_opencv(
                                temp_output_path,
                                video_frames,
                                fps,
                                None  # No audio at this stage
                            )
                            if not success:
                                self.logger.error(f"OpenCV fallback also failed for interval {i+1}")
                                retries += 1
                                continue
                    else:
                        # Use OpenCV directly
                        success = write_video_opencv(
                            temp_output_path,
                            video_frames,
                            fps,
                            None  # No audio at this stage
                        )
                        if not success:
                            self.logger.error(f"OpenCV writing failed for interval {i+1}")
                            retries += 1
                            continue

                    # If audio is required, extract and merge it
                    if has_audio and main_clip_obj.audio:
                        self.logger.info(f"Processing interval with audio: {temp_audiofile_path}")
                        try:
                            # Extract audio for the interval
                            audio_chunk = main_clip_obj.audio.subclip(start, end)
                            if audio_chunk is None:
                                raise ValueError("Audio clip is None, cannot extract audio.")
                            
                            # Write audio to a temporary file
                            audio_chunk.write_audiofile(
                                temp_audiofile_path,
                                codec='pcm_s16le',
                                ffmpeg_params=['-ar', '44100'],
                                logger=None
                            )
                            
                            # Merge audio with video using FFmpeg
                            temp_video_path = temp_output_path.replace('.mp4', '_temp.mp4')
                            cmd = [
                                'ffmpeg',
                                '-i', temp_output_path,
                                '-i', temp_audiofile_path,
                                '-c:v', 'copy',
                                '-c:a', 'aac',
                                '-map', '0:v:0',
                                '-map', '1:a:0',
                                '-shortest',
                                temp_video_path
                            ]
                            subprocess.run(cmd, check=True)
                            
                            # Replace the original video with the one with audio
                            os.remove(temp_output_path)
                            os.rename(temp_video_path, temp_output_path)
                            self.logger.info(f"Successfully wrote final video with audio: {temp_output_path}")
                            
                        except Exception as e:
                            self.logger.warning(f"Could not extract/write audio for interval {i+1}: {e}")
                            self.logger.info("Processing interval without audio (audio unavailable or extraction failed)")
                        finally:
                            if audio_chunk is not None:
                                audio_chunk.close()

                    processed_clips_paths.append(temp_output_path)
                    chunk_success = True

                    if progress_callback:
                        progress_callback((i + 1) / total_intervals)
                        self.logger.info(f"Interval {i+1}/{total_intervals} completed ({((i+1)/total_intervals)*100:.2f}% of total intervals)")
                except Exception as e:
                    self.logger.error(f"Error processing interval {i+1}: {e}")
                    self.logger.debug(traceback.format_exc())
                    retries += 1
                    self.memory_monitor.force_cleanup()
                    if writer:
                        try:
                            writer.close()
                        except Exception as wc_err:
                            self.logger.warning(f"Error closing writer during except: {wc_err}")
                    if retries > self.max_retries:
                        self.logger.error(f"Failed interval {i+1} after retries.")
                        break
                finally:
                    if temp_audiofile_path and os.path.exists(temp_audiofile_path):
                        try:
                            os.remove(temp_audiofile_path)
                            self.logger.debug(f"Removed temp audio file: {temp_audiofile_path}")
                        except Exception as rm_err:
                            self.logger.warning(f"Could not remove temp audio {temp_audiofile_path}: {rm_err}")

            if not processed_clips_paths:
                self.logger.error("No highlight intervals processed successfully.")
                return False
            self.logger.info(f"Concatenating {len(processed_clips_paths)} highlight clips...")
            try:
                final_clips = [VideoFileClip(p) for p in processed_clips_paths if os.path.exists(p)]
                if not final_clips:
                    raise ValueError("Failed to load any processed clips for concatenation.")
                final_video = concatenate_videoclips(final_clips, method="compose")
                self.logger.info(f"Writing final highlight video: {output_path}")
                final_output_params = {k: v for k, v in output_params.items() if k != 'preset'}
                final_video.write_videofile(output_path, **final_output_params)
                for clip_obj in final_clips:
                    clip_obj.close()
                final_video.close()
                self.logger.info("Highlight concatenation successful.")
                return True
            except Exception as e:
                self.logger.error(f"Error during highlight concatenation/write: {e}")
                self.logger.debug(traceback.format_exc())
                return False
        except Exception as e:
            self.logger.error(f"Error opening/reading video {input_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
        finally:
            if main_clip_obj:
                main_clip_obj.close()
            self.cleanup()
            gc.collect()

# ======================================== AUTONOMOUS POSE ASSESSOR ========================================
class AutonomousPoseAssessor:
    """Evaluates human poses for activity detection"""
    def __init__(self):
        if mp is None:
            raise ImportError("Mediapipe required.")
        self.mp_pose_solution = mp.solutions.pose
        self.mp_draw_utils = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self._activity_rules = {
            "BBall_JumpShot": self._rule_jumpshot,
            "BBall_Layup": self._rule_layup,
            "BBall_Defense": self._rule_defense,
            "Soccer_Volley": self._rule_volley,
            "Soccer_Sidekick": self._rule_sidekick,
            "Soccer_Header": self._rule_header,
            "Human": lambda l, t, m: 0.8
        }
        self.ml_pose_classifier = None
        print("INFO: AutonomousPoseAssessor initialized with rules:", list(self._activity_rules.keys()))

    def _get_landmark(self, landmarks, landmark_enum):
        try:
            lm = landmarks[landmark_enum]
            return lm if hasattr(lm, 'x') else None
        except IndexError:
            return None
        except Exception as e:
            print(f"WARN: Landmark access error {landmark_enum}: {e}")
            return None

    def _calc_dist(self, p1, p2):
        if p1 and p2:
            try:
                if hasattr(p1, 'x') and hasattr(p1, 'y') and hasattr(p2, 'x') and hasattr(p2, 'y'):
                    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                else:
                    print("WARN: Landmark missing x/y in _calc_dist.")
                    return float('inf')
            except Exception as e:
                print(f"WARN: Error in distance calc: {e}")
                return float('inf')
        return float('inf')

    def _calc_angle(self, p1, p2, p3):
        if not all([p1, p2, p3]):
            return None
        try:
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v3 = np.array([p3.x - p2.x, p3.y - p2.y])
            m1 = np.linalg.norm(v1)
            m3 = np.linalg.norm(v3)
        except Exception as e:
            print(f"WARN: Angle vector error: {e}")
            return None
        if m1 < 1e-9 or m3 < 1e-9:
            return None
        try:
            dot = np.dot(v1, v3)
            cos = np.clip(dot / (m1 * m3), -1.0, 1.0)
            return np.degrees(np.arccos(cos))
        except Exception as e:
            print(f"WARN: Angle calculation error: {e}")
            return None

    def _rule_jumpshot(self, landmarks, torso_len, motion_mag):
        get = lambda e: self._get_landmark(landmarks, e)
        lsh, rsh, lel, rel, lwr, rwr = get(self.mp_pose_solution.PoseLandmark.LEFT_SHOULDER), get(self.mp_pose_solution.PoseLandmark.RIGHT_SHOULDER), get(self.mp_pose_solution.PoseLandmark.LEFT_ELBOW), get(self.mp_pose_solution.PoseLandmark.RIGHT_ELBOW), get(self.mp_pose_solution.PoseLandmark.LEFT_WRIST), get(self.mp_pose_solution.PoseLandmark.RIGHT_WRIST)
        lhip, rhip, lkn, rkn, lank, rank = get(self.mp_pose_solution.PoseLandmark.LEFT_HIP), get(self.mp_pose_solution.PoseLandmark.RIGHT_HIP), get(self.mp_pose_solution.PoseLandmark.LEFT_KNEE), get(self.mp_pose_solution.PoseLandmark.RIGHT_KNEE), get(self.mp_pose_solution.PoseLandmark.LEFT_ANKLE), get(self.mp_pose_solution.PoseLandmark.RIGHT_ANKLE)
        lk_ang = self._calc_angle(lhip, lkn, lank)
        rk_ang = self._calc_angle(rhip, rkn, rank)
        le_ang = self._calc_angle(lsh, lel, lwr)
        re_ang = self._calc_angle(rsh, rel, rwr)
        if all(a is not None for a in [lk_ang, rk_ang, le_ang, re_ang]) and all(p is not None for p in [lwr, rwr, lsh, rsh]):
            knees_bent = (lk_ang < 160 and rk_ang < 160)
            elbows_extended = (le_ang > 160 and re_ang > 160)
            wrists_above_shoulders = (lwr.y < lsh.y and rwr.y < rsh.y)
            if knees_bent and elbows_extended and wrists_above_shoulders:
                mb = min(1.0, motion_mag / 10.0)
                return 0.7 + 0.2 * mb
        return 0.0

    def _rule_layup(self, landmarks, torso_len, motion_mag):
        get = lambda e: self._get_landmark(landmarks, e)
        lwr, rwr = get(self.mp_pose_solution.PoseLandmark.LEFT_WRIST), get(self.mp_pose_solution.PoseLandmark.RIGHT_WRIST)
        lhip, rhip, lkn, rkn, lank, rank = get(self.mp_pose_solution.PoseLandmark.LEFT_HIP), get(self.mp_pose_solution.PoseLandmark.RIGHT_HIP), get(self.mp_pose_solution.PoseLandmark.LEFT_KNEE), get(self.mp_pose_solution.PoseLandmark.RIGHT_KNEE), get(self.mp_pose_solution.PoseLandmark.LEFT_ANKLE), get(self.mp_pose_solution.PoseLandmark.RIGHT_ANKLE)
        nose = get(self.mp_pose_solution.PoseLandmark.NOSE)
        lk_ang = self._calc_angle(lhip, lkn, lank)
        rk_ang = self._calc_angle(rhip, rkn, rank)
        if all(a is not None for a in [lk_ang, rk_ang]) and all(p is not None for p in [lwr, rwr, nose]):
            right_l = (rwr.y < nose.y) and (lk_ang < 130) and (rk_ang > 150)
            left_l = (lwr.y < nose.y) and (rk_ang < 130) and (lk_ang > 150)
            if right_l or left_l:
                return 0.75
        return 0.0

    def _rule_defense(self, landmarks, torso_len, motion_mag):
        get = lambda e: self._get_landmark(landmarks, e)
        lsh, rsh, lwr, rwr = get(self.mp_pose_solution.PoseLandmark.LEFT_SHOULDER), get(self.mp_pose_solution.PoseLandmark.RIGHT_SHOULDER), get(self.mp_pose_solution.PoseLandmark.LEFT_WRIST), get(self.mp_pose_solution.PoseLandmark.RIGHT_WRIST)
        lhip, rhip, lkn, rkn, lank, rank = get(self.mp_pose_solution.PoseLandmark.LEFT_HIP), get(self.mp_pose_solution.PoseLandmark.RIGHT_HIP), get(self.mp_pose_solution.PoseLandmark.LEFT_KNEE), get(self.mp_pose_solution.PoseLandmark.RIGHT_KNEE), get(self.mp_pose_solution.PoseLandmark.LEFT_ANKLE), get(self.mp_pose_solution.PoseLandmark.RIGHT_ANKLE)
        lk_ang = self._calc_angle(lhip, lkn, lank)
        rk_ang = self._calc_angle(rhip, rkn, rank)
        if all(a is not None for a in [lk_ang, rk_ang]) and all(p is not None for p in [lhip, rhip, lank, rank, lwr, rwr, lsh, rsh]):
            kb = (lk_ang < 160 and rk_ang < 160)
            hy = (lhip.y + rhip.y) / 2
            ay = (lank.y + rank.y) / 2
            hl = hy > (ay - 0.1 * torso_len)
            adl = abs(lwr.x - lsh.x)
            adr = abs(rwr.x - rsh.x)
            aw = (adl > 0.3 * torso_len and adr > 0.3 * torso_len)
            if kb and hl and aw and motion_mag < 3.0:
                return 0.65
        return 0.0

    def _rule_volley(self, landmarks, torso_len, motion_mag):
        get = lambda e: self._get_landmark(landmarks, e)
        lhip, rhip, lkn, rkn, lank, rank = get(self.mp_pose_solution.PoseLandmark.LEFT_HIP), get(self.mp_pose_solution.PoseLandmark.RIGHT_HIP), get(self.mp_pose_solution.PoseLandmark.LEFT_KNEE), get(self.mp_pose_solution.PoseLandmark.RIGHT_KNEE), get(self.mp_pose_solution.PoseLandmark.LEFT_ANKLE), get(self.mp_pose_solution.PoseLandmark.RIGHT_ANKLE)
        if all(p is not None for p in [lhip, rhip, lkn, rkn, lank, rank]):
            rv = (rkn.y < rhip.y) and (rank.y < rkn.y)
            lv = (lkn.y < lhip.y) and (lank.y < lkn.y)
            if rv or lv:
                mb = min(1.0, motion_mag / 5.0)
                return 0.6 + 0.2 * mb
        return 0.0

    def _rule_sidekick(self, landmarks, torso_len, motion_mag):
        get = lambda e: self._get_landmark(landmarks, e)
        lhip, rhip, lkn, rkn, lank, rank = get(self.mp_pose_solution.PoseLandmark.LEFT_HIP), get(self.mp_pose_solution.PoseLandmark.RIGHT_HIP), get(self.mp_pose_solution.PoseLandmark.LEFT_KNEE), get(self.mp_pose_solution.PoseLandmark.RIGHT_KNEE), get(self.mp_pose_solution.PoseLandmark.LEFT_ANKLE), get(self.mp_pose_solution.PoseLandmark.RIGHT_ANKLE)
        lk_ang = self._calc_angle(lhip, lkn, lank)
        rk_ang = self._calc_angle(rhip, rkn, rank)
        if all(p is not None for p in [lank, rank, lkn, rkn]) and all(a is not None for a in [lk_ang, rk_ang]):
            rs = abs(rank.x - rkn.x) > 0.2 * torso_len and (90 < rk_ang < 120)
            ls = abs(lank.x - lkn.x) > 0.2 * torso_len and (90 < lk_ang < 120)
            if rs or ls:
                return 0.6
        return 0.0

    def _rule_header(self, landmarks, torso_len, motion_mag):
        get = lambda e: self._get_landmark(landmarks, e)
        lsh, rsh, nose = get(self.mp_pose_solution.PoseLandmark.LEFT_SHOULDER), get(self.mp_pose_solution.PoseLandmark.RIGHT_SHOULDER), get(self.mp_pose_solution.PoseLandmark.NOSE)
        if all(p is not None for p in [nose, lsh, rsh]):
            msx = (lsh.x + rsh.x) / 2
            msy = (lsh.y + rsh.y) / 2
            hvx = nose.x - msx
            hvy = nose.y - msy
            if abs(hvy) > 1e-6:
                av = math.degrees(math.atan2(hvx, -hvy))
            if abs(av) > 45:
                return 0.6
        return 0.0

    def evaluate_pose_autonomously(self, pose_data, motion_mag=0.0):
        """Evaluate pose for activity detection"""
        if not pose_data or not pose_data.pose_landmarks:
            return 0.0
        landmarks = pose_data.pose_landmarks.landmark
        lsh = self._get_landmark(landmarks, self.mp_pose_solution.PoseLandmark.LEFT_SHOULDER)
        rsh = self._get_landmark(landmarks, self.mp_pose_solution.PoseLandmark.RIGHT_SHOULDER)
        lhip = self._get_landmark(landmarks, self.mp_pose_solution.PoseLandmark.LEFT_HIP)
        rhip = self._get_landmark(landmarks, self.mp_pose_solution.PoseLandmark.RIGHT_HIP)
        if not all([lsh, rsh, lhip, rhip]):
            return 0.0
        dl = self._calc_dist(lsh, lhip)
        dr = self._calc_dist(rsh, rhip)
        vd = [d for d in [dl, dr] if d != float('inf')]
        torso = sum(vd) / len(vd) if vd else 0
        if torso < 1e-6:
            return 0.0
        max_score = 0.0
        for rule_func in self._activity_rules.values():
            try:
                max_score = max(max_score, rule_func(landmarks, torso, motion_mag))
            except Exception as e:
                print(f"WARN: Rule func error: {e}")
        return np.clip(max_score, 0.0, 1.0)

    def visualize_pose(self, bgr_frame, pose_data):
        """Visualize pose landmarks on frame"""
        img = bgr_frame.copy()
        if pose_data and pose_data.pose_landmarks:
            self.mp_draw_utils.draw_landmarks(
                img,
                pose_data.pose_landmarks,
                self.mp_pose_solution.POSE_CONNECTIONS,
                landmark_drawing_spec=self.styles.get_default_pose_landmarks_style()
            )
        return img

# ======================================== TEMPORAL SMOOTHER (HMM) ========================================
class TemporalSmootherHMM:
    """Smooths binary sequences using Hidden Markov Models"""
    def __init__(self):
        print("INFO: TemporalSmootherHMM initialized.")

    def _calculate_dynamic_transitions(self, score_sequence, window=15, base_p_stay=0.75, sensitivity=0.2):
        n = len(score_sequence)
        trans_probs = np.zeros((n, 2, 2))
        seq_arr = np.array(score_sequence)
        for t in range(n):
            start = max(0, t - window)
            history = seq_arr[start:t]
            if len(history) > 0:
                lm = np.mean(history)
                ps1 = np.clip(base_p_stay + sensitivity * (lm - 0.5) * 2, 0.1, 0.95)
                ps0 = np.clip(base_p_stay + sensitivity * (0.5 - lm) * 2, 0.1, 0.95)
                trans_probs[t, 0, 0] = ps0
                trans_probs[t, 0, 1] = 1.0 - ps0
                trans_probs[t, 1, 0] = 1.0 - ps1
                trans_probs[t, 1, 1] = ps1
            else:
                trans_probs[t, :, :] = [[base_p_stay, 1 - base_p_stay], [1 - base_p_stay, base_p_stay]]
        return trans_probs

    def smooth_sequence(self, score_sequence, confidence_sequence, smoothing_radius, use_dynamic_transitions=True):
        """Smooth binary sequence with HMM"""
        if not isinstance(score_sequence, list) or not score_sequence:
            return []
        n = len(score_sequence)
        obs = np.array(score_sequence)  # Assume input is already binary 0/1
        if not np.all(np.isin(obs, [0, 1])):
            print("WARN: HMM received non-binary scores, applying 0.5 threshold.")
            obs = np.array([1 if s > 0.5 else 0 for s in score_sequence])
        conf = np.array(confidence_sequence) if confidence_sequence is not None and len(confidence_sequence) == n else np.full(n, 0.9)
        states = [0, 1]
        n_s = len(states)
        log_eps = 1e-10
        log_b = np.zeros((n, n_s, 2))
        for t in range(n):
            c = np.clip(conf[t], 0.1, 0.99)
            log_b[t, 1, 1] = np.log(c + log_eps)
            log_b[t, 1, 0] = np.log(1.0 - c + log_eps)
            log_b[t, 0, 0] = np.log(c + log_eps)
            log_b[t, 0, 1] = np.log(1.0 - c + log_eps)
        log_pi = np.log(np.array([0.5, 0.5]) + log_eps)
        if use_dynamic_transitions:
            print("INFO: Using dynamic HMM transitions.")
            log_a = np.log(self._calculate_dynamic_transitions(score_sequence, window=max(5, smoothing_radius)) + log_eps)
        else:
            print("INFO: Using static HMM transitions.")
            ps = 1.0 - (1.0 / (smoothing_radius + 2.0)) if smoothing_radius > 0 else 0.5
            psw = 1.0 - ps
            log_a_static = np.log(np.array([[ps, psw], [psw, ps]]) + log_eps)
        V = np.full((n, n_s), -np.inf)
        path = np.zeros((n, n_s), dtype=int)
        obs_0 = obs[0]
        for s in states:
            V[0, s] = log_pi[s] + log_b[0, s, obs_0]
            path[0, s] = s
        for t in range(1, n):
            obs_t = obs[t]
            current_log_a = log_a[t - 1] if use_dynamic_transitions else log_a_static
            for s in states:
                lp = V[t - 1, :] + current_log_a[:, s]
                bp = np.argmax(lp)
                V[t, s] = lp[bp] + log_b[t, s, obs_t]
                path[t, s] = bp
        sm = np.zeros(n, dtype=int)
        sm[n - 1] = np.argmax(V[n - 1, :])
        for t in range(n - 2, -1, -1):
            sm[t] = path[t + 1, sm[t + 1]]
        return sm.tolist()

# ======================================== FRAME PIPELINE (Enhancer) ========================================
class FramePipeline:
    """Processes video frames with optional upsampling"""
    def __init__(self):
        self.upsampler = None
        self.ready = False
        self.dims = (256, 256)
        print("INFO: FramePipeline initialized.")

    def configure_upsampler(self, mid="edsr", fac=2):
        """Configure DNN-based upsampler"""
        if mid.lower() in ['esrgan', 'realesrgan']:
            print(f"WARN: {mid} requested but not implemented.")
            mid = 'edsr'
        if not hasattr(cv2, 'dnn_superres'):
            print("ERROR: DNN SuperRes unavailable.")
            return False
        try:
            fn = f"{mid.upper()}_x{fac}.pb"
            p = os.path.join("models", fn)
        except Exception as e:
            print(f"ERROR path gen: {e}")
            return False
        if not os.path.exists(p):
            print(f"ERROR model missing: {p}")
            return False
        try:
            print(f"INFO: Loading CV DNN upsampler: {p}...")
            self.upsampler = cv2.dnn_superres.DnnSuperResImpl_create()
            self.upsampler.readModel(p)
            self.upsampler.setModel(mid.lower(), fac)
            self.ready = True
            print("INFO: CV DNN Upsampler configured.")
            return True
        except Exception as e:
            print(f"ERROR config upsampler: {e}")
            self.upsampler = None
            self.ready = False
            return False

    def _apply_upsampling(self, frame):
        if not self.ready or self.upsampler is None or frame is None:
            return frame
        try:
            return self.upsampler.upsample(frame)
        except Exception as e:
            print(f"ERROR applying upsampling: {e}")
            return frame

    def _normalize_size(self, frame):
        if frame is None:
            return None
        try:
            h, w = frame.shape[:2]
        except Exception as e:
            print(f"WARN: Shape error: {e}")
            return frame
        if h == 0 or w == 0:
            print(f"WARN: Invalid dims ({w}x{h}).")
            return frame
        try:
            tw, th = self.dims
            iar = w / h
            tar = tw / th
            if iar > tar:
                nw = tw
                nh = max(1, int(nw / iar))
            else:
                nh = th
                nw = max(1, int(nh * iar))
            nw = min(nw, tw)
            nh = min(nh, th)
            if nh > 0 and nw > 0:
                return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                print(f"WARN: Invalid resize dims ({nw}x{nh}).")
                return frame
        except Exception as e:
            print(f"ERROR resize: {e}")
            return frame

    def process_frame(self, frame, enable_upsample=False):
        """Process a single frame"""
        processed = frame
        if enable_upsample:
            processed = self._apply_upsampling(processed)
        processed = self._normalize_size(processed)
        return processed

# ======================================== AUDIO INSIGHTS EXTRACTOR =====================================
class AudioInsightsExtractor:
    """Extracts audio features from video"""
    def __init__(self):
        self.ok = audio_deps_ok
        print(f"INFO: AudioInsightsExtractor initialized (Libs OK: {self.ok}).")

    def extract_audio_segment(self, vid_path, aud_path="temp_audio.wav"):
        """Extract audio from video"""
        if not self.ok or VideoFileClip is None:
            print("ERROR: Cannot save audio segment.")
            return False
        print(f"INFO: Saving audio from {os.path.basename(vid_path)}...")
        ok = False
        try:
            with VideoFileClip(vid_path) as clip:
                if clip.audio:
                    clip.audio.write_audiofile(aud_path, codec='pcm_s16le', logger=None)
                    ok = True
                else:
                    print("INFO: No audio track.")
                    ok = False
            if ok:
                print(f"INFO: Audio saved to {aud_path}")
            return ok
        except Exception as e:
            print(f"ERROR saving audio: {e}")
            return False

    def analyze_audio_features(self, aud_path, fps, frames, rms_thresh=0.3, mfcc_pct=85):
        """Analyze audio features"""
        results = {'rms': [0.0] * frames, 'mfcc': [0.0] * frames, 'ml_event': [0.0] * frames}
        if not self.ok or not os.path.exists(aud_path):
            return results
        print(f"INFO: Analyzing audio features in {os.path.basename(aud_path)}...")
        try:
            y, sr = librosa.load(aud_path, sr=None)
            hop = max(1, int(sr / fps) if fps > 0 else 512)
        except Exception as e:
            print(f"ERR load audio: {e}")
            return results
        try:
            rms = librosa.feature.rms(y=y, hop_length=hop)[0]
            rn = (rms - np.min(rms)) / (np.max(rms) - np.min(rms)) if np.max(rms) > 1e-6 else np.zeros_like(rms)
            sc = [1.0 if rn[i] > rms_thresh else 0.0 for i in range(min(frames, len(rn)))]
            results['rms'] = (sc + [0.0] * frames)[:frames]
            print(f"INFO: RMS done. {int(sum(results['rms']))} frames>{rms_thresh}.")
        except Exception as e:
            print(f"ERROR RMS analysis: {e}")
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
            mnrg = np.max(np.abs(mfccs), axis=0)
        except Exception as e_mfcc:
            print(f"ERROR calculating MFCC: {e_mfcc}")
            mnrg = []
        if len(mnrg) > 0:
            try:
                thresh = np.percentile(mnrg, mfcc_pct)
                sc = [1.0 if mnrg[i] > thresh else 0.0 for i in range(min(frames, len(mnrg)))]
                results['mfcc'] = (sc + [0.0] * frames)[:frames]
                print(f"INFO: Spectral done. {int(sum(results['mfcc']))} frames>{mfcc_pct}th%.")
            except Exception as e:
                print(f"ERROR spectral scoring: {e}")
        return results

# ======================================== ENHANCED OBJECT TRACKER =====================================
class EnhancedObjectTracker:
    """Tracks objects across frames"""
    def __init__(self, max_d=30, max_dist=75):
        self.nID = 0
        self.live_tracks = OrderedDict()
        self.dis = OrderedDict()
        self.trkData = OrderedDict()
        self.maxD = max_d
        self.maxDist = max_dist
        self.appearance_extractor = None
        print("INFO: EnhancedObjectTracker initialized.")

    def _get_detection_centroid_bbox(self, lms, shape):
        if not lms:
            return None, None
        try:
            h, w = shape[:2]
            xc, yc = [], []
            if hasattr(lms, 'landmark'):
                xc = [lm.x * w for lm in lms.landmark if hasattr(lm, 'visibility') and lm.visibility > 0.1 and hasattr(lm, 'x')]
                yc = [lm.y * h for lm in lms.landmark if hasattr(lm, 'visibility') and lm.visibility > 0.1 and hasattr(lm, 'y')]
            else:
                print("WARN: landmarks object missing 'landmark' attribute.")
                return None, None
            if not xc or not yc:
                return None, None
            x0, x1 = int(min(xc)), int(max(xc))
            y0, y1 = int(min(yc)), int(max(yc))
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            box = (x0, y0, x1 - x0, y1 - y0)
            return (cx, cy), box
        except Exception as e:
            print(f"ERROR in _get_detection_centroid_bbox: {e}")
            print(traceback.format_exc())
            return None, None

    def update_tracks(self, pose_res_list, f_idx, frame):
        """Update object tracks"""
        cents = []
        bboxes = []
        feats = []
        shape = frame.shape
        for res in pose_res_list:
            c, b = self._get_detection_centroid_bbox(res.pose_landmarks if res else None, shape)
            if c:
                cents.append(c)
                bboxes.append(b)
                feats.append(None)
        if len(cents) == 0:
            rem_ids = []
            for oid in list(self.dis.keys()):
                self.dis[oid] += 1
                if self.dis[oid] > self.maxD:
                    rem_ids.append(oid)
            for oid in rem_ids:
                self._dereg_track(oid)
            return self.live_tracks, {}
        cur_bboxes = dict(zip(range(len(bboxes)), bboxes))
        if len(self.live_tracks) == 0:
            for i in range(len(cents)):
                self._reg_track(cents[i], f_idx)
        else:
            oids = list(self.live_tracks.keys())
            ocs = list(self.live_tracks.values())
            if ocs and cents:
                D = np.array([[np.linalg.norm(np.array(oc) - np.array(nc)) for nc in cents] for oc in ocs]) if ocs and cents else np.array([])
                if D.size == 0:
                    for i in range(len(cents)):
                        self._reg_track(cents[i], f_idx)
                    return self.live_tracks, cur_bboxes
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                usedR, usedC = set(), set()
                for r, c in zip(rows, cols):
                    if r in usedR or c in usedC:
                        continue
                    if D[r, c] < self.maxDist:
                        oid = oids[r]
                        self.live_tracks[oid] = cents[c]
                        self.trkData[oid].append((f_idx, cents[c]))
                        self.dis[oid] = 0
                        usedR.add(r)
                        usedC.add(c)
                unusedR = set(range(D.shape[0])) - usedR
                unusedC = set(range(len(cents))) - usedC
                for r in unusedR:
                    oid = oids[r]
                    self.dis[oid] += 1
                    if self.dis[oid] > self.maxD:
                        self._dereg_track(oid)
                for c in unusedC:
                    self._reg_track(cents[c], f_idx)
            else:  # No existing tracks
                for i in range(len(cents)):
                    self._reg_track(cents[i], f_idx)
        return self.live_tracks, cur_bboxes

    def _reg_track(self, c, fi):
        self.live_tracks[self.nID] = c
        self.dis[self.nID] = 0
        self.trkData[self.nID] = [(fi, c)]
        self.nID += 1

    def _dereg_track(self, oid):
        print(f"INFO: Deregister track {oid}")
        self.live_tracks.pop(oid, None)
        self.dis.pop(oid, None)

# ======================================== MASTER WORKFLOW COORDINATOR =====================================
class MasterWorkflowCoordinator:
    """Manages the full highlight generation workflow with improved video synthesis"""
    def __init__(self):
        logger.info("Initializing MasterWorkflowCoordinator...")
        self.pose_evaluator = AutonomousPoseAssessor()
        self.smoother = TemporalSmootherHMM()
        self.frame_pipeline = FramePipeline()
        self.audio_extractor = AudioInsightsExtractor()
        self.tracker = EnhancedObjectTracker()
        self.chunked_processor = ChunkedVideoProcessor(logger_name="Workflow_ChunkProcessor")
        self.temp_audio_path = "temp_master_audio.wav"
        self.output_dir = "starleague_engine_output"
        create_directory_if_needed(self.output_dir)
        create_directory_if_needed("models")
        logger.info("MasterWorkflowCoordinator initialized.")

    def _cleanup(self):
        """Cleanup coordinator-specific files (chunked processor handles its own)"""
        if os.path.exists(self.temp_audio_path):
            try:
                os.remove(self.temp_audio_path)
                logger.info(f"Cleaned up {self.temp_audio_path}")
            except:
                pass  # Ignore cleanup errors
        self.chunked_processor.cleanup()

    def _run_frame_analysis_stream(self, config):
        """Analyze video frames and yield results"""
        logger.info("--- Workflow: Frame Analysis Stream ---")
        vid_path = config['video_path']
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {vid_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_fr <= 0:
            raise ValueError(f"Invalid props (FPS:{fps}, Frames:{total_fr})")
        logger.info(f"Video: {total_fr} fr @ {fps:.2f} FPS")
        aud_ok = self.audio_extractor.extract_audio_segment(vid_path, self.temp_audio_path)
        audio_features = self.audio_extractor.analyze_audio_features(
            self.temp_audio_path, fps, total_fr, config['audio_rms_threshold'], config['audio_mfcc_percentile']
        ) if aud_ok else {'rms': [0.0] * total_fr, 'mfcc': [0.0] * total_fr, 'ml_event': [0.0] * total_fr}
        if config.get('enable_upsampling'):
            self.frame_pipeline.configure_upsampler(config['upsample_model'], config['upsample_scale'])
        self.tracker = EnhancedObjectTracker()
        proc_cnt = 0
        shape = None
        prev_gray = None
        start_t = time.time()
        with self.pose_evaluator.mp_pose_solution.Pose(
            static_image_mode=False,
            min_detection_confidence=config['confidence'],
            model_complexity=config['complexity']
        ) as pose:
            for idx in range(total_fr):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Read fail fr {idx+1}")
                    break
                if shape is None:
                    shape = frame.shape
                aud_rms_sc = audio_features['rms'][idx]
                aud_mfcc_sc = audio_features['mfcc'][idx]
                aud_ml_sc = audio_features['ml_event'][idx]
                if config['skip_interval'] > 1 and (idx + 1) % config['skip_interval'] != 0:
                    yield idx, 0.0, aud_rms_sc, aud_mfcc_sc, aud_ml_sc, 0.0, {}, {}
                    continue
                proc_cnt += 1
                cur_fr = self.frame_pipeline.process_frame(frame, config['enable_upsampling'])
                if cur_fr is None:
                    yield idx, 0.0, aud_rms_sc, aud_mfcc_sc, aud_ml_sc, 0.0, {}, {}
                    continue
                rgb = cv2.cvtColor(cur_fr, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                pose_data = pose.process(rgb)
                rgb.flags.writeable = True
                m_score = 0.0
                cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    try:
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        m_score = np.clip(np.mean(mag) / 15.0, 0.0, 1.0)
                    except:
                        pass
                prev_gray = cur_gray
                pose_score = self.pose_evaluator.evaluate_pose_autonomously(pose_data, motion_mag=m_score * 15.0)
                tr_objs, cur_bboxes = {}, {}
                if config.get('track_enabled', True):
                    tr_objs, cur_bboxes = self.tracker.update_tracks([pose_data] if pose_data else [], idx, frame)
                yield idx, pose_score, aud_rms_sc, aud_mfcc_sc, aud_ml_sc, m_score, tr_objs, cur_bboxes
                if proc_cnt % 30 == 0:
                    elps = time.time() - start_t
                    cfps = proc_cnt / elps if elps > 0 else 0
                    print(f" Processed {proc_cnt} ({idx+1} total). FPS:{cfps:.1f}", end='\r')
                if config['frame_limit'] and proc_cnt >= config['frame_limit']:
                    print(f"\nINFO: Frame limit.")
                    break
        cap.release()
        print(f"\n--- Frame Analysis Stream Finished ({proc_cnt} frames) ---")
        dur = time.time() - start_t
        print(f"Analysis Time: {dur:.2f}s")
        yield total_fr, proc_cnt, fps, self.tracker.trkData

    def execute_highlight_generation(self, config):
        """Generate highlight video using improved pipeline"""
        print("\n======= Master Workflow: Starting Highlight Generation =======")
        start_time = time.time()
        pipeline_ok = False
        try:
            frame_data = {}
            total_fr = 0
            proc_fr = 0
            fps = 30.0
            final_tracks = {}
            for res in self._run_frame_analysis_stream(config):
                if len(res) == 8:
                    idx, ps, ar, am, aml, ms, tr, bb = res
                    frame_data[idx] = {'p': ps, 'r': ar, 'm': am, 'ml': aml, 'o': ms}
                elif len(res) == 4:
                    total_fr, proc_fr, fps, final_tracks = res
                    break
            if total_fr == 0 or not frame_data:
                raise ValueError("Frame analysis failed.")

            print("\n--- Workflow: Scoring & Smoothing ---")
            combined_scores = []
            pose_confidences = []
            raw_binary_labels = []
            w = config['weights']
            INTEREST_THRESHOLD = 0.7
            print(f"INFO: Using combined score interest threshold: {INTEREST_THRESHOLD}")
            for i in range(total_fr):
                sc = frame_data.get(i, {'p': 0, 'r': 0, 'm': 0, 'ml': 0, 'o': 0})
                score = (sc['p'] * w['pose'] + sc['r'] * w['audio_rms'] + sc['m'] * w['audio_mfcc'] +
                         sc['o'] * w['motion'] + sc['ml'] * w.get('audio_ml', 0.0))
                combined_scores.append(score)
                pose_confidences.append(sc['p'])
                raw_binary_labels.append(1 if score >= INTEREST_THRESHOLD else 0)
            print(f"INFO: Raw interesting frames (before HMM): {sum(raw_binary_labels)}")
            if sum(raw_binary_labels) == 0:
                display_user_alert("No moments met the initial interest threshold.")
                return False

            smoothing_radius = max(1, min(30, int(config['padding_seconds'] * fps)))
            print(f"INFO: HMM smoothing (Radius Param: {smoothing_radius})...")
            final_lbls = self.smoother.smooth_sequence(raw_binary_labels, pose_confidences, smoothing_radius, use_dynamic_transitions=False)
            final_count = sum(final_lbls)
            print(f"INFO: Final frames for inclusion (after HMM): {final_count}")
            if final_count == 0:
                display_user_alert("No highlights generated after smoothing.")
                return False

            intervals = []
            start_t = None
            frame_dur = 1.0 / fps
            for i, lbl in enumerate(final_lbls):
                t = i * frame_dur
                if lbl == 1 and start_t is None:
                    start_t = t
                elif lbl == 0 and start_t is not None:
                    intervals.append((start_t, t))
                    start_t = None
            if start_t is not None:
                intervals.append((start_t, total_fr * frame_dur))
            print(f"INFO: Identified {len(intervals)} intervals.")
            if not intervals:
                display_user_alert("Error: Could not identify time intervals from smoothed labels.")
                return False

            print("\n--- Workflow: Building Final Reel (using Enhanced Pipeline) ---")
            out_dir = config.get('output_dir', self.output_dir)
            out_path = os.path.join(out_dir, os.path.basename(config['output_filename']))
            print(f"Output: {out_path}")
            output_params = {
                'codec': config.get('codec', 'libx264'),
                'audio_codec': config.get('audio_codec', 'aac'),
                'remove_temp': True,
                'threads': config.get('threads', self.chunked_processor.threads),
                'logger': 'bar'
            }

            def progress_callback(progress):
                if 'prog_var' in globals():
                    prog_var.set(progress)
                    app.update_idletasks()

            pipeline_ok = self.chunked_processor.process_highlight_generation(
                input_path=config['video_path'],
                output_path=out_path,
                intervals=intervals,
                output_params=output_params,
                progress_callback=progress_callback
            )

            if pipeline_ok:
                success_msg = f"Highlight video generated!\nSaved to: {out_path}"
                print(success_msg)
                display_user_alert(success_msg)
                if config.get('notify_email'):
                    send_email(
                        "Highlight Generation Completed",
                        f"Highlight video generated successfully.\nSaved to: {out_path}",
                        config['notify_email'],
                        config.get('from_email', 'sender@example.com'),
                        config.get('smtp_server', 'smtp.example.com'),
                        config.get('smtp_port', 587),
                        config.get('smtp_user', 'user'),
                        config.get('smtp_password', 'pass')
                    )
            else:
                display_user_alert("Highlight generation failed during chunk processing/writing.")
                if config.get('notify_email'):
                    send_email(
                        "Highlight Generation Failed",
                        "Highlight generation failed during chunk processing/writing.",
                        config['notify_email'],
                        config.get('from_email', 'sender@example.com'),
                        config.get('smtp_server', 'smtp.example.com'),
                        config.get('smtp_port', 587),
                        config.get('smtp_user', 'user'),
                        config.get('smtp_password', 'pass')
                    )

        except Exception as e:
            err_msg = f"CRITICAL WORKFLOW ERROR: {e}"
            print(f"{err_msg}\n{traceback.format_exc()}")
            display_user_alert(err_msg)
            if config.get('notify_email'):
                send_email(
                    "Highlight Generation Failed",
                    f"Critical workflow error: {str(e)}",
                    config['notify_email'],
                    config.get('from_email', 'sender@example.com'),
                    config.get('smtp_server', 'smtp.example.com'),
                    config.get('smtp_port', 587),
                    config.get('smtp_user', 'user'),
                    config.get('smtp_password', 'pass')
                )
            pipeline_ok = False
        finally:
            self._cleanup()
            duration = time.time() - start_time
            print(f"======= Workflow Total Time: {duration:.2f}s ========")
            return pipeline_ok

# ======================================== LICENSE VALIDATION (DISABLED) =====================================
def validate_product_license(uses_per_year=1000):
    print("--- Skipping Product License Validation (DISABLED) ---")
    return True

# ======================================== GRAPHICAL USER INTERFACE ========================================
def launch_user_interface():
    """Launch the GUI for the StarLeague Highlight Engine"""
    logger.info("Launching User Interface...")

    # Check for display environment (for Linux compatibility)
    if not os.environ.get("DISPLAY"):
        logger.warning("DISPLAY environment variable not set. GUI may not appear on some systems.")

    # Verify Tkinter functionality
    try:
        root_test = tkinter.Tk()
        root_test.destroy()
        logger.info("Tkinter root test successful.")
    except Exception as e:
        logger.error(f"Tkinter initialization failed: {e}")
        display_user_alert("CRITICAL ERROR: Tkinter failed to initialize.\nCannot launch GUI.")
        return

    # Configure CustomTkinter appearance
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")

    # Initialize main window
    app = customtkinter.CTk()
    app.title("StarLeague Highlight Engine v3.8.8")
    app.geometry("600x700")
    app.resizable(False, False)

    # Center the window
    app.update_idletasks()
    ws, hs = app.winfo_screenwidth(), app.winfo_screenheight()
    x, y = (ws - 600) // 2, (hs - 700) // 2
    app.geometry(f"600x700+{x}+{y}")
    app.attributes('-topmost', True)
    app.focus_force()

    # Input variables
    video_path_var = tkinter.StringVar()
    output_filename_var = tkinter.StringVar(value="highlight_output.mp4")
    padding_var = tkinter.StringVar(value="2.0")
    confidence_var = tkinter.StringVar(value="0.5")
    complexity_var = tkinter.StringVar(value="1")
    skip_var = tkinter.StringVar(value="1")
    frame_limit_var = tkinter.StringVar(value="0")
    upsample_var = tkinter.BooleanVar(value=False)
    upsample_model_var = tkinter.StringVar(value="edsr")
    upsample_scale_var = tkinter.StringVar(value="2")
    audio_rms_var = tkinter.StringVar(value="0.3")
    audio_mfcc_var = tkinter.StringVar(value="85")
    notify_email_var = tkinter.StringVar(value="")
    pipeline_var = tkinter.StringVar(value="FFMPEG" if not USE_OPENCV else "OpenCV")
    global prog_var
    prog_var = tkinter.DoubleVar(value=0.0)

    # Flag to prevent GUI updates after closure
    window_closed = False

    def on_closing():
        """Handle window closure"""
        nonlocal window_closed
        window_closed = True
        app.destroy()
        logger.info("GUI window closed.")

    app.protocol("WM_DELETE_WINDOW", on_closing)

    # Main frame
    frame = customtkinter.CTkFrame(app, corner_radius=10)
    frame.pack(pady=20, padx=20, fill="both", expand=True)

    # Title
    customtkinter.CTkLabel(frame, text="StarLeague Highlight Engine", font=("Arial", 20, "bold")).pack(pady=10)

    # Video selection
    def select_video():
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            video_path_var.set(path)
            video_label.configure(text=f"Selected: {os.path.basename(path)}")

    video_frame = customtkinter.CTkFrame(frame)
    video_frame.pack(fill="x", padx=10, pady=5)
    customtkinter.CTkButton(video_frame, text="Select Video", command=select_video).pack(side="left", padx=5)
    video_label = customtkinter.CTkLabel(video_frame, text="No video selected")
    video_label.pack(side="left", padx=5)

    # Output filename
    output_frame = customtkinter.CTkFrame(frame)
    output_frame.pack(fill="x", padx=10, pady=5)
    customtkinter.CTkLabel(output_frame, text="Output Filename:").pack(side="left", padx=5)
    customtkinter.CTkEntry(output_frame, textvariable=output_filename_var).pack(side="left", fill="x", expand=True, padx=5)

    # Pipeline selection
    pipeline_frame = customtkinter.CTkFrame(frame)
    pipeline_frame.pack(fill="x", padx=10, pady=5)
    customtkinter.CTkLabel(pipeline_frame, text="Pipeline:").pack(side="left", padx=5)
    customtkinter.CTkOptionMenu(pipeline_frame, variable=pipeline_var, values=["FFMPEG", "OpenCV"]).pack(side="left", padx=5)

    # Parameters frame
    params_frame = customtkinter.CTkFrame(frame)
    params_frame.pack(fill="x", padx=10, pady=5)

    # Parameter inputs
    params = [
        ("Padding (s):", padding_var, 0),
        ("Confidence:", confidence_var, 1),
        ("Model Complexity:", complexity_var, 2, ["0", "1", "2"]),
        ("Skip Interval:", skip_var, 3),
        ("Frame Limit (0=unlimited):", frame_limit_var, 4),
        ("Audio RMS Threshold:", audio_rms_var, 5),
        ("Audio MFCC Percentile:", audio_mfcc_var, 6)
    ]

    for label_text, var, row, values in [(p[0], p[1], p[2], p[4] if len(p) > 4 else None) for p in params]:
        customtkinter.CTkLabel(params_frame, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky="w")
        if values:
            customtkinter.CTkOptionMenu(params_frame, variable=var, values=values).grid(row=row, column=1, padx=5, pady=2)
        else:
            customtkinter.CTkEntry(params_frame, textvariable=var, width=100).grid(row=row, column=1, padx=5, pady=2)

    # Upsampling options
    upsample_frame = customtkinter.CTkFrame(frame)
    upsample_frame.pack(fill="x", padx=10, pady=5)
    customtkinter.CTkCheckBox(upsample_frame, text="Enable Upsampling", variable=upsample_var).pack(side="left", padx=5)
    customtkinter.CTkLabel(upsample_frame, text="Model:").pack(side="left", padx=5)
    customtkinter.CTkOptionMenu(upsample_frame, variable=upsample_model_var, values=["edsr"]).pack(side="left", padx=5)
    customtkinter.CTkLabel(upsample_frame, text="Scale:").pack(side="left", padx=5)
    customtkinter.CTkOptionMenu(upsample_frame, variable=upsample_scale_var, values=["2", "3", "4"]).pack(side="left", padx=5)

    # Email notification
    email_frame = customtkinter.CTkFrame(frame)
    email_frame.pack(fill="x", padx=10, pady=5)
    customtkinter.CTkLabel(email_frame, text="Notify Email:").pack(side="left", padx=5)
    customtkinter.CTkEntry(email_frame, textvariable=notify_email_var).pack(side="left", fill="x", expand=True, padx=5)

    # Progress bar
    progress_frame = customtkinter.CTkFrame(frame)
    progress_frame.pack(fill="x", padx=10, pady=10)
    customtkinter.CTkLabel(progress_frame, text="Progress:").pack(side="left", padx=5)
    customtkinter.CTkProgressBar(progress_frame, variable=prog_var).pack(side="left", fill="x", expand=True, padx=5)

    # Run button callback
    def run_button_callback():
        if window_closed:
            logger.info("Run button clicked, but window is closed. Ignoring.")
            return

        video_path = video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            display_user_alert("Please select a valid video file.")
            return

        try:
            config = {
                'video_path': video_path,
                'output_filename': output_filename_var.get(),
                'output_dir': "starleague_engine_output",
                'padding_seconds': float(padding_var.get()),
                'confidence': float(confidence_var.get()),
                'complexity': int(complexity_var.get()),
                'skip_interval': int(skip_var.get()),
                'frame_limit': int(frame_limit_var.get()) or None,
                'enable_upsampling': upsample_var.get(),
                'upsample_model': upsample_model_var.get(),
                'upsample_scale': int(upsample_scale_var.get()),
                'audio_rms_threshold': float(audio_rms_var.get()),
                'audio_mfcc_percentile': float(audio_mfcc_var.get()),
                'weights': {'pose': 0.5, 'audio_rms': 0.2, 'audio_mfcc': 0.2, 'motion': 0.1, 'audio_ml': 0.0},
                'notify_email': notify_email_var.get().strip() or None,
                'codec': 'libx264',
                'audio_codec': 'aac',
                'threads': max(1, os.cpu_count() // 2)
            }
        except ValueError:
            display_user_alert("Invalid input values. Please check your parameters.")
            return

        # Update pipeline selection
        global USE_OPENCV
        USE_OPENCV = pipeline_var.get() == "OpenCV"
        logger.info(f"Pipeline set to {pipeline_var.get()} (USE_OPENCV={USE_OPENCV})")

        # Disable run button and reset progress
        run_button.configure(state="disabled")
        prog_var.set(0.0)

        def run_workflow():
            try:
                coordinator = MasterWorkflowCoordinator()
                success = coordinator.execute_highlight_generation(config)
                if not window_closed:
                    app.after(0, lambda: run_button.configure(state="normal"))
            except Exception as e:
                logger.error(f"Workflow thread error: {e}")
                if not window_closed:
                    app.after(0, lambda: display_user_alert(f"Workflow failed: {str(e)}"))
                    app.after(0, lambda: run_button.configure(state="normal"))

        # Run workflow in a separate thread
        threading.Thread(target=run_workflow).start()

    # Run button
    run_button = customtkinter.CTkButton(frame, text="Generate Highlights", command=run_button_callback)
    run_button.pack(pady=20)

    # Start the main loop
    try:
        app.mainloop()
        logger.info("GUI mainloop exited normally.")
    except Exception as e:
        logger.error(f"GUI mainloop error: {e}")
        if not window_closed:
            display_user_alert(f"GUI crashed: {str(e)}")

# ======================================== MAIN EXECUTION ========================================
if __name__ == "__main__":
    print("StarLeague Highlight Engine v3.8.8")
    print("===================================")
    if not validate_product_license():
        logger.error("License validation failed.")
        exit(1)
    launch_user_interface()