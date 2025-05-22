# ========================================================================
#                       IMPORTS
# ========================================================================
import tkinter
from tkinter import filedialog
import customtkinter

import cv2
import math
import numpy as np # Added for vector operations
import time
# requests, json, datetime removed (no Gumroad)
import os

# --- Required Analysis Libraries ---
try:
    import librosa
    import librosa.display
except ImportError:
    print("ERROR: librosa library not found. Please install it: pip install librosa")
    exit()
try:
    import moviepy.editor as mp
except ImportError:
    print("ERROR: moviepy library not found. Please install it: pip install moviepy")
    print("       You might also need ffmpeg installed on your system.")
    exit()
try:
    import mediapipe as mp_solution # Use alias to avoid conflict with moviepy
except ImportError:
     print("ERROR: mediapipe library not found. Please install it: pip install mediapipe")
     exit()

# ========================================================================
#                       HELPER FUNCTIONS (UI Alert - Unchanged)
# ========================================================================
def tk_write(tk_string1):
    try:
        window = tkinter.Toplevel()
        window.title("StarLeague - Info")
        window.geometry('600x300')
        tk_string1 = str(tk_string1)
        text1 = tkinter.Label(window, text=tk_string1, wraplength=580, justify="left")
        text1.pack(pady=20, padx=20, expand=True, fill="both")
        window.grab_set()
        window.focus_set()
        window.wait_window()
    except Exception as e:
        print(f"Error creating tk_write window: {e}\nMessage was: {tk_string1}")

# ========================================================================
#                       AUDIO ANALYSIS UTILITIES (Unchanged)
# ========================================================================
class PegasusAudioUtils:
    # --- Keeping the class from the previous version ---
    def extract_audio(self, video_path, audio_output_path="temp_audio.wav"):
        # ... (identical to previous version) ...
        print(f"Attempting to extract audio from: {video_path}")
        try:
            if not os.path.exists(video_path):
                 raise FileNotFoundError(f"Video file not found: {video_path}")
            video_clip = mp.VideoFileClip(video_path)
            audio_clip = video_clip.audio
            if audio_clip is None:
                print("Warning: The video file appears to have no audio track.")
                video_clip.close()
                return None
            audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le', logger=None)
            audio_clip.close()
            video_clip.close()
            print(f"Audio successfully extracted to: {audio_output_path}")
            return audio_output_path
        except FileNotFoundError as fnf_err:
             tk_write(f"Error: Input video file not found.\n{fnf_err}")
             print(f"Error: {fnf_err}")
             return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = (f"Error during audio extraction: {e}\n\n"
                       f"- Ensure the video file is not corrupted.\n"
                       f"- Ensure 'ffmpeg' is installed and accessible in your system's PATH.\n"
                       f"- Check file permissions.")
            tk_write(err_msg)
            print(f"Audio Extraction Error: {err_msg}")
            if os.path.exists(audio_output_path):
                try: os.remove(audio_output_path)
                except: pass
            return None

    def analyze_audio(self, audio_path, target_sr=22050):
        # ... (identical to previous version) ...
        print(f"Analyzing audio file: {audio_path}")
        try:
            if not os.path.exists(audio_path):
                 raise FileNotFoundError(f"Temporary audio file not found: {audio_path}")
            y, sr = librosa.load(audio_path, sr=target_sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            tempo, beat_samples = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units='samples')
            beat_times = librosa.samples_to_time(beat_samples, sr=sr)
            print(f"Detected Tempo: {tempo:.2f} BPM, Found {len(beat_times)} beats.")
            hop_length = 512
            rms_energy = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)[0]
            print("Audio analysis complete.")
            return {
                "sr": sr, "tempo": tempo, "beat_samples": beat_samples,
                "beat_times": beat_times, "rms_energy": rms_energy,
                "audio_hop_length": hop_length
            }
        except FileNotFoundError as fnf_err:
             tk_write(f"Error: Audio analysis failed - file not found.\n{fnf_err}")
             print(f"Error: {fnf_err}")
             return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk_write(f"Error during audio analysis: {e}\n\nPlease ensure the audio file is valid.")
            print(f"Audio Analysis Error: {e}")
            return None
        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Removed temporary audio file: {audio_path}")
                except Exception as del_e:
                    print(f"Warning: Could not remove temporary audio file {audio_path}: {del_e}")

# ========================================================================
#          POSE/VISUAL UTILITIES (REVAMPED - ADVANCED FEATURES)
# ========================================================================
class PegasusPoseUtils:
    """
    Extracts advanced physiological/performance features from MediaPipe Pose landmarks.
    Focuses on energy, expansiveness, posture, and sharpness.
    """
    def __init__(self, visibility_threshold=0.5):
        """Initialize Pose utility class."""
        self.mp_pose = mp_solution.solutions.pose
        self.mp_drawing = mp_solution.solutions.drawing_utils
        self.mp_drawing_styles = mp_solution.solutions.drawing_styles
        self.visibility_threshold = visibility_threshold

        # Define keypoint sets for different features
        self.kinetic_keypoints = [
            self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        self.gesture_volume_keypoints = [
            self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            # self.mp_pose.PoseLandmark.NOSE # Include head if desired
        ]
        self.torso_keypoints = {
            "left_shoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            "right_shoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP,
            "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP,
        }

    def _get_landmark_pos(self, landmarks, landmark_enum):
        """Safely get landmark position (x, y, z) as numpy array if visible."""
        try:
            lm = landmarks.landmark[landmark_enum.value]
            if lm.visibility >= self.visibility_threshold:
                return np.array([lm.x, lm.y, lm.z])
            else:
                return None
        except (IndexError, AttributeError, KeyError):
            return None

    def _calculate_velocity(self, pos_prev, pos_curr, dt):
        """Calculate velocity vector."""
        if pos_prev is None or pos_curr is None or dt <= 1e-6:
            return np.array([0.0, 0.0, 0.0])
        return (pos_curr - pos_prev) / dt

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 between vectors p2->p1 and p2->p3."""
        if p1 is None or p2 is None or p3 is None: return None
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if mag1 * mag2 == 0: return None
        cos_theta = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def calculate_dynamic_features(self, pose_results_prev, pose_results_curr, pose_results_next, dt):
        """
        Calculates advanced performance features based on pose landmarks
        from three consecutive frames (previous, current, next).

        Args:
            pose_results_prev: MediaPipe Pose results from the previous frame.
            pose_results_curr: MediaPipe Pose results from the current frame.
            pose_results_next: MediaPipe Pose results from the next frame.
            dt: Time difference between frames (seconds).

        Returns:
            A dictionary containing calculated features for the current frame:
            - kinetic_energy_proxy: Sum of squared speeds of key joints.
            - gesture_volume: Volume of the bounding box around upper body gesture points.
            - postural_lean_angle: Forward/backward lean angle of the torso.
            - movement_jerk_proxy: Average magnitude of acceleration change (sharpness).
            - shoulder_hip_alignment: Vertical alignment factor (hip relative to shoulder).
        """
        features = {
            "kinetic_energy_proxy": 0.0,
            "gesture_volume": 0.0,
            "postural_lean_angle": 0.0, # 0 = upright, positive = forward lean
            "movement_jerk_proxy": 0.0,
            "shoulder_hip_alignment": 0.0, # 0 = hip vertically below shoulder, positive = hip behind
        }

        landmarks_prev = pose_results_prev.pose_landmarks if pose_results_prev else None
        landmarks_curr = pose_results_curr.pose_landmarks if pose_results_curr else None
        landmarks_next = pose_results_next.pose_landmarks if pose_results_next else None

        if not landmarks_curr or dt <= 1e-6: # Need current landmarks at minimum
            return features

        velocities_curr = {}
        accelerations_curr = {}
        num_kinetic_points = 0
        total_sq_speed = 0.0
        total_jerk_mag = 0.0
        num_jerk_points = 0

        # --- Calculate Velocities (Current) and Accelerations (Current) ---
        for kp_enum in self.kinetic_keypoints:
            pos_prev = self._get_landmark_pos(landmarks_prev, kp_enum)
            pos_curr = self._get_landmark_pos(landmarks_curr, kp_enum)
            pos_next = self._get_landmark_pos(landmarks_next, kp_enum)

            vel_curr = self._calculate_velocity(pos_prev, pos_curr, dt)
            vel_next = self._calculate_velocity(pos_curr, pos_next, dt) # Velocity between curr and next

            if pos_curr is not None: # If current point exists
                velocities_curr[kp_enum] = vel_curr
                speed_sq = np.sum(vel_curr**2)
                total_sq_speed += speed_sq
                num_kinetic_points += 1

                # Acceleration requires velocity at next step
                accel_curr = self._calculate_velocity(vel_curr, vel_next, dt) # Accel is change in velocity
                accelerations_curr[kp_enum] = accel_curr

                # Jerk requires acceleration at next step (needs 4 frames total, proxy needs 3)
                # Proxy: Use magnitude of acceleration difference between prev->curr and curr->next
                if pos_prev is not None and pos_next is not None: # Need all 3 points for accel diff
                     vel_prev = self._calculate_velocity(pos_prev-dt, pos_prev, dt) # Need hypothetical frame before prev? NO - use vel_curr vs vel_next
                     accel_prev = self._calculate_velocity(self._calculate_velocity(landmarks_prev-dt, pos_prev, dt), vel_curr, dt) # Getting too complex
                     # Simpler Jerk Proxy: Magnitude of current acceleration
                     jerk_proxy_mag = np.linalg.norm(accel_curr)
                     total_jerk_mag += jerk_proxy_mag
                     num_jerk_points += 1


        # Finalize Kinetic Energy and Jerk Proxies
        if num_kinetic_points > 0:
            # Use average squared speed or sum? Sum reflects total energy better.
            features["kinetic_energy_proxy"] = total_sq_speed # Higher value = more energy
            # Scale? Depends on expected range. TODO: Tune scaling
            features["kinetic_energy_proxy"] *= 10.0 # Arbitrary scaling

        if num_jerk_points > 0:
            features["movement_jerk_proxy"] = total_jerk_mag / num_jerk_points # Average jerk magnitude
            # Scale? TODO: Tune scaling
            features["movement_jerk_proxy"] *= 5.0 # Arbitrary scaling

        # --- Gesture Volume ---
        points_for_volume = []
        for kp_enum in self.gesture_volume_keypoints:
            pos = self._get_landmark_pos(landmarks_curr, kp_enum)
            if pos is not None:
                points_for_volume.append(pos)

        if len(points_for_volume) >= 2:
            points_np = np.array(points_for_volume)
            min_coords = np.min(points_np, axis=0)
            max_coords = np.max(points_np, axis=0)
            dimensions = max_coords - min_coords
            # Use 2D area or 3D volume? Area (X*Y) is simpler and often sufficient.
            volume = dimensions[0] * dimensions[1] # Area in normalized coordinates
            features["gesture_volume"] = volume * 100.0 # Scale for better range

        # --- Postural Lean and Alignment ---
        ls = self._get_landmark_pos(landmarks_curr, self.torso_keypoints["left_shoulder"])
        rs = self._get_landmark_pos(landmarks_curr, self.torso_keypoints["right_shoulder"])
        lh = self._get_landmark_pos(landmarks_curr, self.torso_keypoints["left_hip"])
        rh = self._get_landmark_pos(landmarks_curr, self.torso_keypoints["right_hip"])

        if ls is not None and rs is not None and lh is not None and rh is not None:
            shoulder_mid = (ls + rs) / 2.0
            hip_mid = (lh + rh) / 2.0

            # Lean Angle (using Y and Z coordinates - Z represents depth)
            # Vector from hip to shoulder
            torso_vec = shoulder_mid - hip_mid
            # Project torso_vec onto YZ plane (side view)
            torso_vec_yz = np.array([torso_vec[1], torso_vec[2]]) # Y, Z components
            # Vertical direction vector in YZ plane
            vertical_vec = np.array([1.0, 0.0]) # Points "down" in image coordinates (positive Y)

            dot_yz = np.dot(torso_vec_yz, vertical_vec)
            mag_torso_yz = np.linalg.norm(torso_vec_yz)
            mag_vertical = np.linalg.norm(vertical_vec)

            if mag_torso_yz * mag_vertical > 1e-6:
                cos_lean_angle = np.clip(dot_yz / (mag_torso_yz * mag_vertical), -1.0, 1.0)
                lean_angle_rad = np.arccos(cos_lean_angle)
                # Determine sign based on Z coordinate difference (depth)
                # If shoulder_mid[2] < hip_mid[2], it's leaning forward (positive angle)
                lean_angle_deg = np.degrees(lean_angle_rad)
                if shoulder_mid[2] < hip_mid[2]:
                     features["postural_lean_angle"] = lean_angle_deg
                else:
                     features["postural_lean_angle"] = -lean_angle_deg

            # Shoulder-Hip Vertical Alignment (using Y coordinates mainly)
            # Difference in Y coords relative to torso height
            dy = shoulder_mid[1] - hip_mid[1] # Positive if shoulder is lower (image coords)
            torso_height = np.linalg.norm(torso_vec)
            if torso_height > 1e-6:
                 # How much is hip Y coord offset from shoulder Y coord?
                 # Let's redefine: alignment = hip_y - shoulder_y. Positive means hip is lower.
                 alignment = hip_mid[1] - shoulder_mid[1]
                 features["shoulder_hip_alignment"] = alignment / torso_height # Normalize by torso height


        # Clamp features to reasonable ranges if needed (or handle during scoring)
        # features["kinetic_energy_proxy"] = min(features["kinetic_energy_proxy"], 100.0) # Example clamp
        # features["gesture_volume"] = min(features["gesture_volume"], 50.0) # Example clamp

        return features


    def drawLandmarksOnImage(self, imageInput, poseProcessingInput):
        # ... (identical to previous version) ...
        annotated_image = imageInput.copy()
        if poseProcessingInput and poseProcessingInput.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                poseProcessingInput.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

# ========================================================================
#         DYNAMIC SEGMENT SELECTOR (NEW CLASS - Replaces Smoothing)
# ========================================================================
class DynamicSegmentSelector:
    """
    Refines binary highlight labels based on context, scores, and dynamics.
    Replaces simple neighborhood modification with more intelligent segment handling.
    """
    def __init__(self, config, fps):
        self.fps = fps
        self.min_segment_len = int(config.get('min_highlight_duration_sec', 0.2) * fps)
        self.max_gap_len = int(config.get('max_gap_to_merge_sec', 0.3) * fps)
        self.merge_gap_score_thresh = config.get('merge_gap_score_threshold', 0.3) # Avg score in gap to allow merge
        self.trim_edge_score_thresh = config.get('trim_edge_score_threshold', 0.35) # Trim frames below this at edges
        # self.extend_decay_factor = config.get('extend_decay_factor', 0.8) # How much score can drop to extend
        # self.extend_max_frames = int(config.get('extend_max_sec', 0.15) * fps)

    def _find_segments(self, labels):
        """Find start and end indices of consecutive '1' segments."""
        segments = []
        start_idx = -1
        for i, label in enumerate(labels):
            if label == 1 and start_idx == -1:
                start_idx = i
            elif label == 0 and start_idx != -1:
                segments.append((start_idx, i)) # End index is exclusive
                start_idx = -1
        if start_idx != -1: # Handle segment ending at the last frame
            segments.append((start_idx, len(labels)))
        return segments

    def refine_segments(self, binary_labels, scores): # Add pose_features later if needed
        """
        Applies merging, trimming, and filtering to refine highlight segments.

        Args:
            binary_labels: Initial list of 0s and 1s based on score thresholding.
            scores: List of original scores (0-1) for each frame.
            # pose_features_list: List of feature dictionaries from PegasusPoseUtils (optional).

        Returns:
            Refined list of binary labels.
        """
        n = len(binary_labels)
        if n == 0: return []

        refined_labels = np.array(binary_labels) # Work with numpy array

        # --- Stage 1: Initial Segment Identification & Min Length Filter ---
        segments = self._find_segments(refined_labels)
        print(f"Initial segments found: {len(segments)}")

        # Apply minimum length filter first
        filtered_segments = []
        for start, end in segments:
            if (end - start) >= self.min_segment_len:
                filtered_segments.append((start, end))
            else:
                refined_labels[start:end] = 0 # Zero out segments that are too short
        segments = filtered_segments
        print(f"Segments after min duration filter ({self.min_segment_len} frames): {len(segments)}")

        if not segments: return refined_labels.tolist()

        # --- Stage 2: Merge Gaps ---
        merged_segments = []
        if not segments: return refined_labels.tolist()

        current_start, current_end = segments[0]

        for i in range(1, len(segments)):
            next_start, next_end = segments[i]
            gap_start = current_end
            gap_end = next_start
            gap_len = gap_end - gap_start

            # Check if gap is short enough to consider merging
            if 0 < gap_len <= self.max_gap_len:
                # Check average score within the gap
                gap_scores = scores[gap_start:gap_end]
                avg_gap_score = np.mean(gap_scores) if gap_scores else 0

                # If average score in gap is low enough (below initial threshold but not *too* low)
                if avg_gap_score < self.merge_gap_score_thresh: # Allow merge if gap isn't high energy
                    print(f"Merging gap between {current_end} and {next_start} (len={gap_len}, avg_score={avg_gap_score:.2f})")
                    # Merge by extending the current segment's end
                    current_end = next_end
                    # Fill the gap in the label array
                    refined_labels[gap_start:gap_end] = 1
                else:
                    # Gap is too significant, finalize the current segment and start new
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            else:
                # Gap is too long or non-existent, finalize current segment
                merged_segments.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        # Add the last segment
        merged_segments.append((current_start, current_end))
        segments = merged_segments
        print(f"Segments after merging: {len(segments)}")

        # --- Stage 3: Trim Low-Energy Edges ---
        # Create a final label array based on merged segments before trimming
        final_labels = np.zeros(n, dtype=int)
        for start, end in segments:
            final_labels[start:end] = 1

        trimmed_segments = []
        for start, end in segments:
            original_start, original_end = start, end
            # Trim start
            while start < end and scores[start] < self.trim_edge_score_thresh:
                start += 1
            # Trim end (iterate backwards from end-1)
            while end > start and scores[end - 1] < self.trim_edge_score_thresh:
                end -= 1

            # If trimming made the segment too short, discard it
            if (end - start) >= self.min_segment_len:
                trimmed_segments.append((start, end))
                # Update final_labels: zero out the trimmed portions
                if start > original_start: final_labels[original_start:start] = 0
                if end < original_end: final_labels[end:original_end] = 0
            else:
                 # Segment became too short after trimming, zero out the whole original merged segment
                 final_labels[original_start:original_end] = 0
                 print(f"Segment {original_start}-{original_end} discarded after trimming.")

        print(f"Segments after trimming: {len(trimmed_segments)}")
        print(f"Final highlight frames: {np.sum(final_labels)}")

        return final_labels.tolist()


# ========================================================================
#                      IMAGE UTILITIES (Unchanged)
# ========================================================================
class PegasusImageUtils:
    # ... (identical to previous version) ...
    def resizeTARGET(self, image, TARGET_HEIGHT=256, TARGET_WIDTH=256):
        h, w = image.shape[:2]
        if h == 0 or w == 0: return image
        scale = min(TARGET_HEIGHT / h, TARGET_WIDTH / w)
        if scale >= 1.0: return image
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ========================================================================
#                      MAIN ANALYSIS CLASS (UPDATED Integration)
# ========================================================================
class PegasusMain:
    def __init__(self):
        self.pegasusPoseUtils = PegasusPoseUtils() # Uses new advanced version
        self.pegasusImageUtils = PegasusImageUtils()
        self.pegasusAudioUtils = PegasusAudioUtils()
        # self.pegasusVideoUtils removed (replaced by DynamicSegmentSelector)
        self.mp_pose = mp_solution.solutions.pose # Keep instance

    def calculate_rap_highlight_score(self, audio_energy_frame, pose_features, config):
        """ Calculates score using audio and ADVANCED pose features. """
        weights = {
            'audio_energy': config.get('audio_weight', 0.5),
            'kinetic_proxy': config.get('kinetic_weight', 0.3),
            'gesture_volume': config.get('gesture_volume_weight', 0.1),
            'jerk_proxy': config.get('sharpness_weight', 0.1),
            # Add weights for lean, alignment etc. if desired
        }
        # --- Normalization (Needs careful tuning based on observed feature ranges) ---
        norm_audio = min(audio_energy_frame / config.get('norm_max_rms', 0.5), 1.0)
        norm_kinetic = min(pose_features.get('kinetic_energy_proxy', 0.0) / config.get('norm_max_kinetic', 50.0), 1.0)
        norm_volume = min(pose_features.get('gesture_volume', 0.0) / config.get('norm_max_volume', 30.0), 1.0)
        norm_jerk = min(pose_features.get('movement_jerk_proxy', 0.0) / config.get('norm_max_jerk', 10.0), 1.0)

        score = (norm_audio * weights['audio_energy'] +
                 norm_kinetic * weights['kinetic_proxy'] +
                 norm_volume * weights['gesture_volume'] +
                 norm_jerk * weights['jerk_proxy'])
        return max(0.0, min(score, 1.0))

    def generate_initial_labels(self, scores, audio_data, video_fps, config):
        """ Generates initial 0/1 labels using threshold and beat boost. """
        # --- Identical to previous generate_highlight_labels, renamed ---
        num_frames = len(scores)
        if num_frames == 0: return []
        score_threshold = config.get('score_threshold', 0.5)
        beat_boost = config.get('beat_boost', 0.1)
        boosted_scores = np.array(scores, dtype=float)
        beat_times = audio_data.get('beat_times', [])
        beat_frames_video = [int(round(t * video_fps)) for t in beat_times]
        boost_radius = int(config.get('beat_boost_radius_sec', 0.1) * video_fps)
        for beat_frame_idx in beat_frames_video:
            start_idx = max(0, beat_frame_idx - boost_radius)
            end_idx = min(num_frames, beat_frame_idx + boost_radius + 1)
            boosted_scores[start_idx:end_idx] = np.minimum(boosted_scores[start_idx:end_idx] + beat_boost, 1.0)
        labels = (boosted_scores >= score_threshold).astype(int)
        return labels.tolist()


    def analyzeVideo(self, videoPath, analysisConfig):
        """ Main analysis pipeline using advanced pose features and dynamic segment selection. """
        print("="*30 + "\nStarting Rap Video Analysis (Advanced v2.1)...\n" + "="*30)
        start_time = time.time()

        # --- Config & Initial Setup ---
        TARGET_HEIGHT = analysisConfig.get('resolution_height', 256)
        TARGET_WIDTH = analysisConfig.get('resolution_width', 256)
        drawLandmarks = analysisConfig.get('draw_landmarks_debug', False)

        # 1. Audio Analysis
        pid = os.getpid(); timestamp = int(time.time())
        temp_audio_path = f"temp_audio_{pid}_{timestamp}.wav"
        audio_path = self.pegasusAudioUtils.extract_audio(videoPath, temp_audio_path)
        if not audio_path: return []
        audio_data = self.pegasusAudioUtils.analyze_audio(audio_path)
        if not audio_data: return []

        # 2. Video Processing & Feature Extraction
        capture = cv2.VideoCapture(videoPath)
        if not capture.isOpened(): /* ... error handling ... */ return []
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps <= 0 or total_video_frames <= 0: /* ... error handling ... */ return []
        frame_time_diff = 1.0 / video_fps
        print(f"Video Properties: FPS={video_fps:.2f}, Total Frames={total_video_frames}")

        all_scores = []
        all_pose_features = [] # Store features for segment selection
        pose_results_buffer = [None, None, None] # Stores results for [prev, curr, next]
        frame_count = 0

        # Audio mapping setup (as before)
        audio_sr = audio_data['sr']; rms_energy = audio_data['rms_energy']
        audio_hop_length = audio_data['audio_hop_length']; num_audio_frames = len(rms_energy)
        def get_audio_index_for_video_frame(idx):
            time_sec = (idx + 0.5) * frame_time_diff
            audio_idx = int(round(time_sec * audio_sr / audio_hop_length))
            return max(0, min(audio_idx, num_audio_frames - 1))

        print("Processing video frames & extracting advanced features...")
        with self.mp_pose.Pose(
            static_image_mode=analysisConfig.get('static_mode', False),
            model_complexity=analysisConfig.get('model_complexity', 0),
            min_detection_confidence=analysisConfig.get('min_pose_confidence', 0.5),
            min_tracking_confidence=0.5) as pose_detector:

            while True:
                success, frame = capture.read()
                if not success: break

                # Process frame for pose
                image_resized = self.pegasusImageUtils.resizeTARGET(frame, TARGET_HEIGHT, TARGET_WIDTH)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                current_results = pose_detector.process(image_rgb)
                image_rgb.flags.writeable = True

                # Update results buffer
                pose_results_buffer.pop(0) # Remove oldest
                pose_results_buffer.append(current_results) # Add current

                # Need at least 2 frames (prev, curr) for basic features, 3 for jerk
                if frame_count >= 1: # Can calculate features based on buffer[0] and buffer[1]
                    pose_features = self.pegasusPoseUtils.calculate_dynamic_features(
                        pose_results_buffer[0], # Prev
                        pose_results_buffer[1], # Curr (The one we score)
                        pose_results_buffer[2], # Next (Needed for jerk/accel)
                        frame_time_diff
                    )
                    all_pose_features.append(pose_features) # Store features for frame 'curr'

                    # Get audio energy for 'curr' frame
                    audio_idx = get_audio_index_for_video_frame(frame_count - 1) # Index matches pose_features
                    current_audio_energy = rms_energy[audio_idx]

                    # Calculate score for 'curr' frame
                    current_score = self.calculate_rap_highlight_score(
                        current_audio_energy,
                        pose_features,
                        analysisConfig
                    )
                    all_scores.append(current_score)

                    # Debug Drawing for 'curr' frame
                    if drawLandmarks:
                       debug_image = self.pegasusPoseUtils.drawLandmarksOnImage(image_resized, pose_results_buffer[1])
                       # Display features on image
                       y_offset = 20
                       cv2.putText(debug_image, f"F:{frame_count-1} S:{current_score:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1); y_offset += 15
                       cv2.putText(debug_image, f"A:{current_audio_energy:.3f} K:{pose_features['kinetic_energy_proxy']:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1); y_offset += 15
                       cv2.putText(debug_image, f"V:{pose_features['gesture_volume']:.1f} J:{pose_features['movement_jerk_proxy']:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1); y_offset += 15
                       cv2.putText(debug_image, f"L:{pose_features['postural_lean_angle']:.1f} Al:{pose_features['shoulder_hip_alignment']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                       cv2.imshow("Debug - Pose Features & Score", debug_image)
                       if cv2.waitKey(1) & 0xFF == ord('q'): break

                # --- Progress Update ---
                if frame_count % 100 == 0 and frame_count > 0:
                     elapsed = time.time() - start_time
                     eta = (elapsed / frame_count) * (total_video_frames - frame_count)
                     print(f"  Processed frame {frame_count}/{total_video_frames} | ETA: {eta:.1f}s")
                frame_count += 1

        capture.release()
        if drawLandmarks: cv2.destroyAllWindows()
        print(f"Finished processing {frame_count} frames.")

        # Adjust score/feature list lengths if processing stopped early
        num_scored_frames = len(all_scores)
        if num_scored_frames == 0: print("No scores calculated."); return []
        print(f"Calculated scores/features for {num_scored_frames} frames.")

        # 3. Generate Initial Binary Labels
        print("Generating initial highlight labels...")
        initial_labels = self.generate_initial_labels(all_scores, audio_data, video_fps, analysisConfig)

        # 4. Refine Segments using Dynamic Selector
        print("Refining highlight segments dynamically...")
        segment_selector = DynamicSegmentSelector(analysisConfig, video_fps)
        final_labels = segment_selector.refine_segments(initial_labels, all_scores) # Pass scores, maybe features later

        end_time = time.time()
        print(f"Analysis pipeline completed in {end_time - start_time:.2f} seconds.")
        print("="*30)

        # Pad/truncate final_labels to match total_video_frames for consistency
        final_labels_adjusted = final_labels + [0] * (total_video_frames - len(final_labels))
        return final_labels_adjusted[:total_video_frames]

    # --- buildHighlightVideo (Unchanged from previous version) ---
    def buildHighlightVideo(self, videoPath, frameLabelList, outVideoPath):
        # ... (identical to previous version, including error checks and codec) ...
        print(f"Starting highlight video construction: {outVideoPath}")
        start_time = time.time()
        capture = cv2.VideoCapture(videoPath)
        if not capture.isOpened(): tk_write(f"Error opening video: {videoPath}"); return
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if fps <= 0 or size[0] <= 0 or size[1] <= 0: tk_write("Invalid video props"); capture.release(); return

        if len(frameLabelList) != total_frames: # Adjust label list length
             print(f"Warning: Label list length mismatch ({len(frameLabelList)} vs {total_frames}). Adjusting.")
             frameLabelList = (frameLabelList + [0] * total_frames)[:total_frames]

        num_highlight_frames = sum(frameLabelList)
        if num_highlight_frames == 0: tk_write("No highlight frames selected."); capture.release(); return
        print(f"Selecting {num_highlight_frames}/{total_frames} frames for output at {fps:.2f} FPS, size {size}.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            out = cv2.VideoWriter(outVideoPath, fourcc, float(fps), size)
            if not out.isOpened(): raise IOError("VideoWriter failed to open.")
        except Exception as e: tk_write(f"VideoWriter Error: {e}"); capture.release(); return

        frame_count, written_count = 0, 0
        while True:
            success, frame = capture.read();
            if not success: break
            try:
                if frameLabelList[frame_count] == 1: out.write(frame); written_count += 1
            except IndexError: break # Safeguard
            frame_count += 1
            if frame_count % 500 == 0: print(f"  Checked {frame_count}/{total_frames} frames...")

        capture.release(); out.release()
        print(f"Finished writing {written_count} frames to {outVideoPath} ({time.time() - start_time:.2f}s).")


    # --- runMain (Minor adjustment for clarity) ---
    def runMain(self, videoFilePath, analysisConfig, outputVideoPath):
        """ Main execution flow: Analyze, then Build. """
        finalFrameLabels = self.analyzeVideo(videoFilePath, analysisConfig)
        if finalFrameLabels and sum(finalFrameLabels) > 0:
            self.buildHighlightVideo(videoFilePath, finalFrameLabels, outputVideoPath)
        elif not finalFrameLabels: print("Analysis failed. Video not built.")
        else: print("No highlights found. Video not built.")


# ========================================================================
#                      APP INTERFACE (Updated Config Params)
# ========================================================================
def mainApp():
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    app = customtkinter.CTk()
    app.geometry("800x800") # Even taller for new params
    app.title("StarLeague - Rap Video Editor v2.1 (Advanced)")
    app.pack_propagate(False) # Prevent widgets from resizing window

    headerlabel = customtkinter.CTkLabel(master=app, text="StarLeague Rap Highlight Generator", text_font=("Arial", 28))
    headerlabel.pack(pady=(15, 10))

    # --- Scrollable Frame for Config ---
    scrollable_frame = customtkinter.CTkScrollableFrame(master=app, label_text="Analysis Configuration")
    scrollable_frame.pack(pady=10, padx=20, fill="x")
    # config_frame = scrollable_frame # Use inner frame provided by scrollable

    # Helper function (as before)
    def create_slider_row(parent, label_text, variable, from_, to_, steps, default_val, format_str="{:.2f}"):
        # ... (identical to previous version) ...
        row_frame = customtkinter.CTkFrame(master=parent, fg_color="transparent")
        row_frame.pack(fill="x", pady=5, padx=5)
        label = customtkinter.CTkLabel(master=row_frame, text=label_text, width=190, anchor="w") # Wider label
        label.pack(side="left", padx=(5, 5))
        slider = customtkinter.CTkSlider(master=row_frame, variable=variable, from_=from_, to=to_, number_of_steps=steps)
        slider.pack(side="left", fill="x", expand=True, padx=5)
        value_label = customtkinter.CTkLabel(master=row_frame, text=format_str.format(default_val), width=70, anchor="e")
        value_label.pack(side="right", padx=(5, 5))
        slider.configure(command=lambda v, lbl=value_label: lbl.configure(text=format_str.format(v)))
        return slider, value_label

    # -- UI Variables --
    audio_weight_var = tkinter.DoubleVar(value=0.5)
    kinetic_weight_var = tkinter.DoubleVar(value=0.3)
    gesture_vol_weight_var = tkinter.DoubleVar(value=0.1)
    sharpness_weight_var = tkinter.DoubleVar(value=0.1)
    threshold_var = tkinter.DoubleVar(value=0.40) # Score threshold
    beat_boost_var = tkinter.DoubleVar(value=0.15)
    min_duration_var = tkinter.DoubleVar(value=0.25) # Min highlight sec
    max_gap_var = tkinter.DoubleVar(value=0.3) # Max gap to merge sec
    merge_thresh_var = tkinter.DoubleVar(value=0.3) # Avg score in gap to allow merge
    trim_thresh_var = tkinter.DoubleVar(value=0.35) # Trim edges below this score
    complexity_var = tkinter.IntVar(value=1) # Default balanced

    # -- Create Sliders --
    create_slider_row(scrollable_frame, "Audio Energy Weight:", audio_weight_var, 0.0, 1.0, 20, 0.5)
    create_slider_row(scrollable_frame, "Kinetic Energy Weight:", kinetic_weight_var, 0.0, 1.0, 20, 0.3)
    create_slider_row(scrollable_frame, "Gesture Volume Weight:", gesture_vol_weight_var, 0.0, 1.0, 20, 0.1)
    create_slider_row(scrollable_frame, "Movement Sharpness Weight:", sharpness_weight_var, 0.0, 1.0, 20, 0.1)
    create_slider_row(scrollable_frame, "Highlight Score Threshold:", threshold_var, 0.1, 0.9, 40, 0.40)
    create_slider_row(scrollable_frame, "Beat Emphasis Boost:", beat_boost_var, 0.0, 0.5, 25, 0.15)
    create_slider_row(scrollable_frame, "Min Highlight Duration:", min_duration_var, 0.1, 2.0, 19, 0.25, format_str="{:.1f}s")
    create_slider_row(scrollable_frame, "Max Gap to Merge:", max_gap_var, 0.0, 1.0, 20, 0.3, format_str="{:.1f}s")
    create_slider_row(scrollable_frame, "Merge Gap Score Threshold:", merge_thresh_var, 0.0, 0.8, 40, 0.3)
    create_slider_row(scrollable_frame, "Trim Edge Score Threshold:", trim_thresh_var, 0.0, 0.8, 40, 0.35)

    # -- Model Complexity Radio ---
    # ... (identical radio button setup as previous version) ...
    complexity_frame = customtkinter.CTkFrame(master=scrollable_frame, fg_color="transparent")
    complexity_frame.pack(fill="x", pady=5, padx=5)
    complexity_label = customtkinter.CTkLabel(master=complexity_frame, text="Analysis Quality:", width=190, anchor="w")
    complexity_label.pack(side="left", padx=(5,5))
    radio_frame = customtkinter.CTkFrame(master=complexity_frame, fg_color="transparent")
    radio_frame.pack(side="left", padx=5)
    radio_fast = customtkinter.CTkRadioButton(master=radio_frame, text="Fast", variable=complexity_var, value=0)
    radio_balanced = customtkinter.CTkRadioButton(master=radio_frame, text="Balanced", variable=complexity_var, value=1)
    radio_accurate = customtkinter.CTkRadioButton(master=radio_frame, text="Accurate", variable=complexity_var, value=2)
    radio_fast.pack(side="left", padx=5); radio_balanced.pack(side="left", padx=5); radio_accurate.pack(side="left", padx=5)

    # --- Status Label ---
    status_label = customtkinter.CTkLabel(master=app, text="Ready.", text_font=("Arial", 12), anchor="w")
    status_label.pack(pady=(5, 5), padx=20, fill="x")

    # --- Action Button Logic ---
    def run_analysis_button_action():
        # ... (File dialog as before) ...
        filetypes = (('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv'), ('All files', '*.*'))
        videoFilename = filedialog.askopenfilename(title='Choose video', filetypes=filetypes)
        if not videoFilename: status_label.configure(text="Cancelled."); return

        base, ext = os.path.splitext(videoFilename)
        outputVideoName = f"{base}-RapHighlights_Adv{ext}"

        # --- Gather ALL config params ---
        analysisConfig = {
            'model_complexity': complexity_var.get(),
            'audio_weight': audio_weight_var.get(),
            'kinetic_weight': kinetic_weight_var.get(),
            'gesture_volume_weight': gesture_vol_weight_var.get(),
            'sharpness_weight': sharpness_weight_var.get(),
            'score_threshold': threshold_var.get(),
            'beat_boost': beat_boost_var.get(),
            'beat_boost_radius_sec': 0.1, # Keep fixed for now
            'min_highlight_duration_sec': min_duration_var.get(),
            'max_gap_to_merge_sec': max_gap_var.get(),
            'merge_gap_score_threshold': merge_thresh_var.get(),
            'trim_edge_score_threshold': trim_thresh_var.get(),

            # Fixed/Internal Params (can be made configurable too)
            'min_pose_confidence': 0.5, 'resolution_height': 256, 'resolution_width': 256,
            'norm_max_rms': 0.5, 'norm_max_kinetic': 50.0, 'norm_max_volume': 30.0, 'norm_max_jerk': 10.0,
            # 'draw_landmarks_debug': True, # Uncomment for debug visuals
        }

        status_label.configure(text=f"Starting analysis...\n{os.path.basename(videoFilename)}", text_color="yellow")
        app.update_idletasks()

        try:
            pegasusMain = PegasusMain()
            pegasusMain.runMain(videoFilename, analysisConfig, outputVideoName)
            successMessage = f"Processing complete!\nOutput: {outputVideoName}"
            status_label.configure(text=successMessage, text_color="light green")
        except Exception as e:
            import traceback; error_details = traceback.format_exc()
            print(f"!!! ERROR during runMain !!!\n{error_details}\n!!! -------- !!!")
            errMsg = f"An unexpected error occurred:\n{e}\n\nCheck console output for details."
            tk_write(errMsg)
            status_label.configure(text=f"Error during processing. See console.", text_color="red")
        app.update_idletasks()

    # --- Buttons (as before) ---
    button_frame = customtkinter.CTkFrame(master=app, fg_color="transparent")
    button_frame.pack(pady=10, padx=20, fill="x", side="bottom") # Move buttons to bottom

    run_button = customtkinter.CTkButton(master=button_frame, text="Select Video & Generate Highlights",
                                       fg_color="purple", hover_color="#E0B0FF",
                                       command=run_analysis_button_action, height=40, text_font=("Arial", 14))
    run_button.pack(pady=5, padx=50, fill="x")
    endButton = customtkinter.CTkButton(master=button_frame, text="Close Program",
                                      fg_color="#B00020", hover_color="#CF6679",
                                      command=app.destroy, height=30)
    endButton.pack(pady=(5, 10), padx=150, fill="x")

    # --- Footer (as before) ---
    footer_frame = customtkinter.CTkFrame(master=app, fg_color="transparent")
    footer_frame.pack(side="bottom", fill="x", pady=5, padx=20)
    faqLabel = customtkinter.CTkLabel(master=footer_frame, text="thepegasusai.com", text_font=("Arial", 10), anchor="e")
    faqLabel.pack(side="right")

    app.mainloop()


# ========================================================================
#                       APPLICATION ENTRY POINT (No License Check)
# ========================================================================
if __name__ == "__main__":
    print("Application starting (No License Check)...")
    try:
        mainApp() # Directly run the app
    except ImportError as e:
         tk_write(f"Missing Library Error:\n{e}\n\nPlease install requirements.")
         print(f"Startup Error: Missing library - {e}")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"!!! UNHANDLED STARTUP ERROR !!!\n{error_details}\n!!! --------- !!!")
        tk_write(f"A critical error occurred on startup:\n{e}\n\nCheck console.")
        print(f"Unhandled Startup Error: {e}")
    print("Application finished.")