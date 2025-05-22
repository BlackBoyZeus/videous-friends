import cv2
import numpy as np
import mediapipe as mp
import os
from collections import deque

# --- Configuration ---
# Base directory for videos (USE YOUR ACTUAL PATH)
base_dir = '/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/output_videous_chef/final_renders/'

# Video file name (relative to base_dir - USE YOUR ACTUAL FILENAME)
video_file = 'videous_chef_PhysicsMC_20250331_112126.mp4' # Example filename

# Construct full video path
video_path = os.path.join(base_dir, video_file)

# --- Effect Parameters ---
# History & Trails
MAX_HISTORY = 18       # Number of frames for the trail/reverb effect
TRAIL_START_ALPHA = 0.7 # Starting opacity/intensity for the newest trail points
TRAIL_END_ALPHA = 0.0   # Ending opacity/intensity (fully faded)
TRAIL_RADIUS = 2        # Base radius of trail points
TRAIL_BLUR_KERNEL = (5, 5) # Kernel size for blurring trails (0,0 to disable)

# Tint & Mask
GOLD_TINT_COLOR = (30, 165, 210) # BGR: Slightly adjusted golden/bronze hue
TINT_STRENGTH = 0.40    # Blending strength (0.0 to 1.0)
MASK_BLUR_KERNEL = (15, 15) # Kernel size for softening mask edges (0,0 to disable)
SEGMENTATION_THRESHOLD = 0.6 # Confidence threshold for pose segmentation mask (0.0 to 1.0)

# Optical Flow
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# --- Initialization ---
print("Initializing...")

# Extract video name for output file
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Create the output directory if it doesn't exist
output_dir_name = 'pixel_sense_olympus_output_refined' # New output folder name
output_dir = os.path.join(base_dir, output_dir_name)
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        exit()

# Construct output video path
output_filename = f"{video_name}_olympus_aura_refined.mp4" # Refined filename
output_path = os.path.join(output_dir, output_filename)
print(f"Output video will be saved to: {output_path}")

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    print("Warning: Could not read FPS accurately. Setting to 30.")
    fps = 30.0
print(f"Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS")

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    print(f"Error initializing VideoWriter for path: {output_path}")
    cap.release()
    exit()

# Initialize Mediapipe Face Mesh and Pose (with Segmentation enabled)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True) # <<< Enable Segmentation

# Initialize variables for Optical Flow and History
prev_gray = None
prev_face_points = np.empty((0, 1, 2), dtype=np.float32)
prev_pose_points = np.empty((0, 1, 2), dtype=np.float32)

face_history = deque(maxlen=MAX_HISTORY)
pose_history = deque(maxlen=MAX_HISTORY)

frame_number = 0
print("Initialization complete. Starting video processing...")

# --- Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("\nEnd of video or error reading frame.")
        break

    frame_number += 1
    if frame_number % 30 == 0:
        print(f"Processing frame {frame_number}...")

    overlay_trails = np.zeros_like(frame, dtype=np.uint8)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Landmark Detection & Segmentation ---
    face_results = mp_face_mesh.process(rgb_frame)
    pose_results = mp_pose.process(rgb_frame)

    # Extract current points
    current_face_points_list = []
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            points = np.array([[lm.x * frame_width, lm.y * frame_height]
                               for lm in face_landmarks.landmark if hasattr(lm, 'x')], dtype=np.float32)
            if points.size > 0:
                current_face_points_list.append(points.reshape(-1, 1, 2))
    current_face_points = np.concatenate(current_face_points_list, axis=0) if current_face_points_list else np.empty((0, 1, 2), dtype=np.float32)

    current_pose_points = np.empty((0, 1, 2), dtype=np.float32)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        pose_pts = np.array([[lm.x * frame_width, lm.y * frame_height]
                             for lm in landmarks if hasattr(lm, 'x')], dtype=np.float32)
        if pose_pts.size > 0:
            current_pose_points = pose_pts.reshape(-1, 1, 2)

    # --- Optical Flow Calculation ---
    tracked_face_points = np.empty((0, 1, 2), dtype=np.float32)
    tracked_pose_points = np.empty((0, 1, 2), dtype=np.float32)
    if prev_gray is not None:
        if prev_face_points.size > 0:
            try:
                new_face_points, status_face, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_face_points, None, **lk_params)
                if new_face_points is not None: tracked_face_points = new_face_points[status_face.flatten() == 1].reshape(-1, 1, 2)
            except cv2.error: tracked_face_points = np.empty((0, 1, 2), dtype=np.float32)
        if prev_pose_points.size > 0:
            try:
                new_pose_points, status_pose, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_pose_points, None, **lk_params)
                if new_pose_points is not None: tracked_pose_points = new_pose_points[status_pose.flatten() == 1].reshape(-1, 1, 2)
            except cv2.error: tracked_pose_points = np.empty((0, 1, 2), dtype=np.float32)

    # --- Update History ---
    face_points_to_add = tracked_face_points if tracked_face_points.size > 0 else current_face_points
    pose_points_to_add = tracked_pose_points if tracked_pose_points.size > 0 else current_pose_points
    if face_points_to_add.size > 0: face_history.append(face_points_to_add.reshape(-1, 2))
    if pose_points_to_add.size > 0: pose_history.append(pose_points_to_add.reshape(-1, 2))

    # --- Render Effects ---

    # 1. Create Combined Mask (Face Hull + Pose Segmentation)
    # Start with an empty mask
    combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Add Face Convex Hulls
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            points = np.array([[int(lm.x * frame_width), int(lm.y * frame_height)]
                               for lm in face_landmarks.landmark if hasattr(lm, 'x')], dtype=np.int32)
            if len(points) > 2:
                try:
                    hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(combined_mask, hull, 255) # Add hull to combined mask
                except Exception as e: print(f"Frame {frame_number}: Warning - Face hull error: {e}")

    # Add Pose Segmentation Mask
    if pose_results.segmentation_mask is not None:
        try:
            # Create a binary mask from the segmentation output based on the threshold
            seg_mask_condition = pose_results.segmentation_mask > SEGMENTATION_THRESHOLD
            seg_mask_uint8 = seg_mask_condition.astype(np.uint8) * 255
            # Combine the segmentation mask with the existing face hulls using bitwise OR
            combined_mask = cv2.bitwise_or(combined_mask, seg_mask_uint8)
        except Exception as e: print(f"Frame {frame_number}: Warning - Pose segmentation processing error: {e}")

    # 2. Soften the Combined Mask Edges (Optional)
    if MASK_BLUR_KERNEL[0] > 0 and MASK_BLUR_KERNEL[1] > 0:
        # Apply Gaussian blur to the mask itself
        tint_mask_final = cv2.GaussianBlur(combined_mask, MASK_BLUR_KERNEL, 0)
    else:
        tint_mask_final = combined_mask # Use the sharp mask if blur kernel is (0,0)


    # 3. Apply Golden Tint using the Softened Mask
    golden_layer = np.zeros_like(frame, dtype=np.uint8)
    golden_layer[:] = GOLD_TINT_COLOR
    blended_full = cv2.addWeighted(frame, 1.0 - TINT_STRENGTH, golden_layer, TINT_STRENGTH, 0)
    inverse_mask = cv2.bitwise_not(tint_mask_final) # Use the final (potentially blurred) mask
    original_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    tinted_foreground = cv2.bitwise_and(blended_full, blended_full, mask=tint_mask_final)
    frame_with_tint = cv2.add(original_background, tinted_foreground)


    # 4. Draw Trails/Aura on the Overlay
    num_face_hist = len(face_history)
    for i, points in enumerate(face_history):
        if points is not None and points.size > 0:
            progress = i / max(1, num_face_hist - 1) if num_face_hist > 1 else 1.0
            current_alpha = TRAIL_START_ALPHA + (TRAIL_END_ALPHA - TRAIL_START_ALPHA) * progress
            current_alpha = max(0.0, min(1.0, current_alpha))
            trail_color_face = (int(GOLD_TINT_COLOR[0] * current_alpha),
                                int(GOLD_TINT_COLOR[1] * current_alpha),
                                int(GOLD_TINT_COLOR[2] * current_alpha))
            for point in points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    cv2.circle(overlay_trails, (x, y), radius=TRAIL_RADIUS, color=trail_color_face, thickness=-1, lineType=cv2.LINE_AA)

    num_pose_hist = len(pose_history)
    for i, points in enumerate(pose_history):
         if points is not None and points.size > 0:
            progress = i / max(1, num_pose_hist - 1) if num_pose_hist > 1 else 1.0
            current_alpha = TRAIL_START_ALPHA + (TRAIL_END_ALPHA - TRAIL_START_ALPHA) * progress
            current_alpha = max(0.0, min(1.0, current_alpha))
            trail_color_pose = (int(GOLD_TINT_COLOR[0] * current_alpha * 0.8),
                                int(GOLD_TINT_COLOR[1] * current_alpha * 0.8),
                                int(GOLD_TINT_COLOR[2] * current_alpha * 0.8))
            for point in points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    cv2.circle(overlay_trails, (x, y), radius=TRAIL_RADIUS + 1, color=trail_color_pose, thickness=-1, lineType=cv2.LINE_AA)

    # 5. Blur the Trail Overlay (Optional)
    if TRAIL_BLUR_KERNEL[0] > 0 and TRAIL_BLUR_KERNEL[1] > 0:
        overlay_trails = cv2.GaussianBlur(overlay_trails, TRAIL_BLUR_KERNEL, 0)

    # 6. Combine Tinted Frame with Trails Overlay
    output_frame = cv2.add(frame_with_tint, overlay_trails)

    # --- Update Previous State ---
    prev_gray = gray_frame.copy()
    prev_face_points = face_points_to_add.reshape(-1, 1, 2)
    prev_pose_points = pose_points_to_add.reshape(-1, 1, 2)

    # --- Output and Display ---
    out.write(output_frame)
    cv2.imshow('Olympus Aura Effect - Refined', output_frame) # Updated window title

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nQuit key pressed.")
        break

# --- Cleanup ---
print("\nReleasing resources...")
cap.release()
out.release()
cv2.destroyAllWindows()
# Explicitly close mediapipe objects
if 'mp_face_mesh' in locals() and mp_face_mesh: mp_face_mesh.close()
if 'mp_pose' in locals() and mp_pose: mp_pose.close()
print(f"Processing finished. Output saved to: {output_path}")