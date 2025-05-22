import cv2
import numpy as np
import mediapipe as mp
import sys

# Optional: Specify a video file path here, or leave as an empty string for webcam.
video_path = "/Users/blblackboyzeusackboyzeus/Downloads/pegasusEditorMacOS/output_videous_chef/final_renders/videous_chef_PhysicsMC_20250331_112126.mp4"  # <-- Replace with your video file path, or set to "" for webcam

# Initialize MediaPipe Face Mesh and Pose
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Capture Video Stream (use video file if provided, else webcam)
if video_path:
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()

if not ret:
    print("Failed to capture initial frame. Check your camera permissions or video file path.")
    cap.release()
    sys.exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_landmarks = None
flow = None  # Initialize flow variable

# Hue offset for color wheel rotation effect
hue_offset = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increase hue offset each frame (OpenCV hue range: 0-179)
    hue_offset = (hue_offset + 1) % 180

    # Convert frame to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Process Face Mesh and Pose detections
    face_results = face_mesh.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    landmarks = []

    # Extract face landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                landmarks.append((x, y))
    
    # Extract body (pose) landmarks
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            landmarks.append((x, y))
    
    # Compute Optical Flow if we have previous landmarks and current landmarks exist
    if prev_landmarks and landmarks:
        prev_pts = np.array(prev_landmarks, dtype=np.float32)
        curr_pts = np.array(landmarks, dtype=np.float32)
        flow, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, curr_pts, None, **lk_params)
        
        if flow is not None and flow.shape[0] == prev_pts.shape[0]:
            for i, (new, old) in enumerate(zip(flow, prev_pts)):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                # Draw a trail line for the motion (base color will be modulated later)
                cv2.line(frame, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (255, 105, 180), 1)
                # Draw a dot at the new landmark position
                cv2.circle(frame, (int(x_new), int(y_new)), 3, (173, 216, 230), -1)
        else:
            flow = np.zeros_like(prev_pts)
    
    # Update previous landmarks and frame for next iteration
    prev_landmarks = landmarks
    prev_gray = frame_gray.copy()

    # Apply a distortion effect (sinusoidal mapping)
    rows, cols, _ = frame.shape
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    distortion = np.sin(map_y / 20) * 5
    map_x = np.clip(map_x + distortion, 0, cols - 1).astype(np.float32)
    map_y = map_y.astype(np.float32)
    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Flow-Based Particle Trails: Compute glow intensity for each landmark based on motion magnitude
    if flow is not None and len(landmarks) > 0 and flow.shape[0] == len(landmarks):
        magnitude = np.linalg.norm(flow - np.array(prev_pts), axis=1)
    else:
        magnitude = np.zeros(len(landmarks))
        
    for i, (x, y) in enumerate(landmarks):
        glow_intensity = min(255, int(magnitude[i] * 12))
        cv2.circle(frame, (x, y), 5, (glow_intensity, 50, 255), -1)

    # Convert frame to HSV, rotate the hue, then convert back to BGR
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame[..., 0] = (hsv_frame[..., 0].astype(int) + hue_offset) % 180
    frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

    cv2.imshow('Full-Body Rotating Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
