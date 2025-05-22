import cv2
import numpy as np
import mediapipe as mp

# Initialize Face Mesh and Optical Flow
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Capture Video Stream
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()

if not ret:
    print("Failed to capture initial frame. Check your camera permissions.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_landmarks = None
flow = None  # Initialize flow variable

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                landmarks.append((x, y))

    if prev_landmarks and landmarks:
        prev_pts = np.array(prev_landmarks, dtype=np.float32)
        curr_pts = np.array(landmarks, dtype=np.float32)
        flow, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, curr_pts, None, **lk_params)
        
        if flow is not None and flow.shape[0] == prev_pts.shape[0]:
            for i, (new, old) in enumerate(zip(flow, prev_pts)):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                # Draw a pink line for the trail
                cv2.line(frame, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (255, 105, 180), 1)
                # Draw light blue celestial dots
                cv2.circle(frame, (int(x_new), int(y_new)), 3, (173, 216, 230), -1)
        else:
            flow = np.zeros_like(prev_pts)
    
    # Update previous landmarks and frame for the next iteration
    prev_landmarks = landmarks
    prev_gray = frame_gray.copy()

    # Apply Celestial Aura Effect - Deep cosmic blue overlay
    celestial_overlay = np.full(frame.shape, (30, 0, 90), dtype=np.uint8)
    frame = cv2.addWeighted(frame, 0.85, celestial_overlay, 0.15, 0)

    # Flow-Based Particle Trails (if optical flow available)
    if flow is not None and len(landmarks) > 0 and flow.shape[0] == len(landmarks):
        magnitude = np.linalg.norm(flow - np.array(prev_pts), axis=1)
    else:
        magnitude = np.zeros(len(landmarks))
        
    for i, (x, y) in enumerate(landmarks):
        glow_intensity = min(255, int(magnitude[i] * 12))
        # Draw a pink-purple glow effect based on motion
        cv2.circle(frame, (x, y), 5, (glow_intensity, 50, 255), -1)

    # Mythological Distortion Effect using a sinusoidal mapping
    rows, cols, _ = frame.shape
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    distortion = np.sin(map_y / 20) * 5
    # Convert both map_x and map_y to float32 after applying distortion
    map_x = np.clip(map_x + distortion, 0, cols - 1).astype(np.float32)
    map_y = map_y.astype(np.float32)
    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Olympus Celestial Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
