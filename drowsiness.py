# 🔹 Step 1: Import Required Libraries
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import pygame

# 🔹 Step 2: Define EAR Calculation Functions
def euclidean_distance(pt1, pt2):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(eye):
    """Compute Eye Aspect Ratio (EAR) from 6 eye landmarks."""
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 🔹 Step 3: Alarm Function
def play_alarm():
    """Play an alarm sound using pygame."""
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# 🔹 Step 4: Define Eye Landmark Indices (from MediaPipe face mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 🔹 Step 5: Initialize MediaPipe and Webcam
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture(0)

# EAR Threshold and Drowsy Timing Settings
EAR_THRESHOLD = 0.25
DROWSY_TIME = 3  # seconds

start_time = None  # Start time for when eyes close
alarm_on = False   # Alarm flag

# 🔹 Step 6: Main Video Loop
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror image for selfie view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # Get left and right eye coordinates
        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

        # Draw eye landmarks on frame
        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        # Compute average EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Drowsiness Detection Logic
        if avg_ear < EAR_THRESHOLD:
            if start_time is None:
                start_time = time.time()
            elapsed = time.time() - start_time

            if elapsed > DROWSY_TIME:
                cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm).start()
        else:
            start_time = None
            alarm_on = False

        # Show EAR on frame
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show output frame
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔹 Step 7: Cleanup
cap.release()
cv2.destroyAllWindows()
