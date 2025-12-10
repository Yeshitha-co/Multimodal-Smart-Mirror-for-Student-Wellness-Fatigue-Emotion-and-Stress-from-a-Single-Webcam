# fatigue_module.py

import cv2
import mediapipe as mp
import numpy as np
import math

# ---------- Mediapipe FaceMesh setup ----------

mp_face_mesh_solution = mp.solutions.face_mesh
face_mesh = mp_face_mesh_solution.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

# ---------- Landmark indices ----------

RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]

MOUTH_LEFT_CORNER  = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_UPPER_MID    = 0
MOUTH_LOWER_MID    = 17

NOSE_TIP        = 1
CHIN            = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER= 33

def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

# ---------- Get landmarks from a single video frame ----------

def get_landmarks_from_frame(frame_bgr):
    """
    frame_bgr: OpenCV BGR frame
    returns: landmarks array of shape (478, 2) or None
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0]
    points = np.array([[p.x * w, p.y * h] for p in lm.landmark])
    return points

# ---------- EAR / MAR / head pose ----------

def eye_aspect_ratio(landmarks, eye_idx):
    p1 = landmarks[eye_idx[0]]
    p2 = landmarks[eye_idx[1]]
    p3 = landmarks[eye_idx[2]]
    p4 = landmarks[eye_idx[3]]
    p5 = landmarks[eye_idx[4]]
    p6 = landmarks[eye_idx[5]]

    d1 = euclidean_dist(p2, p6)   # vertical
    d2 = euclidean_dist(p3, p5)   # vertical
    d3 = euclidean_dist(p1, p4)   # horizontal

    return (d1 + d2) / (2.0 * d3 + 1e-6)

def mouth_aspect_ratio(landmarks):
    left_corner  = landmarks[MOUTH_LEFT_CORNER]
    right_corner = landmarks[MOUTH_RIGHT_CORNER]
    upper_mid    = landmarks[MOUTH_UPPER_MID]
    lower_mid    = landmarks[MOUTH_LOWER_MID]

    mouth_width  = euclidean_dist(left_corner, right_corner)
    mouth_height = euclidean_dist(upper_mid, lower_mid)

    return mouth_height / (mouth_width + 1e-6)

def yawn_flag_from_mar(mar, threshold=0.6):
    return int(mar > threshold)

def head_tilt_angle_deg(landmarks):
    left_eye  = landmarks[LEFT_EYE_CORNER]
    right_eye = landmarks[RIGHT_EYE_CORNER]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg < -90:
        angle_deg += 180
    elif angle_deg > 90:
        angle_deg -= 180
    return angle_deg

def head_pitch_proxy(landmarks):
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    nose_chin = euclidean_dist(nose, chin)

    left_eye  = landmarks[LEFT_EYE_CORNER]
    right_eye = landmarks[RIGHT_EYE_CORNER]
    eye_dist  = euclidean_dist(left_eye, right_eye)

    return nose_chin / (eye_dist + 1e-6)

# ---------- MAIN FEATURE FUNCTION ----------

def compute_fatigue_features_from_frame(frame_bgr):
    """
    Returns:
      feats: dict with 'ear', 'mar', 'yawn_flag', 'tilt_deg', 'pitch_ratio'
      landmarks: (478, 2) or None
    """
    lm = get_landmarks_from_frame(frame_bgr)
    if lm is None:
        return None, None

    left_ear  = eye_aspect_ratio(lm, LEFT_EYE_EAR)
    right_ear = eye_aspect_ratio(lm, RIGHT_EYE_EAR)
    ear       = 0.5 * (left_ear + right_ear)

    mar       = mouth_aspect_ratio(lm)
    yawn_flag = yawn_flag_from_mar(mar)
    tilt_deg  = head_tilt_angle_deg(lm)
    pitch_val = head_pitch_proxy(lm)

    feats = {
        "ear": ear,
        "mar": mar,
        "yawn_flag": yawn_flag,
        "tilt_deg": tilt_deg,
        "pitch_ratio": pitch_val,
    }
    return feats, lm

# ---------- For visualization in the mirror ----------

def draw_landmarks_on_frame(frame_bgr, landmarks, color=(0, 255, 0)):
    frame = frame_bgr.copy()
    if landmarks is not None:
        for (x, y) in landmarks.astype(int):
            cv2.circle(frame, (x, y), 1, color, -1)
    return frame
