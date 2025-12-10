# app.py  --- Streamlit Smart Mirror: Fatigue + Emotion (NovelEffNet-B3) + rPPG + RL

import cv2
import time
import numpy as np
import joblib
from collections import deque

import streamlit as st

import torch
import torch.nn as nn
import timm  # for EfficientNet-B3 backbone

from fatigue_module import (
    compute_fatigue_features_from_frame,
    draw_landmarks_on_frame,
)

# ============================================================
# 0. GLOBAL DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. FATIGUE MODEL (MLP on EAR / MAR / etc.)
# ============================================================

mlp_model = joblib.load("fatigue_mlp.joblib")
# Training convention: 0 = Drowsy, 1 = Non-drowsy
# This model was trained offline and saved as fatigue_mlp.joblib.


def mlp_drowsy_prob(feats):
    """
    feats: dict with keys 'ear', 'mar', 'yawn_flag', 'tilt_deg', 'pitch_ratio'
    returns:
        p_drowsy_scalar: float, assumed P(class0=drowsy)
        full_proba: np.array shape (2,) = [p(class0), p(class1)]
    """
    x = np.array([[feats["ear"],
                   feats["mar"],
                   feats["yawn_flag"],
                   feats["tilt_deg"],
                   feats["pitch_ratio"]]], dtype=float)
    proba = mlp_model.predict_proba(x)[0]  # [P(class0), P(class1)]
    return float(proba[0]), proba


# ============================================================
# 2. EMOTION MODEL (NovelEmotionModel w/ MicroExpEnhancer & Routing)
# ============================================================

# IMPORTANT: keep this order consistent with training label_to_idx
EMOTION_CLASSES = [
    "neutral", "happy", "sad", "angry",
    "surprise", "disgust", "fear", "tired"
]


class MicroExpEnhancer(nn.Module):
    def __init__(self, in_ch=3, factor=0.2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # residual micro-expression enhancement
        return x + self.factor * self.conv(x)


class DynamicEmotionRouting(nn.Module):
    def __init__(self, feat_dim=1536, hidden_dim=512):
        super().__init__()
        self.expert_upper = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.expert_lower = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Linear(feat_dim, 2)

    def forward(self, feat):
        # feat: [B, feat_dim]
        gate = torch.softmax(self.gate(feat), dim=-1)  # [B,2]
        up = self.expert_upper(feat)                   # [B,H]
        low = self.expert_lower(feat)                  # [B,H]
        out = gate[:, 0:1] * up + gate[:, 1:2] * low   # [B,H]
        return out, gate


class NovelEmotionModel(nn.Module):
    def __init__(self, num_emotions=8):
        super().__init__()
        self.micro = MicroExpEnhancer()
        # EfficientNet-B3 backbone, no final classifier (num_classes=0)
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0
        )
        # feature dimension for effnet-b3 is 1536
        self.routing = DynamicEmotionRouting(feat_dim=1536, hidden_dim=512)
        self.head_emotion = nn.Linear(512, num_emotions)
        self.head_valence = nn.Linear(512, 1)
        self.head_arousal = nn.Linear(512, 1)

    def forward(self, x):
        # x: [B,3,300,300]
        x = self.micro(x)
        feat = self.backbone(x)           # [B,1536]
        routed, gate = self.routing(feat) # [B,512], [B,2]

        return {
            "logits": self.head_emotion(routed),
            "valence": self.head_valence(routed),
            "arousal": self.head_arousal(routed),
            "gate": gate
        }


# Instantiate and load your trained weights
emotion_model = NovelEmotionModel(num_emotions=len(EMOTION_CLASSES)).to(device)
emotion_state = torch.load("emotion_model_novel_finetuned.pth",
                           map_location=device)
emotion_model.load_state_dict(emotion_state)
emotion_model.eval()


def emotion_predict_from_bgr(face_bgr):
    """
    Run NovelEmotionModel on a BGR face crop.
    Preprocessing matches training:
      - Resize to 300x300
      - Convert to RGB
      - Scale to [0,1]
      - ToTensor (C,H,W)
    Returns:
        label: str or None
        conf: float or None
        probs_np: np.array shape (8,) or None
    """
    if face_bgr is None or face_bgr.size == 0:
        return None, None, None

    # BGR -> RGB, resize to 300x300, scale to [0,1], CHW
    img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (300, 300))
    img = img_resized.astype(np.float32) / 255.0          # [H,W,C], 0–1
    img = np.transpose(img, (2, 0, 1))                    # [C,H,W]
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = emotion_model(img_tensor)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)

    probs_np = probs[0].detach().cpu().numpy()  # pure model probabilities

    idx = int(np.argmax(probs_np))
    max_conf = float(probs_np[idx])

    CONF_THRESH = 0.5
    if max_conf < CONF_THRESH:
        # low confidence – keep probabilities (for smoothing) but no hard label
        return None, None, probs_np

    label = EMOTION_CLASSES[idx]
    return label, max_conf, probs_np


def crop_face_from_landmarks(frame_bgr, landmarks, expand_ratio=1.0):
    if landmarks is None:
        return None

    h, w, _ = frame_bgr.shape
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]

    x_min = max(int(xs.min()), 0)
    x_max = min(int(xs.max()), w - 1)
    y_min = max(int(ys.min()), 0)
    y_max = min(int(ys.max()), h - 1)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    box_w = x_max - x_min
    box_h = y_max - y_min
    size = int(max(box_w, box_h) * expand_ratio)

    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame_bgr[y1:y2, x1:x2]


# ============================================================
# 3. rPPG / STRESS MODEL (UltraLightNet)
# ============================================================

RPPG_IMG_SIZE = 36
RPPG_FRAME_DEPTH = 180  # temporal window for rPPG


class UltraLightNet(nn.Module):
    """
    Extremely simple model - just enough to learn BPM (~20K params)
    """
    def __init__(self):
        super(UltraLightNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),  # 36 -> 18
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                            # 18 -> 9

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),                            # 9 -> 3
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        x = self.conv(x)
        x = x.reshape(B, T, -1)

        x = x.mean(dim=1)  # average over time
        bpm = self.fc(x)
        return bpm.squeeze(-1)


def calculate_stress_score(bpm, prediction_error=0.0):
    """
    Map BPM to stress score in [0,100].
    """
    # Heart-rate stress component
    if bpm < 60:
        hr_stress = 0.2
    elif bpm <= 80:
        hr_stress = (bpm - 60) / 60.0          # 0.0 -> ~0.33
    elif bpm <= 100:
        hr_stress = 0.33 + (bpm - 80) / 40.0   # 0.33 -> ~0.83
    else:
        hr_stress = 0.83 + min((bpm - 100) / 100.0, 0.17)  # up to 1.0

    # Prediction confidence penalty (at runtime usually 0)
    confidence_penalty = min(prediction_error / 20.0, 0.15)

    stress_score = (hr_stress + confidence_penalty) * 100.0
    return float(np.clip(stress_score, 0.0, 100.0))


def extract_rppg_frame(frame_bgr):
    """
    Approximate the training preprocessing:
      - center square crop of full frame
      - resize to 36x36
      - BGR -> RGB
      - scale to [0,1]
    Returns np.array [H,W,C] float32.
    """
    h, w = frame_bgr.shape[:2]
    size = min(h, w)
    y = (h - size) // 2
    x = (w - size) // 2
    crop = frame_bgr[y:y + size, x:x + size]

    crop = cv2.resize(crop, (RPPG_IMG_SIZE, RPPG_IMG_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    return crop


# Load rPPG model weights
rppg_model = UltraLightNet().to(device)
rppg_state = torch.load("best_model.pth", map_location=device)
rppg_model.load_state_dict(rppg_state)
rppg_model.eval()

# ============================================================
# 5. SIMPLE RL AGENT (Q-learning)
# ============================================================

class RLAgent:
    def __init__(self, num_actions=3, lr=0.1, gamma=0.9):
        self.Q = {}
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.last_state = None
        self.last_action = None

    def _to_key(self, state):
        emo, fat, stress = state
        stress_bin = int(stress // 20)
        return (emo, round(fat, 1), stress_bin)

    def get_action(self, state):
        key = self._to_key(state)
        if key not in self.Q:
            self.Q[key] = np.zeros(self.num_actions)
        action = int(np.argmax(self.Q[key]))
        self.last_state = key
        self.last_action = action
        return action

    def update(self, new_state, reward):
        if self.last_state is None:
            return
        new_key = self._to_key(new_state)
        if new_key not in self.Q:
            self.Q[new_key] = np.zeros(self.num_actions)
        old_val = self.Q[self.last_state][self.last_action]
        future = np.max(self.Q[new_key])
        self.Q[self.last_state][self.last_action] = old_val + \
            self.lr * (reward + self.gamma * future - old_val)


def compute_reward(prev_state, new_state):
    emo_prev, fat_prev, stress_prev = prev_state
    emo_new, fat_new, stress_new = new_state

    reward = 0

    # Emotion improvement score
    emo_rank = {
        "angry": 0, "sad": 0, "fear": 0, "disgust": 0,
        "neutral": 1,
        "happy": 2, "surprise": 2, "tired": 0
    }
    reward += (emo_rank.get(emo_new, 1) - emo_rank.get(emo_prev, 1)) * 2

    # Fatigue reduction
    reward += (fat_prev - fat_new) * 3

    # Stress reduction
    reward += (stress_prev - stress_new) * 0.1

    return reward


# ============================================================
# 6. STREAMLIT STATE & UI
# ============================================================

# Session state init
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "prob_history" not in st.session_state:
    st.session_state.prob_history = deque(maxlen=10)
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=10)
if "rppg_buffer" not in st.session_state:
    st.session_state.rppg_buffer = deque(maxlen=RPPG_FRAME_DEPTH)
if "stress_history" not in st.session_state:
    st.session_state.stress_history = deque(maxlen=10)
if "drowsy_streak" not in st.session_state:
    st.session_state.drowsy_streak = 0
if "last_bpm" not in st.session_state:
    st.session_state.last_bpm = None
if "last_stress" not in st.session_state:
    st.session_state.last_stress = None
if "rl_agent" not in st.session_state:
    st.session_state.rl_agent = RLAgent(num_actions=3)
if "prev_state" not in st.session_state:
    st.session_state.prev_state = None
if "last_intervention_time" not in st.session_state:
    st.session_state.last_intervention_time = 0.0
if "last_action" not in st.session_state:
    st.session_state.last_action = 0
if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0.0


def start_camera():
    st.session_state.run_camera = True


def stop_camera():
    st.session_state.run_camera = False


st.title("Smart Mirror Wellness Assistant")

col1, col2 = st.columns(2)
col1.button("▶ Start webcam", on_click=start_camera)
col2.button("⏹ Stop webcam", on_click=stop_camera)

# Mode just for display (all signals together here)
mode = st.radio(
    "Display mode:",
    ["All signals"],
    horizontal=True,
)

frame_placeholder = st.empty()
info_placeholder = st.empty()
fatigue_bar_placeholder = st.empty()
stress_bar_placeholder = st.empty()
alert_placeholder = st.empty()
rl_placeholder = st.empty()

ALERT_THRESH = 0.80
ALERT_STREAK_FRAMES = 10

frame_idx = 0
prev_time = time.time()
fps = 0.0

# ============================================================
# 7. MAIN STREAMLIT CAPTURE LOOP
# ============================================================

if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam")
    else:
        while st.session_state.run_camera and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            frame_idx += 1

            # ---------------- rPPG BUFFER UPDATE ----------------
            rppg_frame = extract_rppg_frame(frame)
            st.session_state.rppg_buffer.append(rppg_frame)

            # Once we have enough frames, run rPPG every 10 frames
            if (
                len(st.session_state.rppg_buffer) >= RPPG_FRAME_DEPTH
                and (frame_idx % 10 == 0)
            ):
                frames_np = np.stack(list(st.session_state.rppg_buffer), axis=0)  # [T,H,W,C]
                frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)   # [T,C,H,W]
                frames_tensor = frames_tensor.unsqueeze(0).to(device)             # [1,T,C,H,W]

                with torch.no_grad():
                    pred_bpm = rppg_model(frames_tensor)
                bpm_val = float(pred_bpm.detach().cpu().item())
                st.session_state.last_bpm = bpm_val

                # No ground-truth error at runtime → assume 0
                stress_val = calculate_stress_score(bpm_val, prediction_error=0.0)
                st.session_state.last_stress = stress_val
                st.session_state.stress_history.append(stress_val)

            # Compute avg_stress for RL + display
            if st.session_state.stress_history:
                avg_stress = float(sum(st.session_state.stress_history) /
                                   len(st.session_state.stress_history))
            elif st.session_state.last_stress is not None:
                avg_stress = st.session_state.last_stress
            else:
                avg_stress = 50.0

            # ---------------- FATIGUE FEATURES ----------------
            feats, landmarks = compute_fatigue_features_from_frame(frame)
            last_feats = feats

            # ---------------- EMOTION ----------------
            if landmarks is not None:
                face_crop = crop_face_from_landmarks(frame, landmarks, expand_ratio=1.0)
                emo_label_inst, emo_conf_inst, probs_np = emotion_predict_from_bgr(face_crop)
                if probs_np is not None:
                    st.session_state.emotion_history.append(probs_np)
            else:
                face_crop = None
                emo_label_inst, emo_conf_inst, probs_np = None, None, None

            if st.session_state.emotion_history:
                avg_probs = np.mean(st.session_state.emotion_history, axis=0)
                emo_idx = int(np.argmax(avg_probs))
                emotion_label = EMOTION_CLASSES[emo_idx]
                emotion_conf = float(avg_probs[emo_idx])
            else:
                avg_probs = None
                emotion_label = emo_label_inst
                emotion_conf = emo_conf_inst

            # For fusion: "positivity" = happy + surprise
            if avg_probs is not None:
                idx_happy = EMOTION_CLASSES.index("happy")
                idx_surprise = EMOTION_CLASSES.index("surprise")
                positive_prob = float(avg_probs[idx_happy] + avg_probs[idx_surprise])
            else:
                positive_prob = 0.0

            # ---------------- FATIGUE MLP + MULTIMODAL FUSION ----------------
            last_full_proba = None
            if feats is not None:
                feats_for_mlp = feats.copy()

                # If clearly happy, don't treat open mouth as strong yawn
                if emotion_label is not None and emotion_label.lower() == "happy":
                    feats_for_mlp["yawn_flag"] = 0
                    feats_for_mlp["mar"] = min(feats_for_mlp["mar"], 0.30)

                # get both probabilities for debugging
                p_raw, full_proba = mlp_drowsy_prob(feats_for_mlp)
                last_full_proba = full_proba

                # Emotion-aware discount: more happy/surprised → less drowsy probability
                p_raw_adjusted = p_raw * (1.0 - 0.4 * positive_prob)

                # Simple open-eye heuristic (for sanity)
                ear_val = feats_for_mlp["ear"]
                mar_val = feats_for_mlp["mar"]
                yawn_flag = feats_for_mlp["yawn_flag"]
                open_eye_heuristic = (ear_val > 0.23 and mar_val < 0.35 and yawn_flag == 0)
                if open_eye_heuristic and p_raw_adjusted > 0.5:
                    p_raw_adjusted = 0.3

                st.session_state.prob_history.append(p_raw_adjusted)
            else:
                if not st.session_state.prob_history:
                    st.session_state.prob_history.append(0.0)

            if st.session_state.prob_history:
                p_drowsy = float(sum(st.session_state.prob_history) /
                                 len(st.session_state.prob_history))
            else:
                p_drowsy = 0.0

            # Maintain drowsy streak
            if p_drowsy > ALERT_THRESH:
                st.session_state.drowsy_streak += 1
            else:
                st.session_state.drowsy_streak = 0

            # ---------------- RL STATE & UPDATE ----------------
            curr_state = (
                emotion_label if emotion_label is not None else "neutral",
                p_drowsy,
                avg_stress
            )

            now = time.time()
            # Every 3 minutes -> RL recommends an action
            if now - st.session_state.last_intervention_time > 180:
                action = st.session_state.rl_agent.get_action(curr_state)
                st.session_state.last_action = action

                # (same semantics as your original code)
                if action == 1:
                    print("RL Suggestion: Take a break...")
                elif action == 2:
                    print("RL Suggestion: Take a nap...")

                st.session_state.prev_state = curr_state
                st.session_state.last_intervention_time = now

            # Evaluate improvement after 15 sec
            if (
                st.session_state.prev_state is not None
                and (now - st.session_state.last_intervention_time) > 15
            ):
                new_state = curr_state
                reward = compute_reward(st.session_state.prev_state, new_state)
                st.session_state.rl_agent.update(new_state, reward)
                st.session_state.last_reward = reward
                print("RL Reward:", reward)
                st.session_state.prev_state = None

            # ---------------- BUILD DISPLAY FRAME ----------------
            frame_disp = frame.copy()
            if landmarks is not None:
                frame_disp = draw_landmarks_on_frame(frame_disp, landmarks)

            frame_disp = cv2.flip(frame_disp, 1)

            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = curr_time

            # Show frame in Streamlit
            frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # ---------------- TEXT & BARS ----------------
            lines = [f"**FPS:** {fps:.1f}"]

            # Drowsy info
            if feats is not None:
                ear = feats.get("ear", 0.0)
                mar = feats.get("mar", 0.0)
                tilt = feats.get("tilt_deg", 0.0)
                lines.append(
                    f"**EAR:** {ear:.3f} &nbsp;&nbsp; "
                    f"**MAR:** {mar:.3f} &nbsp;&nbsp; "
                    f"**Tilt:** {tilt:.1f}° &nbsp;&nbsp; "
                    f"**P(drowsy):** {p_drowsy:.2f}"
                )
            else:
                lines.append("**Fatigue:** No face detected.")

            fatigue_bar_placeholder.progress(
                min(max(p_drowsy, 0.0), 1.0)
            )

            # Emotion info
            if emotion_label is not None:
                lines.append(
                    f"**Emotion:** {emotion_label.capitalize()} ({(emotion_conf or 0.0):.2f})"
                )
            else:
                lines.append("**Emotion:** low confidence / no face.")

            # BPM + Stress info
            if st.session_state.last_bpm is None or st.session_state.last_stress is None:
                lines.append("**BPM / Stress:** collecting signal...")
            else:
                lines.append(
                    f"**BPM:** {st.session_state.last_bpm:.1f} &nbsp;&nbsp; "
                    f"**Stress:** {avg_stress:.1f} / 100"
                )
                stress_bar_placeholder.progress(
                    min(max(avg_stress / 100.0, 0.0), 1.0)
                )

            info_placeholder.markdown("<br>".join(lines), unsafe_allow_html=True)

            # Drowsy alert
            if st.session_state.drowsy_streak >= ALERT_STREAK_FRAMES:
                alert_placeholder.markdown(
                    "<span style='color:red; font-size:18px;'>⚠️ You look drowsy!</span>",
                    unsafe_allow_html=True,
                )
            else:
                alert_placeholder.empty()

            # RL info
            action_text = {
                0: "0 = No special action",
                1: "1 = Take a short break suggestion",
                2: "2 = Take a power nap suggestion",
            }.get(st.session_state.last_action, "Unknown")

            rl_placeholder.markdown(
                f"**RL Last Action:** {action_text}  &nbsp;&nbsp; "
                f"**Last Reward:** {st.session_state.last_reward:.2f}"
            )

        cap.release()
else:
    st.info("Click **▶ Start webcam** to begin.")
