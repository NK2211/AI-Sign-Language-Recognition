"""
Live 23-class sign recognition — fixed to match the training pipeline.

What changed vs webcam_fusion_23class.py
1. MediaPipe runs on the UN-flipped frame (training videos were not flipped).
   The frame is mirrored only for display.
2. Signs are captured as whole segments (rest -> sign -> rest) and resampled to
   32 frames exactly like extract_landmarks.py, instead of feeding 32
   consecutive webcam frames (~1-2 s of a sign) to a model trained on whole
   clips squeezed into 32 frames.
3. Normalization and geometric features are copied from the training scripts
   (including the "pose missing" behaviour).
4. Real timestamps are passed to MediaPipe.

Controls
  SPACE  start / stop a manual recording (useful while tuning thresholds)
  A      toggle automatic motion-based segmentation
  Q      quit
"""

import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\fusion_grouped_23class.pth"
HOLISTIC_MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\holistic_landmarker.task"


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 32
POSE_INDICES = [11, 12, 13, 14, 15, 16]

# Training used IMAGE mode (landmarker.detect). Keep True to match it.
# Set False for VIDEO mode (faster tracking, slightly different landmarks).
USE_IMAGE_MODE = True

# Motion-based segmentation. Units: shoulder widths per frame,
# averaged over MOTION_SMOOTH frames. Watch the "motion" value on screen
# and adjust for your camera / frame rate.
AUTO_SEGMENT = True
START_MOTION = 0.04
STOP_MOTION = 0.015
MOTION_SMOOTH = 5
PRE_ROLL_FRAMES = 8      # rest frames kept before the sign (clips start at rest)
STOP_IDLE_FRAMES = 12    # still frames that end a sign
MIN_SIGN_FRAMES = 15
MAX_SIGN_FRAMES = 240

CONFIDENCE_THRESHOLD = 50.0

# View checks. Training clips show the upper body with hands in frame
# (hands were detected in about 97% of training frames, even at rest).
MIN_SHOULDER_VISIBILITY = 0.5
MIN_HAND_COVERAGE = 0.6   # share of captured frames with at least one hand
DRAW_LANDMARKS = True


# ============================================================
# MODEL (identical to train_fusion_grouped.py)
# ============================================================

class FusionModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.spatial_bigru = nn.GRU(144, 128, num_layers=2, batch_first=True,
                                    dropout=0.3, bidirectional=True)
        self.geometric_gru = nn.GRU(60, 128, num_layers=2, batch_first=True,
                                    dropout=0.3)
        self.geometric_projection = nn.Linear(128, 256)
        self.attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, spatial, geometric):
        spatial_out, _ = self.spatial_bigru(spatial)
        geometric_out, _ = self.geometric_gru(geometric)
        fused = spatial_out + self.geometric_projection(geometric_out)
        attended, _ = self.attention(fused, fused, fused)
        fused = self.norm(fused + attended)
        return self.classifier(self.dropout(fused.mean(dim=1)))


device = torch.device("cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASSES = checkpoint["classes"]          # use the order saved at training time
model = FusionModel(len(CLASSES))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded {MODEL_PATH} ({len(CLASSES)} classes)")
print("The model can only recognise these signs:\n  " + ", ".join(CLASSES))


# ============================================================
# FEATURES — copied from extract_landmarks.py / extract_geometric_features.py
# ============================================================

def landmarks_from_result(result):
    """48x3 raw landmarks, same layout as extract_landmarks.py."""
    lm = np.zeros((48, 3), dtype=np.float32)
    pose = result.pose_landmarks
    if pose:
        for i, idx in enumerate(POSE_INDICES):
            if idx < len(pose):
                lm[i] = [pose[idx].x, pose[idx].y, pose[idx].z]
    if result.left_hand_landmarks:
        for i, p in enumerate(result.left_hand_landmarks):
            lm[6 + i] = [p.x, p.y, p.z]
    if result.right_hand_landmarks:
        for i, p in enumerate(result.right_hand_landmarks):
            lm[27 + i] = [p.x, p.y, p.z]
    return lm


def normalize_landmarks(lm):
    """Exactly as in training: pose missing -> centre 0, scale 1."""
    center = (lm[0] + lm[1]) / 2.0
    dist = np.linalg.norm(lm[0] - lm[1])
    if dist < 1e-6:
        dist = 1.0
    return ((lm - center) / dist).astype(np.float32)


def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    """Same nearest-index resampling as extract_landmarks.py."""
    sequence = np.array(sequence, dtype=np.float32)
    if len(sequence) == target_length:
        return sequence
    idx = np.round(np.linspace(0, len(sequence) - 1, target_length)).astype(int)
    return sequence[idx]


def _dist(a, b):
    return np.linalg.norm(a - b)


def _angle(a, b, c):
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-8:
        return 0.0
    return np.arccos(np.clip(np.dot(ba, bc) / den, -1.0, 1.0))


def geometric_features(f):
    ls, rs, le, re, lw, rw = f[0], f[1], f[2], f[3], f[4], f[5]
    feats = [
        _dist(ls, rs), _dist(ls, le), _dist(le, lw), _dist(ls, lw),
        _dist(rs, re), _dist(re, rw), _dist(rs, rw), _dist(lw, rw),
        _angle(ls, le, lw), _angle(rs, re, rw),
        _dist(lw, rs), _dist(rw, ls),
    ]
    feats += [_dist(lw, f[i]) for i in range(6, 27)]
    feats += [_dist(rw, f[i]) for i in range(27, 48)]
    feats += (lw - ls).tolist() + (rw - rs).tolist()
    return np.array(feats, dtype=np.float32)


def _visibility(lm):
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else float(v)


def check_view(result):
    """Is the signer framed like the training videos?"""
    pose = result.pose_landmarks
    if not pose or len(pose) < 17:
        return False, "No person found - face the camera"
    if min(_visibility(pose[11]), _visibility(pose[12])) < MIN_SHOULDER_VISIBILITY:
        return False, "Shoulders not visible - move back"
    if not result.left_hand_landmarks and not result.right_hand_landmarks:
        return False, "Hands not visible - move back, keep hands in frame"
    return True, "View OK"


def hand_wrists(result, norm):
    """Normalized hand-wrist points that were actually detected (not guessed)."""
    pts = {}
    if result.left_hand_landmarks:
        pts["L"] = norm[6]
    if result.right_hand_landmarks:
        pts["R"] = norm[27]
    return pts


def draw_landmarks(view, result):
    """Draw what MediaPipe tracks, mirrored to match the preview."""
    h, w = view.shape[:2]
    to_px = lambda p: (int((1 - p.x) * w), int(p.y * h))
    pose = result.pose_landmarks
    if pose and len(pose) >= 17:
        for a, b in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
            if min(_visibility(pose[a]), _visibility(pose[b])) >= 0.5:
                cv2.line(view, to_px(pose[a]), to_px(pose[b]), (255, 200, 0), 2)
        for i in POSE_INDICES:
            ok = _visibility(pose[i]) >= 0.5
            cv2.circle(view, to_px(pose[i]), 6,
                       (255, 200, 0) if ok else (0, 0, 255), -1)
    for hand, color in [(result.left_hand_landmarks, (0, 255, 0)),
                        (result.right_hand_landmarks, (255, 0, 255))]:
        if hand:
            for p in hand:
                cv2.circle(view, to_px(p), 3, color, -1)


def predict_segment(frames):
    seq = resample_sequence(frames)                        # (32, 48, 3)
    spatial = seq.reshape(SEQUENCE_LENGTH, 144)
    geometric = np.stack([geometric_features(f) for f in seq])
    with torch.no_grad():
        logits = model(torch.from_numpy(spatial)[None],
                       torch.from_numpy(geometric)[None])
        probs = torch.softmax(logits, dim=1)[0].numpy()
    top = probs.argsort()[::-1][:3]
    return [(CLASSES[i], probs[i] * 100) for i in top]


# ============================================================
# MEDIAPIPE
# ============================================================

options = vision.HolisticLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH),
    running_mode=(vision.RunningMode.IMAGE if USE_IMAGE_MODE
                  else vision.RunningMode.VIDEO),
)


# ============================================================
# MAIN LOOP
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
motion_hist = deque(maxlen=MOTION_SMOOTH)
segment = []          # normalized frames
segment_hands = []    # True where at least one hand was detected
recording = False
manual = False
idle_count = 0
prev_wrists = {}
last_result = [("Waiting...", 0.0)]
status_msg = ""
t0 = time.monotonic()

with vision.HolisticLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detection on the ORIGINAL frame — do not flip before MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if USE_IMAGE_MODE:
            result = landmarker.detect(image)
        else:
            result = landmarker.detect_for_video(
                image, int((time.monotonic() - t0) * 1000))

        norm = normalize_landmarks(landmarks_from_result(result))
        view_ok, view_msg = check_view(result)
        has_hand = bool(result.left_hand_landmarks or result.right_hand_landmarks)

        # Motion from DETECTED hand wrists only. Pose wrists outside the frame
        # are guesses that jump every frame and fake huge motion.
        wrists = hand_wrists(result, norm)
        common = [k for k in wrists if k in prev_wrists]
        if view_ok and common:
            motion_hist.append(max(_dist(wrists[k], prev_wrists[k]) for k in common))
        else:
            motion_hist.append(0.0)
        prev_wrists = wrists
        motion = float(np.mean(motion_hist)) if motion_hist else 0.0

        # ---------------- segmentation ----------------
        if recording:
            segment.append(norm)
            segment_hands.append(has_hand)
            if not manual:
                still = motion < STOP_MOTION or not view_ok
                idle_count = idle_count + 1 if still else 0
                if idle_count >= STOP_IDLE_FRAMES or len(segment) >= MAX_SIGN_FRAMES:
                    recording = False
        elif AUTO_SEGMENT and view_ok and motion > START_MOTION:
            recording, manual, idle_count = True, False, 0
            segment = [f for f, _ in pre_roll] + [norm]
            segment_hands = [h for _, h in pre_roll] + [has_hand]
        else:
            pre_roll.append((norm, has_hand))

        if not recording and segment:
            coverage = float(np.mean(segment_hands)) if segment_hands else 0.0
            if len(segment) < MIN_SIGN_FRAMES:
                status_msg = f"Discarded: only {len(segment)} frames"
            elif coverage < MIN_HAND_COVERAGE:
                status_msg = (f"Discarded: hands visible in {coverage:.0%} "
                              "of frames - keep hands in view")
            else:
                last_result = predict_segment(segment)
                status_msg = f"Predicted from {len(segment)} frames"
                print(f"{len(segment)} frames, hands {coverage:.0%} -> " +
                      ", ".join(f"{c} {p:.1f}%" for c, p in last_result))
            if status_msg.startswith("Discarded"):
                print(status_msg)
                last_result = [("-", 0.0)]   # don't leave an old prediction on screen
            segment, segment_hands = [], []
            pre_roll.clear()

        # ---------------- display (mirrored only here) ----------------
        view = cv2.flip(frame, 1)
        if DRAW_LANDMARKS:
            draw_landmarks(view, result)
        cv2.rectangle(view, (0, 0), (view.shape[1], 175), (0, 0, 0), -1)
        name, conf = last_result[0]
        shown = name if conf >= CONFIDENCE_THRESHOLD else "Uncertain"
        cv2.putText(view, f"Prediction: {shown} ({conf:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(view, "Top-3: " + " | ".join(
            f"{c} {p:.0f}%" for c, p in last_result), (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        state = "RECORDING" if recording else ("auto" if AUTO_SEGMENT else "manual")
        cv2.putText(view, f"{state}  motion={motion:.3f}  frames={len(segment)}"
                    f"  {status_msg}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if recording else (200, 200, 200), 2)
        cv2.putText(view, view_msg, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if view_ok else (0, 200, 255), 2)
        cv2.putText(view, "SPACE record  A auto on/off  Q quit",
                    (20, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.imshow("Sign Recognition (fixed)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            AUTO_SEGMENT = not AUTO_SEGMENT
        if key == ord(" "):
            if recording:
                recording = False
            else:
                recording, manual = True, True
                segment = [f for f, _ in pre_roll]
                segment_hands = [h for _, h in pre_roll]

cap.release()
cv2.destroyAllWindows()
