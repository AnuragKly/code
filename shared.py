# shared.py  —  Config · Detection · Drawing helpers
# Shared by main.py, mode_game.py, mode_free.py, mode_teach.py
#
# Improvements over v1:
#   • Feature vector now includes z (depth) → 63 floats instead of 42
#   • Handedness-aware thumb detection (works for both left and right hand)
#   • k-NN classifier (k=5, majority vote) replaces fragile single centroid
#   • GestureBuffer: rolling majority-vote over last N frames → no flickering
#   • Confidence score (0–1) returned alongside every detection
#   • draw_skeleton encodes depth via joint brightness
#   • draw_confidence_bar shared helper for Free + Game modes
#   • draw_panel supports rounded corners

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from collections import deque, Counter

# ════════════════════════════════════════════════════
#  CONFIG  (edit these to tune behaviour)
# ════════════════════════════════════════════════════
MIRROR            = True
CONFIDENCE        = 0.72        # MediaPipe detection/tracking threshold
FPS_SMOOTH        = 20          # frames for FPS rolling average

TIME_LIMIT        = 10.0        # seconds per game round
HOLD_TIME         = 1.0         # seconds to hold gesture before counting
ROUNDS_TO_WIN     = 5

TEACH_SAMPLES     = 40          # frames captured per teaching session
TEACH_HOLD        = 1.5         # seconds of stillness before sampling
CUSTOM_SIGNS_FILE = "custom_signs.json"

# Classifier
KNN_K             = 5           # neighbours for k-NN vote
CLASSIFY_THRESH   = 0.42        # max distance to accept a custom match

# Smoothing buffer
SMOOTH_WINDOW     = 8           # rolling window size (frames)
SMOOTH_MIN_VOTES  = 4           # votes needed to confirm a label
# ─────────────────────────────────────────────────────

FINGERTIP_IDS = {"THUMB": 4, "INDEX": 8, "MIDDLE": 12, "RING": 16, "PINKY": 20}

FINGER_COLORS = {
    "THUMB":  (0,   200, 255),
    "INDEX":  (0,   255, 150),
    "MIDDLE": (255, 100, 100),
    "RING":   (200,  50, 255),
    "PINKY":  (50,  200, 255),
}

BUILT_IN_GESTURES = [
    "FIST",
    "OPEN",
    "POINT",
    "PEACE",
    "THUMBS UP",
    "CALL ME",
    "3 FINGERS",
]


# ════════════════════════════════════════════════════
#  CUSTOM SIGN STORAGE
# ════════════════════════════════════════════════════

def load_custom_signs():
    """Return {name: [[float,...], ...]} from disk, or {}."""
    if os.path.exists(CUSTOM_SIGNS_FILE):
        try:
            with open(CUSTOM_SIGNS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_custom_signs(signs):
    with open(CUSTOM_SIGNS_FILE, "w") as f:
        json.dump(signs, f)


# ════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (63 floats: x, y, z × 21 landmarks)
# ════════════════════════════════════════════════════

def landmarks_to_feature(lm):
    """
    Build a translation- and scale-invariant 63-float feature vector.
    All (x, y, z) coords are offset by the wrist (lm[0]) position then
    normalised by the maximum absolute value.
    z (depth) distinguishes curled vs extended fingers far better than
    x/y alone, especially for FIST vs relaxed OPEN.
    """
    base_x, base_y, base_z = lm[0].x, lm[0].y, lm[0].z
    coords = []
    for l in lm:
        coords.append(l.x - base_x)
        coords.append(l.y - base_y)
        coords.append(l.z - base_z)
    max_d = max(abs(v) for v in coords) or 1e-6
    return [v / max_d for v in coords]


# ════════════════════════════════════════════════════
#  k-NN CLASSIFIER
# ════════════════════════════════════════════════════

def classify_custom(feature, custom_signs):
    """
    k-NN majority vote (k = KNN_K) across all stored samples.

    Returns (label, confidence):
        confidence ∈ [0, 1] = (vote_fraction) × (1 - normalised_distance)
    Returns (None, 0.0) when no sign is close enough.

    Why better than centroid:
      Each sample votes individually, so natural hand-pose variance across
      sessions is tolerated without the centroid drifting to a bad average.
    """
    if not custom_signs:
        return None, 0.0

    feat = np.array(feature, dtype=np.float32)

    neighbours = []
    for label, samples in custom_signs.items():
        for s in samples:
            s_arr = np.array(s, dtype=np.float32)

        # Handle shape mismatch
        if feat.shape != s_arr.shape:
            # Convert feat (63 → 42) if needed
            if feat.shape[0] == 63 and s_arr.shape[0] == 42:
                feat_use = feat.reshape(-1, 3)[:, :2].flatten()
            # Convert s (42 → 63) if needed
            elif feat.shape[0] == 42 and s_arr.shape[0] == 63:
                s_arr = s_arr.reshape(-1, 3)[:, :2].flatten()
                feat_use = feat
            else:
                continue  # skip incompatible data
        else:
            feat_use = feat

        d = float(np.linalg.norm(feat_use - s_arr))
        neighbours.append((d, label))

    if not neighbours:
        return None, 0.0

    neighbours.sort(key=lambda x: x[0])
    k          = min(KNN_K, len(neighbours))
    top_k      = neighbours[:k]
    best_dist  = top_k[0][0]

    if best_dist >= CLASSIFY_THRESH:
        return None, 0.0

    vote_counts          = Counter(label for _, label in top_k)
    winner, votes        = vote_counts.most_common(1)[0]
    vote_conf            = votes / k
    prox_conf            = max(0.0, 1.0 - best_dist / CLASSIFY_THRESH)
    confidence           = vote_conf * prox_conf

    return winner, round(confidence, 3)


# ════════════════════════════════════════════════════
#  BUILT-IN GESTURE DETECTION  (handedness-aware)
# ════════════════════════════════════════════════════

def detect_builtin_gesture(landmarks, is_right_hand: bool):
    """
    Rule-based detection for 7 built-in gestures.

    is_right_hand comes from MediaPipe's handedness classification
    (already corrected for mirror flip in main.py).
    The thumb open/close test is axis-flipped for left hands so that
    ✊, 👍 and 🤙 register correctly regardless of which hand is used.
    """
    def tip_above_pip(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y

    if is_right_hand:
        thumb_open = landmarks[4].x < landmarks[3].x
    else:
        thumb_open = landmarks[4].x > landmarks[3].x

    index_open  = tip_above_pip(8,  6)
    middle_open = tip_above_pip(12, 10)
    ring_open   = tip_above_pip(16, 14)
    pinky_open  = tip_above_pip(20, 18)
    fingers     = [index_open, middle_open, ring_open, pinky_open]

    if not any(fingers) and not thumb_open:                                                return "FIST"
    if all(fingers) and thumb_open:                                                        return "OPEN"
    if index_open and not middle_open and not ring_open and not pinky_open:                return "POINT"
    if index_open and middle_open and not ring_open and not pinky_open:                    return "PEACE"
    if index_open and middle_open and ring_open and not pinky_open:                        return "FINGERS"
    if thumb_open and pinky_open and not index_open and not middle_open and not ring_open: return "CALL ME"
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open: return "THUMBS UP"
    if not index_open and not ring_open and not pinky_open and middle_open and not thumb_open: return "FUCK YOU"
    return None


# ════════════════════════════════════════════════════
#  SMOOTHING BUFFER  (rolling majority vote)
# ════════════════════════════════════════════════════

class GestureBuffer:
    """
    Eliminates single-frame flickers in the detected label.

    Each frame pushes one raw (label, confidence) result.
    The buffer only confirms a label when it wins a majority vote
    across the last SMOOTH_WINDOW frames AND exceeds SMOOTH_MIN_VOTES.
    The returned confidence is the mean confidence of the winning frames.
    """

    def __init__(self, window=SMOOTH_WINDOW, min_votes=SMOOTH_MIN_VOTES):
        self._window    = window
        self._min_votes = min_votes
        self._history   = deque(maxlen=window)
        self._stable    = (None, 0.0)

    def push(self, label, confidence):
        """Feed one frame. Returns smoothed (label, confidence)."""
        self._history.append((label, confidence))

        if len(self._history) < self._min_votes:
            return self._stable

        labels = [l for l, _ in self._history if l is not None]
        if not labels:
            self._stable = (None, 0.0)
            return self._stable

        winner, votes = Counter(labels).most_common(1)[0]
        if votes >= self._min_votes:
            avg_conf     = float(np.mean([c for l, c in self._history if l == winner]))
            self._stable = (winner, avg_conf)
        else:
            self._stable = (None, 0.0)

        return self._stable

    def reset(self):
        self._history.clear()
        self._stable = (None, 0.0)


# ════════════════════════════════════════════════════
#  UNIFIED RAW DETECTOR  (one frame, no smoothing)
# ════════════════════════════════════════════════════

def detect_gesture_raw(lm, is_right_hand, custom_signs):
    """
    Single-frame detection. Smoothing is applied by the caller via GestureBuffer.
    Returns (label, gesture_type, confidence).
      gesture_type : "builtin" | "custom" | None
      confidence   : 1.0 for built-ins, 0–1 for custom
    """
    builtin = detect_builtin_gesture(lm, is_right_hand)
    if builtin:
        return builtin, "builtin", 1.0

    if custom_signs:
        feat        = landmarks_to_feature(lm)
        label, conf = classify_custom(feat, custom_signs)
        if label:
            return f"[{label}]", "custom", conf

    return None, None, 0.0


# ════════════════════════════════════════════════════
#  DRAWING HELPERS
# ════════════════════════════════════════════════════

def draw_skeleton(frame, landmarks, w, h):
    """
    Draw hand skeleton. Fingertip brightness is modulated by z-depth
    so fingers closer to the camera appear visually brighter.
    """
    for conn in mp.solutions.hands.HAND_CONNECTIONS:
        a, b = conn
        x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
        x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (55, 85, 110), 1, cv2.LINE_AA)

    z_vals  = [l.z for l in landmarks]
    z_min   = min(z_vals)
    z_range = (max(z_vals) - z_min) or 1e-6

    for name, tid in FINGERTIP_IDS.items():
        cx = int(landmarks[tid].x * w)
        cy = int(landmarks[tid].y * h)
        brightness = 0.55 + 0.45 * (1.0 - (landmarks[tid].z - z_min) / z_range)
        base  = FINGER_COLORS[name]
        color = tuple(int(c * brightness) for c in base)
        cv2.circle(frame, (cx, cy), 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 9, color,  1, cv2.LINE_AA)

    for i, lm in enumerate(landmarks):
        if i not in FINGERTIP_IDS.values():
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (60, 90, 115), -1, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.78,
               color=(12, 12, 25), border=(40, 40, 80), radius=8):
    """Semi-transparent rounded-rectangle panel."""
    overlay = frame.copy()
    r = min(radius, w // 2, h // 2)
    if r > 0:
        cv2.rectangle(overlay, (x + r, y),     (x + w - r, y + h), color, -1)
        cv2.rectangle(overlay, (x,     y + r),  (x + w,     y + h - r), color, -1)
        for cx2, cy2 in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
            cv2.circle(overlay, (cx2, cy2), r, color, -1)
    else:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)


def center_text(frame, text, cy, font, scale, color, thickness=1):
    _, w_frame = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, ((w_frame - tw) // 2, cy),
                font, scale, color, thickness, cv2.LINE_AA)


def draw_timer_arc(frame, cx, cy, radius, fraction, color, bg=(35, 35, 55)):
    cv2.circle(frame, (cx, cy), radius, bg, 5, cv2.LINE_AA)
    if fraction > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius),
                    -90, 0, max(1, int(360 * fraction)), color, 5, cv2.LINE_AA)


def draw_confidence_bar(frame, x, y, w, h, confidence, label="Conf"):
    """
    Horizontal bar: red (low) → green (high) confidence.
    Shared between Free Mode and Game Mode overlays.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 35), -1)
    fill = int(w * min(max(confidence, 0.0), 1.0))
    r    = int(255 * (1.0 - confidence))
    g    = int(255 * confidence)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), (0, g, r), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 80), 1)
    cv2.putText(frame, f"{label} {int(confidence * 100)}%",
                (x + 5, y + h - 3), cv2.FONT_HERSHEY_PLAIN, 0.78,
                (200, 200, 220), 1, cv2.LINE_AA)