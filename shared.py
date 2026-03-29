# shared.py  —  Constants, gesture detection, drawing helpers
# Used by all three mode files and main.py

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# ─── CONFIG ──────────────────────────────────────────
MIRROR            = True
CONFIDENCE        = 0.75
FPS_SMOOTH        = 20
TIME_LIMIT        = 10.0      # seconds per round  (Game mode)
HOLD_TIME         = 1.2       # seconds to hold a gesture before it counts
ROUNDS_TO_WIN     = 5         # rounds per game
TEACH_SAMPLES     = 30        # frames captured per sign  (Teach mode)
TEACH_HOLD        = 2.0       # seconds of stillness before sampling starts
CUSTOM_SIGNS_FILE = "custom_signs.json"
CLASSIFY_THRESH   = 0.35      # max centroid distance for custom sign match
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
    "✊ FIST",
    "🖐 OPEN",
    "☝ POINT",
    "✌ PEACE",
    "👍 THUMBS UP",
    "🤙 CALL ME",
    "3 FINGERS",
]


# ════════════════════════════════════════════════════
#  CUSTOM SIGN STORAGE
# ════════════════════════════════════════════════════

def load_custom_signs():
    """Load {name: [feature_vector, ...]} from disk."""
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


def landmarks_to_feature(lm):
    """42-float vector: (x,y) pairs normalised relative to wrist."""
    base_x, base_y = lm[0].x, lm[0].y
    coords = []
    for l in lm:
        coords.append(l.x - base_x)
        coords.append(l.y - base_y)
    max_d = max(abs(v) for v in coords) or 1e-6
    return [v / max_d for v in coords]


def classify_custom(feature, custom_signs):
    """Return (label, distance) or (None, inf)."""
    import numpy as np
    best_label = None
    best_dist  = float("inf")
    feat = np.array(feature)
    for label, samples in custom_signs.items():
        centroid = np.mean(samples, axis=0)
        d = float(np.linalg.norm(feat - centroid))
        if d < best_dist:
            best_dist  = d
            best_label = label
    if best_dist < CLASSIFY_THRESH:
        return best_label, best_dist
    return None, best_dist


# ════════════════════════════════════════════════════
#  GESTURE DETECTION
# ════════════════════════════════════════════════════

def detect_builtin_gesture(landmarks):
    def tip_above_pip(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y

    thumb_open  = landmarks[4].x < landmarks[3].x
    index_open  = tip_above_pip(8,  6)
    middle_open = tip_above_pip(12, 10)
    ring_open   = tip_above_pip(16, 14)
    pinky_open  = tip_above_pip(20, 18)
    fingers     = [index_open, middle_open, ring_open, pinky_open]

    if not any(fingers) and not thumb_open:                                                return "✊ FIST"
    if all(fingers) and thumb_open:                                                        return "🖐 OPEN"
    if index_open and not middle_open and not ring_open and not pinky_open:                return "☝ POINT"
    if index_open and middle_open and not ring_open and not pinky_open:                    return "✌ PEACE"
    if index_open and middle_open and ring_open and not pinky_open:                        return "3 FINGERS"
    if thumb_open and pinky_open and not index_open and not middle_open and not ring_open: return "🤙 CALL ME"
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open: return "👍 THUMBS UP"
    return None


def detect_gesture(lm, custom_signs):
    """Try built-in first, then custom. Returns (label, 'builtin'|'custom'|None)."""
    builtin = detect_builtin_gesture(lm)
    if builtin:
        return builtin, "builtin"
    if custom_signs:
        feat  = landmarks_to_feature(lm)
        label, _ = classify_custom(feat, custom_signs)
        if label:
            return f"[{label}]", "custom"
    return None, None


# ════════════════════════════════════════════════════
#  SHARED DRAWING HELPERS
# ════════════════════════════════════════════════════

def draw_skeleton(frame, landmarks, w, h):
    for conn in mp.solutions.hands.HAND_CONNECTIONS:
        a, b = conn
        x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
        x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (50, 80, 100), 1, cv2.LINE_AA)
    for name, tid in FINGERTIP_IDS.items():
        cx = int(landmarks[tid].x * w)
        cy = int(landmarks[tid].y * h)
        cv2.circle(frame, (cx, cy), 6, FINGER_COLORS[name], -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, FINGER_COLORS[name],  1, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.75, color=(12, 12, 25), border=(40, 40, 80)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1)


def center_text(frame, text, cy, font, scale, color, thickness=1):
    h_frame, w_frame = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, ((w_frame - tw) // 2, cy),
                font, scale, color, thickness, cv2.LINE_AA)


def draw_timer_arc(frame, cx, cy, radius, fraction, color, bg_color=(40, 40, 60)):
    cv2.circle(frame, (cx, cy), radius, bg_color, 4, cv2.LINE_AA)
    if fraction > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0,
                    int(360 * fraction), color, 4, cv2.LINE_AA)