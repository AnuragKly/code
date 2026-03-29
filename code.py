import cv2
import mediapipe as mp
import numpy as np
import time
import os
from collections import deque
from datetime import datetime
 
 
# ─── CONFIG ───────────-───────────────────────────────
TRAIL_LENGTH       = 60       # how many points to keep in trail
TRAIL_FADE         = True     # trail fades out over time
SHOW_ALL_FINGERS   = True     # track all 5 fingertips, not just index
MIRROR             = True     # mirror webcam feed
CONFIDENCE         = 0.75     # detection confidence threshold
FPS_SMOOTH         = 20       # frames to average FPS over
SCREENSHOT_DIR     = "screenshots"
# ─────────────────────────────────────────────────────
 
 
# MediaPipe finger tip landmark IDs
FINGERTIP_IDS = {
    "THUMB":  4,
    "INDEX":  8,
    "MIDDLE": 12,
    "RING":   16,
    "PINKY":  20,
}
 
FINGER_COLORS = {
    "THUMB":  (0,   200, 255),   # orange
    "INDEX":  (0,   255, 150),   # green
    "MIDDLE": (255, 100, 100),   # blue
    "RING":   (200,  50, 255),   # purple
    "PINKY":  (50,  200, 255),   # yellow
}
 
# Simple gesture detection
def detect_gesture(landmarks, w, h):
    """Return gesture label based on which fingers are extended."""
    def tip_above_pip(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y
 
    thumb_open  = landmarks[4].x < landmarks[3].x  # crude thumb check
    index_open  = tip_above_pip(8,  6)
    middle_open = tip_above_pip(12, 10)
    ring_open   = tip_above_pip(16, 14)
    pinky_open  = tip_above_pip(20, 18)
 
    fingers = [index_open, middle_open, ring_open, pinky_open]
    count   = sum(fingers)
 
    if not any(fingers) and not thumb_open:
        return "✊ FIST"
    if all(fingers) and thumb_open:
        return "🖐 OPEN"
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "☝ POINT"
    if index_open and middle_open and not ring_open and not pinky_open:
        return "✌ PEACE"
    if index_open and middle_open and ring_open and not pinky_open:
        return "3 FINGERS"
    if thumb_open and pinky_open and not index_open and not middle_open and not ring_open:
        return "🤙 CALL ME"
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "👍 THUMBS UP"
    if pinky_open and not index_open and not middle_open and not ring_open:
        return "🤙 PINKY"
    return f"  {count} UP"
 
 
def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1, alpha=1.0):
    """Draw a filled or outlined rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.ellipse(overlay, (cx, cy), (r, r), 0, 0, 360, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
 
 
def draw_hud(frame, h, w, fps, hand_count, gesture, show_gestures):
    """Render the HUD overlay — top bar and bottom info strip."""
    bar_h = 40
    overlay = frame.copy()
 
    # Top bar background
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
 
    # Title
    cv2.putText(frame, "FINGER TRACKER", (12, 27),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)
 
    # FPS
    fps_color = (0, 255, 100) if fps >= 25 else (0, 180, 255) if fps >= 15 else (0, 80, 255)
    cv2.putText(frame, f"FPS {fps:04.1f}", (w - 120, 27),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, fps_color, 1, cv2.LINE_AA)
 
    # Hand count indicator
    hands_txt = f"HANDS: {hand_count}"
    cv2.putText(frame, hands_txt, (w // 2 - 50, 27),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)
 
    # Bottom strip
    strip_y = h - 32
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, strip_y), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
 
    hints = "[Q] Quit  [S] Screenshot  [T] Trail  [G] Gesture  [R] Reset"
    cv2.putText(frame, hints, (10, h - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (80, 80, 120), 1, cv2.LINE_AA)
 
    if show_gestures and gesture:
        cv2.putText(frame, gesture, (w - 200, h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 200), 1, cv2.LINE_AA)
 
 
def draw_finger_info(frame, name, tip_x, tip_y, norm_x, norm_y, color, idx):
    """Draw a small info bubble near the fingertip."""
    bx, by = tip_x + 12, tip_y - 28
    label = f"{name}"
    coord = f"({norm_x:.2f}, {norm_y:.2f})"
 
    # Glow circle on tip
    cv2.circle(frame, (tip_x, tip_y), 18, (*color, 60), -1)
    cv2.circle(frame, (tip_x, tip_y), 11, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (tip_x, tip_y), 4,  color, -1, cv2.LINE_AA)
 
    # Cross-hair lines
    cv2.line(frame, (tip_x - 20, tip_y), (tip_x - 12, tip_y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (tip_x + 12, tip_y), (tip_x + 20, tip_y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (tip_x, tip_y - 20), (tip_x, tip_y - 12), color, 1, cv2.LINE_AA)
    cv2.line(frame, (tip_x, tip_y + 12), (tip_x, tip_y + 20), color, 1, cv2.LINE_AA)
 
    # Label bubble
    cv2.putText(frame, label, (bx, by),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(frame, coord, (bx, by + 14),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (180, 180, 220), 1, cv2.LINE_AA)
 
 
def draw_trail(frame, trail, color, fade):
    """Draw a smooth fading trail for a fingertip."""
    pts = list(trail)
    n   = len(pts)
    for i in range(1, n):
        if pts[i - 1] is None or pts[i] is None:
            continue
        if fade:
            alpha = i / n
            c = tuple(int(ch * alpha) for ch in color)
        else:
            c = color
        thickness = max(1, int(3 * (i / n)))
        cv2.line(frame, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)
 
 
def draw_skeleton(frame, landmarks, w, h):
    """Draw hand skeleton connections."""
    CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
    for conn in CONNECTIONS:
        a, b = conn
        x1 = int(landmarks[a].x * w)
        y1 = int(landmarks[a].y * h)
        x2 = int(landmarks[b].x * w)
        y2 = int(landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (50, 80, 100), 1, cv2.LINE_AA)
 
 
def draw_coords_panel(frame, finger_data, h, w):
    """Draw a coordinate readout panel on the right side."""
    px, py = w - 190, 50
    panel_w, panel_h = 180, len(finger_data) * 48 + 24
 
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 8, py - 8),
                  (px + panel_w, py + panel_h), (12, 12, 25), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (px - 8, py - 8),
                  (px + panel_w, py + panel_h), (30, 30, 60), 1)
 
    cv2.putText(frame, "POSITIONS", (px, py + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (80, 80, 140), 1, cv2.LINE_AA)
 
    for i, (name, nx, ny, color) in enumerate(finger_data):
        row_y = py + 26 + i * 48
        cv2.putText(frame, name, (px, row_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"X  {nx:.3f}", (px, row_y + 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (200, 200, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Y  {ny:.3f}", (px, row_y + 26),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (200, 200, 240), 1, cv2.LINE_AA)
 
 
# ─── MAIN ────────────────────────────────────────────
 
def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
 
    mp_hands    = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
    )
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open webcam. Check camera permissions.")
        return
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
    # Trails: one deque per finger per hand slot (max 2 hands)
    trails = {
        hand_idx: {name: deque(maxlen=TRAIL_LENGTH) for name in FINGERTIP_IDS}
        for hand_idx in range(2)
    }
 
    fps_times    = deque(maxlen=FPS_SMOOTH)
    show_trail   = True
    show_gestures= True
    gesture_label= ""
 
    print("\n🟢  Finger Tracker running — press Q or ESC to quit\n")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  Frame grab failed.")
            break
 
        if MIRROR:
            frame = cv2.flip(frame, 1)
 
        h, w = frame.shape[:2]
        t0   = time.time()
 
        # ── MediaPipe inference ───────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb)
 
        hand_count  = 0
        finger_data = []   # for side panel
 
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
 
            for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                lm = hand_lm.landmark
 
                # Skeleton
                draw_skeleton(frame, lm, w, h)
 
                # Gesture
                if show_gestures:
                    gesture_label = detect_gesture(lm, w, h)
 
                fingers_to_show = FINGERTIP_IDS if SHOW_ALL_FINGERS else {"INDEX": 8}
 
                for name, tip_id in fingers_to_show.items():
                    color  = FINGER_COLORS[name]
                    norm_x = lm[tip_id].x
                    norm_y = lm[tip_id].y
                    tip_x  = int(norm_x * w)
                    tip_y  = int(norm_y * h)
 
                    # Trail
                    trails[hand_idx][name].append((tip_x, tip_y))
                    if show_trail:
                        draw_trail(frame, trails[hand_idx][name], color, TRAIL_FADE)
 
                    # Fingertip dot + crosshair + label
                    draw_finger_info(frame, name, tip_x, tip_y, norm_x, norm_y, color, hand_idx)
 
                    if name == "INDEX":   # populate side panel with index finger
                        finger_data.append((name, norm_x, norm_y, color))
 
                # Collect all fingers for panel (first hand only to keep it clean)
                if hand_idx == 0:
                    finger_data = [
                        (name, lm[tid].x, lm[tid].y, FINGER_COLORS[name])
                        for name, tid in FINGERTIP_IDS.items()
                    ]
        else:
            # Clear trails when no hand visible
            gesture_label = ""
            for hi in trails.values():
                for trail in hi.values():
                    trail.append(None)
 
        # ── FPS ───────────────────────────────────────
        fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
 
        # ── Overlays ──────────────────────────────────
        draw_hud(frame, h, w, fps, hand_count, gesture_label, show_gestures)
 
        if finger_data:
            draw_coords_panel(frame, finger_data, h, w)
 
        cv2.imshow("FINGER TRACKER", frame)
 
        # ── Key handling ──────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):          # Q or ESC
            break
        elif key == ord('s'):              # Screenshot
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"finger_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"📸  Screenshot saved → {path}")
        elif key == ord('t'):              # Toggle trail
            show_trail = not show_trail
            print(f"Trail: {'ON' if show_trail else 'OFF'}")
        elif key == ord('g'):              # Toggle gesture
            show_gestures = not show_gestures
            gesture_label = ""
            print(f"Gesture labels: {'ON' if show_gestures else 'OFF'}")
        elif key == ord('r'):              # Reset trails
            for hi in trails.values():
                for trail in hi.values():
                    trail.clear()
            print("Trails reset.")
 
    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()
    print("\n👋  Tracker closed.")
 
 
if __name__ == "__main__":
    main()