import cv2
import mediapipe as mp
import numpy as np
import time
import random
from collections import deque
 
 
# ─── CONFIG ──────────────────────────────────────────
MIRROR         = True
CONFIDENCE     = 0.75
FPS_SMOOTH     = 20
TIME_LIMIT     = 10.0      # seconds per round
HOLD_TIME      = 1.2       # seconds to hold gesture before it counts
ROUNDS_TO_WIN  = 5         # rounds per game
# ─────────────────────────────────────────────────────
 
 
FINGERTIP_IDS = {"THUMB": 4, "INDEX": 8, "MIDDLE": 12, "RING": 16, "PINKY": 20}
 
FINGER_COLORS = {
    "THUMB":  (0,   200, 255),
    "INDEX":  (0,   255, 150),
    "MIDDLE": (255, 100, 100),
    "RING":   (200,  50, 255),
    "PINKY":  (50,  200, 255),
}
 
# All detectable gestures
GESTURES = [
    "✊ FIST",
    "🖐 OPEN",
    "☝ POINT",
    "✌ PEACE",
    "👍 THUMBS UP",
    "🤙 CALL ME",
    "3 FINGERS",
]
 
# Game states
STATE_MENU    = "MENU"
STATE_PLAYING = "PLAYING"
STATE_SUCCESS = "SUCCESS"
STATE_FAIL    = "FAIL"
STATE_WIN     = "WIN"
STATE_LOSE    = "LOSE"
 
 
def detect_gesture(landmarks):
    def tip_above_pip(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y
 
    thumb_open  = landmarks[4].x < landmarks[3].x
    index_open  = tip_above_pip(8,  6)
    middle_open = tip_above_pip(12, 10)
    ring_open   = tip_above_pip(16, 14)
    pinky_open  = tip_above_pip(20, 18)
    fingers     = [index_open, middle_open, ring_open, pinky_open]
    count       = sum(fingers)
 
    if not any(fingers) and not thumb_open:           return "✊ FIST"
    if all(fingers) and thumb_open:                   return "🖐 OPEN"
    if index_open and not middle_open and not ring_open and not pinky_open: return "☝ POINT"
    if index_open and middle_open and not ring_open and not pinky_open:     return "✌ PEACE"
    if index_open and middle_open and ring_open and not pinky_open:         return "3 FINGERS"
    if thumb_open and pinky_open and not index_open and not middle_open and not ring_open: return "🤙 CALL ME"
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open: return "👍 THUMBS UP"
    return None
 
 
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
        cv2.circle(frame, (cx, cy), 8, FINGER_COLORS[name], 1, cv2.LINE_AA)
 
 
def draw_panel(frame, x, y, w, h, alpha=0.75, color=(12, 12, 25), border=(40, 40, 80)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1)
 
 
def center_text(frame, text, cy, font, scale, color, thickness=1):
    h_frame, w_frame = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, ((w_frame - tw) // 2, cy),
                font, scale, color, thickness, cv2.LINE_AA)
 
 
def draw_timer_arc(frame, cx, cy, radius, fraction, color, bg_color=(40, 40, 60)):
    """Draw a circular countdown arc."""
    cv2.circle(frame, (cx, cy), radius, bg_color, 4, cv2.LINE_AA)
    if fraction > 0:
        angle = int(360 * fraction)
        axes  = (radius, radius)
        cv2.ellipse(frame, (cx, cy), axes, -90, 0, angle, color, 4, cv2.LINE_AA)
 
 
def draw_menu(frame, score, best):
    h, w = frame.shape[:2]
    draw_panel(frame, w//2 - 280, h//2 - 200, 560, 400, alpha=0.88)
 
    center_text(frame, "GESTURE GAME", h//2 - 160,
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 180), 2)
    center_text(frame, f"Score {ROUNDS_TO_WIN} correct gestures to win!",
                h//2 - 110, cv2.FONT_HERSHEY_PLAIN, 1.1, (160, 160, 200))
    center_text(frame, f"You have {int(TIME_LIMIT)}s per gesture.",
                h//2 - 88, cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 120, 160))
 
    center_text(frame, "GESTURES:", h//2 - 55,
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 80, 130))
    gestures_per_col = 4
    col_w = 240
    for i, g in enumerate(GESTURES):
        col = i // gestures_per_col
        row = i %  gestures_per_col
        tx  = w//2 - 220 + col * col_w
        ty  = h//2 - 35 + row * 22
        cv2.putText(frame, g, (tx, ty),
                    cv2.FONT_HERSHEY_PLAIN, 1.05, (180, 220, 180), 1, cv2.LINE_AA)
 
    # Best score
    if best > 0:
        center_text(frame, f"BEST  {best} / {ROUNDS_TO_WIN}",
                    h//2 + 120, cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 180, 255))
 
    # Pulse effect for SPACE prompt
    pulse = int(180 + 75 * abs(np.sin(time.time() * 2.5)))
    center_text(frame, "PRESS  SPACE  TO  START",
                h//2 + 155, cv2.FONT_HERSHEY_DUPLEX, 0.65, (pulse, pulse, 255), 1)
 
 
def draw_playing(frame, target, current_gesture, time_left, hold_progress,
                 score, round_num, match):
    h, w = frame.shape[:2]
 
    # ── Top HUD ──
    draw_panel(frame, 0, 0, w, 50, alpha=0.80, color=(8, 8, 18), border=(30, 30, 60))
    cv2.putText(frame, f"SCORE: {score}/{ROUNDS_TO_WIN}",
                (14, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ROUND {round_num}",
                (w//2 - 50, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 255), 1, cv2.LINE_AA)
 
    # Timer arc top-right
    arc_cx, arc_cy = w - 50, 50
    fraction = time_left / TIME_LIMIT
    arc_color = (0, 255, 100) if fraction > 0.5 else (0, 200, 255) if fraction > 0.25 else (0, 60, 255)
    draw_timer_arc(frame, arc_cx, arc_cy, 28, fraction, arc_color)
    cv2.putText(frame, f"{time_left:.1f}", (arc_cx - 18, arc_cy + 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, arc_color, 1, cv2.LINE_AA)
 
    # ── Target gesture box ──
    box_y = h - 175
    draw_panel(frame, w//2 - 240, box_y, 480, 130, alpha=0.85,
               color=(10, 10, 22), border=(50, 50, 100))
 
    cv2.putText(frame, "MAKE THIS GESTURE:", (w//2 - 155, box_y + 26),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (80, 80, 140), 1, cv2.LINE_AA)
 
    target_color = (0, 255, 180) if not match else (0, 255, 80)
    center_text(frame, target, box_y + 75,
                cv2.FONT_HERSHEY_DUPLEX, 1.3, target_color, 2)
 
    # ── Current gesture readout ──
    if current_gesture:
        detected_color = (0, 255, 80) if match else (180, 180, 220)
        cv2.putText(frame, f"YOU: {current_gesture}",
                    (14, h - 14), cv2.FONT_HERSHEY_PLAIN, 1.1, detected_color, 1, cv2.LINE_AA)
 
    # ── Hold progress bar ──
    if match and hold_progress > 0:
        bar_x, bar_y, bar_w, bar_h_sz = w//2 - 200, box_y - 22, 400, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_sz),
                      (30, 30, 50), -1)
        fill = int(bar_w * min(hold_progress, 1.0))
        bar_color = (0, 200 + int(55 * hold_progress), 80)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h_sz),
                      bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_sz),
                      (60, 60, 100), 1)
        cv2.putText(frame, "HOLD!", (bar_x + bar_w + 8, bar_y + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
 
 
def draw_result(frame, success, target, score, round_num):
    h, w = frame.shape[:2]
    if success:
        color  = (0, 255, 120)
        title  = "NICE ONE!"
        sub    = f"That was  {target}"
    else:
        color  = (0, 80, 255)
        title  = "TIME'S UP!"
        sub    = f"It was:  {target}"
 
    draw_panel(frame, w//2 - 240, h//2 - 100, 480, 200, alpha=0.90)
    center_text(frame, title, h//2 - 50, cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)
    center_text(frame, sub,   h//2 + 10, cv2.FONT_HERSHEY_PLAIN,  1.2, (200, 200, 220))
    center_text(frame, f"Score: {score} / {ROUNDS_TO_WIN}",
                h//2 + 45, cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 255))
 
    pulse = int(160 + 95 * abs(np.sin(time.time() * 3)))
    center_text(frame, "SPACE  to continue",
                h//2 + 80, cv2.FONT_HERSHEY_PLAIN, 1.1, (pulse, pulse, 255))
 
 
def draw_endscreen(frame, won, score, best):
    h, w = frame.shape[:2]
    draw_panel(frame, w//2 - 280, h//2 - 180, 560, 360, alpha=0.92)
 
    if won:
        title = "YOU WIN!"
        color = (0, 255, 150)
        sub   = "Flawless gesture skills."
    else:
        title = "GAME OVER"
        color = (0, 80, 255)
        sub   = "Better luck next time!"
 
    center_text(frame, title, h//2 - 110, cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
    center_text(frame, sub,   h//2 - 50,  cv2.FONT_HERSHEY_PLAIN,  1.2, (180, 180, 220))
    center_text(frame, f"Final score:  {score} / {ROUNDS_TO_WIN}",
                h//2 + 10, cv2.FONT_HERSHEY_DUPLEX, 0.75, (220, 220, 255), 1)
    center_text(frame, f"Best ever:    {best} / {ROUNDS_TO_WIN}",
                h//2 + 50, cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 180, 255), 1)
 
    pulse = int(160 + 95 * abs(np.sin(time.time() * 2.5)))
    center_text(frame, "SPACE  to play again   |   Q  to quit",
                h//2 + 120, cv2.FONT_HERSHEY_PLAIN, 1.1, (pulse, pulse, 255))
 
 
# ─── MAIN ────────────────────────────────────────────
 
def main():
    mp_hands    = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
    )
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open webcam.")
        return
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
    state         = STATE_MENU
    score         = 0
    round_num     = 0
    best_score    = 0
    target        = None
    round_start   = 0.0
    hold_start    = None   # when current hold began
    current_gest  = None
    fps_times     = deque(maxlen=FPS_SMOOTH)
 
    def new_round():
        nonlocal target, round_start, hold_start, current_gest
        remaining = [g for g in GESTURES if g != target]  # avoid repeats
        target      = random.choice(remaining)
        round_start = time.time()
        hold_start  = None
        current_gest= None
 
    print("\n🟢  Gesture Game running — press SPACE to start, Q to quit\n")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if MIRROR:
            frame = cv2.flip(frame, 1)
 
        h, w = frame.shape[:2]
        t0   = time.time()
 
        # ── MediaPipe ────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb)
 
        current_gest = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            draw_skeleton(frame, lm, w, h)
            current_gest = detect_gesture(lm)
 
        # ── State machine ────────────────────────────
        if state == STATE_MENU:
            draw_menu(frame, score, best_score)
 
        elif state == STATE_PLAYING:
            time_left = TIME_LIMIT - (time.time() - round_start)
            match     = (current_gest == target)
 
            # Hold logic
            if match:
                if hold_start is None:
                    hold_start = time.time()
                hold_progress = (time.time() - hold_start) / HOLD_TIME
            else:
                hold_start    = None
                hold_progress = 0.0
 
            draw_playing(frame, target, current_gest, max(0, time_left),
                         hold_progress, score, round_num, match)
 
            # Success: held long enough
            if match and hold_progress >= 1.0:
                score += 1
                state  = STATE_SUCCESS if score < ROUNDS_TO_WIN else STATE_WIN
 
            # Timeout
            elif time_left <= 0:
                state = STATE_FAIL if score < ROUNDS_TO_WIN else STATE_WIN
 
        elif state == STATE_SUCCESS:
            draw_playing(frame, target, current_gest, 0, 1.0, score, round_num, True)
            draw_result(frame, True, target, score, round_num)
 
        elif state == STATE_FAIL:
            draw_playing(frame, target, current_gest, 0, 0, score, round_num, False)
            draw_result(frame, False, target, score, round_num)
 
        elif state in (STATE_WIN, STATE_LOSE):
            best_score = max(best_score, score)
            draw_endscreen(frame, state == STATE_WIN, score, best_score)
 
        # ── FPS ──────────────────────────────────────
        fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        cv2.putText(frame, f"{fps:.0f} fps", (8, h - 8),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (50, 50, 80), 1, cv2.LINE_AA)
 
        cv2.imshow("GESTURE GAME", frame)
 
        # ── Keys ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
 
        if key in (ord('q'), 27):
            break
 
        elif key == ord(' '):
            if state == STATE_MENU:
                score     = 0
                round_num = 1
                target    = None
                new_round()
                state = STATE_PLAYING
 
            elif state in (STATE_SUCCESS, STATE_FAIL):
                round_num += 1
                new_round()
                state = STATE_PLAYING
 
            elif state in (STATE_WIN, STATE_LOSE):
                score     = 0
                round_num = 1
                target    = None
                new_round()
                state = STATE_PLAYING
 
    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()
    print("\n👋  Thanks for playing!")
 
 
if __name__ == "__main__":
    main()