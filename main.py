# main.py  —  Entry point & mode integrator
#
# Owns the webcam, MediaPipe, the main menu, and the GestureBuffer.
# Delegates each frame to the active mode object.
#
# File layout:
#   main.py        ← you are here  (webcam · routing · main menu · smoothing)
#   mode_game.py   ← Game Mode     (rules · scoring · difficulty · UI)
#   mode_free.py   ← Free Mode     (live detection display · history · sidebar)
#   mode_teach.py  ← Teach Mode    (capture · motion guard · quality · list)
#   shared.py      ← Config · feature extraction · k-NN · GestureBuffer · drawing
#
# Improvements over v1:
#   • GestureBuffer in main.py smooths detection for all modes uniformly
#   • Handedness extracted from MediaPipe result and passed to detector
#   • detect_gesture_raw() replaces old detect_gesture() throughout
#   • Webcam resolution auto-detected with graceful fallback
#   • FPS counter uses exponential moving average (smoother than rolling mean)
#   • Main menu shows live hand skeleton while idle

import cv2
import numpy as np
import time
from collections import deque

import mediapipe as mp

from shared import (
    MIRROR, CONFIDENCE, FPS_SMOOTH,
    load_custom_signs, detect_gesture_raw,
    draw_skeleton, draw_panel, center_text,
    GestureBuffer,
)
from modegame  import GameMode
from modefree  import FreeMode
from modeteach import TeachMode


# ─── App-level states ─────────────────────────────────
ST_MENU  = "MENU"
ST_GAME  = "GAME"
ST_FREE  = "FREE"
ST_TEACH = "TEACH"


# ════════════════════════════════════════════════════
#  MAIN MENU DRAWING
# ════════════════════════════════════════════════════

def draw_main_menu(frame, best_score, custom_signs):
    h, w = frame.shape[:2]
    draw_panel(frame, w//2 - 320, h//2 - 250, 640, 500, alpha=0.90, radius=16)

    center_text(frame, "GESTURE  APP",
                h//2 - 212, cv2.FONT_HERSHEY_DUPLEX, 1.45, (0, 255, 180), 2)
    center_text(frame, "Hand gesture recognition — three modes",
                h//2 - 170, cv2.FONT_HERSHEY_PLAIN, 1.05, (100, 100, 160))

    modes = [
        ("1", "GAME MODE",
         f"Score {5} correct gestures to win"),
        ("2", "FREE MODE",
         "Live detection — confidence · history · finger states"),
        ("3", "TEACH MODE",
         f"Record custom signs  ({len(custom_signs)} saved)  •  k-NN classifier"),
    ]

    for i, (key_label, title, desc) in enumerate(modes):
        by = h//2 - 118 + i * 106
        draw_panel(frame, w//2 - 275, by, 550, 88, alpha=0.72,
                   color=(16, 18, 42), border=(52, 58, 135), radius=10)
        cv2.putText(frame, f"[{key_label}]", (w//2 - 260, by + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, title, (w//2 - 195, by + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (w//2 - 195, by + 66),
                    cv2.FONT_HERSHEY_PLAIN, 0.92, (110, 110, 168), 1, cv2.LINE_AA)

    if best_score > 0:
        center_text(frame, f"BEST GAME SCORE:  {best_score} / 5",
                    h//2 + 208, cv2.FONT_HERSHEY_DUPLEX, 0.52, (100, 180, 255))

    pulse = int(165 + 90 * abs(np.sin(time.time() * 2.0)))
    center_text(frame, "Press  1 / 2 / 3  to select   |   Q  to quit",
                h//2 + 228, cv2.FONT_HERSHEY_PLAIN, 0.98, (pulse, pulse, 200))


# ════════════════════════════════════════════════════
#  WEBCAM SETUP
# ════════════════════════════════════════════════════

def open_webcam(preferred_w=1280, preferred_h=720):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  preferred_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_h)
    # Read back actual resolution (camera may not support preferred)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Webcam opened at {actual_w}×{actual_h}")
    return cap


# ════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════

def main():
    print("\n🟢  Gesture App starting…")

    # ── MediaPipe ────────────────────────────────────
    mp_hands    = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
    )

    # ── Webcam ───────────────────────────────────────
    cap = open_webcam()
    if cap is None:
        print("❌  Could not open webcam.")
        return

    # ── App state ────────────────────────────────────
    state        = ST_MENU
    custom_signs = load_custom_signs()
    best_score   = 0

    # Exponential moving average for FPS
    fps_ema      = 30.0
    fps_alpha    = 2.0 / (FPS_SMOOTH + 1)

    # Single GestureBuffer shared across modes (reset on mode switch)
    gesture_buf = GestureBuffer()

    # ── Mode objects ─────────────────────────────────
    game_mode  = GameMode()
    free_mode  = FreeMode()
    teach_mode = TeachMode()

    print("   1 = Game   2 = Free   3 = Teach   Q = quit\n")

    while True:
        t_frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame read failed — retrying…")
            time.sleep(0.05)
            continue

        if MIRROR:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ── MediaPipe processing ──────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb)

        lm            = None
        is_right_hand = True     # default; overwritten below
        raw_label     = None
        raw_type      = None
        raw_conf      = 0.0

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            draw_skeleton(frame, lm, w, h)

            # Handedness: MediaPipe labels from *its* perspective (unmirrored).
            # After we flip the frame, "Left" becomes the hand on the right of screen.
            if results.multi_handedness:
                mp_label  = results.multi_handedness[0].classification[0].label
                # With MIRROR=True the image is flipped, so handedness is inverted
                if MIRROR:
                    is_right_hand = (mp_label == "Left")
                else:
                    is_right_hand = (mp_label == "Right")

            raw_label, raw_type, raw_conf = detect_gesture_raw(
                lm, is_right_hand, custom_signs
            )

        # Push raw result through smoothing buffer
        smoothed_label, smoothed_conf = gesture_buf.push(raw_label, raw_conf)
        smoothed_type = raw_type if smoothed_label else None

        # ════════════════════════════════════════════
        #  STATE ROUTING
        # ════════════════════════════════════════════

        if state == ST_MENU:
            draw_main_menu(frame, best_score, custom_signs)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('1'):
                gesture_buf.reset()
                game_mode.reset()
                game_mode.set_custom_signs(custom_signs)
                state = ST_GAME
            elif key == ord('2'):
                gesture_buf.reset()
                state = ST_FREE
            elif key == ord('3'):
                gesture_buf.reset()
                state = ST_TEACH

        elif state == ST_GAME:
            result = game_mode.update(frame, smoothed_label, smoothed_conf)
            best_score = max(best_score, game_mode.best_score)
            if result == "MAIN_MENU":
                gesture_buf.reset()
                state = ST_MENU

        elif state == ST_FREE:
            result = free_mode.update(
                frame, lm,
                smoothed_label, smoothed_type, smoothed_conf,
                custom_signs
            )
            if result == "MAIN_MENU":
                gesture_buf.reset()
                state = ST_MENU

        elif state == ST_TEACH:
            result = teach_mode.update(frame, lm, custom_signs)
            if result == "MAIN_MENU":
                # Sync updated custom signs to game mode
                game_mode.set_custom_signs(custom_signs)
                gesture_buf.reset()
                state = ST_MENU

        # ── FPS overlay ──────────────────────────────
        elapsed = time.time() - t_frame_start
        if elapsed > 0:
            fps_ema = fps_alpha * (1.0 / elapsed) + (1 - fps_alpha) * fps_ema
        cv2.putText(frame, f"{fps_ema:.0f} fps",
                    (8, h - 8), cv2.FONT_HERSHEY_PLAIN, 0.85,
                    (38, 38, 58), 1, cv2.LINE_AA)

        cv2.imshow("GESTURE APP", frame)

    # ── Cleanup ──────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()
    print("\n👋  Thanks for playing!")


if __name__ == "__main__":
    main()