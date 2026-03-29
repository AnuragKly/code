# main.py  —  Entry point & mode integrator
#
# Owns the webcam, MediaPipe, and the main menu.
# Delegates each frame to the active mode object.
#
# File layout:
#   main.py        ← you are here  (routing + main menu + webcam loop)
#   mode_game.py   ← Game Mode     (edit to change game rules / UI)
#   mode_free.py   ← Free Mode     (edit to change live-detection display)
#   mode_teach.py  ← Teach Mode    (edit to change how signs are recorded)
#   shared.py      ← Config, gesture detection, drawing helpers (shared by all)

import cv2
import numpy as np
import time
from collections import deque

import mediapipe as mp

from shared import (
    MIRROR, CONFIDENCE, FPS_SMOOTH,
    load_custom_signs, detect_gesture, draw_skeleton, draw_panel, center_text,
)
from mode_game  import GameMode
from mode_free  import FreeMode
from mode_teach import TeachMode


# ─── App-level states ─────────────────────────────────
ST_MAIN_MENU = "MAIN_MENU"
ST_GAME      = "GAME"
ST_FREE      = "FREE"
ST_TEACH     = "TEACH"


# ════════════════════════════════════════════════════
#  MAIN MENU DRAWING
# ════════════════════════════════════════════════════

def draw_main_menu(frame, best_score, custom_signs):
    h, w = frame.shape[:2]
    draw_panel(frame, w//2 - 310, h//2 - 235, 620, 470, alpha=0.90)

    center_text(frame, "GESTURE  APP",
                h//2 - 198, cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 180), 2)
    center_text(frame, "Choose a mode:",
                h//2 - 150, cv2.FONT_HERSHEY_PLAIN, 1.1, (140, 140, 200))

    modes = [
        ("1", "GAME MODE",  "Score 5 correct gestures to win"),
        ("2", "FREE MODE",  "Live gesture detection — no pressure"),
        ("3", "TEACH MODE", f"Record custom hand signs  ({len(custom_signs)} saved)"),
    ]
    for i, (key_label, title, desc) in enumerate(modes):
        by = h//2 - 100 + i * 98
        draw_panel(frame, w//2 - 260, by, 520, 82, alpha=0.70,
                   color=(18, 20, 42), border=(55, 60, 130))
        cv2.putText(frame, f"[{key_label}]", (w//2 - 245, by + 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, title, (w//2 - 188, by + 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (w//2 - 188, by + 62),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, (120, 120, 175), 1, cv2.LINE_AA)

    if best_score > 0:
        center_text(frame, f"BEST GAME SCORE:  {best_score} / 5",
                    h//2 + 200, cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 180, 255))

    pulse = int(170 + 85 * abs(np.sin(time.time() * 2.0)))
    center_text(frame, "Press  1 / 2 / 3  to select   |   Q  to quit",
                h//2 + 222, cv2.FONT_HERSHEY_PLAIN, 0.95, (pulse, pulse, 200))


# ════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════

def main():
    # ── MediaPipe setup ──────────────────────────────
    mp_hands    = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
    )

    # ── Webcam ───────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── App state ────────────────────────────────────
    state        = ST_MAIN_MENU
    custom_signs = load_custom_signs()   # shared mutable dict
    best_score   = 0
    fps_times    = deque(maxlen=FPS_SMOOTH)

    # ── Mode objects (created once, reused) ──────────
    game_mode  = GameMode()
    free_mode  = FreeMode()
    teach_mode = TeachMode()

    print("\n🟢  Gesture App running")
    print("    1 = Game   2 = Free   3 = Teach   Q = quit\n")

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

        lm           = None
        current_gest = None
        gest_type    = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            draw_skeleton(frame, lm, w, h)
            current_gest, gest_type = detect_gesture(lm, custom_signs)

        # ════════════════════════════════════════════
        #  ROUTING
        # ════════════════════════════════════════════

        if state == ST_MAIN_MENU:
            draw_main_menu(frame, best_score, custom_signs)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('1'):
                game_mode.reset()
                game_mode.set_custom_signs(custom_signs)
                state = ST_GAME
            elif key == ord('2'):
                state = ST_FREE
            elif key == ord('3'):
                state = ST_TEACH

        elif state == ST_GAME:
            result = game_mode.update(frame, current_gest)
            # Sync best score back
            best_score = max(best_score, game_mode.best_score)
            if result == "MAIN_MENU":
                state = ST_MAIN_MENU

        elif state == ST_FREE:
            result = free_mode.update(frame, current_gest, gest_type, custom_signs)
            if result == "MAIN_MENU":
                state = ST_MAIN_MENU

        elif state == ST_TEACH:
            # custom_signs is mutated in-place by teach_mode; pass by reference
            result = teach_mode.update(frame, lm, custom_signs)
            if result == "MAIN_MENU":
                # Refresh game mode's target pool if new signs were added
                game_mode.set_custom_signs(custom_signs)
                state = ST_MAIN_MENU

        # ── FPS overlay (subtle, bottom-left) ───────
        fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        cv2.putText(frame, f"{fps:.0f} fps", (8, h - 8),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (40, 40, 60), 1, cv2.LINE_AA)

        cv2.imshow("GESTURE APP", frame)

    # ── Cleanup ──────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()
    print("\n👋  Thanks for playing!")


if __name__ == "__main__":
    main()