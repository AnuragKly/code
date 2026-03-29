# mode_game.py  —  Game Mode
#
# The player must perform a randomly chosen gesture within TIME_LIMIT seconds.
# Holding the correct gesture for HOLD_TIME counts as a success.
# Score ROUNDS_TO_WIN rounds to win the game.
#
# States returned to main.py:
#   "MAIN_MENU"   — user pressed M to go back
#   (stays internal otherwise)

import cv2
import numpy as np
import time
import random

from shared import (
    TIME_LIMIT, HOLD_TIME, ROUNDS_TO_WIN,
    BUILT_IN_GESTURES,
    draw_panel, center_text, draw_timer_arc,
)

# ─── Internal sub-states ─────────────────────────────
_ST_PLAYING = "playing"
_ST_SUCCESS = "success"
_ST_FAIL    = "fail"
_ST_WIN     = "win"


class GameMode:
    """Encapsulates all Game Mode logic and drawing."""

    def __init__(self):
        self.reset()

    # ── Public API ───────────────────────────────────

    def reset(self):
        """Start a fresh game."""
        self._state      = _ST_PLAYING
        self._score      = 0
        self._round      = 1
        self._target     = None
        self._round_start= 0.0
        self._hold_start = None
        self._new_round()

    def update(self, frame, current_gesture):
        """
        Called every frame.
        Returns "MAIN_MENU" if the user exits, else None (stay in game).
        current_gesture: string label or None (from shared.detect_gesture)
        """
        key = cv2.waitKey(1) & 0xFF

        if self._state == _ST_PLAYING:
            self._update_playing(frame, current_gesture)
        elif self._state == _ST_SUCCESS:
            self._draw_playing(frame, current_gesture, 0, 1.0, True)
            self._draw_result(frame, success=True)
        elif self._state == _ST_FAIL:
            self._draw_playing(frame, current_gesture, 0, 0.0, False)
            self._draw_result(frame, success=False)
        elif self._state == _ST_WIN:
            self._draw_endscreen(frame)

        return self._handle_keys(key)

    @property
    def best_score(self):
        return self._best if hasattr(self, "_best") else 0

    # ── Internal helpers ─────────────────────────────

    def _new_round(self):
        pool           = [g for g in self._all_targets() if g != self._target]
        self._target     = random.choice(pool)
        self._round_start= time.time()
        self._hold_start = None

    def _all_targets(self):
        """Built-in gestures only. Override or extend to add custom signs."""
        return list(BUILT_IN_GESTURES)

    def set_custom_signs(self, custom_signs):
        """Called by main.py so game knows about custom signs as targets."""
        self._custom_sign_names = list(custom_signs.keys())

    def _all_targets(self):
        base = list(BUILT_IN_GESTURES)
        if hasattr(self, "_custom_sign_names"):
            base += [f"[{n}]" for n in self._custom_sign_names]
        return base

    # ── Drawing ──────────────────────────────────────

    def _update_playing(self, frame, current_gesture):
        time_left = TIME_LIMIT - (time.time() - self._round_start)
        match     = (current_gesture == self._target)

        if match:
            if self._hold_start is None:
                self._hold_start = time.time()
            hold_prog = (time.time() - self._hold_start) / HOLD_TIME
        else:
            self._hold_start = None
            hold_prog        = 0.0

        self._draw_playing(frame, current_gesture, max(0, time_left), hold_prog, match)

        if match and hold_prog >= 1.0:
            self._score += 1
            self._state = _ST_WIN if self._score >= ROUNDS_TO_WIN else _ST_SUCCESS
        elif time_left <= 0:
            self._state = _ST_FAIL

    def _draw_playing(self, frame, current_gesture, time_left, hold_prog, match):
        h, w = frame.shape[:2]

        # Top HUD
        draw_panel(frame, 0, 0, w, 50, alpha=0.80, color=(8, 8, 18), border=(30, 30, 60))
        cv2.putText(frame, f"SCORE: {self._score}/{ROUNDS_TO_WIN}",
                    (14, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ROUND {self._round}",
                    (w//2 - 50, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 255), 1, cv2.LINE_AA)

        # Timer arc
        arc_cx, arc_cy = w - 50, 50
        frac      = time_left / TIME_LIMIT
        arc_color = (0, 255, 100) if frac > 0.5 else (0, 200, 255) if frac > 0.25 else (0, 60, 255)
        draw_timer_arc(frame, arc_cx, arc_cy, 28, frac, arc_color)
        cv2.putText(frame, f"{time_left:.1f}", (arc_cx - 18, arc_cy + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, arc_color, 1, cv2.LINE_AA)

        # Target box
        box_y = h - 175
        draw_panel(frame, w//2 - 240, box_y, 480, 130, alpha=0.85,
                   color=(10, 10, 22), border=(50, 50, 100))
        cv2.putText(frame, "MAKE THIS GESTURE:", (w//2 - 155, box_y + 26),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (80, 80, 140), 1, cv2.LINE_AA)
        target_color = (0, 255, 80) if match else (0, 255, 180)
        center_text(frame, self._target, box_y + 75,
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, target_color, 2)

        # Current gesture readout
        if current_gesture:
            dcol = (0, 255, 80) if match else (180, 180, 220)
            cv2.putText(frame, f"YOU: {current_gesture}",
                        (14, h - 14), cv2.FONT_HERSHEY_PLAIN, 1.1, dcol, 1, cv2.LINE_AA)

        # Hold progress bar
        if match and hold_prog > 0:
            bx, by, bw, bht = w//2 - 200, box_y - 22, 400, 12
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bht), (30, 30, 50), -1)
            fill = int(bw * min(hold_prog, 1.0))
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bht),
                          (0, 200 + int(55 * hold_prog), 80), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bht), (60, 60, 100), 1)
            cv2.putText(frame, "HOLD!", (bx + bw + 8, by + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

    def _draw_result(self, frame, success):
        h, w = frame.shape[:2]
        color = (0, 255, 120) if success else (0, 80, 255)
        title = "NICE ONE!"     if success else "TIME'S UP!"
        sub   = f"That was  {self._target}" if success else f"It was:  {self._target}"

        draw_panel(frame, w//2 - 240, h//2 - 100, 480, 200, alpha=0.90)
        center_text(frame, title, h//2 - 50, cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)
        center_text(frame, sub,   h//2 + 10, cv2.FONT_HERSHEY_PLAIN,  1.2, (200, 200, 220))
        center_text(frame, f"Score: {self._score} / {ROUNDS_TO_WIN}",
                    h//2 + 45, cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 255))
        pulse = int(160 + 95 * abs(np.sin(time.time() * 3)))
        center_text(frame, "SPACE  to continue   |   M  for menu",
                    h//2 + 80, cv2.FONT_HERSHEY_PLAIN, 1.0, (pulse, pulse, 255))

    def _draw_endscreen(self, frame):
        h, w = frame.shape[:2]
        if not hasattr(self, "_best"):
            self._best = 0
        self._best = max(self._best, self._score)

        draw_panel(frame, w//2 - 280, h//2 - 180, 560, 360, alpha=0.92)
        center_text(frame, "YOU WIN!", h//2 - 110,
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 150), 3)
        center_text(frame, "Flawless gesture skills.",
                    h//2 - 50, cv2.FONT_HERSHEY_PLAIN, 1.2, (180, 180, 220))
        center_text(frame, f"Final score:  {self._score} / {ROUNDS_TO_WIN}",
                    h//2 + 10, cv2.FONT_HERSHEY_DUPLEX, 0.75, (220, 220, 255), 1)
        center_text(frame, f"Best ever:    {self._best} / {ROUNDS_TO_WIN}",
                    h//2 + 50, cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 180, 255), 1)
        pulse = int(160 + 95 * abs(np.sin(time.time() * 2.5)))
        center_text(frame, "SPACE  to play again   |   M  for menu",
                    h//2 + 120, cv2.FONT_HERSHEY_PLAIN, 1.1, (pulse, pulse, 255))

    # ── Key handling ─────────────────────────────────

    def _handle_keys(self, key):
        if key == ord('m'):
            return "MAIN_MENU"

        if self._state in (_ST_SUCCESS, _ST_FAIL):
            if key == ord(' '):
                self._round += 1
                self._state  = _ST_PLAYING
                self._new_round()

        elif self._state == _ST_WIN:
            if key == ord(' '):
                self._score = 0
                self._round = 1
                self._state = _ST_PLAYING
                self._new_round()

        return None   # stay in game mode