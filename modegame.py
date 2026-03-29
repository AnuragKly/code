# mode_game.py  —  Game Mode
#
# The player performs a randomly chosen gesture within a time limit.
# Hold the correct gesture for HOLD_TIME seconds to score.
# Score ROUNDS_TO_WIN rounds to win the game.
#
# Improvements over v1:
#   • Difficulty scaling: TIME_LIMIT shrinks slightly each round (capped at 5 s)
#   • Streak counter: consecutive hits shown in HUD with flash effect
#   • Confidence bar shown when current gesture is detected
#   • _handle_keys called before draw so ESC key in result screen is instant
#   • Custom signs included as targets automatically via set_custom_signs()
#   • Game Over state added (if player runs out of time 3× in a row)
#   • _best persisted as instance attr, not checked with hasattr

import cv2
import numpy as np
import time
import random

from shared import (
    TIME_LIMIT, HOLD_TIME, ROUNDS_TO_WIN,
    BUILT_IN_GESTURES,
    draw_panel, center_text, draw_timer_arc, draw_confidence_bar,
)

_ST_PLAYING  = "playing"
_ST_SUCCESS  = "success"
_ST_FAIL     = "fail"
_ST_WIN      = "win"
_ST_LOSE     = "lose"   # 3 failures in a row ends the game


class GameMode:
    """All Game Mode logic and rendering."""

    def __init__(self):
        self._best         = 0
        self._custom_names = []
        self.reset()

    # ── Public API ───────────────────────────────────

    def reset(self):
        self._state       = _ST_PLAYING
        self._score       = 0
        self._round       = 1
        self._streak      = 0
        self._fails       = 0          # consecutive failures
        self._target      = None
        self._round_start = 0.0
        self._hold_start  = None
        self._flash_until = 0.0        # timestamp for streak flash
        self._new_round()

    def set_custom_signs(self, custom_signs):
        """Sync custom sign names so they can appear as targets."""
        self._custom_names = list(custom_signs.keys())

    def update(self, frame, current_gesture, confidence):
        """
        Called every frame by main.py.
        current_gesture : smoothed label or None
        confidence      : 0–1 float
        Returns "MAIN_MENU" if user exits, else None.
        """
        key = cv2.waitKey(1) & 0xFF

        if self._state == _ST_PLAYING:
            self._update_playing(frame, current_gesture, confidence)
        elif self._state == _ST_SUCCESS:
            self._draw_playing(frame, current_gesture, confidence, 0, 1.0, True)
            self._draw_result(frame, success=True)
        elif self._state == _ST_FAIL:
            self._draw_playing(frame, current_gesture, confidence, 0, 0.0, False)
            self._draw_result(frame, success=False)
        elif self._state == _ST_WIN:
            self._best = max(self._best, self._score)
            self._draw_endscreen(frame, won=True)
        elif self._state == _ST_LOSE:
            self._best = max(self._best, self._score)
            self._draw_endscreen(frame, won=False)

        return self._handle_keys(key)

    @property
    def best_score(self):
        return self._best

    # ── Internals ────────────────────────────────────

    def _all_targets(self):
        return list(BUILT_IN_GESTURES) + [f"[{n}]" for n in self._custom_names]

    def _round_time_limit(self):
        """Difficulty scaling: each round removes 0.3 s, floor at 5 s."""
        return max(5.0, TIME_LIMIT - (self._round - 1) * 0.3)

    def _new_round(self):
        pool             = [g for g in self._all_targets() if g != self._target]
        self._target     = random.choice(pool) if pool else random.choice(self._all_targets())
        self._round_start= time.time()
        self._hold_start = None

    def _update_playing(self, frame, current_gesture, confidence):
        tl    = self._round_time_limit()
        time_left = tl - (time.time() - self._round_start)
        match     = (current_gesture == self._target)

        if match:
            if self._hold_start is None:
                self._hold_start = time.time()
            hold_prog = (time.time() - self._hold_start) / HOLD_TIME
        else:
            self._hold_start = None
            hold_prog        = 0.0

        self._draw_playing(frame, current_gesture, confidence,
                           max(0.0, time_left), hold_prog, match)

        if match and hold_prog >= 1.0:
            self._score  += 1
            self._streak += 1
            self._fails   = 0
            self._flash_until = time.time() + 0.6
            self._state = _ST_WIN if self._score >= ROUNDS_TO_WIN else _ST_SUCCESS

        elif time_left <= 0:
            self._streak = 0
            self._fails += 1
            self._state  = _ST_LOSE if self._fails >= 3 else _ST_FAIL

    # ── Drawing ──────────────────────────────────────

    def _draw_playing(self, frame, current_gesture, confidence,
                      time_left, hold_prog, match):
        h, w = frame.shape[:2]

        # ── Top HUD ──────────────────────────────────
        draw_panel(frame, 0, 0, w, 52, alpha=0.82,
                   color=(8, 8, 18), border=(30, 30, 60), radius=0)

        cv2.putText(frame, f"SCORE  {self._score} / {ROUNDS_TO_WIN}",
                    (14, 34), cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 180), 1, cv2.LINE_AA)

        tl_str = f"ROUND {self._round}"
        (tw, _), _ = cv2.getTextSize(tl_str, cv2.FONT_HERSHEY_DUPLEX, 0.68, 1)
        cv2.putText(frame, tl_str,
                    (w // 2 - tw // 2, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (200, 200, 255), 1, cv2.LINE_AA)

        # Streak flash
        if self._streak >= 2:
            flash = time.time() < self._flash_until
            scol  = (0, 255, 100) if flash else (0, 180, 80)
            cv2.putText(frame, f"🔥 x{self._streak}",
                        (w - 110, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, scol, 1, cv2.LINE_AA)

        # Difficulty indicator (tiny, top-right corner)
        tl_limit = self._round_time_limit()
        cv2.putText(frame, f"Limit {tl_limit:.0f}s",
                    (w - 100, h - 8), cv2.FONT_HERSHEY_PLAIN, 0.75,
                    (50, 50, 80), 1, cv2.LINE_AA)

        # ── Timer arc ────────────────────────────────
        arc_cx, arc_cy = w - 55, 62
        frac      = time_left / tl_limit
        arc_color = (0, 255, 100) if frac > 0.5 else (0, 200, 255) if frac > 0.25 else (0, 60, 255)
        draw_timer_arc(frame, arc_cx, arc_cy, 30, frac, arc_color)
        tstr = f"{time_left:.1f}"
        (tw, _), _ = cv2.getTextSize(tstr, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(frame, tstr, (arc_cx - tw // 2, arc_cy + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, arc_color, 1, cv2.LINE_AA)

        # ── Target gesture box ────────────────────────
        box_y = h - 185
        bdr   = (0, 180, 80) if match else (50, 50, 100)
        draw_panel(frame, w//2 - 250, box_y, 500, 140, alpha=0.88,
                   color=(10, 10, 22), border=bdr, radius=10)

        cv2.putText(frame, "MAKE THIS GESTURE",
                    (w//2 - 155, box_y + 26),
                    cv2.FONT_HERSHEY_PLAIN, 1.05, (70, 70, 130), 1, cv2.LINE_AA)

        target_color = (0, 255, 80) if match else (0, 240, 180)
        center_text(frame, self._target, box_y + 85,
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, target_color, 2)

        # ── Detected gesture + confidence ─────────────
        if current_gesture:
            dcol = (0, 255, 80) if match else (170, 170, 210)
            cv2.putText(frame, f"YOU: {current_gesture}",
                        (14, h - 20), cv2.FONT_HERSHEY_PLAIN, 1.1, dcol, 1, cv2.LINE_AA)
            if confidence < 1.0:   # skip bar for built-ins (always 1.0)
                draw_confidence_bar(frame, 14, h - 16, 180, 12, confidence, "Conf")

        # ── Hold progress bar ─────────────────────────
        if match and hold_prog > 0:
            bx, by2, bw, bht = w//2 - 210, box_y - 24, 420, 14
            cv2.rectangle(frame, (bx, by2), (bx + bw, by2 + bht), (25, 25, 45), -1)
            fill = int(bw * min(hold_prog, 1.0))
            g    = 160 + int(95 * hold_prog)
            cv2.rectangle(frame, (bx, by2), (bx + fill, by2 + bht), (0, g, 70), -1)
            cv2.rectangle(frame, (bx, by2), (bx + bw, by2 + bht), (60, 60, 100), 1)
            cv2.putText(frame, "HOLD!",
                        (bx + bw + 10, by2 + 11),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)

    def _draw_result(self, frame, success):
        h, w = frame.shape[:2]
        color = (0, 255, 120) if success else (0, 80, 255)
        title = "NICE ONE!" if success else "TIME'S UP!"
        sub   = f"That was  {self._target}" if success else f"It was:  {self._target}"

        draw_panel(frame, w//2 - 250, h//2 - 110, 500, 220, alpha=0.92, radius=12)
        center_text(frame, title, h//2 - 55, cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)
        center_text(frame, sub,   h//2 + 10, cv2.FONT_HERSHEY_PLAIN,  1.2, (200, 200, 220))
        center_text(frame, f"Score: {self._score} / {ROUNDS_TO_WIN}",
                    h//2 + 45, cv2.FONT_HERSHEY_DUPLEX, 0.62, (180, 180, 255))
        pulse = int(160 + 95 * abs(np.sin(time.time() * 3)))
        center_text(frame, "SPACE  continue   |   M  menu",
                    h//2 + 85, cv2.FONT_HERSHEY_PLAIN, 1.05, (pulse, pulse, 255))

    def _draw_endscreen(self, frame, won):
        h, w = frame.shape[:2]
        title = "YOU WIN!" if won else "GAME OVER"
        color = (0, 255, 150) if won else (0, 80, 255)
        sub   = "Excellent gesture skills!" if won else "3 misses — better luck next time!"

        draw_panel(frame, w//2 - 290, h//2 - 190, 580, 380, alpha=0.93, radius=14)
        center_text(frame, title, h//2 - 120, cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
        center_text(frame, sub,   h//2 - 60,  cv2.FONT_HERSHEY_PLAIN,  1.15, (180, 180, 220))
        center_text(frame, f"Final score  {self._score} / {ROUNDS_TO_WIN}",
                    h//2 - 5, cv2.FONT_HERSHEY_DUPLEX, 0.78, (220, 220, 255), 1)
        center_text(frame, f"Best ever    {self._best} / {ROUNDS_TO_WIN}",
                    h//2 + 42, cv2.FONT_HERSHEY_DUPLEX, 0.62, (100, 180, 255), 1)

        pulse = int(155 + 100 * abs(np.sin(time.time() * 2.5)))
        center_text(frame, "SPACE  play again   |   M  menu",
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

        elif self._state in (_ST_WIN, _ST_LOSE):
            if key == ord(' '):
                self.reset()

        return None