# mode_free.py  —  Free Mode
#
# Live gesture detection with no scoring or time pressure.
# Shows built-in and custom labels, confidence bar, and a gesture history log.
#
# Improvements over v1:
#   • Confidence bar rendered for every detected gesture
#   • Gesture history: last 6 detected labels shown as a fading log
#   • Per-finger open/closed state displayed as a mini indicator strip
#   • Custom signs panel includes sample count and a quality hint
#   • Hint footer explains detection method

import cv2
import time
from collections import deque

from shared import (
    BUILT_IN_GESTURES,
    draw_panel, center_text, draw_confidence_bar,
)


class FreeMode:
    """All Free Mode logic and rendering."""

    def __init__(self):
        # History log: (label, gesture_type, timestamp)
        self._history = deque(maxlen=6)
        self._last_pushed = None   # avoid logging the same label repeatedly

    # ── Public API ───────────────────────────────────

    def update(self, frame, lm, current_gesture, gesture_type, confidence, custom_signs):
        """
        Called every frame by main.py.
        lm              : raw MediaPipe landmark list (or None) for finger-state strip
        current_gesture : smoothed label or None
        gesture_type    : "builtin" | "custom" | None
        confidence      : 0–1 float
        Returns "MAIN_MENU" on M key, else None.
        """
        # Push to history only when label changes
        if current_gesture and current_gesture != self._last_pushed:
            self._history.append((current_gesture, gesture_type, time.time()))
            self._last_pushed = current_gesture
        elif not current_gesture:
            self._last_pushed = None

        self._draw(frame, lm, current_gesture, gesture_type, confidence, custom_signs)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            return "MAIN_MENU"
        return None

    # ── Drawing ──────────────────────────────────────

    def _draw(self, frame, lm, current_gesture, gesture_type, confidence, custom_signs):
        h, w = frame.shape[:2]

        # ── Top bar ──────────────────────────────────
        draw_panel(frame, 0, 0, w, 52, alpha=0.82,
                   color=(8, 8, 18), border=(30, 30, 60), radius=0)
        cv2.putText(frame, "FREE MODE  —  show any gesture",
                    (14, 34), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "M = menu",
                    (w - 125, 34), cv2.FONT_HERSHEY_PLAIN, 1.05, (70, 70, 115), 1, cv2.LINE_AA)

        # ── Main detection readout ────────────────────
        box_h = 95
        box_y = h - box_h - 12
        bdr   = (0, 180, 80) if current_gesture else (40, 40, 80)
        draw_panel(frame, w//2 - 290, box_y, 580, box_h, alpha=0.88,
                   color=(10, 10, 22), border=bdr, radius=10)

        if current_gesture:
            tag_color = (0, 200, 255) if gesture_type == "custom" else (0, 255, 180)
            center_text(frame, current_gesture, box_y + 52,
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, tag_color, 2)
            tag = "(custom sign)" if gesture_type == "custom" else "(built-in)"
            center_text(frame, tag, box_y + 76,
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (90, 90, 140))

            # Confidence bar (only meaningful for custom; built-ins always 1.0)
            bar_x = w//2 - 100
            draw_confidence_bar(frame, bar_x, box_y + 6, 200, 14, confidence, "Conf")
        else:
            center_text(frame, "No gesture detected",
                        box_y + 54, cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 50, 80))

        # ── Finger-state strip ────────────────────────
        if lm:
            self._draw_finger_strip(frame, lm, w, h)

        # ── Gesture history log ───────────────────────
        self._draw_history(frame, h)

        # ── Custom signs sidebar ──────────────────────
        if custom_signs:
            self._draw_custom_sidebar(frame, custom_signs, h)

        # ── Footer hint ───────────────────────────────
        hint = "Built-in: rule-based  |  Custom: 5-NN classifier with z-depth"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_PLAIN, 0.78, 1)
        cv2.putText(frame, hint, ((w - tw) // 2, h - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.78, (38, 38, 65), 1, cv2.LINE_AA)

    def _draw_finger_strip(self, frame, lm, w, h):
        """5 small circles: filled = finger open, hollow = closed."""
        names  = ["T", "I", "M", "R", "P"]
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]
        colors  = list(__import__("shared").FINGER_COLORS.values())

        sx = w // 2 - len(names) * 22
        sy = frame.shape[0] - 120

        for i, (name, tip, pip, col) in enumerate(zip(names, tip_ids, pip_ids, colors)):
            cx = sx + i * 44
            # Thumb uses x-axis comparison; fingers use y-axis
            if i == 0:
                is_open = lm[4].x < lm[3].x   # simplified; handedness-corrected in detect
            else:
                is_open = lm[tip].y < lm[pip].y

            if is_open:
                cv2.circle(frame, (cx, sy), 12, col, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, sy), 14, col,  1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, sy), 12, (25, 25, 45), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, sy), 14, col, 1, cv2.LINE_AA)

            cv2.putText(frame, name, (cx - 4, sy + 28),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, col, 1, cv2.LINE_AA)

    def _draw_history(self, frame, h):
        """Fading log of recent gestures, bottom-left."""
        if not self._history:
            return
        now = time.time()
        lx, ly = 12, h - 140
        cv2.putText(frame, "Recent:", (lx, ly),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (55, 55, 90), 1, cv2.LINE_AA)
        for i, (label, gtype, ts) in enumerate(reversed(self._history)):
            age   = now - ts
            alpha = max(0, 1.0 - age / 8.0)   # fade over 8 seconds
            col   = (0, 200, 255) if gtype == "custom" else (0, 255, 160)
            col   = tuple(int(c * alpha) for c in col)
            cv2.putText(frame, label, (lx, ly + 16 + i * 18),
                        cv2.FONT_HERSHEY_PLAIN, 0.88, col, 1, cv2.LINE_AA)

    def _draw_custom_sidebar(self, frame, custom_signs, h):
        """Right-side panel listing trained custom signs."""
        w = frame.shape[1]
        px, py, pw = w - 210, 60, 200
        n   = len(custom_signs)
        ph  = 28 + n * 24 + 16
        draw_panel(frame, px, py, pw, ph, alpha=0.75,
                   color=(12, 14, 28), border=(40, 50, 100), radius=8)
        cv2.putText(frame, "Custom signs", (px + 8, py + 18),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, (80, 80, 140), 1, cv2.LINE_AA)
        for i, (name, samples) in enumerate(custom_signs.items()):
            n_s   = len(samples)
            # Quality hint: < 20 samples = weak, ≥ 40 = good
            quality = "✓" if n_s >= 40 else "~" if n_s >= 20 else "!"
            q_col   = (0, 200, 100) if n_s >= 40 else (0, 180, 255) if n_s >= 20 else (0, 80, 255)
            text    = f"  {quality} [{name}] {n_s}"
            cv2.putText(frame, text, (px + 6, py + 38 + i * 24),
                        cv2.FONT_HERSHEY_PLAIN, 0.88, q_col, 1, cv2.LINE_AA)