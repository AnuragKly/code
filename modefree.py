# mode_free.py  —  Free Mode
#
# Live gesture detection with no scoring or time pressure.
# Shows both built-in and custom sign labels.
# Press M to return to main menu.

import cv2

from shared import draw_panel, center_text


class FreeMode:
    """Encapsulates all Free Mode logic and drawing."""

    # ── Public API ───────────────────────────────────

    def update(self, frame, current_gesture, gesture_type, custom_signs):
        """
        Called every frame.
        Returns "MAIN_MENU" if M is pressed, else None.

        current_gesture : string label or None
        gesture_type    : "builtin" | "custom" | None
        custom_signs    : dict passed in from main.py (read-only here)
        """
        self._draw(frame, current_gesture, gesture_type, custom_signs)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            return "MAIN_MENU"
        return None

    # ── Drawing ──────────────────────────────────────

    def _draw(self, frame, current_gesture, gesture_type, custom_signs):
        h, w = frame.shape[:2]

        # ── Top bar ──────────────────────────────────
        draw_panel(frame, 0, 0, w, 50, alpha=0.80, color=(8, 8, 18), border=(30, 30, 60))
        cv2.putText(frame, "FREE MODE  —  show any gesture",
                    (14, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "M = menu",
                    (w - 130, 32), cv2.FONT_HERSHEY_PLAIN, 1.0, (80, 80, 120), 1, cv2.LINE_AA)

        # ── Detection readout box ─────────────────────
        box_h = 80
        box_y = h - box_h - 10
        draw_panel(frame, w//2 - 280, box_y, 560, box_h, alpha=0.85,
                   color=(10, 10, 22), border=(50, 50, 100))

        if current_gesture:
            tag_color = (0, 200, 255) if gesture_type == "custom" else (0, 255, 180)
            center_text(frame, current_gesture, box_y + 48,
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, tag_color, 2)
            tag = "(custom sign)" if gesture_type == "custom" else "(built-in)"
            center_text(frame, tag, box_y + 70,
                        cv2.FONT_HERSHEY_PLAIN, 0.85, (100, 100, 150))
        else:
            center_text(frame, "No gesture detected",
                        box_y + 48, cv2.FONT_HERSHEY_PLAIN, 1.1, (60, 60, 90))

        # ── Sidebar: list custom signs ────────────────
        if custom_signs:
            lx, ly = 10, 60
            cv2.putText(frame, "Custom signs:", (lx, ly),
                        cv2.FONT_HERSHEY_PLAIN, 0.95, (80, 80, 130), 1, cv2.LINE_AA)
            for i, name in enumerate(custom_signs.keys()):
                n_samples = len(custom_signs[name])
                cv2.putText(frame, f"  [{name}]  ({n_samples})", (lx, ly + 20 + i * 20),
                            cv2.FONT_HERSHEY_PLAIN, 0.88, (100, 155, 210), 1, cv2.LINE_AA)

        # ── Hint: confidence info ─────────────────────
        cv2.putText(frame, "Built-in: rule-based   |   Custom: nearest-centroid",
                    (w//2 - 255, h - 8), cv2.FONT_HERSHEY_PLAIN, 0.8, (40, 40, 70),
                    1, cv2.LINE_AA)