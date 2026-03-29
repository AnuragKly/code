# mode_teach.py  —  Teach Mode
#
# Lets the user record new custom hand signs.
# Signs are stored as feature-vector samples in custom_signs.json via shared.py.
#
# Sub-states (internal):
#   MENU     — pick action
#   NAME     — type the sign name
#   CAPTURE  — hold hand still → samples are captured automatically
#   DONE     — confirmation screen
#   LIST     — browse / delete saved signs
#
# Returns to main.py:
#   "MAIN_MENU"      — user pressed M from MENU
#   custom_signs     — updated dict (passed back on every frame so main always has latest)

import cv2
import numpy as np
import time

from shared import (
    TEACH_SAMPLES, TEACH_HOLD,
    landmarks_to_feature, save_custom_signs,
    draw_panel, center_text,
)

# Internal sub-states
_ST_MENU    = "menu"
_ST_NAME    = "name"
_ST_CAPTURE = "capture"
_ST_DONE    = "done"
_ST_LIST    = "list"


class TeachMode:
    """Encapsulates all Teach Mode logic and drawing."""

    def __init__(self):
        self._state      = _ST_MENU
        self._typed_name = ""
        self._samples    = []
        self._hold_start = None
        self._list_sel   = 0

    # ── Public API ───────────────────────────────────

    def update(self, frame, lm, custom_signs):
        """
        Called every frame.
        lm           : MediaPipe landmark list or None
        custom_signs : dict (may be mutated here when saving)

        Returns "MAIN_MENU" if user exits, else None.
        Always mutates custom_signs in-place; caller keeps the same reference.
        """
        key = cv2.waitKey(1) & 0xFF

        if self._state == _ST_MENU:
            return self._update_menu(frame, custom_signs, key)
        elif self._state == _ST_NAME:
            return self._update_name(frame, custom_signs, key)
        elif self._state == _ST_CAPTURE:
            return self._update_capture(frame, lm, custom_signs, key)
        elif self._state == _ST_DONE:
            return self._update_done(frame, custom_signs, key)
        elif self._state == _ST_LIST:
            return self._update_list(frame, custom_signs, key)

        return None

    # ════════════════════════════════════════════════
    #  MENU
    # ════════════════════════════════════════════════

    def _update_menu(self, frame, custom_signs, key):
        self._draw_menu(frame, custom_signs)
        if key == ord('n'):
            self._typed_name = ""
            self._samples    = []
            self._state      = _ST_NAME
        elif key == ord('l'):
            self._list_sel = 0
            self._state    = _ST_LIST
        elif key == ord('m'):
            return "MAIN_MENU"
        return None

    def _draw_menu(self, frame, custom_signs):
        h, w = frame.shape[:2]
        draw_panel(frame, w//2 - 300, h//2 - 230, 600, 460, alpha=0.90)
        center_text(frame, "TEACH MODE",
                    h//2 - 195, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 255), 2)

        options = [
            ("N", "New sign",   "Teach the program a new gesture"),
            ("L", "List signs", f"View / delete saved signs  ({len(custom_signs)} saved)"),
            ("M", "Main menu",  "Go back"),
        ]
        for i, (key_label, title, desc) in enumerate(options):
            by = h//2 - 145 + i * 95
            draw_panel(frame, w//2 - 260, by, 520, 80, alpha=0.70,
                       color=(15, 20, 40), border=(50, 60, 120))
            cv2.putText(frame, f"[{key_label}]", (w//2 - 245, by + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, title, (w//2 - 190, by + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (w//2 - 190, by + 60),
                        cv2.FONT_HERSHEY_PLAIN, 0.90, (120, 120, 170), 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════
    #  NAME INPUT
    # ════════════════════════════════════════════════

    def _update_name(self, frame, custom_signs, key):
        self._draw_name(frame)
        if key == 27:                         # ESC → back to menu
            self._state = _ST_MENU
        elif key == 13:                       # ENTER → start capture
            if self._typed_name.strip():
                self._samples    = []
                self._hold_start = None
                self._state      = _ST_CAPTURE
        elif key == 8:                        # BACKSPACE
            self._typed_name = self._typed_name[:-1]
        elif 32 <= key < 127:
            self._typed_name += chr(key)
        return None

    def _draw_name(self, frame):
        h, w = frame.shape[:2]
        draw_panel(frame, w//2 - 280, h//2 - 120, 560, 240, alpha=0.90)
        center_text(frame, "NAME YOUR SIGN",
                    h//2 - 85, cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 1)
        center_text(frame, "Type a name, then press ENTER",
                    h//2 - 55, cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 120, 170))

        bx, bw_box, bht = w//2 - 200, 400, 44
        by = h//2 - 20
        cv2.rectangle(frame, (bx, by), (bx + bw_box, by + bht), (30, 30, 60), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw_box, by + bht), (80, 80, 180), 2)
        cursor   = "|" if int(time.time() * 2) % 2 == 0 else " "
        display  = self._typed_name + cursor
        cv2.putText(frame, display, (bx + 10, by + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)

        center_text(frame, "ESC  to cancel",
                    h//2 + 95, cv2.FONT_HERSHEY_PLAIN, 0.9, (60, 60, 100))

    # ════════════════════════════════════════════════
    #  CAPTURE
    # ════════════════════════════════════════════════

    def _update_capture(self, frame, lm, custom_signs, key):
        hand_detected = lm is not None

        if hand_detected:
            if self._hold_start is None:
                self._hold_start = time.time()
            hold_frac = (time.time() - self._hold_start) / TEACH_HOLD
        else:
            self._hold_start = None
            hold_frac        = 0.0

        self._draw_capture(frame, hand_detected, hold_frac)

        # Capture samples once hold threshold is met
        if hand_detected and hold_frac >= 1.0 and len(self._samples) < TEACH_SAMPLES:
            self._samples.append(landmarks_to_feature(lm))

        # Done collecting
        if len(self._samples) >= TEACH_SAMPLES:
            name = self._typed_name.strip()
            if name not in custom_signs:
                custom_signs[name] = []
            custom_signs[name].extend(self._samples)
            save_custom_signs(custom_signs)
            self._samples = []
            self._state   = _ST_DONE

        if key == 27:          # ESC → cancel
            self._samples = []
            self._state   = _ST_MENU

        return None

    def _draw_capture(self, frame, hand_detected, hold_frac):
        h, w = frame.shape[:2]

        # Top bar
        draw_panel(frame, 0, 0, w, 55, alpha=0.82, color=(8, 8, 18), border=(30, 30, 60))
        cv2.putText(frame, f"TEACHING:  \"{self._typed_name}\"",
                    (14, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 1, cv2.LINE_AA)

        # Progress bar (bottom)
        progress = len(self._samples) / TEACH_SAMPLES
        bx, by, bw_bar, bht = 14, h - 55, w - 28, 18
        cv2.rectangle(frame, (bx, by), (bx + bw_bar, by + bht), (20, 20, 40), -1)
        fill   = int(bw_bar * min(progress, 1.0))
        pcol   = (0, 200 + int(55 * progress), 80)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bht), pcol, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw_bar, by + bht), (50, 50, 100), 1)
        cv2.putText(frame, f"{len(self._samples)}/{TEACH_SAMPLES} samples",
                    (bx + 4, by + 13), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 220), 1, cv2.LINE_AA)

        # Status message
        if not hand_detected:
            msg, col = "Show your hand to the camera", (0, 120, 255)
        elif hold_frac < 1.0:
            msg, col = f"Hold still...  {hold_frac * TEACH_HOLD:.1f}s / {TEACH_HOLD:.1f}s", (0, 200, 255)
        else:
            msg, col = "Capturing...", (0, 255, 100)

        center_text(frame, msg,   h - 80, cv2.FONT_HERSHEY_DUPLEX, 0.65, col, 1)
        center_text(frame, "ESC  to cancel", h - 15, cv2.FONT_HERSHEY_PLAIN, 0.9, (50, 50, 80))

    # ════════════════════════════════════════════════
    #  DONE
    # ════════════════════════════════════════════════

    def _update_done(self, frame, custom_signs, key):
        name     = self._typed_name.strip()
        n_stored = len(custom_signs.get(name, []))
        self._draw_done(frame, name, n_stored)

        if key == ord(' '):
            self._typed_name = ""
            self._samples    = []
            self._state      = _ST_NAME
        elif key == ord('m'):
            self._state = _ST_MENU
        return None

    def _draw_done(self, frame, name, n_stored):
        h, w = frame.shape[:2]
        draw_panel(frame, w//2 - 260, h//2 - 120, 520, 240, alpha=0.92)
        center_text(frame, "SIGN SAVED!",
                    h//2 - 75, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 150), 2)
        center_text(frame, f"\"{name}\"  ({n_stored} samples)",
                    h//2 - 25, cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 220, 255))
        center_text(frame, "Detectable in Free Mode and Game Mode.",
                    h//2 + 20, cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 140, 180))
        pulse = int(160 + 95 * abs(np.sin(time.time() * 3)))
        center_text(frame, "SPACE  to teach another   |   M  for teach menu",
                    h//2 + 80, cv2.FONT_HERSHEY_PLAIN, 1.0, (pulse, pulse, 255))

    # ════════════════════════════════════════════════
    #  LIST
    # ════════════════════════════════════════════════

    def _update_list(self, frame, custom_signs, key):
        names = list(custom_signs.keys())
        self._list_sel = max(0, min(self._list_sel, len(names) - 1))
        self._draw_list(frame, custom_signs, names)

        if key in (82, ord('k')):    # UP arrow or k
            self._list_sel = max(0, self._list_sel - 1)
        elif key in (84, ord('j')):  # DOWN arrow or j
            self._list_sel = min(len(names) - 1, self._list_sel + 1)
        elif key == ord('d') and names:
            del custom_signs[names[self._list_sel]]
            save_custom_signs(custom_signs)
            self._list_sel = max(0, self._list_sel - 1)
        elif key == ord('m'):
            self._state = _ST_MENU
        return None

    def _draw_list(self, frame, custom_signs, names):
        h, w = frame.shape[:2]
        draw_panel(frame, w//2 - 300, 60, 600, h - 120, alpha=0.90)
        center_text(frame, "SAVED SIGNS", 100,
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 1)

        if not names:
            center_text(frame, "No custom signs saved yet.",
                        h//2, cv2.FONT_HERSHEY_PLAIN, 1.1, (80, 80, 110))
        else:
            for i, name in enumerate(names):
                by   = 130 + i * 38
                col  = (0, 255, 150) if i == self._list_sel else (180, 180, 220)
                mark = ">>  " if i == self._list_sel else "    "
                n    = len(custom_signs[name])
                cv2.putText(frame, f"{mark}{name}  ({n} samples)",
                            (w//2 - 260, by), cv2.FONT_HERSHEY_DUPLEX, 0.65, col, 1, cv2.LINE_AA)

        center_text(frame, "UP/DOWN  select   |   D  delete   |   M  back",
                    h - 30, cv2.FONT_HERSHEY_PLAIN, 0.9, (80, 80, 130))