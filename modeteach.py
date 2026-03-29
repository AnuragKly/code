# mode_teach.py  —  Teach Mode
#
# Record custom hand signs saved as feature-vector samples.
#
# Improvements over v1:
#   • Motion guard: captures only when the hand is sufficiently still
#     (measures frame-to-frame landmark displacement, rejects jitter)
#   • Per-sign retrain: existing sign can be extended or fully replaced
#   • Quality score shown after capture (sample variance → stability rating)
#   • Visual live preview of captured feature quality during capture
#   • List screen shows quality rating (✓ good / ~ ok / ! weak)
#   • ESC anywhere in teach flow goes back one level (not all the way to menu)

import cv2
import numpy as np
import time
from collections import deque

from shared import (
    TEACH_SAMPLES, TEACH_HOLD,
    landmarks_to_feature, save_custom_signs,
    draw_panel, center_text, draw_confidence_bar,
)

_ST_MENU     = "menu"
_ST_NAME     = "name"
_ST_CAPTURE  = "capture"
_ST_DONE     = "done"
_ST_LIST     = "list"
_ST_CONFIRM  = "confirm_replace"   # new: confirm overwrite

# Motion guard threshold: max allowed mean landmark displacement per frame
MOTION_THRESH = 0.012


class TeachMode:
    """All Teach Mode logic and rendering."""

    def __init__(self):
        self._state        = _ST_MENU
        self._typed_name   = ""
        self._samples      = []
        self._hold_start   = None
        self._list_sel     = 0
        self._prev_feature = None   # for motion guard
        self._quality      = 0.0    # post-capture quality score

    # ── Public API ───────────────────────────────────

    def update(self, frame, lm, custom_signs):
        """
        Called every frame by main.py.
        lm           : MediaPipe landmark list or None
        custom_signs : dict — mutated in-place when signs are saved/deleted
        Returns "MAIN_MENU" if user fully exits, else None.
        """
        key = cv2.waitKey(1) & 0xFF

        dispatch = {
            _ST_MENU:    self._update_menu,
            _ST_NAME:    self._update_name,
            _ST_CAPTURE: self._update_capture,
            _ST_DONE:    self._update_done,
            _ST_LIST:    self._update_list,
            _ST_CONFIRM: self._update_confirm,
        }
        return dispatch[self._state](frame, lm, custom_signs, key)

    # ════════════════════════════════════════════════
    #  MENU
    # ════════════════════════════════════════════════

    def _update_menu(self, frame, lm, custom_signs, key):
        self._draw_menu(frame, custom_signs)
        if key == ord('n'):
            self._typed_name   = ""
            self._samples      = []
            self._prev_feature = None
            self._state        = _ST_NAME
        elif key == ord('l'):
            self._list_sel = 0
            self._state    = _ST_LIST
        elif key in (ord('m'), 27):
            return "MAIN_MENU"
        return None

    def _draw_menu(self, frame, custom_signs):
        h, w = frame.shape[:2]
        draw_panel(frame, w//2 - 300, h//2 - 240, 600, 480, alpha=0.92, radius=14)
        center_text(frame, "TEACH MODE",
                    h//2 - 205, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 255), 2)
        center_text(frame, "Train new hand signs for detection",
                    h//2 - 168, cv2.FONT_HERSHEY_PLAIN, 1.0, (90, 90, 140))

        options = [
            ("N", "New sign",   "Teach a new gesture from scratch"),
            ("L", "List signs", f"Browse / delete saved signs  ({len(custom_signs)} stored)"),
            ("M / ESC", "Main menu",  "Go back"),
        ]
        for i, (key_label, title, desc) in enumerate(options):
            by = h//2 - 130 + i * 100
            draw_panel(frame, w//2 - 265, by, 530, 84, alpha=0.72,
                       color=(15, 20, 42), border=(52, 62, 130), radius=8)
            cv2.putText(frame, f"[{key_label}]", (w//2 - 250, by + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, title, (w//2 - 155, by + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (240, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (w//2 - 155, by + 62),
                        cv2.FONT_HERSHEY_PLAIN, 0.90, (110, 110, 165), 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════
    #  NAME INPUT
    # ════════════════════════════════════════════════

    def _update_name(self, frame, lm, custom_signs, key):
        self._draw_name(frame, custom_signs)
        if key == 27:                          # ESC → back
            self._state = _ST_MENU
        elif key == 13:                        # ENTER
            name = self._typed_name.strip()
            if name:
                if name in custom_signs:
                    self._state = _ST_CONFIRM  # ask before overwriting
                else:
                    self._samples    = []
                    self._hold_start = None
                    self._prev_feature = None
                    self._state      = _ST_CAPTURE
        elif key == 8:
            self._typed_name = self._typed_name[:-1]
        elif 32 <= key < 127:
            if len(self._typed_name) < 24:     # cap name length
                self._typed_name += chr(key)
        return None

    def _draw_name(self, frame, custom_signs):
        h, w = frame.shape[:2]
        name_exists = self._typed_name.strip() in custom_signs
        draw_panel(frame, w//2 - 290, h//2 - 140, 580, 280, alpha=0.92, radius=12)
        center_text(frame, "NAME YOUR SIGN",
                    h//2 - 108, cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 200, 255), 1)
        center_text(frame, "Type a name then press ENTER",
                    h//2 - 75, cv2.FONT_HERSHEY_PLAIN, 1.0, (110, 110, 165))

        bx, bw_box, bht = w//2 - 215, 430, 46
        by = h//2 - 48
        border_col = (180, 60, 60) if name_exists else (80, 80, 200)
        cv2.rectangle(frame, (bx, by), (bx + bw_box, by + bht), (28, 28, 55), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw_box, by + bht), border_col, 2)
        cursor  = "|" if int(time.time() * 2) % 2 == 0 else " "
        display = self._typed_name + cursor
        cv2.putText(frame, display, (bx + 10, by + 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.88, (255, 255, 255), 1, cv2.LINE_AA)

        if name_exists:
            center_text(frame, f"  \"{self._typed_name.strip()}\" already exists — ENTER to add samples",
                        h//2 + 15, cv2.FONT_HERSHEY_PLAIN, 0.95, (180, 100, 80))
        center_text(frame, "ESC  to cancel",
                    h//2 + 110, cv2.FONT_HERSHEY_PLAIN, 0.92, (55, 55, 95))

    # ════════════════════════════════════════════════
    #  CONFIRM REPLACE
    # ════════════════════════════════════════════════

    def _update_confirm(self, frame, lm, custom_signs, key):
        self._draw_confirm(frame)
        if key == ord('a'):    # Add samples to existing
            self._samples      = []
            self._hold_start   = None
            self._prev_feature = None
            self._state        = _ST_CAPTURE
        elif key == ord('r'):  # Replace entirely
            name = self._typed_name.strip()
            if name in custom_signs:
                del custom_signs[name]
                save_custom_signs(custom_signs)
            self._samples      = []
            self._hold_start   = None
            self._prev_feature = None
            self._state        = _ST_CAPTURE
        elif key == 27:
            self._state = _ST_NAME
        return None

    def _draw_confirm(self, frame):
        h, w = frame.shape[:2]
        name = self._typed_name.strip()
        draw_panel(frame, w//2 - 280, h//2 - 130, 560, 260, alpha=0.93, radius=12)
        center_text(frame, f"\"{name}\" already exists",
                    h//2 - 95, cv2.FONT_HERSHEY_DUPLEX, 0.82, (200, 160, 60), 1)

        opts = [
            ("[A]", "Add samples", "Keep existing + record more"),
            ("[R]", "Replace",     "Delete existing and start fresh"),
            ("[ESC]", "Cancel",    "Go back"),
        ]
        for i, (k, title, desc) in enumerate(opts):
            by = h//2 - 60 + i * 65
            draw_panel(frame, w//2 - 240, by, 480, 56, alpha=0.70,
                       color=(18, 18, 38), border=(60, 60, 120), radius=6)
            cv2.putText(frame, k, (w//2 - 225, by + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, title, (w//2 - 155, by + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.68, (240, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (w//2 - 155, by + 50),
                        cv2.FONT_HERSHEY_PLAIN, 0.88, (110, 110, 160), 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════
    #  CAPTURE
    # ════════════════════════════════════════════════

    def _update_capture(self, frame, lm, custom_signs, key):
        hand_ok = lm is not None

        # Motion guard
        still = False
        if hand_ok:
            feat = landmarks_to_feature(lm)
            if self._prev_feature is not None:
                motion = float(np.mean(np.abs(
                    np.array(feat) - np.array(self._prev_feature)
                )))
                still = motion < MOTION_THRESH
            self._prev_feature = feat
        else:
            self._prev_feature = None
            self._hold_start   = None

        # Hold timer (only counts while still)
        if hand_ok and still:
            if self._hold_start is None:
                self._hold_start = time.time()
            hold_frac = (time.time() - self._hold_start) / TEACH_HOLD
        else:
            self._hold_start = None
            hold_frac        = 0.0

        self._draw_capture(frame, hand_ok, still, hold_frac)

        # Capture once hold threshold met and hand is still
        if hand_ok and still and hold_frac >= 1.0 and len(self._samples) < TEACH_SAMPLES:
            self._samples.append(landmarks_to_feature(lm))

        if len(self._samples) >= TEACH_SAMPLES:
            self._quality = self._compute_quality(self._samples)
            name = self._typed_name.strip()
            if name not in custom_signs:
                custom_signs[name] = []
            custom_signs[name].extend(self._samples)
            save_custom_signs(custom_signs)
            self._samples = []
            self._state   = _ST_DONE

        if key == 27:
            self._samples      = []
            self._prev_feature = None
            self._state        = _ST_MENU
        return None

    @staticmethod
    def _compute_quality(samples):
        """
        Quality score 0–1 based on intra-class variance.
        Low variance (consistent poses) → high quality.
        """
        arr = np.array(samples)
        var = float(np.mean(np.var(arr, axis=0)))
        # Empirically: var < 0.002 is great, > 0.01 is poor
        return float(np.clip(1.0 - var / 0.01, 0.0, 1.0))

    def _draw_capture(self, frame, hand_ok, still, hold_frac):
        h, w = frame.shape[:2]
        n = len(self._samples)

        # Top bar
        draw_panel(frame, 0, 0, w, 56, alpha=0.84,
                   color=(8, 8, 18), border=(30, 30, 60), radius=0)
        cv2.putText(frame, f"TEACHING:  \"{self._typed_name}\"",
                    (14, 36), cv2.FONT_HERSHEY_DUPLEX, 0.78, (0, 220, 255), 1, cv2.LINE_AA)

        # Sample progress bar
        progress = n / TEACH_SAMPLES
        bx, by_bar, bw_bar, bht = 14, h - 58, w - 28, 20
        cv2.rectangle(frame, (bx, by_bar), (bx + bw_bar, by_bar + bht), (18, 18, 38), -1)
        fill  = int(bw_bar * min(progress, 1.0))
        g     = int(160 + 95 * progress)
        if fill > 0:
            cv2.rectangle(frame, (bx, by_bar), (bx + fill, by_bar + bht), (0, g, 70), -1)
        cv2.rectangle(frame, (bx, by_bar), (bx + bw_bar, by_bar + bht), (50, 50, 100), 1)
        cv2.putText(frame, f"{n} / {TEACH_SAMPLES} samples",
                    (bx + 5, by_bar + 14), cv2.FONT_HERSHEY_PLAIN, 0.88,
                    (200, 200, 220), 1, cv2.LINE_AA)

        # Status message
        if not hand_ok:
            msg, col = "Show your hand to the camera", (0, 100, 255)
        elif not still:
            msg, col = "Hold still — movement detected", (0, 160, 255)
        elif hold_frac < 1.0:
            msg, col = (f"Hold still...  {hold_frac * TEACH_HOLD:.1f}s / {TEACH_HOLD:.1f}s",
                        (0, 200, 255))
        else:
            msg, col = "Capturing...", (0, 255, 100)

        center_text(frame, msg, h - 80, cv2.FONT_HERSHEY_DUPLEX, 0.68, col, 1)

        # Hold arc (centre-bottom area)
        if hand_ok and hold_frac > 0:
            arc_cx, arc_cy = w // 2, h - 140
            arc_col = (0, int(100 + 155 * hold_frac), 80)
            cv2.ellipse(frame, (arc_cx, arc_cy), (22, 22), -90, 0,
                        int(360 * min(hold_frac, 1.0)), arc_col, 3, cv2.LINE_AA)

        center_text(frame, "ESC  to cancel", h - 14,
                    cv2.FONT_HERSHEY_PLAIN, 0.88, (45, 45, 80))

    # ════════════════════════════════════════════════
    #  DONE
    # ════════════════════════════════════════════════

    def _update_done(self, frame, lm, custom_signs, key):
        name     = self._typed_name.strip()
        n_stored = len(custom_signs.get(name, []))
        self._draw_done(frame, name, n_stored)
        if key == ord(' '):
            self._typed_name   = ""
            self._samples      = []
            self._prev_feature = None
            self._state        = _ST_NAME
        elif key in (ord('m'), 27):
            self._state = _ST_MENU
        return None

    def _draw_done(self, frame, name, n_stored):
        h, w = frame.shape[:2]
        draw_panel(frame, w//2 - 270, h//2 - 140, 540, 280, alpha=0.93, radius=14)
        center_text(frame, "SIGN SAVED!",
                    h//2 - 100, cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 150), 2)
        center_text(frame, f"\"{name}\"  ({n_stored} total samples)",
                    h//2 - 48, cv2.FONT_HERSHEY_DUPLEX, 0.72, (200, 220, 255))

        # Quality bar
        qx = w//2 - 150
        draw_confidence_bar(frame, qx, h//2 - 18, 300, 18, self._quality, "Quality")

        q_text = ("Excellent" if self._quality > 0.75
                  else "Good" if self._quality > 0.5
                  else "Fair — try recording in better light or more consistently")
        center_text(frame, q_text, h//2 + 18, cv2.FONT_HERSHEY_PLAIN, 0.92, (140, 180, 160))
        center_text(frame, "Detectable in Free Mode and Game Mode.",
                    h//2 + 44, cv2.FONT_HERSHEY_PLAIN, 0.92, (110, 130, 165))

        import numpy as np
        import time as _t
        pulse = int(155 + 100 * abs(np.sin(_t.time() * 3)))
        center_text(frame, "SPACE  teach another   |   M  teach menu",
                    h//2 + 95, cv2.FONT_HERSHEY_PLAIN, 1.02, (pulse, pulse, 255))

    # ════════════════════════════════════════════════
    #  LIST
    # ════════════════════════════════════════════════

    def _update_list(self, frame, lm, custom_signs, key):
        names = list(custom_signs.keys())
        self._list_sel = max(0, min(self._list_sel, len(names) - 1))
        self._draw_list(frame, custom_signs, names)

        if key in (82, ord('k'), ord('w')):   # UP
            self._list_sel = max(0, self._list_sel - 1)
        elif key in (84, ord('j'), ord('s')): # DOWN
            self._list_sel = min(len(names) - 1, self._list_sel + 1)
        elif key == ord('d') and names:
            del custom_signs[names[self._list_sel]]
            save_custom_signs(custom_signs)
            self._list_sel = max(0, self._list_sel - 1)
        elif key in (ord('m'), 27):
            self._state = _ST_MENU
        return None

    def _draw_list(self, frame, custom_signs, names):
        h, w = frame.shape[:2]
        draw_panel(frame, w//2 - 310, 58, 620, h - 118, alpha=0.92, radius=12)
        center_text(frame, "SAVED SIGNS", 98,
                    cv2.FONT_HERSHEY_DUPLEX, 0.92, (0, 200, 255), 1)

        if not names:
            center_text(frame, "No custom signs saved yet.",
                        h//2, cv2.FONT_HERSHEY_PLAIN, 1.1, (70, 70, 105))
        else:
            for i, name in enumerate(names):
                by      = 128 + i * 40
                sel     = (i == self._list_sel)
                col     = (0, 255, 150) if sel else (170, 170, 210)
                n_s     = len(custom_signs[name])
                quality = "✓ " if n_s >= 40 else "~ " if n_s >= 20 else "! "
                q_col   = (0, 200, 100) if n_s >= 40 else (0, 180, 255) if n_s >= 20 else (0, 80, 255)

                if sel:
                    cv2.rectangle(frame, (w//2 - 295, by - 20),
                                  (w//2 + 295, by + 14), (20, 30, 55), -1)

                cv2.putText(frame, quality, (w//2 - 290, by),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, q_col, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{name}  ({n_s} samples)",
                            (w//2 - 255, by), cv2.FONT_HERSHEY_DUPLEX, 0.65, col, 1, cv2.LINE_AA)

        center_text(frame, "W/S or ↑↓  select   |   D  delete   |   M / ESC  back",
                    h - 30, cv2.FONT_HERSHEY_PLAIN, 0.92, (70, 70, 120))