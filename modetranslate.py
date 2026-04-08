# mode_translate.py  —  Translate Mode
#
# Accumulates a sequence of smoothed gesture labels, then sends them to
# the Anthropic API and displays the interpreted sentence on screen.
#
# Controls:
#   SPACE  — flush current buffer and translate now
#   C      — clear buffer without translating
#   ESC/B  — back to main menu

import cv2
import time
import threading
import anthropic

from shared import draw_panel, center_text

# How long a gesture must be held (seconds) before it's added to the log
HOLD_SECONDS   = 1.0
# Auto-translate after this many seconds of no new gesture being added
AUTO_TRANSLATE = 4.0
# Max gestures in a sequence before auto-flush
MAX_GESTURES   = 8


class TranslateMode:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.reset()

    def reset(self):
        self.gesture_log    = []       # list of gesture names in sequence
        self.last_label     = None     # last smoothed label seen
        self.hold_start     = None     # when current label started being held
        self.last_added_at  = None     # time the last gesture was appended
        self.translation    = ""       # latest Claude response
        self.translating    = False    # API call in flight
        self.status_msg     = "Show gestures one at a time — SPACE to translate"

    # ── called from main.py each frame ───────────────────────────────
    def update(self, frame, smoothed_label, smoothed_conf):
        h, w = frame.shape[:2]
        now  = time.time()

        # ── Gesture hold logic ──────────────────────────────────────
        if smoothed_label and smoothed_label != "unknown":
            if smoothed_label != self.last_label:
                # New label — start timing
                self.last_label = smoothed_label
                self.hold_start = now
            elif (self.hold_start and
                  now - self.hold_start >= HOLD_SECONDS and
                  smoothed_label not in (self.gesture_log[-1:] or [None]) and
                  len(self.gesture_log) < MAX_GESTURES):
                # Held long enough and not a duplicate of the last one
                self.gesture_log.append(smoothed_label)
                self.last_added_at = now
                self.hold_start    = now + 999   # prevent re-adding same hold
                self.status_msg    = f"Added: {smoothed_label}"
        else:
            self.last_label = None
            self.hold_start = None

        # ── Auto-translate after silence ────────────────────────────
        if (self.gesture_log and
                not self.translating and
                self.last_added_at and
                now - self.last_added_at >= AUTO_TRANSLATE):
            self._start_translation()

        # ── Max-length auto-flush ───────────────────────────────────
        if len(self.gesture_log) >= MAX_GESTURES and not self.translating:
            self._start_translation()

        # ── Draw UI ─────────────────────────────────────────────────
        self._draw(frame, smoothed_label, smoothed_conf, w, h)

        # ── Key handling ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('b')):          # ESC or B → menu
            return "MAIN_MENU"
        elif key == ord(' '):              # SPACE → translate now
            if self.gesture_log and not self.translating:
                self._start_translation()
        elif key == ord('c'):              # C → clear
            self.gesture_log   = []
            self.translation   = ""
            self.translating   = False
            self.last_added_at = None
            self.status_msg    = "Buffer cleared"

        return None

    # ── Background API call ──────────────────────────────────────────
    def _start_translation(self):
        if not self.gesture_log or self.translating:
            return
        self.translating = True
        self.status_msg  = "Translating…"
        snapshot = list(self.gesture_log)
        thread   = threading.Thread(
            target=self._call_api, args=(snapshot,), daemon=True
        )
        thread.start()

    def _call_api(self, gesture_sequence):
        prompt = f"""You are a sign-language interpreter assistant.
The user performed these hand gestures in sequence: {gesture_sequence}

Interpret this gesture sequence as a short natural-language phrase or sentence.
- Be concise (1 sentence max).
- If the gestures clearly map to ASL or common signs, use that meaning.
- Otherwise, make a reasonable creative interpretation.
- Reply with ONLY the interpreted sentence, no explanation."""

        try:
            message = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}]
            )
            self.translation   = message.content[0].text.strip()
            self.status_msg    = "Done! Press C to clear and try again"
        except Exception as e:
            self.translation = ""
            self.status_msg  = f"API error: {e}"
        finally:
            self.gesture_log   = []
            self.last_added_at = None
            self.translating   = False

    # ── Rendering ────────────────────────────────────────────────────
    def _draw(self, frame, smoothed_label, smoothed_conf, w, h):
        # Top bar
        draw_panel(frame, 0, 0, w, 50, alpha=0.82, color=(10, 12, 30),
                   border=(40, 44, 100), radius=0)
        cv2.putText(frame, "TRANSLATE MODE",
                    (14, 32), cv2.FONT_HERSHEY_DUPLEX, 0.72,
                    (0, 230, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "SPACE=translate  C=clear  B=back",
                    (w - 340, 32), cv2.FONT_HERSHEY_PLAIN, 0.95,
                    (80, 80, 130), 1, cv2.LINE_AA)

        # Current detected gesture
        if smoothed_label and smoothed_label != "unknown":
            hold_pct = 0.0
            if self.hold_start:
                elapsed  = time.time() - self.hold_start
                hold_pct = min(elapsed / HOLD_SECONDS, 1.0)
            label_color = (0, int(200 * hold_pct), int(255 * hold_pct))
            cv2.putText(frame, f"Detected: {smoothed_label}  ({smoothed_conf:.0%})",
                        (14, h - 90), cv2.FONT_HERSHEY_DUPLEX, 0.68,
                        label_color, 1, cv2.LINE_AA)
            # Hold progress bar
            bar_w = int((w - 28) * hold_pct)
            cv2.rectangle(frame, (14, h - 72), (14 + bar_w, h - 58),
                          (0, 180, 220), -1)
            cv2.rectangle(frame, (14, h - 72), (w - 14, h - 58),
                          (40, 40, 60), 1)

        # Gesture token strip
        strip_y = 64
        draw_panel(frame, 8, strip_y, w - 16, 52, alpha=0.70,
                   color=(14, 16, 38), border=(36, 42, 100), radius=8)
        if self.gesture_log:
            token_x = 20
            for i, g in enumerate(self.gesture_log):
                label = g[:12]
                tw    = max(len(label) * 9, 72)
                cv2.rectangle(frame, (token_x, strip_y + 8),
                              (token_x + tw, strip_y + 44),
                              (30, 120, 200), -1)
                cv2.rectangle(frame, (token_x, strip_y + 8),
                              (token_x + tw, strip_y + 44),
                              (60, 160, 255), 1)
                cv2.putText(frame, label,
                            (token_x + 6, strip_y + 31),
                            cv2.FONT_HERSHEY_PLAIN, 0.92,
                            (220, 240, 255), 1, cv2.LINE_AA)
                token_x += tw + 8
                if token_x > w - 100:
                    break
        else:
            center_text(frame, "No gestures yet — hold a gesture to add it",
                        strip_y + 32, cv2.FONT_HERSHEY_PLAIN, 0.90,
                        (60, 60, 90))

        # Translation output box
        if self.translation or self.translating:
            box_y = h // 2 - 60
            draw_panel(frame, w // 2 - 280, box_y, 560, 120,
                       alpha=0.92, color=(8, 24, 50),
                       border=(30, 80, 160), radius=12)
            if self.translating:
                pulse = int(140 + 115 * abs(__import__('numpy').sin(time.time() * 3)))
                center_text(frame, "Asking Claude…",
                            box_y + 60, cv2.FONT_HERSHEY_DUPLEX, 0.82,
                            (pulse, pulse, 255))
            else:
                # Word-wrap the translation across two lines if needed
                words     = self.translation.split()
                line1, line2 = [], []
                for word in words:
                    if len(" ".join(line1 + [word])) <= 42:
                        line1.append(word)
                    else:
                        line2.append(word)
                center_text(frame, " ".join(line1),
                            box_y + 52, cv2.FONT_HERSHEY_DUPLEX, 0.76,
                            (220, 240, 255))
                if line2:
                    center_text(frame, " ".join(line2),
                                box_y + 84, cv2.FONT_HERSHEY_DUPLEX, 0.76,
                                (220, 240, 255))

        # Status bar
        cv2.putText(frame, self.status_msg,
                    (14, h - 14), cv2.FONT_HERSHEY_PLAIN, 0.90,
                    (100, 100, 160), 1, cv2.LINE_AA)