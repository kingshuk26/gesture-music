"""
🎵 Gesture Music Control App
Compatible with mediapipe 0.10.30+
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import ctypes
import warnings
warnings.filterwarnings("ignore")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ─── Download model if needed ────────────────────────────────
import urllib.request, pathlib

MODEL_PATH = pathlib.Path("hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not MODEL_PATH.exists():
    print("Downloading hand landmarker model (~9 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!\n")

# ─── Palette ─────────────────────────────────────────────────
ACCENT = (0, 220, 160)
WARN   = (50, 160, 255)
DARK   = (30, 30, 44)

GESTURES = {
    "fist":          ("Fist  -  Play / Pause",    "PLAY_PAUSE"),
    "open_palm":     ("Open Palm  -  Stop",        "STOP"),
    "one_finger":    ("1 Finger  -  Prev Track",   "PREV"),
    "two_fingers":   ("2 Fingers  -  Next Track",  "NEXT"),
    "three_fingers": ("3 Fingers  -  Volume Up",   "VOL_UP"),
    "four_fingers":  ("4 Fingers  -  Volume Down", "VOL_DOWN"),
    "thumbs_up":     ("Thumbs Up  -  Like",        "LIKE"),
    "pinch":         ("Pinch  -  Mute Toggle",     "MUTE"),
}

ACTION_COOLDOWN = 1.3
PINCH_THRESH    = 0.06

# ─── Windows media keys ──────────────────────────────────────
VK_MAP = {
    "PLAY_PAUSE": 0xB3, "STOP": 0xB2,
    "NEXT": 0xB0,       "PREV": 0xB1,
    "VOL_UP": 0xAF,     "VOL_DOWN": 0xAE, "MUTE": 0xAD,
}

def send_media_key(action: str):
    vk = VK_MAP.get(action)
    if vk:
        EXT = 0x1; UP = 0x2
        ctypes.windll.user32.keybd_event(vk, 0, EXT, 0)
        ctypes.windll.user32.keybd_event(vk, 0, EXT | UP, 0)

def execute_action(action: str):
    print(f"  >> {action}")
    if action == "LIKE":
        print("  * Liked track!")
        return
    send_media_key(action)

# ─── Gesture detection ───────────────────────────────────────

def fingers_up(lm, hand_label: str):
    tips  = [4, 8, 12, 16, 20]
    bases = [2, 6, 10, 14, 18]
    thumb = lm[tips[0]].x < lm[bases[0]].x if hand_label == "Right" \
            else lm[tips[0]].x > lm[bases[0]].x
    result = [thumb]
    for i in range(1, 5):
        result.append(lm[tips[i]].y < lm[bases[i]].y)
    return result

def classify_gesture(lm, hand_label: str):
    up    = fingers_up(lm, hand_label)
    count = sum(up[1:])
    pinch_d = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
    if pinch_d < PINCH_THRESH and not up[2] and not up[3]:
        return "pinch"
    if up[0] and count == 0:   return "thumbs_up"
    if not any(up):             return "fist"
    if all(up):                 return "open_palm"
    if count == 1 and up[1]:   return "one_finger"
    if count == 2 and up[1] and up[2]: return "two_fingers"
    if count == 3 and up[1] and up[2] and up[3]: return "three_fingers"
    if count == 4 and up[1] and up[2] and up[3] and up[4]: return "four_fingers"
    return None

# ─── Draw hand landmarks ─────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 120), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, ACCENT, -1)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1)

# ─── UI ──────────────────────────────────────────────────────

def draw_ui(frame, gesture, last_label, fps, action_time, history):
    h, w = frame.shape[:2]
    PANEL = 270

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL, h), (8, 8, 18), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "GESTURE MUSIC", (14, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, "CONTROL", (14, 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, ACCENT, 1, cv2.LINE_AA)
    cv2.line(frame, (14, 68), (PANEL - 14, 68), ACCENT, 1)

    y0 = 88
    for key, (label, _) in GESTURES.items():
        active = gesture == key
        clr = ACCENT if active else (110, 110, 135)
        if active:
            cv2.rectangle(frame, (8, y0 - 14), (PANEL - 8, y0 + 5), (0, 55, 40), -1)
        cv2.putText(frame, label, (14, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)
        y0 += 22

    cv2.putText(frame, f"fps {fps:.0f}", (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 90), 1, cv2.LINE_AA)

    if gesture and gesture in GESTURES:
        label, _ = GESTURES[gesture]
        badge_w  = max(280, len(label) * 12 + 40)
        bx = (w - badge_w) // 2
        by = h - 80
        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx, by), (bx + badge_w, by + 44), DARK, -1)
        cv2.addWeighted(ov2, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + 44), ACCENT, 1)
        cv2.putText(frame, label, (bx + 14, by + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, ACCENT, 1, cv2.LINE_AA)

    elapsed = time.time() - action_time
    if last_label and elapsed < 2.0:
        fade  = max(0, 1.0 - elapsed / 2.0)
        color = tuple(int(c * fade) for c in WARN)
        cv2.putText(frame, f">> {last_label}", (PANEL + 14, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, color, 1, cv2.LINE_AA)

    dot_x = PANEL + 14
    cv2.putText(frame, "activity", (dot_x, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (65, 65, 85), 1, cv2.LINE_AA)
    for i, active in enumerate(history[-14:]):
        cv2.circle(frame, (dot_x + i * 14, h - 14), 4,
                   ACCENT if active else (45, 45, 60), -1)

    cv2.putText(frame, "Q / ESC  quit", (w - 170, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 90), 1, cv2.LINE_AA)

# ─── Main ────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("Gesture Music Control - running")
    print("Press Q or ESC to quit\n")
    for key, (label, action) in GESTURES.items():
        print(f"  {label}")
    print()

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    last_gesture      = None
    gesture_lock_time = 0.0
    last_action_label = ""
    action_time       = 0.0
    history           = []
    prev_t            = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)
            gesture  = None

            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    hand_label = "Right"
                    if result.handedness and i < len(result.handedness):
                        hand_label = result.handedness[i][0].display_name

                    draw_hand(frame, hand_landmarks)
                    gesture = classify_gesture(hand_landmarks, hand_label)

                    now = time.time()
                    if (gesture
                            and gesture != last_gesture
                            and now - gesture_lock_time > ACTION_COOLDOWN):
                        _, action = GESTURES.get(gesture, ("", None))
                        if action:
                            execute_action(action)
                            last_action_label = GESTURES[gesture][0]
                            action_time       = now
                            gesture_lock_time = now

                    last_gesture = gesture
            else:
                last_gesture = None

            now    = time.time()
            fps    = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            history.append(gesture is not None)
            if len(history) > 120:
                history.pop(0)

            draw_ui(frame, gesture, last_action_label, fps, action_time, history)
            cv2.imshow("Gesture Music Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nBye!")


if __name__ == "__main__":
    main()