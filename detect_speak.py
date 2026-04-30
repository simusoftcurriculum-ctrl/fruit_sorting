"""
Fruit Detector — Live Webcam Inference
Requires: fruit_classifier.pth + class_names.json (produced by train.py)

Controls:
  Q = Quit
  M = Toggle speech on/off
  S = Mute speech for 10 seconds
  F = Freeze frame
  D = Debug mode (prints all class probabilities)

Fix included:
- The app now stays silent when no fruit is confidently visible.
- It uses stronger rejection checks:
  - confidence threshold
  - top1 vs top2 margin threshold
  - entropy threshold
  - brightness check
  - texture/detail check
- If you later train with a "background" class, this script supports it too.
"""

import cv2
import sys
import time
import json
import queue
import threading
import subprocess
import platform
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Load model ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
MODEL_PATH = BASE / "fruit_classifier.pth"
CLASS_PATH = BASE / "class_names.json"

for p in (MODEL_PATH, CLASS_PATH):
    if not p.exists():
        print(f"ERROR: {p} not found — run train.py first!")
        sys.exit(1)

with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

print(f"Model loaded | classes: {CLASS_NAMES} | device: {DEVICE}\n")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Class metadata ────────────────────────────────────────────────────────────
INFO = {
    "freshapples": ("Fresh Apple", True,
                    "Fresh apple detected! Apples are crunchy and rich in fiber.",
                    (50, 200, 50)),
    "freshbanana": ("Fresh Banana", True,
                    "Fresh banana detected! Bananas are rich in potassium.",
                    (0, 210, 255)),
    "freshoranges": ("Fresh Orange", True,
                     "Fresh orange detected! Oranges are packed with Vitamin C.",
                     (0, 140, 255)),
    "freshmangoes": ("Fresh Mango", True,
                     "Fresh mango detected! Mangoes are very sweet to eat.",
                     (0, 220, 255)),
    "rottenapples": ("Rotten Apple", False,
                     "Warning! This apple is rotten. Do not eat it.",
                     (0, 0, 200)),
    "rottenbanana": ("Rotten Banana", False,
                     "Warning! This banana is rotten. Throw it away.",
                     (0, 0, 180)),
    "rottenoranges": ("Rotten Orange", False,
                      "Warning! This orange is rotten. Do not consume it.",
                      (0, 0, 180)),
    "rottenmangoes": ("Rotten Mango", False,
                      "Warning! This mango is rotten.",
                      (30, 120, 160)),
    "background": ("Background", True, "", (80, 80, 80)),
}

# ── TTS setup ─────────────────────────────────────────────────────────────────
OS = platform.system()
TTS = None
speech_enabled = True

if OS == "Windows":
    try:
        import win32com.client
        win32com.client.Dispatch("SAPI.SpVoice")
        TTS = "sapi"
        print("TTS: SAPI OK")
    except Exception:
        pass

if TTS is None:
    try:
        import pyttsx3 as _p
        _p.init().stop()
        TTS = "pyttsx3"
        print("TTS: pyttsx3 OK")
    except Exception:
        pass

if TTS is None and OS == "Linux":
    for _c in ["espeak-ng", "espeak"]:
        try:
            if subprocess.run([_c, "--version"], capture_output=True, timeout=3).returncode == 0:
                TTS = _c
                print(f"TTS: {_c} OK")
                break
        except Exception:
            pass

if TTS is None and OS == "Darwin":
    try:
        subprocess.run(["say", "hi"], capture_output=True, timeout=3, check=True)
        TTS = "say"
        print("TTS: say OK")
    except Exception:
        pass

if TTS is None:
    print("WARNING: No TTS engine found. Speech disabled.")


def _say(text):
    try:
        if TTS == "sapi":
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = 0
            speaker.Volume = 100
            speaker.Speak(text)
        elif TTS == "pyttsx3":
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        elif TTS in ("espeak-ng", "espeak"):
            subprocess.run([TTS, "-s", "145", "-a", "200", "-v", "en", text],
                           capture_output=True, timeout=10)
        elif TTS == "say":
            subprocess.run(["say", "-r", "170", text], capture_output=True, timeout=10)
    except Exception as e:
        print(f"TTS error: {e}")


_q = queue.Queue(maxsize=1)


def _worker():
    while True:
        text = _q.get()
        if text is None:
            break
        _say(text)
        _q.task_done()


threading.Thread(target=_worker, daemon=True).start()


def speak(text):
    if not speech_enabled or TTS is None or not text:
        return
    try:
        _q.put_nowait(text)
    except queue.Full:
        pass


# ── Thresholds ────────────────────────────────────────────────────────────────
CONF_THR = 0.88
ENTROPY_THR = 0.65
MARGIN_THR = 0.35
MIN_BRIGHTNESS = 40
MIN_STDDEV = 18
STABLE_NEED = 12
SPEAK_COOLDOWN = 8.0

FRUIT_CLASSES = {
    "freshapples", "freshbanana", "freshoranges", "freshmangoes",
    "rottenapples", "rottenbanana", "rottenoranges", "rottenmangoes",
}


def predict(roi_bgr):
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top2_vals, top2_idx = torch.topk(probs, 2)
    conf = float(top2_vals[0])
    second_conf = float(top2_vals[1])
    cls = CLASS_NAMES[int(top2_idx[0])]
    margin = conf - second_conf
    entropy = float(-(probs * torch.log(probs + 1e-9)).sum())

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    stddev = float(gray.std())

    top2 = [
        (CLASS_NAMES[int(top2_idx[0])], float(top2_vals[0])),
        (CLASS_NAMES[int(top2_idx[1])], float(top2_vals[1])),
    ]
    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return cls, conf, second_conf, margin, entropy, brightness, stddev, top2, all_probs


def is_passing(cls, conf, margin, entropy, brightness, stddev):
    if cls not in FRUIT_CLASSES:
        return False
    if conf < CONF_THR:
        return False
    if margin < MARGIN_THR:
        return False
    if entropy > ENTROPY_THR:
        return False
    if brightness < MIN_BRIGHTNESS:
        return False
    if stddev < MIN_STDDEV:
        return False
    return True


def get_roi(h, w):
    side = int(min(h, w) * 0.45)
    cx, cy = w // 2, h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    return x1, y1, x1 + side, y1 + side


# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    sys.exit(1)

print("=== Fruit Detector ===")
print("Q=Quit  M=Speech toggle  S=Mute 10s  F=Freeze  D=Debug\n")

prev_time = time.time()
silence_until = 0.0
frozen = False
debug = False

stable_cls = None
stable_count = 0
confirmed = None
last_spoke_at = {}
last_top2 = []

while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            continue

    h, w = frame.shape[:2]
    now = time.time()

    rx1, ry1, rx2, ry2 = get_roi(h, w)
    roi = frame[ry1:ry2, rx1:rx2]

    cls, conf, second_conf, margin, entropy, brightness, stddev, top2, all_probs = predict(roi)
    last_top2 = top2
    ok = is_passing(cls, conf, margin, entropy, brightness, stddev)

    if debug:
        print(
            f"\n{cls:20s} conf={conf:.3f} margin={margin:.3f} "
            f"entropy={entropy:.3f} bright={brightness:.1f} std={stddev:.1f} "
            f"ok={ok} stable={stable_count}/{STABLE_NEED}"
        )
        for name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"  {name:20s} {prob:.4f}  {bar}")

    # ── Stability tracking ────────────────────────────────────────────────────
    if ok:
        if cls == stable_cls:
            stable_count += 1
        else:
            stable_cls = cls
            stable_count = 1
            confirmed = None        # reset confirmed when fruit class changes
            last_spoke_at = {}      # clear cooldowns so new fruit speaks immediately
    else:
        stable_cls = None
        stable_count = 0
        confirmed = None            # reset confirmed when detection fails

    if stable_count >= STABLE_NEED:
        confirmed = stable_cls

    # ── Speak only when fruit is truly confirmed ──────────────────────────────
    if (confirmed
            and confirmed in FRUIT_CLASSES
            and confirmed == stable_cls          # must still match current detection
            and stable_count >= STABLE_NEED      # must still be stably detected
            and now > silence_until):
        last_spoke = last_spoke_at.get(confirmed, 0)
        if now - last_spoke > SPEAK_COOLDOWN:
            _, _, phrase, _ = INFO[confirmed]
            if phrase:
                speak(phrase)
                print(f"SPEAK: {phrase}")
                last_spoke_at[confirmed] = now

    # ── ROI border color ──────────────────────────────────────────────────────
    if confirmed and confirmed in FRUIT_CLASSES:
        roi_col = INFO[confirmed][3]
    elif ok and stable_cls:
        roi_col = (60, 200, 60)
    else:
        roi_col = (80, 80, 80)

    # ── Draw ROI ──────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_col, 3)

    # ── Labels ────────────────────────────────────────────────────────────────
    if confirmed and confirmed in FRUIT_CLASSES:
        disp, fresh, _, color = INFO[confirmed]
        lbl = f"{disp}  {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        lx, ly = rx1, ry2 + lh + 14
        cv2.rectangle(frame, (lx - 4, ly - lh - 8), (lx + lw + 8, ly + 6), color, -1)
        cv2.putText(frame, lbl, (lx + 2, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        badge = "FRESH  OK" if fresh else "ROTTEN  X"
        bcol = (0, 200, 80) if fresh else (0, 0, 220)
        cv2.putText(frame, badge, (rx1 + 6, ry1 + 36),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, bcol, 2, cv2.LINE_AA)

    elif ok and stable_cls and stable_cls in FRUIT_CLASSES:
        prog = stable_count / STABLE_NEED
        bw = rx2 - rx1
        cv2.putText(frame, "Scanning...", (rx1, ry2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 60), 2)
        cv2.rectangle(frame, (rx1, ry2 + 36), (rx2, ry2 + 50), (50, 50, 50), -1)
        cv2.rectangle(frame, (rx1, ry2 + 36),
                      (rx1 + int(bw * prog), ry2 + 50), (0, 210, 210), -1)
    else:
        msg = "Place fruit here"
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        mx = rx1 + (rx2 - rx1) // 2 - mw // 2
        cv2.putText(frame, msg, (mx, ry2 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 2)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Top1: {cls}  {conf:.0%}  margin:{margin:.2f}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if last_top2:
        t1, t2 = last_top2[0], last_top2[1]
        cv2.putText(frame,
                    f"Top2: {t1[0]} {t1[1]:.0%}  |  {t2[0]} {t2[1]:.0%}",
                    (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

    max_e = math.log(len(CLASS_NAMES))
    cv2.rectangle(frame, (20, h - 95), (270, h - 77), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 95),
                  (20 + int(250 * conf), h - 77),
                  (0, 200, 80) if conf >= CONF_THR else (60, 60, 180), -1)
    cv2.putText(frame, f"Conf: {conf:.0%}  (need {CONF_THR:.0%})",
                (20, h - 99), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.rectangle(frame, (20, h - 69), (270, h - 51), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 69),
                  (20 + int(250 * min(margin / max(MARGIN_THR, 1e-6), 1.0)), h - 51),
                  (0, 200, 80) if margin >= MARGIN_THR else (60, 60, 180), -1)
    cv2.putText(frame, f"Margin: {margin:.2f}  (need {MARGIN_THR:.2f})",
                (20, h - 73), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.rectangle(frame, (20, h - 43), (270, h - 27), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 43),
                  (20 + int(250 * entropy / max_e), h - 27),
                  (0, 80, 220) if entropy > ENTROPY_THR else (0, 200, 80), -1)
    cv2.putText(frame,
                f"Entropy: {entropy:.2f}  ({'HIGH' if entropy > ENTROPY_THR else 'OK'})",
                (20, h - 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    curr = time.time()
    fps = 1 / max(curr - prev_time, 1e-6)
    prev_time = curr
    cv2.putText(frame, f"FPS:{fps:.0f}",
                (w - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    speech_state = (
        "OFF" if not speech_enabled
        else f"MUTED {int(silence_until - now) + 1}s" if now < silence_until
        else "ON"
    )
    speech_color = (0, 80, 220) if speech_state != "ON" else (0, 200, 80)
    cv2.putText(frame, f"SPEECH:{speech_state}",
                (w - 170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speech_color, 2)

    if frozen:
        cv2.putText(frame, "FROZEN", (w // 2 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    if debug:
        cv2.putText(frame,
                    f"DBG {cls} c={conf:.2f} m={margin:.2f} e={entropy:.2f} s={stable_count}",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 240), 1)

    cv2.putText(frame, "Q:Quit  M:Speech  S:Mute10s  F:Freeze  D:Debug",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (110, 110, 110), 1)

    cv2.imshow("Fruit Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        speech_enabled = not speech_enabled
        print(f"Speech {'ON' if speech_enabled else 'OFF'}")
    elif key == ord("s"):
        silence_until = now + 10
        print("Muted 10s")
    elif key == ord("f"):
        frozen = not frozen
        print("Frozen" if frozen else "Unfrozen")
    elif key == ord("d"):
        debug = not debug
        print(f"Debug {'ON' if debug else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
_q.put(None)
print("Done.")