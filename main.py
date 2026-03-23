import tensorflow as tf
import numpy as np
import requests
import cv2
import threading
import time
from flask import Flask, jsonify
from flask_cors import CORS

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("🚀 STARTING SYSTEM")

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("waste_classifier_model_camera_fixed.keras")
class_names = ['biological', 'dry', 'metal']

# ===== URLs =====
CAM_URL = "http://10.239.48.196/capture"
ESP     = "http://10.239.48.41"

CHECK_DELAY = 0.8
last_time   = 0

# ===== SHARED STATE (read by Flask dashboard API) =====
state = {
    "esp_status":      "DISCONNECTED",
    "cam_status":      "DISCONNECTED",
    "last_prediction": "NONE",
    "last_label":      "NONE",
    "last_confidence": 0.0,
    "last_action":     "IDLE",
    "last_object":     "NONE",
    "trigger":         "NONE",
    "total":           0,
    "counts": {
        "BIOLOGICAL": 0,
        "DRY":        0,
        "METAL":      0,
    }
}
state_lock = threading.Lock()

# ===== FLASK API =====
flask_app = Flask(__name__)
CORS(flask_app)  # allows dashboard (local HTML file) to call this API

@flask_app.route("/state")
def get_state():
    with state_lock:
        return jsonify(state)

@flask_app.route("/health")
def health():
    return jsonify({"ok": True})

def run_flask():
    flask_app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)

# Start Flask in background thread
threading.Thread(target=run_flask, daemon=True).start()
print("📡 Dashboard API running on http://localhost:5050/state")

# ===== SAFE REQUEST =====
def safe_get(url):
    try:
        return requests.get(url, timeout=5)
    except:
        return None

# ===== CLEAN POPUP =====
def popup(frame, label, confidence):
    img = frame.copy()
    h, w, _ = img.shape
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    text = f"{label} ({confidence:.2f})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    x = (w - text_size[0]) // 2
    y = h - 30
    color_map = {"DRY": (0, 255, 0), "METAL": (255, 0, 0)}
    color = color_map.get(label, (0, 255, 255))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow("Captured", img)
    cv2.waitKey(3000)
    cv2.destroyWindow("Captured")

# ===== DRAW LIVE STATUS =====
def draw_overlay(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    with state_lock:
        esp_s = state["esp_status"]
        cam_s = state["cam_status"]
        pred  = state["last_prediction"]
    cv2.putText(frame, f"ESP: {esp_s}",  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"CAM: {cam_s}",  (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"PRED: {pred}",  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

# ===== MAIN LOOP =====
while True:
    try:
        # ===== ESP CHECK =====
        res = safe_get(ESP + "/check")

        if res is None:
            with state_lock:
                state["esp_status"] = "DISCONNECTED"
                state["trigger"]    = "NONE"
            print("❌ ESP DISCONNECTED")
            time.sleep(1)
            continue
        else:
            with state_lock:
                state["esp_status"] = "CONNECTED"

        trigger = res.text.strip()
        with state_lock:
            state["trigger"] = trigger

        # ===== CAMERA =====
        cam_res = safe_get(CAM_URL)

        if cam_res is None:
            with state_lock:
                state["cam_status"] = "DISCONNECTED"
            print("❌ CAMERA DISCONNECTED")
            continue
        else:
            with state_lock:
                state["cam_status"] = "CONNECTED"

        img_arr = np.array(bytearray(cam_res.content), dtype=np.uint8)
        frame   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ CAMERA FRAME ERROR")
            continue

        # ===== DETECTION =====
        if trigger == "DETECTED" and time.time() - last_time > 5:

            print("\n📦 OBJECT DETECTED")

            img = cv2.resize(frame, (224, 224))
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            pred       = model.predict(img, verbose=0)
            idx        = np.argmax(pred)
            confidence = float(np.max(pred))
            label      = class_names[idx].upper()

            pred_str = f"{label} ({confidence:.2f})"
            print(f"🧠 Prediction: {pred_str}")

            # Update shared state for dashboard
            with state_lock:
                state["last_prediction"] = pred_str
                state["last_label"]      = label
                state["last_confidence"] = confidence
                state["total"]          += 1
                state["counts"][label]  += 1

            # ===== SEND TO ESP =====
            reply = safe_get(ESP + "/" + label)
            if reply:
                print("📡 Sent:", label)
                print("📥 ESP Reply:", reply.text.strip())
            else:
                print("❌ ESP SEND FAILED")

            # ===== GET STATUS FROM ESP =====
            status_res = safe_get(ESP + "/status")
            if status_res:
                status_text = status_res.text
                print("📊 ESP STATUS:\n" + status_text)
                # Parse and store
                for line in status_text.split("\n"):
                    if line.startswith("Object:"):
                        with state_lock:
                            state["last_object"] = line.replace("Object:", "").strip()
                    if line.startswith("Action:"):
                        with state_lock:
                            state["last_action"] = line.replace("Action:", "").strip()

            # ===== POPUP =====
            threading.Thread(target=popup, args=(frame, label, confidence)).start()

            last_time = time.time()

        # ===== SHOW LIVE =====
        display = draw_overlay(frame)
        cv2.imshow("Smart Waste System", display)

        if cv2.waitKey(1) == ord('q'):
            print("🛑 EXIT")
            break

        time.sleep(CHECK_DELAY)

    except Exception as e:
        print("❌ ERROR:", e)

cv2.destroyAllWindows()