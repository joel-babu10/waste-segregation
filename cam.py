import tensorflow as tf
import numpy as np
import requests
import cv2
from PIL import Image
from io import BytesIO
import time
import threading

# ✅ IMPORTANT
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("waste_classifier_model_camera_fixed.keras")

# ✅ CORRECT ORDER
class_names = ['biological', 'dry', 'metal']

# ===== ESP32 CAM URL =====
url = "http://10.239.48.196/capture"

last_capture_time = 0
capture_interval = 15  # seconds


# ===== POPUP FUNCTION =====
def show_prediction_popup(frame, label):
    popup = frame.copy()

    cv2.putText(popup, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Captured Prediction", popup)
    cv2.waitKey(3000)
    cv2.destroyWindow("Captured Prediction")


while True:
    try:
        # ===== GET FRAME =====
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert('RGB')

        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ===== LIVE FEED =====
        cv2.imshow("Live ESP32 Camera", frame)

        current_time = time.time()

        # ===== EVERY 15 SEC =====
        if current_time - last_capture_time >= capture_interval:

            # ===== PREPROCESS (FIXED) =====
            img_resized = cv2.resize(frame, (224, 224))
            img_array = preprocess_input(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            # ===== PREDICT =====
            prediction = model.predict(img_array, verbose=0)

            print("Raw prediction:", prediction)

            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # ✅ CONFIDENCE FILTER
            if confidence < 0.6:
                label = "Uncertain"
            else:
                label = f"{predicted_class} ({confidence:.2f})"

            print("Captured:", label)

            # ===== POPUP THREAD =====
            threading.Thread(target=show_prediction_popup,
                             args=(frame, label)).start()

            last_capture_time = current_time

        # EXIT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)

cv2.destroyAllWindows()