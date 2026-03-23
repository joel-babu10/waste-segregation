import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# ✅ IMPORTANT
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("waste_classifier_model_camera_fixed.keras")

# ✅ MUST MATCH TRAINING ORDER
classes = ['biological', 'dry', 'metal']

# ===== TEST FOLDER =====
test_dir = "test_images"

correct = 0
total = 0

for class_name in classes:
    class_path = os.path.join(test_dir, class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # ===== LOAD IMAGE =====
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)

        # 🔥 FIX: PREPROCESS PROPERLY
        img_array = preprocess_input(img_array)

        img_array = np.expand_dims(img_array, axis=0)

        # ===== PREDICT =====
        prediction = model.predict(img_array, verbose=0)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        print(f"{img_name} → Predicted: {predicted_class} ({confidence:.2f}) | Actual: {class_name}")

        if predicted_class == class_name:
            correct += 1

        total += 1

# ===== ACCURACY =====
accuracy = (correct / total) * 100
print(f"\n🔥 Accuracy: {accuracy:.2f}%")