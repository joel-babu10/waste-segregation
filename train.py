import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ✅ IMPORTANT: ADD THIS
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== DATASET PATH =====
dataset_path = r"C:\Users\joelv\ws\Dataset"

# ===== SETTINGS =====
img_size = (224, 224)
batch_size = 32

# ===== LOAD CLASSES =====
class_names = sorted(os.listdir(dataset_path))
print("🔥 Classes:", class_names)

image_paths = []
labels = []

for label, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    
    for file in os.listdir(class_folder):
        image_paths.append(os.path.join(class_folder, file))
        labels.append(label)

# ===== IDENTIFY CAMERA IMAGES =====
cam_images = []
cam_labels = []

for path, label in zip(image_paths, labels):
    name = os.path.basename(path)

    try:
        number = int(name.split("_")[1].split(".")[0])
    except:
        continue

    if (
        ("biological_" in name and 700 <= number <= 717) or
        ("plastic_" in name and 257 <= number <= 272) or
        ("metal_" in name and 361 <= number <= 370)
    ):
        cam_images.append(path)
        cam_labels.append(label)

# ===== REMOVE CAM IMAGES FROM MAIN DATA =====
remaining = [(p, l) for p, l in zip(image_paths, labels) if p not in cam_images]
remaining_paths, remaining_labels = zip(*remaining)

# ===== SPLIT =====
train_paths, val_paths, train_labels, val_labels = train_test_split(
    list(remaining_paths),
    list(remaining_labels),
    test_size=0.2,
    stratify=remaining_labels,
    random_state=42
)

# ===== ADD CAMERA IMAGES TO TRAIN =====
train_paths.extend(cam_images)
train_labels.extend(cam_labels)

print(f"\n✅ Camera images added: {len(cam_images)}")

# ===== IMAGE LOADER =====
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)

    # 🔥 FIX: USE CORRECT PREPROCESSING
    img = preprocess_input(img)

    return img, label

# ===== DATASET =====
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# ===== AUGMENTATION =====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
])

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ===== PERFORMANCE =====
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ===== MODEL =====
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# ===== FREEZE =====
for layer in base_model.layers[:-60]:
    layer.trainable = False

# ===== HEAD =====
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

# ===== COMPILE =====
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00007),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===== CALLBACKS =====
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)

# ===== TRAIN =====
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stop, reduce_lr]
)

# ===== SAVE =====
model.save("waste_classifier_model_camera_fixed.keras")

print("🔥 MODEL TRAINED WITH CORRECT PREPROCESSING")