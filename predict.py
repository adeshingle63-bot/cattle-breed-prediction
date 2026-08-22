import json
import numpy as np
import keras
from PIL import Image

# ==========================================
# CONFIG — adjust if your training used
# different values
# ==========================================
CONFIG_PATH = "models/config.json"
WEIGHTS_PATH = "models/model.weights.h5"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)   # must match training image size exactly

# ==========================================
# REBUILD MODEL FROM CONFIG + WEIGHTS
# (runs once, when the script starts)
# ==========================================
print("Rebuilding model from config...")
with open(CONFIG_PATH, "r") as f:
    config_dict = json.load(f)

model = keras.saving.deserialize_keras_object(config_dict)
model.load_weights(WEIGHTS_PATH)
print("Model rebuilt and weights loaded successfully.")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"{len(class_names)} classes found: {class_names}")


# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_breed(image_path):
    # 1. Load image and force 3-channel RGB
    img = Image.open(image_path).convert("RGB")

    # 2. Resize to match training input size
    img = img.resize(IMG_SIZE)

    # 3. Convert to array — DO NOT normalize here.
    #    EfficientNetB0 has its own internal rescaling layer
    #    and expects raw [0, 255] pixel values, exactly like
    #    image_dataset_from_directory gave it during training.
    img_array = np.array(img).astype("float32")

    # 4. Add batch dimension: (224,224,3) -> (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Run prediction
    predictions = model.predict(img_array)
    probs = predictions[0]

    # 6. Get top prediction
    top_idx = np.argmax(probs)
    top_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    # 7. Get top-3 predictions for extra context
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

    return top_class, confidence, top3


# ==========================================
# RUN SCRIPT
# ==========================================
if __name__ == "__main__":
    image_path = "data/test_cow.jpg"   # <-- change to your actual filename

    breed, confidence, top3 = predict_breed(image_path)

    print("\n===== PREDICTION RESULT =====")
    print(f"Predicted breed : {breed}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print("\nTop 3 guesses:")
    for name, p in top3:
        print(f"  {name:20s} {p * 100:.2f}%")