print("SCRIPT STARTED")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

print("IMPORTS DONE")

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 224

# Your trained ResNet weights
WEIGHTS_PATH = "/Users/lasithcharuka/Desktop/spoof/resnet_anti_spoof.weights.h5"

# Class mapping from training:
# {'real': 0, 'spoof': 1}
# So prediction close to 0 = REAL
# prediction close to 1 = SPOOF

# Adaptive thresholds
REAL_THRESHOLD = 0.25
SPOOF_THRESHOLD = 0.75

# Camera index:
# 0 = MacBook camera
# 1/2/3 = phone camera / external camera
CAMERA_INDEX = 0

# -----------------------------
# Build ResNet50 model
# -----------------------------
print("Building ResNet50 model...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Must match your fine-tuning setup
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.build((None, IMG_SIZE, IMG_SIZE, 3))

print("Loading trained ResNet weights...")
model.load_weights(WEIGHTS_PATH)
print("WEIGHTS LOADED")

# -----------------------------
# Face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Face cascade not loaded.")
    exit()

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
    print("Try changing CAMERA_INDEX to 1, 2, or 3.")
    exit()

print("WEBCAM OPENED")
print("REAL score zone    : prediction < 0.25")
print("UNCERTAIN zone     : 0.25 <= prediction <= 0.75")
print("SPOOF score zone   : prediction > 0.75")
print("Press q to quit.")

# -----------------------------
# Webcam loop
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # If no face detected
    if len(faces) == 0:
        cv2.putText(
            frame,
            "No face detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    else:
        # Use only the largest detected face
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        x, y, w, h = faces[0]

        # If face is too small, don't predict
        if w < 70 or h < 70:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                frame,
                "Move closer to camera",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:
            # Dynamic margin around face
            margin = int(0.25 * w)

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                # Resize for ResNet50
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                # Preprocess for ResNet50
                face_array = np.expand_dims(face_rgb, axis=0)
                face_array = preprocess_input(face_array)

                # Predict
                prediction = model.predict(face_array, verbose=0)[0][0]

                # -----------------------------
                # Adaptive Liveness Decision
                # -----------------------------
                if prediction < REAL_THRESHOLD:
                    label = "REAL"
                    confidence = 1 - prediction
                    color = (0, 255, 0)

                    feedback = "Attendance verification allowed"

                elif prediction > SPOOF_THRESHOLD:
                    label = "SPOOF"
                    confidence = prediction
                    color = (0, 0, 255)

                    feedback = "Spoof detected - verification blocked"

                else:
                    label = "UNCERTAIN"
                    confidence = 1 - abs(prediction - 0.5)
                    color = (0, 255, 255)

                    feedback = "Move closer / improve lighting"

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label
                label_text = f"{label}: {confidence * 100:.2f}%"
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # Raw score
                score_text = f"Score: {prediction:.4f}"
                cv2.putText(
                    frame,
                    score_text,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # Feedback
                cv2.putText(
                    frame,
                    feedback,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

    cv2.imshow("ResNet50 Adaptive Anti-Spoofing Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("WEBCAM CLOSED")