print("SCRIPT STARTED")

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

print("IMPORTS DONE")

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 128

WEIGHTS_PATH = "/Users/lasithcharuka/Desktop/spoof/resnet_anti_spoof.weights.h5"

# Class mapping from training:
# {'real': 0, 'spoof': 1}
# prediction close to 0 = REAL
# prediction close to 1 = SPOOF

REAL_THRESHOLD = 0.25
SPOOF_THRESHOLD = 0.70

# Temporal voting settings
FRAME_WINDOW = 10
MIN_VALID_FRAMES = 5
prediction_history = deque(maxlen=FRAME_WINDOW)

# Camera index
# 0 = MacBook camera
# 1/2/3 = phone/external camera
CAMERA_INDEX = 0

# Quality thresholds
MIN_FACE_SIZE = 50
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 225
MIN_BLUR_SCORE = 25

# ============================================================
# BUILD RESNET50 MODEL
# ============================================================

print("Building ResNet50 model...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_face_quality(face_crop, w, h):
    """
    Checks whether the detected face crop is good enough.
    Returns:
        quality_ok: True/False
        feedback: message to display
    """

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False, "Move closer / make face larger"

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray_face)

    if brightness < MIN_BRIGHTNESS:
        return False, "Improve lighting"

    if brightness > MAX_BRIGHTNESS:
        return False, "Reduce bright light / glare"

    blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    if blur_score < MIN_BLUR_SCORE:
        return False, "Keep still / image blurry"

    return True, "Quality OK"


def classify_frame(prediction):
    """
    Adaptive confidence decision.
    prediction close to 0 = real
    prediction close to 1 = spoof
    """

    if prediction < REAL_THRESHOLD:
        return "REAL"

    elif prediction > SPOOF_THRESHOLD:
        return "SPOOF"

    else:
        return "UNCERTAIN"


def get_temporal_decision(history):
    """
    Multi-frame voting decision.
    """

    if len(history) < MIN_VALID_FRAMES:
        return "COLLECTING"

    real_count = list(history).count("REAL")
    spoof_count = list(history).count("SPOOF")
    uncertain_count = list(history).count("UNCERTAIN")

    # Strong spoof voting
    if spoof_count >= 4:
        return "SPOOF"

    # Strong real voting
    if real_count >= 4:
        return "REAL"

    return "UNCERTAIN"


def draw_status(frame, message, color):
    """
    Draws main status message on top of screen.
    """

    cv2.putText(
        frame,
        message,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Face cascade not loaded.")
    exit()

# ============================================================
# OPEN WEBCAM
# ============================================================

# AVFOUNDATION works better on macOS
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
    print("Try CAMERA_INDEX = 1, 2, or 3.")
    exit()

print("WEBCAM OPENED")
print("Quality-Aware Temporal Liveness Verification started.")
print("Press q to quit.")

# ============================================================
# WEBCAM LOOP
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # More sensitive face detection for phone wallpaper/photo
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        prediction_history.clear()

        draw_status(
            frame,
            "No valid face detected - verification not allowed",
            (0, 255, 255)
        )

    else:
        # Use largest detected face only
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        x, y, w, h = faces[0]

        # Dynamic margin around detected face
        margin = int(0.30 * w)

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            prediction_history.clear()

            draw_status(
                frame,
                "Invalid face crop - retry",
                (0, 255, 255)
            )

        else:
            quality_ok, feedback = check_face_quality(face, w, h)

            if not quality_ok:
                prediction_history.clear()

                color = (0, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    feedback,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                draw_status(
                    frame,
                    "Quality check failed - verification paused",
                    color
                )

            else:
                # Resize face for ResNet50
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                # Preprocess
                face_array = np.expand_dims(face_rgb, axis=0)
                face_array = preprocess_input(face_array)

                # Predict
                prediction = model.predict(face_array, verbose=0)[0][0]

                # Frame-level label
                frame_label = classify_frame(prediction)
                prediction_history.append(frame_label)

                # Multi-frame decision
                final_decision = get_temporal_decision(prediction_history)

                if final_decision == "REAL":
                    color = (0, 255, 0)
                    message = "REAL - Attendance verification allowed"

                elif final_decision == "SPOOF":
                    color = (0, 0, 255)
                    message = "SPOOF - Verification blocked"

                elif final_decision == "UNCERTAIN":
                    color = (0, 255, 255)
                    message = "UNCERTAIN - Retry / improve conditions"

                else:
                    color = (255, 255, 0)
                    message = "Collecting stable frames..."

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Main system message
                draw_status(frame, message, color)

                # Frame label
                cv2.putText(
                    frame,
                    f"Frame label: {frame_label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                # Raw model score
                cv2.putText(
                    frame,
                    f"Score: {prediction:.4f}",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # Vote count summary
                real_count = list(prediction_history).count("REAL")
                spoof_count = list(prediction_history).count("SPOOF")
                uncertain_count = list(prediction_history).count("UNCERTAIN")

                vote_text = f"Votes R:{real_count} S:{spoof_count} U:{uncertain_count}"

                cv2.putText(
                    frame,
                    vote_text,
                    (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    cv2.imshow("Quality-Aware Temporal Liveness Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
print("WEBCAM CLOSED")