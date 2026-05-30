import cv2
import os
import numpy as np
from datetime import datetime
from src.spoof_explainability import SpoofExplainability


MODEL_PATH = "models/spoof_model.keras"
OUTPUT_DIR = "output/spoof_explainability"

os.makedirs(OUTPUT_DIR, exist_ok=True)

explainer = SpoofExplainability(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Face cascade not loaded.")
    exit()

# 0 = MacBook camera
# Try 1, 2, or 3 for external/phone camera
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened.")
print("Press SPACE to generate and save a new Grad-CAM heatmap.")
print("Press q to quit.")

last_face = None
latest_heatmap_view = None
latest_result_text = "No heatmap captured yet"


def create_placeholder(height, width):
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "Press SPACE",
        (40, height // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )
    cv2.putText(
        placeholder,
        "to generate Grad-CAM",
        (40, height // 2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    return placeholder


while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    live_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) > 0:
        # Use largest detected face
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)

        x, y, w, h = faces[0]

        margin = int(0.25 * w)

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        last_face = frame[y1:y2, x1:x2]

        cv2.rectangle(live_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            live_frame,
            "Face detected - press SPACE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:
        cv2.putText(
            live_frame,
            "No face detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    # Prepare right-side panel
    height, width = live_frame.shape[:2]

    if latest_heatmap_view is None:
        right_panel = create_placeholder(height, width)
    else:
        right_panel = cv2.resize(latest_heatmap_view, (width, height))

    # Add labels to panels
    cv2.putText(
        live_frame,
        "LIVE CAMERA",
        (30, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        right_panel,
        "LATEST GRAD-CAM",
        (30, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        right_panel,
        latest_result_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Combine live feed and heatmap in one window
    combined_view = np.hstack((live_frame, right_panel))

    cv2.imshow("Spoof Explainability Live Test - Press SPACE / q", combined_view)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # SPACE key
    if key == 32:
        if last_face is None or last_face.size == 0:
            print("No face available for explanation.")
            continue

        try:
            result = explainer.explain_face(last_face)

            score = result["score"]
            label = result["label"]
            confidence = result["confidence"]
            overlay = result["overlay"]

            print("--------------------------------")
            print("Spoof model explanation generated")
            print("Prediction:", label)
            print("Spoof score:", score)
            print("Confidence:", confidence)
            print("--------------------------------")

            latest_result_text = f"{label} | Score: {score:.4f}"

            # Add readable text background to overlay
            cv2.rectangle(
                overlay,
                (5, 5),
                (390, 50),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                overlay,
                latest_result_text,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_path = os.path.join(
                OUTPUT_DIR,
                f"spoof_gradcam_{label}_{timestamp}.jpg"
            )

            cv2.imwrite(save_path, overlay)

            latest_heatmap_view = overlay.copy()

            print("Saved heatmap:", save_path)
            print("You can press SPACE again to capture another heatmap.")

        except Exception as e:
            print("ERROR while generating Grad-CAM:")
            print(e)

cap.release()
cv2.destroyAllWindows()
print("Closed.")