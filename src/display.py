"""
Display Module: draws inference results onto frame
"""

import cv2
from src import attendance_log

# Colours
GREEN = (0, 200, 0)
RED = (0,0,220)
YELLOW = (0, 200, 220)
WHITE = (255, 255, 255)


def annotate_frame(frame, faces, greeting=None):
    """
    Draw bounding box and inference results onto frame,
    return annotated frames. If a Greeting is provided, overlay a banner.
    """
    for face in faces:
        x,y,w,h = face["box"]
        result = face["result"]

        identity = result.get("identity")
        liveness = result.get("liveness")
        emotion = result.get ("emotion")

        #checking is system locked
        locked = result.get("locked", False)

        if locked:
            colour = (0,0,200)
        elif identity is None:
            colour = YELLOW
        elif liveness:
            colour = GREEN
        else:
            colour = RED

        cv2.rectangle(frame, (x,y), (x+w, y+h), colour, 2)

        if locked:
            # get remaining lockout time from attendance_log
            remaining = attendance_log.get_remaining_seconds()
            label = f"SYSTEM LOCKED {remaining}s"
        elif liveness is False:
            label = "SPOOF DETECTED"
        elif identity:
            label = identity 
        else:
            label = "Unknown"
            
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        if emotion:
              cv2.putText(frame, emotion, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        if liveness is False:
            cv2.putText(frame, "SPOOF DETECTED", (x, y + h + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

    if greeting is not None:
        _draw_greeting_banner(frame, greeting)

    return frame


def _draw_greeting_banner(frame, greeting):
    """Translucent dark bar across the top with the curated greeting message."""
    h, w = frame.shape[:2]
    banner_h = 70

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame, greeting.message, (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2,
    )

    sub = f"detected: {greeting.emotion} ({greeting.dominance * 100:.0f}%)"
    cv2.putText(
        frame, sub, (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )