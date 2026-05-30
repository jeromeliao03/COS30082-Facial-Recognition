"""
Display Module: draws inference results onto frame
"""

import cv2
from src import attendance_log

# Colours
GREEN = (0, 200, 0)
RED = (0,0,220)
YELLOW = (0, 200, 220)
ORANGE = (0, 140, 255)
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
        emotion = result.get("emotion")
        collecting = result.get("tracking") == "collecting"

        #checking is system locked
        locked = result.get("locked", False)

        if locked:
            colour = (0,0,200)
        elif collecting:
            colour = ORANGE
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
        elif collecting:
            label = "Recognising..."
        elif liveness is False:
            label = "SPOOF DETECTED"
        elif identity:
            label = identity 
        else:
            label = "Unknown"
            
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        if emotion:
              cv2.putText(frame, emotion, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    if greeting is not None:
        _draw_greeting_banner(frame, greeting)

    return frame


def _draw_greeting_banner(frame, greeting):
    """Translucent dark bar across the top with the curated greeting message.
    Wraps long messages to multiple lines and grows the banner to fit."""
    h, w = frame.shape[:2]
    padding_x = 15
    line_gap = 28
    sub_gap = 22
    top_offset = 32

    font = cv2.FONT_HERSHEY_SIMPLEX
    msg_scale, msg_thick = 0.7, 2
    sub_scale, sub_thick = 0.5, 1

    max_text_w = w - 2 * padding_x
    lines = _wrap_text(greeting.message, font, msg_scale, msg_thick, max_text_w)

    banner_h = top_offset + (len(lines) - 1) * line_gap + sub_gap + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (padding_x, top_offset + i * line_gap),
            font, msg_scale, WHITE, msg_thick,
        )

    sub = f"detected: {greeting.emotion} ({greeting.dominance * 100:.0f}%)"
    sub_y = top_offset + (len(lines) - 1) * line_gap + sub_gap
    cv2.putText(
        frame, sub, (padding_x, sub_y),
        font, sub_scale, (200, 200, 200), sub_thick,
    )


def _wrap_text(text, font, scale, thickness, max_width):
    """Word-wrap text to fit within max_width pixels using cv2 font metrics."""
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        (text_w, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if text_w <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines