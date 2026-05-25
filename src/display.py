"""
Display Module: draws inference results onto frame
"""

import cv2

# Colours
GREEN = (0, 200, 0)
RED = (0,0,220)
YELLOW = (0, 200, 220)
WHITE = (255, 255, 255)


def annotate_frame(frame, faces):
    """
    Draw bounding box and inference results onto frame, 
    return annotated frames
    """
    for face in faces:
        x,y,w,h = face["box"]
        result = face["result"]

        identity = result.get("identity")
        liveness = result.get("liveness")
        emotion = result.get ("emotion")

        if identity is None:
            colour = YELLOW
        elif liveness:
            colour = GREEN
        else:
            colour = RED

        cv2.rectangle(frame, (x,y), (x+w, y+h), colour, 2)

        label = identity if identity else "UNKNOWN"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        if emotion:
              cv2.putText(frame, emotion, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        if liveness is False:                                                          
            cv2.putText(frame, "SPOOF DETECTED", (x, y + h + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        return frame