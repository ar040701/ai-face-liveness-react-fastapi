import cv2
import numpy as np

from config.settings import (
    SEVERE_BRIGHTNESS_MIN,
    SEVERE_BRIGHTNESS_MAX,
    SEVERE_BLUR_MIN,
)


def check_image_quality(frame, landmarks=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_ok = SEVERE_BRIGHTNESS_MIN <= brightness <= SEVERE_BRIGHTNESS_MAX
    blur_ok = blur_score >= SEVERE_BLUR_MIN

    face_size_ok = True
    face_center_ok = True
    face_area_ratio = None
    face_box = None

    if landmarks is not None:
        h, w, _ = frame.shape

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min = max(0, min(xs))
        x_max = min(w, max(xs))
        y_min = max(0, min(ys))
        y_max = min(h, max(ys))

        face_width = x_max - x_min
        face_height = y_max - y_min

        face_area_ratio = (face_width * face_height) / float(w * h)

        face_center_x = (x_min + x_max) / 2
        face_center_y = (y_min + y_max) / 2

        frame_center_x = w / 2
        frame_center_y = h / 2

        center_offset_x = abs(face_center_x - frame_center_x) / w
        center_offset_y = abs(face_center_y - frame_center_y) / h

        # Relaxed. Do not use these as strict blockers in development.
        face_size_ok = face_area_ratio >= 0.03
        face_center_ok = center_offset_x <= 0.40 and center_offset_y <= 0.40

        face_box = {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "face_area_ratio": round(face_area_ratio, 3),
        }

    severe_quality_fail = not brightness_ok or not blur_ok

    quality_ok = brightness_ok and blur_ok and face_size_ok and face_center_ok

    message = "Image quality good"

    if brightness < SEVERE_BRIGHTNESS_MIN:
        message = "Image too dark"

    elif brightness > SEVERE_BRIGHTNESS_MAX:
        message = "Image too bright"

    elif blur_score < SEVERE_BLUR_MIN:
        message = "Image too blurry"

    elif not face_size_ok:
        message = "Move closer to camera"

    elif not face_center_ok:
        message = "Center your face"

    return {
        "quality_ok": quality_ok,
        "severe_quality_fail": severe_quality_fail,
        "message": message,
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "brightness_ok": brightness_ok,
        "blur_ok": blur_ok,
        "face_size_ok": face_size_ok,
        "face_center_ok": face_center_ok,
        "face_area_ratio": round(face_area_ratio, 3) if face_area_ratio is not None else None,
        "face_box": face_box,
    }