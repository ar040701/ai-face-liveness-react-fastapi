from config.settings import HEAD_TURN_RATIO_THRESHOLD

NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454


def detect_head_turn(landmarks, frame_shape):
    h, w, _ = frame_shape

    nose_x = landmarks[NOSE_TIP].x * w
    left_x = landmarks[LEFT_FACE].x * w
    right_x = landmarks[RIGHT_FACE].x * w

    face_width = abs(right_x - left_x)

    if face_width == 0:
        return "CENTER"

    face_center = (left_x + right_x) / 2

    offset_ratio = (nose_x - face_center) / face_width

    # If direction feels reversed in your app, swap LEFT and RIGHT below.
    if offset_ratio > HEAD_TURN_RATIO_THRESHOLD:
        return "LEFT"

    if offset_ratio < -HEAD_TURN_RATIO_THRESHOLD:
        return "RIGHT"

    return "CENTER"