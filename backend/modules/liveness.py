import random
import time

from modules.blink_detection import detect_blink
from modules.head_pose import detect_head_turn
from modules.anti_spoof import AntiSpoofModel
from modules.face_landmarker import FaceLandmarkerDetector
from modules.image_quality import check_image_quality

from config.settings import (
    RANDOM_CHALLENGE_COUNT,
    CHALLENGE_WARNING_SECONDS,
    CHALLENGE_TIMEOUT_SECONDS,
)


face_detector = FaceLandmarkerDetector()
anti_spoof_model = AntiSpoofModel()


class LivenessState:
    def __init__(self):
        self.blink_count = 0
        self.was_blinking = False

        self.looked_left = False
        self.looked_right = False

        self.available_challenges = [
            "BLINK",
            "TURN_LEFT",
            "TURN_RIGHT",
        ]

        self.challenges = random.sample(
            self.available_challenges,
            k=min(RANDOM_CHALLENGE_COUNT, len(self.available_challenges)),
        )

        self.current_challenge_index = 0
        self.completed_challenges = []

        self.challenge_started_at = time.time()
        self.verified = False


state = LivenessState()


def reset_liveness():
    global state
    state = LivenessState()


def crop_face_from_landmarks(frame, landmarks, padding=0.25):
    h, w, _ = frame.shape

    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x_min = int(max(0, min(xs)))
    x_max = int(min(w, max(xs)))
    y_min = int(max(0, min(ys)))
    y_max = int(min(h, max(ys)))

    face_width = x_max - x_min
    face_height = y_max - y_min

    pad_x = int(face_width * padding)
    pad_y = int(face_height * padding)

    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    face_crop = frame[y_min:y_max, x_min:x_max]

    if face_crop.size == 0:
        return frame

    return face_crop


def format_challenge(challenge):
    if challenge == "BLINK":
        return "Blink now"

    if challenge == "TURN_LEFT":
        return "Turn your head left"

    if challenge == "TURN_RIGHT":
        return "Turn your head right"

    if challenge is None:
        return "All challenges completed"

    return str(challenge)


def get_current_challenge():
    if state.current_challenge_index >= len(state.challenges):
        return None

    return state.challenges[state.current_challenge_index]


def get_base_response():
    current_challenge = get_current_challenge()

    return {
        "blink_count": state.blink_count,
        "head_left": state.looked_left,
        "head_right": state.looked_right,
        "current_challenge": current_challenge,
        "current_challenge_text": format_challenge(current_challenge),
        "completed_challenges": state.completed_challenges,
        "challenges": state.challenges,
        "challenge_ok": current_challenge is None,
    }


def challenge_elapsed_seconds():
    return time.time() - state.challenge_started_at


def challenge_time_left():
    remaining = CHALLENGE_TIMEOUT_SECONDS - challenge_elapsed_seconds()
    return max(0, int(remaining))


def challenge_warning():
    elapsed = challenge_elapsed_seconds()
    return elapsed >= CHALLENGE_WARNING_SECONDS


def challenge_timed_out():
    elapsed = challenge_elapsed_seconds()
    return elapsed >= CHALLENGE_TIMEOUT_SECONDS


def move_to_next_challenge():
    state.current_challenge_index += 1
    state.challenge_started_at = time.time()


def check_current_challenge(is_blinking, head_direction):
    current_challenge = get_current_challenge()

    if current_challenge is None:
        return True

    challenge_passed = False

    if current_challenge == "BLINK":
        if is_blinking and not state.was_blinking:
            challenge_passed = True

    elif current_challenge == "TURN_LEFT":
        if head_direction == "LEFT":
            challenge_passed = True

    elif current_challenge == "TURN_RIGHT":
        if head_direction == "RIGHT":
            challenge_passed = True

    if challenge_passed:
        state.completed_challenges.append(current_challenge)
        move_to_next_challenge()

    return get_current_challenge() is None


def build_common_response(
    *,
    base,
    status,
    message,
    head_direction="N/A",
    ear=None,
    spoof_result=None,
    spoof_ok=False,
    quality_result=None,
    challenge_ok=False,
    time_left=None,
    warning=False,
):
    response = {
        **base,
        "status": status,
        "message": message,
        "head_direction": head_direction,
        "spoof_model_available": anti_spoof_model.model_available,
        "spoof_score": None,
        "real_score": None,
        "fake_score": None,
        "anti_spoof_decision": None,
        "spoof_ok": spoof_ok,
        "quality_ok": None,
        "quality_message": None,
        "severe_quality_fail": None,
        "brightness": None,
        "blur_score": None,
        "face_area_ratio": None,
        "brightness_ok": None,
        "blur_ok": None,
        "face_size_ok": None,
        "face_center_ok": None,
        "challenge_ok": challenge_ok,
        "challenge_warning": warning,
        "challenge_time_left": time_left,
    }

    if ear is not None:
        response["ear"] = round(ear, 3)

    if spoof_result is not None:
        response["spoof_model_available"] = spoof_result.get("available", False)
        response["spoof_score"] = round(spoof_result.get("score", 0.0), 3)
        response["real_score"] = round(spoof_result.get("real_score", 0.0), 3)
        response["fake_score"] = round(spoof_result.get("fake_score", 0.0), 3)
        response["anti_spoof_decision"] = spoof_result.get("decision")

    if quality_result is not None:
        response["quality_ok"] = quality_result.get("quality_ok")
        response["quality_message"] = quality_result.get("message")
        response["severe_quality_fail"] = quality_result.get("severe_quality_fail")
        response["brightness"] = quality_result.get("brightness")
        response["blur_score"] = quality_result.get("blur_score")
        response["face_area_ratio"] = quality_result.get("face_area_ratio")
        response["brightness_ok"] = quality_result.get("brightness_ok")
        response["blur_ok"] = quality_result.get("blur_ok")
        response["face_size_ok"] = quality_result.get("face_size_ok")
        response["face_center_ok"] = quality_result.get("face_center_ok")

    return response


def check_liveness(frame):
    base = get_base_response()

    if state.verified:
        return build_common_response(
            base=base,
            status="LIVE",
            message="Live person already verified",
            head_direction="CENTER",
            spoof_result={
                "available": anti_spoof_model.model_available,
                "score": 1.0,
                "real_score": 1.0,
                "fake_score": 0.0,
                "decision": "REAL",
            },
            spoof_ok=True,
            quality_result={
                "quality_ok": True,
                "message": "Verified",
                "severe_quality_fail": False,
                "brightness": None,
                "blur_score": None,
                "face_area_ratio": None,
                "brightness_ok": None,
                "blur_ok": None,
                "face_size_ok": None,
                "face_center_ok": None,
            },
            challenge_ok=True,
            time_left=0,
            warning=False,
        )

    landmarks = face_detector.detect(frame)

    if landmarks is None:
        return build_common_response(
            base=base,
            status="NO_FACE",
            message="No face detected",
            head_direction="N/A",
            spoof_ok=False,
            quality_result={
                "quality_ok": False,
                "message": "No face detected",
                "severe_quality_fail": False,
                "brightness": None,
                "blur_score": None,
                "face_area_ratio": None,
                "brightness_ok": None,
                "blur_ok": None,
                "face_size_ok": None,
                "face_center_ok": None,
            },
            challenge_ok=False,
            time_left=challenge_time_left(),
            warning=False,
        )

    quality_result = check_image_quality(frame, landmarks)

    is_blinking, ear = detect_blink(landmarks, frame.shape)
    head_direction = detect_head_turn(landmarks, frame.shape)

    if is_blinking and not state.was_blinking:
        state.blink_count += 1

    if head_direction == "LEFT":
        state.looked_left = True

    if head_direction == "RIGHT":
        state.looked_right = True

    challenge_ok = check_current_challenge(
        is_blinking=is_blinking,
        head_direction=head_direction,
    )

    state.was_blinking = is_blinking

    face_crop = crop_face_from_landmarks(frame, landmarks)
    spoof_result = anti_spoof_model.predict(face_crop)

    anti_spoof_decision = spoof_result.get("decision")
    spoof_ok = spoof_result["available"] and anti_spoof_decision == "REAL"

    severe_quality_fail = quality_result.get("severe_quality_fail", False)
    timed_out = challenge_timed_out()
    warning = challenge_warning()
    time_left = challenge_time_left()

    # Production priority:
    # 1. Severe image issue
    # 2. Anti-spoof model error
    # 3. Spoof
    # 4. Live
    # 5. Challenge warning
    # 6. Timeout
    # 7. Checking

    if severe_quality_fail:
        status = "BAD_IMAGE"
        message = quality_result["message"]

    elif not spoof_result["available"] or anti_spoof_decision == "ERROR":
        status = "ANTI_SPOOF_ERROR"
        message = "Anti-spoof model missing or failed"

    elif anti_spoof_decision == "SPOOF":
        status = "SPOOF"
        message = "Spoof detected"

    elif challenge_ok and spoof_ok:
        state.verified = True
        status = "LIVE"
        message = "Live person verified"

    elif warning and not timed_out:
        status = "CHALLENGE_WARNING"
        message = f"Please complete the challenge. Time left: {time_left}s"

    elif timed_out:
        reset_liveness()
        base = get_base_response()
        status = "TIMEOUT"
        message = "Challenge expired. Please try again."

    elif anti_spoof_decision == "UNCERTAIN":
        status = "CHECKING"
        message = "Hold still and face the camera"

    else:
        status = "CHECKING"
        message = f"Challenge: {format_challenge(get_current_challenge())}"

    base = get_base_response()

    return build_common_response(
        base=base,
        status=status,
        message=message,
        head_direction=head_direction,
        ear=ear,
        spoof_result=spoof_result,
        spoof_ok=spoof_ok,
        quality_result=quality_result,
        challenge_ok=challenge_ok,
        time_left=time_left,
        warning=warning,
    )