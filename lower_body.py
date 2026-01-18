import numpy as np

from bicep_curl import calculate_angle


def _init_state():
    return {
        "count": 0,
        "stage": None,
        "last_angle": None,
        "active_leg": None,
    }


def _leg_angle(hip, knee, ankle, confidence_threshold):
    if min(hip[2], knee[2], ankle[2]) < confidence_threshold:
        return None
    return calculate_angle(hip[:2], knee[:2], ankle[:2])


def _update_rep_state(state, angle, down_angle, up_angle):
    if angle > up_angle:
        if state.get("stage") == "down":
            state["count"] += 1
        state["stage"] = "up"
    elif angle < down_angle:
        state["stage"] = "down"


def calculate_squat_score(depth, knee_status, left_knee_angle, right_knee_angle):
    """
    Calculate overall squat quality score out of 100.
    """
    score = 0

    if depth > 60:
        depth_score = 33
    elif depth > 50:
        depth_score = 29
    elif depth > 40:
        depth_score = 24
    elif depth > 30:
        depth_score = 16
    else:
        depth_score = 7
    score += depth_score

    if knee_status == "Knees in position":
        knee_score = 33
    else:
        knee_score = 13
    score += knee_score

    knee_angle_diff = abs(left_knee_angle - right_knee_angle)
    if knee_angle_diff < 10:
        symmetry_score = 34
    elif knee_angle_diff < 20:
        symmetry_score = 27
    elif knee_angle_diff < 35:
        symmetry_score = 20
    else:
        symmetry_score = 10
    score += symmetry_score

    return round(score, 1)


def calculate_lunge_score(front_knee_depth, back_knee_depth, back_status, symmetry_angle):
    """
    Calculate overall lunge quality score out of 100.
    """
    score = 0

    if front_knee_depth > 60:
        front_score = 33
    elif front_knee_depth > 50:
        front_score = 29
    elif front_knee_depth > 40:
        front_score = 24
    elif front_knee_depth > 30:
        front_score = 16
    else:
        front_score = 7
    score += front_score

    if back_knee_depth > 50:
        back_score = 33
    elif back_knee_depth > 40:
        back_score = 29
    elif back_knee_depth > 30:
        back_score = 24
    elif back_knee_depth > 20:
        back_score = 16
    else:
        back_score = 7
    score += back_score

    if back_status == "Back straight":
        posture_score = 34
    elif back_status == "Back slightly forward":
        posture_score = 22
    else:
        posture_score = 10
    score += posture_score

    return round(score, 1)


def analyze_squat_form(keypoints, confidence_threshold=0.25, frame_shape=None):
    """
    Analyze squat form from front view.
    Returns: (depth_percent, feedback, left_knee_angle, right_knee_angle, knee_status)
    """
    shaped = np.squeeze(keypoints)

    left_hip_idx = 11
    right_hip_idx = 12
    left_knee_idx = 13
    right_knee_idx = 14
    left_ankle_idx = 15
    right_ankle_idx = 16

    if frame_shape is not None:
        frame_height, frame_width = frame_shape[0], frame_shape[1]
        left_hip = shaped[left_hip_idx][:2] * [frame_height, frame_width]
        right_hip = shaped[right_hip_idx][:2] * [frame_height, frame_width]
        left_knee = shaped[left_knee_idx][:2] * [frame_height, frame_width]
        right_knee = shaped[right_knee_idx][:2] * [frame_height, frame_width]
        left_ankle = shaped[left_ankle_idx][:2] * [frame_height, frame_width]
        right_ankle = shaped[right_ankle_idx][:2] * [frame_height, frame_width]
    else:
        left_hip = shaped[left_hip_idx][:2]
        right_hip = shaped[right_hip_idx][:2]
        left_knee = shaped[left_knee_idx][:2]
        right_knee = shaped[right_knee_idx][:2]
        left_ankle = shaped[left_ankle_idx][:2]
        right_ankle = shaped[right_ankle_idx][:2]

    left_hip_conf = shaped[left_hip_idx][2]
    right_hip_conf = shaped[right_hip_idx][2]
    left_knee_conf = shaped[left_knee_idx][2]
    right_knee_conf = shaped[right_knee_idx][2]
    left_ankle_conf = shaped[left_ankle_idx][2]
    right_ankle_conf = shaped[right_ankle_idx][2]

    visible_count = sum(
        [
            left_hip_conf > confidence_threshold,
            right_hip_conf > confidence_threshold,
            left_knee_conf > confidence_threshold,
            right_knee_conf > confidence_threshold,
            left_ankle_conf > confidence_threshold,
            right_ankle_conf > confidence_threshold,
        ]
    )
    if visible_count < 3:
        return None, "Insufficient visibility", 0, 0, "N/A"

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    squat_depth_percent = max(0, min(100, (170 - avg_knee_angle) / 0.8))

    knees_in_position = True
    ankle_conf_threshold = 0.2

    if left_knee_conf > confidence_threshold and left_ankle_conf > ankle_conf_threshold:
        knee_to_ankle_dx = left_ankle[0] - left_knee[0]
        knee_to_ankle_dy = left_ankle[1] - left_knee[1]
        if knee_to_ankle_dy > 5:
            if knee_to_ankle_dx < -15 or knee_to_ankle_dx > 20:
                knees_in_position = False

    if right_knee_conf > confidence_threshold and right_ankle_conf > ankle_conf_threshold:
        knee_to_ankle_dx = right_ankle[0] - right_knee[0]
        knee_to_ankle_dy = right_ankle[1] - right_knee[1]
        if knee_to_ankle_dy > 5:
            if knee_to_ankle_dx > 15 or knee_to_ankle_dx < -20:
                knees_in_position = False

    knee_status = "Knees in position" if knees_in_position else "Knees out of position"

    if squat_depth_percent < 15:
        feedback = "Go deeper"
    elif squat_depth_percent > 60:
        feedback = "Excellent depth"
    elif squat_depth_percent > 35:
        feedback = "Good depth"
    else:
        feedback = f"Depth: {squat_depth_percent:.0f}%"

    return (
        squat_depth_percent,
        feedback,
        left_knee_angle,
        right_knee_angle,
        knee_status,
    )


def analyze_lunge_form(keypoints, confidence_threshold=0.25, frame_shape=None):
    """
    Analyze lunge form from front view.
    Returns: (front_knee_depth, back_knee_depth, back_status, feedback, symmetry_angle)
    """
    shaped = np.squeeze(keypoints)

    left_shoulder_idx = 5
    right_shoulder_idx = 6
    left_hip_idx = 11
    right_hip_idx = 12
    left_knee_idx = 13
    right_knee_idx = 14
    left_ankle_idx = 15
    right_ankle_idx = 16

    if frame_shape is not None:
        frame_height, frame_width = frame_shape[0], frame_shape[1]
        left_shoulder = shaped[left_shoulder_idx][:2] * [frame_height, frame_width]
        right_shoulder = shaped[right_shoulder_idx][:2] * [frame_height, frame_width]
        left_hip = shaped[left_hip_idx][:2] * [frame_height, frame_width]
        right_hip = shaped[right_hip_idx][:2] * [frame_height, frame_width]
        left_knee = shaped[left_knee_idx][:2] * [frame_height, frame_width]
        right_knee = shaped[right_knee_idx][:2] * [frame_height, frame_width]
        left_ankle = shaped[left_ankle_idx][:2] * [frame_height, frame_width]
        right_ankle = shaped[right_ankle_idx][:2] * [frame_height, frame_width]
    else:
        left_shoulder = shaped[left_shoulder_idx][:2]
        right_shoulder = shaped[right_shoulder_idx][:2]
        left_hip = shaped[left_hip_idx][:2]
        right_hip = shaped[right_hip_idx][:2]
        left_knee = shaped[left_knee_idx][:2]
        right_knee = shaped[right_knee_idx][:2]
        left_ankle = shaped[left_ankle_idx][:2]
        right_ankle = shaped[right_ankle_idx][:2]

    left_hip_conf = shaped[left_hip_idx][2]
    right_hip_conf = shaped[right_hip_idx][2]
    left_knee_conf = shaped[left_knee_idx][2]
    right_knee_conf = shaped[right_knee_idx][2]
    left_ankle_conf = shaped[left_ankle_idx][2]
    right_ankle_conf = shaped[right_ankle_idx][2]
    left_shoulder_conf = shaped[left_shoulder_idx][2]
    right_shoulder_conf = shaped[right_shoulder_idx][2]

    visible_count = sum(
        [
            left_hip_conf > confidence_threshold,
            right_hip_conf > confidence_threshold,
            left_knee_conf > confidence_threshold,
            right_knee_conf > confidence_threshold,
            left_ankle_conf > confidence_threshold,
            right_ankle_conf > confidence_threshold,
        ]
    )
    if visible_count < 3:
        return None, None, None, "Insufficient visibility", 0

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    if left_knee_angle < right_knee_angle:
        front_knee_depth = max(0, min(100, (170 - left_knee_angle) / 0.8))
        back_knee_depth = max(0, min(100, (170 - right_knee_angle) / 1.5))
    else:
        front_knee_depth = max(0, min(100, (170 - right_knee_angle) / 0.8))
        back_knee_depth = max(0, min(100, (170 - left_knee_angle) / 1.5))

    back_status = "N/A"
    if left_shoulder_conf > confidence_threshold and right_shoulder_conf > confidence_threshold:
        avg_shoulder = [
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
        ]
        avg_hip = [
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
        ]

        shoulder_hip_dx = abs(avg_shoulder[0] - avg_hip[0])
        shoulder_hip_dy = abs(avg_shoulder[1] - avg_hip[1])
        if shoulder_hip_dy > 0:
            lean_angle = np.degrees(np.arctan(shoulder_hip_dx / shoulder_hip_dy))
        else:
            lean_angle = 0

        if lean_angle < 25:
            back_status = "Back straight"
        else:
            back_status = "Back too forward"

    if front_knee_depth < 15:
        feedback = "Front knee: Go deeper"
    elif front_knee_depth > 60:
        feedback = "Front knee: Excellent"
    else:
        feedback = "Good form"

    symmetry_angle = abs(left_knee_angle - right_knee_angle)
    return front_knee_depth, back_knee_depth, back_status, feedback, symmetry_angle


def detect_squat_movement(current_depth, is_currently_squatting):
    """
    Detect if person is actively squatting using hysteresis thresholds.
    """
    if current_depth is None:
        return is_currently_squatting

    squat_start_threshold = 25
    squat_end_threshold = 15

    if not is_currently_squatting:
        if current_depth > squat_start_threshold:
            is_currently_squatting = True
    else:
        if current_depth < squat_end_threshold:
            is_currently_squatting = False

    return is_currently_squatting


def detect_lunge_movement(current_front_depth, is_currently_lunging):
    """
    Detect if person is actively lunging using hysteresis thresholds.
    """
    if current_front_depth is None:
        return is_currently_lunging

    lunge_start_threshold = 25
    lunge_end_threshold = 12

    if not is_currently_lunging:
        if current_front_depth > lunge_start_threshold:
            is_currently_lunging = True
    else:
        if current_front_depth < lunge_end_threshold:
            is_currently_lunging = False

    return is_currently_lunging


def _init_squat_state():
    return {
        "count": 0,
        "score": 0,
        "is_active": False,
        "registered": False,
        "last_depth": None,
        "last_knee_status": "N/A",
        "last_score": 0,
        "last_feedback": None,
        "last_knee_angles": (0.0, 0.0),
    }


def _init_lunge_state():
    return {
        "count": 0,
        "score": 0,
        "is_active": False,
        "registered": False,
        "last_front_depth": None,
        "last_back_depth": None,
        "last_back": "N/A",
        "last_score": 0,
        "last_feedback": None,
        "last_symmetry": 0.0,
    }


def update_squat_state(keypoints, state, frame_shape, counter_active, confidence_threshold=0.4):
    if state is None:
        state = _init_squat_state()

    depth, feedback, left_knee_angle, right_knee_angle, knee_status = analyze_squat_form(
        keypoints, confidence_threshold, frame_shape
    )

    if depth is not None and state["is_active"]:
        current_score = calculate_squat_score(
            depth, knee_status, left_knee_angle, right_knee_angle
        )
        if abs(current_score - state["last_score"]) > 5:
            state["last_score"] = current_score
    elif not state["is_active"]:
        state["last_score"] = 0

    if depth is not None:
        state["is_active"] = detect_squat_movement(depth, state["is_active"])
        state["last_depth"] = depth
        state["last_knee_status"] = knee_status
        state["last_feedback"] = feedback
        state["last_knee_angles"] = (left_knee_angle, right_knee_angle)

        if counter_active:
            if state["is_active"] and not state["registered"]:
                state["count"] += 1
                squat_points = int((state["last_score"] / 100) * 20) + 10
                state["score"] += squat_points
                state["registered"] = True
            elif not state["is_active"]:
                state["registered"] = False
    else:
        state["is_active"] = False
        state["registered"] = False

    return state


def update_lunge_state(keypoints, state, frame_shape, counter_active, confidence_threshold=0.4):
    if state is None:
        state = _init_lunge_state()

    (
        front_depth,
        back_depth,
        back_status,
        feedback,
        symmetry,
    ) = analyze_lunge_form(keypoints, confidence_threshold, frame_shape)

    if front_depth is not None and state["is_active"]:
        current_score = calculate_lunge_score(
            front_depth, back_depth, back_status, symmetry
        )
        if abs(current_score - state["last_score"]) > 5:
            state["last_score"] = current_score
    elif not state["is_active"]:
        state["last_score"] = 0

    if front_depth is not None:
        state["is_active"] = detect_lunge_movement(front_depth, state["is_active"])
        state["last_front_depth"] = front_depth
        state["last_back_depth"] = back_depth
        state["last_back"] = back_status
        state["last_feedback"] = feedback
        state["last_symmetry"] = symmetry

        if counter_active:
            if state["is_active"] and not state["registered"]:
                state["count"] += 1
                lunge_points = int((state["last_score"] / 100) * 20) + 10
                state["score"] += lunge_points
                state["registered"] = True
            elif not state["is_active"]:
                state["registered"] = False
    else:
        state["is_active"] = False
        state["registered"] = False

    return state


def count_squats(keypoints, state, confidence_threshold=0.4, down_angle=100, up_angle=160):
    """
    Counts squats using knee angles from both legs (average when both are visible).
    """
    if state is None:
        state = _init_state()

    shaped = np.squeeze(keypoints)
    left_hip, left_knee, left_ankle = shaped[11], shaped[13], shaped[15]
    right_hip, right_knee, right_ankle = shaped[12], shaped[14], shaped[16]

    left_angle = _leg_angle(left_hip, left_knee, left_ankle, confidence_threshold)
    right_angle = _leg_angle(right_hip, right_knee, right_ankle, confidence_threshold)

    angles = []
    legs = []
    if left_angle is not None:
        angles.append(left_angle)
        legs.append("left")
    if right_angle is not None:
        angles.append(right_angle)
        legs.append("right")

    if not angles:
        return state

    if len(angles) == 2:
        angle = (angles[0] + angles[1]) / 2.0
        active_leg = "both"
    else:
        angle = angles[0]
        active_leg = legs[0]

    state["last_angle"] = angle
    state["active_leg"] = active_leg
    _update_rep_state(state, angle, down_angle, up_angle)
    return state


def count_lunges(keypoints, state, confidence_threshold=0.4, down_angle=95, up_angle=160):
    """
    Counts lunges using the most bent knee angle that is confidently visible.
    """
    if state is None:
        state = _init_state()

    shaped = np.squeeze(keypoints)
    left_hip, left_knee, left_ankle = shaped[11], shaped[13], shaped[15]
    right_hip, right_knee, right_ankle = shaped[12], shaped[14], shaped[16]

    left_angle = _leg_angle(left_hip, left_knee, left_ankle, confidence_threshold)
    right_angle = _leg_angle(right_hip, right_knee, right_ankle, confidence_threshold)

    angles = []
    legs = []
    if left_angle is not None:
        angles.append(left_angle)
        legs.append("left")
    if right_angle is not None:
        angles.append(right_angle)
        legs.append("right")

    if not angles:
        return state

    min_index = int(np.argmin(angles))
    angle = angles[min_index]
    active_leg = legs[min_index]

    state["last_angle"] = angle
    state["active_leg"] = active_leg
    _update_rep_state(state, angle, down_angle, up_angle)
    return state
