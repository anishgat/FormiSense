import numpy as np


def calculate_angle(point_a, point_b, point_c):
    """
    Calculates the angle at point_b formed by point_a and point_c.
    Points are expected as (y, x) coordinates.
    """
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)

    radians = np.arctan2(c[0] - b[0], c[1] - b[1]) - np.arctan2(a[0] - b[0], a[1] - b[1])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def count_bicep_curls(keypoints, state, confidence_threshold=0.4, up_angle=90, down_angle=160):
    """
    Updates curl counts and stages for both arms using elbow angles.
    - keypoints: Model output keypoints (1, 1, 17, 3).
    - state: Dict with left/right counts and stages.
    """
    if state is None:
        state = {
            "left_count": 0,
            "right_count": 0,
            "left_stage": None,
            "right_stage": None,
        }

    shaped = np.squeeze(keypoints)

    # Indices per request: right uses 5/7/9, left uses 6/8/10
    right_shoulder, right_elbow, right_wrist = shaped[5], shaped[7], shaped[9]
    left_shoulder, left_elbow, left_wrist = shaped[6], shaped[8], shaped[10]

    def update_arm(side, shoulder, elbow, wrist):
        if min(shoulder[2], elbow[2], wrist[2]) < confidence_threshold:
            return
        angle = calculate_angle(shoulder[:2], elbow[:2], wrist[:2])
        stage_key = f"{side}_stage"
        count_key = f"{side}_count"

        if angle > down_angle:
            state[stage_key] = "down"
        if angle < up_angle and state.get(stage_key) == "down":
            state[count_key] += 1
            state[stage_key] = "up"

    update_arm("left", left_shoulder, left_elbow, left_wrist)
    update_arm("right", right_shoulder, right_elbow, right_wrist)

    return state
