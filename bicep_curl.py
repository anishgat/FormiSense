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


def _init_rep_metrics():
    return {
        "rep_active": False,
        "min_angle": None,
        "max_angle": None,
        "baseline_elbow_offset": None,
        "upper_arm_length": None,
        "max_elbow_drift": 0.0,
        "scores": [],
        "last_score": None,
        "last_rep": None,
    }


def _reset_rep_metrics(metrics, shoulder, elbow, angle):
    metrics["rep_active"] = True
    metrics["min_angle"] = angle
    metrics["max_angle"] = angle
    elbow_offset = np.array(elbow[:2]) - np.array(shoulder[:2])
    metrics["baseline_elbow_offset"] = elbow_offset
    metrics["upper_arm_length"] = float(np.linalg.norm(elbow_offset))
    metrics["max_elbow_drift"] = 0.0


def _update_rep_metrics(metrics, shoulder, elbow, angle):
    if not metrics["rep_active"]:
        return
    if metrics["min_angle"] is None or angle < metrics["min_angle"]:
        metrics["min_angle"] = angle
    if metrics["max_angle"] is None or angle > metrics["max_angle"]:
        metrics["max_angle"] = angle
    baseline = metrics["baseline_elbow_offset"]
    if baseline is not None:
        elbow_offset = np.array(elbow[:2]) - np.array(shoulder[:2])
        drift = float(np.linalg.norm(elbow_offset - baseline))
        if drift > metrics["max_elbow_drift"]:
            metrics["max_elbow_drift"] = drift


def _score_range(value, ideal_min, ideal_max, min_val, max_val):
    if value is None:
        return None
    if value < min_val or value > max_val:
        return 0.0
    if ideal_min <= value <= ideal_max:
        return 1.0
    if value < ideal_min:
        return (value - min_val) / (ideal_min - min_val)
    return (max_val - value) / (max_val - ideal_max)


def _score_ratio(value, ideal_max, max_val):
    if value is None:
        return None
    if value <= ideal_max:
        return 1.0
    if value >= max_val:
        return 0.0
    return 1.0 - (value - ideal_max) / (max_val - ideal_max)


def _compute_rep_score(metrics):
    min_angle = metrics["min_angle"]
    max_angle = metrics["max_angle"]
    top_score = _score_range(min_angle, 40, 50, 30, 70)
    bottom_score = _score_range(max_angle, 165, 170, 150, 180)
    rom_score = None
    if top_score is not None and bottom_score is not None:
        rom_score = (top_score + bottom_score) / 2.0

    elbow_ratio = None
    if metrics["upper_arm_length"] and metrics["upper_arm_length"] > 0:
        elbow_ratio = metrics["max_elbow_drift"] / metrics["upper_arm_length"]
    elbow_score = _score_ratio(elbow_ratio, 0.05, 0.2)

    weights = {
        "rom": 40,
        "elbow": 35,
    }
    weighted_scores = {
        "rom": rom_score,
        "elbow": elbow_score,
    }

    total = 0.0
    total_weight = 0.0
    for key, score in weighted_scores.items():
        if score is None:
            continue
        total += score * weights[key]
        total_weight += weights[key]

    if total_weight == 0:
        return None

    return {
        "total_score": round((total / total_weight) * 100.0, 1),
        "min_angle": min_angle,
        "max_angle": max_angle,
        "elbow_drift_ratio": elbow_ratio,
        "rom_score": rom_score,
        "elbow_score": elbow_score,
    }


def count_bicep_curls(keypoints, state, side="right", confidence_threshold=0.4, up_angle=90, down_angle=160):
    """
    Updates curl counts and stages for a single arm using elbow angles.
    - keypoints: Model output keypoints (1, 1, 17, 3).
    - state: Dict with count, stage, and per-rep scoring metrics.
    - side: "left" or "right" to select the tracked arm.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    if state is None or state.get("side") != side:
        state = {
            "count": 0,
            "stage": None,
            "metrics": _init_rep_metrics(),
            "side": side,
        }

    shaped = np.squeeze(keypoints)

    # Indices: right uses 5/7/9, left uses 6/8/10
    if side == "right":
        shoulder, elbow, wrist = shaped[5], shaped[7], shaped[9]
    else:
        shoulder, elbow, wrist = shaped[6], shaped[8], shaped[10]

    if min(shoulder[2], elbow[2], wrist[2]) < confidence_threshold:
        return state

    angle = calculate_angle(shoulder[:2], elbow[:2], wrist[:2])
    metrics = state.setdefault("metrics", _init_rep_metrics())

    _update_rep_metrics(metrics, shoulder, elbow, angle)

    if angle > down_angle:
        if state.get("stage") != "down":
            if state.get("stage") == "up":
                rep_score = _compute_rep_score(metrics)
                if rep_score is not None:
                    metrics["scores"].append(rep_score)
                    metrics["last_score"] = rep_score["total_score"]
                    metrics["last_rep"] = rep_score
            _reset_rep_metrics(metrics, shoulder, elbow, angle)
        state["stage"] = "down"
    elif angle < up_angle and state.get("stage") == "down":
        state["count"] += 1
        state["stage"] = "up"

    return state
