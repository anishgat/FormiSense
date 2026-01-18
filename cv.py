# P3: Real-Time Pose Detection Using MoveNet Lightning and OpenCV
import tkinter as tk
from tkinter import ttk

import tensorflow as tf
import numpy as np
import cv2
from bicep_curl import count_bicep_curls
from lower_body import update_lunge_state, update_squat_state

# Load the TensorFlow Lite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

# Function to draw keypoints (joints) on the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    """
    Draws keypoints on the frame if their confidence exceeds the threshold.
    - frame: The image on which keypoints are drawn.
    - keypoints: The predicted keypoints from the model.
    - confidence_threshold: The minimum confidence value for rendering keypoints.
    """
    y, x, c = frame.shape  # Get the frame dimensions
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # Scale keypoints to image size
    
    for kp in shaped:
        ky, kx, kp_conf = kp  # Extract coordinates and confidence
        if kp_conf > confidence_threshold:  # Only draw keypoints above threshold
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)  # Draw a green circle

def select_exercise():
    root = tk.Tk()
    root.title("Select Exercise")
    root.resizable(False, False)

    selection = {"value": None}
    selected = tk.StringVar(value="bicep_curl")

    ttk.Label(root, text="Choose an exercise to start").pack(pady=10)

    options = [
        ("bicep_curl", "Bicep curls"),
        ("squat", "Squats"),
        ("lunge", "Lunges"),
    ]
    for value, label in options:
        ttk.Radiobutton(root, text=label, value=value, variable=selected).pack(
            anchor="w", padx=20, pady=2
        )

    def start():
        selection["value"] = selected.get()
        root.destroy()

    def on_close():
        selection["value"] = None
        root.destroy()

    ttk.Button(root, text="Start", command=start).pack(pady=12)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return selection["value"]


# Define connections between keypoints and their colors
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm',
    (0, 6): 'c', (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c',
    (5, 6): 'y', (5, 11): 'm', (6, 12): 'c', (11, 12): 'y',
    (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}

UPPER_BODY_EDGES = {
    (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c', (5, 6): 'y'
}

LOWER_BODY_EDGES = {
    (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c', (11, 12): 'y'
}

# Function to draw connections (bones) between keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold):
    """
    Draws connections (edges) between keypoints if both keypoints have confidence 
    above the threshold.
    - frame: The image on which connections are drawn.
    - keypoints: The predicted keypoints from the model.
    - edges: A dictionary mapping pairs of keypoints to their colors.
    - confidence_threshold: The minimum confidence value for rendering connections.
    """
    y, x, c = frame.shape  # Get the frame dimensions
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # Scale keypoints to image size
    
    for edge, color in edges.items():  # Loop through each connection
        p1, p2 = edge  # Get the indices of the two keypoints
        y1, x1, c1 = shaped[p1]  # First keypoint coordinates and confidence
        y2, x2, c2 = shaped[p2]  # Second keypoint coordinates and confidence
        
        # Draw a line only if both keypoints exceed the confidence threshold
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red line


SMOOTH_FRAMES = 5
keypoint_buffer = []


def smooth_keypoints(keypoints, buffer, smooth_frames=5):
    """
    Apply temporal smoothing to keypoints by averaging over multiple frames.
    """
    buffer.append(keypoints.copy())
    if len(buffer) > smooth_frames:
        buffer.pop(0)
    return np.mean(buffer, axis=0)


# Initialize webcam capture
exercise = select_exercise()
if not exercise:
    raise SystemExit("No exercise selected.")

cap = cv2.VideoCapture(0)  # Open webcam (use '0' if there's only one camera)
exercise_state = None
tracked_side = "right"
counter_active = False
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Exit loop if no frame is captured
    
    frame = cv2.flip(frame, 1)  # Mirror the camera view for a more natural display
    
    # Preprocess the frame for the model
    img = frame.copy()  # Make a copy of the frame
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)  # Resize with padding
    input_image = tf.cast(img, dtype=tf.float32)  # Convert to float32 for the model
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run inference on the input image
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()  # Invoke the model
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])  # Get predictions
    keypoints_with_scores = smooth_keypoints(
        keypoints_with_scores, keypoint_buffer, SMOOTH_FRAMES
    )
    
    # Render keypoints and connections on the frame
    if exercise == "bicep_curl":
        edges_to_draw = UPPER_BODY_EDGES
    else:
        edges_to_draw = LOWER_BODY_EDGES
    draw_connections(frame, keypoints_with_scores, edges_to_draw, 0.4)  # Draw connections
    draw_keypoints(frame, keypoints_with_scores, 0.4)  # Draw keypoints
    
    # Update exercise counts
    if exercise == "bicep_curl":
        exercise_state = count_bicep_curls(
            keypoints_with_scores, exercise_state, side=tracked_side
        )
        if exercise_state is not None:
            score = None
            last_rep = None
            metrics = exercise_state.get("metrics")
            if metrics:
                score = metrics.get("last_score")
                last_rep = metrics.get("last_rep")
            score_text = f"{score:.1f}" if score is not None else "--"

            rom = last_rep.get("rom_score") if last_rep else None
            elbow = last_rep.get("elbow_score") if last_rep else None
            def format_component(value):
                if value is None:
                    return "--"
                return f"{value * 100.0:.0f}"

            components_text = (
                f"ROM {format_component(rom)}  ELB {format_component(elbow)}"
            )
            side_label = "L" if tracked_side == "left" else "R"
            cv2.putText(
                frame,
                f"{side_label} Curls: {exercise_state['count']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{side_label} Score: {score_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{side_label} {components_text}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    elif exercise == "squat":
        exercise_state = update_squat_state(
            keypoints_with_scores, exercise_state, frame.shape, counter_active
        )
        if exercise_state is not None:
            squat_status = "SQUATTING" if exercise_state["is_active"] else "STANDING"
            squat_color = (0, 255, 255) if exercise_state["is_active"] else (200, 200, 200)
            counter_status = "Counter: ON" if counter_active else "Counter: OFF"
            counter_color = (0, 255, 0) if counter_active else (0, 0, 255)
            back_status = exercise_state.get("last_back") or "N/A"
            last_score = exercise_state.get("last_score", 0)
            score_color = (
                (0, 255, 0)
                if last_score > 70
                else (0, 165, 255)
                if last_score > 50
                else (0, 0, 255)
            )
            feedback = exercise_state.get("last_feedback")
            cv2.putText(
                frame,
                f"Status: {squat_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                squat_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                counter_status,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                counter_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Squats: {exercise_state['count']}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Total Score: {exercise_state['score']}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Back: {back_status}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if back_status == "Back straight" else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Squat Score: {last_score:.1f}/100",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                score_color,
                2,
                cv2.LINE_AA,
            )
            if feedback:
                cv2.putText(
                    frame,
                    feedback,
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    elif exercise == "lunge":
        exercise_state = update_lunge_state(
            keypoints_with_scores, exercise_state, frame.shape, counter_active
        )
        if exercise_state is not None:
            lunge_status = "LUNGING" if exercise_state["is_active"] else "STANDING"
            lunge_color = (0, 255, 255) if exercise_state["is_active"] else (200, 200, 200)
            counter_status = "Counter: ON" if counter_active else "Counter: OFF"
            counter_color = (0, 255, 0) if counter_active else (0, 0, 255)
            back_status = exercise_state.get("last_back") or "N/A"
            last_score = exercise_state.get("last_score", 0)
            score_color = (
                (0, 255, 0)
                if last_score > 70
                else (0, 165, 255)
                if last_score > 50
                else (0, 0, 255)
            )
            feedback = exercise_state.get("last_feedback")
            cv2.putText(
                frame,
                f"Status: {lunge_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                lunge_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                counter_status,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                counter_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Lunges: {exercise_state['count']}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Total Score: {exercise_state['score']}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Back: {back_status}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if back_status == "Back straight" else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Lunge Score: {last_score:.1f}/100",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                score_color,
                2,
                cv2.LINE_AA,
            )
            if feedback:
                cv2.putText(
                    frame,
                    feedback,
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    
    # Display the output frame
    cv2.imshow('MoveNet Lightning', frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s') and exercise in ("squat", "lunge"):
        counter_active = not counter_active
    if key == ord('r') and exercise in ("squat", "lunge"):
        exercise_state = None

# Release resources
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
