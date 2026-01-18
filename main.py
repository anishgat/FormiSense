import time

import customtkinter
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from bicep_curl import count_bicep_curls
from lower_body import update_lunge_state, update_squat_state

MODEL_PATH = "3.tflite"
CONFIDENCE_THRESHOLD = 0.4
SMOOTH_FRAMES = 5
INPUT_SIZE = (192, 192)

FILTER_TYPES = [
    "Bicep curl",
    "Squat",
    "Lunges",
]

UPPER_BODY_EDGES = {
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
}

LOWER_BODY_EDGES = {
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
    (11, 12): "y",
}


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for ky, kx, kp_conf in shaped:
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge in edges:
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def smooth_keypoints(keypoints, buffer, smooth_frames):
    buffer.append(keypoints.copy())
    if len(buffer) > smooth_frames:
        buffer.pop(0)
    return np.mean(buffer, axis=0)


class App(customtkinter.CTk):
    def __init__(self) -> None:
        super().__init__()

        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        self.title("FormiSense")
        self.geometry("1100x700")
        self.minsize(900, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.session_active = False
        self.cap = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.keypoint_buffer = []
        self.exercise_state = None
        self.last_frame_time = None
        self.frame_after_id = None

        self._build_sidebar()
        self._build_preview()
        self._build_footer()
        self._load_model()
        self._on_exercise_change()

    def _build_sidebar(self) -> None:
        self.sidebar = customtkinter.CTkFrame(self, corner_radius=12, width=260)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

        title = customtkinter.CTkLabel(
            self.sidebar,
            text="FormiSense",
            font=customtkinter.CTkFont(size=22, weight="bold"),
        )
        title.pack(anchor="w", padx=16, pady=(16, 4))

        subtitle = customtkinter.CTkLabel(
            self.sidebar, text="Choose exercise", text_color="#9aa3b2"
        )
        subtitle.pack(anchor="w", padx=16, pady=(0, 12))

        self.filter_var = customtkinter.StringVar(value=FILTER_TYPES[0])
        exercise_frame = customtkinter.CTkFrame(self.sidebar, fg_color="transparent")
        exercise_frame.pack(fill="x", padx=16, pady=(0, 12))
        for filter_type in FILTER_TYPES:
            rb_filter = customtkinter.CTkRadioButton(
                exercise_frame,
                text=filter_type,
                variable=self.filter_var,
                value=filter_type,
                command=self._on_exercise_change,
            )
            rb_filter.pack(anchor="w", pady=4)

        side_frame = customtkinter.CTkFrame(self.sidebar, fg_color="transparent")
        side_frame.pack(fill="x", padx=16, pady=(0, 12))
        side_label = customtkinter.CTkLabel(side_frame, text="Tracked side")
        side_label.pack(anchor="w")
        self.side_var = customtkinter.StringVar(value="Right")
        self.side_menu = customtkinter.CTkOptionMenu(
            side_frame, values=["Right", "Left"], variable=self.side_var
        )
        self.side_menu.pack(fill="x", pady=(6, 0))

        counter_frame = customtkinter.CTkFrame(self.sidebar, fg_color="transparent")
        counter_frame.pack(fill="x", padx=16, pady=(0, 12))
        self.counter_var = customtkinter.BooleanVar(value=False)
        self.counter_switch = customtkinter.CTkSwitch(
            counter_frame, text="Rep counter", variable=self.counter_var
        )
        self.counter_switch.pack(anchor="w")

        spacer = customtkinter.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        actions = customtkinter.CTkFrame(self.sidebar, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=(0, 16))
        actions.grid_columnconfigure(0, weight=1)

        self.start_button = customtkinter.CTkButton(
            actions, text="Start Session", command=self.start_session
        )
        self.start_button.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.stop_button = customtkinter.CTkButton(
            actions,
            text="Stop",
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.stop_session,
        )
        self.stop_button.grid(row=1, column=0, sticky="ew")

    def _build_preview(self) -> None:
        self.preview = customtkinter.CTkFrame(self, corner_radius=12)
        self.preview.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        self.preview.grid_rowconfigure(1, weight=1)
        self.preview.grid_columnconfigure(0, weight=1)

        header = customtkinter.CTkFrame(self.preview, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 4))
        header.grid_columnconfigure(0, weight=1)

        self.status_label = customtkinter.CTkLabel(
            header, text="Camera preview", font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.fps_label = customtkinter.CTkLabel(header, text="FPS: --", text_color="#9aa3b2")
        self.fps_label.grid(row=0, column=1, sticky="e")

        self.image_display = customtkinter.CTkLabel(
            self.preview, text="Waiting for camera feed...", corner_radius=10
        )
        self.image_display.grid(row=1, column=0, sticky="nsew", padx=20, pady=16)

    def _build_footer(self) -> None:
        self.footer = customtkinter.CTkFrame(self, corner_radius=12)
        self.footer.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))
        self.footer.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.footer.grid_rowconfigure(1, weight=1)

        self.rep_label = customtkinter.CTkLabel(self.footer, text="Reps: 0")
        self.rep_label.grid(row=0, column=0, sticky="w", padx=16, pady=12)

        score_frame = customtkinter.CTkFrame(self.footer, fg_color="transparent")
        score_frame.grid(row=0, column=1, sticky="nsew", padx=16, pady=(8, 4))
        self.score_caption = customtkinter.CTkLabel(
            score_frame, text="Score", text_color="#9aa3b2"
        )
        self.score_caption.grid(row=0, column=0, sticky="w")
        self.score_value_label = customtkinter.CTkLabel(
            score_frame,
            text="--",
            font=customtkinter.CTkFont(size=28, weight="bold"),
            text_color="#fbbf24",
        )
        self.score_value_label.grid(row=1, column=0, sticky="w")

        self.side_label = customtkinter.CTkLabel(self.footer, text="Side: Right")
        self.side_label.grid(row=0, column=2, sticky="w", padx=16, pady=12)

        self.detail_label = customtkinter.CTkLabel(self.footer, text="Details: --")
        self.detail_label.grid(row=0, column=3, sticky="e", padx=16, pady=12)

        self.feedback_label = customtkinter.CTkLabel(
            self.footer,
            text="Feedback: --",
            wraplength=900,
            justify="left",
        )
        self.feedback_label.grid(row=1, column=0, columnspan=4, sticky="w", padx=16, pady=(0, 12))

    def _load_model(self) -> None:
        try:
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as exc:
            self.interpreter = None
            self.status_label.configure(text=f"Model load failed: {exc}")

    def _on_exercise_change(self) -> None:
        exercise = self.filter_var.get()
        is_bicep = exercise == "Bicep curl"
        counter_enabled = exercise in ("Squat", "Lunges")

        self.side_menu.configure(state="normal" if is_bicep else "disabled")
        self.counter_switch.configure(state="normal" if counter_enabled else "disabled")
        if not counter_enabled:
            self.counter_var.set(False)
        self.exercise_state = None
        self.keypoint_buffer = []
        self._reset_footer()
        if self.session_active:
            self.status_label.configure(text=f"Tracking: {exercise}")
        else:
            self.status_label.configure(text=f"Ready: {exercise}")

    def _reset_footer(self) -> None:
        self.rep_label.configure(text="Reps: 0")
        self.score_value_label.configure(text="--")
        self.side_label.configure(text=f"Side: {self.side_var.get()}")
        self.detail_label.configure(text="Details: --")
        self.feedback_label.configure(text="Feedback: --")

    def _bicep_feedback_sentence(self, last_rep):
        if not last_rep:
            return "Feedback: Complete a full curl with steady elbows."
        rom_score = last_rep.get("rom_score")
        elbow_score = last_rep.get("elbow_score")
        if rom_score is not None and rom_score < 0.7:
            return "Feedback: Curl higher and fully extend to improve your range of motion."
        if elbow_score is not None and elbow_score < 0.7:
            return "Feedback: Keep your elbow close to your side to reduce drift."
        return "Feedback: Great form, keep your movement slow and controlled."

    def _lower_body_feedback_sentence(self, label_prefix, knee_status, back_status, feedback):
        if feedback == "Insufficient visibility":
            return "Feedback: Keep your full body in view for better tracking."

        cues = []
        if label_prefix == "Squat":
            if knee_status != "Knees in position":
                cues.append("keep your knees tracking over your toes")
            if feedback == "Go deeper":
                cues.append("go a bit deeper while staying controlled")
            elif feedback in ("Good depth", "Excellent depth"):
                cues.append("maintain that depth")
            elif feedback and feedback.startswith("Depth:"):
                cues.append("aim for a deeper, controlled squat")
        else:
            if back_status == "Back too forward":
                cues.append("keep your torso more upright")
            if feedback == "Front knee: Go deeper":
                cues.append("drop your front knee a bit deeper")
            elif feedback in ("Front knee: Excellent", "Good form"):
                cues.append("keep the same depth and control")

        if not cues:
            cues.append("hold a steady tempo and posture")

        sentence = " and ".join(cues)
        return f"Feedback: {sentence}."

    def start_session(self) -> None:
        if self.session_active:
            return
        if self.interpreter is None:
            self._load_model()
        if self.interpreter is None:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text="Camera not available")
            self.cap.release()
            self.cap = None
            return

        self.session_active = True
        self.exercise_state = None
        self.keypoint_buffer = []
        self.last_frame_time = None
        self.status_label.configure(text=f"Tracking: {self.filter_var.get()}")
        self._update_frame()

    def stop_session(self) -> None:
        self.session_active = False
        if self.frame_after_id is not None:
            self.after_cancel(self.frame_after_id)
            self.frame_after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.image_display.configure(text="Camera stopped", image=None)
        self.image_display.image = None
        self.fps_label.configure(text="FPS: --")
        self.status_label.configure(text="Session stopped")

    def _on_close(self) -> None:
        self.stop_session()
        self.destroy()

    def _update_frame(self) -> None:
        if not self.session_active or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_session()
            return

        frame = cv2.flip(frame, 1)
        input_image = tf.image.resize_with_pad(
            np.expand_dims(frame, axis=0), INPUT_SIZE[0], INPUT_SIZE[1]
        )
        input_image = tf.cast(input_image, dtype=tf.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], np.array(input_image))
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]["index"])
        if SMOOTH_FRAMES > 1:
            keypoints_with_scores = smooth_keypoints(
                keypoints_with_scores, self.keypoint_buffer, SMOOTH_FRAMES
            )

        exercise = self.filter_var.get()
        edges_to_draw = UPPER_BODY_EDGES if exercise == "Bicep curl" else LOWER_BODY_EDGES
        draw_connections(frame, keypoints_with_scores, edges_to_draw, CONFIDENCE_THRESHOLD)
        draw_keypoints(frame, keypoints_with_scores, CONFIDENCE_THRESHOLD)

        self._update_exercise_state(exercise, keypoints_with_scores, frame.shape)
        self._update_image(frame)
        self._update_fps()

        self.frame_after_id = self.after(10, self._update_frame)

    def _update_exercise_state(self, exercise, keypoints, frame_shape) -> None:
        if exercise == "Bicep curl":
            side = self.side_var.get().lower()
            self.exercise_state = count_bicep_curls(keypoints, self.exercise_state, side=side)
            self._render_bicep_metrics()
            return
        if exercise == "Squat":
            self.exercise_state = update_squat_state(
                keypoints, self.exercise_state, frame_shape, self.counter_var.get()
            )
            self._render_lower_body_metrics("Squat")
            return
        if exercise == "Lunges":
            self.exercise_state = update_lunge_state(
                keypoints, self.exercise_state, frame_shape, self.counter_var.get()
            )
            self._render_lower_body_metrics("Lunge")

    def _render_bicep_metrics(self) -> None:
        if self.exercise_state is None:
            return
        metrics = self.exercise_state.get("metrics", {})
        score = metrics.get("last_score")
        last_rep = metrics.get("last_rep") or {}

        rom = last_rep.get("rom_score")
        elbow = last_rep.get("elbow_score")

        def format_component(value):
            if value is None:
                return "--"
            return f"{value * 100.0:.0f}"

        score_text = f"{score:.1f}" if score is not None else "--"
        self.rep_label.configure(text=f"Reps: {self.exercise_state.get('count', 0)}")
        self.score_value_label.configure(text=score_text)
        self.side_label.configure(text=f"Side: {self.side_var.get()}")
        self.detail_label.configure(
            text=f"ROM {format_component(rom)}  ELB {format_component(elbow)}"
        )
        self.feedback_label.configure(text=self._bicep_feedback_sentence(last_rep))

    def _render_lower_body_metrics(self, label_prefix) -> None:
        if self.exercise_state is None:
            return
        count = self.exercise_state.get("count", 0)
        total_score = self.exercise_state.get("score", 0)
        last_score = self.exercise_state.get("last_score", 0)
        knee_status = self.exercise_state.get("last_knee_status") or "N/A"
        back_status = self.exercise_state.get("last_back") or "N/A"
        feedback = self.exercise_state.get("last_feedback") or "Hold steady"

        activity = self.exercise_state.get("is_active", False)
        active_label = "SQUATTING" if label_prefix == "Squat" else "LUNGING"
        status_text = active_label if activity else "STANDING"
        self.status_label.configure(text=status_text)

        self.rep_label.configure(text=f"Reps: {count}")
        self.score_value_label.configure(text=f"{last_score:.1f}")
        self.side_label.configure(text=f"Total: {total_score}")
        detail_left = knee_status if label_prefix == "Squat" else back_status
        self.detail_label.configure(text=detail_left)
        self.feedback_label.configure(
            text=self._lower_body_feedback_sentence(
                label_prefix, knee_status, back_status, feedback
            )
        )

    def _update_image(self, frame) -> None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_width = self.image_display.winfo_width()
        label_height = self.image_display.winfo_height()

        if label_width > 1 and label_height > 1:
            height, width = frame_rgb.shape[:2]
            scale = min(label_width / width, label_height / height)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.image_display.configure(image=photo, text="")
        self.image_display.image = photo

    def _update_fps(self) -> None:
        now = time.time()
        if self.last_frame_time is not None:
            delta = now - self.last_frame_time
            if delta > 0:
                fps = 1.0 / delta
                self.fps_label.configure(text=f"FPS: {fps:.1f}")
        self.last_frame_time = now


if __name__ == "__main__":
    app = App()
    app.mainloop()
