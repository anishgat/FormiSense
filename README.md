# FormiSense

FormiSense is a real-time exercise form tracker that uses MoveNet pose estimation to
analyze webcam video and give feedback on bicep curls, squats, and lunges. The app
renders keypoints, tracks reps, and scores movement quality so users can improve
form during a live session.

## Key Features
- Real-time pose tracking with MoveNet Lightning.
- Exercise modes for bicep curls, squats, and lunges.
- Rep counting with optional scoring for lower-body exercises.
- Form feedback for range of motion, knee tracking, and posture.
- Smoothing across frames to reduce jitter.
- CustomTkinter desktop UI with live preview and metrics.

## Tech Stack
- Python 3
- TensorFlow Lite (MoveNet Lightning model)
- OpenCV (camera capture and rendering)
- NumPy (pose math and smoothing)
- CustomTkinter + Tkinter (desktop UI)
- Pillow (frame conversion for UI)
