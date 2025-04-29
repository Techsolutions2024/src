# YoloSafetyMonitor

This project is a PyQt5 application for monitoring workplace safety using the YOLOv11 object detection model. It detects safety equipment such as helmets, gloves, safety vests, and glasses on persons in video feeds or camera streams, and logs violations when safety equipment is missing.

## Features

- Load a YOLOv11 model for safety equipment detection.
- Open video files or live camera feed for monitoring.
- Detect persons and check if they are wearing required safety equipment.
- Highlight violations with bounding boxes and labels.
- Save images of violations with timestamps.
- Display a log of violations in the GUI.

## Requirements

- Python 3.x
- OpenCV
- PyQt5
- Ultralytics package

## How to Run

1. Install the required packages:
   ```
   pip install opencv-python PyQt5 ultralytics
   ```

2. Run the application:
   ```
   python main.py
   ```

3. Use the buttons in the GUI to load a YOLOv8 model, open a video file, or start the camera feed.

## Notes

- The application saves violation images in the `violations` folder.
- The model should be a YOLOv8 `.pt` file trained to detect safety equipment classes.
