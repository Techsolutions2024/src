import sys
import cv2
import os
import uuid
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QTextEdit, QHBoxLayout, QGridLayout, QListWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer


class YoloSafetyMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦺 Giám sát An toàn Lao động với YOLOv8")
        self.model = None
        self.class_names = []
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.video_label = QLabel("🔴 Camera Feed")
        self.violation_label = QLabel("❌ Ảnh vi phạm")
        self.violation_log = QListWidget()

        self.load_model_btn = QPushButton("📦 Tải Mô Hình")
        self.load_video_btn = QPushButton("📂 Mở Video")
        self.open_cam_btn = QPushButton("🎥 Mở Camera")
        self.stop_btn = QPushButton("⛔ Dừng")
        self.class_text = QTextEdit()
        self.class_text.setReadOnly(True)

        # Layout
        layout = QGridLayout()
        layout.addWidget(self.video_label, 0, 0)
        layout.addWidget(self.violation_label, 0, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.load_video_btn)
        btn_layout.addWidget(self.open_cam_btn)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout, 1, 0, 1, 2)
        layout.addWidget(QLabel("📚 Các lớp trong mô hình:"), 2, 0)
        layout.addWidget(self.class_text, 3, 0)
        layout.addWidget(QLabel("📋 Nhật ký vi phạm:"), 2, 1)
        layout.addWidget(self.violation_log, 3, 1)

        self.setLayout(layout)

        self.load_model_btn.clicked.connect(self.load_model)
        self.load_video_btn.clicked.connect(self.load_video)
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.stop_btn.clicked.connect(self.stop_video)

        os.makedirs("violations", exist_ok=True)

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Chọn mô hình YOLOv8 (.pt)", "", "Model Files (*.pt)")
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.class_names = self.model.names
                self.class_text.setText("\n".join([f"{i}: {name}" for i, name in self.class_names.items()]))
            except Exception as e:
                self.class_text.setText(f"Lỗi khi load model:\n{str(e)}")

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.start(30)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def check_violation(self, result):
        persons = []
        violations = []

        for box in result.boxes:
            if int(box.cls[0]) == 0:  # person
                persons.append(box)

        for person in persons:
            x1, y1, x2, y2 = map(int, person.xyxy[0])
            sub_items = {"helmet": False, "gloves": False, "safety-vest": False, "glasses": False}

            for box in result.boxes:
                if box is person:
                    continue
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                cx, cy = map(int, box.xywh[0][:2])
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    if label == "helmet": sub_items["helmet"] = True
                    elif label == "gloves": sub_items["gloves"] = True
                    elif label in ["safety-vest", "safety-suit"]: sub_items["safety-vest"] = True
                    elif label == "glasses": sub_items["glasses"] = True

            missing = [k for k, v in sub_items.items() if not v]
            if missing:
                violations.append((x1, y1, x2, y2, missing))

        return violations

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return

            if self.model:
                results = self.model.predict(source=frame, conf=0.25, verbose=False)
                result = results[0]
                violations = self.check_violation(result)
                annotated_frame = result.plot()

                for x1, y1, x2, y2, missing in violations:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Thiếu: {', '.join(missing)}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    person_crop = frame[y1:y2, x1:x2]
                    file_id = str(uuid.uuid4())[:8]
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"violations/{timestamp}_{file_id}.jpg"
                    cv2.imwrite(filename, person_crop)

                    self.violation_label.setPixmap(QPixmap(filename).scaled(320, 240))
                    self.violation_log.addItem(f"[{timestamp}] {label}")

                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.video_label.setPixmap(pixmap)
            else:
                self.video_label.setText("⚠️ Chưa có mô hình được tải.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloSafetyMonitor()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())