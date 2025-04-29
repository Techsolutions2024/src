import sys
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QTextEdit, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer


class YoloTestApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 Kiểm tra YOLOv8 với Video / Camera")
        self.model = None
        self.class_names = []
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # GUI Elements
        self.video_label = QLabel("Chưa có hình ảnh")
        self.load_model_btn = QPushButton("📦 Tải Mô Hình")
        self.load_video_btn = QPushButton("📂 Mở Video")
        self.open_cam_btn = QPushButton("🎥 Mở Camera")
        self.stop_btn = QPushButton("⛔ Dừng")
        self.class_text = QTextEdit()
        self.class_text.setReadOnly(True)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.load_model_btn)
        hlayout.addWidget(self.load_video_btn)
        hlayout.addWidget(self.open_cam_btn)
        hlayout.addWidget(self.stop_btn)
        layout.addLayout(hlayout)

        layout.addWidget(QLabel("📚 Các lớp trong mô hình:"))
        layout.addWidget(self.class_text)
        self.setLayout(layout)

        # Sự kiện
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_video_btn.clicked.connect(self.load_video)
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.stop_btn.clicked.connect(self.stop_video)

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

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.model:
                    results = self.model.predict(source=frame, conf=0.25, verbose=False)
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame

                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.video_label.setPixmap(pixmap)
            else:
                self.stop_video()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloTestApp()
    window.resize(960, 720)
    window.show()
    sys.exit(app.exec_())
