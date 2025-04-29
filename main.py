import sys
import cv2
import os
import uuid
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QTextEdit, QHBoxLayout, QGridLayout, QListWidget, QScrollArea
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
        # Remove single violation_label and replace with scroll area for multiple images
        self.violation_scroll_area = QScrollArea()
        self.violation_scroll_area.setWidgetResizable(True)
        self.violation_container = QWidget()
        self.violation_grid = QGridLayout()
        self.violation_container.setLayout(self.violation_grid)
        self.violation_scroll_area.setWidget(self.violation_container)

        self.violation_log = QListWidget()

        self.load_model_btn = QPushButton("📦 Tải Mô Hình")
        self.load_video_btn = QPushButton("📂 Mở Video")
        self.open_cam_btn = QPushButton("🎥 Mở Camera")
        self.stop_btn = QPushButton("⛔ Dừng")
        # Remove QTextEdit for class_text and replace with scroll area for violation folder images
        # self.class_text = QTextEdit()
        # self.class_text.setReadOnly(True)
        self.violations_folder_scroll_area = QScrollArea()
        self.violations_folder_scroll_area.setWidgetResizable(True)
        self.violations_folder_container = QWidget()
        self.violations_folder_grid = QGridLayout()
        self.violations_folder_container.setLayout(self.violations_folder_grid)
        self.violations_folder_scroll_area.setWidget(self.violations_folder_container)

        # Layout
        layout = QGridLayout()
        layout.addWidget(self.video_label, 0, 0)
        # Add scroll area for violation images instead of single label
        layout.addWidget(self.violation_scroll_area, 0, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.load_video_btn)
        btn_layout.addWidget(self.open_cam_btn)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout, 1, 0, 1, 2)
        layout.addWidget(QLabel("📚 Ảnh vi phạm trong thư mục violations:"), 2, 0)
        layout.addWidget(self.violations_folder_scroll_area, 3, 0)
        layout.addWidget(QLabel("📋 Nhật ký vi phạm:"), 2, 1)
        layout.addWidget(self.violation_log, 3, 1)

        self.setLayout(layout)

        self.load_model_btn.clicked.connect(self.load_model)
        self.load_video_btn.clicked.connect(self.load_video)
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.stop_btn.clicked.connect(self.stop_video)

        os.makedirs("violations", exist_ok=True)

        # Load violation folder images on startup
        self.load_violations_folder_images()

    def load_violations_folder_images(self):
        # Clear existing images
        for i in reversed(range(self.violations_folder_grid.count())):
            widget_to_remove = self.violations_folder_grid.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        col_count = 3
        row = 0
        col = 0

        # List all image files in violations folder
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        files = [f for f in os.listdir("violations") if f.lower().endswith(image_extensions)]
        files.sort()

        for filename in files:
            filepath = os.path.join("violations", filename)
            pixmap = QPixmap(filepath).scaled(160, 120)
            label = QLabel()
            label.setPixmap(pixmap)
            self.violations_folder_grid.addWidget(label, row, col)
            col += 1
            if col >= col_count:
                col = 0
                row += 1

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Chọn mô hình YOLOv8 (.pt)", "", "Model Files (*.pt)")
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.class_names = self.model.names
                # No longer set class_text since replaced by images
                # self.class_text.setText("\n".join([f"{i}: {name}" for i, name in self.class_names.items()]))
            except Exception as e:
                # No longer set class_text
                # self.class_text.setText(f"Lỗi khi load model:\n{str(e)}")
                pass

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

                # Clear previous violation images from grid
                for i in reversed(range(self.violation_grid.count())):
                    widget_to_remove = self.violation_grid.itemAt(i).widget()
                    if widget_to_remove is not None:
                        widget_to_remove.setParent(None)

                col_count = 3  # Number of columns in grid
                row = 0
                col = 0

                for x1, y1, x2, y2, missing in violations:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Thiếu: {', '.join(missing)}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    person_crop = frame[y1:y2, x1:x2]
                    file_id = str(uuid.uuid4())[:8]
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"violations/{timestamp}_{file_id}.jpg"
                    cv2.imwrite(filename, person_crop)

                    # Create QLabel for each violation image and add to grid
                    violation_pixmap = QPixmap(filename).scaled(160, 120)
                    violation_img_label = QLabel()
                    violation_img_label.setPixmap(violation_pixmap)
                    self.violation_grid.addWidget(violation_img_label, row, col)
                    col += 1
                    if col >= col_count:
                        col = 0
                        row += 1

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
