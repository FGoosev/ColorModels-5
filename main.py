import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QComboBox, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from skimage.feature import ORB, corner_harris, corner_peaks, hog
from skimage.color import rgb2gray
from skimage.transform import integral_image
from skimage.feature import (plot_matches, match_descriptors, BRIEF, corner_harris, corner_peaks)
from skimage.feature import hog
from skimage.feature import (ORB, match_descriptors, plot_matches)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Лабораторная_5')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(800, 400)

        self.open_image_button = QPushButton('Загрузить изображение', self)
        self.open_image_button.clicked.connect(self.open_image)

        self.open_video_button = QPushButton('Загрузить видео', self)
        self.open_video_button.clicked.connect(self.open_video)

        self.effect_combo = QComboBox(self)
        self.effect_combo.addItem("Харрис")
        self.effect_combo.addItem("SIFT")
        self.effect_combo.addItem("SURF")
        self.effect_combo.addItem("FAST")
        self.effect_combo.addItem("Удалить фон")
        self.effect_combo.addItem("Размытие объектов")
        self.effect_combo.currentIndexChanged.connect(self.apply_effect)
        self.effect_combo.setEnabled(False)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.apply_effect)
        self.slider.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.open_image_button)
        layout.addWidget(self.open_video_button)
        layout.addWidget(self.effect_combo)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image = None
        self.original_image = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def open_image(self):
        self.timer.stop()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.original_image = self.image.copy()
            self.display_image(self.image)
            self.slider.setEnabled(False)
            self.effect_combo.setEnabled(True)

    def open_video(self):
        self.timer.stop()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if file_name:
            self.video_capture = cv2.VideoCapture(file_name)
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.slider.setEnabled(True)
            self.effect_combo.setEnabled(True)
            self.timer.start(30)  # Set timer to call update_frame every 30ms

    def update_frame(self):
        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.original_image = frame.copy()
                self.apply_effect()
            else:
                self.timer.stop()

    def display_image(self, img):
        height, width = img.shape[:2]
        max_height = 400
        max_width = 800
        scale_factor = min(max_width / width, max_height / height)

        new_size = (int(width * scale_factor), int(height * scale_factor))
        resized_image = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(resized_image, resized_image.shape[1], resized_image.shape[0], resized_image.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(out_image))
        self.image_label.setScaledContents(True)

    def apply_effect(self):
        if self.original_image is not None:
            effect = self.effect_combo.currentText()
            if effect == "Харрис":
                self.apply_harris()
            elif effect == "SIFT":
                self.apply_sift()
            elif effect == "SURF":
                self.apply_surf()
            elif effect == "FAST":
                self.apply_fast()
            elif effect == "Удалить фон":
                self.apply_bg_subtraction()
            elif effect == "Размытие объектов":
                self.apply_blur_moving_objects()

    def apply_harris(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        self.image = self.original_image.copy()
        self.image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
        self.display_image(self.image)

    def apply_sift(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(gray, None)
        self.image = cv2.drawKeypoints(self.original_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.display_image(self.image)

    def apply_surf(self):
        gray = rgb2gray(self.original_image)
        integral_img = integral_image(gray)
        surf = ORB(n_keypoints=500)
        surf.detect_and_extract(integral_img)
        keypoints = surf.keypoints
        self.image = self.original_image.copy()
        for kp in keypoints:
            y, x = kp
            cv2.circle(self.image, (int(x), int(y)), 3, (0, 255, 0), -1)
        self.display_image(self.image)

    def apply_fast(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
        self.image = cv2.drawKeypoints(self.original_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.display_image(self.image)

    def apply_bg_subtraction(self):
        fg_mask = self.bg_subtractor.apply(self.original_image)
        bg_removed = cv2.bitwise_and(self.original_image, self.original_image, mask=fg_mask)
        self.display_image(bg_removed)

    def apply_blur_moving_objects(self):
        fg_mask = self.bg_subtractor.apply(self.original_image)
        blurred_frame = cv2.GaussianBlur(self.original_image, (21, 21), 0)
        mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]
        blurred_objects = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
        static_background = cv2.bitwise_and(self.original_image, self.original_image, mask=cv2.bitwise_not(mask))
        final_frame = cv2.add(static_background, blurred_objects)
        self.display_image(final_frame)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
