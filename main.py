import sys
import numpy as np
import cv2
import time
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, find_peaks
from utils.heart_rate import cpu_POS

class HeartRateMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.r_signal, self.g_signal, self.b_signal = [], [], []
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // int(self.fps))

    def initUI(self):
        self.setWindowTitle('Real-Time Heart Rate Monitor')
        self.setGeometry(100, 100, 1200, 600)

        self.video_label = QLabel(self)
        self.hr_label = QLabel('Heart Rate: -- BPM', self)
        self.hr_label.setAlignment(Qt.AlignCenter)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-3, 3)
        self.plot_curve = self.plot_widget.plot()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.hr_label)
        left_layout.addWidget(self.plot_widget)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.video_label)

        main_layout = QGridLayout()
        main_layout.addLayout(left_layout, 0, 0)
        main_layout.addLayout(right_layout, 0, 1)

        self.setLayout(main_layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame_rgb.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)

                forehead_x = x + width // 4
                forehead_y = y // 2 + 55
                forehead_width = width // 2
                forehead_height = height // 5

                roi = frame_rgb[forehead_y:forehead_y + forehead_height, forehead_x:forehead_x + forehead_width]
                cv2.rectangle(frame_rgb, (forehead_x, forehead_y), (forehead_x + forehead_width, forehead_y + forehead_height), (0, 255, 0), 2)

                self.r_signal.append(np.mean(roi[:, :, 0]))
                self.g_signal.append(np.mean(roi[:, :, 1]))
                self.b_signal.append(np.mean(roi[:, :, 2]))

        if len(self.r_signal) >= self.fps * 5:
            rgb_signals = np.array([self.r_signal, self.g_signal, self.b_signal])
            rgb_signals = rgb_signals.reshape(1, 3, -1)
            rppg_signal = cpu_POS(rgb_signals, fps=self.fps)
            rppg_signal = rppg_signal.reshape(-1)

            filtered_signal = self.bandpass_filter(rppg_signal, 0.75, 3.0, self.fps, order=5)
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
            smoothed_signal = self.moving_average(normalized_signal, window_size=int(self.fps/2))

            peaks, _ = find_peaks(smoothed_signal, distance=self.fps/2)
            peak_intervals = np.diff(peaks) / self.fps
            heart_rate = 60.0 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0

            self.hr_label.setText(f'Heart Rate: {heart_rate:.2f} BPM')
            self.plot_curve.setData(smoothed_signal)
            self.r_signal, self.g_signal, self.b_signal = [], [], []

        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def bandpass_filter(self, signal, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y

    def moving_average(self, signal, window_size):
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HeartRateMonitor()
    ex.show()
    sys.exit(app.exec_())