import sys
import numpy as np
import cv2
import os
import requests
import tqdm
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, find_peaks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.heart_rate import cpu_POS

def download_model_face_detection():
    """
    Mengunduh model pose landmarker dari MediaPipe.
    Model ini digunakan untuk mendeteksi pose tubuh dalam video.
    """
    # Create models directory if it doesn't exist
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    filename = os.path.join(model_dir, "face_detector.task")
    
    # Check if file already exists and is not empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} already exists and is valid, skipping download.")
        return filename
    
    # Download with progress bar
    try:
        print(f"Downloading model to {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filename, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        
        # Verify downloaded file
        if os.path.getsize(filename) == 0:
            raise ValueError("Downloaded file is empty")
            
        print("Download completed successfully!")
        return filename
        
    except Exception as e:
        print(f"Error downloading the model: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Clean up partial download
        raise


class HeartRateMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.r_signal, self.g_signal, self.b_signal = [], [], []
        self.face_detector = self.initialize_face_detector()
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

    def initialize_face_detector(self):
        model_path = download_model_face_detection()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        return vision.FaceDetector.create_from_options(options)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.face_detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                h, w, _ = frame_rgb.shape
                x, y = int(bbox.origin_x), int(bbox.origin_y)
                width, height = int(bbox.width), int(bbox.height)

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