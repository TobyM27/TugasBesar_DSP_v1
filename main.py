import sys
import numpy as np
import cv2
import os
import subprocess
import requests
from tqdm import tqdm
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
import platform
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

def download_model_pose_detection():
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")
    
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} already exists and is valid, skipping download.")
        return filename
    
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
        
        if os.path.getsize(filename) == 0:
            raise ValueError("Downloaded file is empty")
            
        print("Download completed successfully!")
        return filename
        
    except Exception as e:
        print(f"Error downloading the model: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        raise

## Problem specific for Mac
def check_gpu():
    """
    Memeriksa ketersediaan GPU pada sistem.
    Returns:
        str: "NVIDIA" untuk GPU NVIDIA, "MLX" untuk Apple Silicon, atau "CPU" jika tidak ada GPU
    """
    system = platform.system()
    print(f"System: {system}")
    # Check for NVIDIA GPU
    if system == "Linux" or system == "Windows":
        try:
            nvidia_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return "NVIDIA"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "CPU"

    # Check for Apple MLX
    elif system == "Darwin":  # macOS
        try:
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU: {cpu_info}")
            if "Apple" in cpu_info:  # This indicates Apple Silicon (M1/M2/M3)
                return "MLX"
        except subprocess.CalledProcessError:
            pass
    return "CPU"

## The length to be around should width to ensure the landmarks are detected and can proceed to optical flow 
def get_initial_roi(image, landmarker, x_size=100, y_size=30, shift_x=0, shift_y=-30):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    
    detection_result = landmarker.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in first frame!")
        
    landmarks = detection_result.pose_landmarks[0]
    
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
    
    center_x += shift_x
    center_y += shift_y
    
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)
    
    if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
        raise ValueError("Invalid ROI dimensions")
        
    return (left_x, top_y, right_x, bottom_y)

class HeartRateMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        video_backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_AVFOUNDATION # Menggunakan CAP_DSHOW untuk Windows dan CAP_AVFOUNDATION untuk macOS
        self.cap = cv2.VideoCapture(0, video_backend)  
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Set a default FPS value if the camera does not provide it
        self.r_signal, self.g_signal, self.b_signal = [], [], []
        self.resp_signal = []
        self.face_detector = self.initialize_face_detector()
        self.pose_landmarker = self.initialize_pose_landmarker()
        self.features = None  # Initialize features attribute
        self.left_x = None
        self.top_y = None
        self.right_x = None
        self.bottom_y = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // int(self.fps))

    def initUI(self):
        self.setWindowTitle('Real-Time Heart Rate and Respiration Monitor')
        self.setGeometry(100, 100, 1200, 800)

        self.video_label = QLabel(self)
        self.hr_label = QLabel('Heart Rate: -- BPM (Beat Per Minute)', self)
        self.hr_label.setAlignment(Qt.AlignCenter)
        self.resp_label = QLabel('Respiration Rate: -- BPM (Breath Per Minute)', self)
        self.resp_label.setAlignment(Qt.AlignCenter)

        self.plot_widget_hr = pg.PlotWidget()
        self.plot_widget_hr.setYRange(-3, 3)
        self.plot_curve_hr = self.plot_widget_hr.plot()

        self.plot_widget_resp = pg.PlotWidget()
        self.plot_widget_resp.setYRange(-3, 3)
        self.plot_curve_resp = self.plot_widget_resp.plot()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.hr_label)
        left_layout.addWidget(self.plot_widget_hr)
        left_layout.addWidget(self.resp_label)
        left_layout.addWidget(self.plot_widget_resp)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.video_label)

        main_layout = QGridLayout()
        main_layout.addLayout(left_layout, 0, 0)
        main_layout.addLayout(right_layout, 0, 1)

        self.setLayout(main_layout)

    def initialize_face_detector(self):
        model_path = download_model_face_detection()
        BaseOptions = mp.tasks.BaseOptions
        gpu_checked = check_gpu()
        if gpu_checked == "NVIDIA":
            delegate = python.BaseOptions.Delegate.GPU
        else:
            delegate = python.BaseOptions.Delegate.CPU
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=delegate
            )
        )
        return vision.FaceDetector.create_from_options(options)

    def initialize_pose_landmarker(self):
        model_path = download_model_pose_detection()
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        gpu_checked = check_gpu()

        if gpu_checked == "NVIDIA":
            delegate = BaseOptions.Delegate.GPU
        else:
            delegate = BaseOptions.Delegate.CPU
        
        options_image = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate = delegate
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        
        return mp.tasks.vision.PoseLandmarker.create_from_options(options_image)

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
                h, w, _ = frame.shape
                x, y = int(bbox.origin_x), int(bbox.origin_y)
                width, height = int(bbox.width), int(bbox.height)

                forehead_x = x + width // 4
                forehead_y = y // 2 + 55
                forehead_width = width // 2
                forehead_height = height // 5

                roi_forehead = frame[forehead_y:forehead_y + forehead_height, forehead_x:forehead_x + forehead_width]
                cv2.rectangle(frame, (forehead_x, forehead_y), (forehead_x + forehead_width, forehead_y + forehead_height), (0, 255, 0), 2)

                self.r_signal.append(np.mean(roi_forehead[:, :, 0]))
                self.g_signal.append(np.mean(roi_forehead[:, :, 1]))
                self.b_signal.append(np.mean(roi_forehead[:, :, 2]))

        if self.features is None:
            # Initialize ROI and feature detection
            self.initialize_features(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.features) > 10:
            new_features, status, error = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.features, None, **self.lk_params)
            good_old = self.features[status == 1]
            good_new = new_features[status == 1]
            mask = np.zeros_like(frame)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
            frame = cv2.add(frame, mask)
            if len(good_new) > 0:
                avg_y = np.mean(good_new[:, 1])
                self.resp_signal.append(avg_y)
                self.features = good_new.reshape(-1, 1, 2)
            self.old_gray = frame_gray.copy()
        else:
            # Reinitialize features if tracking fails
            self.initialize_features(frame)

        # Draw ROI rectangle for chest area
        cv2.rectangle(frame, (self.left_x, self.top_y), (self.right_x, self.bottom_y), (0, 0, 255), 2)

        if len(self.r_signal) >= self.fps * 5:  # 10 seconds
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

            self.hr_label.setText(f'Heart Rate: {heart_rate:.2f} BPM (Beat Per Minute)')
            self.plot_curve_hr.setData(smoothed_signal)
            self.r_signal, self.g_signal, self.b_signal = [], [], []

        if len(self.resp_signal) >= self.fps * 5:  # 10 seconds for respiration rate
            resp_signal = np.array(self.resp_signal)
            filtered_resp_signal = self.bandpass_filter(resp_signal, 0.1, 0.5, self.fps, order=5)
            normalized_resp_signal = (filtered_resp_signal - np.mean(filtered_resp_signal)) / np.std(filtered_resp_signal)
            smoothed_resp_signal = self.moving_average(normalized_resp_signal, window_size=int(self.fps/2))

            resp_peaks, _ = find_peaks(smoothed_resp_signal, distance=self.fps)
            resp_intervals = np.diff(resp_peaks) / self.fps
            respiration_rate = 60.0 / np.mean(resp_intervals) if len(resp_intervals) > 0 else 0

            self.resp_label.setText(f'Respiration Rate: {respiration_rate:.2f} BPM (Breath Per Minute)')
            self.plot_curve_resp.setData(smoothed_resp_signal)
            self.resp_signal = []

        # Convert frame to RGB for displaying in video_label
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def initialize_features(self, frame):
        roi_coords = get_initial_roi(frame, self.pose_landmarker)
        self.left_x, self.top_y, self.right_x, self.bottom_y = roi_coords
        old_frame = frame.copy()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        roi_chest = old_gray[self.top_y:self.bottom_y, self.left_x:self.right_x]
        self.features = cv2.goodFeaturesToTrack(roi_chest, maxCorners=50, qualityLevel=0.2, minDistance=5, blockSize=3)
        if self.features is None:
            raise ValueError("No features found to track!")
        self.features = np.float32(self.features)
        self.features[:,:,0] += self.left_x
        self.features[:,:,1] += self.top_y
        self.old_gray = old_gray
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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