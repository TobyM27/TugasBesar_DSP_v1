# Mengimport library yang diperlukan untuk tugas besar ini
import sys
import numpy as np
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, find_peaks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.heart_rate import cpu_POS
from utils.download_model import download_model_face_detection, download_model_pose_detection
from utils.check_gpu import check_gpu

class HeartRateMonitor(QWidget):
    """
    Main Kelas untuk menampilkan GUI dan menghitung detak jantung dan pernapasan secara real-time.
    Kelas ini akan menyimpan nilai sinyal rppg dan nilai sinyal pernafasan dari pose detection
    """
    def __init__(self):
        super().__init__()
        self.initUI()

        ## Platform Specific Camera Backend
        video_backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_AVFOUNDATION # Menggunakan CAP_DSHOW untuk Windows dan CAP_AVFOUNDATION untuk macOS
        self.cap = cv2.VideoCapture(0, video_backend)  
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  

        ## Properties for Storeing value
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
        """
        Metode untuk mempersiapkan kelas PyQT sebagai GUI untuk overlay dari feed live time video dan sinyal (rppg + resp)
        Karena kelas merupakan anak dari QWidget, untuk insialisasi tidak perlu menggunakan keyword QWidget.
        """
        self.setWindowTitle('Real-Time Heart Rate and Respiration Monitor')
        self.setGeometry(100, 100, 1200, 800)

        ## Prepare Plot for Combining with Layout 
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

        ## Making Layout instance and insert the plot into the layout
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
        """
        Metode untuk menginisialisasi fungsi face detection dari mediapipe untuk proses RPPG
        """
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
        """
        Metode untuk menginisialisasi fungsi pose detection dari mediapipe untuk proses resp signal
        """
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
        """
        Metode penting dalam kelas ini, akan diambi frame orng lalu ditentukan beberapa proses seperti 
        deteksi wajah dengan mediapipe dan pose detection, lalu menentukan bounding box untuk proses selanjutnya.
        Untuk RPPG ditetapkan daerah dahi sebagai ROI, dan untuk Resp signal ditetapkan daerah sekitar baru sebagai ROI.
        Untuk Pose Detection akan diterapkan Optical Flow untuk mengurangi kinerja beban tracking setiap frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.face_detector.detect(mp_image)

        ## Mediapipe Face Detection
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


        ## Mediapipe Pose Detection with Optical Flow
        if self.features is None:
            # Initialize ROI and feature detection
            self.initialize_features(frame)

        """
        Proses paling kompleks, dimana akan dilakukan pengecekan terhadap features optical flow (10 buah) dan jika tidak ada features 
        maka akan ditrack ulang oleh mediapipe. Meskipun dengan Gray Image tetapi model ini sangat berat untuk pose detection sehinnga cukup sulit 
        untuk optimasi pada momen momen awal ketika belum ditentukan optical flownya. Sehingga diterapkannya Tracking Failure in case tidak ada features.
        """
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

        
        """
        Proses untuk kalkulasi nilai RPPG dan Resp lalu di set sebagai sinyal oleh PyQt Graph.
        dan juga akan ditampikan video live feed dan ditempel ke PyQT
        """
        if len(self.r_signal) >= self.fps * 10:  # 10 seconds
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

        if len(self.resp_signal) >= self.fps * 10:  # 10 seconds for respiration rate
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
        """
        Metode untuk mendapatkan nilai features untuk keperluan optical flow dan membuat object Lucas Kanade sebagai argument dari optical flow itu sendiri.
        frame: cv2Object = frame sumber dari kamera untuk melakukan deteksi ROI dada dan bahu
        """

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
        """
        Metode untuk melakukan bandpass filters
        signal: parray  = sinyal target filtering
        lowcut: float = batas bawah frekuensi
        highcut : float = batas atas frekuensi
        fs : int = nilai sampling rate
        order : int = tingkat pengurangan amplitudo / frekuensi 
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y

    def moving_average(self, signal, window_size):
        """
        I'm forgor about this
        """
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

    def closeEvent(self, event):
        """
        Metode turunan dari kelas QWidget untuk melepaskan resource yang tidak dibutuhkan lagi.
        """
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

## The length to be around should width to ensure the landmarks are detected and can proceed to optical flow 
def get_initial_roi(image, landmarker, x_size=100, y_size=30, shift_x=0, shift_y=-30):

    """
    Mengambil ROI dari webcam untuk mendeteksi sinyal respirasi berdasarkan pergerakan posisi bahu pasien.

    Args:
        image (np.ndarray): Frame dari webcam
        pose_landmarker (task): MediaPipe pose detector yang sudah didownload
        x_size (int): Ukuran ROI pada sumbu x
        y_size (int): Ukuran ROI pada sumbu y
        shift_x (int): Pergeseran ROI pada sumbu x (dimana nilai positif akan menggeser ROI ke kanan dan sebaliknya)
        shift_y (int): Pergeseran ROI pada sumbu y (dimana nilai positif akan menggeser ROI ke bawah dan sebaliknya)

    Returns:
        tuple: Koordinat ROI (left_x, top_y, right_x, bottom_y)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Mengubah warna BGR ke RGB
    height, width = image.shape[:2] # Mengambil dimensi frame webcam
    
    # Membuat gambar MediaPipe dari frame webcam
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    
    # Mendeteksi pose dari frame webcam
    detection_result = landmarker.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in first frame!")
    
    # Mendeteksi tubuh pengguna dari landmark pertama
    landmarks = detection_result.pose_landmarks[0]
    
    # Mengambil landmark bahu kiri dan kanan
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    # Menghitung posisi tengah dari bahu kiri dan bahu kanan
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
    
    # Mengaplikasikan shift terhadap titik tengah
    center_x += shift_x
    center_y += shift_y
    
    # Manghitung batasan ROI berdasarkan posisi tengah dan ukuran ROI
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)
    
    # Mevalidasi ukuran ROI
    if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
        raise ValueError("Invalid ROI dimensions")
        
    return (left_x, top_y, right_x, bottom_y)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HeartRateMonitor()
    ex.show()
    sys.exit(app.exec_())