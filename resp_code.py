# Mengimport library-library yang akan digunakan untuk mendeteksi sinyal respirasi pada tugas besar DSP ini. 
import os
import requests
from tqdm import tqdm
import platform
import subprocess
import mediapipe as mp
from glob import glob
import re
from datetime import timedelta 
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# menginstall model pose landmarker dari MediaPipe. 
def download_model():
    """
    Menambahkan comment pada bagian kode ini 
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")

    #memeriksa apakah file sudah ada dan tidak kosong
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} sudah ada. Melewati proses download.")
        return filename
    
    # menampilkan proses download dengan tqdm
    try:
        print(f"Mengunduh model ke direktori {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(filename, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        # memverifikasi file download
        if os.path.getsize(filename) == 0:
            raise ValueError("Model file kosong.")
        print("Proses download telah selesai. Model siap digunakan.")
        return filename
    except Exception as e:
        print(f"Proses download gagal: {e}")
        if os.path.exists(filename):
            os.remove(filename) 

def check_gpu():
    """
    menambahkan comment 
    """
    system = platform.system()
    print(f"System yang sedang digunakan : {system}")
    # Memeriksa apabila ada GPU yang tersedia pada laptop pengguna
    if system == "Linux" or system == "Windows":
        try:
            nvidia_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return "NVIDIA"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "CPU"
    # Memeriksa apabila laptop pengguna sedang menggunakan Apple dengan arsitektur MLX
    elif system == "Darwin":
        try:
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU yang digunakan: {cpu_info}")
            if "Apple" in cpu_info:
                return "MLX"
        except subprocess.CalledProcessError:
            pass
    return "CPU"

def initialize_pose_landmarker():
    """
    Menginialisasi pose landmarker yang sudah didownload pada folder 'models'
    """
    model_path = download_model() 

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    gpu_check = check_gpu()

    # program akan berjalan pada GPU jika tersedia, jika tidak maka akan berjalan pada CPU dari pengguna
    if gpu_check == "NVIDIA":
        delegate = BaseOptions.Delegate.GPU
    else:
        delegate = BaseOptions.Delegate.CPU

    # Membuat landmarker untuk frame awal
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        ),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return PoseLandmarker.create_from_options(options)

def roi_enhancement(roi):
    """ 
    Menambahkan comment pada bagian kode ini
    """
    if roi is None or roi.size == 0:
        raise ValueError("ROI tidak diberikan")
    
    # Mengubah ROI menjadi grayscale
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Mengapllikasikan fungsi CLAHE (Contrast Limited Adaptive Histogram Equalization) pada ROI
        clahe = cv2.createCLAHE(chiplimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Mengaplikasikan Gaussian Blur pada ROI untuk memperkaya fitur tepi-tepi pada gambar pasien
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return enhanced
    except cv2.error as e:
        raise ValueError(f"Error dalam memproses ROI: {str(e)}")
    
def get_initial_roi(image, pose_landmarker, x_size=100, y_size=150, shift_x=0, shift_y=0):
    """
    Mengambil ROI dari webcam untuk mendeteksi sinyal respirasi berdasarkan pergerakan posisi bahu pasien

    Args:
        image (np.ndarray): Frame dari webcam
        pose_landmarker : MediaPipe pose detector yang sudah didownload

    Returns:
        tuple: Koordinat ROI (left_x, top_y, right_x, bottom_y)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Membuat gambar MediaPip dari frame pertama
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Mendeteksi pose dari frame pertama
    detection_result = pose_landmarker.detect(mp_image)

    if not detection_result.pose_landmarks:
        raise ValueError("Tidak ada pose yang terdeteksi pada frame!")
    
    landmarks = detection_result.pose_landmarks[0]
    height, width = image.shape[:2]

    # Mengambil posisi dari bahu kiri dan bahu kanan
    left_shoulder = landmarks.landmarks[11] 
    right_shoulder = landmarks.landmarks[12]

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
        raise ValueError("Ukuran ROI tidak valid")
    return (left_x, top_y, right_x, bottom_y)

def process_respiration_webcam():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True : 
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()