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

    # menginisialisasi pose landmarker yang telah didownload 
    """
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,)
    """