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

#memanggil fungsi download_model - menguji fungsi download_model
download_model()