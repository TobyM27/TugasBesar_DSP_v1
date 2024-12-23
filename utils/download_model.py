import os
import requests
from tqdm import tqdm

def download_model_face_detection():
    """
    Mengunduh model face_detector dari MediaPipe.
    Model ini digunakan untuk mendeteksi muka pengguna dalam video.
    """
    # Membuat direktori model jika belum ada 
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    filename = os.path.join(model_dir, "face_detector.task")
    
    # Memeriksa apabila file sudah terbuat dan tidak kosong
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"File model {filename} sudah terunduh. Melewati proses pengunduhan...")
        return filename
    
    # Proses pengunduhan dengan progress bar (yang menggunakan library tqdm)
    try:
        print(f"Mengunduh model face detector ke direktori: {filename}...")
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
            raise ValueError("File yang terunduh kosong")
            
        print("Proses pengunduhan model face detector berhasil!")  
        return filename
    
    except Exception as e:
        print(f"Terdapat error dalam proses pengunduhan model face detector: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Clean up partial download
        raise

def download_model_pose_detection():
    """
    Mengunduh model pose landmarker dari MediaPipe.
    Model ini digunakan untuk mendeteksi pose tubuh pengguna dalam video (khususnya dalam bagian bahu / torso atas).
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")
    
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"File model {filename} sudah terunduh. Melewati proses pengunduhan...")
        return filename
    
    try:
        print(f"Mengunduh model pose landmarker ke direktori: {filename}...")
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
            raise ValueError("File yang terunduh kosong")
            
        print("Proses pengunduhan model pose landmarker berhasil!")
        return filename
        
    except Exception as e:
        print(f"Terdapat error dalam proses pengunduhan model face detector: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        raise

