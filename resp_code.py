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
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.signal import butter, filtfilt, find_peaks

# menginstall model pose landmarker dari MediaPipe. 
def download_model():
    """
    Mengunduh model pose landmaker dari MediaPipe
    Model pose landmarker akan digunakan untuk mendeteksi pose dari pasien
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")

    #memeriksa apakah file sudah ada dan tidak kosong
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} sudah ada. Melewati proses download.")
        return filename
    
    # menampilkan proses download dengan library tqdm
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

def roi_enhancement(roi):
    """ 
    Dengan fungsi ini, kualitas Region of Interest (ROI) dapat ditingkatkan dengan teknik pemrosesan gambar. 

    Args:
        roi (np.ndarray): ROI dalam dari frame video 

    Returns: 
        numpy.ndarray: ROI yang telah ditingkatkan kualitasnya
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
    
def get_respiration_initial_roi(image, pose_landmarker, x_size=100, y_size=150, shift_x=0, shift_y=0):
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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    # Membuat gambar MediaPip dari frame pertama
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Mendeteksi pose dari frame pertama
    detection_result = pose_landmarker.detect(mp_image)

    if not detection_result.pose_landmarks:
        raise ValueError("Tidak ada pose yang terdeteksi pada frame!")
    
    landmarks = detection_result.pose_landmarks[0]

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

def prepare_plot():
    """
    Menyiapkan plot untuk menampilkan visualisasi gerakan bahu pasien dengan matplotlib.

    Returns:
        tuple: Figure dan Axis dari plot
    """
    fig = plt.figure(figsize=(4, 3), facecolor='none')
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.7) # Membuat plot semi-transparan
    ax.set_xlabel('Waktu (detik)')
    ax.set_ylabel('Y position(px/pixels)')
    ax.set_title("Pergerakan Bahu Pasien")
    ax.grid(True, alpha=1) 
    return fig, ax

def plot_shoulders_movement(timestamps, y_positions):
    """
    Membuat plot untuk pergerakan bahu terhadap indeks waktu.

    Args:
        timestamps: larik waktu
        y_positions: larik posisi y bahu 
    """
    plt.figure(figsize=(12,6))
    plt.plot(timestamps, y_positions, label='Average Y Position', color='green')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Y Position (px/pixels)')
    plt.title('Pergerakan Bahu Pasien per Detik')
    plt.legend()
    plt.grid(True)
    plt.show()

def bandpass_filter(signal, lowcut, highcut, fs, order=5): # panggil fungsi ini dari program main.py dari real-time-heart-rate-detection
    """
    Menerapkan filter bandpass Butterworth pada sinyal input.

    Args:
        signal (np.ndarray): Sinyal input.
        lowcut (float): Frekuensi cut-off rendah dalam Hz.
        highcut (float): Frekuensi cut-off tinggi dalam Hz.
        fs (float): Frekuensi sampling dalam Hz.
        order (int): Orde dari filter.

    Returns:
        np.ndarray: Sinyal yang telah difilter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def process_respiration_webcam(pose_landmarker,x_size=300, y_size=250, shift_x=0, shift_y=0, save_video=False):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) # Mengambil fps dari webcam
    # Membatasi ukuran frame webcam menjadi 1280x720
    width = int(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280))
    height = int(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720))

    # Menginisialisasi video writer untuk menyimpan video hasil real-time
    if save_video:
        data_directory = 'data'
        output_dir = os.path.join(data_directory, 'respiration_output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'percobaan_shoulder_track_webcam.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWrite(output_path, fourcc, fps, (width, height))

    respiration_signal = [] # Inisialisasi sinyal respirasi

    # Menyiapkan plot 
    fig,ax = prepare_plot()
    timestamps = []
    y_positions = []
    start_time = time.time()

    # Membaca frame pertama dan mengambil ROI
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Tidak dapat mengakses webcam")
    
    try: 
        # Mengambil ROI dari frame pertama sewaktu memulai webcam
        roi_coords = get_respiration_initial_roi(first_frame, pose_landmarker, x_size, y_size, shift_x, shift_y)
        left_x, top_y, right_x, bottom_y = roi_coords

        # Inisialiasi ROI dengan Optical Flow
        old_frame = first_frame.copy()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Menginisialisasi ROI dan pendeteksi fitur
        roi = old_gray[top_y:bottom_y, left_x:right_x]
        features = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.15, minDistance=3, blockSize=7)

        if features is None:
            raise ValueError("Tidak ada fitur yang terdeteksi pada ROI!")
        
        # Menyesuaikan koordinat fitur ke frame penuh
        features = np.float32(features) 
        features[:,:,0] += left_x
        features[:,:,1] += top_y 

        # LK parameters (Lucas-Kanade)
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        frame_count = 0
        start_time = time.time()

        # Looping untuk mendeteksi pergerakan bahu pasien hingga program dihentikan (pengguna menekan q)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            current_time = time.time() - start_time

            if len(features) > 10:
                # Menghitung Optical Flow
                new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, features, None, **lk_params)

                good_old = features[status == 1]
                good_new = new_features[status == 1]

                # Mengambar garis untuk menunjukkan pergerakan bahu
                mask = np.zeros_like(frame)
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
                frame = cv2.add(frame,mask)

                # Mengupdate tracking dan grafik plot
                if len(good_new) > 0:
                    average_y = np.mean(good_new[:,1])
                    y_positions.append(average_y)
                    timestamps.append(current_time)
                    features = good_new.reshape(-1, 1, 2)

                    # Mengupdate plot 
                    ax.clear()
                    ax.set_facecolor('none')
                    #ax.patch.set_alpha(0.7)
                    ax.plot(timestamps, y_positions, 'g-', linewidth=2) # Tolong modifikasi line ini
                    ax.set_xlabel('Waktu (detik)')
                    ax.set_ylabel('Posisi Y (px/pixels)')
                    ax.set_title('Pergerakan Bahu Pasien')
                    ax.grid(True, alpha=0.5)
                    
                    # Mengkonversi dan melakukan overlay plot pada frame
                    fig.canvas.draw()
                    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plot_height = int(height * 0.3)
                    plot_width = int(width * 0.3)
                    plot_img = cv2.resize(plot_img, (plot_width, plot_height))
                    
                    # Overlay plot pada frame
                    y_offset = 20 
                    x_offset = width - plot_width - 20
                    frame[y_offset:y_offset + plot_height, x_offset:x_offset + plot_width] = plot_img

                    # Menghitung sinyal respirasi
                    roi = frame[top_y:bottom_y, left_x:right_x]
                    average_intensity = np.mean(roi[:, :, 1])
                    respiration_signal.append(average_intensity)

                    # Mengambil sampel sinyal respirasi setiap 5 detik
                    if len(respiration_signal) >= fps * 5:
                    # Memproses sinyal respirasi
                        filtered_signal = bandpass_filter(respiration_signal, 0.75, 3.0, fps, order=5)
                        cv2.putText(frame, f"Respiration Intensity: {filtered_signal[-1]:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Melakukan deteksi ulang jika diperlukan
                roi = frame_gray[top_y:bottom_y, left_x:right_x]
                features = cv2.goodFeaturesToTrack(roi, 
                                                   maxCorners=60, 
                                                   qualityLevel=0.15, 
                                                   minDistance=3, 
                                                   blockSize=7)
                if features is not None:
                    features = features + np.array([[left_x, top_y]], dtype=np.float32)
            
            # Mengambar ROI pada frame
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 2)

            # Menampilkan frame 
            cv2.imshow('Pendeteksi Bahu', frame) 

            # Menyimpan frame 
            out.write(frame)
            
            # Mengupdate hasil tracking
            old_gray = frame_gray.copy()
            frame_count += 1
                
            # Menghentikan program apabila tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Membersihkan frame cv2
        cap.release()
        out.release
        cv2.destroyAllWindows()
        plt.close(fig)
        # Mengembalikan hasil sinyal respirasi
        return timestamps, y_positions
    
    except Exception as e:
        print(f"error during tahap pemrosesan webcam: {str(e)}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    """
    Fungsi utama dari program respiration. 
    Di tahap ini, program menginisialisasi model pose landmark dan memulai webcam untuk mendeteksi gerakan bahu pasien.
    """
    detector_image = None
    # Mengisiasi model landmarker
    try: 
        # Mengunduh model pose landmarker dari MediaPipe apabila belum ada. Jika ada, maka model akan langsung diinisiasi
        model_path = download_model()

        # Mempersiapan pose landmarker
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
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Membuat landmarker dari opsi yang sudah diinisiasi
        respiration_detector = PoseLandmarker.create_from_options(options)

        # Memulai proses deteksi gerakan bahu pasien secara real time
        timestamps, y_positions = process_respiration_webcam(respiration_detector,
                                                             x_size=300, 
                                                             y_size=250, 
                                                             shift_x=0, 
                                                             shift_y=0, 
                                                             save_video=False)
        print("Done!")
    except Exception as e:
        print("Terdeteksi error: ", str(e))
        raise
    finally:
        if respiration_detector:
            respiration_detector.close()

if __name__ == "__main__":
    main()