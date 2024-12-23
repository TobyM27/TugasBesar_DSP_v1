import os
import requests
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt

def download_pose_model():
    """
    Downloads the MediaPipe Pose Landmarker model if not already available.
    Returns the file path of the model.
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker.task")
    
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
    
    return model_path

def initialize_pose_landmarker():
    """
    Initializes and returns the MediaPipe Pose Landmarker.
    """
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = download_pose_model()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return PoseLandmarker.create_from_options(options)

def get_respiration_roi(image, pose_landmarker):
    """
    Detects the shoulders and returns the Region of Interest (ROI) for respiration signal extraction.

    Args:
        image (np.ndarray): Input video frame.
        pose_landmarker: MediaPipe pose detector.

    Returns:
        tuple: Coordinates of the ROI (left_x, top_y, right_x, bottom_y).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = pose_landmarker.detect(mp_image)

    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in the frame!")

    landmarks = detection_result.pose_landmarks[0]
    height, width = image.shape[:2]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)

    x_size, y_size = 100, 150
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)

    return (left_x, top_y, right_x, bottom_y)

def low_pass_filter(data, cutoff, fs, order=5):
    """
    Applies a low-pass Butterworth filter to the input signal.

    Args:
        data (np.ndarray): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def process_respiration_from_webcam():
    """
    Processes respiration signal in real-time from a webcam feed.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    pose_landmarker = initialize_pose_landmarker()

    respiration_signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            left_x, top_y, right_x, bottom_y = get_respiration_roi(frame, pose_landmarker)
            roi = frame[top_y:bottom_y, left_x:right_x]
            avg_intensity = np.mean(roi[:, :, 1])  # Green channel
            respiration_signal.append(avg_intensity)

            # Apply low-pass filter in real-time (only for display purposes)
            if len(respiration_signal) > 30:  # Ensure sufficient data points
                filtered_signal = low_pass_filter(respiration_signal, cutoff=0.5, fs=30)
                cv2.putText(frame, f"Respiration Intensity: {filtered_signal[-1]:.2f}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except ValueError:
            continue

        # Display the frame with ROI
        cv2.imshow('Respiration Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_respiration_from_webcam()
