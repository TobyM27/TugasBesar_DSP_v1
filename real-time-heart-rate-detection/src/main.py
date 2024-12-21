import numpy as np
import cv2
import time
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from utils.heart_rate import cpu_POS

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def main():
    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    r_signal, g_signal, b_signal = [], [], []

    # Mediapipe Face Detection Initialization
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame using face_detection
        results = face_detection.process(frame_rgb)

        if results.detections:  # If there are faces detected
            for detection in results.detections:  # Loop through all the detected faces
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)

                # Define forehead ROI
                forehead_x = x + width // 4  # Start slightly to the right of the left edge
                forehead_y = y // 2 + 55  # Start at the top of the bounding box
                forehead_width = width // 2  # Half the width of the bounding box
                forehead_height = height // 5  # Quarter the height of the bounding box

                # Extract the forehead ROI
                roi = frame[forehead_y:forehead_y + forehead_height, forehead_x:forehead_x + forehead_width]

                # Optional: Draw the forehead ROI on the frame for visualization
                cv2.rectangle(frame, 
                              (forehead_x, forehead_y), 
                              (forehead_x + forehead_width, forehead_y + forehead_height), 
                              (0, 255, 0), 2)

                # Calculate mean pixel values for the RGB channels
                r_signal.append(np.mean(roi[:, :, 0]))
                g_signal.append(np.mean(roi[:, :, 1]))
                b_signal.append(np.mean(roi[:, :, 2]))

        # Display the frame
        cv2.imshow('Video Feed', frame)

        # Sample heart rate every 5 seconds
        if len(r_signal) >= fps * 5:
            # Prepare the signal for heart rate estimation
            rgb_signals = np.array([r_signal, g_signal, b_signal])
            rgb_signals = rgb_signals.reshape(1, 3, -1)
            rppg_signal = cpu_POS(rgb_signals, fps=fps)
            rppg_signal = rppg_signal.reshape(-1)

            # Bandpass filter the signal
            filtered_signal = bandpass_filter(rppg_signal, 0.75, 3.0, fps, order=5)

            # Normalize the signal
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

            # Apply moving average smoothing
            smoothed_signal = moving_average(normalized_signal, window_size=int(fps/2))

            # Find peaks in the signal
            peaks, _ = find_peaks(smoothed_signal, distance=fps/2)
            peak_intervals = np.diff(peaks) / fps
            heart_rate = 60.0 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0

            print(f'Detected Heart Rate: {heart_rate:.2f} BPM')
            r_signal, g_signal, b_signal = [], [], []  # Reset signals after sampling

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()