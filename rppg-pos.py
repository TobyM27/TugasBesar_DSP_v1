## Import Dependencies
import numpy as np
import mediapipe as mp
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
import scipy.signal as signal

## Core method Cpu POS 
def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    """
    eps: A small constant (10^-9) used to prevent division by zero in normalization steps.
    X: The input signal, which is a 3D array where:
    e: Number of estimators or regions in the frame (like different parts of the face).
    c: Color channels (3 for RGB).
    f: Number of frames.
    w: Window length, determined by the camera's frame rate (fps). For example, at 20 fps, w would be 32 (which corresponds to about 1.6 seconds of video).
    """
    eps = 10**-9
    X = signal
    e, c, f = X.shape # Number of estimators, color channels, and frames
    w = int(1.6 * kargs['fps']) # Window length in frames

    """
    P: A fixed 2x3 matrix used for the projection step. It defines how to transform the color channels (RGB) into a new space.
    Q: This is a stack of the matrix P repeated e times, where each P corresponds to an estimator (region of interest) in the video.
    """
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis = 0)

    """
    H: A matrix to store the estimated heart rate signal over time for each estimator.
    n: The current frame in the sliding window.
    m: The start index of the sliding window (calculating which frames are part of the current window).
    """
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window 
        m = n - w + 1

        """
        Temporal Normalization (Equation 5 from the paper): This step ensures that the signal is invariant to global lighting changes and other noise factors.
        """
        Cn = X[:, :, m:(n+1)]
        M = 1.0 / (np.mean(Cn, axis = 2) + eps)
        M = np.expand_dims(M, axis=2) # shape [e, c, w]
        Cn = np.multiply(Cn, M)

        """
        Projection (Equation 6 from the paper): This step transforms the RGB values into a space where the signal from blood flow (heart rate) is more distinct.
        """
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1) 

        """
        Tuning (Equation 7 from the paper): This step adjusts the projected components to make the heart rate signal clearer.
        """
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha - np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)

        """
        Overlap-Adding (Equation 8 from the paper): This step combines the processed signals from each frame to form the final output heart rate signal.
        """
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)  # Add the tuned signal to the output matrix

    return H

## Open the Original Video
original_video = cv2.VideoCapture("Samples/subject_s53.mp4") # 35 Hz

# Target sampling rate
target_rate = 35  # Hz

## Mediapipe Face Detection Initialization
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

## Since we've already the Original video on Cv2 Object, just prepare for the RPPG method

## original_video as the variable

# Prepare variables for the RPPG method
r_signal, g_signal, b_signal = [], [], []
f_count = 0
f_total = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))


"""
Our theoretical here, since the subject is moving along the video.
"""

## Main funciton
if __name__ == "__main__":
    try:
        while original_video.isOpened():
            print(f'Processing Frame {f_count}/{f_total}', end='\r')
            ret, frame = original_video.read()
            
            ### 3. Mendeteksi area wajah menggunakan mediapipe
            
            ### 3.1 Mengkonversi frame ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            ### 3.2 Memproses frame menggunakan face_detection
            results = face_detection.process(frame_rgb)
            
            if results.detections: # If there are faces detected
                for detection in results.detections: # Loop through all the detected faces
                    ### 3.3 Mendapatkan bounding box dari wajah
                    bbox = detection.location_data.relative_bounding_box
                    ### 3.4 Mendapatkan lebar dan tinggi frame
                    h, w, _ = frame.shape
                    ### 3.5 Mengkonversi bounding box ke koordinat piksel
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    ### 3.6 Mengkonversi lebar dan tinggi bounding box ke koordinat piksel
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    
                    # ### 3.7 Melakukan penyesuaian pada bounding box
                    # bbox_size_from_center = 70
                    
                    # bbox_center_x = x + width // 2
                    # bbox_center_y = y + height // 2
                    # new_x = bbox_center_x - bbox_size_from_center
                    # new_y = bbox_center_y - bbox_size_from_center
                    # new_width = bbox_size_from_center * 2
                    # new_height = bbox_size_from_center * 2
                    
                    ### 3.8 Menggambar bounding box pada frame
                    # cv2.rectangle(frame, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 255, 0), 2)
                    
                    
                    ### 4 Mendapatkan nilai rata-rata piksel dari ROI dan menambahkannya ke signal
                    # roi = frame[new_y:new_y+new_height, new_x:new_x+new_width]
                    # cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    # roi = frame[y:y+height, x:x+width]
                    # r_signal.append(np.mean(roi[:, :, 0]))
                    # g_signal.append(np.mean(roi[:, :, 1]))
                    # b_signal.append(np.mean(roi[:, :, 2]))
                                    # Define forehead ROI
                    forehead_x = x + width // 4  # Start slightly to the right of the left edge
                    forehead_y = y // 2 + 55 # Start at the top of the bounding box
                    forehead_width = width // 2  # Half the width of the bounding box
                    forehead_height = height // 5  # Quarter the height of the bounding box
                    
                    # Extract the forehead ROI
                    roi = frame[forehead_y:forehead_y+forehead_height, forehead_x:forehead_x+forehead_width]
                    
                    # Optional: Draw the forehead ROI on the frame for visualization
                    cv2.rectangle(frame, 
                                (forehead_x, forehead_y), 
                                (forehead_x + forehead_width, forehead_y + forehead_height), 
                                (0, 255, 0), 2)

                    # Calculate mean pixel values for the RGB channels
                    r_signal.append(np.mean(roi[:, :, 0]))
                    g_signal.append(np.mean(roi[:, :, 1]))
                    b_signal.append(np.mean(roi[:, :, 2]))

            
            if not ret:
                break

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            f_count += 1
        original_video.release()
        cv2.destroyAllWindows()

    except Exception as e:
        original_video.release()
        cv2.destroyAllWindows()

