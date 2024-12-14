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