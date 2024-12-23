import platform
import subprocess

## Problem specific for Mac
def check_gpu():
    """
    Memeriksa ketersediaan GPU pada sistem.
    Returns:
        str: "NVIDIA" untuk GPU NVIDIA, "MLX" untuk Apple Silicon, atau "CPU" jika tidak ada GPU
    """
    system = platform.system()
    print(f"System: {system}")
    # Memeriksa apakah sistem adalah Windows atau Linux 
    if system == "Linux" or system == "Windows":
        try:
            nvidia_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return "NVIDIA" 
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "CPU"

    # Memeriksa apakah sistem adalah macOS dengan arsitektur Apple Silicon
    elif system == "Darwin":  # macOS
        try:
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU: {cpu_info}")
            if "Apple" in cpu_info:  # Ini mencakupi semua chip buatan Apple (M1/M2/M3)
                return "MLX"
        except subprocess.CalledProcessError:
            pass
    return "CPU" # Menggunakan CPU sebagai default
