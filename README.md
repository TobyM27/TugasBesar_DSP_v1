# Tugas Besar Mata Kuliah Digital Processing Signal (IF3024)

## Dosen Pengampu: **Martin Clinton Tosima Manullang, S.T., M.T..**

# **Real Time HR & Resp Signal**

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="100" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" alt="Jupyter Lab Logo" width="100" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" alt="VS Code Logo" width="100" />
</p>

---

## **Anggota Kelompok**

| **Nama**                    | **NIM**   | **ID GITHUB**                                                               |
| --------------------------- | --------- | --------------------------------------------------------------------------- |
| Fransiskus Xaverius Gunawan | 121140010 | <a href="https://github.com/fransiskus-121140010">@fransiskus-121140010</a> |
| Arsyadana Estu Aziz         | 121140068 | <a href="https://github.com/archiseino">@archiseino</a>                     |
| Tobyanto Putra Mandiri      | 121140099 | <a href="https://github.com/TobyM27">@TobyM27</a>                           |

---

## **Deskripsi Proyek**

Proyek ini merupakan tugas akhir dari mata kuliah "Pengolahan Sinyal Digital IF(3024)" yang dapat
digunakan untuk memperoleh sinyal respirasi dan sinyal remote-photopletysmography (rPPG) dari input
video web-cam secara real-time.
Program ini memperoleh sinyal respirasi dengan menggunakan pose-landmarker dari MediaPipe
untuk menghitung pergerakan bahu saat pengguna melakukan pernapasan. Sedangkan untuk sinyal
rPPG, program ini menggunakan face-detector dari MediaPipe dan algoritma Plane Orthogonal-to-
Skin (POS) untuk menghitung detak jantung secara non-kontak dengan menganalisis perubahan warna
pada wajah pengguna[

---

## **Teknologi yang Digunakan**

Berikut adalah teknologi dan alat yang digunakan dalam proyek ini:

| Logo                                                                                                                           | Nama Teknologi | Fungsi                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------- | -------------------------------------------------------------------------------- |
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="60">            | Python         | Bahasa pemrograman utama untuk pengembangan filter.                              |
| <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" alt="Jupyter Lab Logo" width="60">             | Jupyter Lab    | Lingkungan pengembangan untuk menjalankan dan menguji skrip Python.              |
| <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" alt="VS Code Logo" width="60"> | VS Code        | Editor teks untuk mengedit skrip secara efisien dengan dukungan ekstensi Python. |

---

## **Library yang Digunakan**

Berikut adalah daftar library Python yang digunakan dalam proyek ini, beserta fungsinya:

| **Library**                | **Fungsi**                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `cv2`                      | Digunakan untuk menangkap gambar dari kamera dan memproses gambar secara langsung.                 |
| `mediapipe`                | Digunakan untuk mendeteksi landmark wajah, seperti posisi hidung, untuk mendeteksi gerakan kepala. |
| `jupyternotebook`          | Digunakan sebagai media explorasi model dan landmark yang sesuai.                                  |
| `PyQt`, `PyQtGraph`        | Digunakan untuk membuat tampilan GUI untuk sinyal dan live video feed.                             |
| `Scipy`, `numpy`, `pandas` | Digunakan untuk bahan oprasi pembuatan program                                                     |

---

## **Fitur**

### **1. RPPG (Remote Photoplethysmography)**

- Bagian ini rPPG menggunakan teknologi visi komputer untuk mengekstrak informasi tentang perubahan penyerapan cahaya pada kulit wajah.
  Sinyal yang didapat akan di proses menjadi sinyal detak jantung. Referensi yang digunakan pada tugas kali ini adalah metode POS dari Pak [Wenjing Wang](https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf)

### **2. Resp Signal**

- Bagian ini menggunakan metode Pose Detection dan Optical Tracking dengan Tracking Failure untuk mendeteksi gerakan bahu dan dada untuk mensimmulasikan gerakan pernafasan.

---

## Logbook

### Minggu 1

- Inisialisasi github repo dan github project management untuk manajemen tugas besar "Digital Signal Processing"
- Pembagian tugas antara Arsyadana, Fransiskus, dan Tobyanto

### Minggu 2

- Pembuatan branch baru pengembangan RPPG (dengan nama branch ambatulearn).
  - Membuat file `pos.ipynb` yang sudah diisikan dengan pemanggilan library dan fungsi download model dari mediapipe.

### Minggu 3

- Pembuatan branch baru pengembangan respiration (dengan nama branch respiration_code).
  - Membuat file `resp_code.py` yang sudah diisikan dengan pemanggilan library dan fungsi download model dari mediapipe.
  - Menjalankan kode `resp_code.py` untuk mendownload model dari mediapipe.

### Minggu 4

- Pembuatan project overleaf untuk laporan tugas besar DSP.
- Pembuatan GUI untuk menampilkan sinyal detak jantung dan pernafasan secara real-time (dengan Library PyQt).
- Penggabungan modul antara RPPG dan Respiration untuk menampilkan sinyal detak jantung dan pernafasan secara real-time.
- Refactor kode untuk memisahkan fungsi-fungsi yang berbeda ke dalam file yang berbeda dan membuat docstring pada tiap fungsi.
- Pembuatan requirements.txt dan environment.yml untuk memudahkan instalasi library yang dibutuhkan.

## How to run this

Dengan asumsi bahwa Anda sudah mempunyai environment manager seperti conda. maka buat environment baru seperti ini. Clone / fork lalu jalankan perintah ini.

### 2. Dengan environment.yml

```yaml
conda env create -f environment.yml
```

Lalu bukan environment yang sudah dibuat sebelumnya dengan.

```yaml
conda activate real_time_hr_respiration
```

Lalu jalankan perintah ini untuk menjalankan program.

```yaml
python main.py
```

### 2. Dengan requirements.txt

Atau jika Anda mengalami masalah dalam menggunakan environment.yml, anda bisa menggunakan requirements.txt. Jalankan perintah ini.

```yaml
conda create -n real_time_hr_respiration python=3.10.16
```

Lalu buka environment yang sudah dibuat sebelumnya dengan

```yaml
conda activate real_time_hr_respiration
```

Jalankan perintah ini untuk menginstall library yang dibutuhkan.

```yaml
pip install -r requirements.txt
```

Lalu jalankan perintah ini untuk menjalankan program.

```yaml
python main.py
```
