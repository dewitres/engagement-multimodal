# engagement-multimodal

Script Python untuk ekstraksi fitur wajah (EAR, MAR, head pose, gaze) dari gambar kamera mahasiswa — bagian dari penelitian model engagement mahasiswa berbasis multimodal.

## Fitur yang Diekstrak

- **Eye Aspect Ratio (EAR)** — kiri, kanan, rata-rata (indikator mata terbuka/tertutup)
- **Mouth Aspect Ratio (MAR)** — indikator mulut terbuka (menguap)
- **Head Pose** — pitch, yaw, roll (orientasi kepala dalam derajat)
- **Blink indicator** — deteksi mata tertutup berdasarkan threshold EAR
- **Gaze Direction** — arah pandangan berbasis iris (appearance-based)
- **Engagement label** — focused / distracted / fatigued (rule-based awal)

## Instalasi

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

### Lokal

1. Siapkan struktur folder data:
   ```
   data_kamera/
     ├── mahasiswa_001/
     │     ├── log_120__webcam-119-74-16-1677018846411.png
     │     └── ...
     ├── mahasiswa_002/
     │     └── ...
     └── ...
   ```

2. Sesuaikan `CONFIG["DATA_DIR"]` di dalam script jika perlu

3. Jalankan:
   ```bash
   python feature_extraction_kamera.py
   ```

### Google Colab

```python
# Sel 1 — install
!pip install -r requirements.txt -q

# Sel 2 — mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Sel 3 — clone repo
!git clone https://github.com/dewitres/engagement-multimodal.git
%cd engagement-multimodal

# Sel 4 — jalankan
import sys
sys.path.insert(0, '/content/engagement-multimodal')
from feature_extraction_kamera import CONFIG, run_extraction

CONFIG["DATA_DIR"]   = "/content/drive/MyDrive/data_kamera"
CONFIG["OUTPUT_DIR"] = "/content/drive/MyDrive/output_features"

df = run_extraction()
```

## Output

| File | Deskripsi |
|------|-----------|
| `features_kamera.csv` | Satu baris per gambar dengan semua fitur |
| `features_kamera_summary.csv` | Statistik agregat per mahasiswa |
| `extraction_log.txt` | Log error dan gambar yang gagal diproses |

## Dependencies

- MediaPipe (deteksi landmark wajah)
- OpenCV (pemrosesan gambar dan PnP solver)
- pandas, numpy (manipulasi data)
- pytz (timezone WIB)

## Konteks Penelitian

Script ini merupakan modul ekstraksi fitur visual dalam penelitian engagement multimodal mahasiswa, yang menggabungkan data kamera dengan log aktivitas LMS untuk membangun model prediksi tingkat keterlibatan dalam pembelajaran online.
