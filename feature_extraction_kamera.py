"""
============================================================
FEATURE EXTRACTION - DATA KAMERA
Penelitian: Model Engagement Mahasiswa Berbasis Multimodal
============================================================

Fitur yang diekstraksi:
  1. Eye Aspect Ratio (EAR) - kiri, kanan, rata-rata
  2. Mouth Aspect Ratio (MAR)
  3. Head Pose Estimation (pitch, yaw, roll)
  4. Blink indicator (EAR < threshold)
  5. Gaze Direction (appearance-based)
  6. Engagement label awal (rule-based dari fitur geometris)

Struktur folder yang diharapkan:
  data_kamera/
    ├── mahasiswa_001/
    │     ├── log_120__webcam-119-74-16-1677018846411.png
    │     ├── log_121__webcam-119-74-16-1677018861411.png
    │     └── ...
    ├── mahasiswa_002/
    │     └── ...
    └── ...

Output:
  features_kamera.csv  - satu baris per gambar, semua fitur
  features_kamera_summary.csv - statistik per mahasiswa
  extraction_log.txt   - log error dan gambar yang gagal
============================================================
"""

import os
import re
import cv2
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import mediapipe as mp
from pathlib import Path

# ── Coba import opsional ──
try:
    import pytz
    WIB = pytz.timezone('Asia/Jakarta')
    USE_PYTZ = True
except ImportError:
    USE_PYTZ = False
    print("[INFO] pytz tidak tersedia, timestamp dalam UTC")

# ============================================================
# KONFIGURASI — SESUAIKAN DENGAN SETUP ANDA
# ============================================================
CONFIG = {
    # Path ke folder utama yang berisi subfolder per mahasiswa
    "DATA_DIR"        : "data_kamera",         # ganti sesuai path Anda

    # Path output
    "OUTPUT_DIR"      : "output_features",

    # Threshold EAR untuk deteksi mata tertutup / mengantuk
    # Nilai normal: ~0.25-0.30; di bawah threshold = mata hampir tertutup
    "EAR_THRESHOLD"   : 0.21,

    # Threshold MAR untuk deteksi mulut terbuka (menguap)
    "MAR_THRESHOLD"   : 0.6,

    # Threshold head pose (derajat) untuk deteksi kepala menyimpang
    "YAW_THRESHOLD"   : 30,    # menoleh kiri/kanan
    "PITCH_THRESHOLD" : 25,    # menunduk/mendongak

    # Tampilkan preview gambar saat ekstraksi (set False untuk batch)
    "SHOW_PREVIEW"    : False,

    # Ekstensi file gambar yang dicari
    "IMG_EXTENSIONS"  : ['.png', '.jpg', '.jpeg'],

    # Cetak progress setiap N gambar
    "PRINT_EVERY"     : 50,
}

# ============================================================
# INISIALISASI MEDIAPIPE
# ============================================================
mp_face_mesh    = mp.solutions.face_mesh
mp_face_detect  = mp.solutions.face_detection
mp_drawing      = mp.solutions.drawing_utils

# Landmark indices MediaPipe Face Mesh
# Referensi: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Mata kiri (dari perspektif kamera = kanan orang)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Mata kanan (dari perspektif kamera = kiri orang)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Mulut
MOUTH     = [61, 291, 13, 14, 17, 0]
# Landmark untuk head pose (6 titik 3D reference)
NOSE_TIP      = 4
CHIN          = 152
LEFT_EYE_LM   = 263
RIGHT_EYE_LM  = 33
LEFT_MOUTH    = 287
RIGHT_MOUTH   = 57

# ============================================================
# FUNGSI UTILITAS
# ============================================================

def parse_timestamp(filename):
    """
    Ekstrak Unix timestamp (milidetik) dari nama file.
    Contoh: log_120__webcam-119-74-16-1677018846411.png
             → 1677018846411 ms → datetime
    """
    # Cari angka panjang (>10 digit) di nama file → Unix timestamp ms
    matches = re.findall(r'(\d{13})', filename)
    if matches:
        ts_ms  = int(matches[-1])          # ambil yang terakhir
        ts_sec = ts_ms / 1000.0
        dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        if USE_PYTZ:
            dt_wib = dt_utc.astimezone(WIB)
            return dt_wib, ts_ms
        return dt_utc, ts_ms
    return None, None


def euclidean(p1, p2):
    """Jarak Euclidean antara dua titik 2D."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def get_eye_landmarks(landmarks, indices, img_w, img_h):
    """Ambil koordinat pixel dari indeks landmark mata."""
    return [(int(landmarks[i].x * img_w),
             int(landmarks[i].y * img_h)) for i in indices]


def compute_EAR(eye_pts):
    """
    Eye Aspect Ratio (Soukupova & Cech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    Indeks: p1=kiri, p4=kanan, p2&p6=atas-bawah vertikal 1,
            p3&p5=atas-bawah vertikal 2
    Nilai normal ~0.25-0.30; mendekati 0 = mata tertutup
    """
    v1 = euclidean(eye_pts[1], eye_pts[5])
    v2 = euclidean(eye_pts[2], eye_pts[4])
    h  = euclidean(eye_pts[0], eye_pts[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_MAR(mouth_pts):
    """
    Mouth Aspect Ratio (Daza et al., 2020):
    MAR = jarak vertikal mulut / jarak horizontal mulut
    Nilai tinggi (>0.6) = mulut terbuka lebar → kemungkinan menguap
    """
    # Titik: [kiri, kanan, atas-tengah, bawah-tengah, atas-luar, bawah-luar]
    v1 = euclidean(mouth_pts[2], mouth_pts[5])
    v2 = euclidean(mouth_pts[3], mouth_pts[4])
    h  = euclidean(mouth_pts[0], mouth_pts[1])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_head_pose(landmarks, img_w, img_h):
    """
    Head Pose Estimation menggunakan Perspective-n-Point (PnP).
    Mengembalikan (pitch, yaw, roll) dalam derajat.

    Pitch  > 0: mendongak; < 0: menunduk
    Yaw    > 0: menoleh kanan; < 0: menoleh kiri
    Roll   > 0: miring kanan; < 0: miring kiri

    Referensi: Murphy-Chutorian & Trivedi (2009)
    """
    # 3D model points (koordinat wajah generik dalam mm)
    model_pts = np.array([
        (0.0,    0.0,    0.0),    # nose tip
        (0.0,   -330.0, -65.0),   # chin
        (-225.0, 170.0, -135.0),  # left eye corner
        (225.0,  170.0, -135.0),  # right eye corner
        (-150.0,-150.0, -125.0),  # left mouth
        (150.0, -150.0, -125.0),  # right mouth
    ], dtype=np.float64)

    # 2D image points dari landmarks
    def lm_px(idx):
        return (landmarks[idx].x * img_w,
                landmarks[idx].y * img_h)

    img_pts = np.array([
        lm_px(NOSE_TIP),
        lm_px(CHIN),
        lm_px(LEFT_EYE_LM),
        lm_px(RIGHT_EYE_LM),
        lm_px(LEFT_MOUTH),
        lm_px(RIGHT_MOUTH),
    ], dtype=np.float64)

    # Camera matrix (estimasi dari ukuran gambar)
    focal  = img_w
    center = (img_w / 2, img_h / 2)
    cam_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # asumsi tidak ada distorsi lensa

    success, rot_vec, trans_vec = cv2.solvePnP(
        model_pts, img_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return None, None, None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    # Konversi ke sudut Euler
    sy = math.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2( rot_mat[2,1], rot_mat[2,2])
        yaw   = math.atan2(-rot_mat[2,0], sy)
        roll  = math.atan2( rot_mat[1,0], rot_mat[0,0])
    else:
        pitch = math.atan2(-rot_mat[1,2], rot_mat[1,1])
        yaw   = math.atan2(-rot_mat[2,0], sy)
        roll  = 0

    return (math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll))


def estimate_gaze(landmarks, img_w, img_h):
    """
    Estimasi arah pandang mata (appearance-based gaze estimation).
    Menggunakan rasio posisi iris terhadap bounding box mata.

    Mengembalikan:
      gaze_x: -1 (kiri) hingga +1 (kanan)
      gaze_y: -1 (atas) hingga +1 (bawah)
      gaze_label: 'center'|'left'|'right'|'up'|'down'
    """
    # Iris landmarks (tersedia di MediaPipe FaceMesh dengan refine_landmarks=True)
    # Iris kiri: 474-477; Iris kanan: 469-472
    try:
        # Pusat iris kanan (dari perspektif kamera = iris kiri orang)
        iris_r = landmarks[468]
        # Bounding box mata kanan
        eye_r_pts = get_eye_landmarks(landmarks, RIGHT_EYE, img_w, img_h)
        ex_min = min(p[0] for p in eye_r_pts)
        ex_max = max(p[0] for p in eye_r_pts)
        ey_min = min(p[1] for p in eye_r_pts)
        ey_max = max(p[1] for p in eye_r_pts)

        ex_range = ex_max - ex_min
        ey_range = ey_max - ey_min

        if ex_range < 1 or ey_range < 1:
            return 0.0, 0.0, 'unknown'

        iris_px = (iris_r.x * img_w, iris_r.y * img_h)
        gaze_x  = (iris_px[0] - ex_min) / ex_range * 2 - 1  # [-1, 1]
        gaze_y  = (iris_px[1] - ey_min) / ey_range * 2 - 1  # [-1, 1]

        # Label arah pandang
        THRESHOLD = 0.3
        if abs(gaze_x) < THRESHOLD and abs(gaze_y) < THRESHOLD:
            label = 'center'
        elif gaze_x < -THRESHOLD:
            label = 'left'
        elif gaze_x > THRESHOLD:
            label = 'right'
        elif gaze_y < -THRESHOLD:
            label = 'up'
        else:
            label = 'down'

        return round(gaze_x, 4), round(gaze_y, 4), label

    except (IndexError, AttributeError):
        return 0.0, 0.0, 'unknown'


def rule_based_engagement(ear_avg, mar, yaw, pitch, gaze_label):
    """
    Label engagement rule-based awal dari fitur geometris.
    Digunakan sebagai sanity check sebelum model ML dilatih.

    Label:
      'focused'    - semua indikator normal
      'distracted' - kepala atau pandangan menyimpang
      'fatigued'   - EAR rendah atau MAR tinggi (mengantuk/menguap)
      'unknown'    - tidak dapat ditentukan
    """
    cfg = CONFIG
    if ear_avg < 0 or mar < 0:
        return 'unknown'
    if ear_avg < cfg['EAR_THRESHOLD'] or mar > cfg['MAR_THRESHOLD']:
        return 'fatigued'
    if (yaw is not None and abs(yaw) > cfg['YAW_THRESHOLD']) or \
       (pitch is not None and abs(pitch) > cfg['PITCH_THRESHOLD']) or \
       gaze_label in ['left', 'right', 'up']:
        return 'distracted'
    return 'focused'


# ============================================================
# EKSTRAKSI FITUR UTAMA
# ============================================================

def extract_features_from_image(img_path, face_mesh):
    """
    Ekstrak semua fitur dari satu gambar.
    Mengembalikan dictionary fitur atau None jika gagal.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "Gagal membaca gambar"

    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None, "Wajah tidak terdeteksi"

    # Ambil landmark wajah pertama (satu mahasiswa per frame)
    lm = results.multi_face_landmarks[0].landmark

    # ── EAR ──
    left_pts  = get_eye_landmarks(lm, LEFT_EYE,  img_w, img_h)
    right_pts = get_eye_landmarks(lm, RIGHT_EYE, img_w, img_h)
    ear_left  = compute_EAR(left_pts)
    ear_right = compute_EAR(right_pts)
    ear_avg   = (ear_left + ear_right) / 2.0

    # ── MAR ──
    mouth_pts_raw = [(int(lm[i].x * img_w),
                      int(lm[i].y * img_h)) for i in MOUTH]
    mar = compute_MAR(mouth_pts_raw)

    # ── Head Pose ──
    pitch, yaw, roll = compute_head_pose(lm, img_w, img_h)

    # ── Gaze ──
    gaze_x, gaze_y, gaze_label = estimate_gaze(lm, img_w, img_h)

    # ── Label rule-based ──
    engage_label = rule_based_engagement(ear_avg, mar, yaw, pitch, gaze_label)

    # ── Flag kondisi ──
    is_drowsy     = int(ear_avg < CONFIG['EAR_THRESHOLD'])
    is_yawning    = int(mar > CONFIG['MAR_THRESHOLD'])
    is_distracted = int(
        (yaw  is not None and abs(yaw)   > CONFIG['YAW_THRESHOLD']) or
        (pitch is not None and abs(pitch) > CONFIG['PITCH_THRESHOLD']) or
        gaze_label in ['left', 'right']
    )

    return {
        # Identifikasi
        'file'             : img_path.name,

        # EAR
        'ear_left'         : round(ear_left,  4),
        'ear_right'        : round(ear_right, 4),
        'ear_avg'          : round(ear_avg,   4),
        'is_drowsy'        : is_drowsy,

        # MAR
        'mar'              : round(mar, 4),
        'is_yawning'       : is_yawning,

        # Head Pose
        'pitch'            : round(pitch, 4) if pitch is not None else None,
        'yaw'              : round(yaw,   4) if yaw   is not None else None,
        'roll'             : round(roll,  4) if roll  is not None else None,

        # Gaze
        'gaze_x'           : gaze_x,
        'gaze_y'           : gaze_y,
        'gaze_label'       : gaze_label,
        'is_distracted'    : is_distracted,

        # Label engagement rule-based
        'engagement_label' : engage_label,

        # Metadata gambar
        'img_width'        : img_w,
        'img_height'       : img_h,
        'face_detected'    : 1,
    }, None


# ============================================================
# PIPELINE UTAMA
# ============================================================

def run_extraction():
    data_dir   = Path(CONFIG["DATA_DIR"])
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "extraction_log.txt"
    log_lines = []

    all_records  = []
    total_ok     = 0
    total_fail   = 0
    total_no_ts  = 0

    print("=" * 60)
    print("  FEATURE EXTRACTION — DATA KAMERA")
    print("=" * 60)
    print(f"  Input  : {data_dir.resolve()}")
    print(f"  Output : {output_dir.resolve()}")
    print()

    # ── Inisialisasi FaceMesh ──
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,       # diperlukan untuk iris (gaze)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Iterasi per subfolder mahasiswa ──
    subfolders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"  Ditemukan {len(subfolders)} subfolder mahasiswa\n")

    for folder in subfolders:
        mahasiswa_id = folder.name
        img_files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in CONFIG["IMG_EXTENSIONS"]
        ])

        if not img_files:
            log_lines.append(f"[WARN] {mahasiswa_id}: tidak ada gambar")
            continue

        print(f"  [{mahasiswa_id}] — {len(img_files)} gambar")
        mhs_ok   = 0
        mhs_fail = 0

        for i, img_path in enumerate(img_files):

            # Ekstrak timestamp dari nama file
            dt, ts_ms = parse_timestamp(img_path.name)

            # Ekstrak fitur
            features, error = extract_features_from_image(img_path, face_mesh)

            if features is None:
                total_fail += 1
                mhs_fail   += 1
                log_lines.append(
                    f"[FAIL] {mahasiswa_id}/{img_path.name}: {error}")
                # Tetap simpan baris kosong agar sinkronisasi temporal terjaga
                all_records.append({
                    'mahasiswa_id'   : mahasiswa_id,
                    'file'           : img_path.name,
                    'timestamp_ms'   : ts_ms,
                    'datetime'       : dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None,
                    'date'           : dt.strftime('%Y-%m-%d') if dt else None,
                    'hour'           : dt.hour if dt else None,
                    'face_detected'  : 0,
                    'engagement_label': 'missing',
                    # Semua fitur None
                    **{k: None for k in [
                        'ear_left','ear_right','ear_avg','is_drowsy',
                        'mar','is_yawning',
                        'pitch','yaw','roll',
                        'gaze_x','gaze_y','gaze_label','is_distracted',
                        'img_width','img_height'
                    ]}
                })
                continue

            # Gabungkan timestamp + identitas + fitur
            record = {
                'mahasiswa_id' : mahasiswa_id,
                'timestamp_ms' : ts_ms,
                'datetime'     : dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None,
                'date'         : dt.strftime('%Y-%m-%d') if dt else None,
                'hour'         : dt.hour if dt else None,
            }
            record.update(features)
            all_records.append(record)

            total_ok += 1
            mhs_ok   += 1

            if (i + 1) % CONFIG["PRINT_EVERY"] == 0:
                print(f"    ... {i+1}/{len(img_files)} diproses")

        print(f"    Selesai: {mhs_ok} berhasil, {mhs_fail} gagal")

    face_mesh.close()

    # ── Simpan hasil ke CSV ──
    if not all_records:
        print("\n[ERROR] Tidak ada data yang berhasil diekstraksi.")
        return

    df = pd.DataFrame(all_records)

    # Urutkan berdasarkan mahasiswa + timestamp
    df = df.sort_values(['mahasiswa_id', 'timestamp_ms'], na_position='last')

    output_path = output_dir / "features_kamera.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  Fitur disimpan ke: {output_path}")
    print(f"  Total baris       : {len(df)}")
    print(f"  Wajah terdeteksi  : {total_ok}")
    print(f"  Gagal (missing)   : {total_fail}")

    # ── Statistik per mahasiswa ──
    df_ok = df[df['face_detected'] == 1].copy()
    if not df_ok.empty:
        summary = df_ok.groupby('mahasiswa_id').agg(
            total_frame       = ('file', 'count'),
            frame_terdeteksi  = ('face_detected', 'sum'),
            ear_avg_mean      = ('ear_avg', 'mean'),
            ear_avg_std       = ('ear_avg', 'std'),
            mar_mean          = ('mar', 'mean'),
            yaw_abs_mean      = ('yaw', lambda x: x.abs().mean()),
            pitch_abs_mean    = ('pitch', lambda x: x.abs().mean()),
            pct_drowsy        = ('is_drowsy', 'mean'),
            pct_yawning       = ('is_yawning', 'mean'),
            pct_distracted    = ('is_distracted', 'mean'),
            pct_focused       = ('engagement_label',
                                  lambda x: (x == 'focused').mean()),
            pct_fatigued      = ('engagement_label',
                                  lambda x: (x == 'fatigued').mean()),
            pct_distracted_lb = ('engagement_label',
                                  lambda x: (x == 'distracted').mean()),
        ).round(4).reset_index()

        summary_path = output_dir / "features_kamera_summary.csv"
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"  Ringkasan per mhs : {summary_path}")

        # Tampilkan statistik cepat
        print("\n  ── Statistik Global ──")
        print(f"  Rata-rata EAR     : {df_ok['ear_avg'].mean():.4f}")
        print(f"  Rata-rata MAR     : {df_ok['mar'].mean():.4f}")
        print(f"  % Mengantuk       : {df_ok['is_drowsy'].mean()*100:.1f}%")
        print(f"  % Menguap         : {df_ok['is_yawning'].mean()*100:.1f}%")
        print(f"  % Distraksi       : {df_ok['is_distracted'].mean()*100:.1f}%")
        print()
        print("  ── Distribusi Label Engagement ──")
        label_dist = df_ok['engagement_label'].value_counts(normalize=True)*100
        for lbl, pct in label_dist.items():
            print(f"  {lbl:<15}: {pct:.1f}%")

    # ── Simpan log ──
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    if log_lines:
        print(f"\n  Log error        : {log_path} ({len(log_lines)} entri)")

    print("\n  Ekstraksi selesai!")
    print("=" * 60)

    return df


# ============================================================
# JALANKAN
# ============================================================
if __name__ == "__main__":

    # ── Instalasi otomatis (uncomment jika belum install) ──
    # import subprocess, sys
    # subprocess.check_call([sys.executable, "-m", "pip", "install",
    #     "mediapipe", "opencv-python", "pandas", "numpy"])

    # ── Untuk Google Colab: mount drive terlebih dahulu ──
    # from google.colab import drive
    # drive.mount('/content/drive')
    # CONFIG["DATA_DIR"]   = "/content/drive/MyDrive/data_kamera"
    # CONFIG["OUTPUT_DIR"] = "/content/drive/MyDrive/output_features"

    df = run_extraction()


# ============================================================
# CATATAN PENGGUNAAN
# ============================================================
"""
CARA MENJALANKAN:

1. Windows (Command Prompt / Anaconda Prompt):
   pip install mediapipe opencv-python pandas numpy
   python feature_extraction_kamera.py

2. Google Colab:
   - Upload script ini ke Colab
   - Uncomment bagian mount drive di atas
   - Sesuaikan CONFIG["DATA_DIR"] dengan path di Google Drive
   - Jalankan sel

STRUKTUR OUTPUT (features_kamera.csv):
  mahasiswa_id  : ID/nama subfolder mahasiswa
  timestamp_ms  : Unix timestamp milidetik dari nama file
  datetime      : Waktu dalam format YYYY-MM-DD HH:MM:SS (WIB)
  date          : Tanggal (YYYY-MM-DD)
  hour          : Jam akses (0-23)
  file          : Nama file gambar
  ear_left      : EAR mata kiri [0-1]
  ear_right     : EAR mata kanan [0-1]
  ear_avg       : Rata-rata EAR [0-1] — indikator utama kelelahan
  is_drowsy     : 1 jika ear_avg < 0.21 (mengantuk)
  mar           : Mouth Aspect Ratio [0-1]
  is_yawning    : 1 jika mar > 0.6 (menguap)
  pitch         : Sudut kepala atas-bawah (derajat)
  yaw           : Sudut kepala kiri-kanan (derajat)
  roll          : Sudut kepala miring (derajat)
  gaze_x        : Arah pandang horizontal [-1=kiri, +1=kanan]
  gaze_y        : Arah pandang vertikal [-1=atas, +1=bawah]
  gaze_label    : center|left|right|up|down|unknown
  is_distracted : 1 jika yaw/pitch melebihi threshold atau gaze menyimpang
  engagement_label: focused|distracted|fatigued|unknown (rule-based)
  face_detected : 1 jika wajah berhasil terdeteksi, 0 jika gagal
  img_width     : Lebar gambar (pixel)
  img_height    : Tinggi gambar (pixel)

LANGKAH SELANJUTNYA SETELAH EKSTRAKSI:
  1. Cek features_kamera_summary.csv — lihat mhs dengan banyak missing
  2. Merge dengan data log LMS berdasarkan mahasiswa_id + timestamp
  3. Lanjutkan ke feature engineering: normalisasi, windowing 15 detik
  4. Latih Vision Engine (MobileNetV3) menggunakan fitur + label
"""
