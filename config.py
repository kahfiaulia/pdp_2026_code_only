"""
config.py - Konfigurasi Terpusat
Deteksi Retinopati Diabetik dengan MViTv2 + Explainability Hibrida

Pilih device (CPU atau GPU) dengan meng-comment/uncomment baris yang sesuai.
"""

import os
import torch
import multiprocessing

# ============================================================
# PILIH DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Informasi device yang digunakan
# ============================================================
if multiprocessing.current_process().name == "MainProcess":
    print(f"[CONFIG] Device yang digunakan: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CONFIG] CUDA Version: {torch.version.cuda}")
        print(f"[CONFIG] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[CONFIG] Mode CPU aktif - training akan lebih lambat")

# ============================================================
# PATH DATASET
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "APTOS 2019 Diabetic Retinopathy Dataset")

# Path ke folder gambar (nested structure: train_images/train_images/)
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train_images", "train_images")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "val_images", "val_images")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test_images", "test_images")

# Path ke CSV label
TRAIN_CSV = os.path.join(DATASET_DIR, "train_1.csv")
VAL_CSV = os.path.join(DATASET_DIR, "valid.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explainability")

# Buat direktori output jika belum ada
for d in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, METRICS_DIR, EXPLAIN_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# HYPERPARAMETER MODEL
# ============================================================
# MViTv2 Model
MODEL_NAME = "mvitv2_tiny"  # Opsi: "mvitv2_tiny", "mvitv2_small", "mvitv2_base"
NUM_CLASSES = 5
PRETRAINED = True           # Gunakan pretrained ImageNet weights

# ============================================================
# HYPERPARAMETER TRAINING
# ============================================================
IMG_SIZE = 384              # Resolusi input (384x384 piksel, lebih baik untuk deteksi lesi kecil)
BATCH_SIZE = 8              # Kurangi jika kehabisan memori GPU/RAM (dikurangi karena resolusi lebih besar)
NUM_WORKERS = 0            # Jumlah worker untuk data loading (kurangi di Windows jika error)
EPOCHS = 20                 # Jumlah epoch training
WARMUP_EPOCHS = 3           # Epoch warmup sebelum cosine annealing (stabilisasi transfer learning)
FREEZE_EPOCHS = 3           # Jumlah epoch pertama di mana backbone MViTv2 di-freeze
LEARNING_RATE = 1e-4        # Learning rate awal
WEIGHT_DECAY = 1e-4         # Weight decay untuk AdamW
PATIENCE = 7                # Early stopping patience
MIN_DELTA = 1e-4            # Minimum improvement untuk early stopping

# ============================================================
# ORDINAL LOSS SETTINGS (OrdinalCrossEntropyLoss)
# ============================================================
# Dipakai untuk mengganti label_smoothing biasa dengan soft target
# berbasis jarak ordinal + penalti regresi ordinal. Tujuannya menekan
# kesalahan klasifikasi yang melompat jauh (mis. Moderate -> Proliferative)
# karena QWK menghukum kesalahan tersebut secara kuadratik.
LOSS_DISTANCE_POWER = 2.0      # 2.0 = penalti kuadratik (selaras dgn definisi QWK), coba 1.0/1.5 jika tidak stabil
LOSS_SMOOTHING_STRENGTH = 0.1  # setara label_smoothing yang dipakai sebelumnya
LOSS_ORDINAL_WEIGHT = 0.4      # bobot komponen regresi ordinal; mulai 0.4, tuning range 0.2-0.6

# ============================================================
# MIXED PRECISION TRAINING
# ============================================================
USE_AMP = True if DEVICE.type == "cuda" else False

# ============================================================
# LABEL KELAS RETINOPATI DIABETIK
# ============================================================
CLASS_NAMES = [
    "No DR",            # 0 - Tidak ada retinopati diabetik
    "Mild",             # 1 - Ringan (NPDR ringan)
    "Moderate",         # 2 - Sedang (NPDR sedang)
    "Severe",           # 3 - Berat (NPDR berat)
    "Proliferative"     # 4 - Proliferatif (PDR)
]

CLASS_LABELS = {i: name for i, name in enumerate(CLASS_NAMES)}

# ============================================================
# EXPLAINABILITY SETTINGS
# ============================================================
EXPLAIN_NUM_SAMPLES = 5        # Jumlah sampel per kelas untuk explainability
EXPLAIN_IG_STEPS = 50          # Jumlah steps untuk Integrated Gradients
EXPLAIN_COLORMAP = "jet"       # Colormap untuk heatmap

# ============================================================
# IMAGENET NORMALIZATION
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# RANDOM SEED (Reproducibility)
# ============================================================
SEED = 42

# ============================================================
# VALIDASI KONFIGURASI
# ============================================================
def validate_config():
    """Validasi bahwa semua path dan konfigurasi sudah benar."""
    errors = []
    
    if not os.path.exists(DATASET_DIR):
        errors.append(f"Dataset directory tidak ditemukan: {DATASET_DIR}")
    if not os.path.exists(TRAIN_IMG_DIR):
        errors.append(f"Train image directory tidak ditemukan: {TRAIN_IMG_DIR}")
    if not os.path.exists(VAL_IMG_DIR):
        errors.append(f"Validation image directory tidak ditemukan: {VAL_IMG_DIR}")
    if not os.path.exists(TRAIN_CSV):
        errors.append(f"Train CSV tidak ditemukan: {TRAIN_CSV}")
    if not os.path.exists(VAL_CSV):
        errors.append(f"Validation CSV tidak ditemukan: {VAL_CSV}")
    
    if errors:
        print("\n[CONFIG ERROR] Konfigurasi bermasalah:")
        for e in errors:
            print(f"  [X] {e}")
        return False
    
    print("\n[CONFIG] [OK] Semua path tervalidasi")
    print(f"[CONFIG] Dataset: {DATASET_DIR}")
    print(f"[CONFIG] Model: {MODEL_NAME} (pretrained={PRETRAINED})")
    print(f"[CONFIG] Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"[CONFIG] Batch size: {BATCH_SIZE}")
    print(f"[CONFIG] Epochs: {EPOCHS}")
    print(f"[CONFIG] Learning rate: {LEARNING_RATE}")
    print(f"[CONFIG] Mixed Precision: {USE_AMP}")
    return True


if __name__ == "__main__":
    validate_config()