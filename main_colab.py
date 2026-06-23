"""
main_colab.py - Entry Point untuk Google Colab
Deteksi Retinopati Diabetik dengan MViTv2 + Explainability Hibrida

Jalankan di Google Colab Notebook (GPU):
    # Clone repo & masuk ke folder
    !git clone <REPO_URL> /content/code
    %cd /content/code
    !python main_colab.py --source kaggle --mode train

Sumber dataset yang didukung:
  --source kaggle  : Download otomatis dari Kaggle (autentikasi via OAuth login)
  --source drive   : Mount Google Drive & gunakan dataset dari folder Drive

Mode yang tersedia:
  --mode train      : Melatih model dari awal atau melanjutkan training
  --mode evaluate   : Mengevaluasi model pada test set
  --mode explain    : Menjalankan analisis explainability
  --mode test_data  : Memverifikasi dataloader dan preprocessing
"""

import argparse
import os
import sys
import subprocess
import glob

# ============================================================
# COLAB PATH CONFIGURATION
# ============================================================

COLAB_WORKING_DIR = "/content"
COLAB_DATASET_BASE = os.path.join(COLAB_WORKING_DIR, "dataset")

# Google Drive default path (sesuaikan jika berbeda)
GDRIVE_MOUNT_POINT = "/content/drive"
GDRIVE_DATASET_DEFAULT = os.path.join(
    GDRIVE_MOUNT_POINT, "MyDrive", "datasets", "aptos2019"
)

# Kaggle dataset identifier
KAGGLE_DATASET_SLUG = "mariaherrerot/aptos2019"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _is_colab():
    """Deteksi apakah sedang berjalan di Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _install_dependencies():
    """Install dependensi yang belum tersedia di Colab runtime."""
    print("\n[COLAB] Memeriksa dan menginstall dependensi...")
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_file]
        )
        print("[COLAB] Dependensi dari requirements.txt terinstall.")
    else:
        # Fallback: install paket esensial
        essentials = ["timm", "scikit-learn", "captum", "opencv-python-headless"]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + essentials
        )
        print("[COLAB] Paket esensial terinstall.")


def _first_existing(*paths):
    """Kembalikan path pertama yang ada di filesystem."""
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _resolve_image_dir(dataset_dir, split):
    """
    Temukan folder gambar untuk split train/val/test.
    Mendukung struktur nested (train_images/train_images/) dan flat.
    """
    split_map = {"train": "train", "val": "val", "test": "test"}
    prefix = split_map.get(split, split)

    candidates = [
        os.path.join(dataset_dir, f"{prefix}_images", f"{prefix}_images"),
        os.path.join(dataset_dir, f"{prefix}_images"),
    ]
    found = _first_existing(*candidates)
    if found:
        return found

    # Fallback: semua gambar di satu folder (train_images)
    fallback = os.path.join(dataset_dir, "train_images")
    if os.path.isdir(fallback):
        return fallback

    return candidates[0]


def _resolve_csv(dataset_dir, split):
    """Temukan file CSV untuk split train/val/test."""
    csv_candidates = {
        "train": ["train_1.csv", "train.csv"],
        "val": ["valid.csv", "val.csv", "validation.csv"],
        "test": ["test.csv"],
    }
    for name in csv_candidates.get(split, []):
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(dataset_dir, csv_candidates[split][0])


# ============================================================
# DATASET SOURCE HANDLERS
# ============================================================

def setup_dataset_kaggle():
    """
    Download dataset APTOS 2019 dari Kaggle menggunakan Kaggle API.
    Autentikasi via OAuth login (kaggle auth login) — tidak perlu upload kaggle.json.
    """
    dataset_dir = COLAB_DATASET_BASE

    if os.path.isdir(dataset_dir) and any(
        f.endswith(".csv") for f in os.listdir(dataset_dir)
    ):
        print(f"[COLAB] Dataset sudah ada di {dataset_dir}, skip download.")
        return dataset_dir

    # ---- Install kaggle CLI terlebih dahulu ----
    print("[COLAB] Menginstall Kaggle CLI...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "kaggle"]
    )

    # ---- Cek apakah sudah terautentikasi ----
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        print("\n[COLAB] Kaggle belum terautentikasi.")
        print("[COLAB] Menjalankan 'kaggle auth login' — ikuti instruksi di browser...\n")
        try:
            subprocess.check_call(["kaggle", "auth", "login"])
            print("\n[COLAB] Autentikasi Kaggle berhasil!")
        except subprocess.CalledProcessError:
            print("\n[ERROR] Autentikasi Kaggle gagal.")
            print("  Pastikan Anda mengikuti instruksi OAuth di browser.")
            print("  Alternatif: letakkan kaggle.json secara manual di ~/.kaggle/kaggle.json")
            sys.exit(1)
    else:
        print(f"[COLAB] Kaggle sudah terautentikasi (ditemukan {kaggle_json}).")

    # ---- Download dataset ----
    print(f"\n[COLAB] Mendownload dataset '{KAGGLE_DATASET_SLUG}'...")
    os.makedirs(dataset_dir, exist_ok=True)
    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET_SLUG,
        "-p", dataset_dir,
        "--unzip",
    ])

    print(f"[COLAB] Dataset berhasil didownload ke {dataset_dir}")

    # Cek apakah ada subfolder hasil unzip (kadang Kaggle buat nested folder)
    _flatten_if_nested(dataset_dir)

    return dataset_dir


def setup_dataset_drive(drive_path=None):
    """
    Gunakan dataset dari Google Drive.
    Mount Drive jika belum ter-mount, lalu gunakan path yang diberikan.
    """
    # Mount Google Drive jika di Colab dan belum ter-mount
    if _is_colab() and not os.path.ismount(GDRIVE_MOUNT_POINT):
        print("[COLAB] Mounting Google Drive...")
        from google.colab import drive

        drive.mount(GDRIVE_MOUNT_POINT)
        print("[COLAB] Google Drive berhasil di-mount.")

    # Tentukan path dataset
    dataset_dir = drive_path or GDRIVE_DATASET_DEFAULT

    if not os.path.isdir(dataset_dir):
        print(f"\n[ERROR] Folder dataset tidak ditemukan: {dataset_dir}")
        print("  Pastikan dataset APTOS 2019 sudah ada di Google Drive Anda.")
        print(f"  Path default: {GDRIVE_DATASET_DEFAULT}")
        print("  Atau gunakan --drive_path untuk menentukan path yang benar.")
        print("\n  Contoh struktur folder yang diharapkan:")
        print(f"    {dataset_dir}/")
        print(f"    ├── train_images/")
        print(f"    │   └── train_images/   (opsional, nested)")
        print(f"    ├── train_1.csv")
        print(f"    ├── valid.csv")
        print(f"    └── test.csv")
        sys.exit(1)

    print(f"[COLAB] Menggunakan dataset dari Google Drive: {dataset_dir}")
    return dataset_dir


def _flatten_if_nested(dataset_dir):
    """
    Jika hasil unzip menghasilkan satu subfolder saja (mis. aptos2019/),
    pindahkan semua isinya ke dataset_dir langsung.
    """
    entries = os.listdir(dataset_dir)
    if len(entries) == 1:
        single = os.path.join(dataset_dir, entries[0])
        if os.path.isdir(single):
            print(f"[COLAB] Flatten nested folder: {entries[0]}/")
            import shutil

            for item in os.listdir(single):
                src = os.path.join(single, item)
                dst = os.path.join(dataset_dir, item)
                shutil.move(src, dst)
            os.rmdir(single)


# ============================================================
# COLAB CONFIG SETUP
# ============================================================

def setup_colab_config(dataset_dir):
    """Override config.py dengan path dan setting khusus Colab."""
    import config

    code_dir = os.path.dirname(os.path.abspath(__file__))

    config.BASE_DIR = code_dir
    config.DATASET_DIR = dataset_dir

    config.TRAIN_IMG_DIR = _resolve_image_dir(dataset_dir, "train")
    config.VAL_IMG_DIR = _resolve_image_dir(dataset_dir, "val")
    config.TEST_IMG_DIR = _resolve_image_dir(dataset_dir, "test")

    config.TRAIN_CSV = _resolve_csv(dataset_dir, "train")
    config.VAL_CSV = _resolve_csv(dataset_dir, "val")
    config.TEST_CSV = _resolve_csv(dataset_dir, "test")

    config.OUTPUT_DIR = os.path.join(COLAB_WORKING_DIR, "outputs")
    config.CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "checkpoints")
    config.PLOT_DIR = os.path.join(config.OUTPUT_DIR, "plots")
    config.METRICS_DIR = os.path.join(config.OUTPUT_DIR, "metrics")
    config.EXPLAIN_DIR = os.path.join(config.OUTPUT_DIR, "explainability")

    for d in [
        config.OUTPUT_DIR,
        config.CHECKPOINT_DIR,
        config.PLOT_DIR,
        config.METRICS_DIR,
        config.EXPLAIN_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # Optimasi untuk lingkungan Colab GPU
    import torch

    config.NUM_WORKERS = 2  # Colab stabil dengan 2 workers
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        if gpu_mem >= 15:  # A100 / V100 (Colab Pro)
            config.BATCH_SIZE = 16
        else:  # T4 (Colab gratis)
            config.BATCH_SIZE = 8
        config.USE_AMP = True
    else:
        config.BATCH_SIZE = 4
        config.USE_AMP = False

    return config


def print_colab_info(cfg):
    """Cetak ringkasan konfigurasi Colab."""
    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n" + "=" * 60)
    print("GOOGLE COLAB ENVIRONMENT")
    print("=" * 60)
    print(f"  Runtime GPU : {gpu_name}")
    print(f"  Dataset dir : {cfg.DATASET_DIR}")
    print(f"  Train CSV   : {cfg.TRAIN_CSV}")
    print(f"  Val CSV     : {cfg.VAL_CSV}")
    print(f"  Test CSV    : {cfg.TEST_CSV}")
    print(f"  Train imgs  : {cfg.TRAIN_IMG_DIR}")
    print(f"  Val imgs    : {cfg.VAL_IMG_DIR}")
    print(f"  Test imgs   : {cfg.TEST_IMG_DIR}")
    print(f"  Output dir  : {cfg.OUTPUT_DIR}")
    print(f"  Batch size  : {cfg.BATCH_SIZE}")
    print(f"  Num workers : {cfg.NUM_WORKERS}")
    print(f"  AMP         : {cfg.USE_AMP}")
    print("=" * 60 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MViTv2 Diabetic Retinopathy Detection (Google Colab)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["kaggle", "drive"],
        help="Sumber dataset: 'kaggle' (download dari Kaggle) atau 'drive' (Google Drive)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "explain", "test_data"],
        help="Mode eksekusi: train, evaluate, explain, atau test_data",
    )
    parser.add_argument(
        "--drive_path",
        type=str,
        default=None,
        help=(
            "Path kustom ke folder dataset di Google Drive "
            f"(default: {GDRIVE_DATASET_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path ke checkpoint untuk melanjutkan training atau evaluasi",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Jumlah sampel per kelas untuk mode explain",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path ke satu gambar spesifik untuk dianalisis (mode explain)",
    )
    parser.add_argument(
        "--skip_install",
        action="store_true",
        help="Skip auto-install dependensi (jika sudah diinstall manual)",
    )

    args = parser.parse_args()

    # ---- 1. Install dependensi ----
    if not args.skip_install:
        _install_dependencies()

    # ---- 2. Siapkan dataset berdasarkan sumber ----
    if args.source == "kaggle":
        dataset_dir = setup_dataset_kaggle()
    elif args.source == "drive":
        dataset_dir = setup_dataset_drive(drive_path=args.drive_path)

    # ---- 3. Override config ----
    cfg = setup_colab_config(dataset_dir)
    print_colab_info(cfg)

    # ---- 4. Validasi ----
    from config import validate_config

    if not validate_config():
        print("\n[COLAB] Validasi gagal. Isi folder dataset:")
        if os.path.isdir(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                depth = root.replace(dataset_dir, "").count(os.sep)
                if depth > 2:
                    continue
                indent = "  " * depth
                print(f"{indent}{os.path.basename(root)}/")
                for f in files[:10]:
                    print(f"{indent}  {f}")
                if len(files) > 10:
                    print(f"{indent}  ... ({len(files) - 10} file lainnya)")
        else:
            print(f"  Dataset tidak ditemukan: {dataset_dir}")
        sys.exit(1)

    # ---- 5. Eksekusi mode ----
    if args.mode == "train":
        from train import train

        train(resume_checkpoint=args.resume)

    elif args.mode == "evaluate":
        from evaluate import evaluate

        evaluate(checkpoint_path=args.resume)

    elif args.mode == "explain":
        from explainability import run_explainability, explain_single_image

        if args.image:
            explain_single_image(args.image, checkpoint_path=args.resume)
        else:
            run_explainability(
                checkpoint_path=args.resume,
                num_samples_per_class=args.num_samples,
            )

    elif args.mode == "test_data":
        from dataset import get_dataloaders

        print("\nMenjalankan test DataLoader...")
        get_dataloaders()
        print("\nTest DataLoader selesai!")


if __name__ == "__main__":
    main()
