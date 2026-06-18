"""
main_kaggle.py - Entry Point untuk Kaggle
Deteksi Retinopati Diabetik dengan MViTv2 + Explainability Hibrida

Jalankan di Kaggle Notebook (GPU):
    !cd /kaggle/working/code && python main_kaggle.py --mode train

Dataset: /kaggle/input/datasets/mariaherrerot/aptos2019

Mode yang tersedia:
- train: Melatih model dari awal atau melanjutkan training
- evaluate: Mengevaluasi model pada test set
- explain: Menjalankan analisis explainability
- test_data: Memverifikasi dataloader dan preprocessing
"""

import argparse
import os
import sys

# ============================================================
# KAGGLE PATH CONFIGURATION
# Harus dijalankan SEBELUM import modul lain (train, dataset, dll.)
# ============================================================

KAGGLE_DATASET_DIR = "/kaggle/input/datasets/mariaherrerot/aptos2019"
KAGGLE_WORKING_DIR = "/kaggle/working"


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


def setup_kaggle_config():
    """Override config.py dengan path dan setting khusus Kaggle."""
    import config

    code_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = KAGGLE_WORKING_DIR if os.path.isdir(KAGGLE_WORKING_DIR) else code_dir

    config.BASE_DIR = code_dir
    config.DATASET_DIR = KAGGLE_DATASET_DIR

    config.TRAIN_IMG_DIR = _resolve_image_dir(KAGGLE_DATASET_DIR, "train")
    config.VAL_IMG_DIR = _resolve_image_dir(KAGGLE_DATASET_DIR, "val")
    config.TEST_IMG_DIR = _resolve_image_dir(KAGGLE_DATASET_DIR, "test")

    config.TRAIN_CSV = _resolve_csv(KAGGLE_DATASET_DIR, "train")
    config.VAL_CSV = _resolve_csv(KAGGLE_DATASET_DIR, "val")
    config.TEST_CSV = _resolve_csv(KAGGLE_DATASET_DIR, "test")

    config.OUTPUT_DIR = os.path.join(working_dir, "outputs")
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

    # Optimasi untuk lingkungan Kaggle GPU
    import torch

    config.NUM_WORKERS = 8
    if torch.cuda.is_available():
        config.BATCH_SIZE = 16
        config.USE_AMP = True
    else:
        config.BATCH_SIZE = 8
        config.USE_AMP = False

    return config


def print_kaggle_info(cfg):
    """Cetak ringkasan konfigurasi Kaggle."""
    print("\n" + "=" * 60)
    print("KAGGLE ENVIRONMENT")
    print("=" * 60)
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
    print("=" * 60 + "\n")


def main():
    cfg = setup_kaggle_config()
    print_kaggle_info(cfg)

    from config import validate_config

    parser = argparse.ArgumentParser(
        description="MViTv2 Diabetic Retinopathy Detection (Kaggle)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "explain", "test_data"],
        help="Mode eksekusi: train, evaluate, explain, atau test_data",
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

    args = parser.parse_args()

    if not validate_config():
        print("\n[KAGGLE] Validasi gagal. Isi folder dataset:")
        if os.path.isdir(KAGGLE_DATASET_DIR):
            for root, dirs, files in os.walk(KAGGLE_DATASET_DIR):
                depth = root.replace(KAGGLE_DATASET_DIR, "").count(os.sep)
                if depth > 2:
                    continue
                indent = "  " * depth
                print(f"{indent}{os.path.basename(root)}/")
                for f in files[:10]:
                    print(f"{indent}  {f}")
                if len(files) > 10:
                    print(f"{indent}  ... ({len(files) - 10} file lainnya)")
        else:
            print(f"  Dataset tidak ditemukan: {KAGGLE_DATASET_DIR}")
            print("  Tambahkan dataset 'mariaherrerot/aptos2019' ke notebook Kaggle.")
        sys.exit(1)

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
