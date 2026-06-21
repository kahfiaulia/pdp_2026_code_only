"""
dataset.py - Custom PyTorch Dataset untuk APTOS 2019
Deteksi Retinopati Diabetik dengan MViTv2

Menangani:
- Loading citra fundus retina dari CSV + folder gambar
- Praproses (crop, CLAHE, resize)
- Augmentasi data (training only)
- Normalisasi ImageNet
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR,
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
    CLASS_WEIGHT_BETA
)
from preprocessing import preprocess_fundus_image


class DRDataset(Dataset):
    """
    Dataset untuk Diabetic Retinopathy Classification.
    
    Memuat citra fundus retina dan label dari CSV file.
    Mendukung augmentasi dan praproses khusus retina.
    
    Args:
        csv_file (str): Path ke CSV file (kolom: id_code, diagnosis)
        img_dir (str): Path ke direktori gambar
        transform (albumentations.Compose): Transformasi augmentasi
        is_training (bool): Mode training (aktifkan augmentasi)
        apply_preprocessing (bool): Terapkan praproses retina (crop + CLAHE)
    """
    
    def __init__(self, csv_file, img_dir, transform=None, 
                 is_training=False, apply_preprocessing=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_training = is_training
        self.apply_preprocessing = apply_preprocessing
        
        # Filter baris kosong
        self.df = self.df.dropna(subset=["id_code", "diagnosis"])
        self.df = self.df.reset_index(drop=True)
        
        # Konversi label ke integer
        self.df["diagnosis"] = self.df["diagnosis"].astype(int)
        
        print(f"[DATASET] Loaded {len(self.df)} samples dari {csv_file}")
        print(f"[DATASET] Distribusi kelas:")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            count = (self.df["diagnosis"] == cls_idx).sum()
            print(f"  Kelas {cls_idx} ({cls_name}): {count} gambar")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Mengambil satu sampel (gambar, label).
        
        Returns:
            tuple: (image_tensor, label_tensor)
        """
        row = self.df.iloc[idx]
        img_name = row["id_code"]
        label = int(row["diagnosis"])
        
        # Load gambar
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Praproses retina (crop area hitam + CLAHE)
        if self.apply_preprocessing:
            img = preprocess_fundus_image(
                img, img_size=IMG_SIZE, 
                apply_crop=True, apply_contrast=True
            )
        else:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Augmentasi (Albumentations)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        
        return img, torch.tensor(label, dtype=torch.long)


def get_train_transforms():
    """
    Transformasi augmentasi untuk data training.
    
    Augmentasi yang diterapkan:
    - Horizontal & Vertical Flip
    - Rotasi ±30 derajat
    - Perubahan brightness & contrast
    - Shift, Scale, Rotate
    - Coarse Dropout (cutout)
    - Normalisasi ImageNet
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=0.1,
            scale=(0.85, 1.15),
            rotate=(-30, 30),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(IMG_SIZE // 10, IMG_SIZE // 10),
            hole_width_range=(IMG_SIZE // 10, IMG_SIZE // 10),
            fill=0,
            p=0.3
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Transformasi untuk data validasi/test (tanpa augmentasi).
    Hanya normalisasi ImageNet.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_dataloaders():
    """
    Membuat DataLoader untuk train, validation, dan test.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Dataset
    train_dataset = DRDataset(
        csv_file=TRAIN_CSV,
        img_dir=TRAIN_IMG_DIR,
        transform=get_train_transforms(),
        is_training=True
    )
    
    val_dataset = DRDataset(
        csv_file=VAL_CSV,
        img_dir=VAL_IMG_DIR,
        transform=get_val_transforms(),
        is_training=False
    )
    
    test_dataset = DRDataset(
        csv_file=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=get_val_transforms(),
        is_training=False
    )
    
    # Hitung sample weights untuk WeightedRandomSampler
    class_counts = train_dataset.df["diagnosis"].value_counts().sort_index()
    class_weights_arr = 1.0 / class_counts.values
    sample_weights = [class_weights_arr[label] for label in train_dataset.df["diagnosis"]]
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\n[DATALOADER] Train: {len(train_dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"[DATALOADER] Val:   {len(val_dataset)} samples, "
          f"{len(val_loader)} batches")
    print(f"[DATALOADER] Test:  {len(test_dataset)} samples, "
          f"{len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def compute_class_weights(csv_file, beta=CLASS_WEIGHT_BETA):
    """
    Menghitung class weights untuk menangani class imbalance.
    
    Menggunakan Effective Number of Samples (Class-Balanced Loss, Cui et al., 2019):
    effective_num = 1.0 - beta^count
    weights = (1.0 - beta) / effective_num

    CATATAN PENTING soal pemilihan beta:
    Sensitivitas formula ini terhadap beta sangat tergantung skala n (jumlah sampel
    per kelas). Untuk dataset sekelas APTOS (n berkisar ~150-1500), beta=0.9999
    (default umum di literatur untuk dataset besar seperti ImageNet/iNaturalist)
    menghasilkan weight dengan gradasi tajam (~0.2x - 2.0x). Begitu beta diturunkan
    ke 0.99 ke bawah, beta^n untuk n>~150 sudah mendekati 0, sehingga SEMUA weight
    collapse ke ~1.0 (efek weighting hilang total, bukan sekadar melemah).
    
    beta=0.99 adalah titik tengah yang masih punya gradasi berarti (~0.9x - 1.2x)
    namun jauh lebih halus dari 0.9999 -- cocok dipakai bersamaan dengan
    WeightedRandomSampler di get_dataloaders() supaya tidak terjadi double
    compensation (sampler menangani exposure frequency, loss weight cukup jadi
    penyeimbang ringan).
    
    Args:
        csv_file (str): Path ke CSV training data
        beta (float): Parameter ENS, default 0.99 (lihat catatan di atas).
            Gunakan 0.9999 jika TIDAK memakai WeightedRandomSampler di dataloader.
    
    Returns:
        torch.Tensor: Tensor class weights
    """
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["diagnosis"])
    
    class_counts = df["diagnosis"].value_counts().sort_index()
    num_classes = len(class_counts)
    
    # Class-Balanced Loss formula
    counts = np.array([class_counts.get(i, 1) for i in range(num_classes)])
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    
    # Normalize weights agar jumlahnya sama dengan num_classes
    weights = weights / np.sum(weights) * num_classes
    
    weights_tensor = torch.FloatTensor(weights)
    
    print(f"\n[CLASS WEIGHTS] beta={beta}")
    print(f"[CLASS WEIGHTS] Class distribution & weights:")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        count = class_counts.get(i, 0)
        print(f"  Kelas {i} ({name}): {count} samples, weight={w:.4f}")
    
    return weights_tensor


if __name__ == "__main__":
    """Test dataset loading."""
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("TEST DATASET LOADING")
    print("=" * 60)
    
    # Test class weights
    weights = compute_class_weights(TRAIN_CSV)
    
    # Test dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Visualisasi batch pertama
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Denormalisasi untuk visualisasi
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            img = images[i] * std + mean  # Denormalisasi
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"Label: {labels[i].item()} ({CLASS_NAMES[labels[i].item()]})")
        ax.axis("off")
    
    plt.suptitle("Sample Training Batch", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "outputs", "plots", "sample_batch.png"), dpi=150)
    plt.show()
    print("\nDataset test selesai!")