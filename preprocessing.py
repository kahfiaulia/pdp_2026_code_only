"""
preprocessing.py - Praproses Citra Fundus Retina
Deteksi Retinopati Diabetik dengan MViTv2

Fungsi-fungsi praproses:
1. Crop area hitam di sekitar gambar retina
2. Peningkatan kontras dengan CLAHE
3. Ben Graham preprocessing untuk normalisasi citra fundus
"""

import cv2
import numpy as np
from PIL import Image


def crop_image_from_gray(img, tol=7):
    """
    Crop area hitam/gelap di sekitar gambar retina.
    
    Citra fundus retina biasanya memiliki area hitam di sekitar 
    lingkaran retina. Fungsi ini mendeteksi dan memotong area tersebut.
    
    Args:
        img (numpy.ndarray): Gambar input (BGR atau RGB)
        tol (int): Threshold toleransi untuk area gelap (default=7)
    
    Returns:
        numpy.ndarray: Gambar yang sudah di-crop
    """
    if img.ndim == 2:
        # Gambar grayscale
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        # Gambar berwarna - gunakan grayscale untuk membuat mask
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        # Cek apakah mask valid
        if not mask.any():
            return img
            
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        
        return img


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE meningkatkan kontras lokal pada citra, sehingga struktur retina 
    seperti pembuluh darah, mikroaneurisma, dan eksudat menjadi lebih jelas.
    
    Args:
        img (numpy.ndarray): Gambar input RGB
        clip_limit (float): Batas clipping untuk kontras (default=2.0)
        tile_grid_size (tuple): Ukuran grid untuk area adaptif (default=(8,8))
    
    Returns:
        numpy.ndarray: Gambar dengan kontras yang ditingkatkan
    """
    # Konversi ke LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Terapkan CLAHE pada channel L (lightness)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Konversi kembali ke RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def ben_graham_preprocessing(img, sigmaX=10):
    """
    Teknik Ben Graham untuk normalisasi citra fundus retina.
    
    Metode ini mengurangi variasi pencahayaan pada citra fundus dengan 
    mengurangi komponen pencahayaan global (Gaussian blur) dari gambar asli,
    kemudian menambahkan nilai konstan untuk mempertahankan kecerahan.
    
    Referensi: Ben Graham, Kaggle Diabetic Retinopathy Competition Winner
    
    Args:
        img (numpy.ndarray): Gambar input RGB
        sigmaX (int): Sigma untuk Gaussian blur (default=10)
    
    Returns:
        numpy.ndarray: Gambar yang sudah dinormalisasi
    """
    # Terapkan weighted addition: gambar asli - Gaussian blur + 128
    processed = cv2.addWeighted(
        img, 4,                                          # Gambar asli * 4
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,      # Gaussian blur * (-4)
        128                                               # Offset konstan
    )
    
    return processed


def preprocess_fundus_image(img, img_size=224, apply_crop=True, 
                            apply_contrast=True, use_ben_graham=False):
    """
    Pipeline praproses lengkap untuk citra fundus retina.
    
    Urutan praproses:
    1. Crop area hitam (opsional)
    2. Resize ke ukuran target
    3. Peningkatan kontras CLAHE atau Ben Graham (opsional)
    
    Args:
        img (numpy.ndarray): Gambar input RGB
        img_size (int): Ukuran target (default=224)
        apply_crop (bool): Terapkan cropping area hitam
        apply_contrast (bool): Terapkan peningkatan kontras CLAHE
        use_ben_graham (bool): Gunakan Ben Graham preprocessing
    
    Returns:
        numpy.ndarray: Gambar yang sudah diproses
    """
    # Step 1: Crop area hitam
    if apply_crop:
        img = crop_image_from_gray(img)
    
    # Step 2: Resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # Step 3: Peningkatan kontras
    if use_ben_graham:
        img = ben_graham_preprocessing(img)
    elif apply_contrast:
        img = apply_clahe(img)
    
    return img


def load_and_preprocess(image_path, img_size=224):
    """
    Load dan preprocess satu gambar dari path.
    
    Args:
        image_path (str): Path ke file gambar
        img_size (int): Ukuran target
    
    Returns:
        numpy.ndarray: Gambar yang sudah diproses (RGB)
    """
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    
    # Konversi BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Terapkan praproses
    img = preprocess_fundus_image(img, img_size=img_size)
    
    return img


if __name__ == "__main__":
    """Test preprocessing pada satu gambar."""
    import os
    import matplotlib.pyplot as plt
    from config import TRAIN_IMG_DIR, TRAIN_CSV, IMG_SIZE
    import pandas as pd
    
    # Load satu gambar untuk test
    df = pd.read_csv(TRAIN_CSV)
    sample = df.iloc[0]
    img_path = os.path.join(TRAIN_IMG_DIR, f"{sample['id_code']}.png")
    
    # Load gambar original
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    processed = load_and_preprocess(img_path, img_size=IMG_SIZE)
    
    # Visualisasi
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original)
    axes[0].set_title(f"Original - Label: {sample['diagnosis']}")
    axes[0].axis("off")
    
    axes[1].imshow(processed)
    axes[1].set_title("Preprocessed (Crop + CLAHE)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "outputs", "plots", "preprocessing_test.png"), dpi=150)
    plt.show()
    print("Preprocessing test selesai!")
