"""
utils.py - Fungsi Utilitas
Deteksi Retinopati Diabetik dengan MViTv2

Berisi fungsi-fungsi bantu:
- Seed setting (reproducibility)
- Checkpoint management
- Plotting utilities
- Logging
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seed untuk reproducibility.
    
    Args:
        seed (int): Nilai seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[UTILS] Random seed set to {seed}")


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, 
                    val_accuracy, val_qwk=None, filepath=None):
    """
    Simpan model checkpoint.
    
    Args:
        model: Model PyTorch
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch (int): Epoch saat ini
        val_loss (float): Validation loss
        val_accuracy (float): Validation accuracy
        val_qwk (float, optional): Validation Quadratic Weighted Kappa
        filepath (str): Path untuk menyimpan checkpoint
    """
    # PENTING: paksa semua nilai numerik jadi Python primitif murni (int/float),
    # bukan numpy scalar (np.float64, np.int64, dst). PyTorch >=2.6 default
    # torch.load(weights_only=True) MENOLAK unpickle numpy scalar di dalam
    # checkpoint dict (error "Unsupported global: numpy._core.multiarray.scalar"),
    # meski isinya cuma angka biasa. int()/float() di sini menghindarinya sejak awal.
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "val_qwk": float(val_qwk) if val_qwk is not None else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(checkpoint, filepath)
    qwk_str = f" | Val QWK: {val_qwk:.4f}" if val_qwk is not None else ""
    print(f"[CHECKPOINT] Saved: {filepath}{qwk_str}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path ke checkpoint
        model: Model PyTorch
        optimizer: Optimizer (opsional)
        scheduler: Scheduler (opsional)
        device: Device target
    
    Returns:
        dict: Informasi checkpoint
    """
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    except Exception as e:
        # Checkpoint lama (sebelum fix numpy-scalar) bisa berisi tipe seperti
        # numpy.float64 yang ditolak weights_only=True di PyTorch >=2.6.
        # Fallback ke weights_only=False -- AMAN di sini karena file ini hasil
        # save_checkpoint() kita sendiri, bukan checkpoint dari sumber pihak ketiga.
        print(f"[CHECKPOINT] weights_only=True gagal ({type(e).__name__}), "
              f"mencoba ulang dengan weights_only=False (checkpoint lama)...")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    print(f"[CHECKPOINT] Loaded: {filepath}")
    qwk_val = checkpoint.get("val_qwk")
    qwk_str = f", Val QWK: {qwk_val:.4f}" if qwk_val is not None else ""
    print(f"[CHECKPOINT] Epoch: {checkpoint['epoch']}, "
          f"Val Loss: {checkpoint['val_loss']:.4f}, "
          f"Val Acc: {checkpoint['val_accuracy']:.4f}{qwk_str}")
    
    return checkpoint


class EarlyStopping:
    """
    Early Stopping untuk menghentikan training ketika
    validation loss tidak membaik.
    
    Args:
        patience (int): Jumlah epoch untuk menunggu perbaikan
        min_delta (float): Minimum perubahan yang dianggap perbaikan
        mode (str): 'min' untuk loss, 'max' untuk accuracy
    """
    
    def __init__(self, patience=7, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n[EARLY STOPPING] Triggered after {self.counter} epochs "
                      f"without improvement")
                return True
        
        return False


class MetricsTracker:
    """
    Tracker untuk menyimpan metrik training per epoch.
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_qwks = []
        self.learning_rates = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, lr, val_qwk=None):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.train_accuracies.append(float(train_acc))
        self.val_accuracies.append(float(val_acc))
        self.learning_rates.append(float(lr))
        # Pakai NaN kalau qwk tidak diberikan, supaya index tetap selaras
        # dengan list lain meski dipanggil tanpa val_qwk di beberapa tempat.
        self.val_qwks.append(float(val_qwk) if val_qwk is not None else float("nan"))
    
    def get_best_epoch(self):
        """
        Mendapatkan epoch terbaik.
        Kalau val_qwk tersedia (tidak semuanya NaN), pakai QWK tertinggi
        sebagai kriteria utama -- ini konsisten dengan kriteria pemilihan
        best_model.pth di train.py. Kalau tidak ada data QWK sama sekali
        (training lama / belum pakai ordinal loss), fallback ke val_loss terendah.
        """
        qwk_array = np.array(self.val_qwks)
        has_qwk = len(qwk_array) > 0 and not np.all(np.isnan(qwk_array))

        if has_qwk:
            best_idx = int(np.nanargmax(qwk_array))
        else:
            best_idx = int(np.argmin(self.val_losses))

        return {
            "epoch": best_idx + 1,
            "val_accuracy": float(self.val_accuracies[best_idx]),
            "val_loss": float(self.val_losses[best_idx]),
            "val_qwk": float(qwk_array[best_idx]) if has_qwk else None,
            "train_accuracy": float(self.train_accuracies[best_idx]),
            "train_loss": float(self.train_losses[best_idx])
        }


def plot_training_curves(metrics_tracker, save_path):
    """
    Plot kurva training (loss, accuracy, QWK jika tersedia, dan learning rate).
    
    Args:
        metrics_tracker (MetricsTracker): Objek tracker metrik
        save_path (str): Path untuk menyimpan plot
    """
    qwk_array = np.array(getattr(metrics_tracker, "val_qwks", []))
    has_qwk = len(qwk_array) > 0 and not np.all(np.isnan(qwk_array))

    n_plots = 4 if has_qwk else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))

    epochs = range(1, len(metrics_tracker.train_losses) + 1)
    
    # Plot Loss
    axes[0].plot(epochs, metrics_tracker.train_losses, 'b-o', 
                 label='Training Loss', markersize=3)
    axes[0].plot(epochs, metrics_tracker.val_losses, 'r-o', 
                 label='Validation Loss', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(epochs, metrics_tracker.train_accuracies, 'b-o', 
                 label='Training Accuracy', markersize=3)
    axes[1].plot(epochs, metrics_tracker.val_accuracies, 'r-o', 
                 label='Validation Accuracy', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    next_idx = 2

    # Plot QWK (hanya kalau datanya ada)
    if has_qwk:
        axes[next_idx].plot(epochs, qwk_array, 'm-o',
                     label='Validation QWK', markersize=3)
        best_idx = int(np.nanargmax(qwk_array))
        axes[next_idx].scatter([best_idx + 1], [qwk_array[best_idx]],
                        color='gold', s=80, zorder=5, edgecolor='black',
                        label=f'Best: {qwk_array[best_idx]:.4f} (epoch {best_idx+1})')
        axes[next_idx].set_xlabel('Epoch')
        axes[next_idx].set_ylabel('QWK')
        axes[next_idx].set_title('Validation QWK')
        axes[next_idx].legend()
        axes[next_idx].grid(True, alpha=0.3)
        next_idx += 1
    
    # Plot Learning Rate
    axes[next_idx].plot(epochs, metrics_tracker.learning_rates, 'g-o', markersize=3)
    axes[next_idx].set_xlabel('Epoch')
    axes[next_idx].set_ylabel('Learning Rate')
    axes[next_idx].set_title('Learning Rate Schedule')
    axes[next_idx].grid(True, alpha=0.3)
    axes[next_idx].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.suptitle('MViTv2 - Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] Training curves saved: {save_path}")


def plot_class_distribution(csv_file, class_names, save_path):
    """
    Visualisasi distribusi kelas pada dataset.
    
    Args:
        csv_file (str): Path ke CSV file
        class_names (list): Daftar nama kelas
        save_path (str): Path untuk menyimpan plot
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["diagnosis"])
    
    counts = df["diagnosis"].value_counts().sort_index()
    
    colors = sns.color_palette("viridis", len(class_names))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(class_names)), 
                  [counts.get(i, 0) for i in range(len(class_names))],
                  color=colors, edgecolor='black', linewidth=0.5)
    
    # Tambahkan label di atas bar
    for bar, count in zip(bars, [counts.get(i, 0) for i in range(len(class_names))]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Kelas Retinopati Diabetik', fontsize=12)
    ax.set_ylabel('Jumlah Gambar', fontsize=12)
    ax.set_title('Distribusi Kelas Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([f"{i}\n{name}" for i, name in enumerate(class_names)], 
                        fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] Class distribution saved: {save_path}")


def denormalize_image(tensor, mean=None, std=None):
    """
    Denormalisasi tensor gambar untuk visualisasi.
    
    Args:
        tensor (torch.Tensor): Gambar tensor [C, H, W] atau [H, W, C]
        mean (list): Mean normalisasi
        std (list): Std normalisasi
    
    Returns:
        numpy.ndarray: Gambar denormalisasi [H, W, C] range [0, 1]
    """
    from config import IMAGENET_MEAN, IMAGENET_STD
    
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    if isinstance(tensor, torch.Tensor):
        img = tensor.clone().detach().cpu()
        if img.dim() == 3 and img.shape[0] == 3:
            # [C, H, W] -> [H, W, C]
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            img = img * std_t + mean_t
            img = img.permute(1, 2, 0).numpy()
        else:
            img = img.numpy()
    else:
        img = tensor
    
    return np.clip(img, 0, 1)


def print_separator(title="", char="=", length=60):
    """Print separator line."""
    if title:
        padding = (length - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * length)


if __name__ == "__main__":
    """Test utility functions."""
    from config import TRAIN_CSV, CLASS_NAMES, PLOT_DIR
    
    print_separator("TEST UTILITIES")
    
    set_seed(42)
    
    # Test class distribution plot
    plot_class_distribution(
        TRAIN_CSV, CLASS_NAMES,
        os.path.join(PLOT_DIR, "class_distribution.png")
    )
    
    # Test early stopping
    es = EarlyStopping(patience=3)
    test_losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]
    for i, loss in enumerate(test_losses):
        stopped = es(loss)
        print(f"  Epoch {i+1}: loss={loss:.2f}, counter={es.counter}, stop={stopped}")
    
    print("\nUtilities test selesai!")