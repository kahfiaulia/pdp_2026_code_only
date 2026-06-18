"""
evaluate.py - Evaluasi Model pada Test Set
Deteksi Retinopati Diabetik dengan MViTv2

Menghasilkan:
- Classification Report (Precision, Recall, F1-Score per kelas)
- Confusion Matrix (heatmap)
- ROC-AUC Curve per kelas
- Overall Metrics (Accuracy, Weighted/Macro F1, Cohen's Kappa)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, cohen_kappa_score,
    roc_curve, auc, f1_score
)

from config import (
    DEVICE, NUM_CLASSES, CLASS_NAMES,
    CHECKPOINT_DIR, PLOT_DIR, METRICS_DIR,
    MODEL_NAME, IMG_SIZE
)
from model import load_model_for_inference
from dataset import get_dataloaders
from utils import print_separator


def get_predictions(model, dataloader, device, use_tta=True):
    """
    Mendapatkan prediksi model pada seluruh dataset dengan Test-Time Augmentation (opsional).
    
    Args:
        model: Model PyTorch
        dataloader: DataLoader
        device: Device
        use_tta: Jika True, gunakan Test-Time Augmentation (Original + HFlip + VFlip)
    
    Returns:
        tuple: (all_labels, all_predictions, all_probabilities)
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print(f"[EVAL] Test-Time Augmentation (TTA): {'Aktif' if use_tta else 'Tidak Aktif'}")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            if use_tta:
                # Original
                out_orig = model(images)
                prob_orig = F.softmax(out_orig, dim=1)
                
                # Horizontal Flip
                images_hflip = torch.flip(images, dims=[3])
                out_hflip = model(images_hflip)
                prob_hflip = F.softmax(out_hflip, dim=1)
                
                # Vertical Flip
                images_vflip = torch.flip(images, dims=[2])
                out_vflip = model(images_vflip)
                prob_vflip = F.softmax(out_vflip, dim=1)
                
                # Average probabilities
                probs = (prob_orig + prob_hflip + prob_vflip) / 3.0
            else:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
            _, predicted = torch.max(probs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, 
                          normalize=False):
    """
    Visualisasi Confusion Matrix sebagai heatmap.
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        class_names: Nama kelas
        save_path: Path untuk menyimpan
        normalize: Normalisasi per baris (persentase)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'MViTv2 - {title}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] Confusion matrix saved: {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """
    Plot ROC Curve per kelas (One-vs-Rest).
    
    Args:
        y_true: Label sebenarnya
        y_probs: Probabilitas prediksi [N, num_classes]
        class_names: Nama kelas
        save_path: Path untuk menyimpan
    """
    from sklearn.preprocessing import label_binarize
    
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    all_auc = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, 
            label='Random (AUC = 0.5000)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('MViTv2 - ROC Curve (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] ROC curves saved: {save_path}")
    
    return all_auc


def plot_per_class_metrics(report_dict, class_names, save_path):
    """
    Plot metrik per kelas (Precision, Recall, F1-Score) sebagai grouped bar chart.
    
    Args:
        report_dict: Dictionary dari classification_report
        class_names: Nama kelas
        save_path: Path untuk menyimpan
    """
    metrics_names = ['precision', 'recall', 'f1-score']
    
    data = {metric: [] for metric in metrics_names}
    for cls_name in class_names:
        for metric in metrics_names:
            data[metric].append(report_dict[cls_name][metric])
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        bars = ax.bar(x + i * width, data[metric], width, 
                      label=metric.capitalize(), color=color, edgecolor='black',
                      linewidth=0.5)
        # Tambahkan nilai di atas bar
        for bar, val in zip(bars, data[metric]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Kelas', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('MViTv2 - Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] Per-class metrics saved: {save_path}")


def evaluate(checkpoint_path=None):
    """
    Evaluasi lengkap model pada test set.
    
    Args:
        checkpoint_path (str): Path ke model checkpoint
    
    Returns:
        dict: Semua metrik evaluasi
    """
    print_separator("EVALUATION PIPELINE")
    
    # ========================================
    # 1. Load Model
    # ========================================
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint tidak ditemukan: {checkpoint_path}")
        print("[ERROR] Jalankan training terlebih dahulu!")
        return None
    
    print(f"\n[1/4] Loading model from {checkpoint_path}...")
    model = load_model_for_inference(
        checkpoint_path, 
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES, 
        device=DEVICE,
        img_size=IMG_SIZE
    )
    
    # ========================================
    # 2. Load Test Data
    # ========================================
    print("\n[2/4] Loading test data...")
    _, _, test_loader = get_dataloaders()
    
    # ========================================
    # 3. Get Predictions dengan TTA
    # ========================================
    print("\n[3/4] Getting predictions...")
    y_true, y_pred, y_probs = get_predictions(model, test_loader, DEVICE, use_tta=True)
    
    # ========================================
    # 4. Calculate Metrics
    # ========================================
    print("\n[4/4] Computing metrics...")
    
    # Classification Report
    report = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    
    # Overall Metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    print_separator("CLASSIFICATION REPORT")
    print(report_text)
    
    print_separator("OVERALL METRICS")
    print(f"  Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Weighted F1-Score:     {weighted_f1:.4f}")
    print(f"  Macro F1-Score:        {macro_f1:.4f}")
    print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    
    # ========================================
    # 5. Visualizations
    # ========================================
    
    # Confusion Matrix (raw counts)
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        os.path.join(PLOT_DIR, "confusion_matrix.png"),
        normalize=False
    )
    
    # Confusion Matrix (normalized)
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        os.path.join(PLOT_DIR, "confusion_matrix_normalized.png"),
        normalize=True
    )
    
    # ROC Curves
    auc_scores = plot_roc_curves(
        y_true, y_probs, CLASS_NAMES,
        os.path.join(PLOT_DIR, "roc_curves.png")
    )
    
    # Per-Class Metrics Bar Chart
    plot_per_class_metrics(
        report, CLASS_NAMES,
        os.path.join(PLOT_DIR, "per_class_metrics.png")
    )
    
    # ========================================
    # 6. Save Results
    # ========================================
    
    # Save classification report
    report_path = os.path.join(METRICS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MViTv2 - Classification Report\n")
        f.write("Diabetic Retinopathy Detection\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
        f.write("\n\nOverall Metrics:\n")
        f.write(f"  Accuracy:              {accuracy:.4f}\n")
        f.write(f"  Weighted F1-Score:     {weighted_f1:.4f}\n")
        f.write(f"  Macro F1-Score:        {macro_f1:.4f}\n")
        f.write(f"  Cohen's Kappa (QWK):   {kappa:.4f}\n")
        f.write(f"\nPer-Class AUC:\n")
        for name, auc_val in zip(CLASS_NAMES, auc_scores):
            f.write(f"  {name}: {auc_val:.4f}\n")
        f.write(f"  Mean AUC: {np.mean(auc_scores):.4f}\n")
    
    print(f"\n[METRICS] Classification report saved: {report_path}")
    
    # Save as JSON
    import json
    results = {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "quadratic_weighted_kappa": float(qwk),
        "per_class_auc": {name: float(a) for name, a in zip(CLASS_NAMES, auc_scores)},
        "mean_auc": float(np.mean(auc_scores)),
        "per_class_report": {name: report[name] for name in CLASS_NAMES}
    }
    
    json_path = os.path.join(METRICS_DIR, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[METRICS] JSON results saved: {json_path}")
    
    print_separator("EVALUATION COMPLETE")
    
    return results


if __name__ == "__main__":
    results = evaluate()
