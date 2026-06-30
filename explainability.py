"""
explainability.py - Framework Explainability Hibrida
Deteksi Retinopati Diabetik dengan MViTv2

Implementasi tiga metode explainability:
1. Grad-CAM++ - Lokalisasi area retina yang berkontribusi dominan
2. Occlusion - Analisis sensitivitas berbasis perturbasi
3. Integrated Gradients - Analisis sensitivitas input

Pendekatan hibrida: ketiga metode saling melengkapi untuk menghasilkan
interpretasi yang lebih komprehensif dan robust.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from tqdm import tqdm

from config import (
    DEVICE, NUM_CLASSES, CLASS_NAMES, IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    CHECKPOINT_DIR, EXPLAIN_DIR, MODEL_NAME,
    EXPLAIN_NUM_SAMPLES, EXPLAIN_IG_STEPS, EXPLAIN_COLORMAP,
    EXPLAIN_OCCLUSION_WINDOW, EXPLAIN_OCCLUSION_STRIDE
)
from model import load_model_for_inference
from utils import denormalize_image, print_separator


# ============================================================
# 1. GRAD-CAM++ 
# ============================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++ untuk visualisasi area penting pada citra retina.
    
    Grad-CAM++ adalah generalisasi dari Grad-CAM yang menggunakan 
    weighted combination dari positive partial derivatives untuk 
    menghasilkan lokalisasi yang lebih akurat, terutama untuk 
    objek-objek kecil seperti mikroaneurisma.
    
    Referensi: Chattopadhay et al., "Grad-CAM++: Generalized 
    Gradient-based Visual Explanations for Deep Convolutional Networks"
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Model PyTorch
            target_layer: Layer target untuk mengekstrak gradien
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward dan backward hooks pada target layer."""
        def forward_hook(module, input, output):
            # Handle berbagai format output
            if isinstance(output, torch.Tensor):
                self.activations = output.detach()
            elif isinstance(output, tuple):
                self.activations = output[0].detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
    def remove_hooks(self):
        """Hapus hooks untuk mencegah memory leak."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor (torch.Tensor): Input gambar [1, 3, H, W]
            target_class (int): Kelas target (None = kelas prediksi)
        
        Returns:
            numpy.ndarray: Heatmap [H, W] dinormalisasi [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("[WARNING] Gradients or activations not captured")
            return np.zeros((IMG_SIZE, IMG_SIZE))
        
        gradients = self.gradients
        activations = self.activations
        
        # Reshape jika diperlukan (MViTv2 output bisa [B, N, C] bukan [B, C, H, W])
        if gradients.dim() == 3:
            # [B, N, C] -> perlu reshape ke spatial
            B, N, C = gradients.shape
            H = W = int(np.sqrt(N))
            if H * W != N:
                # Ada CLS token, hapus
                H = W = int(np.sqrt(N - 1))
                gradients = gradients[:, 1:, :]  # Hapus CLS token
                activations = activations[:, 1:, :]
            gradients = gradients.permute(0, 2, 1).reshape(B, C, H, W)
            activations = activations.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Grad-CAM++ weights
        # alpha = relu(gradient²) / (2*gradient² + sum(activation * gradient³) + eps)
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha_numer = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_numer / alpha_denom
        
        # Hanya gradien positif
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize ke ukuran input
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), 
                           mode='bilinear', align_corners=False)
        
        cam = cam.squeeze().cpu().numpy()
        
        # Normalisasi
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
        self.remove_hooks()
        
        return cam


# ============================================================
# 2. OCCLUSION
# ============================================================

def compute_occlusion(model, input_tensor, target_class=None, device="cpu",
                      sliding_window_shapes=None, strides=None):
    """
    Compute Occlusion attribution.
    
    Occlusion adalah metode perturbasi yang secara sistematis menutupi 
    (occlude) area input dengan sliding window dan mengukur perubahan 
    prediksi. Area yang menyebabkan penurunan prediksi terbesar 
    dianggap paling penting.
    
    Metode ini bersifat model-agnostic dan tidak bergantung pada gradien,
    sehingga memberikan perspektif yang berbeda dari Grad-CAM++ dan 
    Integrated Gradients.
    
    Menggunakan Captum library.
    
    Referensi: Zeiler & Fergus, "Visualizing and Understanding 
    Convolutional Networks" (2014)
    
    Args:
        model: Model PyTorch
        input_tensor (torch.Tensor): Input gambar [1, 3, H, W]
        target_class (int): Kelas target (None = kelas prediksi)
        device: Device
        sliding_window_shapes (tuple): Ukuran sliding window (C, H, W). 
            Default menggunakan EXPLAIN_OCCLUSION_WINDOW dari config.
        strides (tuple): Stride untuk sliding window (C, H, W).
            Default menggunakan EXPLAIN_OCCLUSION_STRIDE dari config.
    
    Returns:
        numpy.ndarray: Attribution map [H, W]
    """
    from captum.attr import Occlusion
    
    model.eval()
    input_tensor = input_tensor.to(device)
    
    if sliding_window_shapes is None:
        sliding_window_shapes = (3, EXPLAIN_OCCLUSION_WINDOW, EXPLAIN_OCCLUSION_WINDOW)
    if strides is None:
        strides = (3, EXPLAIN_OCCLUSION_STRIDE, EXPLAIN_OCCLUSION_STRIDE)
    
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()
    
    try:
        occlusion = Occlusion(model)
        attribution = occlusion.attribute(
            input_tensor,
            target=target_class,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=0
        )
        
        # Konversi ke heatmap
        attr_map = attribution.squeeze().cpu().detach().numpy()
        
        # Rata-rata channel untuk mendapat [H, W]
        if attr_map.ndim == 3:
            attr_map = np.mean(attr_map, axis=0)
        
        # Normalisasi
        attr_map = np.abs(attr_map)
        if attr_map.max() > 0:
            attr_map = attr_map / (attr_map.max() + 1e-8)
        
        return attr_map
    
    except Exception as e:
        print(f"[WARNING] Occlusion gagal: {e}")
        print("[WARNING] Menggunakan Simple Gradient sebagai fallback")
        return compute_simple_gradient(model, input_tensor, target_class, device)


def compute_simple_gradient(model, input_tensor, target_class=None, device="cpu"):
    """
    Fallback: Simple Gradient Attribution.
    
    Menghitung gradient dari output terhadap input sebagai
    proxy untuk attribution map jika Occlusion gagal.
    
    Args:
        model: Model PyTorch
        input_tensor (torch.Tensor): Input gambar [1, 3, H, W]
        target_class (int): Kelas target
        device: Device
    
    Returns:
        numpy.ndarray: Gradient map [H, W]
    """
    model.eval()
    input_tensor = input_tensor.to(device).detach().clone().requires_grad_(True)
    
    output = model(input_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    gradients = input_tensor.grad.squeeze().cpu().numpy()
    
    # Rata-rata channel
    if gradients.ndim == 3:
        gradients = np.mean(np.abs(gradients), axis=0)
    
    # Normalisasi
    if gradients.max() > 0:
        gradients = gradients / (gradients.max() + 1e-8)
    
    return gradients


# ============================================================
# 3. INTEGRATED GRADIENTS
# ============================================================

def compute_integrated_gradients(model, input_tensor, target_class=None,
                                  n_steps=50, device="cpu"):
    """
    Compute Integrated Gradients.
    
    Integrated Gradients menganalisis sensitivitas prediksi terhadap 
    perubahan input dengan mengintegrasikan gradien sepanjang path 
    dari baseline (gambar hitam) ke input.
    
    Referensi: Sundararajan et al., "Axiomatic Attribution for Deep Networks"
    
    Args:
        model: Model PyTorch
        input_tensor (torch.Tensor): Input gambar [1, 3, H, W]
        target_class (int): Kelas target (None = kelas prediksi)
        n_steps (int): Jumlah langkah integrasi
        device: Device
    
    Returns:
        numpy.ndarray: Attribution map [H, W]
    """
    from captum.attr import IntegratedGradients
    
    model.eval()
    input_tensor = input_tensor.to(device).detach().clone().requires_grad_(True)
    
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()
    
    # Baseline: random noise kecil
    baseline = (torch.randn_like(input_tensor) * 0.01).to(device)
    
    try:
        ig = IntegratedGradients(model)
        attribution = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=4,
            return_convergence_delta=False
        )
        
        # Konversi ke heatmap
        attr_map = attribution.squeeze().cpu().detach().numpy()
        
        # Rata-rata channel
        if attr_map.ndim == 3:
            attr_map = np.mean(attr_map, axis=0)
        
        # Normalisasi (simpan tanda untuk analisis positif/negatif)
        attr_abs = np.abs(attr_map)
        if attr_abs.max() > 0:
            attr_map = attr_abs / (attr_abs.max() + 1e-8)
        
        return attr_map
    
    except Exception as e:
        print(f"[WARNING] Integrated Gradients gagal: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE))


# ============================================================
# VISUALISASI
# ============================================================

def create_heatmap_overlay(image, heatmap, colormap=EXPLAIN_COLORMAP, alpha=0.5):
    """
    Overlay heatmap pada gambar asli.
    
    Args:
        image (numpy.ndarray): Gambar asli [H, W, 3] range [0, 1]
        heatmap (numpy.ndarray): Heatmap [H, W] range [0, 1]
        colormap (str): Nama colormap matplotlib
        alpha (float): Transparansi overlay
    
    Returns:
        numpy.ndarray: Gambar dengan overlay heatmap
    """
    # Resize heatmap jika diperlukan
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Terapkan colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # [H, W, 3]
    
    # Overlay
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_single_sample(image_tensor, model, predicted_class, true_class,
                            device, save_path=None, sample_id="sample"):
    """
    Visualisasi explainability untuk satu sampel dengan ketiga metode.
    
    Menghasilkan figure dengan layout:
    - Baris 1: Gambar Asli | Grad-CAM++ | Occlusion | Integrated Gradients
    - Baris 2: Overlay per metode
    
    Args:
        image_tensor (torch.Tensor): Input tensor [1, 3, H, W]
        model: Model PyTorch
        predicted_class (int): Kelas prediksi
        true_class (int): Kelas sebenarnya
        device: Device
        save_path (str): Path untuk menyimpan
        sample_id (str): ID sampel
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Denormalisasi gambar untuk visualisasi
    orig_image = denormalize_image(image_tensor.squeeze(0))
    
    # ========================================
    # Generate attributions
    # ========================================
    
    # 1. Grad-CAM++
    print(f"  Computing Grad-CAM++...")
    target_layer = model.get_target_layer()
    gradcam = GradCAMPlusPlus(model, target_layer)
    gradcam_map = gradcam.generate(image_tensor, target_class=predicted_class)
    
    # 2. Occlusion
    print(f"  Computing Occlusion...")
    occlusion_map = compute_occlusion(model, image_tensor, target_class=predicted_class, 
                                      device=device)
    
    # 3. Integrated Gradients
    print(f"  Computing Integrated Gradients...")
    ig_map = compute_integrated_gradients(
        model, image_tensor, target_class=predicted_class,
        n_steps=EXPLAIN_IG_STEPS, device=device
    )
    
    # ========================================
    # Visualisasi
    # ========================================
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.2)
    
    # Status prediksi
    status = "[TRUE] BENAR" if predicted_class == true_class else "[FALSE] SALAH"
    fig.suptitle(
        f"Explainability Analysis - {sample_id}\n"
        f"True: {CLASS_NAMES[true_class]} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} | {status}",
        fontsize=14, fontweight='bold'
    )
    
    # --- Baris 1: Heatmaps ---
    # Gambar asli
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_image)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis("off")
    
    # Grad-CAM++
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(gradcam_map, cmap=EXPLAIN_COLORMAP, vmin=0, vmax=1)
    ax2.set_title("Grad-CAM++\n(Area Dominan)", fontsize=12, fontweight='bold')
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Occlusion
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(occlusion_map, cmap=EXPLAIN_COLORMAP, vmin=0, vmax=1)
    ax3.set_title("Occlusion\n(Sensitivitas Perturbasi)", fontsize=12, fontweight='bold')
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Integrated Gradients
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(ig_map, cmap=EXPLAIN_COLORMAP, vmin=0, vmax=1)
    ax4.set_title("Integrated Gradients\n(Sensitivitas Input)", fontsize=12, 
                  fontweight='bold')
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # --- Baris 2: Overlay pada gambar asli ---
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(orig_image)
    ax5.set_title("Original Image", fontsize=12)
    ax5.axis("off")
    
    ax6 = fig.add_subplot(gs[1, 1])
    overlay_gc = create_heatmap_overlay(orig_image, gradcam_map)
    ax6.imshow(overlay_gc)
    ax6.set_title("Overlay Grad-CAM++", fontsize=12)
    ax6.axis("off")
    
    ax7 = fig.add_subplot(gs[1, 2])
    overlay_occlusion = create_heatmap_overlay(orig_image, occlusion_map)
    ax7.imshow(overlay_occlusion)
    ax7.set_title("Overlay Occlusion", fontsize=12)
    ax7.axis("off")
    
    ax8 = fig.add_subplot(gs[1, 3])
    overlay_ig = create_heatmap_overlay(orig_image, ig_map)
    ax8.imshow(overlay_ig)
    ax8.set_title("Overlay Integrated Gradients", fontsize=12)
    ax8.axis("off")
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    plt.close(fig)
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        "gradcam": gradcam_map,
        "occlusion": occlusion_map,
        "ig": ig_map
    }


def visualize_comparison_grid(image_tensor, model, predicted_class, true_class,
                               device, save_path=None, sample_id="sample",
                               precomputed_maps=None):
    """
    Visualisasi perbandingan ringkas dalam satu baris.
    
    Layout: Original | Grad-CAM++ | Occlusion | IG | Combined
    
    Args:
        image_tensor: Input tensor
        model: Model
        predicted_class: Kelas prediksi
        true_class: Kelas sebenarnya
        device: Device
        save_path: Path simpan
        sample_id: ID sampel
    
    Returns:
        dict: Attribution maps
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    orig_image = denormalize_image(image_tensor.squeeze(0))
    
    # Gunakan precomputed maps jika ada untuk menghemat waktu
    if precomputed_maps:
        gradcam_map = precomputed_maps["gradcam"]
        occlusion_map = precomputed_maps["occlusion"]
        ig_map = precomputed_maps["ig"]
    else:
        # Generate attributions
        target_layer = model.get_target_layer()
        gradcam = GradCAMPlusPlus(model, target_layer)
        gradcam_map = gradcam.generate(image_tensor, target_class=predicted_class)
        occlusion_map = compute_occlusion(model, image_tensor, target_class=predicted_class, 
                                           device=device)
        ig_map = compute_integrated_gradients(
            model, image_tensor, target_class=predicted_class,
            n_steps=EXPLAIN_IG_STEPS, device=device
        )
    
    # Combined: rata-rata ketiga metode
    combined = (gradcam_map + occlusion_map + ig_map) / 3.0
    if combined.max() > 0:
        combined = combined / (combined.max() + 1e-8)
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    status = "[T]" if predicted_class == true_class else "[F]"
    
    axes[0].imshow(orig_image)
    axes[0].set_title(f"Original\nTrue: {CLASS_NAMES[true_class]}", fontsize=11)
    axes[0].axis("off")
    
    overlay_gc = create_heatmap_overlay(orig_image, gradcam_map)
    axes[1].imshow(overlay_gc)
    axes[1].set_title(f"Grad-CAM++\nPred: {CLASS_NAMES[predicted_class]} {status}", 
                      fontsize=11)
    axes[1].axis("off")
    
    overlay_occlusion = create_heatmap_overlay(orig_image, occlusion_map)
    axes[2].imshow(overlay_occlusion)
    axes[2].set_title("Occlusion", fontsize=11)
    axes[2].axis("off")
    
    overlay_ig = create_heatmap_overlay(orig_image, ig_map)
    axes[3].imshow(overlay_ig)
    axes[3].set_title("Integrated Gradients", fontsize=11)
    axes[3].axis("off")
    
    overlay_combined = create_heatmap_overlay(orig_image, combined)
    axes[4].imshow(overlay_combined)
    axes[4].set_title("Combined (Hybrid)", fontsize=11)
    axes[4].axis("off")
    
    plt.suptitle(f"Hybrid Explainability - {sample_id}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {"gradcam": gradcam_map, "occlusion": occlusion_map, "ig": ig_map, "combined": combined}


# ============================================================
# BATCH EXPLAINABILITY
# ============================================================

def run_explainability(checkpoint_path=None, num_samples_per_class=None):
    """
    Jalankan analisis explainability pada sampel dari setiap kelas.
    
    Menghasilkan visualisasi per sampel dan summary grid per kelas.
    
    Args:
        checkpoint_path (str): Path ke model checkpoint
        num_samples_per_class (int): Jumlah sampel per kelas
    """
    print_separator("EXPLAINABILITY ANALYSIS")
    
    if num_samples_per_class is None:
        num_samples_per_class = EXPLAIN_NUM_SAMPLES
    
    # ========================================
    # 1. Load Model
    # ========================================
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint tidak ditemukan: {checkpoint_path}")
        return
    
    print(f"[1/3] Loading model...")
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
    print(f"\n[2/3] Loading test data...")
    from dataset import get_dataloaders
    _, _, test_loader = get_dataloaders()
    
    # Kumpulkan sampel per kelas
    class_samples = {i: [] for i in range(NUM_CLASSES)}
    
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            cls = lbl.item()
            if len(class_samples[cls]) < num_samples_per_class:
                class_samples[cls].append(img)
        
        # Cek apakah sudah cukup
        if all(len(v) >= num_samples_per_class for v in class_samples.values()):
            break
    
    # ========================================
    # 3. Generate Visualizations
    # ========================================
    print(f"\n[3/3] Generating explainability visualizations...")
    
    for cls_idx in range(NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_idx]
        cls_dir = os.path.join(EXPLAIN_DIR, f"class_{cls_idx}_{cls_name.replace(' ', '_')}")
        os.makedirs(cls_dir, exist_ok=True)
        
        print(f"\n--- Kelas {cls_idx}: {cls_name} ({len(class_samples[cls_idx])} samples) ---")
        
        for s_idx, img_tensor in enumerate(class_samples[cls_idx]):
            sample_id = f"class{cls_idx}_{cls_name}_sample{s_idx+1}"
            input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            
            # Prediksi
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()
            
            print(f"\n  Sample {s_idx+1}: True={cls_name}, "
                  f"Pred={CLASS_NAMES[pred_class]}")
            
            # Visualisasi detail (2 baris)
            maps = visualize_single_sample(
                input_tensor, model, pred_class, cls_idx,
                device=DEVICE,
                save_path=os.path.join(cls_dir, f"{sample_id}_detail.png"),
                sample_id=sample_id
            )
            
            # Visualisasi ringkas (1 baris), reuse precomputed maps
            visualize_comparison_grid(
                input_tensor, model, pred_class, cls_idx,
                device=DEVICE,
                save_path=os.path.join(cls_dir, f"{sample_id}_comparison.png"),
                sample_id=sample_id,
                precomputed_maps=maps
            )
    
    print_separator("EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"Visualisasi tersimpan di: {EXPLAIN_DIR}")


def explain_single_image(image_path, checkpoint_path=None, save_dir=None):
    """
    Analisis explainability untuk satu gambar spesifik.
    
    Args:
        image_path (str): Path ke gambar
        checkpoint_path (str): Path ke checkpoint model
        save_dir (str): Direktori untuk menyimpan hasil
    """
    from preprocessing import load_and_preprocess
    from dataset import get_val_transforms
    
    print_separator(f"EXPLAIN: {os.path.basename(image_path)}")
    
    # Load model
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    model = load_model_for_inference(
        checkpoint_path, model_name=MODEL_NAME,
        num_classes=NUM_CLASSES, device=DEVICE,
        img_size=IMG_SIZE
    )
    
    # Load dan preprocess gambar
    img = load_and_preprocess(image_path, img_size=IMG_SIZE)
    
    # Terapkan transformasi
    transform = get_val_transforms()
    augmented = transform(image=img)
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    
    # Prediksi
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    print(f"Prediksi: {CLASS_NAMES[pred_class]} (confidence: {confidence:.4f})")
    print(f"Probabilitas per kelas:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[0, i].item():.4f}")
    
    # Generate visualisasi
    if save_dir is None:
        save_dir = EXPLAIN_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    visualize_single_sample(
        input_tensor, model, pred_class, pred_class,
        device=DEVICE,
        save_path=os.path.join(save_dir, f"{basename}_explainability.png"),
        sample_id=basename
    )


if __name__ == "__main__":
    run_explainability()
