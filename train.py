"""
train.py - Pipeline Training MViTv2
Deteksi Retinopati Diabetik

Fitur:
- Training dengan class-weighted CrossEntropyLoss
- AdamW optimizer + CosineAnnealingLR scheduler
- Mixed Precision Training (AMP) - opsional untuk GPU
- Early Stopping
- Checkpoint saving (best model)
- Metrics tracking & visualization
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from ordinal_loss import OrdinalCrossEntropyLoss
from sklearn.metrics import cohen_kappa_score

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, MIN_DELTA, USE_AMP, WARMUP_EPOCHS, FREEZE_EPOCHS,
    CHECKPOINT_DIR, PLOT_DIR, TRAIN_CSV,
    MODEL_NAME, NUM_CLASSES, PRETRAINED, IMG_SIZE,
    LOSS_DISTANCE_POWER, LOSS_SMOOTHING_STRENGTH, LOSS_ORDINAL_WEIGHT  # tambahan
)

from model import create_model
from dataset import get_dataloaders, compute_class_weights
from utils import (
    set_seed, save_checkpoint, EarlyStopping, 
    MetricsTracker, plot_training_curves, print_separator
)


def train_one_epoch(model, dataloader, criterion, optimizer, device, 
                    scaler=None, use_amp=False):
    """
    Training untuk satu epoch.
    
    Args:
        model: Model PyTorch
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
        scaler: GradScaler untuk AMP
        use_amp: Gunakan Mixed Precision
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device.type == "cuda":
            # Mixed Precision Training (GPU only)
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard Training (CPU atau GPU tanpa AMP)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Hitung metrik
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    return avg_loss, accuracy, qwk


def train(resume_checkpoint=None):
    """
    Pipeline training lengkap.
    
    Args:
        resume_checkpoint (str): Path ke checkpoint untuk melanjutkan training
    
    Returns:
        tuple: (trained_model, metrics_tracker)
    """
    print_separator("TRAINING PIPELINE")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Mixed Precision: {USE_AMP}")
    print_separator()
    
    # Set seed
    set_seed()
    
    # ========================================
    # 1. Data Loading
    # ========================================
    print("\n[1/5] Loading data...")
    train_loader, val_loader, _ = get_dataloaders()
    
    # ========================================
    # 2. Model
    # ========================================
    print("\n[2/5] Creating model...")
    model = create_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        img_size=IMG_SIZE
    )
    model = model.to(DEVICE)
    
    # ========================================
    # 3. Loss, Optimizer, Scheduler
    # ========================================
    print("\n[3/5] Setting up training components...")
    
    # Class weights untuk menangani imbalance
    class_weights = compute_class_weights(TRAIN_CSV)
    class_weights = class_weights.to(DEVICE)
    
    criterion = OrdinalCrossEntropyLoss(
        num_classes=NUM_CLASSES,
        class_weights=class_weights,
        distance_power=LOSS_DISTANCE_POWER,
        smoothing_strength=LOSS_SMOOTHING_STRENGTH,
        ordinal_weight=LOSS_ORDINAL_WEIGHT,
    ).to(DEVICE)
    
    # Differential Learning Rate: backbone pakai LR lebih kecil
    # karena sudah pretrained, sedangkan classifier head random init
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LEARNING_RATE * 0.1},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    
    # Warmup + Cosine Annealing Scheduler
    # Linear warmup mencegah kerusakan pretrained weights di epoch awal
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )
    
    # Mixed Precision Scaler (hanya untuk GPU)
    scaler = GradScaler("cuda") if USE_AMP else None
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, mode="min")
    
    # Metrics Tracker
    metrics = MetricsTracker()
    
    # Resume training
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        from utils import load_checkpoint
        ckpt = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, DEVICE)
        start_epoch = ckpt["epoch"]
        print(f"Resuming from epoch {start_epoch}")
    
    # ========================================
    # 4. Training Loop
    # ========================================
    print("\n[4/5] Starting training...")
    print_separator()
    
    best_val_acc = 0.0
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        # Fine-Tuning Bertahap: Freeze backbone di epoch awal
        if epoch < FREEZE_EPOCHS:
            if epoch == start_epoch: # Print status at the beginning
                model.freeze_backbone(True)
        elif epoch == FREEZE_EPOCHS:
            print("\n[PHASE 2] Unfreezing backbone for full fine-tuning...")
            model.freeze_backbone(False)
            
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            scaler=scaler, use_amp=USE_AMP
        )
        
        # Validation
        val_loss, val_acc, val_qwk = validate(model, val_loader, criterion, DEVICE)
        
        # Learning rate step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        # Update metrics
        metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr)
        
        # Waktu per epoch
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val QWK: {val_qwk:.4f} "
          f"| LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                val_loss, val_acc, val_qwk,
                os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )
            print(f"  ★ New best model! Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val QWK: {val_qwk:.4f}")
        
        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            val_loss, val_acc, val_qwk,
            os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
        )
        
        # Early stopping check
        if early_stopping(val_qwk):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # ========================================
    # 5. Post-Training
    # ========================================
    print_separator("TRAINING COMPLETE")
    
    best_info = metrics.get_best_epoch()
    print(f"Best Epoch: {best_info['epoch']}")
    print(f"Best Val Accuracy: {best_info['val_accuracy']:.2f}%")
    print(f"Best Val Loss: {best_info['val_loss']:.4f}")
    print(f"Best Val QWK: {best_info['val_qwk']:.4f}")
    print(f"Best Train Accuracy: {best_info['train_accuracy']:.2f}%")
    print(f"Best Train Loss: {best_info['train_loss']:.4f}")
    
    # Plot training curves
    plot_training_curves(
        metrics,
        os.path.join(PLOT_DIR, "training_curves.png")
    )
    
    # Save metrics
    import json
    metrics_data = {
        "train_losses": metrics.train_losses,
        "val_losses": metrics.val_losses,
        "train_accuracies": metrics.train_accuracies,
        "val_accuracies": metrics.val_accuracies,
        "val_qwks": metrics.val_qwks,
        "learning_rates": metrics.learning_rates,
        "best_epoch": best_info
    }
    
    metrics_path = os.path.join(CHECKPOINT_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[METRICS] Saved: {metrics_path}")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = train()
