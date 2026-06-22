"""
model.py - Arsitektur Model MViTv2
Deteksi Retinopati Diabetik dengan MViTv2

Menggunakan timm library untuk load pretrained MViTv2.
Classification head dimodifikasi untuk 5 kelas DR.
"""

import torch
import torch.nn as nn
import timm


class MViTv2Model(nn.Module):
    """
    Multiscale Vision Transformer v2 untuk klasifikasi Retinopati Diabetik.
    
    MViTv2 menangkap representasi fitur multiskala secara hierarkis,
    cocok untuk mendeteksi lesi retina pada berbagai skala (mikroaneurisma 
    kecil hingga neovaskularisasi besar).
    
    Args:
        model_name (str): Nama model timm (default: "mvitv2_tiny")
        num_classes (int): Jumlah kelas output (default: 5)
        pretrained (bool): Gunakan pretrained ImageNet weights
        drop_rate (float): Dropout rate pada classification head
    """
    
    def __init__(self, model_name="mvitv2_tiny", num_classes=5, 
                 pretrained=True, drop_rate=0.3, img_size=224):
        super(MViTv2Model, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained MViTv2 dari timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # Hapus classification head bawaan
            drop_rate=drop_rate,
            img_size=img_size       # Sangat penting untuk interpolasi positional embedding
        )
        
        # Dapatkan dimensi fitur dari backbone
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=drop_rate),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(p=drop_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        print(f"[MODEL] {model_name} loaded (pretrained={pretrained})")
        print(f"[MODEL] Feature dim: {self.feature_dim}")
        print(f"[MODEL] Num classes: {num_classes}")
        print(f"[MODEL] Total parameters: {self.count_parameters():,}")
        print(f"[MODEL] Trainable parameters: {self.count_trainable_parameters():,}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [B, 3, H, W]
        
        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        # Extract features dari backbone
        features = self.backbone(x)     # [B, feature_dim]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x):
        """
        Ekstrak fitur dari backbone (tanpa classification head).
        Berguna untuk analisis fitur dan explainability.
        
        Args:
            x (torch.Tensor): Input tensor [B, 3, H, W]
        
        Returns:
            torch.Tensor: Feature vector [B, feature_dim]
        """
        return self.backbone(x)
    
    def count_parameters(self):
        """Hitung total parameter model."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        """Hitung parameter yang dapat dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self, freeze=True):
        """
        Freeze/unfreeze backbone untuk transfer learning.
        
        Saat freeze=True, hanya classification head yang dilatih.
        Berguna untuk fine-tuning bertahap.
        
        Args:
            freeze (bool): True untuk freeze, False untuk unfreeze
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        trainable = self.count_trainable_parameters()
        print(f"[MODEL] Backbone {status}. Trainable params: {trainable:,}")
    
    def get_target_layer(self):
        """
        Mendapatkan target layer untuk Grad-CAM++.
        
        Mengembalikan layer terakhir dari backbone MViTv2
        yang menghasilkan feature map spasial.
        
        Returns:
            nn.Module: Target layer untuk Grad-CAM
        """
        # Untuk MViTv2, gunakan stage terakhir
        # timm MViTv2 memiliki self.backbone.stages[-1]
        if hasattr(self.backbone, 'stages'):
            return self.backbone.stages[-1]
        elif hasattr(self.backbone, 'blocks'):
            return self.backbone.blocks[-1]
        elif hasattr(self.backbone, 'norm'):
            return self.backbone.norm
        else:
            # Fallback: cari layer terakhir
            modules = list(self.backbone.modules())
            for m in reversed(modules):
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    return m
            return modules[-2]


def create_model(model_name="mvitv2_tiny", num_classes=5, 
                 pretrained=True, drop_rate=0.3, img_size=224):
    """
    Factory function untuk membuat model MViTv2.
    
    Args:
        model_name (str): Nama model
        num_classes (int): Jumlah kelas
        pretrained (bool): Gunakan pretrained weights
        drop_rate (float): Dropout rate
    
    Returns:
        MViTv2Model: Model yang sudah diinisialisasi
    """
    model = MViTv2Model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        img_size=img_size
    )
    return model


def load_model_for_inference(checkpoint_path, model_name="mvitv2_tiny", 
                              num_classes=5, device="cpu", img_size=224):
    """
    Load model dari checkpoint untuk inferensi.
    
    Args:
        checkpoint_path (str): Path ke file checkpoint (.pth)
        model_name (str): Nama model
        num_classes (int): Jumlah kelas
        device (str): Device target
    
    Returns:
        MViTv2Model: Model yang sudah di-load
    """
    model = MViTv2Model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        img_size=img_size
    )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        # Checkpoint lama (sebelum fix numpy-scalar di save_checkpoint) bisa berisi
        # tipe seperti numpy.float64 yang ditolak weights_only=True di PyTorch >=2.6
        # (error: "Unsupported global: numpy._core.multiarray.scalar").
        # Fallback ke weights_only=False -- AMAN karena file ini hasil training kita
        # sendiri, bukan checkpoint dari sumber tidak dikenal.
        print(f"[MODEL] weights_only=True gagal ({type(e).__name__}), "
              f"mencoba ulang dengan weights_only=False (checkpoint lama)...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle checkpoint yang berisi state_dict di dalam dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[MODEL] Loaded checkpoint dari epoch {checkpoint.get('epoch', '?')}")
        print(f"[MODEL] Val accuracy: {checkpoint.get('val_accuracy', '?')}")
        if checkpoint.get("val_qwk") is not None:
            print(f"[MODEL] Val QWK: {checkpoint['val_qwk']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    """Test model creation."""
    from config import MODEL_NAME, NUM_CLASSES, DEVICE, IMG_SIZE
    
    print("=" * 60)
    print("TEST MODEL CREATION")
    print("=" * 60)
    
    model = create_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=True,
        img_size=IMG_SIZE
    )
    model = model.to(DEVICE)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Test target layer
    target_layer = model.get_target_layer()
    print(f"\nTarget layer untuk Grad-CAM: {type(target_layer).__name__}")
    
    print("\nModel test selesai!")