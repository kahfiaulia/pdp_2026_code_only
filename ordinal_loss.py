"""
Ordinal-aware loss untuk DR grading (5 kelas: No DR, Mild, Moderate, Severe, Proliferative)
Drop-in replacement untuk nn.CrossEntropyLoss(weight=..., label_smoothing=0.1)

Dua komponen yang dijumlahkan:
1. Soft-CE dengan target distribution berbasis jarak ordinal (ganti label_smoothing biasa)
2. Ordinal regression penalty (expected value vs label, di-weight oleh class weight)

Tidak perlu ubah arsitektur model -- tetap 5-unit softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(nn.Module):
    """
    CE dengan soft target yang meluruh berdasarkan jarak kelas (bukan uniform smoothing),
    dijumlahkan dengan penalti regresi ordinal (expected class vs true class)^2.

    Args:
        num_classes: jumlah kelas (5 untuk DR grading)
        class_weights: tensor (num_classes,) -- weight per kelas, sama seperti yang sudah Anda pakai
        distance_power: kontrol seberapa cepat target meluruh terhadap jarak.
            1.0 = linear decay, 2.0 = decay lebih cepat (lebih mirip QWK yang kuadratik)
        smoothing_strength: total massa probabilitas yang "dibagi" ke kelas lain (analog label_smoothing=0.1)
        ordinal_weight: bobot untuk komponen regresi ordinal (loss kedua). Mulai dari 0.3-0.5.
        ce_class_weighted: jika True, soft-CE juga di-weight oleh class_weights pada kelas target
        ordinal_class_weighted: jika True, komponen regresi ordinal JUGA di-weight oleh class_weights.
            PENTING jika dataloader Anda sudah memakai WeightedRandomSampler: kombinasi
            sampler (yang sudah menyamakan frekuensi exposure tiap kelas per epoch) DAN
            class_weights di loss bisa jadi double-compensation yang terlalu agresif.
            Set False di sini untuk biarkan sampler menangani imbalance exposure, sementara
            ordinal penalty murni fokus ke jarak kesalahan tanpa weighting tambahan.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: torch.Tensor = None,
        distance_power: float = 2.0,
        smoothing_strength: float = 0.1,
        ordinal_weight: float = 0.4,
        ce_class_weighted: bool = True,
        ordinal_class_weighted: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.distance_power = distance_power
        self.smoothing_strength = smoothing_strength
        self.ordinal_weight = ordinal_weight
        self.ce_class_weighted = ce_class_weighted
        self.ordinal_class_weighted = ordinal_class_weighted

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # precompute matrix jarak antar kelas: dist[i,j] = |i - j|
        idx = torch.arange(num_classes).float()
        dist_matrix = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        self.register_buffer("dist_matrix", dist_matrix)

    def _make_soft_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Bangun target distribution per sampel berdasarkan jarak ordinal.
        targets: (B,) long tensor label index 0..num_classes-1
        return: (B, num_classes) soft target distribution
        """
        B = targets.size(0)
        device = targets.device

        # ambil baris jarak sesuai label tiap sampel -> (B, num_classes)
        dist = self.dist_matrix.to(device)[targets]  # (B, C)

        # bobot mentah meluruh terhadap jarak: w = 1 / (1 + dist^power), lalu di-zero-kan di posisi target
        raw_weight = 1.0 / (1.0 + dist.pow(self.distance_power))
        raw_weight = raw_weight.clone()
        raw_weight.scatter_(1, targets.unsqueeze(1), 0.0)  # nolkan posisi kelas benar dulu

        # normalisasi neighbor weight supaya total = smoothing_strength
        row_sum = raw_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        neighbor_mass = raw_weight / row_sum * self.smoothing_strength

        soft_targets = neighbor_mass.clone()
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing_strength)

        return soft_targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes) raw output dari model (belum softmax)
        targets: (B,) long tensor label index
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        soft_targets = self._make_soft_targets(targets)  # (B, C)

        # --- komponen 1: soft cross-entropy ---
        per_sample_ce = -(soft_targets * log_probs).sum(dim=1)  # (B,)

        if self.ce_class_weighted and self.class_weights is not None:
            sample_w = self.class_weights.to(targets.device)[targets]
            ce_loss = (per_sample_ce * sample_w).sum() / sample_w.sum()
        else:
            ce_loss = per_sample_ce.mean()

        # --- komponen 2: ordinal regression penalty ---
        # expected class index dari distribusi prediksi (soft argmax)
        class_idx = torch.arange(self.num_classes, device=logits.device).float()
        expected_class = (probs * class_idx.unsqueeze(0)).sum(dim=1)  # (B,)
        target_float = targets.float()

        ordinal_penalty = (expected_class - target_float).pow(2)  # (B,)

        if self.ordinal_class_weighted and self.class_weights is not None:
            sample_w = self.class_weights.to(targets.device)[targets]
            ordinal_loss = (ordinal_penalty * sample_w).sum() / sample_w.sum()
        else:
            ordinal_loss = ordinal_penalty.mean()

        total_loss = ce_loss + self.ordinal_weight * ordinal_loss
        return total_loss


# -----------------------------------------------------------------------------
# INTEGRASI KE train.py ANDA
# -----------------------------------------------------------------------------
# Di train.py, bagian [3/5] Setting up training components, ganti:
#
#   criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
#
# menjadi:
#
#   from ordinal_loss import OrdinalCrossEntropyLoss
#   criterion = OrdinalCrossEntropyLoss(
#       num_classes=NUM_CLASSES,
#       class_weights=class_weights,
#       distance_power=2.0,
#       smoothing_strength=0.1,
#       ordinal_weight=0.4,
#   ).to(DEVICE)
#
# Tidak ada perubahan lain yang diperlukan di train_one_epoch() atau validate()
# karena criterion(outputs, labels) dipanggil dengan signature yang sama persis.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Contoh pemakaian -- ganti baris loss Anda yang sekarang
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # SEBELUM (yang Anda pakai sekarang):
    # criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # SESUDAH:
    class_weights = torch.tensor([0.3, 1.2, 0.8, 2.5, 1.5])  # contoh -- pakai weight Anda yang sebenarnya

    criterion = OrdinalCrossEntropyLoss(
        num_classes=5,
        class_weights=class_weights,
        distance_power=2.0,     # mulai dari 2.0, bisa coba 1.0 atau 1.5 juga
        smoothing_strength=0.1, # sama seperti label_smoothing Anda sekarang
        ordinal_weight=0.4,     # mulai dari 0.4, tuning antara 0.2 - 0.6
    )

    # dummy test
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 2, 3, 4, 1, 2])

    loss = criterion(logits, targets)
    print("Loss:", loss.item())
    loss.backward()
    print("Backward OK, grad norm:", logits.grad.norm().item())

    # bandingkan dengan CE biasa untuk sanity check skala
    ce_baseline = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    print("CE baseline (skala referensi):", ce_baseline(logits.detach(), targets).item())