"""
SODALS v3.0 — Faz 3: Python HolographicProjector Training
==========================================================
Rust train_projector.rs'ın Python eşdeğeri.
PhysicsNeuron simülasyonu (Symplectic Euler) → Gerçek Qwen embedding hizalama.

Dual-amaçlı kayıp:
  L = L_cos + λ * L_mag
  L_cos = 1 - cos(projected_wave, target_embedding)
  L_mag = (||projected_wave|| - ||target_embedding||)²
"""

import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from safetensors.torch import load_file, save_file
import numpy as np
from typing import List, Tuple

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
REAL_EMBEDDINGS_PATH = os.path.join(SFT_DIR, "real_teacher_embeddings.safetensors")
PROJECTOR_OUTPUT_PATH = os.path.join(SFT_DIR, "holographic_projector_weights.pt")


# ════════════════════════════════════════════════════════════════════════════
# PhysicsNeuron — Python Port of Rust physics_neuron.rs
# ════════════════════════════════════════════════════════════════════════════

class PhysicsNeuron:
    """
    Rust PhysicsNeuron'un Python portu.
    Symplectic Euler integrator ile kararlı-durum dalga üretimi.

    Denklemler:
      a_t = F_input - k * y_t - b * v_t
      v_{t+1} = v_t + a_t * dt
      y_{t+1} = y_t + v_{t+1} * dt  (symplectic: güncellenmiş hızı kullan)
    """
    def __init__(self, k: float = 1.0, b: float = 0.1, dt: float = 0.01):
        self.y = complex(0.0, 0.0)  # Pozisyon/Amplitüd+Faz
        self.v = complex(0.0, 0.0)  # Hız
        self.k = k   # Yay sabiti / rezonans frekansı
        self.b = b   # Sönümleme katsayısı
        self.dt = dt  # Zaman adımı

    def forward(self, input_force: complex) -> complex:
        """Tek adım Symplectic Euler."""
        accel = input_force - self.k * self.y - self.b * self.v
        self.v = self.v + accel * self.dt
        self.y = self.y + self.v * self.dt
        return self.y

    def simulate_word(self, word: str, dim: int = 256, steps: int = 50) -> List[complex]:
        """
        Bir kelime için dim boyutlu kompleks dalga üret.
        Her boyut farklı bir PhysicsNeuron olarak simüle edilir.
        """
        import hashlib
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        wave = []

        for d in range(dim):
            # Her boyut için benzersiz parametre
            seed = (h + d * 7919) % (2**32)
            k = 0.5 + (seed % 100) / 100.0  # k ∈ [0.5, 1.5]
            b = 0.05 + (seed % 50) / 500.0   # b ∈ [0.05, 0.15]

            neuron = PhysicsNeuron(k=k, b=b, dt=0.01)

            # Giriş kuvveti: kelime hash'ından türetilmiş
            force_re = math.sin(seed * 0.0001)
            force_im = math.cos(seed * 0.0003)
            force = complex(force_re, force_im)

            # 50 adım simülasyon
            for _ in range(steps):
                neuron.forward(force)

            wave.append(neuron.y)

        return wave


# ════════════════════════════════════════════════════════════════════════════
# HolographicProjector — Python Port of Rust interface_layer.rs
# ════════════════════════════════════════════════════════════════════════════

class HolographicProjector(nn.Module):
    """
    Π(Ψ) → V(Ψ): Kompleks dalga → gerçek embedding projeksiyonu.
    Rust HolographicProjector'un PyTorch portu.

    Architecture:
      proj_real: Maps [Re, |z|] → embedding_dim
      proj_imag: Maps [Im, arg(z)] → embedding_dim
      Output: sqrt(proj_real² + proj_imag²)  (magnitude-preserving)
    """
    def __init__(self, wave_dim: int = 256, embedding_dim: int = 3584):
        super().__init__()
        # Her kompleks sayı = 2 gerçek bileşen (Re, Im) + 2 türetilmiş (|z|, arg(z))
        input_dim = wave_dim * 2  # Sadece Re ve Im
        self.proj_real = nn.Linear(input_dim, embedding_dim)
        self.proj_imag = nn.Linear(input_dim, embedding_dim)

    def forward(self, wave_re: torch.Tensor, wave_im: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave_re: [batch, wave_dim] - Gerçek bileşenler
            wave_im: [batch, wave_dim] - Sanal bileşenler
        Returns:
            embedding: [batch, embedding_dim] - Projekte edilmiş gerçek vektör
        """
        # Concatenate Re ve Im
        x_real = torch.cat([wave_re, torch.sqrt(wave_re**2 + wave_im**2 + 1e-8)], dim=-1)
        x_imag = torch.cat([wave_im, torch.atan2(wave_im, wave_re + 1e-8)], dim=-1)

        # Parallel projection
        proj_r = self.proj_real(x_real)
        proj_i = self.proj_imag(x_imag)

        # Magnitude-preserving combination
        output = torch.sqrt(proj_r**2 + proj_i**2 + 1e-8)
        return output


def prepare_training_data(real_embeddings: dict, wave_dim: int = 256
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Her kelime için:
    1. PhysicsNeuron simülasyonu → kompleks dalga (student)
    2. Gerçek Qwen embedding (teacher)
    """
    physics = PhysicsNeuron()
    waves_re = []
    waves_im = []
    targets = []

    words = list(real_embeddings.keys())
    print(f"[DATA] Preparing {len(words)} wave-embedding pairs...")

    for i, word in enumerate(words):
        # Student: PhysicsNeuron dalga üretimi
        wave = physics.simulate_word(word, dim=wave_dim, steps=50)
        re_vec = [c.real for c in wave]
        im_vec = [c.imag for c in wave]
        waves_re.append(re_vec)
        waves_im.append(im_vec)

        # Teacher: Gerçek Qwen embedding
        targets.append(real_embeddings[word])

    waves_re_t = torch.tensor(waves_re, dtype=torch.float32)
    waves_im_t = torch.tensor(waves_im, dtype=torch.float32)
    targets_t = torch.stack(targets)

    print(f"  Waves shape: {waves_re_t.shape}")
    print(f"  Targets shape: {targets_t.shape}")

    return waves_re_t, waves_im_t, targets_t


def train_projector():
    print("=" * 80)
    print("  SODALS v3.0 — TRAINING HOLOGRAPHIC PROJECTOR WITH REAL EMBEDDINGS")
    print("=" * 80)

    # 1. Gerçek embedding'leri yükle
    if not os.path.exists(REAL_EMBEDDINGS_PATH):
        print(f"[ERROR] Real embeddings not found at {REAL_EMBEDDINGS_PATH}")
        print("  Run export_real_teacher_embeddings.py first!")
        return

    real_embeddings = load_file(REAL_EMBEDDINGS_PATH)
    embedding_dim = list(real_embeddings.values())[0].shape[0]
    print(f"[LOAD] {len(real_embeddings)} real embeddings loaded, dim={embedding_dim}")

    # 2. Eğitim verisini hazırla
    wave_dim = 256
    waves_re, waves_im, targets = prepare_training_data(real_embeddings, wave_dim)

    # 3. Projector oluştur
    projector = HolographicProjector(wave_dim=wave_dim, embedding_dim=embedding_dim)
    optimizer = AdamW(projector.parameters(), lr=1e-4, weight_decay=0.01)
    lambda_mag = 0.5

    # 4. Eğitim döngüsü
    batch_size = 32
    num_epochs = 200
    n_samples = waves_re.shape[0]
    best_loss = float("inf")

    print(f"\n[TRAIN] Starting alignment training: {num_epochs} epochs, "
          f"batch={batch_size}, lr=1e-4, λ_mag={lambda_mag}")

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        waves_re_s = waves_re[perm]
        waves_im_s = waves_im[perm]
        targets_s = targets[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_re = waves_re_s[i:i + batch_size]
            batch_im = waves_im_s[i:i + batch_size]
            batch_target = targets_s[i:i + batch_size]

            projected = projector(batch_re, batch_im)

            # Cosine loss
            cos_sim = F.cosine_similarity(projected, batch_target, dim=-1)
            loss_cos = (1 - cos_sim).mean()

            # Magnitude loss
            proj_norm = projected.norm(dim=-1)
            target_norm = batch_target.norm(dim=-1)
            loss_mag = ((proj_norm - target_norm) ** 2).mean()

            loss = loss_cos + lambda_mag * loss_mag

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: Loss={avg_loss:.6f} "
                  f"(cos={loss_cos.item():.6f}, mag={loss_mag.item():.6f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(projector.state_dict(), PROJECTOR_OUTPUT_PATH)

    print(f"\n[SUCCESS] Best projector saved to {PROJECTOR_OUTPUT_PATH}")
    print(f"  Best loss: {best_loss:.6f}")

    # 5. Doğrulama
    projector.load_state_dict(torch.load(PROJECTOR_OUTPUT_PATH, weights_only=True))
    projector.eval()
    with torch.no_grad():
        proj_all = projector(waves_re, waves_im)
        cos_sims = F.cosine_similarity(proj_all, targets, dim=-1)
        print(f"\n  Final Alignment Quality:")
        print(f"    Mean Cosine Similarity: {cos_sims.mean().item():.4f}")
        print(f"    Min  Cosine Similarity: {cos_sims.min().item():.4f}")
        print(f"    Max  Cosine Similarity: {cos_sims.max().item():.4f}")
        print(f"    Std  Cosine Similarity: {cos_sims.std().item():.4f}")


if __name__ == "__main__":
    train_projector()
