# 1. Aşama: NVIDIA CUDA Geliştirme Ortamı (Base Image)
# Ubuntu 22.04 + CUDA 12.1 + CuDNN
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder

# Gerekli araçları yükle
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Rust'ı yükle (Nightly gerekmez, Stable yeterli)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Çalışma dizini oluştur
WORKDIR /app

# Dependency Cache Katmanı
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# --- DÜZELTME BURADA ---
# nvidia-smi hatasını atlatmak için mimariyi elle giriyoruz.
# 86 = RTX 3000/4000 serisi ve A10/A40 gibi kartlar için uygundur.
ENV CUDA_COMPUTE_CAP=86

# 2. Aşama: Derleme
# Projenin tamamını kopyala
COPY . .

# Şimdi O BÜYÜLÜ KOMUTU çalıştır
RUN cargo build --release --bin swe_runner --features cuda

# 3. Aşama: Çalıştırma (Runtime Image)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Derlenmiş binary'i al
COPY --from=builder /app/target/release/swe_runner .

# Çalıştır
CMD ["./swe_runner"]