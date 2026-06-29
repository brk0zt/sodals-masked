# run_bci_inference.py
# PHASE 63: First Ignition (Mock BCI Inference Pipeline)
# Bu betik, SODALS Nörosembolik yığınını (Stack) baştan uca test eder.

import time
import torch
import torch.nn.functional as F

# Kendi yazdığımız modülleri içe aktarıyoruz
from bci_signal_bridge import BCIQuantumBridge
from nn.hsp_model import SODALSHybridPipeline

def build_brain_topology(channels: int, device: torch.device):
    """
    EEG elektrotları arasındaki bağlantıları (Connectome) simüle eder.
    Gerçek senaryoda bu, PLV (Phase Locking Value) ile dinamik hesaplanır.
    Şimdilik test için statik ve seyrek (sparse) bir komşuluk matrisi yaratıyoruz.
    """
    # Basit bir tam bağlı (fully connected) alt-küme veya rastgele ağ
    # Şimdilik her düğümü (elektrotu) rastgele 4 komşuya bağlıyoruz.
    src = torch.arange(channels).repeat_interleave(4)
    dst = torch.randint(0, channels, (channels * 4,))
    
    edge_index = torch.stack([src, dst], dim=0).to(device)
    
    # Bağlantı güçleri (Edge weights)
    edge_weight = torch.rand(edge_index.size(1), device=device)
    
    return edge_index, edge_weight

def main():
    print("="*60)
    print("SODALS BCI-HSP NÖROSEMBOLİK MOTORU: İLK ATEŞLEME")
    print("="*60)

    # CUDA aktif mi kontrol et, değilse CPU'da patlamasın diye uyar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[UYARI] CUDA bulunamadı! Triton kernelleri GPU gerektirir. Sistem CPU'da hata verebilir.")
    
    channels = 64
    in_features = 2  # Genlik (Amplitude) ve Faz (Phase)
    hidden_dim = 128
    out_dim = 5      # 5 Farklı Kognitif Durum / Karar sınıfı
    hd_dim = 1024

    # 1. BCI Köprüsünü Başlat (MOCK modunda çalışacak çünkü cihaz bağlı değil)
    bridge = BCIQuantumBridge(stream_type='MOCK_EEG', channels=channels)

    # 2. Uçtan Uca Modelimizi Başlat
    print("[SİSTEM] HSP Pipeline derleniyor...")
    model = SODALSHybridPipeline(
        in_dim=in_features, 
        hidden_dim=hidden_dim, 
        out_dim=out_dim, 
        hd_dim=hd_dim, 
        sparsity_threshold=0.15 # Düşük enerjili gürültüleri filtrele
    ).to(device)
    model.eval() # Inference moduna al

    # Ağ topolojisini yarat (64 EEG kanalı arasındaki bağlantılar)
    edge_index, edge_weight = build_brain_topology(channels, device)

    # 3. Sıcak Döngü (Hot Loop) İnfazı
    print("\n[SİSTEM] Canlı İnfaz Başlıyor (3 iterasyon)...\n")
    
    with torch.no_grad():
        for step in range(1, 4):
            start_time = time.perf_counter()

            # A. Beyin Dalgasını Oku (Complex64 Phasor Tensor)
            # Mock_eeg_chunk 10Hz Alpha dalgası ve gürültü üretecek
            quantum_phasor = bridge.fetch_and_transform(chunk_size=32)
            if quantum_phasor is None:
                continue

            # B. Complex64'ü donanımın anlayacağı Reel Feature'lara çevir
            amplitudes = torch.abs(quantum_phasor)
            phases = torch.angle(quantum_phasor)
            # Shape: [64, 2] -> 64 Node, her node için (Genlik, Faz)
            x_raw = torch.stack([amplitudes, phases], dim=-1).to(torch.float32).to(device)

            # C. Modeli Çalıştır (Derleme -> Triton SpMM -> HDC Bind -> Bundle -> Karar)
            logits = model(x_raw, edge_index, edge_weight)

            # D. Kararı Yorumla (Softmax)
            probabilities = F.softmax(logits, dim=-1).squeeze()
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item() * 100

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Raporlama
            print(f"--- Iterasyon {step} ---")
            print(f"[*] Sinyal İşleme ve Model Gecikmesi: {latency_ms:.2f} ms")
            print(f"[*] Kognitif Tahmin (Sınıf ID): {predicted_class} (Güven: {confidence:.1f}%)")
            print(f"[*] Çıktı Vektörü (Logits): {logits.cpu().numpy().round(3)}")
            print("-" * 25)
            
            time.sleep(0.5) # Çıktıları okuyabilmen için kısa bir bekleme

    print("\n[BAŞARI] Nörosembolik motor Mock veriyi başarıyla yuttu ve işledi!")

if __name__ == "__main__":
    main()