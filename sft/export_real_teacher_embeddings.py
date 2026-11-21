"""
SODALS v3.0 — Faz 3: Real Teacher Embedding Extractor
======================================================
Rust train_projector.rs'ın "from_vocabulary_simulated()" fonksiyonunun
ürettiği hash-tabanlı SAHTE embedding'leri değil, gerçek Qwen modelinin
embed_tokens katmanından çıkarılan GERÇEK embedding'leri dışa aktarır.

Bu dosya, HolographicProjector'un gerçek veriye hizalanması için
teacher sinyallerini üretir.
"""

import os
import sys
import json
import torch
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
CURRICULUM_PATH = os.path.join(SFT_DIR, "k7_sft_unified_curriculum.jsonl")
OUTPUT_PATH = os.path.join(SFT_DIR, "real_teacher_embeddings.safetensors")


def extract_vocabulary_from_curriculum(jsonl_path: str, tokenizer) -> list:
    """
    SFT müfredatındaki tüm benzersiz kelimeleri çıkar.
    Her kelime için tokenizer'ın ürettiği token ID'lerini sakla.
    """
    word_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for msg in item.get("messages", []):
                content = msg.get("content", "")
                # Basit tokenizasyon: boşluk ve noktalama ile böl
                words = content.split()
                for w in words:
                    clean = w.strip(".,;:!?()[]{}\"'`$\\/#@%^&*+=<>~|")
                    if len(clean) >= 2 and len(clean) <= 50:
                        word_counter[clean] += 1

    # En sık 2000 kelimeyi al
    top_words = [w for w, c in word_counter.most_common(2000)]
    print(f"[VOCAB] Extracted {len(top_words)} unique words from {jsonl_path}")
    return top_words


def extract_real_embeddings(model, tokenizer, words: list, device="cpu") -> dict:
    """
    Her kelime için Qwen'in embed_tokens katmanından gerçek embedding çıkar.
    Token-mean pooling uygula.

    Bu, train_projector.rs'taki extract_real_embedding() fonksiyonunun
    Python eşdeğeridir.
    """
    embed_layer = model.model.embed_tokens  # Qwen'in embedding katmanı
    embeddings = {}
    hidden_dim = model.config.hidden_size

    print(f"[EMBED] Extracting real {hidden_dim}-D embeddings for {len(words)} words...")

    for i, word in enumerate(words):
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            continue

        input_tensor = torch.tensor([token_ids], device=device)
        with torch.no_grad():
            word_embeds = embed_layer(input_tensor)  # [1, n_tokens, hidden_dim]
            # Token-mean pooling (birden fazla token olabilir)
            mean_embed = word_embeds.mean(dim=1).squeeze(0)  # [hidden_dim]

        embeddings[word] = mean_embed.cpu().to(torch.float32)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(words)}] Processed...")

    print(f"[EMBED] Successfully extracted {len(embeddings)} real embeddings")
    return embeddings


def main():
    print("=" * 80)
    print("  SODALS v3.0 — EXTRACTING REAL TEACHER EMBEDDINGS FROM QWEN")
    print("=" * 80)

    # 1. Tokenizer yükle
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 2. Müfredattan kelime çıkar
    words = extract_vocabulary_from_curriculum(CURRICULUM_PATH, tokenizer)

    # 3. Model yükle (CPU, sadece embed_tokens katmanı lazım)
    print("\nLoading Qwen model (CPU, for embed_tokens only)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 4. Gerçek embedding'leri çıkar
    real_embeddings = extract_real_embeddings(model, tokenizer, words, device="cpu")

    # 5. SafeTensors olarak kaydet
    save_file(real_embeddings, OUTPUT_PATH)
    print(f"\n[SUCCESS] Real teacher embeddings saved to {OUTPUT_PATH}")
    print(f"  Words: {len(real_embeddings)}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")

    # 6. Doğrulama: Hash-tabanlı sahte vs gerçek karşılaştırma
    print("\n--- Verification: Real vs Simulated Embedding Quality ---")
    sample_words = list(real_embeddings.keys())[:5]
    for w in sample_words:
        real_emb = real_embeddings[w]
        # Simulated (hash-based, train_projector.rs L100 ile aynı)
        import hashlib
        h = int(hashlib.sha256(w.encode()).hexdigest()[:16], 16)
        sim_emb = torch.tensor([(h + i) * 0.1 for i in range(real_emb.shape[0])],
                               dtype=torch.float32)
        sim_emb = torch.sin(sim_emb) * 0.1

        cos_sim = F.cosine_similarity(real_emb.unsqueeze(0), sim_emb.unsqueeze(0)).item()
        print(f"  '{w}': cos_sim(real, simulated) = {cos_sim:.4f} "
              f"(should be ~0.0 if simulated is noise)")

    del model
    print("\n🎉 Real teacher embeddings extraction complete!")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
