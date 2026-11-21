"""
SODALS v2.1 — Master Interleaved (Domain 1 + Domain 2) Curriculum Builder
==========================================================================
Merges and homogeneously interleaves:
  - 144 Domain 1 STEM samples (Real Analysis, Topology, Homological Algebra)
  - 160 Domain 2 Abstract Algebra samples (with corrected Lie algebra derivations)
Total: 304 samples.

Guarantees:
  1. sp(6) correctly derives dim = 3 * 7 = 21 (prevents 6 * 7 = 42).
  2. sp(4) correctly derives dim = 2 * 5 = 10 (prevents so(4) = 6).
  3. E_6 strictly separates |Φ| = 72 from dim = 6 + 72 = 78 (prevents 6 + 78 = 84).
  4. E_7 strictly separates |Φ| = 126 from dim = 7 + 126 = 133 (prevents 63 trap).
  5. Aut(Q_8) = 24, E_8 = 248, so(8) = 28, Biquadratic = 4 proofs preserved.
  6. Homogeneous 1:1 interleaving (seed=42) prevents catastrophic sequential interference!

Output: sft/master_interleaved_d1_d2_304_curriculum.jsonl
"""

import json
import os
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
D1_PATH = os.path.join(SFT_DIR, "domain1_sft_curriculum.jsonl")
D2_PATH = os.path.join(SFT_DIR, "domain2_sft_curriculum.jsonl")
OUTPUT_PATH = os.path.join(SFT_DIR, "master_interleaved_d1_d2_304_curriculum.jsonl")


def get_corrected_thought_and_answer(topic_id: str):
    """Returns corrected thought and answer for the 4 target Lie topics."""
    if "sp6" in topic_id:
        thought = (
            "Arkhe İzi:\n"
            "  Bilinç (Aksiyom ve İndis Ayrımı):\n"
            "    Simplektik Lie cebiri \\mathfrak{sp}(2n, \\mathbb{C}), 2n \\times 2n antisimetrik simplektik formu J koruyan matrislerdir.\n"
            "    Burada matris boyutu 2n = 6'dır, bu da Cartan rankının n = 3 olduğunu belirler (Cartan tipi C_3).\n"
            "    KRİTİK İNDİS AYRIMI: Rank n = 3'tür, 6 DEĞİLDİR. n yerine 2n kullanmak feci bir katsayı hatası üretir.\n"
            "  Benzerlik (Blok Matris Ayrışımı):\n"
            "    X matrisi 3 \\times 3 bloklara ayrılır: X = \\begin{pmatrix} A & B \\\\ C & D \\end{pmatrix}.\n"
            "    Simplektik kısıt X^T J + J X = 0 gereği: D = -A^T, B = B^T (simetrik), C = C^T (simetrik).\n"
            "  Çıkarım (Boyut ve Parametre Sayımı):\n"
            "    - Blok A: \\mathfrak{gl}(3, \\mathbb{C}) içinde serbest \\implies n^2 = 3^2 = 9 serbest parametre.\n"
            "    - Blok B: 3 \\times 3 simetrik \\implies \\frac{n(n+1)}{2} = \\frac{3 \\times 4}{2} = 6 serbest parametre.\n"
            "    - Blok C: 3 \\times 3 simetrik \\implies \\frac{n(n+1)}{2} = \\frac{3 \\times 4}{2} = 6 serbest parametre.\n"
            "    - Blok D: -A^T ile belirlenir \\implies 0 serbest parametre.\n"
            "    Toplam Boyut = 9 + 6 + 6 = 21.\n"
            "    Genel formül: \\dim(\\mathfrak{sp}(2n)) = n(2n+1). n = 3 için: 3(2 \\times 3 + 1) = 3 \\times 7 = 21.\n"
            "  Şartlar (Hata Çürütme):\n"
            "    6 \\times 7 = 42 hesabı tamamen yanlıştır; çünkü 6 matris boyutudur (2n), rank (n) 3'tür.\n"
            "Cevap: 21."
        )
        return thought, "21"

    elif "sp4" in topic_id:
        thought = (
            "Arkhe İzi:\n"
            "  Bilinç:\n"
            "    \\mathfrak{sp}(4, \\mathbb{C}) simplektik Lie cebirinde matris boyutu 2n = 4 olup Cartan rankı n = 2'dir (Cartan tipi C_2).\n"
            "  Benzerlik:\n"
            "    2 \\times 2 bloklar halinde X = \\begin{pmatrix} A & B \\\\ C & -A^T \\end{pmatrix}, B = B^T, C = C^T.\n"
            "  Çıkarım (Boyut):\n"
            "    - A: 2^2 = 4 parametre.\n"
            "    - B: \\frac{2 \\times 3}{2} = 3 parametre.\n"
            "    - C: \\frac{2 \\times 3}{2} = 3 parametre.\n"
            "    Toplam Boyut = 4 + 3 + 3 = 10.\n"
            "    Formül: n(2n+1) = 2(2 \\times 2 + 1) = 2 \\times 5 = 10.\n"
            "  Şartlar (Dynkin İzomorfizması ve Yanılgı):\n"
            "    n(2n-1) = 2 \\times 3 = 6 formülü ortogonal cebir \\mathfrak{so}(4) (tip D_2)'ye aittir.\n"
            "    Simplektik C_2 ise B_2 = \\mathfrak{so}(5) ile izomorftur ve boyutu kesinlikle 10'dur.\n"
            "Cevap: 10."
        )
        return thought, "10"

    elif "e6" in topic_id:
        thought = (
            "Arkhe İzi:\n"
            "  Bilinç:\n"
            "    E_6 istisnai basit Lie cebirinin rankı r = 6'dır (\\dim(\\mathfrak{h}) = 6).\n"
            "  Benzerlik (Kök Sistemi Cardinalitesi — KESİN AYRIM):\n"
            "    E_6 kök sistemi \\Phi:\n"
            "    - Pozitif kökler: |\\Phi^+| = 36.\n"
            "    - Negatif kökler: |\\Phi^-| = 36.\n"
            "    - Toplam kök sayısı: |\\Phi| = 2 \\times |\\Phi^+| = 2 \\times 36 = 72 kök.\n"
            "    KRİTİK ARİTMETİK UYARI: 2 \\times 36 = 72'dir. Kök sayısı 78 DEĞİLDİR!\n"
            "    78, rank eklendikten sonraki nihai Lie cebiri boyutudur.\n"
            "  Çıkarım (Cartan-Weyl Ayrışımı):\n"
            "    \\dim(E_6) = \\text{rank}(E_6) + |\\Phi| = 6 + 72 = 78.\n"
            "    (Çürütme: |\\Phi| = 78 alıp 6 + 78 = 84 yazmak çembersel bir akıl yürütme hatasıdır; gerçek boyut 78'dir).\n"
            "Cevap: 78."
        )
        return thought, "78"

    elif "e7" in topic_id:
        thought = (
            "Arkhe İzi:\n"
            "  Bilinç:\n"
            "    E_7 istisnai basit Lie cebirinin rankı r = 7'dir (\\dim(\\mathfrak{h}) = 7).\n"
            "  Benzerlik (Kök Sistemi Cardinalitesi — KESİN AYRIM):\n"
            "    E_7 kök sistemi \\Phi:\n"
            "    - Pozitif kök sayısı: |\\Phi^+| = 63.\n"
            "    - Negatif kök sayısı: |\\Phi^-| = 63.\n"
            "    - Toplam kök sayısı: |\\Phi| = 2 \\times 63 = 126 kök.\n"
            "    FALSIFICATION PASS: 63 yalnızca pozitif kök sayısıdır, toplam kök veya boyut değildir.\n"
            "  Çıkarım (Cartan-Weyl Ayrışımı):\n"
            "    \\dim(E_7) = \\text{rank}(E_7) + |\\Phi| = 7 + 126 = 133.\n"
            "Cevap: 133."
        )
        return thought, "133"

    return None, None


def build_interleaved_curriculum():
    print("=" * 80)
    print("BUILDING MASTER INTERLEAVED (DOMAIN 1 + DOMAIN 2) CURRICULUM")
    print("=" * 80)

    # 1. Load Domain 1 (144 samples)
    d1_samples = []
    with open(D1_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["metadata"] = item.get("metadata", {})
                item["metadata"]["domain"] = "Domain_1_Analysis_Topology"
                d1_samples.append(item)
    print(f"[D1] Loaded {len(d1_samples)} Domain 1 samples from {D1_PATH}")

    # 2. Load and correct Domain 2 (160 samples)
    d2_samples = []
    corrected_count = 0
    with open(D2_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sample_id = item["id"]
                thought, answer = get_corrected_thought_and_answer(sample_id)
                if thought is not None:
                    # Update assistant message with corrected first-principles thought
                    user_msg = item["messages"][0]["content"]
                    item["messages"] = [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"<think>\n{thought}\n</think>\n\nAnswer: {answer}"}
                    ]
                    corrected_count += 1

                item["metadata"] = item.get("metadata", {})
                item["metadata"]["domain"] = "Domain_2_Abstract_Algebra"
                d2_samples.append(item)
    print(f"[D2] Loaded {len(d2_samples)} Domain 2 samples from {D2_PATH} (Corrected {corrected_count} Lie samples)")

    # 3. Interleave homogeneously (1:1 balanced distribution)
    # Using fixed seed for deterministic reproducibility
    random.seed(42)

    interleaved = []
    max_len = max(len(d1_samples), len(d2_samples))
    # Shuffle each domain individually first
    random.shuffle(d1_samples)
    random.shuffle(d2_samples)

    # Round-robin interleave
    i1, i2 = 0, 0
    while i1 < len(d1_samples) or i2 < len(d2_samples):
        if i1 < len(d1_samples):
            interleaved.append(d1_samples[i1])
            i1 += 1
        if i2 < len(d2_samples):
            interleaved.append(d2_samples[i2])
            i2 += 1

    print(f"[MERGE] Total Interleaved Samples: {len(interleaved)} (D1: {i1}, D2: {i2})")

    # 4. Save to master curriculum file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in interleaved:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved {len(interleaved)} samples to {OUTPUT_PATH}!")


if __name__ == "__main__":
    build_interleaved_curriculum()
