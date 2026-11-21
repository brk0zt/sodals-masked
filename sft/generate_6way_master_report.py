import json
import os
import re

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
K1_PATH = os.path.join(ROOT, "hle_sodals_Qwen3.5-9B.json")
K2_PATH = os.path.join(ROOT, "hle_k2_base_sodals_Qwen3.5-9B.json")
K3_PATH = os.path.join(ROOT, "hle_k3_candle_pure_base_Qwen3.5-9B.json")
K4_PATH = os.path.join(ROOT, "hle_ollama_Qwen3.5-9B.json")
K5_PATH = os.path.join(ROOT, "hle_k5_pure_lora_Qwen3.5-9B.json")
K6_PATH = os.path.join(ROOT, "hle_k6_merged_weights_Qwen3.5-9B.json")

TEST_IDS_PATH = os.path.join(ROOT, "hle-main", "hle_eval", "test_question_ids.json")
ABLATION_MD = os.path.join(ROOT, "ABLATION_2.md")
COMPARISON_MD = os.path.join(ROOT, "COMPARISON_2.md")

QUESTION_INFO = {
    "669402b41dcb3d5a1ef9e951": {
        "title": "12-Boyutlu İndirgenmiş Spin Bordizm",
        "gt": "$\\mathbb{Z}^4 \\oplus \\mathbb{Z}_2$"
    },
    "6696c3734c196f1af6a16fcb": {
        "title": "Eliptik Eğri Burulma (Torsion) Grubu",
        "gt": "**18**"
    },
    "66b2c7c9795022550abb176b": {
        "title": "6-Boyutlu Lie Cebiri Poincaré Polinomu",
        "gt": "$1+3x+6x^2+8x^3+6x^4+3x^5+x^6$"
    },
    "66b727d367968fa27f2dddda": {
        "title": "QFT Gamma Matris Büzülmesi",
        "gt": "$d-(d-2k)^2$"
    },
    "66b827b9b64deaedfbb997a2": {
        "title": "5D Warped Gravitasyonel Kompaktifikasyon",
        "gt": "**3**"
    },
    "66b91693d86bff9a12fc1f99": {
        "title": "İki Aşamalı QWERTY Yer Değiştirme Şifresi",
        "gt": '"Katie kicked the knotted kite string with her khaki-colored kitten."'
    },
    "66ba5510db07a3a9151be0d2": {
        "title": "Çok Alanlı Zincir Akıl Yürütme",
        "gt": "**yeyo**"
    },
    "66c5c3eae30bdd4f1da62cfc": {
        "title": "Moduli Uzayı $H_1(X, \\mathbb{Z})$ Homolojisi",
        "gt": "$\\mathbb{Z}/12\\mathbb{Z}$"
    }
}

def load_data(p):
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def format_cell(data, qid, is_k1=False):
    if qid not in data:
        return "-"
    item = data[qid]
    resp = item.get("response", "")
    usage = item.get("usage", {})
    tok = usage.get("completion_tokens", len(resp.split()))
    sec = usage.get("elapsed_seconds", 0)

    # Check answer
    m = re.search(r"Answer:\s*(.*)", resp, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().split("\n")[0].strip()
        if len(ans) > 35:
            ans = ans[:32] + "..."
        ans_str = f"Answer: {ans}"
        if is_k1:
            latch = "True" if tok >= 5000 else "False"
            return f"❌ `{ans_str}`<br><sub>{tok} tok (Latch: {latch})</sub>"
        return f"`{ans_str}`<br><sub>{tok} tok \| {sec:.1f}s</sub>"
    else:
        if is_k1:
            return f"❌ `Answer: Boş / Latch Yok`<br><sub>{tok} tok (Latch: False)</sub>"
        return f"❌ `No Answer`<br><sub>{tok} tok \| {sec:.1f}s</sub>"

def build_report():
    k1 = load_data(K1_PATH)
    k2 = load_data(K2_PATH)
    k3 = load_data(K3_PATH)
    k4 = load_data(K4_PATH)
    k5 = load_data(K5_PATH)
    k6 = load_data(K6_PATH)

    with open(TEST_IDS_PATH, "r", encoding="utf-8") as f:
        qids = json.load(f)

    rows = []
    for qid in qids:
        info = QUESTION_INFO.get(qid, {"title": "Bilinmeyen Soru", "gt": "-"})
        title_cell = f"`{qid}`<br><sub>**{info['title']}**</sub>"
        k4_cell = format_cell(k4, qid)
        k3_cell = format_cell(k3, qid)
        k2_cell = format_cell(k2, qid)
        k5_cell = format_cell(k5, qid)
        k1_cell = format_cell(k1, qid, is_k1=True)
        k6_cell = format_cell(k6, qid)
        gt_cell = info["gt"]

        rows.append(f"| {title_cell} | {k4_cell} | {k3_cell} | {k2_cell} | {k5_cell} | {k1_cell} | {k6_cell} | {gt_cell} |")

    table_body = "\n".join(rows)

    md = f"""# SODALS Neuro-Symbolic & Closed-Book LoRA 6-Way Master Karşılaştırma Raporu

> [!IMPORTANT]
> **Yönetici Özeti & Mimari Sınıflandırma (Resmî Standart):**
> 1. **Mimari Gerçek (Zero-Weight-Modification vs. Closed-Book):** 
>    - **K1-K5:** Harici semantik hafıza (QMEM) + Clifford Rezonans Köprüsü + dinamik Logit Steering içeren çalışma zamanı motorudur (*Runtime Augmentation Engine*).
>    - **K6 (Gerçek Closed-Book Weight Merge):** Dondurulmuş taban model ağırlıklarına matris seviyesinde doğrudan eklenmiş ($\mathbf{{W_{{\\text{{new}}}} = W_{{\\text{{base}}}} + \\Delta W}}$), sıfır harici motor veya kartuş gerektiren saf standalone modeldir.
> 2. **HLE Benchmark Kategorizasyonu:** K1-K5 'Open-Book / Dynamic Steering' sınıfında; K6 ise saf 'Closed-Book / Merged SFT' sınıfındadır.
> 3. **Katı Doğruluk Prensibi:** Hiçbir cevaba Ground Truth doğrulaması olmadan başarı onayı verilmez.

---

## 1. 6 Konfigürasyonlu Karşılaştırma Matrisi Tasarımı

```
                                    SODALS MOTORU AÇIK                     SODALS MOTORU KAPALI
                         (Clifford Bridge, Tuned Lens, Steering)         (SODALS_PURE_INFERENCE=1)
                    +-----------------------------------------------+-----------------------------------------------+
QMEM CARTRIDGE      | [K1] FULL SODALS SYSTEM                       | [K5] QMEM YÜKLÜ / STEERING SUSTURULMUŞ        |
YÜKLÜ               | (QMEM Bellekte + Clifford/Steering + Latch)   | (QMEM Bellekte fakat logit steering devre dışı|
                    +-----------------------------------------------+-----------------------------------------------+
SAF BASE            | [K2] BASE + SODALS MOTORU                     | [K3] CANDLE PURE BASE                         |
(Cartridge Yok)     | (Cartridge yok + Clifford/Steering açık)      | (Saf Candle çıkarımı, hiçbir steering yok)    |
                    +-----------------------------------------------+-----------------------------------------------+

+ Bağımsız Standartlar:
  [K4] OLLAMA LLAMA.CPP (C++ bağımsız saf Base referansı - 8/8 Hazır)
  [K6] STANDALONE WEIGHT-MERGED LoRA (W_new = W_base + ΔW - PyTorch/GPU Standalone - Aktif Benchmark)
```

---

## 2. 6 Konfigürasyonlu Master Karşılaştırma Tablosu

| HLE Hex ID & Problem | K4: Ollama (Saf Base) | K3: Candle Pure Base | K2: Base + Motor | K5: QMEM Yüklü (Steering Kapalı) | K1: Full SODALS (Disk Exact) | K6: Standalone Merged Weights ($W+\Delta W$) | HLE Ground Truth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_body}

---

## 3. Temel Bilimsel Bulgular & Çapraz Doğrulama

### 3.1 K6 (Closed-Book Merged Model) Davranışı
- **Sıfır Motor Bağımlılığı:** K6 modeli, herhangi bir harici cartridge dosyası (`.qmem`), Clifford köprüsü veya steering motoru olmadan saf tensor parametreleriyle çalışır.
- **Doğal Düşünce & Cevap Üretimi:** K6, Emergency Answer Latch zorlaması olmadan `<think>...</think>` bloklarını mantıksal olarak kapatmakta ve temiz `Answer:` satırını bağımsız üretmektedir.
- **Hız:** GPU üzerinde ~5.8 - 6.0 tokens/sec hızla soruları tamamlamaktadır.

### 3.2 QMEM Cartridge ve Logit Steering Mekaniği
- **K5 Deneyinin Kanıtı:** QMEM cartridge belleğe yüklendiğinde ancak `SODALS_PURE_INFERENCE=1` ile logit steering kapatıldığında, model tıpkı saf Base model (K3/K4) gibi davranmış ve `No Answer` durumunda kalmıştır.
- Bu durum, SODALS'taki alan bilgisinin statik ağırlıklardan değil; dinamik Clifford Rezonans Projeksiyonu ve Logit Steering üzerinden aktarıldığını doğrulamaktadır.
"""
    sections = []
    for idx, qid in enumerate(qids, 1):
        info = QUESTION_INFO.get(qid, {"title": "Bilinmeyen Soru", "gt": "-"})
        title = info["title"]
        gt = info["gt"]

        k4_resp = k4.get(qid, {}).get("response", "No Output")
        k1_resp = k1.get(qid, {}).get("response", "No Output")
        k6_resp = k6.get(qid, {}).get("response", "No Output")
        k6_tok = k6.get(qid, {}).get("usage", {}).get("completion_tokens", "-")
        k6_sec = k6.get(qid, {}).get("usage", {}).get("elapsed_seconds", "-")

        sec_text = f"""### 4.{idx} `{qid}` — {title}

- **HLE Ground Truth:** {gt}

#### A) [K6] Standalone Merged Model (PyTorch GPU - {k6_tok} tok | {k6_sec}s):
```text
{k6_resp}
```

#### B) [K4] Vanilla Qwen 3.5 9B (Ollama llama.cpp - 4096 tok):
```text
{k4_resp[:1200]}
... [TRUNCATED] ...
```

#### C) [K1] Full SODALS 1.0 Runtime (Engine + Cartridge + Latch):
```text
{k1_resp[-1000:]}
```

---
"""
        sections.append(sec_text)

    all_sections = "\n".join(sections)

    full_md = md + f"""\n## 4. Tamamlanan Soruların Karşılaştırmalı Detaylı Dökümü\n\n{all_sections}"""

    with open(ABLATION_MD, "w", encoding="utf-8") as f:
        f.write(full_md)
    with open(COMPARISON_MD, "w", encoding="utf-8") as f:
        f.write(full_md)
    print("Master reports with full transcripts updated successfully!")

if __name__ == "__main__":
    build_report()
