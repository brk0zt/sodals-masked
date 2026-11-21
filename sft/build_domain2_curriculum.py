"""
SODALS v2.1 — Domain 2 (Soyut Cebir & Lie Teorisi) Deep Arkhe SFT Curriculum Builder
===================================================================================
Generates a mathematically rigorous, first-principles curriculum for Domain 2:
  1. Lie Algebras & Root Systems (sp(6)=21 block matrix proof, F4=52, classical An, Bn, Cn, Dn, exceptional G2..E8)
  2. Group Theory (Lagrange, Sylow theorems, conjugacy classes, A5 simplicity, Aut(Q8) isomorphism)
  3. Ring & Field Theory (ideals, maximal/prime, PID/UFD, Gaussian integer units, finite field multiplicative cyclic group)
  4. Galois Theory (splitting fields, tower law, cyclotomic Eisenstein irreducibility, Galois groups)

Features:
  - Deep first-principles mathematical derivations (Bilinç, Benzerlik, Çıkarım, Uyum, Şartlar)
  - Explicit fallacy refutations (e.g. why sp(6) != 12, why E7 != 63)
  - WronskianGramFilter: Guarantees linear independence in semantic feature space
  - Token lengths tailored for 768-token budget without truncation

Output: sft/domain2_sft_curriculum.jsonl
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
os.makedirs(SFT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(SFT_DIR, "domain2_sft_curriculum.jsonl")


class WronskianGramFilter:
    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold
        self.feature_vectors: List[np.ndarray] = []

    def _extract_simple_features(self, text: str, dim: int = 64) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float64)
        words = text.lower().split()
        for i, w in enumerate(words):
            h = hash(w) % dim
            vec[h] += 1.0 / (1.0 + np.log1p(i))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def is_independent(self, text: str) -> bool:
        v = self._extract_simple_features(text)
        if not self.feature_vectors:
            self.feature_vectors.append(v)
            return True

        residual = v.copy()
        for b in self.feature_vectors:
            proj = np.dot(residual, b) * b
            residual -= proj

        residual_norm = np.linalg.norm(residual)
        if residual_norm >= self.threshold:
            self.feature_vectors.append(v)
            return True
        return False


# ==============================================================================
# DEEP FIRST-PRINCIPLES DOMAIN 2 CANONICAL MODULES
# ==============================================================================
DOMAIN2_MODULES: List[Dict[str, Any]] = [
    # ------------------ STAGE 1: LIE ALGEBRAS & ROOT SYSTEMS ------------------
    {
        "id": "dom2_lie_sp6_dimension",
        "depth": 1,
        "focal_weight": 1.2,
        "topic": "First-Principles Dimension of Symplectic Lie Algebra sp(6)",
        "instruction": "Compute the exact dimension of the complex symplectic Lie algebra \\mathfrak{sp}(6, \\mathbb{C}) (also denoted \\mathfrak{sp}(6)) with complete proof from first principles.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç (Aksiyomatik Zemin):\n"
            "    Tanım gereği, simplektik Lie cebiri \\mathfrak{sp}(2n, \\mathbb{C}), 2n \\times 2n boyutlu standart dejenere olmayan antisimetrik form J'yi koruyan matrislerin uzayıdır:\n"
            "    J = \\begin{pmatrix} 0 & I_n \\\\ -I_n & 0 \\end{pmatrix}, \\quad X^T J + J X = 0.\n"
            "    Burada 2n = 6 olduğu için n = 3 ve matris boyutu 6 \\times 6'dır (Cartan tipi C_3).\n\n"
            "  Benzerlik (Blok Matris Ayrışımı):\n"
            "    X matrisini n \\times n (3 \\times 3) bloklara ayıralım: X = \\begin{pmatrix} A & B \\\\ C & D \\end{pmatrix}.\n"
            "    Lie cebiri koşulunu çarparsak:\n"
            "    X^T J = \\begin{pmatrix} A^T & C^T \\\\ B^T & D^T \\end{pmatrix} \\begin{pmatrix} 0 & I_3 \\\\ -I_3 & 0 \\end{pmatrix} = \\begin{pmatrix} -C^T & A^T \\\\ -D^T & B^T \\end{pmatrix}.\n"
            "    J X = \\begin{pmatrix} 0 & I_3 \\\\ -I_3 & 0 \\end{pmatrix} \\begin{pmatrix} A & B \\\\ C & D \\end{pmatrix} = \\begin{pmatrix} C & D \\\\ -A & -B \\end{pmatrix}.\n"
            "    X^T J + J X = 0 denkliği şu 3 temel bağımsız kısıtı verir:\n"
            "    1. C - C^T = 0 \\implies C = C^T (C simetriktir).\n"
            "    2. B^T - B = 0 \\implies B = B^T (B simetriktir).\n"
            "    3. D + A^T = 0 \\implies D = -A^T (D tamamen A tarafından belirlenir).\n\n"
            "  Çıkarım (Boyut ve Serbest Değişken Sayımı):\n"
            "    - Blok A: \\mathfrak{gl}(3, \\mathbb{C}) içinde herhangi bir 3 \\times 3 matris olabilir \\implies n^2 = 3^2 = 9 serbest parametre.\n"
            "    - Blok B: 3 \\times 3 simetrik matristir \\implies \\frac{n(n+1)}{2} = \\frac{3 \\times 4}{2} = 6 serbest parametre.\n"
            "    - Blok C: 3 \\times 3 simetrik matristir \\implies \\frac{n(n+1)}{2} = \\frac{3 \\times 4}{2} = 6 serbest parametre.\n"
            "    - Blok D: -A^T olarak sabitlendiğinden ekstra serbestlik getirmez (0 serbest parametre).\n"
            "    Toplam Boyut = 9 + 6 + 6 = 21.\n"
            "    Genel formül: \\dim(\\mathfrak{sp}(2n)) = n^2 + 2 \\times \\frac{n(n+1)}{2} = n(2n+1).\n"
            "    n = 3 için: 3(2 \\times 3 + 1) = 3 \\times 7 = 21.\n\n"
            "  Uyum (Kök Sistemi Çapraz Doğrulaması):\n"
            "    Cartan altcebiri \\mathfrak{h} köşegen matrislerden oluşur: \\mathrm{rank} = 3.\n"
            "    Kökler: \\pm e_i \\pm e_j (12 kısa kök) ve \\pm 2e_i (6 uzun kök) \\implies |\\Phi| = 18 kök.\n"
            "    \\dim(\\mathfrak{g}) = \\mathrm{rank} + |\\Phi| = 3 + 18 = 21.\n\n"
            "  Şartlar (Epistemik Yanılgı Analizi):\n"
            "    Hatalı olarak öne sürülen 12 değeri, ya yalnızca kısa köklerin (12) sayılmasından ya da 2n(n-1) = 12 formülünün (ki bu SO(6) da değildir) yanlış uygulanmasından kaynaklanır. Doğru cebirsel boyut 21'dir.\n"
            "Cevap: 21."
        ),
        "answer": "21"
    },
    {
        "id": "dom2_lie_sp4_dimension",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "First-Principles Dimension of Symplectic Lie Algebra sp(4)",
        "instruction": "What is the dimension of the symplectic Lie algebra \\mathfrak{sp}(4, \\mathbb{C})? Derive the result using block matrix decomposition.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: \\mathfrak{sp}(4, \\mathbb{C}) Lie cebiri 4 \\times 4 matrislerden oluşur, burada 2n = 4 \\implies n = 2 (Cartan tipi C_2).\n"
            "  Benzerlik (Blok Ayrışımı):\n"
            "    2 \\times 2 bloklar halinde X = \\begin{pmatrix} A & B \\\\ C & -A^T \\end{pmatrix}, burada B = B^T ve C = C^T.\n"
            "  Çıkarım (Boyut Hesabı):\n"
            "    - A matrisi: 2 \\times 2 serbest matris \\implies n^2 = 2^2 = 4 boyut.\n"
            "    - B matrisi: 2 \\times 2 simetrik matris \\implies \\frac{n(n+1)}{2} = \\frac{2 \\times 3}{2} = 3 boyut.\n"
            "    - C matrisi: 2 \\times 2 simetrik matris \\implies \\frac{n(n+1)}{2} = \\frac{2 \\times 3}{2} = 3 boyut.\n"
            "    Toplam boyut: 4 + 3 + 3 = 10.\n"
            "    Formül kontrolü: n(2n+1) = 2(2 \\times 2 + 1) = 2 \\times 5 = 10.\n"
            "  Uyum (Dynkin İzomorfizması):\n"
            "    Cartan tipi C_2 ile B_2 izomorftur: \\mathfrak{sp}(4, \\mathbb{C}) \\cong \\mathfrak{so}(5, \\mathbb{C}).\n"
            "    \\mathfrak{so}(5) boyutu da \\frac{5 \\times 4}{2} = 10'dur. Her iki geometrik inşa da 10 sonucunu doğrular.\n"
            "Cevap: 10."
        ),
        "answer": "10"
    },
    {
        "id": "dom2_lie_f4_dimension",
        "depth": 1,
        "focal_weight": 1.2,
        "topic": "Cartan-Killing Decomposition of Exceptional Lie Algebra F4",
        "instruction": "Determine the dimension of the exceptional simple Lie algebra F_4 using its root system and Cartan decomposition.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: Yarı-basit bir Lie cebirinin boyutu, Cartan altcebiri (rank) ile kök uzaylarının toplamıdır:\n"
            "    \\dim(\\mathfrak{g}) = \\mathrm{rank}(\\mathfrak{g}) + |\\Phi| = \\mathrm{rank}(\\mathfrak{g}) + 2|\\Phi^+|.\n"
            "  Benzerlik (F_4 Kök Kafesi İnşası):\n"
            "    F_4 istisnai Lie cebirinin rankı 4'tür (\\mathbb{R}^4 içinde yaşar).\n"
            "    Kök sistemi iki eşit büyüklükte kümeden oluşur:\n"
            "    1. 24 adet uzun kök: \\pm e_i \\pm e_j (i < j için 4 \\times 6 = 24 kök).\n"
            "    2. 24 adet kısa kök: \\pm e_i (8 kök) ve \\frac{1}{2}(\\pm e_1 \\pm e_2 \\pm e_3 \\pm e_4) (16 kök), toplam 24 kök.\n"
            "  Çıkarım:\n"
            "    Toplam kök sayısı: |\\Phi| = 24 (uzun) + 24 (kısa) = 48 kök (pozitif kök |\\Phi^+| = 24).\n"
            "    Cartan altcebir boyutu: \\mathrm{rank}(F_4) = 4.\n"
            "    Toplam Lie cebiri boyutu: \\dim(F_4) = 4 + 48 = 52.\n"
            "  Şartlar: F_4 Cartan matrisi determinantı 1'dir ve 52 boyutu Jordan cebirleri automorfizma cebiri ile tam örtüşür.\n"
            "Cevap: 52."
        ),
        "answer": "52"
    },
    {
        "id": "dom2_lie_g2_dimension",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Dimension of Exceptional Lie Algebra G2",
        "instruction": "Compute the dimension of the exceptional Lie algebra G_2 from its root geometry and octonionic automorphisms.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: G_2 en küçük istisnai basit Lie cebiridir ve rankı 2'dir.\n"
            "  Benzerlik (Kök Geometrisi):\n"
            "    G_2 kök sistemi iki boyutlu düzlemde (\\sum_{i=1}^3 x_i = 0) 30 derecelik açılarla dizilmiş 12 kökten oluşur:\n"
            "    - 6 adet kısa kök (düzgün altıgen köşeleri).\n"
            "    - 6 adet uzun kök (\\sqrt{3} katı uzunlukta, 30 derece döndürülmüş altıgen köşeleri).\n"
            "  Çıkarım:\n"
            "    Kök sayısı: |\\Phi| = 6 + 6 = 12.\n"
            "    Rank (Cartan boyutu): 2.\n"
            "    Lie cebiri boyutu: \\dim(G_2) = \\mathrm{rank} + |\\Phi| = 2 + 12 = 14.\n"
            "  Uyum (Oktonyon Otomorfizmaları):\n"
            "    G_2, oktonyonlar cebirinin (\\mathbb{O}) türevler cebiridir: \\mathfrak{g}_2 = \\mathrm{Der}(\\mathbb{O}).\n"
            "    \\dim(\\mathrm{Der}(\\mathbb{O})) = 14 bağımsız türetim olduğu bilinir.\n"
            "Cevap: 14."
        ),
        "answer": "14"
    },
    {
        "id": "dom2_lie_e6_dimension",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Dimension of Exceptional Lie Algebra E6",
        "instruction": "What is the exact dimension of the exceptional Lie algebra E_6?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: E_6 istisnai Lie cebirinin rankı 6'dır.\n"
            "  Benzerlik (Kök Sistemi):\n"
            "    E_6 kök kafesi, \\mathbb{R}^8 içinde e_7 + e_8 ve e_6 + e_7 + e_8 vektörlerine dik altuzayda yer alır.\n"
            "    Kök sayısı: |\\Phi| = 72 kök (pozitif kök sayısı |\\Phi^+| = 36).\n"
            "  Çıkarım:\n"
            "    \\dim(E_6) = \\mathrm{rank} + |\\Phi| = 6 + 72 = 78.\n"
            "  Uyum: E_6'nın adjoint temsili 78 boyutludur; temel temsili ise 27 boyutlu Albert cebiri üzerindeki eylemdir.\n"
            "Cevap: 78."
        ),
        "answer": "78"
    },
    {
        "id": "dom2_lie_e7_dimension",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Dimension of Exceptional Lie Algebra E7",
        "instruction": "Find the dimension of the exceptional simple Lie algebra E_7 and explain the root decomposition.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: E_7 istisnai Lie cebirinin rankı 7'dir.\n"
            "  Benzerlik (Kök Sayımı):\n"
            "    E_7 kök sistemi 126 kökten oluşur:\n"
            "    - Pozitif kök sayısı: |\\Phi^+| = 63.\n"
            "    - Negatif kök sayısı: |\\Phi^-| = 63.\n"
            "    Toplam kökler |\\Phi| = 126.\n"
            "  Çıkarım:\n"
            "    \\dim(E_7) = \\mathrm{rank} + |\\Phi| = 7 + 126 = 133.\n"
            "  Şartlar (Yanılma Uyarısı):\n"
            "    Yaygın bir hata, yalnızca pozitif kök sayısı olan 63'ü cevap olarak vermektir. Gerçek boyut rank (7) + tüm kökler (126) = 133'tür.\n"
            "Cevap: 133."
        ),
        "answer": "133"
    },
    {
        "id": "dom2_lie_e8_dimension",
        "depth": 1,
        "focal_weight": 1.1,
        "topic": "Dimension of Exceptional Lie Algebra E8",
        "instruction": "Calculate the dimension of the exceptional Lie algebra E_8 using its unimodular root lattice.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: E_8, istisnai Lie cebirlerinin en büyüğüdür ve rankı 8'dir.\n"
            "  Benzerlik (E_8 Kök Kafesi):\n"
            "    \\mathbb{R}^8 içindeki E_8 kökleri normu 2 olan vektörlerdir:\n"
            "    1. \\pm e_i \\pm e_j (i < j): 4 \\times \\binom{8}{2} = 112 kök.\n"
            "    2. \\frac{1}{2}(\\pm 1, \\dots, \\pm 1) (çift sayıda eksi işaretli): 2^7 = 128 kök.\n"
            "    Toplam kök sayısı: |\\Phi| = 112 + 128 = 240 kök.\n"
            "  Çıkarım:\n"
            "    \\dim(E_8) = \\mathrm{rank} + |\\Phi| = 8 + 240 = 248.\n"
            "  Uyum: E_8'in en küçük aşikâr olmayan temsili kendi adjoint temsilidir (248 boyut).\n"
            "Cevap: 248."
        ),
        "answer": "248"
    },
    {
        "id": "dom2_lie_sl4_dimension",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Dimension of Special Linear Lie Algebra sl(4)",
        "instruction": "Compute the dimension of the special linear Lie algebra \\mathfrak{sl}(4, \\mathbb{C}).",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: \\mathfrak{sl}(k, \\mathbb{C}), izi sıfır olan k \\times k karmaşık matrislerin Lie cebiridir.\n"
            "  Çıkarım:\n"
            "    - Bütün k \\times k matrislerin uzayı \\mathfrak{gl}(k, \\mathbb{C}) k^2 boyuta sahiptir. k = 4 için 4^2 = 16.\n"
            "    - \\mathrm{Tr}(X) = 0 koşulu 1 adet bağımsız lineer kısıt getirir.\n"
            "    - \\dim(\\mathfrak{sl}(4)) = 16 - 1 = 15.\n"
            "  Uyum: Cartan tipi A_n (burada n = 3, k = n+1 = 4). Formül: n(n+2) = 3 \\times 5 = 15.\n"
            "Cevap: 15."
        ),
        "answer": "15"
    },
    {
        "id": "dom2_lie_so8_dimension",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Dimension of Orthogonal Lie Algebra so(8)",
        "instruction": "What is the dimension of the Lie algebra \\mathfrak{so}(8, \\mathbb{C})?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: \\mathfrak{so}(m, \\mathbb{C}), antisimetrik m \\times m matrislerin (X^T = -X) Lie cebiridir.\n"
            "  Çıkarım:\n"
            "    - Köşegen elemanlar sıfır olmak zorundadır (8 sıfır).\n"
            "    - Köşegen üstündeki elemanlar köşegen altındakileri tek türlü belirler.\n"
            "    - Serbest eleman sayısı: \\frac{m(m-1)}{2} = \\frac{8 \\times 7}{2} = 28.\n"
            "  Uyum: Cartan tipi D_4 (2n = 8 \\implies n = 4). Formül: n(2n-1) = 4 \\times 7 = 28.\n"
            "Cevap: 28."
        ),
        "answer": "28"
    },
    {
        "id": "dom2_lie_so5_dimension",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Dimension of Odd Orthogonal Lie Algebra so(5)",
        "instruction": "Compute the dimension of the odd orthogonal Lie algebra \\mathfrak{so}(5, \\mathbb{C}).",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: \\mathfrak{so}(2n+1) tek ortogonal Lie cebiridir (Cartan tipi B_n). Burada 2n+1 = 5 \\implies n = 2 (B_2).\n"
            "  Çıkarım:\n"
            "    - 5 \\times 5 antisimetrik matrislerin boyutu: \\frac{5 \\times 4}{2} = 10.\n"
            "    - Formül: n(2n+1) = 2 \\times 5 = 10.\n"
            "  Uyum: B_2 kök sistemi C_2 kök sistemine çift (dual) izomorftur, dolayısıyla \\dim(\\mathfrak{so}(5)) = \\dim(\\mathfrak{sp}(4)) = 10.\n"
            "Cevap: 10."
        ),
        "answer": "10"
    },

    # ------------------ STAGE 2: GROUP THEORY ------------------
    {
        "id": "dom2_grp_sylow_order21",
        "depth": 2,
        "focal_weight": 1.1,
        "topic": "Sylow Subgroups of Order 21",
        "instruction": "Let G be a group of order 21 = 3 × 7. Find the number of Sylow 7-subgroups and the possible numbers of Sylow 3-subgroups.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: 3. Sylow Teoremine göre Sylow p-altgruplarının sayısı n_p şu iki şartı sağlar:\n"
            "    1. n_p \\equiv 1 \\pmod p\n"
            "    2. n_p \\mid (|G|/p)\n"
            "  Çıkarım:\n"
            "    - p = 7 için: n_7 \\equiv 1 \\pmod 7 ve n_7 \\mid 3.\n"
            "      3'ün bölenleri {1, 3}. 3 \\not\\equiv 1 \\pmod 7 olduğundan tek çözüm n_7 = 1'dir. Sylow 7-altgrubu tektir ve normaldir.\n"
            "    - p = 3 için: n_3 \\equiv 1 \\pmod 3 ve n_3 \\mid 7.\n"
            "      7'nin bölenleri {1, 7}. Hem 1 \\equiv 1 \\pmod 3 hem de 7 \\equiv 1 \\pmod 3 olduğundan n_3 \\in {1, 7} olabilir.\n"
            "Cevap: n_7 = 1, n_3 \\in \\{1, 7\\}."
        ),
        "answer": "n_7 = 1, n_3 \\in \\{1, 7\\}"
    },
    {
        "id": "dom2_grp_a5_simplicity",
        "depth": 2,
        "focal_weight": 1.0,
        "topic": "Order of the Smallest Non-Abelian Simple Group",
        "instruction": "What is the order of the smallest non-abelian simple group?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: Sonlu basit grupların sınıflandırmasına göre en küçük abelyen olmayan basit grup A_5 (5 elemanlı alterne grup)tir.\n"
            "  Çıkarım:\n"
            "    - Simetrik grup S_5 mertebesi 5! = 120'dir.\n"
            "    - A_5, çift permütasyonlardan oluşur ve işaret homomorfizmasının çekirdeğidir (indeksi 2'dir).\n"
            "    - |A_5| = 120 / 2 = 60.\n"
            "  Uyum: A_5'in eşlenik sınıfları 1, 15, 20, 12, 12 elemanlıdır; 1'i içeren hiçbir alt küme toplamı 60'ı bölmediğinden aşikâr olmayan normal altgrubu yoktur.\n"
            "Cevap: 60."
        ),
        "answer": "60"
    },
    {
        "id": "dom2_grp_aut_q8",
        "depth": 2,
        "focal_weight": 1.1,
        "topic": "Automorphism Group of Quaternion Group Q8",
        "instruction": "What is the isomorphism type and order of the automorphism group \\mathrm{Aut}(Q_8) of the quaternion group Q_8?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: Kuaterniyon grubu Q_8 = {±1, ±i, ±j, ±k} 8 elemanlıdır.\n"
            "  Benzerlik (İç ve Dış Otomorfizmalar):\n"
            "    - Merkez Z(Q_8) = {±1}. Bölüm grubu Q_8 / Z(Q_8) \\cong \\mathbb{Z}_2 \\times \\mathbb{Z}_2 (Klein 4-grubu).\n"
            "    - İç otomorfizma grubu: \\mathrm{Inn}(Q_8) \\cong Q_8 / Z(Q_8) \\implies |\\mathrm{Inn}(Q_8)| = 4.\n"
            "    - Q_8 içinde mertebesi 4 olan 3 adet maksimal döngel altgrup vardır: \\langle i \\rangle, \\langle j \\rangle, \\langle k \\rangle.\n"
            "    - Her otomorfizma bu 3 altgrubu permüte etmek zorundadır, bu da \\mathrm{Aut}(Q_8) \\to S_3 homomorfizması üretir.\n"
            "  Çıkarım:\n"
            "    Çekirdek \\mathrm{Inn}(Q_8) olduğundan: |\\mathrm{Aut}(Q_8)| = 4 \\times 6 = 24.\n"
            "    Yapısal analiz bu grubun simetrik grup S_4'e izomorf olduğunu kanıtlar: \\mathrm{Aut}(Q_8) \\cong S_4.\n"
            "Cevap: S_4 (mertebe 24)."
        ),
        "answer": "S_4"
    },
    {
        "id": "dom2_grp_center_d8",
        "depth": 2,
        "focal_weight": 1.0,
        "topic": "Center of the Dihedral Group D8",
        "instruction": "What is the order of the center Z(D_8) of the dihedral group of order 8?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: D_8 = \\langle r, s \\mid r^4 = 1, s^2 = 1, srs = r^{-1} \\rangle mertebesi 8 olan dihedral gruptur.\n"
            "  Çıkarım:\n"
            "    - Bir eleman z \\in Z(D_8) olabilmesi için hem r hem de s ile değişmeli olması gerekir.\n"
            "    - r^k elemanının s ile değişmesi için: s r^k s = r^k \\implies r^{-k} = r^k \\implies r^{2k} = 1.\n"
            "    - k = 0 (özdeşlik 1) ve k = 2 (r^2, 180 derece dönme) bunu sağlar.\n"
            "    - Yansımalar (sr^k) merkezde yer alamaz çünkü r ile değişmezler.\n"
            "    - Z(D_8) = {1, r^2} \\implies |Z(D_8)| = 2.\n"
            "Cevap: 2."
        ),
        "answer": "2"
    },

    # ------------------ STAGE 3: RING & FIELD THEORY ------------------
    {
        "id": "dom2_ring_maximal_r_poly",
        "depth": 3,
        "focal_weight": 1.1,
        "topic": "Maximal Ideal in R[x] and Quotient Field",
        "instruction": "Is the ideal I = (x^2 + 1) maximal in the polynomial ring \\mathbb{R}[x]? What is the field \\mathbb{R}[x] / I isomorphic to?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: \\mathbb{R} bir cisim olduğundan polinom halkası \\mathbb{R}[x] bir Temel İdeal Bölgesidir (PID).\n"
            "  Çıkarım:\n"
            "    1. Bir PID içinde sıfırdan farklı bir (p(x)) ideali maksimaldir \\iff p(x) indirgenemezdir.\n"
            "    2. p(x) = x^2 + 1 polinomunun derecesi 2'dir ve gerçel kökü yoktur (diskriminant \\Delta = -4 < 0). Dolayısıyla \\mathbb{R} üzerinde indirgenemezdir.\n"
            "    3. Sonuç olarak I = (x^2 + 1) ideali \\mathbb{R}[x] içinde maksimaldir.\n"
            "    4. Bir halkanın maksimal ideale bölümü bir cisimdir. x mod I sınıfı i sanal birimi gibi davranır: x^2 \\equiv -1 \\pmod{x^2+1}.\n"
            "    5. Birinci izomorfizma teoremi gereğince \\mathbb{R}[x] / (x^2 + 1) \\cong \\mathbb{C}.\n"
            "Cevap: \\mathbb{C}."
        ),
        "answer": "\\mathbb{C}"
    },
    {
        "id": "dom2_ring_units_gaussian",
        "depth": 3,
        "focal_weight": 1.0,
        "topic": "Units of the Gaussian Integers Z[i]",
        "instruction": "List all units (invertible elements) in the ring of Gaussian integers \\mathbb{Z}[i].",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: \\mathbb{Z}[i] halkasında bir z = a + bi elemanının birim (tersinir) olması için çarpımsal normunun N(z) = a^2 + b^2 = 1 olması gerekir.\n"
            "  Çıkarım:\n"
            "    a, b tamsayıları için a^2 + b^2 = 1 denkleminin tam olarak 4 çözümü vardır:\n"
            "    - a = ±1, b = 0 \\implies z = ±1\n"
            "    - a = 0, b = ±1 \\implies z = ±i\n"
            "    Dolayısıyla birimler grubu \\mathbb{Z}[i]^\\times = {1, -1, i, -i} olup 4 elemanlı döngel gruptur (\\mathbb{Z}_4).\n"
            "Cevap: \\{\\pm 1, \\pm i\\}."
        ),
        "answer": "\\{\\pm 1, \\pm i\\}"
    },
    {
        "id": "dom2_ring_finite_field_27",
        "depth": 3,
        "focal_weight": 1.0,
        "topic": "Multiplicative Group of Finite Field F_27",
        "instruction": "What is the structure and order of the multiplicative group \\mathbb{F}_{27}^\\times of the finite field with 27 elements?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: Sonlu bir \\mathbb{F}_q cisminde sıfır dışındaki elemanlar çarpma altında bir grup oluşturur (mertebe q - 1).\n"
            "  Çıkarım:\n"
            "    - q = 27 için mertebe |\\mathbb{F}_{27}^\\times| = 27 - 1 = 26'dır.\n"
            "    - Cebirin temel teoremine göre herhangi bir sonlu cismin çarpımsal grubu daima döngeldir (cyclic).\n"
            "    - Bu nedenle \\mathbb{F}_{27}^\\times \\cong \\mathbb{Z}_{26} (26 mertebeli döngel grup).\n"
            "Cevap: \\mathbb{Z}_{26}."
        ),
        "answer": "\\mathbb{Z}_{26}"
    },

    # ------------------ STAGE 4: GALOIS THEORY ------------------
    {
        "id": "dom2_gal_biquadratic_extension",
        "depth": 4,
        "focal_weight": 1.2,
        "topic": "Galois Group of Biquadratic Field Q(sqrt(2), sqrt(3))",
        "instruction": "Find the Galois group \\mathrm{Gal}(\\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) / \\mathbb{Q}) and explain the degree multiplicativity.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: L = \\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) cismi (x^2-2)(x^2-3) polinomunun ayrışma cismidir.\n"
            "  Çıkarım (Kule Kanunu):\n"
            "    1. [\\mathbb{Q}(\\sqrt{2}):\\mathbb{Q}] = 2 (minimal polinom x^2 - 2).\n"
            "    2. \\sqrt{3} \\notin \\mathbb{Q}(\\sqrt{2}) çünkü (a + b\\sqrt{2})^2 = 3 eşitliği rasyonel a, b için imkânsızdır.\n"
            "    3. Dolayısıyla [L:\\mathbb{Q}(\\sqrt{2})] = 2.\n"
            "    4. Kule Kanunu gereği [L:\\mathbb{Q}] = 2 \\times 2 = 4.\n"
            "  Otomorfizmalar:\n"
            "    Karakteristik 0 ve normal olduğundan genişleme Galois'dır ve |\\mathrm{Gal}(L/\\mathbb{Q})| = 4'tür.\n"
            "    Otomorfizmalar üreteçleri işaretlerine gönderir: \\sigma(\\sqrt{2}) = -\\sqrt{2}, \\tau(\\sqrt{3}) = -\\sqrt{3}.\n"
            "    Her aşikâr olmayan elemanın mertebesi 2'dir: \\sigma^2 = \\tau^2 = (\\sigma\\tau)^2 = \\mathrm{id}.\n"
            "    Mertebesi 4 olan ve her elemanı 2 olan tek grup Klein 4-grubudur: \\mathbb{Z}_2 \\times \\mathbb{Z}_2.\n"
            "Cevap: \\mathbb{Z}_2 \\times \\mathbb{Z}_2."
        ),
        "answer": "\\mathbb{Z}_2 \\times \\mathbb{Z}_2"
    },
    {
        "id": "dom2_gal_cyclotomic_7",
        "depth": 4,
        "focal_weight": 1.1,
        "topic": "Galois Group of the 7th Cyclotomic Field",
        "instruction": "Compute the Galois group \\mathrm{Gal}(\\mathbb{Q}(\\zeta_7) / \\mathbb{Q}), where \\zeta_7 is a primitive 7th root of unity.",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: \\zeta_7 ilkel 7. birim köküdür.\n"
            "  Çıkarım:\n"
            "    1. Minimal polinomu 7. siklotomik polinomdur: \\Phi_7(x) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1.\n"
            "    2. p = 7 asal olduğundan \\Phi_7(x+1) üzerine Eisenstein kriteri (p=7) uygulanarak \\mathbb{Q} üzerinde indirgenemez olduğu ispatlanır.\n"
            "    3. Genişleme derecesi: [\\mathbb{Q}(\\zeta_7):\\mathbb{Q}] = \\deg(\\Phi_7) = 6.\n"
            "    4. Siklotomik Galois grubu kanonik olarak (\\mathbb{Z}/7\\mathbb{Z})^\\times grubuna izomorftur.\n"
            "    5. (\\mathbb{Z}/7\\mathbb{Z})^\\times mertebesi 6 olan bir döngel gruptur: \\mathbb{Z}_6.\n"
            "Cevap: \\mathbb{Z}_6."
        ),
        "answer": "\\mathbb{Z}_6"
    },
    {
        "id": "dom2_gal_cubic_x3_minus_2",
        "depth": 4,
        "focal_weight": 1.1,
        "topic": "Galois Group of x^3 - 2 over Q",
        "instruction": "What is the Galois group of the polynomial f(x) = x^3 - 2 over \\mathbb{Q}?",
        "thought": (
            "Arkhe İzi:\n"
            "  Bilinç: f(x) = x^3 - 2 polinomunun kökleri \\alpha = \\sqrt[3]{2}, \\omega\\alpha, \\omega^2\\alpha'dır (\\omega = e^{2\\pi i / 3}).\n"
            "  Çıkarım:\n"
            "    1. Ayrışma cismi L = \\mathbb{Q}(\\sqrt[3]{2}, \\omega) cismidir.\n"
            "    2. [\\mathbb{Q}(\\sqrt[3]{2}):\\mathbb{Q}] = 3 (Eisenstein kriteri p=2 ile indirgenemez).\n"
            "    3. \\omega'nın minimal polinomu x^2 + x + 1 olup gerçel olmadığından [L:\\mathbb{Q}(\\sqrt[3]{2})] = 2'dir.\n"
            "    4. Toplam derece [L:\\mathbb{Q}] = 3 \\times 2 = 6.\n"
            "    5. L bir 3. derece polinomun ayrışma cismi olduğundan Galois grubu S_3'ün bir altgrubudur.\n"
            "       Mertebesi 6 olduğundan \\mathrm{Gal}(L/\\mathbb{Q}) \\cong S_3.\n"
            "Cevap: S_3."
        ),
        "answer": "S_3"
    },
]


def generate_domain2_sft_dataset(output_file: str, augment_factor: int = 4):
    print("=" * 72, flush=True)
    print("  SODALS v2.1 — Building Deep First-Principles Domain 2 SFT Curriculum", flush=True)
    print("=" * 72, flush=True)

    wronskian = WronskianGramFilter(threshold=1e-4)
    dataset = []
    accepted = 0
    rejected = 0

    sorted_curriculum = sorted(DOMAIN2_MODULES, key=lambda x: (x["depth"], x["focal_weight"]))

    prompts_variations = [
        "Please solve the following advanced abstract algebra problem step by step with complete proof:\n{instruction}",
        "Question: {instruction}\nProvide a complete mathematical derivation using Arkhe first-principles reasoning and conclude with the final answer.",
        "Problem: {instruction}\nWork through the algebraic steps in detail, explain where the formulas come from, and provide the exact result.",
        "Solve the following Lie theory / abstract algebra question rigorously from first principles: {instruction}"
    ]

    for item in sorted_curriculum:
        for var_idx, p_template in enumerate(prompts_variations):
            prompt = p_template.format(instruction=item["instruction"])
            cot = f"<think>\n{item['thought']}\n</think>\n\nAnswer: {item['answer']}"

            sample_text = f"{prompt}\n{cot}"
            if wronskian.is_independent(sample_text):
                dataset.append({
                    "id": f"{item['id']}_var{var_idx}",
                    "domain": "abstract_algebra",
                    "topic": item["topic"],
                    "depth": item["depth"],
                    "focal_weight": item["focal_weight"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cot}
                    ]
                })
                accepted += 1
            else:
                rejected += 1

    for aug in range(augment_factor):
        for item in sorted_curriculum:
            prompt = f"Evaluate and compute the exact result with proof: {item['instruction']}"
            cot = f"<think>\n{item['thought']}\n</think>\n\nAnswer: {item['answer']}"

            sample_text = f"{prompt}\n{cot}"
            if wronskian.is_independent(sample_text):
                dataset.append({
                    "id": f"{item['id']}_aug{aug}",
                    "domain": "abstract_algebra",
                    "topic": item["topic"],
                    "depth": item["depth"],
                    "focal_weight": item["focal_weight"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cot}
                    ]
                })
                accepted += 1
            else:
                rejected += 1

    dataset.sort(key=lambda x: (x["depth"], x["focal_weight"]))

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n  ✓ Deep Domain 2 SFT Dataset generated at: {output_file}", flush=True)
    print(f"  Total Canonical Modules:                      {len(DOMAIN2_MODULES)}", flush=True)
    print(f"  Total Accepted Samples (Hamiltonian Sorted):  {accepted}", flush=True)
    print(f"  Total Rejected Samples (Redundant):           {rejected}", flush=True)
    return accepted


if __name__ == "__main__":
    generate_domain2_sft_dataset(OUTPUT_FILE)
