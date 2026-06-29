# SODALS-IoT: Endüstriyel Kestirimci Bakım ve Gürültü Azaltma Sistemi
## Teknik Spesifikasyon, Matematiksel Temeller ve Siber-Fiziksel İçe Bakış (Introspection) Felsefesi

---

## 1. Giriş ve Felsefi Arka Plan: "Fısıldayan Makineler"

Modern endüstriyel tesisler, ham maddeyi işleyen mekanik yığınlar olmanın ötesinde; sürekli entropiye, aşınmaya ve bilgi gürültüsüne maruz kalan **siber-fiziksel organizmalardır**. İnsan beynindeki düşüncelerin söze dökülmeden önceki kas titreşimlerini yakalayan BCI (Beyin-Bilgisayar Arayüzü) subvocal sEMG mantığı, endüstriyel makinelere uygulandığında devrimsel bir felsefi düzlem ortaya çıkarır: **Makinelerin Fısıltısı**.

Nasıl ki bir insan konuşmadan önce boğaz kaslarında mikroskobik elektrik sinyalleri (sEMG) oluşuyorsa, bir rulman, dişli veya motor da aniden arızalanmaz. Fiziksel kırılmadan saatler, hatta günler önce, çevreye yüksek frekanslı akustik emisyonlar, mikro-termal dalgalanmalar ve manyetik alan sapmaları yayarak **fısıldar**.

SODALS-IoT, bu fısıltıları toplayıp siber-fiziksel bir "içe bakış" (introspection) süzgecinden geçirerek, üretimi durdurmadan arızaları tahmin eden entegre bir yapay zeka mimarisidir.

```
+------------------------------------------------------------------------+
|                      SİBER-FİZİKSEL İÇE BAKIŞ                          |
|                                                                        |
|  [Fiziksel Katman] ---> [Sinyal Fısıltısı] ---> [Topolojik Çıkarım]   |
|  (Rulman/Motor)         (Aşınma Mikro-v)       (Wronskian/Jakobiyen)   |
|                                                      |                 |
|                                                      v                 |
|  [Karar Katmanı]  <--- [Semantik Doğrulama] <--- [LSQ Causal Feedback] |
|  (Fiziksel Kısıt)      (EBNF Kısıtlı LLM)       (Aᵀ ile Hata Paylama)  |
+------------------------------------------------------------------------+
```

---

## 2. Mimari Katmanları (BCI'dan IoT'ye Transkripsiyon)

### Katman 0: Sensör Kalibrasyonu ve Bazal Çizgi (Baseline)
```
BCI Karşılığı : Kullanıcı Kalibrasyonu (DTW ile hece → token ID eşleşmesi)
IoT Karşılığı : Makine-İçi Kalibrasyon ve Bazal Sinyal İmzası (Signature)
```

#### Felsefi Boyut
BCI sistemlerinde her insanın ses telleri ve gırtlak yapısı benzersiz olduğu için kalibrasyon **kişi içi** (subject-specific) yapılmalıdır. Endüstride de durum aynıdır: Seri üretimden çıkmış olsa dahi, montaj toleransları, zemin titreşimi ve yük koşulları nedeniyle **her makinenin akustik ve termal parmak izi benzersizdir**. Bu nedenle kalibrasyon "makineye özel" olmalı, her makine kendi normal çalışma bazal çizgisini kendisi öğrenmelidir.

#### Matematiksel Gerçekleştirim
Her sensörden gelen zaman serileri DTW (Dynamic Time Warping) hizalamasından geçirilerek makinenin normal çalışma şablonu çıkarılır ve her sensör kanalı bir "token ID" ile sembolik uzaya taşınır.

```rust
// Zero-Allocation Sensör Kalibrasyon Yapısı
pub struct SensorCalibration<const N: usize> {
    pub sensor_id: u32,
    pub baseline_mean: f64,
    pub baseline_std: f64,
    pub dtw_template: [f64; N],
    pub token_id: u32,
}

impl<const N: usize> SensorCalibration<N> {
    pub fn is_anomalous(&self, reading: f64, deviation_multiplier: f64) -> bool {
        let z_score = (reading - self.baseline_mean).abs() / (self.baseline_std + 1e-9);
        z_score > deviation_multiplier // 3-Sigma kuralı (İstatiksel proses kontrolü)
    }
}
```

---

### Katman 1: EBNF Parse Ağacı → Fiziksel Yasallık Grameri
```
BCI Karşılığı : EEG semantik rolleri (<subject>, <object>, <verb>)
IoT Karşılığı : Fiziksel durum semantiği (<cause>, <effect>, <state>, <action>)
```

#### Felsefi Boyut
İnsan beyninden gelen EEG sinyalleri semantik rollere bölünüp dil bilgisi kurallarına (gramer) göre sınanır. Endüstriyel dünyada ise dil bilgisinin yerini **fiziksel yasallık** (physical legality) alır. Termodinamik ve mekanik yasaları, makinenin durum değişimlerinin "gramerini" belirler. Bir sensör zinciri, fizik yasalarını çiğniyorsa (örneğin motor sıcaklığı artmadan devrin aniden sıfıra inmesi ama akımın tavan yapması), bu girdi "sentaktik olarak geçersizdir" ve LLM kısıtlama katmanında elenmelidir.

#### EBNF (Extended Backus-Naur Form) Tanımı
Makinelerin çalışması, fizik yasalarına dayanan bir durum grameri ile tanımlanır:

```ebnf
(* SODALS-IoT Fiziksel Durum Grameri *)
<system_state>      ::= <operational_phase> + <thermodynamic_balance> + <vibration_syntax>
<operational_phase> ::= <idle> | <ramp_up> | <steady_state> | <cooldown>
<thermodynamic_balance> ::= <heat_source> + <cooling_effect> + <dissipation_rate>
<heat_source>       ::= winding_temp_c | bearing_temp_c
<cooling_effect>    ::= oil_flow_rate | ambient_temp_c
<vibration_syntax>  ::= <vibration_amplitude> + <wronskian_state>
<wronskian_state>   ::= "Lineer Bağımlı (Normal)" | "Lineer Bağımsız (Anomali)"
```

#### Hiyerarşik Parse Ağacı (İnterpolasyon İterasyonu Sonucu)
Sinyal kaybı (packet loss) veya sensör arızası durumunda, sistem interpolasyon iterasyonları ile aşağıdaki parse ağacını inşa eder. Bilinmeyen veya kayıp sensör değeri ($f(x)$), fiziksel kısıtlara göre yaprak düğümlere dağıtılır:

```
                  <system_state> [Valid]
                         |
        +----------------+----------------+
        |                                 |
<thermodynamic_balance>          <vibration_syntax>
        |                                 |
  +-----+-----+                     +-----+-----+
  |           |                     |           |
<source>   <cooling>             <amplitude> <wronskian>
  |           |                     |           |
 [95°C]     [f(x)]               [12 mm/s]    [W ≠ 0]
              |                                 |
      (Eksik Veri Çözümü)                (Yeni Frekans İmzası)
      → Hesaplanır: 2.1 L/dk
```

---

### Katman 2: Wronskian ve Jakobiyen ile Ağırlıklı Otomata

```
BCI Karşılığı : sEMG geçiş olasılıkları + Newton fark analojisi
IoT Karşılığı : Wronskian Diferansiyel Analizi + Jakobiyen Durum Uzayı
```

#### 2.1 Wronskian Determinantı ile Lineer Bağımsızlık ve Anomali Kararı
Makineden gelen iki titreşim sinyali ($f_1(t)$ ve $f_2(t)$) arasındaki ilişkiyi analiz etmek için diferansiyel denklemlerin en temel araçlarından olan Wronskian determinantı kullanılır. 

$$W(f_1, f_2)(t) = \begin{vmatrix} f_1(t) & f_2(t) \\ f_1'(t) & f_2'(t) \end{vmatrix} = f_1(t)f_2'(t) - f_2(t)f_1'(t)$$

* **$W(f_1, f_2) \approx 0$ (Lineer Bağımlılık)**: $f_2$ sinyali, $f_1$ sinyalinin sadece bir türevi, faz kayması veya harmonik yansımasıdır. Yeni bir fiziksel olay (arıza) yoktur.
* **$W(f_1, f_2) \neq 0$ (Lineer Bağımsızlık)**: $f_2$ sinyali, sistemde daha önce var olmayan, bağımsız bir fiziksel kaynaktan (örn. milden çatlama, kafeste kırılma) beslenmektedir. Bu net bir anomali sinyalidir.

```rust
// Zero-Allocation Wronskian Analizi (Hot Path)
pub fn compute_wronskian_hot(f1: &[f64; 2], f2: &[f64; 2], dt: f64) -> f64 {
    let f1_t = f1[1];
    let f2_t = f2[1];
    
    // Merkezi fark ile türev yaklaştırması (O(1) stack operations)
    let f1_prime = (f1[1] - f1[0]) / dt;
    let f2_prime = (f2[1] - f2[0]) / dt;
    
    f1_t * f2_prime - f2_t * f1_prime
}
```

#### 2.2 Jakobiyen Matrisi ile RUL (Kalan Faydalı Ömür) Hesaplaması
Durum değişkenleri $x = [x_1, x_2, x_3, x_4]^T = [\text{Sıcaklık}, \text{Basınç}, \text{Devir}, \text{Titreşim}]^T$ olsun. Sistem dinamiği $dx/dt = f(x, u)$ diferansiyel denklemi ile modellenir. Sistem Jakobiyen matrisi:

$$J = \frac{\partial f}{\partial x} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_4} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_4}{\partial x_1} & \dots & \frac{\partial f_4}{\partial x_4}
\end{bmatrix}$$

Jakobiyen matrisinin özdeğerleri ($\lambda$) hesaplanır:

$$J \cdot v = \lambda \cdot v$$

En büyük gerçel kısma sahip özdeğer ($\lambda_{max}$), sistemin kararsızlığa (arızaya) doğru en hızlı sürüklenen modunu temsil eder. Buradan Kalan Faydalı Ömür (RUL):

$$\text{RUL} = \frac{\Theta_{\text{critical}}}{\lambda_{\text{max}} + \epsilon}$$

#### 2.3 Newton İleri Fark Analojisi ve Ağırlıklı Otomata Geçişleri
BCI'daki heceler arası olasılıksal geçişler, IoT'de fiziksel degradasyon adımlarına karşılık gelir. Newton İleri Fark tablosundaki her üst derece fark ($\Delta^k f$), bir önceki degrade gradyanın ivmesini ve artık hatasını düzeltir:

```
[Mevcut Durum q₀] (Normal)
       |
       |  σ₁: T_001 (Sıcaklık Aşımı) -> δ(q₀, σ₁)
       v
[Degrade Durum q₁] (Yatak Hasarı Başlangıcı) ---> Δ¹f (Hız Değişimi)
       |
       |  σ₂: V_003 (Akustik Emisyon Girişi) -> W ≠ 0
       v
[Kritik Durum q₂] (Termal Kilitlenme Öncesi)  ---> Δ²f (İvmelenme Hızı)
```

**Ağırlıklı Otomata Karar Kuralı:**

$$\delta(q, \sigma) = \operatorname{argmax}_i \left[ w_{\text{sensor}}(\sigma, \text{sensor}_i) \cdot w_{\text{wronskian}}(\text{fault}_i) \cdot w_{\text{jacobian}}(\text{degrade}_i) \right]$$

---

### Katman 3: LSQ (En Küçük Kareler) ve $A^T$ Transpoz ile Causal Feedback

```
BCI Karşılığı : AᵀA w = Aᵀb normal denklem bazlı ağırlık güncellemesi
IoT Karşılığı : Sensör katkı paylama ve hata dağıtım (Blame Attribution)
```

#### Felsefi Boyut
Bir siber-fiziksel sistemin içe bakış (introspection) yapabilmesi için, küresel tahmin hatasını (residual) oluşturan alt parçaları doğru cezalandırması veya ödüllendirmesi gerekir. LSQ (Least Squares) normal denklemindeki transpoz matrisi $A^T$, bu **nedensel atıf** (causal attribution) işleminin kalbidir. 

Aktivasyon matrisi $A$, hangi sensörün hangi zaman diliminde sisteme ne kadar sinyal bastığını gösterir. $A^T$, hata vektörü $(b - Aw)$ ile çarpıldığında, hatayı sensörlerin kendi geçmiş aktivasyon güçleriyle çarparak geri yansıtır. Yani **"hata, en çok sesi çıkana (en aktif olana) paylaştırılır"**.

Bu durum, modern yapay zekadaki **Attention (Dikkat) mekanizmasındaki $Q K^T$ matris çarpımı ile derin bir yapısal benzerliğe sahiptir**: Hangi girdinin hangi çıktı ile ilişkili olduğunu topolojik olarak bağlar.

```
                  LSQ Aᵀ Hata Dağıtım Döngüsü
                  
              +----------------------------------+
              |    Sistem Çıktısı Tahmini (Aw)   |
              +----------------------------------+
                               |
                               v (Hata Hesaplama: b - Aw)
              +----------------------------------+
              |      Global Hata Vektörü (e)     |
              +----------------------------------+
                               |
                               v (Aᵀ ile Geri Yansıtma: Aᵀe)
              +----------------------------------+
              |    Sensör Katkı Ağırlığı (Δw)    |
              |   (En gürültülü sensöre ceza)    |
              +----------------------------------+
```

#### Matematiksel Modelleme
Minimize edilmek istenen hata fonksiyonu:

$$\min_{w} \|Aw - b\|^2 \implies A^T A w = A^T b \implies w^* = (A^T A)^{-1} A^T b$$

Gerçek zamanlı edge cihazında bu ters matris işlemini ($A^T A)^{-1}$ sürekli koşturmak işlemciyi yoracağından, ağırlık güncellemeleri gradient descent formuyla iterative olarak koşturulur:

$$w_{\text{new}} = w_{\text{old}} + \eta \cdot A^T (b - A w_{\text{old}})$$

Burada:
* $A$ : $N \times 128$ sensör aktivasyon matrisi.
* $b$ : $N \times 1$ doğrulanmış arıza tespiti (ground truth).
* $w$ : $128 \times 1$ sensör nedensel ağırlık vektörü.
* $A^T (b - Aw)$ : Hata yönünün sensör bazında yansıması.

---

### Katman 4: LLM Kısıtlı Üretim ve Semantik Doğrulama

Otomata seviyesinde elde edilen tüm fiziksel ve semantik kısıtlar, LLM (Örn. Qwen 8B) çıkarım yaparken logits katmanında maskeleme ve fonetik boost olarak uygulanır.

```
[Otomata Çıktısı] ---> [Logits Processor Masking] ---> [LLM Decode]
                             (Fiziksel Kısıtlar)       (Qwen 8B)
                                                           |
                                                           v
[Uç Çıktı: "Rulman 2 Değişmeli"] <--- [God's Gate] <--- [Ouroboros Loop]
                                     (LU / Picard)      (Eleştiri/İyileştirme)
```

1. **Ouroboros Loop**: LLM çıktısı kendi içinde "Simülasyon $\rightarrow$ Eleştiri $\rightarrow$ İyileştirme" döngüsünden geçerek mantıksal hatalarından temizlenir.
2. **God's Gate (Picard/LU Doğrulama)**: Çıkan nihai kararın fiziksel doğruluğu diferansiyel denklemler (Picard yakınsaması) ve lineer cebir solverları (LU ayrışımı) ile son kez donanım seviyesinde test edilerek PLC'ye "Read-Only" tavsiye raporu olarak iletilir.

Sistemin tam hedef fonksiyonu (Sistemin Tam Çıktı Kararı):

$$\hat{y} = \text{LLM}(\operatorname{constrain}(\text{logits}, \delta^*(q, \sigma)))$$

---

## 3. Golden Teacher RL ile Boyut İndirgeme (128 → 4)

Endüstriyel tesislerdeki "Veri Çöplüğü" (128 sensörün binlerce Hz hızında veri üretmesi) problemini çözmek için BCI'daki EEG kanal indirgeme felsefesi kullanılır.

```
       [128 Sensör Girdisi] (Sıcaklık, Titreşim, Akım, Akış...)
                |
                v
  +───────────────────────────+
  |    Öğretmen Ajan (Twin)   | (Geniş CPU/GPU, Ağır Simülasyon)
  +───────────────────────────+
                |
                | (Knowledge Distillation - Bilgi Damıtma)
                v
  +───────────────────────────+
  |    Öğrenci Ajan (Edge)    | (Kısıtlı Stack Bellek, Zero-Alloc)
  +───────────────────────────+
                |
                v
  [4 Temel Sağlık İndikatörü (Health Indicators)]
  - HI_1: Mekanik Sağlık  (Titreşim/Hız)
  - HI_2: Termal Sağlık   (Isı/Viskozite)
  - HI_3: Elektrik Sağlığı (Güç/Harmonik)
  - HI_4: Hidrolik Sağlık (Akış/Basınç)
```

### Bilgi Damıtma (Knowledge Distillation) ve RL Ödül Fonksiyonu
Öğrenci (Edge) Ajan, 128 sensörü doğrudan okumak yerine, Wronskian, Jakobiyen ve LSQ ağırlık matrisinden gelen topolojik bilgiyi kullanarak en değerli 4 sensörü dinamik olarak seçer. RL ödül fonksiyonu hem doğruluğu maksimize eder hem de işlem süresini cezalandırır:

$$R = \alpha \cdot (\text{Doğruluk}) - \beta \cdot (\text{İşlem Süresi}) - \gamma \cdot (\text{Bellek Tüketimi})$$

---

## 4. İnterpolasyon ve Eksik Sinyal Çözümleme

Endüstriyel tesisler elektromanyetik gürültü (EMI) altındadır. Kablosuz sensörlerin paket kayıpları sıradan ortalamalarla tamir edilemez. SODALS-IoT, eksik verileri fiziksel korunum yasalarına sadık kalarak doldurur.

```
                  İnterpolasyon Karar Ağacı
                  
                   [Sinyal Kaybı Algılandı]
                              |
                     (Wronskian Kontrolü)
                              |
               +--------------+--------------+
               | (W < Threshold)             | (W > Threshold)
               v                             v
     [Newton Fark Polinomu]        [Çok Terimli Regresyon]
     (Fiziksel Yasaya Uygun)       (Yüksek Gradyanlı Düzeltme)
```

Eğer Wronskian determinantı belirlenen eşiğin altındaysa, eksik sinyalin sistemin kararlı fazına ait olduğu doğrulanır ve Newton ileri fark tablosundaki polinomsal interpolasyon stack-only bellek bölgesinde hesaplanarak yerine konur.

---

## 5. Zero-Allocation Rust Çekirdeği Tasarım İlkeleri

Fabrika ortamında çalışan AI motorunun en büyük düşmanı **GC (Garbage Collection) gecikmeleri** veya **Heap fragmentation nedeniyle oluşan beklenmedik duraksamalardır**. SODALS-IoT, çekirdeğin en kritik yollarında (hot paths) heap allocation yapmayı tamamen yasaklar.

```
+───────────────────────────+─────────────────────────────────────────+
| Zero-Alloc Kuralı         | Uygulama Detayı                         |
+───────────────────────────+─────────────────────────────────────────+
| Stack-Only Veri Yapıları  | Vec<T> yerine sabit boyutlu [T; N]      |
| Lock-Free Haberleşme      | Mutex yerine atomik swap ve spinlocklar |
| Ring Buffer Kullanımı     | Dinamik kuyruk yerine halka bellek      |
+───────────────────────────+─────────────────────────────────────────+
```

```rust
// Stack-Allocated Ring Buffer Tasarımı (Zero-Heap Allocation)
pub struct SensorBuffer<const CAP: usize, const SENSORS: usize> {
    pub buffer: [[f64; SENSORS]; CAP],
    pub head: usize,
    pub tail: usize,
    pub count: usize,
}

impl<const CAP: usize, const SENSORS: usize> SensorBuffer<CAP, SENSORS> {
    pub const fn new() -> Self {
        Self {
            buffer: [[0.0; SENSORS]; CAP],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, data: [f64; SENSORS]) {
        self.buffer[self.head] = data;
        self.head = (self.head + 1) % CAP;
        if self.count < CAP {
            self.count += 1;
        } else {
            self.tail = (self.tail + 1) % CAP;
        }
    }
}
```

---

## 6. IEC 62443 ve Sadece Okunur (Read-Only) Güvenlik

Siber güvenlik standartları (IEC 62443 SL-2) uyarınca, AI modelinin çıkardığı kararların PLC (Programlanabilir Mantıksal Denetleyici) kontrol döngüsünü doğrudan manipüle etmesi **kesinlikle yasaktır**. 

```
                                    GÜVENLİK SINIRI
                                    
  [PLC / SCADA] --------------(Sadece Okuma)-------------> [SODALS-IoT Engine]
  (Aktüatör Kontrolü)         (OPC UA / Read-Only Node)    (Wronskian/LLM)
                                                                 |
                                                                 v
  [İşletici Paneli] <--- (Bakım Önerisi / Alarm Raporu) ----------+
```

* **Write/Execute Hak Kısıtı**: OPC UA istemcisinde `write` metotları bulunmaz, sadece sensör verilerine `subscribe` olunabilir.
* **Watchdog Entegrasyonu**: AI motorunun kendi durumunu raporlayan bir heartbeat sinyali donanımsal watchdog devresine iletilir. AI motorunun donması veya çökelmesi durumunda, fiziksel hat otomatik olarak donanım seviyesinde güvenli moduna geçer.

---

## 7. TÜBİTAK BiGG ve Savunma Sanayii / Endüstri 4.0 Uyumluluk Matrisi

Bu teknik altyapı, yerli ve milli teknoloji hamlesi standartlarıyla tam uyumludur:

```
+───────────────────────────+─────────────────────────+──────────────────────────────────────+
| Teknoloji Kriteri         | TÜBİTAK Karşılığı       | Endüstriyel Katma Değeri             |
+───────────────────────────+─────────────────────────+──────────────────────────────────────+
| Wronskian & Jakobiyen     | Matematiksel İnovasyon  | Algoritmik bağımsızlık, düşük hata   |
| Zero-Alloc Rust Core      | Teknik Derinlik         | Donanım maliyet düşüşü               |
| Golden Teacher RL         | Özgün Yapay Zeka        | Veri sıkıştırma ve edge uygulanırlık |
| IEC 62443 Read-Only       | Güvenilirlik & Siber S. | Kritik altyapı koruması              |
+───────────────────────────+─────────────────────────+──────────────────────────────────────+
```

* **Yerli İkame**: Yabancı SCADA ve kestirimci bakım yazılımlarının (Siemens MindSphere vb.) yerine, tamamen yerel sunucularda koşan, bulut bağımlılığı olmayan egemen bir AI katmanı sunar.

---

## 8. Sistem Özeti Formülleri

Sistem kararlarının ve içe bakış süreçlerinin matematiksel bir özeti:

$$\begin{aligned}
\text{Durum Kısıt Maskesi} &: \delta^*(q, \sigma) = \operatorname{argmax}_i \left[ w_{\text{sensor}} \cdot w_{\text{wronskian}} \cdot w_{\text{jacobian}} \right] \\
\text{Nihai Çıktı Kararı} &: \hat{y} = \text{LLM}(\operatorname{constrain}(\text{logits}, \delta^*(q, \sigma))) \\
\text{Causal Blame Güncelleme} &: w_{\text{new}} = w_{\text{old}} + \eta \cdot A^T (b - A w_{\text{old}}) \\
\text{Ömür Öngörüsü (RUL)} &: \text{RUL} = \frac{\Theta_{\text{critical}}}{\lambda_{\text{max}}(J) + \epsilon}
\end{aligned}$$

---

## 9. Başarı Kriterleri ve Limitler

Sistem performansının doğrulanması için aşağıdaki limitler ve hedefler esas alınır:

1. **Boyut İndirgeme Bilgi Koruma**: 128 sensörden 4 indikatöre indirgendiğinde bilgi kaybı $\%5$'in altında kalmalıdır.
2. **Anomali Yakalama Hassasiyeti**: Yanlış pozitif alarm oranı $\%3$'ün altında kalırken, gerçek mekanik arızaları yakalama oranı $\%92$'den yüksek olmalıdır.
3. **Edge Gecikme Sınırı**: Zero-allocation Rust döngüsünün tick süresi $10\text{ms}$ ayarlandığında, her iterasyonun toplam işlem süresi $1\text{ms}$ sınırını geçmemelidir.
4. **Bellek Sınırı**: Çekirdeğin toplam RAM tüketimi $100\text{MB}$ olarak sabitlenmiştir.

---

*Bu doküman, SODALS siber-fiziksel sistemler mimari spesifikasyonunun en güncel matematiksel ve felsefi sürümüdür. TÜBİTAK BiGG, KOSGEB ve Savunma Sanayii projelerinde teknik referans dokümanı olarak kullanılabilir.*

**Versiyon:** 2.0  
**Tarih:** 2026-06-03  
**Yazar:** SODALS-IoT Teknik ve Felsefi Araştırmalar Grubu  
