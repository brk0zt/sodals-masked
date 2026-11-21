// src/crates/sodals_neuro/src/qmem_builder.rs
// ============================================================================
// SODALS QMEM Cartridge Builder
//
// İki kartridj üretir:
//   1. primordial_cognition.qmem   — Töz bilgiler (evrimsel substrat)
//   2. industrial_topology.qmem    — AssetOps / ITBench domain bilgisi
//
// Töz bilgiler hakkında mimari not:
//   Bunlar "butona bas" gibi mekanik kurallar değil; evrimsel olarak
//   kodlanmış, varlığın içinde hazır bulunan "fiziksel ontoloji" katmanıdır.
//   Bir bebek klavye görmemiş olsa da bir butona dokunma ihtiyacı hisseder
//   çünkü "nesne → tepki" nedenselliği doğuştan geliyor.
//   Bunları QMEM'e koyduğumuzda net_g (fizik motoru) her soruda bu zemine
//   yaslanır; net_f (backbone) ise mevcut durumu bu zemine projekte eder.
// ============================================================================

use anyhow::Result;
use crate::compiler::qmem_compiler::compile;
use crate::compiler::qmem_linker::compile_cst_to_network;
use crate::semantic_network::ResonanceNetwork;

// ════════════════════════════════════════════════════════════════════════════
// TÖZ BİLGİLER (PRIMORDIAL COGNITION)
// ════════════════════════════════════════════════════════════════════════════
//
// Temel ontolojik katmanlar:
//
//   I.   NEDENSEL ETKENLİK   — "Ben hareket edersem dünya değişir"
//   II.  NESNE SÜREKLILIĞI   — "Görmediğim şey yok olmaz"
//   III. FİZİKSEL SEZGİ     — Yerçekimi, temas, sınır, içerik
//   IV.  HEDEF-ARALIK        — "Şimdiki durum ≠ istenen durum → harekete geç"
//   V.   GERİBESLEME DÖNGÜSÜ — Ödül → tekrar; ceza → değiştir
//   VI.  ÖRÜNTÜ TANIMASI     — Tekrar → öngör; simetri → tamamla
//   VII. UZAMSAL SEZGİ      — İçinde/dışında; yukarı/aşağı; yakın/uzak
//   VIII.SOSYAL/İLETİŞİM    — Başka etken → etkileşim isteği
//   IX.  ARC-AGI GRİD PRİMLERI — Hücre rengi; dönüşüm kuralı; kopyala-doldur
//
// Faz sabitleri (Euler e^{iθ}):
//   0.0   = eş faz        → güçlü rezonans, yapıcı girişim
//   1.57  = π/2           → dik, kısmi bağlantı
//   3.14  = π (anti-faz)  → karşıt, kısıtlayıcı, engelleyici
//   2.36  = 3π/4          → yumuşak karşıtlık
//   0.78  = π/4           → eğik bağlantı

const PRIMORDIAL_DSL: &str = r#"
// ═══════════════════════════════════════════════════════════════════
// I. NEDENSEL ETKENLİK — Evrenin en temel bilişsel gerçeği:
//    "Ben hareket edersem dünya değişir."
//    Bir bebeğin elini sallayıp mobil salıncağın döndüğünü görmesi.
//    Bu deneyim hiç öğretilmez — sadece yaşanır ve kodlanır.
// ═══════════════════════════════════════════════════════════════════

"kendi_eylemi" CAUSES "durum_degisimi" [0.98, 0.0];
"durum_degisimi" CAUSES "algilanan_fark" [0.95, 0.0];
"algilanan_fark" CAUSES "eylem_guncellemesi" [0.90, 0.0];
"eylem_guncellemesi" CAUSES "kendi_eylemi" [0.80, 0.78];
"dis_degisim" CAUSES "etkenin_varligi" [0.90, 0.0];
"etkenin_varligi" CAUSES "dikkat_cekimi" [0.88, 0.0];
"tekrar_eylem" CAUSES "tekrar_sonuc" [0.92, 0.0];
"tekrar_sonuc" CORRELATES_PHASE "beklenti" [0.88, 0.0];
"beklenti_uyumsuzlugu" CAUSES "dikkat_artisi" [0.95, 3.14];

// ═══════════════════════════════════════════════════════════════════
// II. NESNE SÜREKLİLİĞİ — "Gözden yitti ama hâlâ var."
//     8 aylık bir bebek battaniyenin altındaki oyuncağı aramaya devam eder.
//     Bu arama itkisi öğretilmez — doğuştan gelir.
// ═══════════════════════════════════════════════════════════════════

"nesne" CAUSES "surekliliik" [0.99, 0.0];
"surekliliik" RESONATES_WITH "hafiza_izi" [0.95, 0.0];
"gorsellik_kaybi" MEASURES "nesne" [0.30, 3.14];
"nesne" RESONATES_WITH "sinir" [0.82, 1.57];
"nesne" CORRELATES_PHASE "konum" [0.90, 0.0];
"konum_degisimi" CAUSES "hareket_tahmini" [0.85, 0.0];
"hareket_tahmini" CAUSES "arama_davranisi" [0.78, 0.0];
"gizli_nesne" RESONATES_WITH "nesne" [0.88, 0.0];
"nesne_bulma" CAUSES "beklenti_dogrulamasi" [0.90, 0.0];

// ═══════════════════════════════════════════════════════════════════
// III. FİZİKSEL SEZGİ — "Düşer. Durur. İçinde kalır."
//      Bebek hiç Newton okumamıştır ama elinden bıraktığı nesnenin
//      düşeceğini bilir. Bu sezgi evrimin hediyesidir.
// ═══════════════════════════════════════════════════════════════════

"destek_yok" CAUSES "dusme" [0.99, 3.14];
"dusme" CAUSES "temas" [0.95, 0.0];
"temas" CAUSES "durma" [0.96, 3.14];
"yerçekimi" RESONATES_WITH "asagi" [1.0, 3.14];
"kuvvet" CAUSES "ivme" [0.95, 0.0];
"ivme" CAUSES "hiz_degisimi" [0.92, 0.0];
"engel" CAUSES "yol_blokaji" [0.95, 3.14];
"yol_blokaji" CAUSES "alternatif_arayisi" [0.88, 1.57];
"sinir" CAUSES "icinde_kalma" [0.93, 1.57];
"sinir" CAUSES "disinda_kalma" [0.93, 1.57];
"temas_noktasi" CORRELATES_PHASE "kuvvet_aktarimi" [0.88, 0.0];
"siki_yuzey" RESONATES_WITH "sabitlik" [0.90, 0.0];
"kaygan_yuzey" CAUSES "denge_bozulmasi" [0.82, 3.14];
"agirlik" RESONATES_WITH "aşağı_cekme" [0.95, 3.14];

// ═══════════════════════════════════════════════════════════════════
// IV. HEDEF-ARALIK ALGISI — "Oraya varmam lazım. Buradayım. Fark var."
//     Bir çocuk uzaktaki oyuncağa bakar, arayı kapatmaya çalışır.
//     Bu gap sensing içgüdüsü hedef yönelimli hareketin temelidir.
// ═══════════════════════════════════════════════════════════════════

"mevcut_durum" MEASURES "hedef_durum" [0.90, 0.0];
"durum_farki" CAUSES "eylem_itkisi" [0.95, 0.0];
"eylem_itkisi" CAUSES "yaklasma_hareketi" [0.90, 0.0];
"yaklasma_hareketi" CAUSES "fark_azalma" [0.85, 0.0];
"fark_azalma" CAUSES "ilerleme_sinyali" [0.88, 0.0];
"ilerleme_sinyali" CAUSES "eylem_tekrari" [0.85, 0.0];
"ilerleme_yok" CAUSES "strateji_degisimi" [0.90, 1.57];
"hedef_ulasim" CAUSES "tatmin_sinyali" [0.98, 0.0];
"tatmin_sinyali" CAUSES "eylem_dondurma" [0.90, 3.14];
"durum_farki" CORRELATES_PHASE "motivasyon_seviyesi" [0.85, 0.0];

// ═══════════════════════════════════════════════════════════════════
// V. GERİBESLEME DÖNGÜSÜ — "Ödüle götüren şeyi tekrar yap."
//    Pavlov koşullanması, operant koşullanma — bunlar öğretilmez;
//    sinir sistemi bu döngüyü doğuştan çalıştırır.
// ═══════════════════════════════════════════════════════════════════

"odul_sinyali" CAUSES "yaklasma_itkisi" [0.97, 0.0];
"odul_sinyali" CAUSES "onceki_eylemin_pekistirilmesi" [0.95, 0.0];
"ceza_sinyali" CAUSES "uzaklasma_itkisi" [0.97, 3.14];
"ceza_sinyali" CAUSES "onceki_eylemin_bastirimi" [0.93, 3.14];
"yaklasma_itkisi" CAUSES "eylem_tekrari" [0.90, 0.0];
"eylem_tekrari" CAUSES "beklenti_olusumu" [0.85, 0.0];
"beklenti_olusumu" CAUSES "odul_tahmini" [0.82, 0.0];
"odul_tahmini" CORRELATES_PHASE "odul_sinyali" [0.88, 0.0];
"tahmin_hatasi" CAUSES "ogrenme_guncellemesi" [0.93, 0.0];

// ═══════════════════════════════════════════════════════════════════
// VI. ÖRÜNTÜ TANIMI — "Bu daha önce oldu. Bir sonraki ne olacak?"
//     İnsan beyni örüntü tanıma makinesidir. Bu içgüdü hayatta kalmak
//     için milyonlarca yılda seçilmiştir.
// ═══════════════════════════════════════════════════════════════════

"tekrar_eden_oruntu" CAUSES "gelecek_tahmini" [0.93, 0.0];
"oruntu" CAUSES "kategori_olusumu" [0.88, 0.0];
"kategori_olusumu" CAUSES "tanimlar_genelleme" [0.85, 0.0];
"simetri" RESONATES_WITH "tamamlama_itkisi" [0.90, 0.0];
"eksik_parca" CAUSES "tamamlama_itkisi" [0.93, 0.0];
"tamamlama_itkisi" CAUSES "doldurma_eylemi" [0.88, 0.0];
"anormallik" CAUSES "dikkat_sicaligi" [0.97, 3.14];
"dikkat_sicaligi" CAUSES "analiz_cabalama" [0.90, 0.0];
"benzerlik_algilama" CAUSES "analoji_kurma" [0.88, 0.0];
"analoji_kurma" CAUSES "yeni_duruma_transfer" [0.85, 0.0];
"gorsel_gruplama" CAUSES "kategori_siniri" [0.85, 0.0];
"renk_benzerligi" CAUSES "ayni_kategori" [0.90, 0.0];
"sekil_benzerligi" CAUSES "ayni_tur" [0.88, 0.0];

// ═══════════════════════════════════════════════════════════════════
// VII. UZAMSAL SEZGİ — "Burada. Orada. İçinde. Dışında."
//      Bebek daha konuşmadan önce uzamsal ilişkileri kavrar.
//      "Nesne kutunun içinde mi dışında mı?" sorusu dil öncesidir.
// ═══════════════════════════════════════════════════════════════════

"uzamsal_konum" RESONATES_WITH "referans_noktasi" [0.90, 0.0];
"icinde" CAUSES "sinirlanma" [0.93, 1.57];
"disinda" CAUSES "serbestlik" [0.78, 0.0];
"yukari" RESONATES_WITH "asagi" [0.88, 3.14];
"sol" RESONATES_WITH "sag" [0.85, 3.14];
"yakin" CAUSES "erisilebilirlik" [0.90, 0.0];
"uzak" CAUSES "erisilebilirlik" [0.70, 3.14];
"bitisik" CAUSES "temas_olasiligi" [0.93, 0.0];
"aralik" RESONATES_WITH "mesafe" [0.88, 0.0];
"ust" CAUSES "altta_baski" [0.85, 3.14];
"alt" RESONATES_WITH "destek" [0.90, 0.0];
"merkez" RESONATES_WITH "cevresindekiler" [0.80, 1.57];
"sinir_noktasi" CAUSES "gecis_karari" [0.90, 0.0];

// ═══════════════════════════════════════════════════════════════════
// VIII. SOSYAL/İLETİŞİM SUBSTRASI
//       "Başka bir etken gördüğümde onunla iletişim kurmak isterim."
//       Bebekler doğdukları andan itibaren yüzleri tercih eder.
//       İletişim isteği DNA'ya yazılmıştır.
// ═══════════════════════════════════════════════════════════════════

"baska_etken" CAUSES "etkilesim_itkisi" [0.90, 0.0];
"etkilesim_itkisi" CAUSES "isaret_davranisi" [0.85, 0.0];
"isaret_davranisi" CAUSES "dikkat_yonlendirme" [0.88, 0.0];
"sembol_algilama" CAUSES "anlam_arayisi" [0.93, 0.0];
"anlam_arayisi" CAUSES "yorumlama_cabalama" [0.88, 0.0];
"tepki_alma" CAUSES "iletisim_dogrulamasi" [0.90, 0.0];
"iletisim_dogrulamasi" CAUSES "tekrar_sinyal" [0.85, 0.0];
"taklit_itkisi" CAUSES "basarili_davranis_kopyalama" [0.88, 0.0];
"dikkat_paylasimi" RESONATES_WITH "ortak_zemin" [0.85, 0.0];

// ═══════════════════════════════════════════════════════════════════
// IX. ARC-AGI GRID PRİMLERİ
//     Geometrik muhakemenin temel atomları.
//     Dönüşüm kuralı, renk kimliği, uzamsal topoloji.
//     Bunlar da töz bilgidir: çocuk ilk bakışta bir şeklin
//     döndürülmüş versiyonunu "aynı şey" olarak tanır.
// ═══════════════════════════════════════════════════════════════════

"grid_hucresi" RESONATES_WITH "renk_kimligi" [0.95, 0.0];
"renk_kimligi" CAUSES "kategori_uyeligii" [0.90, 0.0];
"uzamsal_konum" RESONATES_WITH "komsu_hucre" [0.85, 0.78];
"donusum_kurali" CAUSES "kural_uygulama" [0.95, 0.0];
"kural_uygulama" CAUSES "cikti_tahmini" [0.92, 0.0];
"uzamsal_degismezlik" RESONATES_WITH "donusum_kurali" [0.88, 0.0];
"sayi_karsilastirma" CAUSES "cok_ya_da_az" [0.93, 0.0];
"yansima_ekseni" CAUSES "aynalanmis_kopyasi" [0.90, 0.0];
"dondurme_merkezi" CAUSES "dondurulmus_kopya" [0.90, 1.57];
"renk_deseni" CAUSES "tekrar_kalip" [0.88, 0.0];
"tekrar_kalip" CAUSES "sonraki_hucresi_tahmini" [0.90, 0.0];
"eksik_hucre" CAUSES "tamamlama_zorunlulugu" [0.93, 0.0];
"tamamlama_zorunlulugu" CAUSES "doldurma_eylemi" [0.90, 0.0];
"kural_tespiti" CAUSES "kural_genelleme" [0.88, 0.0];
"kural_genelleme" CAUSES "gorulmemis_duruma_uygulama" [0.85, 0.0];
"gorulmemis_duruma_uygulama" CAUSES "transfer_ogrenme" [0.88, 0.0];
"renk_degisimi" CAUSES "kategori_gecisi" [0.85, 0.0];
"desen_bolumleme" CAUSES "alt_oruntu_tespiti" [0.88, 0.0];
"alt_oruntu_tespiti" CAUSES "hiyerarsik_analiz" [0.83, 0.0];
"nesne_sayimi" CAUSES "miktar_temsili" [0.90, 0.0];
"kontur_tespiti" CAUSES "sekil_siniri" [0.90, 0.0];
"sekil_siniri" CAUSES "nesne_kimligi" [0.88, 0.0];

// ═══════════════════════════════════════════════════════════════════
// X. ÇIKIRIM ZİNCİRLERİ — Yukarıdaki primitiflerin bağlantıları
//    Birden fazla Arkhe birlikte çalışarak transfer öğrenmeyi sağlar.
// ═══════════════════════════════════════════════════════════════════

"analoji_kurma" CAUSES "cikari_transferi" [0.88, 0.0];
"cikari_transferi" CAUSES "gorulmemis_duruma_uygulama" [0.85, 0.0];
"fiziksel_sezi" RESONATES_WITH "uzamsal_konum" [0.82, 0.78];
"hedef_durum" RESONATES_WITH "tamamlama_zorunlulugu" [0.88, 0.0];
"renk_kimligi" RESONATES_WITH "kategori_olusumu" [0.85, 0.0];
"oruntu" RESONATES_WITH "donusum_kurali" [0.85, 0.0];
"beklenti" RESONATES_WITH "cikti_tahmini" [0.90, 0.0];
"nesne" RESONATES_WITH "grid_hucresi" [0.80, 0.0];
"simetri" RESONATES_WITH "yansima_ekseni" [0.90, 0.0];
"tekrar_eden_oruntu" RESONATES_WITH "tekrar_kalip" [0.90, 0.0];
"uzaklaşma_itkisi" RESONATES_WITH "alternatif_arayisi" [0.82, 0.78];
"dikkat_artisi" RESONATES_WITH "analiz_cabalama" [0.88, 0.0];
"tamamlama_itkisi" RESONATES_WITH "eksik_hucre" [0.92, 0.0];
"strateji_degisimi" RESONATES_WITH "alternatif_arayisi" [0.88, 0.78];
"etkilesim_itkisi" RESONATES_WITH "kural_tespiti" [0.75, 0.0];
"surekliliik" RESONATES_WITH "uzamsal_degismezlik" [0.88, 0.0];
"tahmin_hatasi" CAUSES "anormallik" [0.90, 3.14];
"anormallik" CAUSES "kural_tespiti" [0.85, 0.0];
"fiziksel_sezi" RESONATES_WITH "sinir" [0.85, 1.57];
"sinir" RESONATES_WITH "sekil_siniri" [0.90, 0.0];
"destek_yok" RESONATES_WITH "tamamlama_itkisi" [0.70, 1.57];
"engel" RESONATES_WITH "yol_blokaji" [0.88, 3.14];
"simetri" CAUSES "tamamlanmis_goruntu" [0.85, 0.0];
"tamamlanmis_goruntu" CAUSES "tatmin_sinyali" [0.80, 0.0];
"fiziksel_sezi" RESONATES_WITH "kuvvet_aktarimi" [0.78, 0.0];
"kuvvet_aktarimi" RESONATES_WITH "donusum_kurali" [0.72, 0.0];
"butona_dokunma_itkisi" CAUSES "tepki_beklentisi" [0.95, 0.0];
"tepki_beklentisi" CAUSES "sonucu_gorme_istegi" [0.92, 0.0];
"sonucu_gorme_istegi" CAUSES "tekrar_eylem" [0.88, 0.0];
"geri_bildirim_aldim" CAUSES "butona_dokunma_itkisi" [0.85, 0.0];
"hareket_eden_nesne" CAUSES "izleme_itkisi" [0.92, 0.0];
"izleme_itkisi" CAUSES "yore_tahmini" [0.88, 0.0];
"yore_tahmini" RESONATES_WITH "hareket_tahmini" [0.90, 0.0];
"tamamlanmis_surec" CAUSES "nesne_hareketi_tamamlandi" [0.95, 0.0];
"nesne_hareketi_tamamlandi" CAUSES "hedef_ulasim" [0.78, 0.0];
"gorsel_nesne" RESONATES_WITH "nesne" [0.90, 0.0];
"gorsel_nesne" CAUSES "dogal_hareket" [0.95, 3.14];
"dogal_hareket" RESONATES_WITH "dusme" [0.95, 0.0];
"oruntu_boslugu" CAUSES "tamamlama_itkisi" [0.93, 0.0];
"tamamlanmis_yapi" CAUSES "tatmin_sinyali" [0.90, 0.0];
"tamamlanmis_yapi" CAUSES "denge_beklentisi" [0.85, 0.0];
"serbest_alan" CAUSES "nesne_uyumu" [0.90, 0.0];
"nesne_uyumu" CAUSES "konumlandirma_karari" [0.88, 0.0];
"konumlandirma_karari" CAUSES "kendi_eylemi" [0.85, 0.0];
"deger_dususu" CAUSES "ceza_sinyali" [0.92, 3.14];
"deger_artisi" CAUSES "odul_sinyali" [0.92, 0.0];
"gorsel_alan" RESONATES_WITH "sinir" [0.90, 1.57];
"uzamsal_iliskiler" CAUSES "konumlandirma_karari" [0.90, 0.0];
"sekil_manipulasyonu" CAUSES "alternatif_uyum" [0.88, 1.57];
"alternatif_uyum" CAUSES "konumlandirma_karari" [0.85, 0.0];
"optimum_konum" CAUSES "minimum_duzensizlik" [0.88, 0.0];
"minimum_duzensizlik" CAUSES "tatmin_sinyali" [0.80, 0.0];
"dinamik_degisim" CAUSES "hizli_tepki" [0.88, 3.14];
"hizli_tepki" CAUSES "acil_karar" [0.85, 0.0];
"sinira_dayanma" CAUSES "ceza_sinyali" [0.98, 3.14];
"sinira_dayanma" CAUSES "durum_sonlanma_tahmini" [0.95, 3.14];
"durum_sonlanma_tahmini" CAUSES "refleks_tepki" [0.92, 0.0];
"refleks_tepki" CAUSES "acil_karar" [0.90, 0.0];
"oruntu_tamamlanmasi" CAUSES "cozum_tatmini" [0.95, 0.0];
"cozum_tatmini" RESONATES_WITH "tatmin_sinyali" [0.95, 0.0];
"form_temsili" RESONATES_WITH "nesne_kimligi" [0.90, 0.0];
"hareket_vektoru" CAUSES "konum_degisimi" [0.92, 3.14];
"yonelim_karari" CAUSES "kendi_eylemi" [0.90, 0.0];
"karsiliksiz_alan" CAUSES "ceza_sinyali" [0.88, 3.14];
"oruntu_eksikligi" CAUSES "kararsizlik" [0.85, 3.14];
"kararsizlik" RESONATES_WITH "durum_farki" [0.90, 0.0];
"kategori_ozelligi" RESONATES_WITH "renk_kimligi" [0.90, 0.0];
"nesne_konumu" RESONATES_WITH "uzamsal_konum" [0.92, 0.0];
"rotasyon_temsili" RESONATES_WITH "dondurulmus_kopya" [0.90, 1.57];
"sinirlayici_faktor" CAUSES "sinira_dayanma" [0.88, 3.14];
"uyumlu_konumlandirma" CAUSES "odul_sinyali" [0.90, 0.0];
"hatali_konumlandirma" CAUSES "ceza_sinyali" [0.85, 3.14];
"muhakeme_verimliligi" MEASURES "cozum_basarisi" [0.88, 0.0];
"cozum_basarisi" CAUSES "davranis_pekistirilmesi" [0.85, 0.0];
"cozum_basarisizligi" CAUSES "strateji_degisimi" [0.90, 1.57];
"surekli_eksiklik" CAUSES "strateji_degisimi" [0.83, 1.57];
"anlik_ipucu" CAUSES "hizli_eylem" [0.90, 0.0];
"hizli_eylem" RESONATES_WITH "acil_karar" [0.88, 0.0];
"gercek_zamanli_degerlendirme" CAUSES "yonelim_karari" [0.85, 0.0];
"nesne_form_tanimi" CAUSES "konumlandirma_plani" [0.88, 0.0];
"konumlandirma_plani" CAUSES "yonelim_karari" [0.85, 0.0];
"bos_uzam" CAUSES "serbest_gecis" [0.90, 0.0];
"serbest_gecis" CAUSES "nesne_uyumu" [0.88, 0.0];
"nesne_uyumu" CAUSES "kararli_denge" [0.90, 0.0];
"kararli_denge" RESONATES_WITH "minimum_duzensizlik" [0.88, 0.0];
"dogal_fizik" RESONATES_WITH "yerçekimi" [0.95, 0.0];
"dogal_fizik" RESONATES_WITH "temas" [0.90, 0.0];
"dogal_fizik" RESONATES_WITH "sinir" [0.88, 1.57];
"muhakeme_surecleri" RESONATES_WITH "dogal_fizik" [0.80, 0.0];
"muhakeme_surecleri" RESONATES_WITH "geri_bildirim_aldim" [0.82, 0.0];
"muhakeme_surecleri" RESONATES_WITH "hedef_durum" [0.80, 0.0];
"muhakeme_surecleri" RESONATES_WITH "kendi_eylemi" [0.85, 0.0];
"muhakeme_surecleri" RESONATES_WITH "odul_sinyali" [0.85, 0.0];
"muhakeme_surecleri" RESONATES_WITH "ceza_sinyali" [0.85, 0.0];
"fiziksel_sezi" RESONATES_WITH "dogal_fizik" [0.95, 0.0];
"hedef_aralik_algisi" RESONATES_WITH "durum_farki" [0.95, 0.0];
"geri_besleme_dongusu" RESONATES_WITH "eylem_guncellemesi" [0.92, 0.0];
"oruntu_tanima" RESONATES_WITH "tekrar_eden_oruntu" [0.92, 0.0];
"nedensel_etkenlik" RESONATES_WITH "kendi_eylemi" [0.95, 0.0];
"nesne_sureklilik" RESONATES_WITH "surekliliik" [0.92, 0.0];
"uzamsal_sezi" RESONATES_WITH "uzamsal_konum" [0.92, 0.0];
"sosyal_substrat" RESONATES_WITH "etkilesim_itkisi" [0.90, 0.0];
"gorsel_muhakeme_primitifi" RESONATES_WITH "donusum_kurali" [0.90, 0.0];
"gorsel_muhakeme_primitifi" RESONATES_WITH "renk_kimligi" [0.90, 0.0];
"gorsel_muhakeme_primitifi" RESONATES_WITH "uzamsal_degismezlik" [0.88, 0.0];
"gorsel_muhakeme_primitifi" RESONATES_WITH "tamamlama_itkisi" [0.88, 0.0];
"gorsel_muhakeme_primitifi" RESONATES_WITH "analoji_kurma" [0.88, 0.0];
"fikri_zemin" RESONATES_WITH "nedensel_etkenlik" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "nesne_sureklilik" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "fiziksel_sezi" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "hedef_aralik_algisi" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "geri_besleme_dongusu" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "oruntu_tanima" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "uzamsal_sezi" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "gorsel_muhakeme_primitifi" [0.90, 0.0];
"fikri_zemin" RESONATES_WITH "muhakeme_surecleri" [0.85, 0.0];
"fikri_zemin" RESONATES_WITH "sosyal_substrat" [0.80, 0.0];
"#;

// ════════════════════════════════════════════════════════════════════════════
// AssetOps / ITBench DOMAIN BİLGİSİ
// ════════════════════════════════════════════════════════════════════════════

const INDUSTRIAL_DSL: &str = r#"
// ═══════════════════════════════════════════════════════════════════
// AssetOps Bench — FailureSensorIQ Domain
// Endüstriyel ekipman arıza tespiti ve kök neden analizi
// ═══════════════════════════════════════════════════════════════════

// Sensör → Fiziksel Durum
"sensor_okuma" MEASURES "fiziksel_durum" [0.92, 0.0];
"titresim_artisi" CAUSES "rulman_asimi" [0.88, 0.0];
"rulman_asimi" CAUSES "sicaklik_artisi" [0.85, 0.0];
"sicaklik_artisi" CAUSES "arizalanma_riski" [0.93, 0.0];
"arizalanma_riski" CAUSES "bakim_alarmi" [0.96, 3.14];
"anormal_ses" CAUSES "mekanik_ariza_ipucu" [0.85, 0.0];
"mekanik_ariza_ipucu" CAUSES "kok_neden_hipotezi" [0.82, 0.0];
"yuksek_titresim" CORRELATES_PHASE "dengesizlik" [0.88, 0.0];
"yuksek_titresim" CORRELATES_PHASE "uyum_bozuklugu" [0.82, 0.78];
"termal_anomali" CORRELATES_PHASE "elektrik_arizasi" [0.80, 1.57];
"basinc_dususu" CAUSES "sizinti_gostergesi" [0.93, 0.0];
"sizinti_gostergesi" CAUSES "kacan_madde_tespiti" [0.88, 0.0];
"MTBF_asimi" CAUSES "degistirme_gerekliligi" [0.97, 0.0];
"asiri_yuk" CAUSES "motor_isi_artisi" [0.90, 0.0];
"motor_isi_artisi" CAUSES "izolasyon_bozulmasi" [0.82, 0.0];
"izolasyon_bozulmasi" CAUSES "kisa_devre_riski" [0.88, 0.0];
"kisa_devre_riski" CAUSES "acil_bakim_alarmi" [0.96, 3.14];
"yag_analizi" MEASURES "yuzey_asimasi" [0.85, 0.0];
"yuzey_asimasi" CAUSES "rulman_asimi" [0.88, 0.0];
"debi_dususu" CAUSES "pompa_verimsizligi" [0.85, 0.0];
"pompa_verimsizligi" CAUSES "basinc_dususu" [0.82, 0.0];
"corrozyon" CAUSES "yapi_zayiflamasi" [0.85, 0.0];
"yapi_zayiflamasi" CAUSES "mekanik_ariza_riski" [0.90, 0.0];
"mekanik_ariza_riski" CAUSES "bakim_alarmi" [0.93, 3.14];
"darbeli_yuk" CAUSES "rulman_asimi" [0.78, 0.0];
"ortam_nem" CAUSES "corrozyon" [0.72, 0.0];
"ortam_sicaklik" MEASURES "ekipman_sicakligi" [0.70, 0.0];
"ekipman_sicakligi" CAUSES "termal_anomali" [0.85, 0.0];
"harmonik_bozulma" CAUSES "motor_isi_artisi" [0.78, 0.0];
"harmonik_bozulma" CAUSES "izolasyon_bozulmasi" [0.72, 0.0];
"enerji_tuketim_anomalisi" CAUSES "verimsizlik_tespiti" [0.88, 0.0];
"verimsizlik_tespiti" CAUSES "kok_neden_hipotezi" [0.82, 0.0];

// Arıza Modu Sınıflandırması
"rulman_asimi" RESONATES_WITH "titresim_ariza_modu" [0.90, 0.0];
"dengesizlik" RESONATES_WITH "titresim_ariza_modu" [0.88, 0.0];
"uyum_bozuklugu" RESONATES_WITH "titresim_ariza_modu" [0.85, 0.0];
"elektrik_arizasi" RESONATES_WITH "elektrik_ariza_modu" [0.90, 0.0];
"izolasyon_bozulmasi" RESONATES_WITH "elektrik_ariza_modu" [0.88, 0.0];
"kisa_devre_riski" RESONATES_WITH "elektrik_ariza_modu" [0.90, 0.0];
"sizinti_gostergesi" RESONATES_WITH "akiskan_ariza_modu" [0.90, 0.0];
"pompa_verimsizligi" RESONATES_WITH "akiskan_ariza_modu" [0.85, 0.0];
"corrozyon" RESONATES_WITH "yapisal_ariza_modu" [0.90, 0.0];
"yapi_zayiflamasi" RESONATES_WITH "yapisal_ariza_modu" [0.90, 0.0];

// Bakım Kararları
"kok_neden_hipotezi" CAUSES "bakim_stratejisi" [0.85, 0.0];
"bakim_stratejisi" CAUSES "onleyici_bakim" [0.80, 0.0];
"bakim_stratejisi" CAUSES "kestirimci_bakim" [0.82, 0.0];
"onleyici_bakim" CAUSES "ariza_onleme" [0.85, 0.0];
"kestirimci_bakim" CAUSES "optimal_bakim_zamanlama" [0.88, 0.0];
"acil_bakim_alarmi" CAUSES "anlik_mudahale" [0.96, 0.0];
"anlik_mudahale" CAUSES "durusu_onleme" [0.90, 0.0];
"ekipman_durumu" MEASURES "kalan_faydali_omur" [0.88, 0.0];
"kalan_faydali_omur" CAUSES "degistirme_planlamasi" [0.85, 0.0];
"seferde_bakim" CAUSES "uretim_kaybi" [0.75, 3.14];
"plansiz_duruş" CAUSES "yuksek_maliyet" [0.90, 3.14];
"plansiz_duruş" CAUSES "guvenlik_riski" [0.85, 3.14];
"planlı_bakim" CAUSES "uretim_kaybi" [0.50, 3.14];
"planlı_bakim" CAUSES "guvenlik_riski" [0.20, 3.14];
"sensor_kalibrasyon" CAUSES "olcum_dogrulugu" [0.90, 0.0];
"olcum_dogrulugu" CAUSES "gercek_durum_temsili" [0.88, 0.0];

// Çok-Sensör Korelasyon
"cok_sensor_anomalisi" CAUSES "sistem_duzeyi_ariza" [0.90, 0.0];
"tek_sensor_anomalisi" CAUSES "sensor_arizasi_olasiligi" [0.70, 0.0];
"sensor_arizasi_olasiligi" CAUSES "dogrulama_gerekliligi" [0.85, 0.0];
"dogrulama_gerekliligi" CAUSES "coğaltılmış_olcum" [0.80, 0.0];
"coğaltılmış_olcum" CAUSES "olcum_dogrulugu" [0.85, 0.0];

// ═══════════════════════════════════════════════════════════════════
// ITBench — IT Altyapı Domain Bilgisi
// Performans degradasyonu ve olay korelasyonu
// ═══════════════════════════════════════════════════════════════════

// Temel Performans Metrikleri
"gecikme_artisi" CAUSES "verimlilik_dususu" [0.88, 0.0];
"cpu_doygunlugu" CAUSES "yanit_suresi_artisi" [0.93, 0.0];
"bellek_baskisi" CAUSES "swap_kullanimi" [0.88, 0.0];
"swap_kullanimi" CAUSES "gecikme_artisi" [0.85, 0.0];
"disk_io_bekleme" CAUSES "uygulama_yavaskalmasi" [0.85, 0.0];
"ag_paket_kaybi" CAUSES "baglanti_kararsizligi" [0.93, 3.14];
"baglanti_kararsizligi" CAUSES "istek_zaman_asimi" [0.88, 3.14];
"istek_zaman_asimi" CAUSES "kullanici_deneyimi_bozulmasi" [0.90, 3.14];

// Kapasite ve Kaynak Yönetimi
"kaynak_tuketimi" MEASURES "sistem_sagligi" [0.92, 3.14];
"asiri_istek_yuku" CAUSES "cpu_doygunlugu" [0.88, 0.0];
"asiri_istek_yuku" CAUSES "bellek_baskisi" [0.82, 0.0];
"bellek_sizintisi" CAUSES "bellek_baskisi" [0.90, 0.0];
"bellek_sizintisi" CAUSES "uygulama_yeniden_baslama" [0.85, 3.14];
"cache_miss_orani_yuksek" CAUSES "disk_io_bekleme" [0.83, 0.0];
"veritabani_sorgu_yavas" CAUSES "uygulama_yavaskalmasi" [0.88, 0.0];
"dizin_eksikligi" CAUSES "veritabani_sorgu_yavas" [0.85, 0.0];

// Olay Korelasyonu
"alarm_korelasyonu" CAUSES "olay_deseni" [0.88, 0.0];
"olay_deseni" CAUSES "kok_neden_analizi" [0.85, 0.0];
"SLA_ihlali" CAUSES "escalasyon_gerekliligi" [0.96, 0.0];
"cascaded_ariza" CAUSES "servis_bozulmasi" [0.93, 0.0];
"cascaded_ariza" CAUSES "ogrenme_firsati" [0.70, 0.0];
"temel_cizgi_sapma" CAUSES "anomali_tespiti" [0.88, 0.0];
"anomali_tespiti" CAUSES "alarm_tetikleme" [0.90, 0.0];
"alarm_tetikleme" CAUSES "olay_yaniti" [0.88, 0.0];
"olay_yaniti" CAUSES "cozum_eylemi" [0.85, 0.0];
"cozum_eylemi" CAUSES "sistem_toparlanma" [0.82, 0.0];
"sistem_toparlanma" CAUSES "servis_devamliligi" [0.88, 0.0];
"alarm_yorgunlugu" CAUSES "kritik_alarm_kacirma_riski" [0.88, 3.14];
"yanlis_pozitif_alarm" CAUSES "alarm_yorgunlugu" [0.80, 0.0];
"gecmis_olay_analizi" CAUSES "kok_neden_analizi" [0.80, 0.0];
"tekrarlayan_olay" CAUSES "altta_yatan_sorun_gostergesi" [0.88, 0.0];
"altta_yatan_sorun_gostergesi" CAUSES "sistemik_duzeltme_gerekliligi" [0.85, 0.0];

// Kapasite Planlama
"trafik_trendi" CAUSES "kapasite_tahmin" [0.85, 0.0];
"kapasite_tahmin" CAUSES "olcekleme_karari" [0.82, 0.0];
"olcekleme_karari" CAUSES "kaynak_tahsisi" [0.85, 0.0];
"anlık_olcekleme" CAUSES "kisa_vadeli_kapasite_artisi" [0.88, 0.0];
"dikey_olcekleme" CAUSES "tek_sunucu_guclenme" [0.82, 0.0];
"yatay_olcekleme" CAUSES "dagitik_kapasite" [0.85, 0.0];
"dagitik_kapasite" CAUSES "tekil_hata_noktasi_eliminasyonu" [0.85, 0.0];
"yedeklilik" CAUSES "yuksek_erislebilirlik" [0.90, 0.0];
"yedeklilik" CAUSES "cascaded_ariza" [0.30, 3.14];

// Güvenlik Boyutu
"guvenlik_aci_noktasi" CAUSES "sistem_tehdit_riski" [0.90, 3.14];
"sistem_tehdit_riski" CAUSES "guvenlik_alarmi" [0.93, 3.14];
"yetkisiz_erisim" CAUSES "veri_ihlali_riski" [0.90, 3.14];
"veri_ihlali_riski" CAUSES "SLA_ihlali" [0.88, 3.14];
"guvenlik_yama" CAUSES "guvenlik_aci_noktasi" [0.80, 3.14];

// Ortak Nedensellik Zincirleri (AssetOps + ITBench köprüleri)
"anomali_skoru" MEASURES "sistem_sagligi" [0.90, 3.14];
"sistem_sagligi" CAUSES "operasyonel_veri" [0.88, 0.0];
"operasyonel_veri" CAUSES "karar_destek_bilgisi" [0.85, 0.0];
"karar_destek_bilgisi" CAUSES "kok_neden_hipotezi" [0.80, 0.0];
"kok_neden_hipotezi" CAUSES "duzeltici_eylem" [0.88, 0.0];
"duzeltici_eylem" CAUSES "durum_iyilestirme" [0.85, 0.0];
"durum_iyilestirme" MEASURES "sistem_sagligi" [0.82, 0.0];
"proaktif_izleme" CAUSES "erken_anomali_tespiti" [0.88, 0.0];
"erken_anomali_tespiti" CAUSES "onleyici_eylem" [0.85, 0.0];
"onleyici_eylem" CAUSES "ariza_onleme" [0.88, 0.0];
"ariza_onleme" CAUSES "surekliliik" [0.90, 0.0];
"surekliliik" RESONATES_WITH "operasyonel_veri" [0.80, 0.0];
"sensor_verisi" RESONATES_WITH "sensor_okuma" [0.95, 0.0];
"olcum_verisii" RESONATES_WITH "kaynak_tuketimi" [0.85, 0.0];
"fiziksel_olcum" RESONATES_WITH "sensor_okuma" [0.92, 0.0];
"dijital_olcum" RESONATES_WITH "kaynak_tuketimi" [0.88, 0.0];
"kurtarma_süresi" MEASURES "olay_yaniti" [0.85, 0.0];
"kurtarma_süresi" CAUSES "SLA_ihlali" [0.82, 3.14];
"MTTR" RESONATES_WITH "kurtarma_süresi" [0.92, 0.0];
"MTBF" RESONATES_WITH "MTBF_asimi" [0.90, 0.0];
"MTTD" MEASURES "anomali_tespiti" [0.88, 0.0];
"performans_baseline" MEASURES "temel_cizgi_sapma" [0.90, 0.0];
"servis_kalitesi" RESONATES_WITH "sistem_sagligi" [0.88, 0.0];
"servis_kalitesi" RESONATES_WITH "kullanici_deneyimi_bozulmasi" [0.82, 3.14];
"konfigürasyon_hatasi" CAUSES "sistem_tehdit_riski" [0.80, 3.14];
"konfigürasyon_hatasi" CAUSES "servis_bozulmasi" [0.82, 3.14];
"izleme_kapsamı" MEASURES "proaktif_izleme" [0.85, 0.0];
"log_analizi" CAUSES "olay_deseni" [0.80, 0.0];
"log_analizi" CAUSES "anomali_tespiti" [0.78, 0.0];
"metrik_toplama" CAUSES "temel_cizgi_sapma" [0.85, 0.0];
"yapay_zeka_anomali_tespiti" CAUSES "erken_anomali_tespiti" [0.88, 0.0];
"tahminsel_analiz" CAUSES "kapasite_tahmin" [0.85, 0.0];
"tahminsel_analiz" CAUSES "kestirimci_bakim" [0.82, 0.0];
"dijital_ikiz" RESONATES_WITH "tahminsel_analiz" [0.82, 0.0];
"dijital_ikiz" RESONATES_WITH "sistem_sagligi" [0.80, 0.0];
"endüstri_40" RESONATES_WITH "yapay_zeka_anomali_tespiti" [0.80, 0.0];
"endüstri_40" RESONATES_WITH "kestirimci_bakim" [0.82, 0.0];
"endüstri_40" RESONATES_WITH "dijital_ikiz" [0.80, 0.0];
"IIoT_sensor" RESONATES_WITH "sensor_okuma" [0.90, 0.0];
"IIoT_sensor" RESONATES_WITH "proaktif_izleme" [0.85, 0.0];
"SCADA" RESONATES_WITH "sensor_okuma" [0.85, 0.0];
"SCADA" CAUSES "operasyonel_veri" [0.82, 0.0];
"PLC" RESONATES_WITH "SCADA" [0.85, 0.0];
"PLC" CAUSES "fiziksel_durum" [0.80, 0.0];
"OEE" MEASURES "ekipman_verimliligi" [0.90, 0.0];
"ekipman_verimliligi" MEASURES "sistem_sagligi" [0.85, 0.0];
"kullanılabilirlik" RESONATES_WITH "ariza_onleme" [0.88, 0.0];
"performans" RESONATES_WITH "gecikme_artisi" [0.85, 3.14];
"kalite" RESONATES_WITH "olcum_dogrulugu" [0.82, 0.0];
"OEE" RESONATES_WITH "kullanılabilirlik" [0.88, 0.0];
"OEE" RESONATES_WITH "performans" [0.88, 0.0];
"OEE" RESONATES_WITH "kalite" [0.85, 0.0];
"guvenli_calisma_siniri" CAUSES "bakim_alarmi" [0.90, 3.14];
"guvenli_calisma_siniri" RESONATES_WITH "arizalanma_riski" [0.88, 3.14];
"bakim_penceresi" CAUSES "planlı_bakim" [0.88, 0.0];
"bakim_penceresi" CAUSES "uretim_kaybi" [0.50, 3.14];
"operasyon_surekliligi" RESONATES_WITH "surekliliik" [0.90, 0.0];
"isletme_kararliligi" RESONATES_WITH "operasyon_surekliligi" [0.88, 0.0];
"varlık_yonetimi" RESONATES_WITH "MTBF" [0.85, 0.0];
"varlık_yonetimi" RESONATES_WITH "MTTR" [0.85, 0.0];
"varlık_yonetimi" RESONATES_WITH "kalan_faydali_omur" [0.88, 0.0];
"varlık_yonetimi" RESONATES_WITH "OEE" [0.82, 0.0];
"#;

// ════════════════════════════════════════════════════════════════════════════
// MATEMATİK DOMAIN — Reel Analiz + Nokta-Küme Topolojisi Temelleri
// ════════════════════════════════════════════════════════════════════════════
//
// Bu DSL, HLE Benchmark matematik kategorisi için genel amaçlı
// ansiklopedik bilgi tabanı inşa eder. Kontaminasyondan bağımsız,
// standart doktora müfredatı sıralamasını takip eder.
//
// Faz imzaları (Clifford cebirinde ortogonallik):
//   Analiz axiom'ları  → θ ≈ 0.0    (sınırlama/epsilon-delta tarzı akıl yürütme)
//   Topoloji axiom'ları → θ ≈ π/2   (dönüşümsel/değişmezlik tarzı akıl yürütme)
//   Köprü bağlantılar  → θ ≈ π/4   (alan-arası geçiş)

const MATHEMATICS_DSL: &str = r#"
// ═══════════════════════════════════════════════════════════════════
// I. METRİK UZAYLAR — Analizin temel yapı taşı
//    Metrik → açık küre → topoloji üretimi.
//    Uzaklık kavramı, matematiksel analizin tüm zeminini kurar.
// ═══════════════════════════════════════════════════════════════════

"metrik_fonksiyonu" CAUSES "uzaklik_olcumu" [0.95, 0.0];
"uzaklik_olcumu" CAUSES "acik_kure" [0.92, 0.0];
"acik_kure" CAUSES "acik_kume_tanimi" [0.90, 0.0];
"acik_kume_tanimi" CAUSES "topoloji_uretimi" [0.88, 1.5708];

"metrik_fonksiyonu" RESONATES_WITH "pozitiflik" [0.95, 0.0];
"metrik_fonksiyonu" RESONATES_WITH "simetri_ozelligi" [0.95, 0.0];
"metrik_fonksiyonu" RESONATES_WITH "ucgen_esitsizligi" [0.95, 0.0];

"acik_kume_tanimi" RESONATES_WITH "kapali_kume_tanimi" [0.90, 3.14];
"kapali_kume_tanimi" CAUSES "sinir_noktasi" [0.88, 0.0];
"sinir_noktasi" RESONATES_WITH "yigilma_noktasi" [0.85, 0.0];

// ═══════════════════════════════════════════════════════════════════
// II. DİZİLER & YAKINSAKLIK — Analiz akıl yürütmesinin çekirdeği
//    Epsilon-delta kanıtı, sınırlama hareketi.
//    Yapısal olarak "küçültme/sıkıştırma" hareketi.
// ═══════════════════════════════════════════════════════════════════

"dizi" CAUSES "yakinsaklik_sorgusu" [0.92, 0.0];
"yakinsaklik_sorgusu" CAUSES "limit_hesabi" [0.90, 0.0];
"limit_hesabi" CAUSES "epsilon_delta_kaniti" [0.95, 0.0];

"cauchy_dizisi" CAUSES "yakinsaklik_onkosulu" [0.90, 0.0];
"cauchy_dizisi" RESONATES_WITH "sinirli_dizi" [0.85, 0.0];

"sinirli_dizi" CAUSES "altdizi_secimi" [0.88, 0.0];
"altdizi_secimi" CAUSES "bolzano_weierstrass" [0.92, 0.0];
"bolzano_weierstrass" CAUSES "yakinsak_altdizi_vardir" [0.95, 0.0];

"monoton_dizi" CAUSES "sinirlilik_kontrolu" [0.90, 0.0];
"sinirlilik_kontrolu" CAUSES "monoton_yakinsaklik_teoremi" [0.95, 0.0];

// ═══════════════════════════════════════════════════════════════════
// III. SÜREKLİLİK — Fonksiyonlar arası köprü
//    Epsilon-delta tanımı, açık küme geri-görüntüsü.
//    Analizin topolojiye açılan kapısı.
// ═══════════════════════════════════════════════════════════════════

"surekli_fonksiyon" CAUSES "epsilon_delta_tanimi" [0.95, 0.0];
"epsilon_delta_tanimi" RESONATES_WITH "epsilon_delta_kaniti" [0.92, 0.0];

"surekli_fonksiyon" CAUSES "duzgun_sureklilik" [0.85, 0.0];
"duzgun_sureklilik" CAUSES "lipschitz_surekliligi" [0.82, 0.0];

"surekli_fonksiyon" CAUSES "acik_geri_goruntu" [0.90, 1.5708];
"acik_geri_goruntu" RESONATES_WITH "acik_kume_tanimi" [0.88, 1.5708];

"surekli_fonksiyon" CAUSES "ara_deger_teoremi" [0.90, 0.0];
"ara_deger_teoremi" RESONATES_WITH "baglantililik" [0.88, 1.5708];

// ═══════════════════════════════════════════════════════════════════
// IV. TAMLIK (COMPLETENESS) — Cauchy + metrik = yakınsaklık
//    Tam metrik uzayda her Cauchy dizisi yakınsar.
//    R'nin tamlığı, analizin temel postulatıdır.
// ═══════════════════════════════════════════════════════════════════

"tam_metrik_uzay" CAUSES "cauchy_yakinsar" [0.95, 0.0];
"cauchy_yakinsar" RESONATES_WITH "yakinsaklik_onkosulu" [0.92, 0.0];

"reel_sayi_dogasi" CAUSES "tam_siralama" [0.95, 0.0];
"tam_siralama" CAUSES "supremum_vardir" [0.92, 0.0];
"supremum_vardir" CAUSES "tam_metrik_uzay" [0.90, 0.0];

"tam_metrik_uzay" CAUSES "baire_kategori_teoremi" [0.88, 0.0];
"baire_kategori_teoremi" CAUSES "yogun_acik_kesisim_yogun" [0.85, 0.0];

"metrik_uzay" CAUSES "tamamlama_islemi" [0.82, 0.0];
"tamamlama_islemi" CAUSES "tam_metrik_uzay" [0.85, 0.0];

// ═══════════════════════════════════════════════════════════════════
// V. TIKIZLIK (COMPACTNESS) — Sonluluğun sonsuzla buluşması
//    Açık örtü → sonlu alt-örtü. Heine-Borel. Dizisel tıkızlık.
//    Yapısal olarak "dönüşüm altında sabit kalan" hareketi.
// ═══════════════════════════════════════════════════════════════════

"tikiz_uzay" CAUSES "acik_ortu" [0.92, 1.5708];
"acik_ortu" CAUSES "sonlu_alt_ortu" [0.95, 1.5708];

"kapali_sinirli_kume" CAUSES "heine_borel_teoremi" [0.95, 0.0];
"heine_borel_teoremi" CAUSES "tikiz_uzay" [0.95, 1.5708];

"tikiz_uzay" RESONATES_WITH "dizisel_tikizlik" [0.92, 1.5708];
"dizisel_tikizlik" CAUSES "yakinsak_altdizi_vardir" [0.90, 0.0];

"surekli_fonksiyon" CORRELATES_PHASE "tikiz_uzay" [0.90, 1.5708];

"tikiz_uzay" CAUSES "sinirli_surekli_fonksiyon" [0.90, 0.0];
"sinirli_surekli_fonksiyon" CAUSES "maksimum_minimum_vardir" [0.92, 0.0];

// ═══════════════════════════════════════════════════════════════════
// VI. BAĞLANTILILIK (CONNECTEDNESS) — Bölünmezlik
//    Topolojik bölünemezlik ve sürekli fonksiyonla korunumu.
//    Ara değer teoreminin topolojik temeli.
// ═══════════════════════════════════════════════════════════════════

"baglantili_uzay" CAUSES "bolunemezlik" [0.90, 1.5708];
"bolunemezlik" CAUSES "ara_deger_ozelligi" [0.88, 1.5708];
"ara_deger_ozelligi" RESONATES_WITH "ara_deger_teoremi" [0.90, 1.5708];

"yol_baglantili_uzay" CAUSES "baglantili_uzay" [0.88, 1.5708];
"yol_baglantili_uzay" CAUSES "surekli_egri_vardir" [0.90, 1.5708];

"baglantili_bilesen" CAUSES "maksimal_baglantili_altkume" [0.85, 1.5708];

// ═══════════════════════════════════════════════════════════════════
// VII. TOPOLOJİK UZAYLAR — Soyut yapılar
//    Topoloji tanımı, Hausdorff, homeomorfizm, topolojik değişmez.
//    Metrik uzayların ötesine geçiş.
// ═══════════════════════════════════════════════════════════════════

"topoloji_tanimi" CAUSES "acik_kume_aksiyomlari" [0.95, 1.5708];
"acik_kume_aksiyomlari" RESONATES_WITH "acik_kume_tanimi" [0.90, 1.5708];

"hausdorff_ozelligi" CAUSES "ayirilabilirlik" [0.88, 1.5708];
"ayirilabilirlik" CAUSES "tekil_limit" [0.90, 1.5708];

"homeomorfizm" CAUSES "topolojik_esdegerlik" [0.95, 1.5708];
"topolojik_esdegerlik" CAUSES "topolojik_degismez" [0.92, 1.5708];
"topolojik_degismez" RESONATES_WITH "tikiz_uzay" [0.85, 1.5708];
"topolojik_degismez" RESONATES_WITH "baglantili_uzay" [0.85, 1.5708];
"topolojik_degismez" RESONATES_WITH "hausdorff_ozelligi" [0.85, 1.5708];

// ═══════════════════════════════════════════════════════════════════
// VIII. AKSİYOM, KURAL & HEDEF TANIMLARI
//    QMEM Axiom/Rule/Goal yapıları.
//    Ezberlenmiş cümleler değil, türetilebilir sonuçlar.
// ═══════════════════════════════════════════════════════════════════

AXIOM "cauchy_dizisi_aksiyomu" DEFINES "yakinsaklik_onkosulu" [0.90, 0.0];
AXIOM "tam_metrik_uzay_aksiyomu" DEFINES "cauchy_yakinsar" [0.95, 0.0];
AXIOM "acik_kapali_kumeler_aksiyomu" DEFINES "topoloji_temeli" [0.85, 1.5708];
AXIOM "surekli_fonksiyon_aksiyomu" DEFINES "acik_geri_goruntu" [0.90, 1.5708];
AXIOM "tikizlik_aksiyomu" DEFINES "sonlu_alt_ortu" [0.92, 1.5708];
AXIOM "baglanti_aksiyomu" DEFINES "bolunemezlik" [0.90, 1.5708];

RULE "yakinsaklik_garantisi" COMPOSE "cauchy_dizisi" AND "tam_metrik_uzay" -> "dizi_yakinsar" [0.90, 0.0];
RULE "tikizlik_korunumu" COMPOSE "surekli_fonksiyon" AND "tikiz_uzay" -> "goruntu_tikiz" [0.88, 1.5708];
RULE "baglanti_korunumu" COMPOSE "surekli_fonksiyon" AND "baglantili_uzay" -> "goruntu_baglantili" [0.88, 1.5708];
RULE "heine_borel_kurali" COMPOSE "kapali_sinirli_kume" AND "reel_sayi_dogasi" -> "tikiz_kume" [0.92, 0.0];
RULE "ekstrema_teoremi" COMPOSE "surekli_fonksiyon" AND "tikiz_uzay" -> "maksimum_minimum_vardir" [0.90, 0.0];
RULE "duzgun_sureklilik_kurali" COMPOSE "surekli_fonksiyon" AND "tikiz_uzay" -> "duzgun_surekli" [0.88, 0.0];

GOAL "dizi_yakinsakligini_kanitla" REQUIRES "cauchy_ve_tamlik_dogrulandi" [0.90, 0.0];
GOAL "tikizlik_goster" REQUIRES "sonlu_alt_ortu_bulundu" [0.92, 1.5708];
GOAL "homeomorfizm_kur" REQUIRES "bijektif_surekli_tersi_surekli" [0.90, 1.5708];
GOAL "ara_deger_uygula" REQUIRES "sureklilik_ve_baglanti_dogrulandi" [0.88, 0.0];

// ═══════════════════════════════════════════════════════════════════
// IX. ÇIKARIM ZİNCİRLERİ — Alanlar arası köprüler
//    Analiz ↔ Topoloji geçişleri. GPR'ın faz imzası
//    ile ayırt edeceği akıl yürütme kalıpları.
// ═══════════════════════════════════════════════════════════════════

"epsilon_delta_tanimi" RESONATES_WITH "acik_kume_tanimi" [0.88, 0.78];
"cauchy_dizisi" RESONATES_WITH "tam_metrik_uzay" [0.92, 0.0];
"sinirli_dizi" RESONATES_WITH "tikiz_uzay" [0.85, 0.78];
"ara_deger_teoremi" RESONATES_WITH "baglantili_uzay" [0.90, 0.78];
"bolzano_weierstrass" RESONATES_WITH "dizisel_tikizlik" [0.90, 0.78];

"homeomorfizm" RESONATES_WITH "surekli_fonksiyon" [0.90, 1.5708];
"tikiz_uzay" RESONATES_WITH "kapali_kume_tanimi" [0.82, 1.5708];

"analiz_muhakemesi" RESONATES_WITH "epsilon_delta_kaniti" [0.92, 0.0];
"analiz_muhakemesi" RESONATES_WITH "limit_hesabi" [0.90, 0.0];
"analiz_muhakemesi" RESONATES_WITH "yakinsaklik_sorgusu" [0.88, 0.0];
"analiz_muhakemesi" RESONATES_WITH "monoton_yakinsaklik_teoremi" [0.85, 0.0];

"topoloji_muhakemesi" RESONATES_WITH "topolojik_degismez" [0.92, 1.5708];
"topoloji_muhakemesi" RESONATES_WITH "homeomorfizm" [0.90, 1.5708];
"topoloji_muhakemesi" RESONATES_WITH "tikiz_uzay" [0.88, 1.5708];
"topoloji_muhakemesi" RESONATES_WITH "baglantili_uzay" [0.88, 1.5708];
"topoloji_muhakemesi" RESONATES_WITH "hausdorff_ozelligi" [0.85, 1.5708];

"matematik_temeli" RESONATES_WITH "analiz_muhakemesi" [0.95, 0.0];
"matematik_temeli" RESONATES_WITH "topoloji_muhakemesi" [0.95, 1.5708];
"matematik_temeli" RESONATES_WITH "metrik_fonksiyonu" [0.90, 0.0];
"matematik_temeli" RESONATES_WITH "surekli_fonksiyon" [0.90, 0.0];
"matematik_temeli" RESONATES_WITH "tam_metrik_uzay" [0.88, 0.0];
"#;

// ════════════════════════════════════════════════════════════════════════════
// DOMAIN 2: SOYUT CEBİR — Grup, Halka, Galois ve Lie Cebirleri
// ════════════════════════════════════════════════════════════════════════════
//
// Bu DSL, Soyut Cebir (Abstract Algebra) ve Lie Teorisi için genel amaçlı
// aksiyomatik ve işlemsel bilgi tabanı inşa eder.
//
// Faz imzaları (Clifford cebirinde ortogonallik):
//   Grup teorisi / Simetri      → θ ≈ 0.0    (permutasyonel/dönel simetri)
//   Halka & Cisim / İdealler    → θ ≈ π/3    (cebirsel kapanış/yapısal çarpım)
//   Galois Teorisi / Eşleme     → θ ≈ π/2    (aracı cisim - altgrup eşlemesi)
//   Lie Cebirleri & Kök Sistemleri → θ ≈ 2π/3 (diferansiyel simetri ve Lie braketi)
//   Köprü bağlantılar           → θ ≈ π/4    (alan-arası cebirsel geçiş)

const ABSTRACT_ALGEBRA_DSL: &str = r#"
// ═══════════════════════════════════════════════════════════════════
// I. GRUP TEORİSİ — Simetrinin Cebirsel Temeli (θ ≈ 0.0)
//    Grup aksiyomları, altgruplar, normal altgruplar, Lagrange ve Sylow
// ═══════════════════════════════════════════════════════════════════

"grup_yapisi" CAUSES "ikili_islem_kapaliligi" [0.98, 0.0];
"ikili_islem_kapaliligi" CAUSES "birlesme_ozelligi" [0.98, 0.0];
"birlesme_ozelligi" CAUSES "birim_eleman_varligi" [0.98, 0.0];
"birim_eleman_varligi" CAUSES "ters_eleman_varligi" [0.98, 0.0];

"altgrup_tanimi" CAUSES "lagrange_teoremi" [0.96, 0.0];
"lagrange_teoremi" CAUSES "altgrup_mertebesi_grup_mertebesini_boler" [0.98, 0.0];
"lagrange_teoremi" CAUSES "indeks_formulu" [0.95, 0.0];

"normal_altgrup" CAUSES "sol_ve_sag_yan_kumeler_esit" [0.96, 0.0];
"normal_altgrup" CAUSES "bolum_grubu_tanimi" [0.95, 0.0];
"bolum_grubu_tanimi" CAUSES "birinci_izomorfizma_teoremi" [0.96, 0.0];

"sylow_p_altgrubu" CAUSES "sylow_varlik_teoremi" [0.95, 0.0];
"sylow_varlik_teoremi" CAUSES "sylow_sayisi_kisiti_mod_p" [0.95, 0.0];
"sylow_sayisi_kisiti_mod_p" CAUSES "sylow_sayisi_boler_m" [0.95, 0.0];
"sylow_esleniklik_teoremi" CAUSES "tum_p_sylowlar_esleniktir" [0.92, 0.0];

"devirli_grup" CAUSES "tek_uretec_vardir" [0.95, 0.0];
"devirli_grup" RESONATES_WITH "abel_grubu" [0.95, 0.0];
"simetrik_grup_Sn" CAUSES "permutasyon_yapisi" [0.92, 0.0];
"alterne_grup_An" CAUSES "cift_permutasyonlar" [0.92, 0.0];
"alterne_grup_An" RESONATES_WITH "basit_grup_n_buyuktur_4" [0.90, 0.0];

// ═══════════════════════════════════════════════════════════════════
// II. HALKA & CİSİM TEORİSİ — İki İşlemli Cebirsel Yapılar (θ ≈ π/3 = 1.0472)
//    Halkalar, idealler, bölüm halkaları, asal/maksimal idealler, PID/UFD
// ═══════════════════════════════════════════════════════════════════

"halka_yapisi" CAUSES "toplamsal_abel_grubu" [0.96, 1.0472];
"halka_yapisi" CAUSES "carpma_dagilma_ozelligi" [0.96, 1.0472];
"degismeli_birimli_halka" CAUSES "tamlik_bolgesi_adayi" [0.92, 1.0472];

"ideal_tanimi" CAUSES "yutan_altkume_ozelligi" [0.95, 1.0472];
"ideal_tanimi" CAUSES "bolum_halkasi_yapisi" [0.95, 1.0472];

"maksimal_ideal" CAUSES "bolum_halkasi_cisimdir" [0.98, 1.0472];
"asal_ideal" CAUSES "bolum_halkasi_tamlik_bolgesidir" [0.96, 1.0472];
"maksimal_ideal" RESONATES_WITH "asal_ideal" [0.92, 1.0472];

"esas_ideal_bolgesi_pid" CAUSES "her_ideal_tek_elemanla_uretulir" [0.95, 1.0472];
"oklid_bolgesi" CAUSES "esas_ideal_bolgesi_pid" [0.95, 1.0472];
"esas_ideal_bolgesi_pid" CAUSES "tekil_carpanlara_ayrilma_ufd" [0.95, 1.0472];

"polinom_halkasi" CAUSES "bolmeli_algoritma" [0.92, 1.0472];
"indirgenemez_polinom" CAUSES "maksimal_ideal_ureteci" [0.95, 1.0472];
"eisenstein_kriteri" CAUSES "polinom_indirgenemezdir" [0.95, 1.0472];
"gauss_lemmasi" CAUSES "Z_uzerinde_indirgenemezse_Q_uzerinde_de_indirgenemez" [0.92, 1.0472];

// ═══════════════════════════════════════════════════════════════════
// III. CİSİM GENİŞLEMELERİ & GALOIS TEORİSİ (θ ≈ π/2 = 1.5708)
//    Cisim genişlemeleri, ayrışma cismi, Galois grubu ve çözülebilirlik
// ═══════════════════════════════════════════════════════════════════

"cisim_genislemesi_L_K" CAUSES "vektor_uzayi_yapisi" [0.95, 1.5708];
"cisim_genislemesi_L_K" CAUSES "genisleme_derecesi" [0.95, 1.5708];
"kule_kanunu" CAUSES "derecelerin_carpimi_kurali" [0.96, 1.5708];

"cebirsel_eleman" CAUSES "minimal_polinom_varligi" [0.95, 1.5708];
"minimal_polinom_varligi" CAUSES "basit_genisleme_derecesi" [0.92, 1.5708];

"ayrisma_cismi" CAUSES "tum_kokler_mevcuttur" [0.95, 1.5708];
"normal_ve_ayrilabilir_genisleme" CAUSES "galois_genislemesi" [0.98, 1.5708];

"galois_genislemesi" CAUSES "galois_grubu_Gal_L_K" [0.98, 1.5708];
"galois_grubu_Gal_L_K" CAUSES "otomorfizma_grubu_mertebesi_dereceye_esit" [0.96, 1.5708];
"temel_galois_teoremi" CAUSES "ara_cisimler_ile_altgruplar_arasi_birebir_esleme" [0.98, 1.5708];

"cozulebilir_galois_grubu" CAUSES "polinom_radikallerle_cozulebilir" [0.95, 1.5708];
"abel_ruffini_teoremi" CAUSES "besinci_ve_yuksek_derece_genel_cozulemez" [0.95, 1.5708];
"sonlu_cisim_F_q" CAUSES "karakteristik_p_kuvveti" [0.95, 1.5708];
"sonlu_cisim_F_q" CAUSES "frobenius_otomorfizmasi" [0.92, 1.5708];

// ═══════════════════════════════════════════════════════════════════
// IV. LİE CEBİRLERİ & KÖK SİSTEMLERİ (θ ≈ 2π/3 = 2.0944)
//    Lie parantezi, Killing formu, kök uzayı ayrışımı,
//    İstisnai ve Klasik Lie Cebirleri Boyut Formülleri
// ═══════════════════════════════════════════════════════════════════

"lie_cebiri_tanimi" CAUSES "vektor_uzayi_arti_lie_parantezi" [0.98, 2.0944];
"lie_parantezi" CAUSES "iki_dogrusallik" [0.95, 2.0944];
"lie_parantezi" CAUSES "antisimetri_veya_sifir_kare" [0.95, 2.0944];
"lie_parantezi" CAUSES "jacobi_ozdesligi" [0.98, 2.0944];

"ad_temsili" CAUSES "killing_formu_tanimi" [0.95, 2.0944];
"killing_formu_dejenere_degildir" CAUSES "yari_basit_lie_cebiri" [0.98, 2.0944];

"cartan_altcebiri" CAUSES "maksimal_torusal_altcebir" [0.95, 2.0944];
"cartan_altcebiri" CAUSES "kok_uzayi_ayrisimi" [0.96, 2.0944];
"kok_uzayi_ayrisimi" CAUSES "kok_sistemi_Phi" [0.96, 2.0944];
"kok_sistemi_Phi" CAUSES "basit_kokler_Delta" [0.95, 2.0944];
"kok_sistemi_Phi" CAUSES "pozitif_kokler_Phi_arti" [0.95, 2.0944];
"kok_sistemi_Phi" CAUSES "weyl_grubu_simetrisi" [0.92, 2.0944];
"cartan_matrisi" CAUSES "dynkin_diyagrami" [0.95, 2.0944];

// İSTİSNAİ LİE CEBİRLERİ BOYUT FORMÜLÜ: dim(g) = rank + |Phi| = rank + 2|Phi^+|
"istisnai_lie_cebiri" CAUSES "cartan_boyut_formulu_rank_arti_kok_sayisi" [0.98, 2.0944];
"g2_lie_cebiri" CAUSES "g2_rank_2_kok_12_boyut_14" [0.98, 2.0944];
"f4_lie_cebiri" CAUSES "f4_rank_4_kok_48_boyut_52" [0.98, 2.0944];
"e6_lie_cebiri" CAUSES "e6_rank_6_kok_72_boyut_78" [0.98, 2.0944];
"e7_lie_cebiri" CAUSES "e7_rank_7_kok_126_boyut_133" [0.98, 2.0944];
"e8_lie_cebiri" CAUSES "e8_rank_8_kok_240_boyut_248" [0.98, 2.0944];

// KLASİK LİE CEBİRLERİ BOYUT FORMÜLLERİ:
// A_n = sl(n+1) -> dim = n(n+2) = (n+1)^2 - 1
"A_n_ozel_dogrusal_sl" CAUSES "sl_boyut_formulu_n_carpi_n_arti_2" [0.98, 2.0944];
// B_n = so(2n+1) -> dim = n(2n+1)
"B_n_tek_ortogonal_so" CAUSES "so_tek_boyut_formulu_n_carpi_2n_arti_1" [0.98, 2.0944];
// C_n = sp(2n) -> dim = n(2n+1)
"C_n_simplektik_sp" CAUSES "sp_boyut_formulu_n_carpi_2n_arti_1" [0.98, 2.0944];
// KRİTİK SP(6) DİAGNOSTİĞİ: 2n = 6 -> n = 3 -> dim(sp(6)) = 3 * (2*3 + 1) = 3 * 7 = 21
"sp_6_lie_cebiri" CAUSES "sp_6_n_esittir_3_icin_C3_ailesidir" [0.98, 2.0944];
"sp_6_n_esittir_3_icin_C3_ailesidir" CAUSES "sp_6_boyut_esittir_21" [0.99, 2.0944];
// D_n = so(2n) -> dim = n(2n-1) (Örnek: D_4 = so(8) -> 4 * 7 = 28)
"D_n_cift_ortogonal_so" CAUSES "so_cift_boyut_formulu_n_carpi_2n_eksi_1" [0.98, 2.0944];

// ═══════════════════════════════════════════════════════════════════
// V. AKSİYOM, KURAL & HEDEF TANIMLARI
// ═══════════════════════════════════════════════════════════════════

AXIOM "grup_tanim_aksiyomu" DEFINES "kapalilik_ve_ters_eleman" [0.95, 0.0];
AXIOM "lagrange_aksiyomu" DEFINES "altgrup_mertebesi_grup_mertebesini_boler" [0.98, 0.0];
AXIOM "sylow_aksiyomu" DEFINES "p_sylow_altgruplari_vardir" [0.95, 0.0];
AXIOM "maksimal_ideal_aksiyomu" DEFINES "bolum_halkasi_cisimdir" [0.98, 1.0472];
AXIOM "asal_ideal_aksiyomu" DEFINES "bolum_halkasi_tamlik_bolgesidir" [0.95, 1.0472];
AXIOM "galois_esleme_aksiyomu" DEFINES "ara_cisimler_ve_altgruplar_birebir_eslenir" [0.98, 1.5708];
AXIOM "cartan_kok_aksiyomu" DEFINES "istisnai_lie_boyutu_rank_arti_koklerdir" [0.98, 2.0944];
AXIOM "simplektik_boyut_aksiyomu" DEFINES "sp_2n_boyutu_n_carpi_2n_arti_1_dir" [0.98, 2.0944];
AXIOM "ozel_dogrusal_boyut_aksiyomu" DEFINES "sl_n_arti_1_boyutu_n_arti_1_kare_eksi_1_dir" [0.95, 2.0944];
AXIOM "ortogonal_boyut_aksiyomu" DEFINES "so_m_boyutu_m_carpimi_m_eksi_1_bolu_2_dir" [0.95, 2.0944];

RULE "lagrange_kurali" COMPOSE "grup_yapisi" AND "altgrup_tanimi" -> "mertebe_bolunur" [0.95, 0.0];
RULE "sylow_sayisi_kurali" COMPOSE "sylow_p_altgrubu" AND "grup_mertebesi" -> "sylow_sayisi_mod_p_ve_bolunurluk" [0.95, 0.0];
RULE "maksimal_ideal_cisim_kurali" COMPOSE "degismeli_birimli_halka" AND "maksimal_ideal" -> "bolum_halkasi_cisimdir" [0.98, 1.0472];
RULE "asal_ideal_tamlik_kurali" COMPOSE "degismeli_birimli_halka" AND "asal_ideal" -> "bolum_halkasi_tamlik_bolgesidir" [0.95, 1.0472];
RULE "galois_derece_kurali" COMPOSE "galois_genislemesi" AND "ayrisma_cismi" -> "galois_grubu_mertebesi_genisleme_derecesine_esittir" [0.96, 1.5708];
RULE "istisnai_lie_boyut_kurali" COMPOSE "cartan_altcebiri" AND "kok_sistemi_Phi" -> "boyut_esittir_rank_arti_kok_sayisi" [0.98, 2.0944];
RULE "simplektik_sp_boyut_kurali" COMPOSE "C_n_simplektik_sp" AND "sp_6_lie_cebiri" -> "sp_6_boyut_esittir_21" [0.99, 2.0944];

GOAL "grup_yapisi_ve_altgruplari_bul" REQUIRES "lagrange_ve_sylow_uygulandi" [0.95, 0.0];
GOAL "bolum_halkasi_yapisi_belirle" REQUIRES "ideal_maksimal_veya_asal_mi_dogrulandi" [0.95, 1.0472];
GOAL "galois_grubu_belirle" REQUIRES "ayrisma_cismi_ve_kokler_incelendi" [0.95, 1.5708];
GOAL "lie_cebiri_boyutunu_kesin_hesapla" REQUIRES "kok_sistemi_veya_klasik_form_tespit_edildi" [0.98, 2.0944];

// ═══════════════════════════════════════════════════════════════════
// VI. ÇIKARIM ZİNCİRLERİ & REZONANS KÖPRÜLERİ
// ═══════════════════════════════════════════════════════════════════

"grup_muhakemesi" RESONATES_WITH "lagrange_teoremi" [0.95, 0.0];
"grup_muhakemesi" RESONATES_WITH "sylow_varlik_teoremi" [0.92, 0.0];
"grup_muhakemesi" RESONATES_WITH "birinci_izomorfizma_teoremi" [0.92, 0.0];

"halka_muhakemesi" RESONATES_WITH "maksimal_ideal" [0.95, 1.0472];
"halka_muhakemesi" RESONATES_WITH "asal_ideal" [0.92, 1.0472];
"halka_muhakemesi" RESONATES_WITH "esas_ideal_bolgesi_pid" [0.90, 1.0472];

"galois_muhakemesi" RESONATES_WITH "galois_grubu_Gal_L_K" [0.96, 1.5708];
"galois_muhakemesi" RESONATES_WITH "temel_galois_teoremi" [0.95, 1.5708];
"galois_muhakemesi" RESONATES_WITH "ayrisma_cismi" [0.92, 1.5708];

"lie_cebiri_muhakemesi" RESONATES_WITH "cartan_altcebiri" [0.96, 2.0944];
"lie_cebiri_muhakemesi" RESONATES_WITH "kok_sistemi_Phi" [0.96, 2.0944];
"lie_cebiri_muhakemesi" RESONATES_WITH "f4_rank_4_kok_48_boyut_52" [0.98, 2.0944];
"lie_cebiri_muhakemesi" RESONATES_WITH "sp_6_boyut_esittir_21" [0.99, 2.0944];

"cebir_temeli" RESONATES_WITH "grup_muhakemesi" [0.95, 0.0];
"cebir_temeli" RESONATES_WITH "halka_muhakemesi" [0.95, 1.0472];
"cebir_temeli" RESONATES_WITH "galois_muhakemesi" [0.95, 1.5708];
"cebir_temeli" RESONATES_WITH "lie_cebiri_muhakemesi" [0.95, 2.0944];
"#;

// ════════════════════════════════════════════════════════════════════════════
// Builder Fonksiyonları
// ════════════════════════════════════════════════════════════════════════════

/// Töz bilgi (primordial cognition) kartridjini derle ve kaydet.
///
/// Bu kartridj net_g (QMEM fizik motoru) olarak kullanıldığında,
/// her inference adımında modele şu zemin sağlanır:
///   - Nesne kalıcılığı → ARC-AGI grid hücresi sürekliliği
///   - Hedef-aralik → tamamlanmamış desene doğru çekim
///   - Geri besleme → doğru adıma devam, yanlış adımdan kaçınma
///   - Fiziksel sezi → yerçekimi/temas/sınır → Tetris/ARC fizik modeli
///   - Örüntü tamamlama → simetri aksından doldurma
pub fn build_primordial_cartridge(output_path: &str) -> Result<()> {
    eprintln!("[QMEM BUILDER] Töz bilgi kartridji derleniyor...");

    let cst = compile(PRIMORDIAL_DSL)
        .map_err(|e| anyhow::anyhow!("Primordial DSL compile error: {}", e))?;

    eprintln!(
        "[QMEM BUILDER] Primordial CST: {} kavram, {} kenar",
        cst.node_count(), cst.edge_count()
    );

    let net = compile_cst_to_network(&cst);

    eprintln!(
        "[QMEM BUILDER] ResonanceNetwork: {} aktif nöron, {} kenar",
        net.n, net.edge_count()
    );

    net.save_cartridge(output_path)?;

    eprintln!("[QMEM BUILDER] ✓ Primordial cartridge: {}", output_path);
    Ok(())
}

fn load_industrial_dsl(base_dir: &str) -> String {
    let paths = [
        format!("{}/industrial_topology.txt", base_dir.trim_end_matches('/')),
        "industrial_topology.txt".to_string(),
        "../industrial_topology.txt".to_string(),
    ];
    for path in &paths {
        if std::path::Path::new(path).exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                eprintln!("[QMEM BUILDER] Loaded industrial topology from file: {}", path);
                return content;
            }
        }
    }
    eprintln!("[QMEM BUILDER] Falling back to default INDUSTRIAL_DSL");
    INDUSTRIAL_DSL.to_string()
}

pub fn build_arxiv_cartridge(input_path: &str, output_path: &str) -> Result<()> {
    eprintln!("[QMEM BUILDER] ArXiv / Custom domain kartridji derleniyor ({})...", input_path);

    let dsl = std::fs::read_to_string(input_path)
        .map_err(|e| anyhow::anyhow!("Failed to read custom DSL from {}: {}", input_path, e))?;

    let cst = compile(&dsl)
        .map_err(|e| anyhow::anyhow!("Custom DSL compile error: {}", e))?;

    eprintln!(
        "[QMEM BUILDER] Custom CST: {} kavram, {} kenar",
        cst.node_count(), cst.edge_count()
    );

    let net = compile_cst_to_network(&cst);

    eprintln!(
        "[QMEM BUILDER] ResonanceNetwork: {} aktif nöron, {} kenar",
        net.n, net.edge_count()
    );

    net.save_cartridge(output_path)?;

    eprintln!("[QMEM BUILDER] ✓ Custom cartridge: {}", output_path);
    Ok(())
}

/// Matematik domain kartridjini derle ve kaydet.
///
/// Bu kartridj Reel Analiz + Nokta-Küme Topolojisi temellerini içerir:
///   - Metrik uzaylar, diziler, yakınsaklık, süreklilik
///   - Tamlık, tıkızlık, bağlantılılık, topolojik uzaylar
///   - Faz imzaları: Analiz → θ≈0, Topoloji → θ≈π/2
pub fn build_mathematics_cartridge(output_path: &str) -> Result<()> {
    eprintln!("[QMEM BUILDER] Matematik domain kartridji derleniyor...");

    let cst = compile(MATHEMATICS_DSL)
        .map_err(|e| anyhow::anyhow!("Mathematics DSL compile error: {}", e))?;

    eprintln!(
        "[QMEM BUILDER] Mathematics CST: {} kavram, {} kenar",
        cst.node_count(), cst.edge_count()
    );

    let net = compile_cst_to_network(&cst);

    eprintln!(
        "[QMEM BUILDER] ResonanceNetwork: {} aktif nöron, {} kenar",
        net.n, net.edge_count()
    );

    net.save_cartridge(output_path)?;

    eprintln!("[QMEM BUILDER] ✓ Mathematics cartridge: {}", output_path);
    Ok(())
}

/// Soyut Cebir (Domain 2: Abstract Algebra) kartridjini derle ve kaydet.
///
/// Bu kartridj Grup, Halka, Cisim, Galois ve Lie Cebirleri temellerini içerir:
///   - Lagrange, Sylow, Bölüm grupları, İzomorfizma teoremleri
///   - Maksimal ve Asal idealler, Bölüm halkaları, PID/UFD, Polinomlar
///   - Galois genişlemeleri, Ayrışma cisimleri, Galois eşlemesi
///   - İstisnai Lie cebirleri: F4=52, G2=14, E6=78, E7=133, E8=248
///   - Klasik Lie cebirleri: sl(n+1), so(2n+1), sp(2n), so(2n)
///   - sp(6) için n=3 -> dim = 21 (Epistemic boundary çözümü)
///   - Faz imzaları: Grup -> theta=0, Halka -> theta=pi/3, Galois -> theta=pi/2, Lie -> theta=2pi/3
pub fn build_abstract_algebra_cartridge(output_path: &str) -> Result<()> {
    eprintln!("[QMEM BUILDER] Soyut Cebir (Domain 2) kartridji derleniyor...");

    let cst = compile(ABSTRACT_ALGEBRA_DSL)
        .map_err(|e| anyhow::anyhow!("Abstract Algebra DSL compile error: {}", e))?;

    eprintln!(
        "[QMEM BUILDER] Abstract Algebra CST: {} kavram, {} kenar",
        cst.node_count(), cst.edge_count()
    );

    let net = compile_cst_to_network(&cst);

    eprintln!(
        "[QMEM BUILDER] ResonanceNetwork: {} aktif nöron, {} kenar",
        net.n, net.edge_count()
    );

    net.save_cartridge(output_path)?;

    eprintln!("[QMEM BUILDER] ✓ Abstract Algebra cartridge: {}", output_path);
    Ok(())
}

/// AssetOps/ITBench endüstriyel domain kartridjini derle ve kaydet.
///
/// Bu kartridj net_g olarak kullanıldığında, FailureSensorIQ ve
/// ITBench benchmark soruları için fizik zemin sağlanır:
///   - Titreşim → rulman aşımı → sıcaklık artışı kausal zinciri
///   - CPU doygunluğu → yanıt süresi artışı kausal zinciri
///   - MTBF/MTTR endüstriyel sabitleri
///   - SLA ihlali → eskalasyon zorunluluğu
///   - Anomali skoru → kök neden hipotezi → düzeltici eylem
pub fn build_industrial_cartridge(output_path: &str) -> Result<()> {
    eprintln!("[QMEM BUILDER] Endüstriyel domain kartridji derleniyor...");

    let base_dir = std::path::Path::new(output_path)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".");
    let dsl = load_industrial_dsl(base_dir);

    let cst = compile(&dsl)
        .map_err(|e| anyhow::anyhow!("Industrial DSL compile error: {}", e))?;

    eprintln!(
        "[QMEM BUILDER] Industrial CST: {} kavram, {} kenar",
        cst.node_count(), cst.edge_count()
    );

    let net = compile_cst_to_network(&cst);

    eprintln!(
        "[QMEM BUILDER] ResonanceNetwork: {} aktif nöron, {} kenar",
        net.n, net.edge_count()
    );

    net.save_cartridge(output_path)?;

    eprintln!("[QMEM BUILDER] ✓ Industrial cartridge: {}", output_path);
    Ok(())
}

/// Tüm kartridjleri derle ve kaydet.
/// `base_dir`: çıktı dizini (ör: "C:/Users/Burak/Desktop/sodals_core")
pub fn build_all_cartridges(base_dir: &str) -> Result<()> {
    let primordial_path = format!("{}/primordial_cognition.qmem", base_dir.trim_end_matches('/'));
    let industrial_path = format!("{}/industrial_topology.qmem", base_dir.trim_end_matches('/'));
    let mathematics_path = format!("{}/mathematics_analysis_topology.qmem", base_dir.trim_end_matches('/'));
    let abstract_algebra_path = format!("{}/abstract_algebra.qmem", base_dir.trim_end_matches('/'));
    let arxiv_path = format!("{}/arxiv_physics.qmem", base_dir.trim_end_matches('/'));

    build_primordial_cartridge(&primordial_path)?;
    build_industrial_cartridge(&industrial_path)?;
    build_mathematics_cartridge(&mathematics_path)?;
    build_abstract_algebra_cartridge(&abstract_algebra_path)?;
    
    // Optional: build arxiv cartridge if arxiv_scraped_topology.txt exists
    let scraped_path = format!("{}/arxiv_scraped_topology.txt", base_dir.trim_end_matches('/'));
    if std::path::Path::new(&scraped_path).exists() {
        if let Err(e) = build_arxiv_cartridge(&scraped_path, &arxiv_path) {
            eprintln!("[QMEM BUILDER] Arxiv cartridge hatasi: {}", e);
        }
    }

    eprintln!("[QMEM BUILDER] ═══════════════════════════════════════════");
    eprintln!("[QMEM BUILDER] Tüm kartridjler hazır:");
    eprintln!("[QMEM BUILDER]   Töz      → {}", primordial_path);
    eprintln!("[QMEM BUILDER]   Domain   → {}", industrial_path);
    eprintln!("[QMEM BUILDER]   Math (D1)→ {}", mathematics_path);
    eprintln!("[QMEM BUILDER]   Algebra(D2)→ {}", abstract_algebra_path);
    if std::path::Path::new(&arxiv_path).exists() {
        eprintln!("[QMEM BUILDER]   ArXiv    → {}", arxiv_path);
    }
    eprintln!("[QMEM BUILDER] ═══════════════════════════════════════════");
    eprintln!("[QMEM BUILDER] Kullanım:");
    eprintln!("[QMEM BUILDER]   LlmEngine::load_qmem_cartridge(&path) → net_g");
    eprintln!("[QMEM BUILDER]   ArkheCognitionPipeline::run(net_f, net_g, prompt, srf)");
    eprintln!("[QMEM BUILDER] ═══════════════════════════════════════════");

    Ok(())
}

/// Tüm kartridjleri birleştir: primordial zemine endüstriyel, analiz ve soyut cebir domain bilgilerini ekle.
/// Birleşik kartridj tüm bilgi katmanlarını tek net_g altında taşır.
pub fn build_merged_cartridge(base_dir: &str, output_filename: &str) -> Result<()> {
    let industrial_dsl = load_industrial_dsl(base_dir);
    let mut merged_dsl = format!(
        "{}\n{}\n{}\n{}",
        PRIMORDIAL_DSL, industrial_dsl, MATHEMATICS_DSL, ABSTRACT_ALGEBRA_DSL
    );
    
    let scraped_path = format!("{}/arxiv_scraped_topology.txt", base_dir.trim_end_matches('/'));
    if std::path::Path::new(&scraped_path).exists() {
        if let Ok(arxiv_dsl) = std::fs::read_to_string(&scraped_path) {
            merged_dsl.push_str("\n");
            merged_dsl.push_str(&arxiv_dsl);
            eprintln!("[QMEM BUILDER] ArXiv DSL merged into main cartridge.");
        }
    }

    eprintln!("[QMEM BUILDER] Birleşik kartridj derleniyor (töz + domain + d1_math + d2_algebra + arxiv)...");

    let cst = compile(&merged_dsl)
        .map_err(|e| anyhow::anyhow!("Merged DSL compile error: {}", e))?;

    eprintln!(
        "[QMEM BUILDER] Merged CST: {} kavram, {} kenar",
        cst.node_count(), cst.edge_count()
    );

    let net = compile_cst_to_network(&cst);
    let output_path = format!("{}/{}", base_dir.trim_end_matches('/'), output_filename);
    net.save_cartridge(&output_path)?;

    eprintln!("[QMEM BUILDER] ✓ Merged cartridge: {}", output_path);
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// TESTS
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primordial_dsl_compiles() {
        let cst = compile(PRIMORDIAL_DSL).expect("Primordial DSL compile hatası");
        assert!(cst.node_count() >= 50, "Primordial: en az 50 kavram bekleniyor");
        assert!(cst.edge_count() >= 80, "Primordial: en az 80 kenar bekleniyor");

        // Kritik töz kavramlar mevcut mu?
        assert!(cst.nodes.contains_key("kendi_eylemi"), "nedensel_etkenlik eksik");
        assert!(cst.nodes.contains_key("nesne"), "nesne_sureklilik eksik");
        assert!(cst.nodes.contains_key("yerçekimi"), "fiziksel sezi eksik");
        assert!(cst.nodes.contains_key("durum_farki"), "hedef_aralik eksik");
        assert!(cst.nodes.contains_key("odul_sinyali"), "geri besleme eksik");
        assert!(cst.nodes.contains_key("donusum_kurali"), "arc_agi_primitifi eksik");
        assert!(cst.nodes.contains_key("fikri_zemin"), "ana toplama düğümü eksik");
    }

    #[test]
    fn test_industrial_dsl_compiles() {
        let cst = compile(INDUSTRIAL_DSL).expect("Industrial DSL compile hatası");
        assert!(cst.node_count() >= 40, "Industrial: en az 40 kavram bekleniyor");
        assert!(cst.edge_count() >= 60, "Industrial: en az 60 kenar bekleniyor");

        // Kritik domain kavramlar
        assert!(cst.nodes.contains_key("titresim_artisi"), "vibration eksik");
        assert!(cst.nodes.contains_key("arizalanma_riski"), "failure risk eksik");
        assert!(cst.nodes.contains_key("SLA_ihlali"), "ITBench SLA eksik");
        assert!(cst.nodes.contains_key("anomali_tespiti"), "anomaly detection eksik");
        assert!(cst.nodes.contains_key("MTBF_asimi"), "MTBF eksik");
        assert!(cst.nodes.contains_key("kok_neden_hipotezi"), "root cause eksik");
    }

    #[test]
    fn test_primordial_builds_network() {
        let cst = compile(PRIMORDIAL_DSL).unwrap();
        let net = compile_cst_to_network(&cst);

        assert!(net.n >= 50);
        assert!(net.holographic_memory.len() >= 50);

        // Her kavramın VSA imzası var mı?
        for sig in &net.holographic_memory {
            assert_eq!(sig.len(), 4096, "VSA imzası 4096-D olmalı");
        }
    }

    #[test]
    fn test_industrial_builds_network() {
        let cst = compile(INDUSTRIAL_DSL).unwrap();
        let net = compile_cst_to_network(&cst);

        assert!(net.n >= 40);
        assert!(net.holographic_memory.len() >= 40);

        // Titreşim → Rulman → Sıcaklık kausal zinciri var mı?
        assert!(net.labels.contains_key("titresim_artisi"));
        assert!(net.labels.contains_key("rulman_asimi"));
        assert!(net.labels.contains_key("sicaklik_artisi"));
    }

    #[test]
    fn test_merged_dsl_compiles() {
        let merged = format!("{}\n{}", PRIMORDIAL_DSL, INDUSTRIAL_DSL);
        let cst = compile(&merged).expect("Merged DSL compile hatası");
        // Birleşik → daha fazla kavram
        assert!(cst.node_count() >= 80);
        assert!(cst.nodes.contains_key("fikri_zemin"));
        assert!(cst.nodes.contains_key("varlık_yonetimi"));
    }

    #[test]
    fn test_mathematics_dsl_compiles() {
        let cst = compile(MATHEMATICS_DSL).expect("Mathematics DSL compile hatası");
        assert!(cst.node_count() >= 40, "Mathematics: en az 40 kavram bekleniyor, {} bulundu", cst.node_count());
        assert!(cst.edge_count() >= 60, "Mathematics: en az 60 kenar bekleniyor, {} bulundu", cst.edge_count());

        // Kritik matematik kavramlar mevcut mu?
        assert!(cst.nodes.contains_key("cauchy_dizisi"), "cauchy_dizisi eksik");
        assert!(cst.nodes.contains_key("tam_metrik_uzay"), "tam_metrik_uzay eksik");
        assert!(cst.nodes.contains_key("tikiz_uzay"), "tikiz_uzay eksik");
        assert!(cst.nodes.contains_key("homeomorfizm"), "homeomorfizm eksik");
        assert!(cst.nodes.contains_key("surekli_fonksiyon"), "surekli_fonksiyon eksik");
        assert!(cst.nodes.contains_key("baglantili_uzay"), "baglantili_uzay eksik");
        assert!(cst.nodes.contains_key("matematik_temeli"), "matematik_temeli toplama düğümü eksik");

        // AXIOM, RULE, GOAL sayıları
        assert!(cst.axioms.len() >= 6, "En az 6 AXIOM bekleniyor, {} bulundu", cst.axioms.len());
        assert!(cst.rules.len() >= 6, "En az 6 RULE bekleniyor, {} bulundu", cst.rules.len());
        assert!(cst.goals.len() >= 4, "En az 4 GOAL bekleniyor, {} bulundu", cst.goals.len());
    }

    #[test]
    fn test_mathematics_builds_network() {
        let cst = compile(MATHEMATICS_DSL).unwrap();
        let net = compile_cst_to_network(&cst);

        assert!(net.n >= 40);
        assert!(net.holographic_memory.len() >= 40);

        // Her kavramın VSA imzası var mı?
        for sig in &net.holographic_memory {
            assert_eq!(sig.len(), 4096, "VSA imzası 4096-D olmalı");
        }

        // Analiz ↔ Topoloji temel kavramlar
        assert!(net.labels.contains_key("cauchy_dizisi"));
        assert!(net.labels.contains_key("tikiz_uzay"));
        assert!(net.labels.contains_key("homeomorfizm"));
    }

    #[test]
    fn test_merged_with_math_dsl_compiles() {
        let merged = format!("{}\n{}\n{}", PRIMORDIAL_DSL, INDUSTRIAL_DSL, MATHEMATICS_DSL);
        let cst = compile(&merged).expect("Merged+Math DSL compile hatası");
        // All three domains combined
        assert!(cst.node_count() >= 120, "Merged+Math: en az 120 kavram bekleniyor, {} bulundu", cst.node_count());
        assert!(cst.nodes.contains_key("fikri_zemin")); // primordial
        assert!(cst.nodes.contains_key("varlık_yonetimi")); // industrial
        assert!(cst.nodes.contains_key("matematik_temeli")); // math
    }

    #[test]
    fn test_abstract_algebra_dsl_compiles() {
        let cst = compile(ABSTRACT_ALGEBRA_DSL).expect("Abstract Algebra DSL compile hatası");
        assert!(cst.node_count() >= 50, "Abstract Algebra: en az 50 kavram bekleniyor, {} bulundu", cst.node_count());
        assert!(cst.edge_count() >= 70, "Abstract Algebra: en az 70 kenar bekleniyor, {} bulundu", cst.edge_count());

        // Kritik cebir ve Lie kavramları mevcut mu?
        assert!(cst.nodes.contains_key("grup_yapisi"), "grup_yapisi eksik");
        assert!(cst.nodes.contains_key("lagrange_teoremi"), "lagrange_teoremi eksik");
        assert!(cst.nodes.contains_key("sylow_p_altgrubu"), "sylow_p_altgrubu eksik");
        assert!(cst.nodes.contains_key("maksimal_ideal"), "maksimal_ideal eksik");
        assert!(cst.nodes.contains_key("asal_ideal"), "asal_ideal eksik");
        assert!(cst.nodes.contains_key("galois_grubu_Gal_L_K"), "galois_grubu eksik");
        assert!(cst.nodes.contains_key("temel_galois_teoremi"), "temel_galois_teoremi eksik");
        assert!(cst.nodes.contains_key("cartan_altcebiri"), "cartan_altcebiri eksik");
        assert!(cst.nodes.contains_key("kok_sistemi_Phi"), "kok_sistemi_Phi eksik");
        assert!(cst.nodes.contains_key("f4_rank_4_kok_48_boyut_52"), "F4 boyut tanımı eksik");
        assert!(cst.nodes.contains_key("sp_6_boyut_esittir_21"), "sp(6) = 21 diagnostik tanımı eksik");
        assert!(cst.nodes.contains_key("cebir_temeli"), "cebir_temeli toplama düğümü eksik");

        // AXIOM, RULE, GOAL sayıları
        assert!(cst.axioms.len() >= 8, "En az 8 AXIOM bekleniyor, {} bulundu", cst.axioms.len());
        assert!(cst.rules.len() >= 6, "En az 6 RULE bekleniyor, {} bulundu", cst.rules.len());
        assert!(cst.goals.len() >= 4, "En az 4 GOAL bekleniyor, {} bulundu", cst.goals.len());
    }

    #[test]
    fn test_abstract_algebra_builds_network() {
        let cst = compile(ABSTRACT_ALGEBRA_DSL).unwrap();
        let net = compile_cst_to_network(&cst);

        assert!(net.n >= 50);
        assert!(net.holographic_memory.len() >= 50);

        // Her kavramın VSA imzası 4096-D mi?
        for sig in &net.holographic_memory {
            assert_eq!(sig.len(), 4096, "VSA imzası 4096-D olmalı");
        }

        // Temel cebirsel etiketler ağda kayıtlı mı?
        assert!(net.labels.contains_key("lagrange_teoremi"));
        assert!(net.labels.contains_key("maksimal_ideal"));
        assert!(net.labels.contains_key("galois_grubu_Gal_L_K"));
        assert!(net.labels.contains_key("sp_6_boyut_esittir_21"));
    }

    #[test]
    fn test_merged_all_domains_dsl_compiles() {
        let merged = format!(
            "{}\n{}\n{}\n{}",
            PRIMORDIAL_DSL, INDUSTRIAL_DSL, MATHEMATICS_DSL, ABSTRACT_ALGEBRA_DSL
        );
        let cst = compile(&merged).expect("Merged All Domains DSL compile hatası");
        assert!(cst.node_count() >= 180, "All Domains: en az 180 kavram bekleniyor, {} bulundu", cst.node_count());
        assert!(cst.nodes.contains_key("fikri_zemin")); // primordial
        assert!(cst.nodes.contains_key("varlık_yonetimi")); // industrial
        assert!(cst.nodes.contains_key("matematik_temeli")); // math (D1)
        assert!(cst.nodes.contains_key("cebir_temeli")); // algebra (D2)
    }
}
