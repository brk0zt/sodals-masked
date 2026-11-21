# SODALS-OS v1.0 — İzole Otonom Günlük Görev Mini İşletim Sistemi

SODALS (Svon) yapay zekasının ana Windows bilgisayarınıza **hiçbir erişimi olmadan** web sayfaları açabilmesi, kalıcı Markdown notlar kaydedebilmesi, otonom günlük görevler (`daily:HH:MM` / `every:Nm`) çalıştırabilmesi ve izole bir Linux çalışma alanında komut yürütebilmesi için kurulmuş **Alpine Linux 3.21.10 (QEMU x86_64) Mikro-VM** ortamıdır.

## Güvenlik & İzolasyon Garantileri

1. **Ayrı Linux Çekirdeği & Sanal Disk:** `Linux 6.12.115-0-virt (x86_64)` çekirdeği, `4 GB` dinamik `.qcow2` sanal disk (`images/sodals-os.qcow2`).
2. **Ana Windows Diskine Sıfır Erişim:** Hiçbir `9p`, `virtiofs`, `SMB` veya `/mnt/c` paylaşımı bağlı değildir (`host_filesystem_mounts = []`, `host_c_drive_accessible = false`).
3. **Yetkisiz Kullanıcı (`svon`):** Misafir servis ve tüm shell komutları root yetkisi olmayan `svon` kullanıcısıyla `/home/svon/workspace` içinde çalışır.
4. **Çift Katmanlı Ağ Koruması:** Hem Alpine `iptables` çıkış kuralı hem de Python SSRF süzgeci yerel ağ (`127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`) erişimini engeller; yalnızca halka açık `http://` ve `https://` web sitelerine izin verir.
5. **Bearer Token Korumalı Loopback Köprüsü:** Windows host yalnızca `127.0.0.1:11440` üzerinden rastgele üretilmiş token (`secrets/api_token.txt`) ile bağlanır.
6. **Anlık Geri Alma (`golden_clean` Snapshot):** İstenildiği an tek komutla fabrika ayarlarına döndürülebilir.

## Hızlı Kullanım Komutları

Proje kök dizininden (`C:\Users\Burak\Desktop\sodals_core`):

| Komut | Açıklama |
|---|---|
| `sodals_os.bat start` | İzole Mini-OS sanal makinesini başlatır |
| `sodals_os.bat dashboard` | Canlı Web Kontrol Panelini (`http://127.0.0.1:11440`) tarayıcıda açar |
| `sodals_os.bat agent` | Mini-OS'i başlatır ve SODALS'ı `--os-agent` (otonom web/not/görev) modunda açar |
| `sodals_os.bat status` | VM durumu, RAM/disk ve izolasyon denetimini gösterir |
| `sodals_os.bat stop` | Mini-OS'i güvenli ACPI sinyaliyle kapatır |
| `sodals_os.bat reset` | Sanal diski `golden_clean` anlık görüntüsüne sıfırlar |

Sohbet terminalinde (`sodals_chat.py`) doğrudan `/os on`, `/os durum` veya `/os <görev>` komutlarıyla da geçiş yapılabilir.

## İsteğe Bağlı: 10x Donanım Hızlandırma (WHPX)

Mevcut kurulum yeniden başlatma gerektirmeden çok çekirdekli `tcg` hızlandırıcısıyla çalışır. Arka plan GPU işleriniz bittiğinde ve bilgisayarı yeniden başlatmaya hazır olduğunuzda Yönetici PowerShell ile aşağıdaki betiği bir kez çalıştırmanız yeterlidir:

```powershell
powershell -ExecutionPolicy Bypass -File .\sodals_os\enable_whpx_admin.ps1
```

Yeniden başlatma sonrasında `vm_manager.py` otomatik olarak `WHPX` donanım sanallaştırmasına geçer.
