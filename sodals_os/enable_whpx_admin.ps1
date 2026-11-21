#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Enables Windows Hypervisor Platform (WHPX) so QEMU accelerates SODALS-OS
    at native Intel i9-13900HX hardware speed (~10x faster boot and execution).
.NOTES
    Run this ONLY when you are ready to reboot Windows (e.g. after background GPU tasks finish).
    Once enabled and rebooted, sodals_os\vm_manager.py auto-detects WHPX automatically.
#>
Write-Host "=== SODALS-OS WHPX Donanim Hizlandirma Etkinlestirme ===" -ForegroundColor Cyan
dism.exe /online /enable-feature /featurename:HypervisorPlatform /all /norestart
$rc = $LASTEXITCODE
Write-Host ""
if ($rc -ne 0 -and $rc -ne 3010) {
    Write-Host "[HATA] DISM HypervisorPlatform etkinlestirmesi basarisiz oldu (exit code: $rc)." -ForegroundColor Red
    exit $rc
}
Write-Host "Tamamlandi! Degisikligin aktif olmasi icin bilgisayari 1 kez yeniden baslatin." -ForegroundColor Green
Write-Host "Yeniden baslatma sonrasinda SODALS-OS otomatik olarak WHPX donanim hizlandirmasina gececektir." -ForegroundColor Green
