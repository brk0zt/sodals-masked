#!/bin/bash
# =============================================================================
# ADIM 1: İşletim Sistemi İzolasyon Stratejisi
# Linux PREEMPT_RT + isolcpus + cgroups v2
# =============================================================================
# Karma Kritiklik (Mixed-Criticality) Bölümlemesi:
#   Çekirdek 0-1: Gerçek Zamanlı (RT) — OPC UA + Sensör polling
#   Çekirdek 2-3+: Non-RT — Qwen 8B, Sentinel, Python hibrit
#
# Kullanım:
#   sudo ./rt_isolation_setup.sh
#
# NOT: Bu script sistem seviyesinde değişiklik yapar.
# Üretim ortamında GRUB ile kalıcı `isolcpus` eklenmelidir.
# =============================================================================

set -euo pipefail

# --- 1.1 CPU İzolasyonu (isolcpus) ---
# `isolcpus` kernel parametresi ile RT çekirdekleri scheduler'dan çıkarılır.
# Kalıcı ayar için /etc/default/grub'a yazılır.
echo "[1/5] isolcpus GRUB yapılandırması kontrol ediliyor..."

RT_CORES="0,1"
NONRT_CORES="2-15"

if grep -q "isolcpus=" /etc/default/grub 2>/dev/null; then
    echo "GRUB'da isolcpus zaten var. Mevcut ayar:"
    grep "GRUB_CMDLINE_LINUX" /etc/default/grub | grep -o "isolcpus=[^\" ]*"
else
    echo "isolcpus ekleniyor..."
    sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"isolcpus=${RT_CORES} nohz_full=${RT_CORES} rcu_nocbs=${RT_CORES} /" /etc/default/grub
    update-grub
    echo "GRUB güncellendi. Sistem yeniden başlatılmalı (reboot)."
fi

# --- 1.2 IRQ Affinity (Kesmeleri RT çekirdeklerinden uzaklaştır) ---
echo "[2/5] IRQ affinity ayarlanıyor (RT çekirdeklerinden IRQ kaldırma)..."
for irq in $(ls /proc/irq/); do
    if [[ -f "/proc/irq/${irq}/smp_affinity" ]]; then
        # RT çekirdeklerine (0,1) IRQ taşımasını engelle
        # Non-RT çekirdeklerine (2-15) mask: 0xFFFC (1111111111111100)
        echo "FFFC" > "/proc/irq/${irq}/smp_affinity" 2>/dev/null || true
    fi
done

# --- 1.3 cgroups v2 ile RT/Non-RT Bölümleme ---
echo "[3/5] cgroups v2 yapılandırması..."

CGROUP2_MOUNT="/sys/fs/cgroup"
if ! mountpoint -q "$CGROUP2_MOUNT"; then
    mount -t cgroup2 none "$CGROUP2_MOUNT" || true
fi

# RT cgroup: Yüksek öncelik, sadece Core 0-1
mkdir -p "${CGROUP2_MOUNT}/sodals_rt"
echo "0-1" > "${CGROUP2_MOUNT}/sodals_rt/cpuset.cpus"
echo "root" > "${CGROUP2_MOUNT}/sodals_rt/cpuset.mems"
echo "1" > "${CGROUP2_MOUNT}/sodals_rt/cpu.uclamp.min"
echo "-19" > "${CGROUP2_MOUNT}/sodals_rt/cpu.prio"

# Non-RT cgroup: Düşük öncelik, Core 2+
mkdir -p "${CGROUP2_MOUNT}/sodals_ai"
echo "2-15" > "${CGROUP2_MOUNT}/sodals_ai/cpuset.cpus"
echo "root" > "${CGROUP2_MOUNT}/sodals_ai/cpuset.mems"
echo "0" > "${CGROUP2_MOUNT}/sodals_ai/cpu.uclamp.min"

# --- 1.4 SCHED_FIFO ve RT Priorite Ayarı ---
echo "[4/5] RT scheduling limitleri ayarlanıyor..."

# /etc/security/limits.d/sodals_rt.conf
 cat << 'EOF' > /etc/security/limits.d/sodals_rt.conf
# SODALS RT süreçleri: SCHED_FIFO + öncelik 99
@sodals_rt soft rtprio 99
@sodals_rt hard rtprio 99
@sodals_rt soft memlock unlimited
@sodals_rt hard memlock unlimited
EOF

# --- 1.5 Kernel Parametreleri (Runtime) ---
echo "[5/5] Kernel RT parametreleri ayarlanıyor..."

# RCU offload (RT çekirdeklerinde RCU callback çalışmaz)
sysctl -w kernel.rcu_nocb_poll=1 2>/dev/null || true

# Timer tick: RT çekirdeklerinde tickless
sysctl -w kernel.watchdog_thresh=15 2>/dev/null || true

# Memory: Swappiness sıfır (RT süreçler swap'e düşmemeli)
sysctl -w vm.swappiness=1
sysctl -w vm.zone_reclaim_mode=0

echo "=== KURULUM TAMAMLANDI ==="
echo ""
echo "Adımlar:"
echo "  1. Sistemi yeniden başlat (reboot) — isolcpus aktif olması için gerekli."
echo "  2. RT sürecini cgroups'a ata:"
echo "     sudo sodals_rt.service'i başlat (systemd unit dosyası ayrıca verilmiştir)."
echo "  3. /proc/cmdline ile isolcpus kontrolü yap."
echo ""
echo "RT Çekirdekler (isole): Core ${RT_CORES}"
echo "Non-RT Çekirdekler: Core ${NONRT_CORES}"
