#!/usr/bin/env python3
r"""
SODALS-OS Host VM Manager (Windows / Linux)
===========================================
Manages the isolated Alpine Linux QEMU micro-VM (`SODALS-OS`) with:
  - Auto-detection of WHPX hardware acceleration vs multi-threaded TCG (zero reboot)
  - Zero host filesystem mounts (host C:\ is physically unreachable from VM)
  - Pure-Python NoCloud `cidata.iso` provisioning (no mkisofs needed)
  - Clean ACPI shutdown via QEMU QMP monitor on 127.0.0.1:11441
  - Instant `.qcow2` snapshot & factory reset (`golden_clean`)
  - Live Web Control Panel launcher (`http://127.0.0.1:11440/?token=...`)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OS_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = OS_ROOT / "images"
GUEST_DIR = OS_ROOT / "guest"
SECRETS_DIR = OS_ROOT / "secrets"
LOGS_DIR = OS_ROOT / "logs"
RUN_DIR = OS_ROOT / "run"

BASE_IMAGE = IMAGES_DIR / "alpine-3.21.10-base.qcow2"
BASE_IMAGE_SHA512 = (
    "c0e7ee119af25d335af0bbb22dd29b13ff6c4c3d6b7d57e391971b41e7e9ce80"
    "b467522978cde2509b374777ccb2239ecadbdb3b86fedb8d720441c28c7fd58d"
)
VM_DISK = IMAGES_DIR / "sodals-os.qcow2"
CIDATA_ISO = IMAGES_DIR / "cidata.iso"
TOKEN_FILE = SECRETS_DIR / "api_token.txt"
PID_FILE = RUN_DIR / "vm.pid"
CONSOLE_LOG = LOGS_DIR / "vm_console.log"
DAEMON_SRC = GUEST_DIR / "os_daemon.py"

API_HOST = "127.0.0.1"
API_PORT = 11440
QMP_PORT = 11441
SNAPSHOT_NAME = "golden_clean"

for _d in (IMAGES_DIR, SECRETS_DIR, LOGS_DIR, RUN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if str(OS_ROOT) not in sys.path:
    sys.path.insert(0, str(OS_ROOT))

from iso_builder import build_nocloud_iso  # noqa: E402


# ---------------------------------------------------------------------------
# QEMU Binary & Accelerator Detection
# ---------------------------------------------------------------------------
def find_qemu_binaries() -> Tuple[Path, Path]:
    candidates = [
        OS_ROOT / "qemu",
        Path(r"C:\Program Files\qemu"),
        Path(r"C:\Program Files (x86)\qemu"),
        Path.home() / "scoop" / "apps" / "qemu" / "current",
    ]
    qemu_exe = shutil.which("qemu-system-x86_64")
    img_exe = shutil.which("qemu-img")

    if not qemu_exe:
        for base in candidates:
            p = base / ("qemu-system-x86_64.exe" if os.name == "nt" else "qemu-system-x86_64")
            if p.exists():
                qemu_exe = str(p)
                break
    if not img_exe:
        for base in candidates:
            p = base / ("qemu-img.exe" if os.name == "nt" else "qemu-img")
            if p.exists():
                img_exe = str(p)
                break

    if not qemu_exe or not img_exe:
        raise FileNotFoundError(
            "QEMU bulunamadı. Kurmak için: winget install --exact --id SoftwareFreedomConservancy.QEMU"
        )
    return Path(qemu_exe), Path(img_exe)


def _whpx_hypervisor_present() -> bool:
    """Queries WinHvPlatform.dll WHvGetCapability(HypervisorPresent) instead of checking registry alone."""
    try:
        import ctypes

        dll = ctypes.WinDLL("WinHvPlatform.dll")
        # WHvCapabilityCodeHypervisorPresent = 0x00000000
        present = ctypes.c_uint64(0)
        written = ctypes.c_uint32(0)
        hr = dll.WHvGetCapability(
            ctypes.c_uint32(0),
            ctypes.byref(present),
            ctypes.c_uint32(ctypes.sizeof(present)),
            ctypes.byref(written),
        )
        return hr == 0 and bool(present.value)
    except Exception:
        return False


def detect_accelerator(qemu_exe: Path) -> Tuple[str, List[str]]:
    """Auto-detect WHPX hardware acceleration on Windows (or KVM on Linux); fallback to multi-threaded TCG."""
    if os.name == "nt":
        try:
            if _whpx_hypervisor_present():
                proc = subprocess.run([str(qemu_exe), "-accel", "help"], capture_output=True, text=True, timeout=5)
                if "whpx" in (proc.stdout + proc.stderr).lower():
                    return "whpx", ["-accel", "whpx,kernel-irqchip=off", "-cpu", "max"]
        except Exception:
            pass
        return "tcg", ["-accel", "tcg,thread=multi", "-cpu", "qemu64"]
    else:
        if os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK):
            return "kvm", ["-accel", "kvm", "-cpu", "host"]
        return "tcg", ["-accel", "tcg,thread=multi", "-cpu", "qemu64"]


def get_or_create_token() -> str:
    if TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_hex(24)
    TOKEN_FILE.write_text(tok, encoding="utf-8")
    return tok


def verify_base_image_sha512() -> bool:
    if not BASE_IMAGE.exists():
        return False
    h = hashlib.sha512()
    with open(BASE_IMAGE, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower() == BASE_IMAGE_SHA512.lower()


# ---------------------------------------------------------------------------
# Cloud-Init NoCloud Seed Generator
# ---------------------------------------------------------------------------
def build_seed_iso(api_token: str) -> Path:
    if not DAEMON_SRC.exists():
        raise FileNotFoundError(f"Guest daemon source not found: {DAEMON_SRC}")

    daemon_gz_b64 = base64.b64encode(gzip.compress(DAEMON_SRC.read_bytes(), compresslevel=9)).decode("ascii")

    openrc_service = """#!/sbin/openrc-run
name="sodals-os"
description="SODALS-OS Isolated Daily Task & Web Automation Daemon"
supervisor="supervise-daemon"
command="/usr/bin/python3"
command_args="/opt/sodals_os/os_daemon.py"
command_user="svon:svon"
pidfile="/run/sodals-os.pid"
output_log="/home/svon/logs/daemon.out.log"
error_log="/home/svon/logs/daemon.err.log"
respawn_delay=3
respawn_max=0

depend() {
    need net localmount sodals-firewall
}
"""
    openrc_b64 = base64.b64encode(openrc_service.encode("utf-8")).decode("ascii")

    firewall_service = """#!/sbin/openrc-run
name="sodals-firewall"
description="SODALS-OS Strict IPv4/IPv6 Egress Firewall (Blocks Host RFC1918/ULA/Link-Local LAN)"

depend() {
    need net
    before sodals-os
}

start() {
    ebegin "Applying SODALS-OS IPv4/IPv6 egress firewall rules"
    if command -v iptables >/dev/null 2>&1; then
        iptables -F OUTPUT 2>/dev/null || true
        iptables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
        iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
        iptables -A OUTPUT -d 10.0.2.3 -p udp --dport 53 -j ACCEPT 2>/dev/null || true
        iptables -A OUTPUT -d 10.0.2.3 -p tcp --dport 53 -j ACCEPT 2>/dev/null || true
        iptables -A OUTPUT -d 10.0.2.2 -p udp --dport 67 -j ACCEPT 2>/dev/null || true
        iptables -A OUTPUT -d 10.0.0.0/8 -j REJECT 2>/dev/null || true
        iptables -A OUTPUT -d 172.16.0.0/12 -j REJECT 2>/dev/null || true
        iptables -A OUTPUT -d 192.168.0.0/16 -j REJECT 2>/dev/null || true
        iptables -A OUTPUT -d 169.254.0.0/16 -j REJECT 2>/dev/null || true
        iptables -A OUTPUT -d 100.64.0.0/10 -j REJECT 2>/dev/null || true
    fi
    if command -v ip6tables >/dev/null 2>&1; then
        ip6tables -F OUTPUT 2>/dev/null || true
        ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
        ip6tables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
        ip6tables -A OUTPUT -d fc00::/7 -j REJECT 2>/dev/null || true
        ip6tables -A OUTPUT -d fe80::/10 -j REJECT 2>/dev/null || true
    fi
    eend 0
}
"""
    fw_b64 = base64.b64encode(firewall_service.encode("utf-8")).decode("ascii")

    setup_script = """#!/bin/sh
set -e
echo "=== SODALS_OS_PROVISION_START ===" > /dev/ttyS0 2>/dev/null || true

# Create unprivileged user svon (no sudo, no root escalation)
if ! id svon >/dev/null 2>&1; then
    adduser -D -s /bin/sh svon
    passwd -l svon || true
fi

mkdir -p /opt/sodals_os /etc/sodals_os /home/svon/workspace /home/svon/notes/daily /home/svon/data /home/svon/logs
chown -R svon:svon /home/svon
chgrp svon /etc/sodals_os/api_token
chmod 0640 /etc/sodals_os/api_token
chmod 0755 /opt/sodals_os/os_daemon.py /etc/init.d/sodals-os /etc/init.d/sodals-firewall

echo "=== SODALS_OS_INSTALLING_PACKAGES ===" > /dev/ttyS0 2>/dev/null || true
apk update
apk add --no-cache python3 py3-beautifulsoup4 curl iptables ip6tables ca-certificates tzdata acpid chrony

ln -sf /usr/share/zoneinfo/Europe/Istanbul /etc/localtime || true
echo "Europe/Istanbul" > /etc/timezone

# Speed up subsequent boots (1s bootloader timeout + disable cloud-init metadata polling)
sed -i 's/^TIMEOUT 100/TIMEOUT 10/' /boot/extlinux.conf 2>/dev/null || true
mkdir -p /etc/cloud
touch /etc/cloud/cloud-init.disabled
for svc in cloud-init-local cloud-init cloud-config cloud-final; do
    rc-update del "$svc" boot 2>/dev/null || true
    rc-update del "$svc" default 2>/dev/null || true
done

cat <<'EOF' > /etc/local.d/99-sodals-boot-marker.start
#!/bin/sh
echo "=== SODALS_OS_BOOT_COMPLETE ===" > /dev/ttyS0 2>/dev/null || true
exit 0
EOF
chmod 0755 /etc/local.d/99-sodals-boot-marker.start

rc-update add acpid default || true
rc-service acpid start || true
rc-update add chronyd default || true
rc-service chronyd start || true

rc-update add local default || true
rc-update add sodals-firewall default
rc-service sodals-firewall start || true

rc-update add sodals-os default
sync
rc-service sodals-os restart
sync

echo "=== SODALS_OS_READY ===" > /dev/ttyS0 2>/dev/null || true
"""
    setup_b64 = base64.b64encode(setup_script.encode("utf-8")).decode("ascii")

    user_data = f"""#cloud-config
hostname: sodals-os
manage_etc_hosts: true
disable_root: true
ssh_pwauth: false
write_files:
  - path: /etc/sodals_os/api_token
    permissions: '0640'
    content: "{api_token}\\n"
  - path: /opt/sodals_os/os_daemon.py
    encoding: gz+b64
    permissions: '0755'
    content: "{daemon_gz_b64}"
  - path: /etc/init.d/sodals-os
    encoding: b64
    permissions: '0755'
    content: "{openrc_b64}"
  - path: /etc/init.d/sodals-firewall
    encoding: b64
    permissions: '0755'
    content: "{fw_b64}"
  - path: /opt/sodals_os/provision.sh
    encoding: b64
    permissions: '0755'
    content: "{setup_b64}"
runcmd:
  - [ /bin/sh, /opt/sodals_os/provision.sh ]
"""

    meta_data = (
        "instance-id: sodals-os-001\n"
        "local-hostname: sodals-os\n"
    )

    return build_nocloud_iso(
        {
            "meta-data": meta_data.encode("utf-8"),
            "user-data": user_data.encode("utf-8"),
        },
        output_path=CIDATA_ISO,
        volume_id="cidata",
    )


# ---------------------------------------------------------------------------
# Runtime HTTP / QMP Helpers
# ---------------------------------------------------------------------------
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_api_healthy(timeout: float = 2.0) -> bool:
    try:
        with _DIRECT_OPENER.open(f"http://{API_HOST}:{API_PORT}/health", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("ok"))
    except Exception:
        return False


def fetch_status(timeout: float = 4.0) -> Optional[Dict[str, Any]]:
    tok = get_or_create_token()
    req = urllib.request.Request(
        f"http://{API_HOST}:{API_PORT}/api/status",
        headers={"Authorization": f"Bearer {tok}"},
    )
    try:
        with _DIRECT_OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _read_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        raw = PID_FILE.read_text(encoding="utf-8").strip()
        if raw.startswith("{"):
            data = json.loads(raw)
            return int(data.get("pid", 0)) or None
        return int(raw)
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    payload = {"pid": int(pid), "exe": "qemu-system-x86_64", "started_at": time.time()}
    PID_FILE.write_text(json.dumps(payload), encoding="utf-8")


def _pid_is_qemu(pid: int) -> bool:
    """Verifies that `pid` is alive AND actually belongs to a QEMU process (prevents PID reuse kills)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = (proc.stdout or "").lower()
        return (f'"{pid}"' in out) and ("qemu-system" in out)
    try:
        os.kill(pid, 0)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="ignore").lower()
        return "qemu-system" in cmdline
    except Exception:
        return False


def _pid_is_alive(pid: int) -> bool:
    return _pid_is_qemu(pid)


def send_qmp_command(cmd_name: str, timeout: float = 4.0) -> bool:
    try:
        with socket.create_connection((API_HOST, QMP_PORT), timeout=timeout) as sock:
            sock.settimeout(timeout)
            f = sock.makefile("rwb", buffering=0)
            greeting = f.readline()
            if not greeting:
                return False
            f.write(b'{"execute":"qmp_capabilities"}\r\n')
            # Read until return for qmp_capabilities
            for _ in range(5):
                ln = f.readline().decode("utf-8", errors="ignore")
                if '"return"' in ln:
                    break
            f.write(json.dumps({"execute": cmd_name}).encode("utf-8") + b"\r\n")
            for _ in range(8):
                ln = f.readline().decode("utf-8", errors="ignore")
                if not ln:
                    break
                if '"return"' in ln:
                    return True
            return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# VM Lifecycle Commands
# ---------------------------------------------------------------------------
def start_vm(
    attach_seed: bool = False,
    wait_sec: int = 180,
    ram_mb: int = 1536,
    cpus: int = 4,
    foreground: bool = False,
) -> Dict[str, Any]:
    if is_api_healthy(timeout=1.5):
        st = fetch_status()
        return {"ok": True, "already_running": True, "status": st}

    if not VM_DISK.exists():
        raise FileNotFoundError("SODALS-OS sanal diski yok. Önce 'provision' çalıştırın.")

    qemu_exe, _ = find_qemu_binaries()
    accel_name, accel_flags = detect_accelerator(qemu_exe)
    share_dir = qemu_exe.parent / "share"

    disk_arg = str(VM_DISK).replace("\\", "/")
    console_arg = str(CONSOLE_LOG).replace("\\", "/")

    cmd = [
        str(qemu_exe),
        "-name",
        "SODALS-OS",
        "-machine",
        "q35",
        *accel_flags,
        "-m",
        f"{ram_mb}M",
        "-smp",
        str(cpus),
        "-display",
        "none",
    ]
    if share_dir.is_dir():
        cmd.extend(["-L", str(share_dir).replace("\\", "/")])
    cmd.extend(
        [
            "-drive",
            f"file={disk_arg},if=virtio,format=qcow2,cache=writeback",
            "-netdev",
            f"user,id=net0,hostfwd=tcp:{API_HOST}:{API_PORT}-:{API_PORT}",
            "-device",
            "virtio-net-pci,netdev=net0",
            "-device",
            "virtio-rng-pci",
            "-qmp",
            f"tcp:{API_HOST}:{QMP_PORT},server,nowait",
            "-serial",
            f"file:{console_arg}",
        ]
    )
    if attach_seed and CIDATA_ISO.exists():
        seed_arg = str(CIDATA_ISO).replace("\\", "/")
        cmd.extend(["-drive", f"file={seed_arg},media=cdrom,readonly=on"])

    CONSOLE_LOG.write_text("", encoding="utf-8")

    popen_kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    _write_pid(proc.pid)

    print(f"[SODALS-OS] VM başlatıldı (PID={proc.pid}, accel={accel_name}, RAM={ram_mb}MB, vCPU={cpus}). Hazır olması bekleniyor...", flush=True)
    deadline = time.time() + wait_sec
    seen_markers = set()
    while time.time() < deadline:
        if is_api_healthy(timeout=1.5):
            st = fetch_status()
            print(f"[SODALS-OS] HAZIR! Panel: http://{API_HOST}:{API_PORT} (Kernel: {(st or {}).get('kernel', '?')})", flush=True)
            if foreground:
                print("[SODALS-OS] Ön plan daemon modunda çalışıyor (durdurmak için Ctrl+C veya sodals_os.bat stop)...", flush=True)
                try:
                    proc.wait()
                except KeyboardInterrupt:
                    stop_vm()
            return {"ok": True, "pid": proc.pid, "accel": accel_name, "status": st}
        if not _pid_is_alive(proc.pid):
            tail = CONSOLE_LOG.read_text(encoding="utf-8", errors="replace")[-1200:] if CONSOLE_LOG.exists() else ""
            raise RuntimeError(f"QEMU beklenmedik şekilde sonlandı (PID={proc.pid}). Konsol:\n{tail}")
        # Print milestone markers from serial console once each
        try:
            txt = CONSOLE_LOG.read_text(encoding="utf-8", errors="replace")
            for marker in ("SODALS_OS_PROVISION_START", "SODALS_OS_INSTALLING_PACKAGES", "SODALS_OS_READY", "SODALS_OS_BOOT_COMPLETE"):
                if marker in txt and marker not in seen_markers:
                    seen_markers.add(marker)
                    print(f"[SODALS-OS] Durum: {marker}", flush=True)
        except Exception:
            pass
        time.sleep(2.0)

    tail = CONSOLE_LOG.read_text(encoding="utf-8", errors="replace")[-1500:] if CONSOLE_LOG.exists() else ""
    raise TimeoutError(f"SODALS-OS {wait_sec}s içinde yanıt vermedi. Son konsol çıktısı:\n{tail}")


def stop_vm(force_after_sec: int = 20) -> Dict[str, Any]:
    pid = _read_pid()
    qmp_sent = send_qmp_command("system_powerdown")
    if qmp_sent:
        print("[SODALS-OS] Temiz ACPI kapatma sinyali gönderildi...", flush=True)
        deadline = time.time() + force_after_sec
        while time.time() < deadline:
            if not is_api_healthy(timeout=1.0) and (pid is None or not _pid_is_alive(pid)):
                if PID_FILE.exists():
                    PID_FILE.unlink()
                print("[SODALS-OS] VM temiz şekilde kapandı.", flush=True)
                return {"ok": True, "graceful": True}
            time.sleep(1.0)

    # Fallback QMP quit or process kill
    send_qmp_command("quit")
    time.sleep(1.0)
    if pid and _pid_is_alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)
        else:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    if PID_FILE.exists():
        PID_FILE.unlink()
    print("[SODALS-OS] VM durduruldu.", flush=True)
    return {"ok": True, "graceful": False}


def provision_vm(disk_gb: int = 4, force: bool = False, skip_snapshot: bool = False) -> Dict[str, Any]:
    qemu_exe, img_exe = find_qemu_binaries()
    if not BASE_IMAGE.exists():
        raise FileNotFoundError(f"Alpine base imajı bulunamadı: {BASE_IMAGE}")
    print("[SODALS-OS] Alpine 3.21.10 SHA-512 bütünlüğü doğrulanıyor...", flush=True)
    if not verify_base_image_sha512():
        raise RuntimeError("Alpine base imaj SHA-512 doğrulaması başarısız!")
    print("[SODALS-OS] SHA-512 DOĞRULANDI.", flush=True)

    if VM_DISK.exists():
        if not force:
            print(f"[SODALS-OS] Mevcut sanal disk bulundu: {VM_DISK}. Başlatılıyor...", flush=True)
            return start_vm(attach_seed=False)
        stop_vm()
        VM_DISK.unlink()

    print(f"[SODALS-OS] {disk_gb} GB izole sanal disk oluşturuluyor ({VM_DISK.name})...", flush=True)
    shutil.copy2(BASE_IMAGE, VM_DISK)
    subprocess.run([str(img_exe), "resize", str(VM_DISK), f"{disk_gb}G"], check=True, capture_output=True)

    tok = get_or_create_token()
    build_seed_iso(tok)
    print(f"[SODALS-OS] Cloud-Init seed ISO oluşturuldu ({CIDATA_ISO.name}). İlk kurulum başlatılıyor...", flush=True)

    res = start_vm(attach_seed=True, wait_sec=300)
    if not skip_snapshot:
        # Allow provision.sh final sync to settle before clean ACPI powerdown
        time.sleep(4.0)
        print(f"[SODALS-OS] Temiz başlangıç anlık görüntüsü ('{SNAPSHOT_NAME}') alınıyor...", flush=True)
        stop_vm(force_after_sec=30)
        # Remove plaintext cidata.iso containing api_token once provisioning is complete
        if CIDATA_ISO.exists():
            try:
                CIDATA_ISO.unlink()
            except OSError:
                pass
        subprocess.run([str(img_exe), "snapshot", "-c", SNAPSHOT_NAME, str(VM_DISK)], check=True, capture_output=True)
        print(f"[SODALS-OS] Snapshot '{SNAPSHOT_NAME}' kaydedildi. VM tekrar başlatılıyor (seed ISO çıkarılmış)...", flush=True)
        res = start_vm(attach_seed=False, wait_sec=180)
    return res


def reset_to_snapshot(snapshot_name: str = SNAPSHOT_NAME) -> Dict[str, Any]:
    _, img_exe = find_qemu_binaries()
    if not VM_DISK.exists():
        raise FileNotFoundError("Sanal disk yok.")
    stop_vm()
    print(f"[SODALS-OS] '{snapshot_name}' anlık görüntüsüne geri dönülüyor...", flush=True)
    subprocess.run([str(img_exe), "snapshot", "-a", snapshot_name, str(VM_DISK)], check=True, capture_output=True)
    return start_vm(attach_seed=False, wait_sec=180)


def _request_bootstrap_code(timeout: float = 4.0) -> Optional[str]:
    tok = get_or_create_token()
    req = urllib.request.Request(
        f"http://{API_HOST}:{API_PORT}/api/auth/bootstrap",
        data=b"{}",
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "X-SODALS-CSRF": "1",
        },
        method="POST",
    )
    try:
        with _DIRECT_OPENER.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok") and data.get("code"):
                return str(data["code"])
    except Exception:
        pass
    return None


def open_dashboard() -> None:
    if not is_api_healthy(timeout=1.5):
        start_vm(attach_seed=False)
    code = _request_bootstrap_code()
    if code:
        url = f"http://{API_HOST}:{API_PORT}/?code={code}"
    else:
        url = f"http://{API_HOST}:{API_PORT}/"
    print(f"[SODALS-OS] Kontrol Paneli açılıyor: http://{API_HOST}:{API_PORT}", flush=True)
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="SODALS-OS Isolated Micro-VM Manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prov = sub.add_parser("provision", help="Provision fresh SODALS-OS Alpine micro-VM")
    p_prov.add_argument("--disk-gb", type=int, default=4)
    p_prov.add_argument("--force", action="store_true")
    p_prov.add_argument("--skip-snapshot", action="store_true")

    p_start = sub.add_parser("start", help="Start SODALS-OS micro-VM")
    p_start.add_argument("--ram-mb", type=int, default=1536)
    p_start.add_argument("--cpus", type=int, default=4)
    p_start.add_argument("--wait-sec", type=int, default=180)
    p_start.add_argument("--foreground", action="store_true", help="Keep launcher attached to QEMU process")

    sub.add_parser("stop", help="Gracefully stop SODALS-OS micro-VM")
    sub.add_parser("status", help="Show SODALS-OS status and isolation audit")
    sub.add_parser("reset", help="Rewind SODALS-OS disk to golden_clean snapshot")
    sub.add_parser("dashboard", help="Open SODALS-OS Live Web Control Panel")

    args = parser.parse_args()
    if args.cmd == "provision":
        res = provision_vm(disk_gb=args.disk_gb, force=args.force, skip_snapshot=args.skip_snapshot)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "start":
        res = start_vm(ram_mb=args.ram_mb, cpus=args.cpus, wait_sec=args.wait_sec, foreground=args.foreground)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "stop":
        res = stop_vm()
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "status":
        running = is_api_healthy(timeout=2.0)
        st = fetch_status() if running else None
        qemu_exe = None
        accel = "unknown"
        try:
            qemu_exe, _ = find_qemu_binaries()
            accel, _ = detect_accelerator(qemu_exe)
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "running": running,
                    "pid": _read_pid(),
                    "accelerator": accel,
                    "dashboard_url": f"http://{API_HOST}:{API_PORT}",
                    "disk_exists": VM_DISK.exists(),
                    "status": st,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.cmd == "reset":
        res = reset_to_snapshot()
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "dashboard":
        open_dashboard()


if __name__ == "__main__":
    main()
