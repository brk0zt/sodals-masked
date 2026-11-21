"""
Pure-Python ISO-9660 + Joliet (Level 3) NoCloud `cidata` Seed Image Builder.
Requires zero external binaries (no genisoimage/mkisofs/oscdimg needed on Windows).
Produces a standards-compliant ISO-9660 + Joliet image that Linux `mount -t iso9660`
and `cloud-init` NoCloud datasource read natively with exact lowercase filenames
(`meta-data`, `user-data`, `network-config`).
"""

from __future__ import annotations

import datetime
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple

SECTOR_SIZE = 2048


def _both16(v: int) -> bytes:
    return struct.pack("<H", v) + struct.pack(">H", v)


def _both32(v: int) -> bytes:
    return struct.pack("<I", v) + struct.pack(">I", v)


def _dec_datetime(dt: datetime.datetime) -> bytes:
    return dt.strftime("%Y%m%d%H%M%S00").encode("ascii") + b"\x00"


def _dir_datetime(dt: datetime.datetime) -> bytes:
    return bytes([dt.year - 1900, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0])


def _make_dir_record(extent_lba: int, data_len: int, flags: int, file_id: bytes, dt: datetime.datetime) -> bytes:
    rec_len = 33 + len(file_id)
    pad = b"\x00" if (rec_len % 2 == 1) else b""
    rec_len += len(pad)
    hdr = bytearray()
    hdr.append(rec_len)
    hdr.append(0)  # Extended attribute record length
    hdr.extend(_both32(extent_lba))
    hdr.extend(_both32(data_len))
    hdr.extend(_dir_datetime(dt))
    hdr.append(flags)
    hdr.append(0)  # File unit size
    hdr.append(0)  # Interleave gap size
    hdr.extend(_both16(1))  # Volume sequence number
    hdr.append(len(file_id))
    hdr.extend(file_id)
    hdr.extend(pad)
    return bytes(hdr)


def _pack_directory_sector(records: List[bytes]) -> bytes:
    buf = bytearray()
    for rec in records:
        rem = SECTOR_SIZE - (len(buf) % SECTOR_SIZE)
        if len(rec) > rem:
            buf.extend(b"\x00" * rem)
        buf.extend(rec)
    if len(buf) == 0:
        buf.extend(b"\x00" * SECTOR_SIZE)
    else:
        rem = len(buf) % SECTOR_SIZE
        if rem:
            buf.extend(b"\x00" * (SECTOR_SIZE - rem))
    return bytes(buf)


def _make_path_table(root_lba: int, big_endian: bool) -> bytes:
    endian = ">" if big_endian else "<"
    # Root directory entry: name len=1, ext_attr=0, lba, parent=1, name=0x00, pad=0x00
    entry = struct.pack(f"{endian}BBIH", 1, 0, root_lba, 1) + b"\x00\x00"
    return entry


def _pvd_filename(name: str) -> bytes:
    # ISO-9660 uppercase d-chars fallback (Linux Joliet overrides with exact UTF-16BE)
    cleaned = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name.upper())
    return (cleaned + ";1").encode("ascii")


def _joliet_filename(name: str) -> bytes:
    if len(name) > 64:
        raise ValueError(f"Joliet filename exceeds 64-character limit ({len(name)} chars): '{name}'")
    return (name + ";1").encode("utf-16-be")


def _make_volume_descriptor(
    vd_type: int,
    volume_id_bytes: bytes,
    volume_space_sectors: int,
    path_table_size: int,
    l_path_lba: int,
    m_path_lba: int,
    root_record: bytes,
    dt: datetime.datetime,
    escape_seq: bytes = b"",
) -> bytes:
    vd = bytearray(SECTOR_SIZE)
    vd[0] = vd_type
    vd[1:6] = b"CD001"
    vd[6] = 1  # Version
    vd[7] = 0
    # System identifier (32 bytes)
    sys_id = b"LINUX".ljust(32, b" ") if vd_type == 1 else "LINUX".encode("utf-16-be").ljust(32, b"\x00")
    vd[8:40] = sys_id[:32]
    # Volume identifier (32 bytes)
    vd[40:72] = volume_id_bytes[:32]
    # Volume space size
    vd[80:88] = _both32(volume_space_sectors)
    # Escape sequences (Joliet SVD only)
    if escape_seq:
        vd[88 : 88 + len(escape_seq)] = escape_seq
    # Volume set size & sequence number
    vd[120:124] = _both16(1)
    vd[124:128] = _both16(1)
    # Logical block size
    vd[128:132] = _both16(SECTOR_SIZE)
    # Path table size & locations
    vd[132:140] = _both32(path_table_size)
    vd[140:144] = struct.pack("<I", l_path_lba)
    vd[148:152] = struct.pack(">I", m_path_lba)
    # Root directory record (34 bytes at offset 156..190)
    assert len(root_record) == 34, f"Expected 34-byte root record, got {len(root_record)}"
    vd[156:190] = root_record
    # Pad text metadata fields
    pad_char = b" " if vd_type == 1 else b"\x00"
    vd[190:813] = pad_char * (813 - 190)
    # Timestamps
    ts = _dec_datetime(dt)
    vd[813:830] = ts
    vd[830:847] = ts
    vd[847:864] = b"0" * 16 + b"\x00"
    vd[864:881] = ts
    vd[881] = 1  # File structure version
    return bytes(vd)


def build_nocloud_iso(files: Dict[str, bytes], output_path: Path, volume_id: str = "cidata") -> Path:
    """
    Writes an ISO-9660 + Joliet Level 3 NoCloud image containing the given root files.
    """
    dt = datetime.datetime.now(datetime.timezone.utc)
    ordered_names = sorted(files.keys())

    # Fixed descriptor & path table sectors:
    # 0..15: System area
    # 16: PVD, 17: SVD (Joliet), 18: Volume Descriptor Set Terminator
    # 19..22: Path tables (PVD L/M, SVD L/M)
    # 23+: Dynamic multi-sector PVD Root Dir, SVD Root Dir, then File extents
    pvd_pt_l_lba = 19
    pvd_pt_m_lba = 20
    svd_pt_l_lba = 21
    svd_pt_m_lba = 22
    pvd_root_lba = 23

    # Pre-calculate directory sizes so multi-sector root directories work for any file count
    dummy_pvd = [
        _make_dir_record(0, 0, 0x02, b"\x00", dt),
        _make_dir_record(0, 0, 0x02, b"\x01", dt),
    ] + [_make_dir_record(0, len(files[n]), 0x00, _pvd_filename(n), dt) for n in ordered_names]
    pvd_root_bytes_len = len(_pack_directory_sector(dummy_pvd))
    pvd_root_sectors = pvd_root_bytes_len // SECTOR_SIZE

    svd_root_lba = pvd_root_lba + pvd_root_sectors
    dummy_svd = [
        _make_dir_record(0, 0, 0x02, b"\x00", dt),
        _make_dir_record(0, 0, 0x02, b"\x01", dt),
    ] + [_make_dir_record(0, len(files[n]), 0x00, _joliet_filename(n), dt) for n in ordered_names]
    svd_root_bytes_len = len(_pack_directory_sector(dummy_svd))
    svd_root_sectors = svd_root_bytes_len // SECTOR_SIZE

    next_lba = svd_root_lba + svd_root_sectors

    file_extents: List[Tuple[str, int, bytes]] = []
    for name in ordered_names:
        data = files[name]
        sectors_needed = max(1, math.ceil(len(data) / SECTOR_SIZE)) if len(data) > 0 else 0
        lba = next_lba if len(data) > 0 else 0
        file_extents.append((name, lba, data))
        next_lba += sectors_needed

    total_sectors = next_lba

    # Build PVD root directory with real LBAs and multi-sector byte length
    pvd_records = [
        _make_dir_record(pvd_root_lba, pvd_root_bytes_len, 0x02, b"\x00", dt),
        _make_dir_record(pvd_root_lba, pvd_root_bytes_len, 0x02, b"\x01", dt),
    ]
    for name, lba, data in file_extents:
        pvd_records.append(_make_dir_record(lba, len(data), 0x00, _pvd_filename(name), dt))
    pvd_root_sec = _pack_directory_sector(pvd_records)

    # Build SVD (Joliet) root directory with real LBAs and multi-sector byte length
    svd_records = [
        _make_dir_record(svd_root_lba, svd_root_bytes_len, 0x02, b"\x00", dt),
        _make_dir_record(svd_root_lba, svd_root_bytes_len, 0x02, b"\x01", dt),
    ]
    for name, lba, data in file_extents:
        svd_records.append(_make_dir_record(lba, len(data), 0x00, _joliet_filename(name), dt))
    svd_root_sec = _pack_directory_sector(svd_records)

    pvd_pt_l = _make_path_table(pvd_root_lba, big_endian=False)
    pvd_pt_m = _make_path_table(pvd_root_lba, big_endian=True)
    svd_pt_l = _make_path_table(svd_root_lba, big_endian=False)
    svd_pt_m = _make_path_table(svd_root_lba, big_endian=True)

    pvd_vol_id = volume_id.encode("ascii").ljust(32, b" ")[:32]
    # Joliet UCS-2BE padded with 0x0020 spaces
    j_vid = volume_id.encode("utf-16-be")
    while len(j_vid) < 32:
        j_vid += b"\x00 "
    svd_vol_id = j_vid[:32]

    pvd = _make_volume_descriptor(
        vd_type=1,
        volume_id_bytes=pvd_vol_id,
        volume_space_sectors=total_sectors,
        path_table_size=len(pvd_pt_l),
        l_path_lba=pvd_pt_l_lba,
        m_path_lba=pvd_pt_m_lba,
        root_record=_make_dir_record(pvd_root_lba, len(pvd_root_sec), 0x02, b"\x00", dt),
        dt=dt,
    )
    svd = _make_volume_descriptor(
        vd_type=2,
        volume_id_bytes=svd_vol_id,
        volume_space_sectors=total_sectors,
        path_table_size=len(svd_pt_l),
        l_path_lba=svd_pt_l_lba,
        m_path_lba=svd_pt_m_lba,
        root_record=_make_dir_record(svd_root_lba, len(svd_root_sec), 0x02, b"\x00", dt),
        dt=dt,
        escape_seq=b"%/E",
    )
    terminator = bytearray(SECTOR_SIZE)
    terminator[0] = 255
    terminator[1:6] = b"CD001"
    terminator[6] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"\x00" * (16 * SECTOR_SIZE))
        f.write(pvd)
        f.write(svd)
        f.write(terminator)
        f.write(pvd_pt_l.ljust(SECTOR_SIZE, b"\x00"))
        f.write(pvd_pt_m.ljust(SECTOR_SIZE, b"\x00"))
        f.write(svd_pt_l.ljust(SECTOR_SIZE, b"\x00"))
        f.write(svd_pt_m.ljust(SECTOR_SIZE, b"\x00"))
        f.write(pvd_root_sec)
        f.write(svd_root_sec)
        for _name, _lba, data in file_extents:
            if not data:
                continue
            f.write(data)
            rem = len(data) % SECTOR_SIZE
            if rem:
                f.write(b"\x00" * (SECTOR_SIZE - rem))
    return output_path
