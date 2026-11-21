#!/usr/bin/env python3
"""
SODALS-OS Guest Daemon (v1.0)
=============================
Runs strictly inside the isolated Alpine Linux QEMU micro-VM as unprivileged
user `svon`. Provides:
  1. Headless Stateful Web Browser (open, click, search, RSS, save-as-note)
     with strict SSRF / private-LAN egress blocking.
  2. Persistent Notes Vault (Markdown files + SQLite search index).
  3. Autonomous Daily Task Scheduler (daily:HH:MM, every:Nm/Nh, manual)
     supporting web_digest, web_watch, shell_job, and note_briefing.
  4. Isolated File Workspace (/home/svon/workspace) & Sandboxed Shell.
  5. Live Web Control Panel (http://127.0.0.1:11440) with audit trail.
"""

from __future__ import annotations

import datetime
import hashlib
import html
import http.client
import http.cookiejar
import ipaddress
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import resource  # POSIX only
except ImportError:
    resource = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------
HOME_DIR = Path(os.environ.get("SODALS_OS_HOME", "/home/svon")).resolve()
WORKSPACE_DIR = HOME_DIR / "workspace"
NOTES_DIR = HOME_DIR / "notes"
DATA_DIR = HOME_DIR / "data"
LOGS_DIR = HOME_DIR / "logs"
DB_PATH = DATA_DIR / "sodals_os.db"
TOKEN_FILE = Path(os.environ.get("SODALS_OS_TOKEN_FILE", "/etc/sodals_os/api_token"))
BIND_HOST = os.environ.get("SODALS_OS_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("SODALS_OS_PORT", "11440"))
MAX_WEB_BYTES = 1_500_000
START_TIME = time.time()

for _d in (WORKSPACE_DIR, NOTES_DIR, NOTES_DIR / "daily", DATA_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_api_token() -> str:
    env_tok = os.environ.get("SODALS_OS_TOKEN", "").strip()
    if env_tok:
        return env_tok
    if TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text(encoding="utf-8", errors="ignore").strip()
        if tok:
            return tok
    fallback = DATA_DIR / "api_token.txt"
    if fallback.exists():
        tok = fallback.read_text(encoding="utf-8", errors="ignore").strip()
        if tok:
            return tok
    tok = secrets.token_hex(24)
    fallback.write_text(tok, encoding="utf-8")
    os.chmod(fallback, 0o600)
    return tok


API_TOKEN = _load_api_token()


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_now() -> datetime.datetime:
    return datetime.datetime.now()


def slugify(text: str, max_len: int = 48) -> str:
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    cleaned = (text or "note").translate(tr_map).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return (cleaned or "note")[:max_len]


# ---------------------------------------------------------------------------
# SQLite Storage (Notes, Daily Tasks, Task Runs, Activity Audit)
# ---------------------------------------------------------------------------
class StateDB:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=15000;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    folder TEXT NOT NULL DEFAULT 'general',
                    tags TEXT NOT NULL DEFAULT '',
                    file_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_run_at TEXT,
                    last_status TEXT,
                    last_summary TEXT,
                    last_hash TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    note_id TEXT
                );
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    ok INTEGER NOT NULL DEFAULT 1
                );
                """
            )

    def log_activity(self, category: str, action: str, detail: str, ok: bool = True) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO activity_log (ts, category, action, detail, ok) VALUES (?, ?, ?, ?, ?)",
                (utc_now_iso(), category, action, (detail or "")[:1000], 1 if ok else 0),
            )
            conn.execute(
                "DELETE FROM activity_log WHERE id NOT IN (SELECT id FROM activity_log ORDER BY id DESC LIMIT 300)"
            )

    def recent_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (max(1, min(200, int(limit))),)
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Notes ---
    @staticmethod
    def _validate_note_id(raw_id: Optional[str]) -> Optional[str]:
        if raw_id is None:
            return None
        cand = str(raw_id).strip()
        if not cand:
            return None
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", cand):
            raise ValueError(
                f"Geçersiz note_id ('{cand}'): yalnızca harf, rakam, '_' ve '-' (en fazla 80 karakter) kabul edilir."
            )
        return cand

    def save_note(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        folder: str = "general",
        note_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_folder = re.sub(r"[^a-zA-Z0-9_-]", "", folder or "general") or "general"
        folder_dir = (NOTES_DIR / safe_folder).resolve()
        notes_root = NOTES_DIR.resolve()
        if folder_dir != notes_root and notes_root not in folder_dir.parents:
            raise PermissionError("Not klasörü NOTES_DIR dışına çıkamaz.")
        folder_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        tag_str = ",".join(sorted({t.strip().lower() for t in (tags or []) if t and t.strip()}))
        validated_id = self._validate_note_id(note_id)
        nid = validated_id or f"{local_now().strftime('%Y%m%d_%H%M%S')}_{slugify(title, 30)}_{secrets.token_hex(2)}"
        file_path = (folder_dir / f"{nid}.md").resolve()
        if notes_root not in file_path.parents:
            raise PermissionError(f"Not dosya yolu NOTES_DIR dışına çıkamaz: {nid}")
        md_doc = (
            f"---\n"
            f"id: {nid}\n"
            f"title: {title}\n"
            f"folder: {safe_folder}\n"
            f"tags: [{tag_str}]\n"
            f"updated_at: {now}\n"
            f"---\n\n"
            f"{content}\n"
        )
        file_path.write_text(md_doc, encoding="utf-8")
        rel_path = str(file_path.relative_to(HOME_DIR))

        with self._lock, self._conn() as conn:
            existing = conn.execute("SELECT created_at FROM notes WHERE id = ?", (nid,)).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT INTO notes (id, title, folder, tags, file_path, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    folder=excluded.folder,
                    tags=excluded.tags,
                    file_path=excluded.file_path,
                    content=excluded.content,
                    updated_at=excluded.updated_at
                """,
                (nid, title, safe_folder, tag_str, rel_path, content, created_at, now),
            )
        self.log_activity("notes", "save_note", f"[{safe_folder}] {title} ({nid})", True)
        return self.get_note(nid) or {"id": nid, "title": title, "file_path": rel_path}

    def list_notes(self, query: str = "", tag: str = "", folder: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT id, title, folder, tags, file_path, substr(content, 1, 240) AS preview, created_at, updated_at FROM notes WHERE 1=1"
        params: List[Any] = []
        if folder:
            sql += " AND folder = ?"
            params.append(folder)
        if tag:
            sql += " AND (',' || tags || ',') LIKE ?"
            params.append(f"%,{tag.lower().strip()},%")
        if query:
            sql += " AND (lower(title) LIKE ? OR lower(content) LIKE ? OR lower(tags) LIKE ?)"
            q = f"%{query.lower().strip()}%"
            params.extend([q, q, q])
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(200, int(limit))))
        with self._lock, self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
            return dict(row) if row else None

    def delete_note(self, note_id: str) -> bool:
        note = self.get_note(note_id)
        if not note:
            return False
        notes_root = NOTES_DIR.resolve()
        fp = (HOME_DIR / note["file_path"]).resolve()
        try:
            if notes_root in fp.parents and fp.exists():
                fp.unlink()
        except Exception:
            pass
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.log_activity("notes", "delete_note", f"Deleted note {note_id}", True)
        return True

    # --- Tasks ---
    VALID_TASK_TYPES = {"web_digest", "web_watch", "note_briefing", "shell_job"}

    @staticmethod
    def validate_schedule(schedule: str) -> str:
        s = (schedule or "").strip().lower()
        if s == "manual":
            return s
        m_daily = re.fullmatch(r"daily:(\d{1,2}):(\d{2})", s)
        if m_daily:
            hh, mm = int(m_daily.group(1)), int(m_daily.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"daily:{hh:02d}:{mm:02d}"
            raise ValueError(f"Geçersiz daily saat/dakika değeri: '{schedule}' (00:00..23:59 olmalı).")
        m_every = re.fullmatch(r"every:(\d+)(m|h)", s)
        if m_every:
            val = int(m_every.group(1))
            if val >= 1:
                return s
            raise ValueError(f"Geçersiz periyot değeri: '{schedule}' (>= 1 olmalı).")
        raise ValueError(
            f"Geçersiz zamanlama biçimi: '{schedule}'. Beklenen: 'manual', 'daily:HH:MM' veya 'every:Nm'/'every:Nh'."
        )

    def create_task(
        self,
        name: str,
        task_type: str,
        schedule: str,
        config: Dict[str, Any],
        enabled: bool = True,
    ) -> Dict[str, Any]:
        ttype = (task_type or "").strip()
        if ttype not in self.VALID_TASK_TYPES:
            raise ValueError(f"Geçersiz görev türü: '{ttype}'. İzin verilenler: {sorted(self.VALID_TASK_TYPES)}")
        clean_sched = self.validate_schedule(schedule)
        tid = f"task_{slugify(name, 24)}_{secrets.token_hex(3)}"
        now = utc_now_iso()
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO tasks (id, name, task_type, schedule, config_json, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (tid, name, ttype, clean_sched, json.dumps(config, ensure_ascii=False), 1 if enabled else 0, now),
            )
        self.log_activity("tasks", "create_task", f"{name} ({ttype}, {clean_sched})", True)
        return self.get_task(tid) or {"id": tid}

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["config"] = json.loads(d.pop("config_json"))
                except Exception:
                    d["config"] = {}
                d["enabled"] = bool(d["enabled"])
                out.append(d)
            return out

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            r = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if not r:
                return None
            d = dict(r)
            try:
                d["config"] = json.loads(d.pop("config_json"))
            except Exception:
                d["config"] = {}
            d["enabled"] = bool(d["enabled"])
            return d

    def set_task_enabled(self, task_id: str, enabled: bool) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.execute("UPDATE tasks SET enabled = ? WHERE id = ?", (1 if enabled else 0, task_id))
            return cur.rowcount > 0

    def delete_task(self, task_id: str) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            if cur.rowcount > 0:
                self.log_activity("tasks", "delete_task", f"Deleted task {task_id}", True)
                return True
            return False

    def record_task_run(
        self,
        task_id: str,
        task_name: str,
        duration_ms: int,
        status: str,
        summary: str,
        note_id: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> None:
        now = utc_now_iso()
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO task_runs (task_id, task_name, started_at, duration_ms, status, summary, note_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, task_name, now, duration_ms, status, (summary or "")[:2000], note_id),
            )
            conn.execute(
                """
                UPDATE tasks SET last_run_at = ?, last_status = ?, last_summary = ?,
                                 last_hash = COALESCE(?, last_hash)
                WHERE id = ?
                """,
                (now, status, (summary or "")[:500], content_hash, task_id),
            )
            conn.execute(
                "DELETE FROM task_runs WHERE id NOT IN (SELECT id FROM task_runs ORDER BY id DESC LIMIT 200)"
            )

    def recent_runs(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM task_runs ORDER BY id DESC LIMIT ?", (max(1, min(200, int(limit))),)
            ).fetchall()
            return [dict(r) for r in rows]

    def counts(self) -> Dict[str, int]:
        with self._lock, self._conn() as conn:
            notes_c = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            tasks_c = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            runs_c = conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
            return {"notes": notes_c, "tasks": tasks_c, "runs": runs_c}


DB = StateDB(DB_PATH)


# ---------------------------------------------------------------------------
# Headless Stateful Web Browser (SSRF-Safe)
# ---------------------------------------------------------------------------
class _FallbackTextParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.in_title = False
        self.skip_depth = 0
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []
        self.links: List[Dict[str, Any]] = []
        self._cur_href: Optional[str] = None
        self._cur_link_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript", "svg"):
            self.skip_depth += 1
        elif t == "title":
            self.in_title = True
        elif t in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "article", "section"):
            self.text_parts.append("\n")
        elif t == "a":
            href = dict(attrs).get("href")
            if href:
                self._cur_href = urllib.parse.urljoin(self.base_url, href)
                self._cur_link_text = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript", "svg") and self.skip_depth > 0:
            self.skip_depth -= 1
        elif t == "title":
            self.in_title = False
        elif t == "a" and self._cur_href:
            label = " ".join("".join(self._cur_link_text).split()).strip()
            if label and self._cur_href.startswith(("http://", "https://")):
                self.links.append({"text": label[:140], "href": self._cur_href})
            self._cur_href = None
            self._cur_link_text = []

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self.skip_depth == 0:
            self.text_parts.append(data)
        if self._cur_href is not None:
            self._cur_link_text.append(data)


def _resolve_and_validate_public_url(url: str) -> Tuple[bool, str, List[str]]:
    parsed = urllib.parse.urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https"):
        return False, "Yalnızca http:// ve https:// URL'leri desteklenir.", []
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "Geçersiz URL: hostname eksik.", []
    if host in ("localhost", "ip6-localhost", "broadcasthost") or host.endswith(".local"):
        return False, f"Güvenlik duvarı: yerel adres engellendi ({host}).", []
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return False, f"DNS çözümlenemedi ({host}): {exc}", []
    resolved_ips: List[str] = []
    for info in infos:
        ip_str = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            check_ip = ip_obj.ipv4_mapped if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped else ip_obj
            if (
                (not check_ip.is_global)
                or check_ip.is_private
                or check_ip.is_loopback
                or check_ip.is_link_local
                or check_ip.is_multicast
                or check_ip.is_reserved
                or check_ip.is_unspecified
            ):
                return False, f"Güvenlik duvarı: özel/yerel ağ IP adresi engellendi ({host} -> {ip_str}).", []
            if ip_str not in resolved_ips:
                resolved_ips.append(ip_str)
        except ValueError:
            return False, f"Geçersiz IP adresi: {ip_str}", []
    if not resolved_ips:
        return False, f"DNS geçerli genel IP döndürmedi ({host}).", []
    return True, "", resolved_ips


def _validate_public_url(url: str) -> Tuple[bool, str]:
    ok, err, _ = _resolve_and_validate_public_url(url)
    return ok, err


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connects strictly to the pre-validated public IP to eliminate DNS rebinding."""

    def __init__(self, host: str, port: Optional[int] = None, timeout: float = 20, **kwargs: Any):
        self._pinned_ip = kwargs.pop("pinned_ip", None)
        super().__init__(host, port=port, timeout=timeout, **kwargs)

    def connect(self) -> None:
        target = self._pinned_ip or self.host
        self.sock = socket.create_connection((target, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects strictly to the pre-validated public IP while preserving original TLS SNI hostname."""

    def __init__(self, host: str, port: Optional[int] = None, timeout: float = 20, **kwargs: Any):
        self._pinned_ip = kwargs.pop("pinned_ip", None)
        super().__init__(host, port=port, timeout=timeout, **kwargs)

    def connect(self) -> None:
        target = self._pinned_ip or self.host
        raw_sock = socket.create_connection((target, self.port), self.timeout, self.source_address)
        server_hostname = self.host
        if self._tunnel_host:
            self.sock = raw_sock
            self._tunnel()
            server_hostname = self._tunnel_host
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=server_hostname)


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request):
        ok, err, ips = _resolve_and_validate_public_url(req.full_url)
        if not ok:
            raise urllib.error.URLError(err)
        pinned = ips[0]
        return self.do_open(lambda host, **kw: _PinnedHTTPConnection(host, pinned_ip=pinned, **kw), req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request):
        ok, err, ips = _resolve_and_validate_public_url(req.full_url)
        if not ok:
            raise urllib.error.URLError(err)
        pinned = ips[0]
        ctx = ssl.create_default_context()
        return self.do_open(
            lambda host, **kw: _PinnedHTTPSConnection(host, pinned_ip=pinned, context=ctx, **kw),
            req,
        )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates EVERY redirect target BEFORE connecting, blocking 301/302 SSRF to localhost/LAN."""

    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        abs_url = urllib.parse.urljoin(req.full_url, newurl)
        ok, err, _ = _resolve_and_validate_public_url(abs_url)
        if not ok:
            raise urllib.error.URLError(f"Yönlendirme engellendi: {err}")
        return super().redirect_request(req, fp, code, msg, headers, abs_url)


def _parse_html_document(raw_html: str, final_url: str, max_chars: int = 10000) -> Tuple[str, str, List[Dict[str, Any]]]:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw_html, "html.parser")
        for bad in soup(["script", "style", "noscript", "svg", "iframe"]):
            bad.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else final_url
        links: List[Dict[str, Any]] = []
        seen_href = set()
        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(final_url, a["href"])
            if not href.startswith(("http://", "https://")) or href in seen_href:
                continue
            label = " ".join(a.get_text(" ", strip=True).split())
            if not label:
                continue
            seen_href.add(href)
            links.append({"index": len(links) + 1, "text": label[:140], "href": href})
            if len(links) >= 40:
                break
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        text = "\n".join(lines)[:max_chars]
        return title[:200], text, links

    parser = _FallbackTextParser(final_url)
    parser.feed(raw_html)
    title = " ".join("".join(parser.title_parts).split()).strip() or final_url
    lines = [ln.strip() for ln in "".join(parser.text_parts).splitlines() if ln.strip()]
    text = "\n".join(lines)[:max_chars]
    dedup_links: List[Dict[str, Any]] = []
    seen = set()
    for lnk in parser.links:
        if lnk["href"] not in seen:
            seen.add(lnk["href"])
            dedup_links.append({"index": len(dedup_links) + 1, "text": lnk["text"], "href": lnk["href"]})
            if len(dedup_links) >= 40:
                break
    return title[:200], text, dedup_links


class MiniBrowser:
    MAX_TABS = 20

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _PinnedHTTPHandler(),
            _PinnedHTTPSHandler(),
            _SafeRedirectHandler(),
            urllib.request.HTTPCookieProcessor(self.jar),
        )
        self.tabs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.history: List[Dict[str, Any]] = []

    def _put_tab(self, tab_id: str, state: Dict[str, Any]) -> None:
        if tab_id in self.tabs:
            self.tabs.move_to_end(tab_id)
        self.tabs[tab_id] = state
        while len(self.tabs) > self.MAX_TABS:
            self.tabs.popitem(last=False)

    def open_url(self, url: str, tab_id: str = "main", max_chars: int = 10000, timeout: int = 20) -> Dict[str, Any]:
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        ok, err = _validate_public_url(url)
        if not ok:
            DB.log_activity("web", "open_blocked", f"{url} -> {err}", False)
            return {"ok": False, "error": err, "url": url}

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) SODALS-OS/1.0 Chrome/128.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )
        t0 = time.time()
        try:
            with self.opener.open(req, timeout=timeout) as resp:
                final_url = resp.geturl()
                ok2, err2 = _validate_public_url(final_url)
                if not ok2:
                    return {"ok": False, "error": f"Yönlendirme engellendi: {err2}", "url": final_url}
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read(MAX_WEB_BYTES)
                status_code = getattr(resp, "status", 200)
        except Exception as exc:
            DB.log_activity("web", "open_error", f"{url} -> {exc}", False)
            return {"ok": False, "error": f"Sayfa açılamadı: {exc}", "url": url}

        charset = "utf-8"
        m = re.search(r"charset=([a-zA-Z0-9_-]+)", content_type, re.IGNORECASE)
        if m:
            charset = m.group(1)
        try:
            decoded = raw.decode(charset, errors="replace")
        except LookupError:
            decoded = raw.decode("utf-8", errors="replace")

        if "xml" in content_type.lower() or decoded.lstrip().startswith("<?xml"):
            title, text, links = self._parse_feed_or_xml(decoded, final_url, max_chars)
        elif "json" in content_type.lower():
            title = final_url
            try:
                text = json.dumps(json.loads(decoded), ensure_ascii=False, indent=2)[:max_chars]
            except Exception:
                text = decoded[:max_chars]
            links = []
        elif "text/plain" in content_type.lower():
            title = final_url
            text = decoded[:max_chars]
            links = []
        else:
            title, text, links = _parse_html_document(decoded, final_url, max_chars)

        elapsed_ms = int((time.time() - t0) * 1000)
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        tab_state = {
            "ok": True,
            "tab_id": tab_id,
            "url": final_url,
            "status_code": status_code,
            "content_type": content_type,
            "title": title,
            "text": text,
            "links": links,
            "content_hash": digest,
            "fetched_at": utc_now_iso(),
            "elapsed_ms": elapsed_ms,
        }
        with self._lock:
            self._put_tab(tab_id, tab_state)
            self.history.insert(
                0,
                {
                    "ts": tab_state["fetched_at"],
                    "tab_id": tab_id,
                    "url": final_url,
                    "title": title,
                    "status_code": status_code,
                    "elapsed_ms": elapsed_ms,
                },
            )
            self.history = self.history[:80]
        DB.log_activity("web", "open", f"[{status_code}] {title} ({final_url})", True)
        return tab_state

    def _parse_feed_or_xml(self, xml_text: str, final_url: str, max_chars: int) -> Tuple[str, str, List[Dict[str, Any]]]:
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_text)
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if items:
                title_el = root.find(".//channel/title") or root.find(".//{http://www.w3.org/2005/Atom}title")
                feed_title = (title_el.text or final_url).strip() if title_el is not None else final_url
                lines = [f"# {feed_title}"]
                links: List[Dict[str, Any]] = []
                for idx, it in enumerate(items[:25], start=1):
                    t_el = it.find("title")
                    if t_el is None:
                        t_el = it.find("{http://www.w3.org/2005/Atom}title")
                    l_el = it.find("link")
                    if l_el is None:
                        l_el = it.find("{http://www.w3.org/2005/Atom}link")
                    d_el = it.find("description")
                    if d_el is None:
                        d_el = it.find("{http://www.w3.org/2005/Atom}summary")
                    t_str = html.unescape((t_el.text or "").strip()) if t_el is not None else f"Öğe {idx}"
                    href = ""
                    if l_el is not None:
                        href = (l_el.text or "").strip() or (l_el.attrib.get("href") or "").strip()
                    desc = re.sub(r"<[^>]+>", " ", html.unescape((d_el.text or "").strip())) if d_el is not None else ""
                    desc = " ".join(desc.split())[:220]
                    lines.append(f"{idx}. **{t_str}**" + (f"\n   {desc}" if desc else "") + (f"\n   URL: {href}" if href else ""))
                    if href.startswith(("http://", "https://")):
                        links.append({"index": len(links) + 1, "text": t_str[:140], "href": href})
                return feed_title, "\n".join(lines)[:max_chars], links
        except Exception:
            pass
        return _parse_html_document(xml_text, final_url, max_chars)

    def click(self, tab_id: str = "main", index: Optional[int] = None, link_text: str = "") -> Dict[str, Any]:
        with self._lock:
            tab = self.tabs.get(tab_id)
        if not tab:
            return {"ok": False, "error": f"'{tab_id}' sekmesinde açık sayfa yok."}
        links = tab.get("links") or []
        target_href = None
        if index is not None:
            for lnk in links:
                if int(lnk.get("index", -1)) == int(index):
                    target_href = lnk["href"]
                    break
        elif link_text:
            needle = link_text.lower().strip()
            for lnk in links:
                if needle in lnk["text"].lower() or needle in lnk["href"].lower():
                    target_href = lnk["href"]
                    break
        if not target_href:
            return {"ok": False, "error": "Eşleşen bağlantı bulunamadı."}
        return self.open_url(target_href, tab_id=tab_id)

    def search(self, query: str, tab_id: str = "main", limit: int = 8) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"ok": False, "error": "Arama sorgusu boş olamaz."}
        results: List[Dict[str, str]] = []

        # Primary: DuckDuckGo HTML endpoint
        ddg_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
        try:
            req = urllib.request.Request(
                ddg_url,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) SODALS-OS/1.0"},
            )
            with self.opener.open(req, timeout=15) as resp:
                raw_html = resp.read(MAX_WEB_BYTES).decode("utf-8", errors="replace")
            if BeautifulSoup is not None:
                soup = BeautifulSoup(raw_html, "html.parser")
                for res in soup.select(".result")[:limit]:
                    a = res.select_one("a.result__a")
                    snip = res.select_one(".result__snippet")
                    if not a:
                        continue
                    href = a.get("href", "")
                    if "uddg=" in href:
                        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        href = qs.get("uddg", [href])[0]
                    if href.startswith(("http://", "https://")):
                        results.append(
                            {
                                "title": a.get_text(" ", strip=True),
                                "url": href,
                                "snippet": snip.get_text(" ", strip=True) if snip else "",
                            }
                        )
        except Exception:
            results = []

        # Fallback: Wikipedia OpenSearch API (works reliably even if DDG throttles)
        if not results:
            for lang in ("tr", "en"):
                wiki_url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                    {"action": "opensearch", "search": q, "limit": limit, "namespace": 0, "format": "json"}
                )
                try:
                    req = urllib.request.Request(wiki_url, headers={"User-Agent": "SODALS-OS/1.0"})
                    with self.opener.open(req, timeout=12) as resp:
                        data = json.loads(resp.read(200_000).decode("utf-8", errors="replace"))
                    if isinstance(data, list) and len(data) >= 4:
                        for t, d, u in zip(data[1], data[2], data[3]):
                            results.append({"title": t, "url": u, "snippet": d})
                    if results:
                        break
                except Exception:
                    continue

        links = [{"index": i + 1, "text": r["title"][:140], "href": r["url"]} for i, r in enumerate(results)]
        lines = [f"# Arama Sonuçları: {q}"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. **{r['title']}**\n   {r['snippet']}\n   URL: {r['url']}")
        text = "\n".join(lines) if results else f"'{q}' için sonuç bulunamadı."

        tab_state = {
            "ok": True,
            "tab_id": tab_id,
            "url": ddg_url,
            "status_code": 200,
            "content_type": "text/search",
            "title": f"Arama: {q}",
            "text": text,
            "results": results,
            "links": links,
            "fetched_at": utc_now_iso(),
        }
        with self._lock:
            self._put_tab(tab_id, tab_state)
            self.history.insert(
                0, {"ts": tab_state["fetched_at"], "tab_id": tab_id, "url": ddg_url, "title": f"Arama: {q}", "status_code": 200, "elapsed_ms": 0}
            )
            self.history = self.history[:80]
        DB.log_activity("web", "search", f"Query '{q}' -> {len(results)} results", True)
        return tab_state

    def state_summary(self) -> Dict[str, Any]:
        with self._lock:
            tabs_out = {}
            for tid, st in self.tabs.items():
                tabs_out[tid] = {
                    "tab_id": tid,
                    "url": st.get("url"),
                    "title": st.get("title"),
                    "status_code": st.get("status_code"),
                    "fetched_at": st.get("fetched_at"),
                    "text_preview": (st.get("text") or "")[:3000],
                    "links": (st.get("links") or [])[:25],
                }
            return {"tabs": tabs_out, "history": self.history[:30]}


BROWSER = MiniBrowser()


# ---------------------------------------------------------------------------
# Isolated Workspace Files & Sandboxed Shell
# ---------------------------------------------------------------------------
def _safe_ws_path(rel: str) -> Path:
    clean = (rel or ".").replace("\\", "/").lstrip("/")
    candidate = (WORKSPACE_DIR / clean).resolve()
    if candidate != WORKSPACE_DIR and WORKSPACE_DIR not in candidate.parents:
        raise PermissionError(f"Çalışma alanı dışına erişim engellendi: {rel}")
    return candidate


def list_workspace_files(rel: str = ".") -> Dict[str, Any]:
    target = _safe_ws_path(rel)
    if not target.exists():
        return {"ok": False, "error": f"Dizin bulunamadı: {rel}"}
    entries = []
    for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:200]:
        st = item.stat()
        entries.append(
            {
                "name": item.name,
                "rel_path": str(item.relative_to(WORKSPACE_DIR)),
                "is_dir": item.is_dir(),
                "size": st.st_size,
                "mtime": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    return {"ok": True, "path": str(target.relative_to(WORKSPACE_DIR)), "entries": entries}


def read_workspace_file(rel: str, max_chars: int = 20000) -> Dict[str, Any]:
    target = _safe_ws_path(rel)
    if not target.is_file():
        return {"ok": False, "error": f"Dosya bulunamadı: {rel}"}
    content = target.read_text(encoding="utf-8", errors="replace")[:max_chars]
    return {"ok": True, "path": str(target.relative_to(WORKSPACE_DIR)), "content": content}


def write_workspace_file(rel: str, content: str) -> Dict[str, Any]:
    target = _safe_ws_path(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    DB.log_activity("files", "write_file", str(target.relative_to(WORKSPACE_DIR)), True)
    return {"ok": True, "path": str(target.relative_to(WORKSPACE_DIR)), "bytes": len(content.encode("utf-8"))}


MAX_SHELL_CAPTURE_BYTES = 256_000


def _apply_posix_shell_rlimits() -> None:
    if resource is None:
        return
    limits = [
        (getattr(resource, "RLIMIT_CPU", None), (180, 180)),
        (getattr(resource, "RLIMIT_FSIZE", None), (64 * 1024 * 1024, 64 * 1024 * 1024)),
        (getattr(resource, "RLIMIT_AS", None), (768 * 1024 * 1024, 768 * 1024 * 1024)),
        (getattr(resource, "RLIMIT_NPROC", None), (64, 64)),
    ]
    for res_id, (soft, hard) in limits:
        if res_id is None:
            continue
        try:
            cur_soft, cur_hard = resource.getrlimit(res_id)
            eff_hard = min(cur_hard, hard) if cur_hard > 0 else hard
            eff_soft = min(soft, eff_hard)
            resource.setrlimit(res_id, (eff_soft, eff_hard))
        except Exception:
            pass


def _read_stream_bounded(stream: Any, max_bytes: int) -> bytes:
    buf = bytearray()
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        if len(buf) < max_bytes:
            need = max_bytes - len(buf)
            buf.extend(chunk[:need])
    return bytes(buf)


def run_sandboxed_shell(command: str, timeout_sec: int = 45) -> Dict[str, Any]:
    cmd = (command or "").strip()
    if not cmd:
        return {"ok": False, "error": "Komut boş olamaz."}
    timeout_sec = max(1, min(180, int(timeout_sec)))
    t0 = time.time()
    proc = None
    try:
        popen_kw: Dict[str, Any] = {
            "cwd": str(WORKSPACE_DIR),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(HOME_DIR),
                "USER": os.environ.get("USER", "svon"),
                "LANG": "C.UTF-8",
            },
        }
        if os.name == "posix":
            popen_kw["start_new_session"] = True
            popen_kw["preexec_fn"] = _apply_posix_shell_rlimits
            shell_argv = ["/bin/sh", "-c", cmd]
        else:
            shell_argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]

        proc = subprocess.Popen(shell_argv, **popen_kw)
        out_box: List[bytes] = [b""]
        err_box: List[bytes] = [b""]

        def _drain(st: Any, box: List[bytes]) -> None:
            try:
                box[0] = _read_stream_bounded(st, MAX_SHELL_CAPTURE_BYTES)
            except Exception:
                pass

        t_out = threading.Thread(target=_drain, args=(proc.stdout, out_box), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, err_box), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
            else:
                proc.kill()
            t_out.join(timeout=1.0)
            t_err.join(timeout=1.0)
            DB.log_activity("shell", "timeout", cmd[:140], False)
            return {"ok": False, "error": f"Zaman aşımı ({timeout_sec}s)", "exit_code": -1}

        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        stdout_s = out_box[0].decode("utf-8", errors="replace")
        stderr_s = err_box[0].decode("utf-8", errors="replace")
        elapsed_ms = int((time.time() - t0) * 1000)
        ok = proc.returncode == 0
        DB.log_activity("shell", "exec", f"[{proc.returncode}] {cmd[:140]}", ok)
        return {
            "ok": ok,
            "exit_code": proc.returncode,
            "stdout": stdout_s[:12000],
            "stderr": stderr_s[:6000],
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        DB.log_activity("shell", "error", f"{cmd[:100]} -> {exc}", False)
        return {"ok": False, "error": str(exc), "exit_code": -1}


# ---------------------------------------------------------------------------
# Autonomous Daily Task Scheduler Engine
# ---------------------------------------------------------------------------
def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    tid = task["id"]
    name = task["name"]
    ttype = task["task_type"]
    cfg = task.get("config") or {}
    note_id: Optional[str] = None
    content_hash: Optional[str] = None

    try:
        if ttype == "web_digest":
            urls = cfg.get("urls") or ([] if not cfg.get("url") else [cfg["url"]])
            query = cfg.get("search_query", "")
            sections: List[str] = [f"# Günlük Özet: {name}", f"Tarih: {local_now().strftime('%Y-%m-%d %H:%M')}\n"]
            for u in urls[:6]:
                res = BROWSER.open_url(str(u), tab_id=f"task_{tid}", max_chars=3500)
                if res.get("ok"):
                    sections.append(f"## {res.get('title', u)}\nKaynak: {res.get('url', u)}\n\n{res.get('text', '')[:2500]}\n")
                else:
                    sections.append(f"## Hata: {u}\n{res.get('error', 'Bilinmeyen hata')}\n")
            if query:
                sres = BROWSER.search(str(query), tab_id=f"task_{tid}", limit=6)
                sections.append(f"## Web Araması: {query}\n\n{sres.get('text', '')[:2500]}\n")
            body = "\n".join(sections)
            today_slug = f"{local_now().strftime('%Y-%m-%d')}_{slugify(name, 28)}"
            saved = DB.save_note(
                title=f"[{local_now().strftime('%Y-%m-%d')}] {name}",
                content=body,
                tags=["daily", "otonom", slugify(name, 20)],
                folder="daily",
                note_id=today_slug,
            )
            note_id = saved["id"]
            summary = f"Web özeti oluşturuldu ve '{note_id}' notuna kaydedildi."
            status = "ok"

        elif ttype == "web_watch":
            url = str(cfg.get("url", ""))
            keyword = str(cfg.get("keyword", "")).strip()
            res = BROWSER.open_url(url, tab_id=f"watch_{tid}", max_chars=8000)
            if not res.get("ok"):
                raise RuntimeError(res.get("error", "Web sayfası okunamadı"))
            content_hash = res.get("content_hash")
            changed = bool(task.get("last_hash") and task["last_hash"] != content_hash)
            kw_found = bool(keyword and keyword.lower() in (res.get("text") or "").lower())
            if changed or kw_found or not task.get("last_hash"):
                reason = []
                if not task.get("last_hash"):
                    reason.append("İlk tarama")
                if changed:
                    reason.append("Sayfa içeriği değişti")
                if kw_found:
                    reason.append(f"Anahtar kelime bulundu: '{keyword}'")
                saved = DB.save_note(
                    title=f"[İzleme] {name} ({', '.join(reason)})",
                    content=f"URL: {res['url']}\nSebep: {', '.join(reason)}\nHash: {content_hash}\n\n{res['text'][:4000]}",
                    tags=["watch", "daily", slugify(name, 20)],
                    folder="daily",
                )
                note_id = saved["id"]
                summary = f"{', '.join(reason)} -> Not kaydedildi ({note_id})."
            else:
                summary = f"Değişiklik yok (hash={content_hash})."
            status = "ok"

        elif ttype == "shell_job":
            cmd = str(cfg.get("command", ""))
            sh_res = run_sandboxed_shell(cmd, timeout_sec=int(cfg.get("timeout_sec", 60)))
            status = "ok" if sh_res.get("ok") else "error"
            out_preview = ((sh_res.get("stdout") or "") + "\n" + (sh_res.get("stderr") or "")).strip()
            if cfg.get("save_note", True):
                saved = DB.save_note(
                    title=f"[Otonom Görev] {name} ({local_now().strftime('%Y-%m-%d %H:%M')})",
                    content=f"Komut: `{cmd}`\nÇıkış Kodu: `{sh_res.get('exit_code')}`\n\n```text\n{out_preview[:6000]}\n```",
                    tags=["shell-job", "daily"],
                    folder="daily",
                )
                note_id = saved["id"]
            summary = f"exit={sh_res.get('exit_code')} | {out_preview[:240]}"

        elif ttype == "note_briefing":
            template = str(cfg.get("template") or "- [ ] Günlük öncelikleri gözden geçir\n- [ ] Notları düzenle")
            today = local_now().strftime("%Y-%m-%d")
            saved = DB.save_note(
                title=f"[{today}] {name}",
                content=f"# {name} ({today})\n\n{template}",
                tags=["briefing", "daily"],
                folder="daily",
                note_id=f"{today}_{slugify(name, 28)}",
            )
            note_id = saved["id"]
            summary = f"Günlük plan notu oluşturuldu ({note_id})."
            status = "ok"

        else:
            raise ValueError(f"Desteklenmeyen görev türü: {ttype}")

    except Exception as exc:
        status = "error"
        summary = f"Görev hatası: {exc}"

    dur_ms = int((time.time() - t0) * 1000)
    DB.record_task_run(tid, name, dur_ms, status, summary, note_id=note_id, content_hash=content_hash)
    DB.log_activity("scheduler", f"run:{status}", f"{name}: {summary[:160]}", status == "ok")
    return {"ok": status == "ok", "task_id": tid, "status": status, "duration_ms": dur_ms, "summary": summary, "note_id": note_id}


def _task_is_due(task: Dict[str, Any], now_dt: datetime.datetime) -> bool:
    if not task.get("enabled"):
        return False
    sched = (task.get("schedule") or "").strip().lower()
    if not sched or sched == "manual":
        return False

    last_run_str = task.get("last_run_at")
    last_dt: Optional[datetime.datetime] = None
    if last_run_str:
        try:
            last_dt = datetime.datetime.strptime(last_run_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        except Exception:
            last_dt = None

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    m_daily = re.fullmatch(r"daily:(\d{1,2}):(\d{2})", sched)
    if m_daily:
        hh, mm = int(m_daily.group(1)), int(m_daily.group(2))
        target_today = now_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if now_dt < target_today:
            return False
        if last_dt is not None:
            last_local = last_dt.astimezone()
            if last_local.date() == now_dt.date():
                return False
        return True

    m_every = re.fullmatch(r"every:(\d+)(m|h)", sched)
    if m_every:
        val = int(m_every.group(1))
        unit = m_every.group(2)
        interval_sec = val * (3600 if unit == "h" else 60)
        if last_dt is None:
            return True
        return (now_utc - last_dt).total_seconds() >= interval_sec

    return False


class SchedulerThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="sodals-os-scheduler")
        self.stop_event = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="sodals-task")
        self._running_ids: Set[str] = set()
        self._running_lock = threading.Lock()

    def _run_guarded(self, task: Dict[str, Any]) -> None:
        tid = task["id"]
        try:
            execute_task(task)
        finally:
            with self._running_lock:
                self._running_ids.discard(tid)

    def run(self) -> None:
        try:
            while not self.stop_event.wait(15.0):
                try:
                    now_dt = local_now()
                    for t in DB.list_tasks():
                        if _task_is_due(t, now_dt):
                            tid = t["id"]
                            with self._running_lock:
                                if tid in self._running_ids:
                                    continue
                                self._running_ids.add(tid)
                            self._pool.submit(self._run_guarded, t)
                except Exception as exc:
                    DB.log_activity("scheduler", "loop_error", str(exc), False)
        finally:
            self._pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Isolation & System Telemetry
# ---------------------------------------------------------------------------
def collect_system_status() -> Dict[str, Any]:
    mem_total_mb = 0
    mem_avail_mb = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_mb = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                mem_avail_mb = int(line.split()[1]) // 1024
    except Exception:
        pass

    du = shutil.disk_usage(str(HOME_DIR))
    host_mounts: List[str] = []
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                fstype = parts[2].lower()
                if fstype in ("9p", "virtiofs", "cifs", "vboxsf", "drvfs", "smbfs"):
                    host_mounts.append(f"{parts[1]} ({fstype})")
    except Exception:
        pass

    return {
        "ok": True,
        "os_name": "SODALS-OS v1.0 (Alpine Linux Micro-VM)",
        "hostname": socket.gethostname(),
        "kernel": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "user": os.environ.get("USER", "svon"),
        "uid": os.getuid() if hasattr(os, "getuid") else -1,
        "uptime_sec": int(time.time() - START_TIME),
        "mem_total_mb": mem_total_mb,
        "mem_avail_mb": mem_avail_mb,
        "disk_free_mb": int(du.free // (1024 * 1024)),
        "isolation": {
            "host_filesystem_mounts": host_mounts,
            "host_c_drive_accessible": os.path.exists("/mnt/c") or bool(host_mounts),
            "unprivileged_user": (os.getuid() != 0) if hasattr(os, "getuid") else True,
            "ssrf_private_lan_guard": True,
        },
        "counts": DB.counts(),
        "browser_tabs": len(BROWSER.tabs),
        "server_time": local_now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Live Web Control Panel HTML (Self-Contained Dark UI)
# ---------------------------------------------------------------------------
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SODALS-OS — İzole Mini İşletim Sistemi Kontrol Paneli</title>
<style>
  :root{--bg:#0b1020;--panel:#131a2e;--panel2:#1a2440;--border:#263356;--text:#e6ecf8;--muted:#8fa1c7;--accent:#5b8cff;--ok:#2ecc71;--warn:#f1c40f;--err:#ff5c77;}
  *{box-sizing:border-box;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;}
  body{margin:0;background:var(--bg);color:var(--text);min-height:100vh;}
  header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;background:var(--panel);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10;}
  .brand{display:flex;align-items:center;gap:12px;font-weight:700;font-size:18px;}
  .badge{padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;background:#103624;color:var(--ok);border:1px solid #1e6c45;}
  .badge.warn{background:#3a2f10;color:var(--warn);border-color:#6d5818;}
  nav{display:flex;gap:8px;padding:12px 24px;background:var(--panel);border-bottom:1px solid var(--border);flex-wrap:wrap;}
  nav button{background:var(--panel2);color:var(--text);border:1px solid var(--border);padding:8px 14px;border-radius:8px;cursor:pointer;font-weight:600;}
  nav button.active{background:var(--accent);border-color:var(--accent);color:#fff;}
  main{max-width:1280px;margin:20px auto;padding:0 20px 60px;}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:18px;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px;}
  .card h3{margin:0 0 8px;font-size:14px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}
  .big{font-size:24px;font-weight:700;}
  .tab{display:none;}
  .tab.active{display:block;}
  input,textarea,select{width:100%;background:#0d1426;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 11px;margin:5px 0 10px;font-size:14px;}
  button.btn{background:var(--accent);color:#fff;border:none;padding:9px 15px;border-radius:8px;cursor:pointer;font-weight:600;}
  button.btn.sec{background:var(--panel2);border:1px solid var(--border);}
  button.btn.del{background:#471925;color:#ff93a6;border:1px solid #75273b;}
  .row{display:flex;gap:10px;align-items:center;}
  pre{background:#090e1c;border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;max-height:420px;white-space:pre-wrap;font-family:Consolas,monospace;font-size:13px;line-height:1.45;}
  .item{border:1px solid var(--border);border-radius:10px;padding:12px;margin-bottom:10px;background:var(--panel2);}
  .muted{color:var(--muted);font-size:12px;}
  .link-btn{display:inline-block;margin:3px 6px 3px 0;padding:4px 8px;background:#162344;border:1px solid var(--border);border-radius:6px;color:#9cc0ff;font-size:12px;cursor:pointer;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left;}
  th{color:var(--muted);}
</style>
</head>
<body>
<header>
  <div class="brand">🖥️ SODALS-OS <span class="badge" id="isoBadge">İzole VM Kontrol Ediliyor...</span></div>
  <div class="muted" id="clockInfo">Bağlanıyor...</div>
</header>
<nav>
  <button class="active" onclick="showTab('overview',this)">Genel Bakış</button>
  <button onclick="showTab('browser',this)">Canlı Tarayıcı</button>
  <button onclick="showTab('notes',this)">Not Defteri</button>
  <button onclick="showTab('tasks',this)">Otonom Günlük Görevler</button>
  <button onclick="showTab('shell',this)">Dosyalar & Terminal</button>
  <button onclick="showTab('activity',this)">Canlı Etkinlik Günlüğü</button>
</nav>
<main>
  <section id="tab-overview" class="tab active">
    <div class="grid">
      <div class="card"><h3>İşletim Sistemi</h3><div class="big" id="stKernel">-</div><div class="muted" id="stUser">-</div></div>
      <div class="card"><h3>VM Bellek & Disk</h3><div class="big" id="stMem">-</div><div class="muted" id="stDisk">-</div></div>
      <div class="card"><h3>İzolasyon Kalkanı</h3><div class="big" id="stIso">-</div><div class="muted" id="stIsoDetail">-</div></div>
      <div class="card"><h3>Notlar / Görevler</h3><div class="big" id="stCounts">-</div><div class="muted" id="stUptime">-</div></div>
    </div>
    <div class="card">
      <h3>Son Otonom Etkinlikler</h3>
      <div id="overviewActivity">Yükleniyor...</div>
    </div>
  </section>

  <section id="tab-browser" class="tab">
    <div class="card" style="margin-bottom:14px;">
      <h3>Mini-OS İzole Web Tarayıcısı</h3>
      <div class="row">
        <input id="webUrl" placeholder="https://example.com veya web araması için sorgu yaz..." style="flex:1;margin:0;">
        <button class="btn" onclick="webOpen()">Sayfayı Aç</button>
        <button class="btn sec" onclick="webSearch()">Web'de Ara</button>
        <button class="btn sec" onclick="webSaveNote()">Not Olarak Kaydet</button>
      </div>
    </div>
    <div class="card">
      <h3 id="webTitle">Açık Sayfa Yok</h3>
      <div class="muted" id="webMeta"></div>
      <div id="webLinks" style="margin:10px 0;"></div>
      <pre id="webBody">Henüz bir sayfa açılmadı.</pre>
    </div>
  </section>

  <section id="tab-notes" class="tab">
    <div class="grid" style="grid-template-columns:1fr 1.4fr;">
      <div class="card">
        <h3>Yeni Not Kaydet</h3>
        <input id="noteTitle" placeholder="Not başlığı">
        <input id="noteTags" placeholder="Etiketler (virgülle: daily, araştırma)">
        <textarea id="noteContent" rows="8" placeholder="Markdown not içeriği..."></textarea>
        <button class="btn" onclick="saveNote()">Notu Kaydet</button>
      </div>
      <div class="card">
        <h3>Kayıtlı Notlar</h3>
        <input id="noteSearch" placeholder="Notlarda ara..." oninput="loadNotes()">
        <div id="notesList">Yükleniyor...</div>
      </div>
    </div>
  </section>

  <section id="tab-tasks" class="tab">
    <div class="grid" style="grid-template-columns:1fr 1.4fr;">
      <div class="card">
        <h3>Yeni Otonom Günlük Görev</h3>
        <input id="taskName" placeholder="Örn: Sabah Teknoloji Özeti">
        <select id="taskType">
          <option value="web_digest">Web / RSS Günlük Özeti (web_digest)</option>
          <option value="web_watch">Web Sayfası Değişiklik İzleme (web_watch)</option>
          <option value="note_briefing">Günlük Plan / Checklist Notu (note_briefing)</option>
          <option value="shell_job">İzole Linux Script / Shell Görevi (shell_job)</option>
        </select>
        <input id="taskSchedule" value="daily:08:30" placeholder="daily:08:30 | every:60m | manual">
        <textarea id="taskConfig" rows="5" placeholder='{"urls": ["https://news.ycombinator.com/rss"]}'></textarea>
        <button class="btn" onclick="createTask()">Görevi Zamanla</button>
      </div>
      <div class="card">
        <h3>Zamanlanmış Görevler</h3>
        <div id="tasksList">Yükleniyor...</div>
        <h3 style="margin-top:18px;">Son Çalıştırma Geçmişi</h3>
        <div id="runsList">Yükleniyor...</div>
      </div>
    </div>
  </section>

  <section id="tab-shell" class="tab">
    <div class="card" style="margin-bottom:14px;">
      <h3>İzole Linux Terminali (/home/svon/workspace — Host C:\ Erişimi Yok)</h3>
      <div class="row">
        <input id="shellCmd" placeholder="uname -a && ls -la" style="flex:1;margin:0;" onkeydown="if(event.key==='Enter')runShell()">
        <button class="btn" onclick="runShell()">Çalıştır</button>
      </div>
      <pre id="shellOut" style="margin-top:10px;">Hazır.</pre>
    </div>
    <div class="card">
      <h3>Workspace Dosyaları</h3>
      <div id="filesList">Yükleniyor...</div>
    </div>
  </section>

  <section id="tab-activity" class="tab">
    <div class="card">
      <h3>Gerçek Zamanlı Denetim Günlüğü</h3>
      <div id="activityTable">Yükleniyor...</div>
    </div>
  </section>
</main>
<script>
async function api(path, opts={}){
  const hdrs = Object.assign({'Content-Type':'application/json','X-SODALS-CSRF':'1'}, opts.headers||{});
  const r = await fetch(path, Object.assign({}, opts, {headers:hdrs}));
  return await r.json();
}
function showTab(id, btn){
  document.querySelectorAll('.tab').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  if(btn) btn.classList.add('active');
  refreshAll();
}
function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

document.addEventListener('click', (ev)=>{
  const btn = ev.target && ev.target.closest ? ev.target.closest('[data-action]') : null;
  if(!btn) return;
  const act = btn.getAttribute('data-action');
  const id = btn.getAttribute('data-id') || '';
  if(act === 'web-click') webClick(Number(id));
  else if(act === 'view-note') viewNote(id);
  else if(act === 'del-note') delNote(id);
  else if(act === 'run-task') runTask(id);
  else if(act === 'del-task') delTask(id);
});

async function loadStatus(){
  const st = await api('/api/status');
  if(!st.ok) return;
  document.getElementById('stKernel').textContent = st.kernel;
  document.getElementById('stUser').textContent = `Host: ${st.hostname} | Kullanıcı: ${st.user} (uid=${st.uid})`;
  document.getElementById('stMem').textContent = `${st.mem_avail_mb} / ${st.mem_total_mb} MB`;
  document.getElementById('stDisk').textContent = `Boş Disk: ${st.disk_free_mb} MB`;
  const safe = !st.isolation.host_c_drive_accessible && st.isolation.unprivileged_user;
  document.getElementById('stIso').textContent = safe ? 'TAM İZOLE ✓' : 'UYARI';
  document.getElementById('stIsoDetail').textContent = `Host Mount: ${st.isolation.host_filesystem_mounts.length} | LAN Engel: Aktif`;
  document.getElementById('isoBadge').textContent = safe ? 'İzole Linux VM ✓ (C:\\ Kapalı)' : 'İzolasyon Uyarısı';
  document.getElementById('stCounts').textContent = `${st.counts.notes} Not / ${st.counts.tasks} Görev`;
  document.getElementById('stUptime').textContent = `Uptime: ${st.uptime_sec}s | Çalıştırma: ${st.counts.runs}`;
  document.getElementById('clockInfo').textContent = `VM Saati: ${st.server_time}`;
}

async function loadWeb(){
  const st = await api('/api/web/state');
  const main = st.tabs && st.tabs.main;
  if(!main) return;
  document.getElementById('webTitle').textContent = main.title || main.url;
  document.getElementById('webMeta').textContent = `${main.url} | HTTP ${main.status_code} | ${main.fetched_at}`;
  document.getElementById('webBody').textContent = main.text_preview || '';
  document.getElementById('webLinks').innerHTML = (main.links||[]).slice(0,18).map(l=>
    `<span class="link-btn" data-action="web-click" data-id="${Number(l.index)}">[${Number(l.index)}] ${esc(l.text.slice(0,48))}</span>`
  ).join('');
}
async function webOpen(){
  const url = document.getElementById('webUrl').value.trim();
  if(!url) return;
  document.getElementById('webBody').textContent = 'Yükleniyor...';
  const r = await api('/api/web/open',{method:'POST',body:JSON.stringify({url})});
  if(!r.ok) document.getElementById('webBody').textContent = 'HATA: '+r.error;
  await loadWeb();
}
async function webSearch(){
  const query = document.getElementById('webUrl').value.trim();
  if(!query) return;
  document.getElementById('webBody').textContent = 'Aranıyor...';
  await api('/api/web/search',{method:'POST',body:JSON.stringify({query})});
  await loadWeb();
}
async function webClick(idx){
  document.getElementById('webBody').textContent = 'Bağlantı açılıyor...';
  await api('/api/web/click',{method:'POST',body:JSON.stringify({index:idx})});
  await loadWeb();
}
async function webSaveNote(){
  const r = await api('/api/web/save_note',{method:'POST',body:JSON.stringify({tab_id:'main'})});
  alert(r.ok ? 'Not kaydedildi: '+r.note.title : 'Hata: '+r.error);
  loadNotes();
}

async function loadNotes(){
  const q = encodeURIComponent(document.getElementById('noteSearch').value.trim());
  const r = await api('/api/notes?q='+q);
  const el = document.getElementById('notesList');
  if(!r.notes || !r.notes.length){ el.innerHTML='<div class="muted">Henüz not yok.</div>'; return; }
  el.innerHTML = r.notes.map(n=>`
    <div class="item">
      <div class="row" style="justify-content:space-between;">
        <strong>${esc(n.title)}</strong>
        <div>
          <button class="btn sec" style="padding:4px 8px;font-size:12px;" data-action="view-note" data-id="${esc(n.id)}">Oku</button>
          <button class="btn del" style="padding:4px 8px;font-size:12px;" data-action="del-note" data-id="${esc(n.id)}">Sil</button>
        </div>
      </div>
      <div class="muted">[${esc(n.folder)}] ${esc(n.tags)} · ${esc(n.updated_at)}</div>
      <div style="margin-top:6px;font-size:13px;">${esc(n.preview)}</div>
    </div>
  `).join('');
}
async function saveNote(){
  const title = document.getElementById('noteTitle').value.trim() || 'Başlıksız Not';
  const tags = document.getElementById('noteTags').value.split(',').map(x=>x.trim()).filter(Boolean);
  const content = document.getElementById('noteContent').value;
  await api('/api/notes',{method:'POST',body:JSON.stringify({title,tags,content})});
  document.getElementById('noteTitle').value=''; document.getElementById('noteContent').value='';
  loadNotes(); loadStatus();
}
async function viewNote(id){
  const r = await api('/api/notes/'+encodeURIComponent(id));
  if(r.ok) alert(r.note.title + '\n\n' + r.note.content);
}
async function delNote(id){
  if(!confirm('Bu not silinsin mi?')) return;
  await api('/api/notes/'+encodeURIComponent(id),{method:'DELETE'});
  loadNotes(); loadStatus();
}

async function loadTasks(){
  const r = await api('/api/tasks');
  const el = document.getElementById('tasksList');
  if(!r.tasks || !r.tasks.length){ el.innerHTML='<div class="muted">Henüz zamanlanmış görev yok.</div>'; }
  else {
    el.innerHTML = r.tasks.map(t=>`
      <div class="item">
        <div class="row" style="justify-content:space-between;">
          <strong>${esc(t.name)}</strong>
          <div>
            <button class="btn" style="padding:4px 9px;font-size:12px;" data-action="run-task" data-id="${esc(t.id)}">Şimdi Çalıştır</button>
            <button class="btn del" style="padding:4px 9px;font-size:12px;" data-action="del-task" data-id="${esc(t.id)}">Sil</button>
          </div>
        </div>
        <div class="muted">Tür: ${esc(t.task_type)} | Zaman: ${esc(t.schedule)} | Son: ${esc(t.last_status||'bekliyor')}</div>
        <div style="margin-top:4px;font-size:12px;">${esc(t.last_summary||'')}</div>
      </div>
    `).join('');
  }
  const rr = await api('/api/tasks/runs');
  const rl = document.getElementById('runsList');
  rl.innerHTML = (rr.runs||[]).slice(0,10).map(x=>`<div class="muted">[${esc(x.started_at)}] <b>${esc(x.task_name)}</b> (${esc(x.status)}, ${Number(x.duration_ms)}ms): ${esc(x.summary)}</div>`).join('') || '<div class="muted">Henüz çalıştırma yok.</div>';
}
async function createTask(){
  let cfg = {};
  try{ cfg = JSON.parse(document.getElementById('taskConfig').value || '{}'); }catch(e){ alert('JSON hatası: '+e); return; }
  const r = await api('/api/tasks',{method:'POST',body:JSON.stringify({
    name: document.getElementById('taskName').value.trim() || 'Günlük Görev',
    task_type: document.getElementById('taskType').value,
    schedule: document.getElementById('taskSchedule').value.trim() || 'daily:08:30',
    config: cfg
  })});
  if(!r.ok) alert('Hata: '+(r.error||'Görev oluşturulamadı'));
  loadTasks(); loadStatus();
}
async function runTask(id){
  const r = await api('/api/tasks/'+encodeURIComponent(id)+'/run',{method:'POST'});
  alert(r.summary || JSON.stringify(r));
  loadTasks(); loadNotes();
}
async function delTask(id){
  await api('/api/tasks/'+encodeURIComponent(id),{method:'DELETE'});
  loadTasks(); loadStatus();
}

async function runShell(){
  const cmd = document.getElementById('shellCmd').value.trim();
  if(!cmd) return;
  document.getElementById('shellOut').textContent = 'Çalıştırılıyor...';
  const r = await api('/api/shell',{method:'POST',body:JSON.stringify({command:cmd})});
  document.getElementById('shellOut').textContent = `[exit=${r.exit_code}]\n${r.stdout||''}${r.stderr?'\n[stderr]\n'+r.stderr:''}${r.error?'\n[error]\n'+r.error:''}`;
  loadFiles();
}
async function loadFiles(){
  const r = await api('/api/files');
  const el = document.getElementById('filesList');
  el.innerHTML = (r.entries||[]).map(f=>`<div class="muted">${f.is_dir?'📁':'📄'} ${esc(f.rel_path)} (${Number(f.size)} B)</div>`).join('') || '<div class="muted">Workspace boş.</div>';
}
async function loadActivity(){
  const r = await api('/api/activity');
  const rows = (r.events||[]).map(e=>`<tr><td>${esc(e.ts)}</td><td>${esc(e.category)}</td><td>${esc(e.action)}</td><td>${esc(e.detail)}</td></tr>`).join('');
  const html = `<table><thead><tr><th>Zaman</th><th>Kategori</th><th>İşlem</th><th>Detay</th></tr></thead><tbody>${rows}</tbody></table>`;
  document.getElementById('activityTable').innerHTML = html;
  document.getElementById('overviewActivity').innerHTML = (r.events||[]).slice(0,8).map(e=>`<div class="muted">[${esc(e.ts)}] <b>${esc(e.category)}/${esc(e.action)}</b>: ${esc(e.detail)}</div>`).join('') || '<div class="muted">Henüz etkinlik yok.</div>';
}
async function refreshAll(){
  await Promise.allSettled([loadStatus(), loadWeb(), loadNotes(), loadTasks(), loadFiles(), loadActivity()]);
}
refreshAll();
setInterval(()=>{ loadStatus(); loadActivity(); }, 6000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# One-Time Dashboard Login Bootstrap Codes (Never expose master token in URL!)
# ---------------------------------------------------------------------------
_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAP_CODES: Dict[str, float] = {}


def _issue_bootstrap_code(ttl_sec: int = 60) -> str:
    now = time.time()
    code = secrets.token_urlsafe(24)
    with _BOOTSTRAP_LOCK:
        # Evict expired codes
        for k in [k for k, exp in _BOOTSTRAP_CODES.items() if exp <= now]:
            _BOOTSTRAP_CODES.pop(k, None)
        _BOOTSTRAP_CODES[code] = now + max(10, int(ttl_sec))
    return code


def _consume_bootstrap_code(code: str) -> bool:
    if not code:
        return False
    now = time.time()
    with _BOOTSTRAP_LOCK:
        exp = _BOOTSTRAP_CODES.pop(code.strip(), None)
    return exp is not None and exp > now


def _parse_int(val: Any, default: int, min_v: int = 0, max_v: int = 100000) -> int:
    try:
        n = int(val)
    except (TypeError, ValueError):
        return default
    return max(min_v, min(max_v, n))


# ---------------------------------------------------------------------------
# HTTP JSON-RPC + Dashboard Server
# ---------------------------------------------------------------------------
class SodalsOSHandler(BaseHTTPRequestHandler):
    server_version = "SODALS-OS/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # Keep stdout clean; audit log lives in SQLite

    def _send_security_headers(self, is_html: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if is_html:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._send_security_headers(is_html=False)
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict[str, Any]:
        length = _parse_int(self.headers.get("Content-Length", "0"), 0, 0, 2_000_000)
        if length <= 0 or length > 2_000_000:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if not host:
            return True
        host_only = host.split(":")[0]
        return host_only in ("127.0.0.1", "localhost", "[::1]")

    def _auth_mode(self) -> Optional[str]:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and secrets.compare_digest(auth[7:].strip(), API_TOKEN):
            return "bearer"
        hdr = self.headers.get("X-SODALS-OS-Token", "").strip()
        if hdr and secrets.compare_digest(hdr, API_TOKEN):
            return "header"
        cookie_hdr = self.headers.get("Cookie", "")
        for part in cookie_hdr.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "sodals_os_token" and secrets.compare_digest(v.strip(), API_TOKEN):
                return "cookie"
        return None

    def _is_authorized(self, require_csrf_on_cookie: bool = False) -> bool:
        if not self._host_allowed():
            return False
        mode = self._auth_mode()
        if mode is None:
            return False
        if require_csrf_on_cookie and mode == "cookie":
            # Block cross-port localhost CSRF: cookie-authenticated mutations must carry custom header & matching Origin
            if self.headers.get("X-SODALS-CSRF") != "1":
                return False
            origin = (self.headers.get("Origin") or "").strip().lower()
            if origin:
                expected_host = (self.headers.get("Host") or "").strip().lower()
                if origin not in (f"http://{expected_host}", f"https://{expected_host}"):
                    return False
        return True

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        # Public liveness probe (no sensitive data)
        if path == "/health":
            self._send_json(200, {"ok": True, "service": "sodals-os", "uptime_sec": int(time.time() - START_TIME)})
            return

        # Dashboard entry with one-time bootstrap code (never leaks permanent master token in URL!)
        if path == "/":
            if not self._host_allowed():
                self._send_json(403, {"ok": False, "error": "Forbidden Host header"})
                return
            qcode = (qs.get("code") or [""])[0].strip()
            if qcode and _consume_bootstrap_code(qcode):
                self.send_response(302)
                self.send_header("Set-Cookie", f"sodals_os_token={API_TOKEN}; Path=/; HttpOnly; SameSite=Strict")
                self.send_header("Location", "/")
                self._send_security_headers(is_html=False)
                self.end_headers()
                return
            if not self._is_authorized():
                body = b"<h2>SODALS-OS: Yetkisiz Erisim</h2><p>Paneli acmak icin <code>sodals_os.bat dashboard</code> komutunu kullanin.</p>"
                self.send_response(401)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._send_security_headers(is_html=True)
                self.end_headers()
                self.wfile.write(body)
                return
            raw = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self._send_security_headers(is_html=True)
            self.end_headers()
            self.wfile.write(raw)
            return

        if not self._is_authorized():
            self._send_json(401, {"ok": False, "error": "Unauthorized: invalid SODALS-OS token"})
            return

        if path == "/api/status":
            self._send_json(200, collect_system_status())
        elif path == "/api/web/state":
            self._send_json(200, {"ok": True, **BROWSER.state_summary()})
        elif path == "/api/notes":
            notes = DB.list_notes(
                query=(qs.get("q") or [""])[0],
                tag=(qs.get("tag") or [""])[0],
                folder=(qs.get("folder") or [""])[0],
                limit=_parse_int((qs.get("limit") or ["50"])[0], 50, 1, 200),
            )
            self._send_json(200, {"ok": True, "notes": notes})
        elif path.startswith("/api/notes/"):
            nid = urllib.parse.unquote(path[len("/api/notes/"):])
            note = DB.get_note(nid)
            if not note:
                self._send_json(404, {"ok": False, "error": "Not bulunamadı"})
            else:
                self._send_json(200, {"ok": True, "note": note})
        elif path == "/api/tasks":
            self._send_json(200, {"ok": True, "tasks": DB.list_tasks()})
        elif path == "/api/tasks/runs":
            self._send_json(200, {"ok": True, "runs": DB.recent_runs(_parse_int((qs.get("limit") or ["30"])[0], 30, 1, 200))})
        elif path == "/api/files":
            try:
                self._send_json(200, list_workspace_files((qs.get("path") or ["."])[0]))
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
        elif path == "/api/files/read":
            try:
                self._send_json(200, read_workspace_file((qs.get("path") or [""])[0]))
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
        elif path == "/api/activity":
            self._send_json(200, {"ok": True, "events": DB.recent_activity(_parse_int((qs.get("limit") or ["50"])[0], 50, 1, 200))})
        else:
            self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:
        if not self._is_authorized(require_csrf_on_cookie=True):
            self._send_json(401, {"ok": False, "error": "Unauthorized"})
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        body = self._read_json()

        if path == "/api/auth/bootstrap":
            if self._auth_mode() not in ("bearer", "header"):
                self._send_json(403, {"ok": False, "error": "Bootstrap code requires Bearer/Header token"})
                return
            code = _issue_bootstrap_code(ttl_sec=60)
            self._send_json(200, {"ok": True, "code": code, "ttl_sec": 60})
        elif path == "/api/web/open":
            res = BROWSER.open_url(
                url=str(body.get("url", "")),
                tab_id=str(body.get("tab_id", "main")),
                max_chars=_parse_int(body.get("max_chars", 10000), 10000, 200, 50000),
            )
            self._send_json(200 if res.get("ok") else 400, res)
        elif path == "/api/web/click":
            raw_idx = body.get("index")
            parsed_idx = _parse_int(raw_idx, -1, 1, 200) if raw_idx is not None else None
            res = BROWSER.click(
                tab_id=str(body.get("tab_id", "main")),
                index=parsed_idx,
                link_text=str(body.get("link_text", "")),
            )
            self._send_json(200 if res.get("ok") else 400, res)
        elif path == "/api/web/search":
            res = BROWSER.search(
                query=str(body.get("query", "")),
                tab_id=str(body.get("tab_id", "main")),
                limit=_parse_int(body.get("limit", 8), 8, 1, 25),
            )
            self._send_json(200 if res.get("ok") else 400, res)
        elif path == "/api/web/save_note":
            tid = str(body.get("tab_id", "main"))
            tab = BROWSER.tabs.get(tid)
            if not tab:
                self._send_json(400, {"ok": False, "error": f"'{tid}' sekmesinde açık sayfa yok."})
                return
            title = str(body.get("title") or tab.get("title") or tab.get("url") or "Web Notu")
            tags = body.get("tags") or ["web"]
            md = f"Kaynak URL: {tab.get('url')}\nAlınma Tarihi: {tab.get('fetched_at')}\n\n{tab.get('text', '')}"
            try:
                saved = DB.save_note(title=title, content=md, tags=tags, folder=str(body.get("folder", "web")))
                self._send_json(200, {"ok": True, "note": saved})
            except (ValueError, PermissionError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
        elif path == "/api/notes":
            try:
                saved = DB.save_note(
                    title=str(body.get("title", "Başlıksız Not")),
                    content=str(body.get("content", "")),
                    tags=body.get("tags") if isinstance(body.get("tags"), list) else [],
                    folder=str(body.get("folder", "general")),
                    note_id=body.get("note_id"),
                )
                self._send_json(200, {"ok": True, "note": saved})
            except (ValueError, PermissionError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
        elif path == "/api/tasks":
            try:
                created = DB.create_task(
                    name=str(body.get("name", "Günlük Görev")),
                    task_type=str(body.get("task_type", "web_digest")),
                    schedule=str(body.get("schedule", "daily:08:30")),
                    config=body.get("config") if isinstance(body.get("config"), dict) else {},
                    enabled=bool(body.get("enabled", True)),
                )
                self._send_json(200, {"ok": True, "task": created})
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
        elif path.startswith("/api/tasks/") and path.endswith("/run"):
            tid = urllib.parse.unquote(path[len("/api/tasks/"):-len("/run")])
            task = DB.get_task(tid)
            if not task:
                self._send_json(404, {"ok": False, "error": "Görev bulunamadı"})
                return
            res = execute_task(task)
            self._send_json(200, res)
        elif path == "/api/files/write":
            try:
                res = write_workspace_file(str(body.get("path", "")), str(body.get("content", "")))
                self._send_json(200, res)
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
        elif path == "/api/shell":
            res = run_sandboxed_shell(
                str(body.get("command", "")),
                timeout_sec=_parse_int(body.get("timeout_sec", 45), 45, 1, 180),
            )
            self._send_json(200, res)
        else:
            self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

    def do_PATCH(self) -> None:
        if not self._is_authorized(require_csrf_on_cookie=True):
            self._send_json(401, {"ok": False, "error": "Unauthorized"})
            return
        parsed = urllib.parse.urlparse(self.path)
        body = self._read_json()
        if parsed.path.startswith("/api/tasks/"):
            tid = urllib.parse.unquote(parsed.path[len("/api/tasks/"):])
            ok = DB.set_task_enabled(tid, bool(body.get("enabled", True)))
            self._send_json(200 if ok else 404, {"ok": ok})
        else:
            self._send_json(404, {"ok": False, "error": "Unknown endpoint"})

    def do_DELETE(self) -> None:
        if not self._is_authorized(require_csrf_on_cookie=True):
            self._send_json(401, {"ok": False, "error": "Unauthorized"})
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/notes/"):
            nid = urllib.parse.unquote(parsed.path[len("/api/notes/"):])
            ok = DB.delete_note(nid)
            self._send_json(200 if ok else 404, {"ok": ok})
        elif parsed.path.startswith("/api/tasks/"):
            tid = urllib.parse.unquote(parsed.path[len("/api/tasks/"):])
            ok = DB.delete_task(tid)
            self._send_json(200 if ok else 404, {"ok": ok})
        else:
            self._send_json(404, {"ok": False, "error": "Unknown endpoint"})


def main() -> None:
    DB.log_activity("system", "boot", f"SODALS-OS daemon started on {BIND_HOST}:{BIND_PORT}", True)
    sched = SchedulerThread()
    sched.start()
    httpd = ThreadingHTTPServer((BIND_HOST, BIND_PORT), SodalsOSHandler)
    print(f"SODALS_OS_DAEMON_LISTENING {BIND_HOST}:{BIND_PORT}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sched.stop_event.set()
        httpd.server_close()


if __name__ == "__main__":
    main()
