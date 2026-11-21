"""
SODALS v2.1 — Offline Local Coding & Automation Agent (Claude Code Style)
=========================================================================
Provides completely offline, zero-network workspace tools for `sodals_chat`:
  - Navigation & search : `list_dir`, `find_files`, `grep_search`, `read_file`
  - Safe code editing   : `edit_file`, `write_file` (with unified diff + undo stack)
  - Local automation    : `run_command` (PowerShell / shell execution with timeout & guards)
  - Robust tool parser  : Parses `<sodals_tool>{...}</sodals_tool>`, `<tool_call>{...}</tool_call>`,
                          and fenced ```sodals_tool``` JSON blocks emitted by 9B models.
"""

from dataclasses import dataclass
import difflib
import fnmatch
import itertools
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".cargo",
    "target",
    "dist",
    "build",
}

MUTATING_TOOLS = {"edit_file", "write_file", "run_command"}

# Block catastrophic system-destructive commands even in auto-approve mode
DANGEROUS_COMMAND_PATTERNS = [
    r"(?i)\bformat\s+[a-z]:",
    r"(?i)\bdiskpart\b",
    r"(?i)\brm\s+-rf\s+[/\\](?:\s|$)",
    r"(?i)\bdel\s+/[sqf]+\s+[a-z]:\\",
    r"(?i)\bRemove-Item\b[^\n]*-Recurse[^\n]*\b[a-z]:\\(?:Windows|Users)?(?:\s|$|['\"])",
    r"(?i)\bshutdown\b",
]


@dataclass
class UndoEntry:
    path: str
    existed_before: bool
    previous_content: Optional[str]
    summary: str
    timestamp: float


@dataclass
class ToolExecutionResult:
    tool: str
    ok: bool
    output: str
    diff: str = ""
    changed_path: Optional[str] = None
    aborted_by_user: bool = False


class LocalWorkspaceToolbox:
    """
    Workspace-scoped offline coding & automation toolkit with undo history and safety gates.
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        restrict_to_workspace: bool = True,
        max_output_chars: int = 8000,
    ):
        self.workspace_root = os.path.realpath(os.path.abspath(workspace_root or os.getcwd()))
        self.restrict_to_workspace = restrict_to_workspace
        self.max_output_chars = max_output_chars
        self.undo_stack: List[UndoEntry] = []

    def set_workspace(self, new_root: str) -> str:
        cand = os.path.realpath(os.path.abspath(os.path.expanduser(new_root.strip().strip('"').strip("'"))))
        if not os.path.isdir(cand):
            raise NotADirectoryError(f"Klasör bulunamadı: {cand}")
        self.workspace_root = cand
        return self.workspace_root

    def resolve_path(self, raw_path: str) -> str:
        cleaned = (raw_path or ".").strip().strip('"').strip("'")
        if not os.path.isabs(cleaned):
            joined = os.path.join(self.workspace_root, cleaned)
        else:
            joined = cleaned
        # Resolve symlinks/junctions so workspace containment cannot be bypassed via symlink escape
        abs_p = os.path.realpath(os.path.abspath(joined))

        if self.restrict_to_workspace:
            root_real = os.path.realpath(self.workspace_root)
            root_norm = os.path.normcase(root_real)
            target_norm = os.path.normcase(abs_p)
            try:
                common = os.path.commonpath([root_norm, target_norm])
            except ValueError as exc:
                raise PermissionError(
                    f"Çalışma dizini dışı erişim engellendi: '{abs_p}' (root: '{self.workspace_root}')"
                ) from exc
            if common != root_norm:
                raise PermissionError(
                    f"Çalışma dizini dışı erişim engellendi: '{abs_p}'. Değiştirmek için `/cwd <yol>` kullanın."
                )
        return abs_p

    def _rel(self, abs_path: str) -> str:
        try:
            rel = os.path.relpath(abs_path, self.workspace_root)
            return rel if not rel.startswith("..") else abs_path
        except ValueError:
            return abs_path

    def _trim(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        half = self.max_output_chars // 2 - 40
        omitted = len(text) - (2 * half)
        return f"{text[:half]}\n\n[... {omitted} karakter kısaltıldı ...]\n\n{text[-half:]}"

    def list_dir(self, path: str = ".", depth: int = 2, max_entries: int = 120) -> ToolExecutionResult:
        try:
            base = self.resolve_path(path)
            if not os.path.isdir(base):
                return ToolExecutionResult("list_dir", False, f"Klasör değil veya yok: {base}")

            depth = max(1, min(4, int(depth)))
            lines: List[str] = [f"[Klasör: {self._rel(base)}]"]
            base_depth = base.rstrip("\\/").count(os.sep)

            for root, dirs, files in os.walk(base):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                cur_depth = root.rstrip("\\/").count(os.sep) - base_depth
                indent = "  " * cur_depth
                if cur_depth > 0:
                    lines.append(f"{indent}{os.path.basename(root)}/")
                if cur_depth + 1 >= depth:
                    dirs[:] = []

                sub_indent = "  " * (cur_depth + 1)
                for fn in sorted(files):
                    fpath = os.path.join(root, fn)
                    try:
                        size = os.path.getsize(fpath)
                        lines.append(f"{sub_indent}{fn} ({size} B)")
                    except OSError:
                        lines.append(f"{sub_indent}{fn}")
                    if len(lines) >= max_entries:
                        lines.append(f"... (en fazla {max_entries} girdi gösterildi)")
                        return ToolExecutionResult("list_dir", True, "\n".join(lines))

            return ToolExecutionResult("list_dir", True, "\n".join(lines))
        except Exception as exc:
            return ToolExecutionResult("list_dir", False, str(exc))

    def find_files(self, pattern: str = "*", path: str = ".", max_results: int = 80) -> ToolExecutionResult:
        try:
            base = self.resolve_path(path)
            if not os.path.isdir(base):
                return ToolExecutionResult("find_files", False, f"Klasör bulunamadı: {base}")

            matches: List[str] = []
            pat = (pattern or "*").strip()
            for root, dirs, files in os.walk(base):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for fn in sorted(files):
                    full = os.path.join(root, fn)
                    rel = self._rel(full).replace("\\", "/")
                    if fnmatch.fnmatch(fn, pat) or fnmatch.fnmatch(rel, pat):
                        matches.append(rel)
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break

            if not matches:
                return ToolExecutionResult("find_files", True, f"Eşleşen dosya yok (desen: {pat}).")
            header = f"Bulunan dosyalar ({len(matches)}):"
            return ToolExecutionResult("find_files", True, header + "\n" + "\n".join(matches))
        except Exception as exc:
            return ToolExecutionResult("find_files", False, str(exc))

    def grep_search(
        self,
        pattern: str,
        path: str = ".",
        file_glob: str = "*",
        max_matches: int = 40,
        case_sensitive: bool = False,
    ) -> ToolExecutionResult:
        try:
            if not pattern:
                return ToolExecutionResult("grep_search", False, "Arama deseni (pattern) boş olamaz.")
            base = self.resolve_path(path)
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error:
                regex = re.compile(re.escape(pattern), flags)

            hits: List[str] = []
            files_to_scan: List[str] = []
            if os.path.isfile(base):
                files_to_scan.append(base)
            elif os.path.isdir(base):
                for root, dirs, files in os.walk(base):
                    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                    for fn in sorted(files):
                        if fnmatch.fnmatch(fn, file_glob):
                            files_to_scan.append(os.path.join(root, fn))
            else:
                return ToolExecutionResult("grep_search", False, f"Yol bulunamadı: {base}")

            for fpath in files_to_scan:
                try:
                    if os.path.getsize(fpath) > 1_500_000:
                        continue
                    with open(fpath, "rb") as fb:
                        if b"\x00" in fb.read(2048):
                            continue
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, start=1):
                            if regex.search(line):
                                rel = self._rel(fpath).replace("\\", "/")
                                hits.append(f"{rel}:{lineno}: {line.rstrip()[:220]}")
                                if len(hits) >= max_matches:
                                    hits.append(f"... (en fazla {max_matches} eşleşme gösterildi)")
                                    return ToolExecutionResult("grep_search", True, "\n".join(hits))
                except OSError:
                    continue

            if not hits:
                return ToolExecutionResult("grep_search", True, f"Eşleşme bulunamadı: {pattern}")
            return ToolExecutionResult("grep_search", True, "\n".join(hits))
        except Exception as exc:
            return ToolExecutionResult("grep_search", False, str(exc))

    @staticmethod
    def _atomic_write_text(abs_p: str, content: str) -> None:
        parent = os.path.dirname(abs_p) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".sodals_tmp_", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, abs_p)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def read_file(self, path: str, start_line: int = 1, end_line: int = 150) -> ToolExecutionResult:
        try:
            abs_p = self.resolve_path(path)
            if not os.path.isfile(abs_p):
                return ToolExecutionResult("read_file", False, f"Dosya bulunamadı: {abs_p}")

            file_size = os.path.getsize(abs_p)
            if file_size > 10 * 1024 * 1024:
                return ToolExecutionResult(
                    "read_file",
                    False,
                    f"Dosya çok büyük ({file_size / (1024 * 1024):.1f} MB > 10 MB): {self._rel(abs_p)}",
                )

            start_line = max(1, int(start_line))
            end_line = max(start_line, min(start_line + 299, int(end_line)))

            with open(abs_p, "rb") as fb:
                if b"\x00" in fb.read(4096):
                    return ToolExecutionResult("read_file", False, f"İkili (binary) dosya satır okunamaz: {abs_p}")

            numbered: List[str] = []
            total = 0
            with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                for idx, raw_line in enumerate(f, start=1):
                    total = idx
                    if start_line <= idx <= end_line:
                        numbered.append(f"{idx}: {raw_line.rstrip(chr(10)).rstrip(chr(13))}")

            if total == 0:
                return ToolExecutionResult("read_file", True, f"[{self._rel(abs_p)}] (boş dosya, 0 satır)")

            s_idx = min(start_line, total)
            e_idx = min(end_line, total)
            if not numbered:
                # Requested start_line exceeded total; show final line cleanly
                with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                    tail = list(itertools.islice(f, max(0, s_idx - 1), e_idx))
                    numbered = [
                        f"{s_idx + off}: {ln.rstrip(chr(10)).rstrip(chr(13))}"
                        for off, ln in enumerate(tail)
                    ]
            header = f"[{self._rel(abs_p)} | Satır {s_idx}-{e_idx} / {total}]"
            return ToolExecutionResult("read_file", True, self._trim(header + "\n" + "\n".join(numbered)))
        except Exception as exc:
            return ToolExecutionResult("read_file", False, str(exc))

    @staticmethod
    def _unified_diff(old_text: str, new_text: str, label: str) -> str:
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{label}",
                tofile=f"b/{label}",
                lineterm="",
                n=3,
            )
        )
        return "\n".join(diff_lines) if diff_lines else "(değişiklik yok)"

    def write_file(self, path: str, content: str, overwrite: bool = False) -> ToolExecutionResult:
        try:
            abs_p = self.resolve_path(path)
            existed = os.path.exists(abs_p)
            if existed and not overwrite:
                return ToolExecutionResult(
                    "write_file",
                    False,
                    f"Dosya zaten var ({self._rel(abs_p)}). Üzerine yazmak için overwrite=true kullanın veya edit_file seçin.",
                )

            old_text = ""
            if existed and os.path.isfile(abs_p):
                with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                    old_text = f.read()

            self._atomic_write_text(abs_p, content)

            rel = self._rel(abs_p)
            self.undo_stack.append(
                UndoEntry(
                    path=abs_p,
                    existed_before=existed,
                    previous_content=old_text if existed else None,
                    summary=f"write_file({rel})",
                    timestamp=time.time(),
                )
            )
            diff_txt = self._unified_diff(old_text, content, rel)
            return ToolExecutionResult(
                "write_file",
                True,
                f"Dosya yazıldı: {rel} ({len(content)} karakter, {len(content.splitlines())} satır)",
                diff=self._trim(diff_txt),
                changed_path=abs_p,
            )
        except Exception as exc:
            return ToolExecutionResult("write_file", False, str(exc))

    def edit_file(self, path: str, target: str, replacement: str) -> ToolExecutionResult:
        try:
            abs_p = self.resolve_path(path)
            if not os.path.isfile(abs_p):
                return ToolExecutionResult("edit_file", False, f"Düzenlenecek dosya bulunamadı: {abs_p}")
            if not target:
                return ToolExecutionResult("edit_file", False, "`target` metni boş olamaz.")

            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    old_text = f.read()
            except UnicodeDecodeError:
                return ToolExecutionResult(
                    "edit_file",
                    False,
                    f"Dosya geçerli UTF-8 değil ({self._rel(abs_p)}); veri bozulmasını önlemek için düzenleme reddedildi.",
                )

            count = old_text.count(target)
            if count == 0:
                return ToolExecutionResult(
                    "edit_file",
                    False,
                    f"`target` metni {self._rel(abs_p)} içinde bulunamadı. Önce `read_file` ile tam satırları doğrulayın.",
                )
            if count > 1:
                return ToolExecutionResult(
                    "edit_file",
                    False,
                    f"`target` metni {count} farklı yerde geçiyor; tekil eşleşme için çevre satırları ekleyin.",
                )

            new_text = old_text.replace(target, replacement, 1)
            self._atomic_write_text(abs_p, new_text)

            rel = self._rel(abs_p)
            self.undo_stack.append(
                UndoEntry(
                    path=abs_p,
                    existed_before=True,
                    previous_content=old_text,
                    summary=f"edit_file({rel})",
                    timestamp=time.time(),
                )
            )
            diff_txt = self._unified_diff(old_text, new_text, rel)
            return ToolExecutionResult(
                "edit_file",
                True,
                f"Dosya güncellendi: {rel}",
                diff=self._trim(diff_txt),
                changed_path=abs_p,
            )
        except Exception as exc:
            return ToolExecutionResult("edit_file", False, str(exc))

    def undo_last_edit(self) -> ToolExecutionResult:
        if not self.undo_stack:
            return ToolExecutionResult("undo", False, "Geri alınacak dosya değişikliği yok.")
        entry = self.undo_stack.pop()
        rel = self._rel(entry.path)
        try:
            if not entry.existed_before:
                if os.path.exists(entry.path):
                    os.remove(entry.path)
                return ToolExecutionResult("undo", True, f"Yeni oluşturulan dosya kaldırıldı: {rel}", changed_path=entry.path)
            with open(entry.path, "w", encoding="utf-8") as f:
                f.write(entry.previous_content or "")
            return ToolExecutionResult("undo", True, f"Önceki haline geri alındı: {rel}", changed_path=entry.path)
        except Exception as exc:
            return ToolExecutionResult("undo", False, f"Geri alma hatası ({rel}): {exc}")

    def run_command(self, command: str, timeout_sec: int = 60) -> ToolExecutionResult:
        cmd = (command or "").strip()
        if not cmd:
            return ToolExecutionResult("run_command", False, "Çalıştırılacak komut boş olamaz.")

        for pat in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pat, cmd):
                return ToolExecutionResult(
                    "run_command",
                    False,
                    f"Güvenlik kilidi: Yıkıcı sistem komutu engellendi ({cmd}).",
                )

        timeout_sec = max(1, min(300, int(timeout_sec)))
        try:
            if os.name == "nt":
                proc_args = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
            else:
                posix_cmd = cmd[2:].lstrip() if cmd.startswith("& ") else cmd
                proc_args = ["/bin/sh", "-c", posix_cmd]

            proc = subprocess.run(
                proc_args,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
            combined = []
            combined.append(f"[exit_code={proc.returncode}]")
            if proc.stdout.strip():
                combined.append(f"[stdout]\n{proc.stdout.rstrip()}")
            if proc.stderr.strip():
                combined.append(f"[stderr]\n{proc.stderr.rstrip()}")
            if len(combined) == 1:
                combined.append("(çıktı yok)")

            return ToolExecutionResult(
                "run_command",
                proc.returncode == 0,
                self._trim("\n".join(combined)),
            )
        except subprocess.TimeoutExpired:
            return ToolExecutionResult("run_command", False, f"Komut zaman aşımına uğradı ({timeout_sec}s): {cmd}")
        except Exception as exc:
            return ToolExecutionResult("run_command", False, str(exc))

    def execute_tool(
        self,
        call: Dict[str, Any],
        auto_approve: bool = False,
        confirm_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ) -> ToolExecutionResult:
        tool_name = str(call.get("tool", "")).strip()
        if not tool_name:
            return ToolExecutionResult("", False, "Araç çağrısında 'tool' alanı eksik.")

        if tool_name in MUTATING_TOOLS and not auto_approve:
            if confirm_callback is None:
                return ToolExecutionResult(
                    tool_name,
                    False,
                    f"'{tool_name}' işlemi kullanıcı onayı gerektiriyor (onay mekanizması kapalı).",
                    aborted_by_user=True,
                )
            if not confirm_callback(tool_name, call):
                return ToolExecutionResult(
                    tool_name,
                    False,
                    f"Kullanıcı '{tool_name}' işlemini reddetti.",
                    aborted_by_user=True,
                )

        if tool_name == "list_dir":
            return self.list_dir(path=call.get("path", "."), depth=call.get("depth", 2))
        if tool_name == "find_files":
            return self.find_files(pattern=call.get("pattern", "*"), path=call.get("path", "."))
        if tool_name == "grep_search":
            return self.grep_search(
                pattern=call.get("pattern", ""),
                path=call.get("path", "."),
                file_glob=call.get("file_glob", "*"),
            )
        if tool_name == "read_file":
            return self.read_file(
                path=call.get("path", ""),
                start_line=call.get("start_line", 1),
                end_line=call.get("end_line", 150),
            )
        if tool_name == "write_file":
            return self.write_file(
                path=call.get("path", ""),
                content=str(call.get("content", "")),
                overwrite=bool(call.get("overwrite", False)),
            )
        if tool_name == "edit_file":
            return self.edit_file(
                path=call.get("path", ""),
                target=str(call.get("target", "")),
                replacement=str(call.get("replacement", "")),
            )
        if tool_name == "run_command":
            raw_to = call.get("timeout_sec", 60)
            try:
                parsed_to = int(raw_to)
            except (TypeError, ValueError):
                return ToolExecutionResult("run_command", False, f"Geçersiz timeout_sec değeri: {raw_to!r}")
            return self.run_command(
                command=str(call.get("command", "")),
                timeout_sec=parsed_to,
            )

        return ToolExecutionResult(tool_name, False, f"Bilinmeyen araç: {tool_name}")


_TOOL_PATTERNS = [
    re.compile(r"<sodals_tool>(.*?)</sodals_tool>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE),
    re.compile(r"```(?:sodals_tool|tool_call)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE),
]


def _extract_json_object(raw_block: str) -> Optional[Dict[str, Any]]:
    s = (raw_block or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end <= start:
        return None
    snippet = s[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError:
        # Tolerate unescaped Windows path backslashes emitted by local 9B models
        escaped = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', snippet)
        try:
            parsed = json.loads(escaped)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) and parsed.get("tool") else None


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extracts JSON tool-call dictionaries emitted by the model."""
    calls: List[Dict[str, Any]] = []
    seen_raw = set()
    body = text or ""

    # Strip `<think>...</think>` AND any unclosed trailing `<think>...` block so
    # truncated reasoning notes about tool syntax are never executed prematurely.
    cleaned = re.sub(r"<think>.*?</think>", "", body, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)

    for pat in _TOOL_PATTERNS:
        for m in pat.finditer(cleaned):
            raw_inner = m.group(1).strip()
            if raw_inner in seen_raw:
                continue
            parsed = _extract_json_object(raw_inner)
            if parsed is not None:
                seen_raw.add(raw_inner)
                calls.append(parsed)
    return calls


def strip_tool_blocks(text: str) -> str:
    """Removes `<think>` and raw `<sodals_tool>` markers for clean user display."""
    out = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    for pat in _TOOL_PATTERNS:
        out = pat.sub("", out)
    return out.strip()
