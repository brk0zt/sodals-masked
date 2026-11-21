"""
SODALS v2.1 — Document Reader & Context Retriever for `sodals_chat` / `sodals_serve`
====================================================================================
Zero-fragile-dependency document ingestion supporting:
  - Plain text & code: .txt, .md, .py, .rs, .js, .ts, .tex, .csv, .tsv, .log, .yaml, .yml, .toml, .ini, .html, .xml
  - Structured data  : .json, .jsonl
  - PDF documents    : .pdf (via pypdf)
  - Word documents   : .docx (native zipfile + XML parser — no python-docx required)
  - Excel sheets     : .xlsx (pandas if available, else native zipfile + XML parser)

Also handles Windows drag-and-drop terminal paths (`"C:\\path\\to\\file.pdf"` or `& 'C:\\...'`),
inline prompt attachments (`C:\\path\\doc.pdf bu belgeyi özetle`), and query-aware chunking
so large documents fit cleanly inside 8 GB VRAM context budgets without sliding-window eviction.
"""

from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import zipfile


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".py", ".rs", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h", ".hpp",
    ".java", ".go", ".sh", ".ps1", ".bat", ".sql", ".tex", ".bib",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".html", ".htm", ".css", ".xml",
}
STRUCTURED_EXTENSIONS = {".json", ".jsonl"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | STRUCTURED_EXTENSIONS | {".pdf", ".docx", ".xlsx"}
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


def _read_zip_member_bounded(zf: zipfile.ZipFile, member_name: str, max_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES) -> bytes:
    info = zf.getinfo(member_name)
    if info.file_size > max_bytes:
        raise ValueError(
            f"Arşiv üyesi güvenli boyut sınırını aşıyor: '{member_name}' ({info.file_size} > {max_bytes} bayt)."
        )
    with zf.open(info, "r") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(
            f"Arşiv üyesi açıldığında güvenli boyut sınırını aştı: '{member_name}' (>{max_bytes} bayt)."
        )
    return data


@dataclass(frozen=True)
class LoadedDocument:
    name: str
    path: str
    ext: str
    text: str
    page_or_section_count: int
    char_count: int
    word_count: int


def _read_text_with_fallback(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1254", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_pdf_text(path: str) -> Tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages.append(f"[Sayfa {idx}]\n{txt}")
    return "\n\n".join(pages), len(reader.pages)


def _extract_docx_text(path: str) -> Tuple[str, int]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    blocks: List[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("Geçerli bir .docx içeriği bulunamadı (word/document.xml eksik).")
        root = ET.fromstring(_read_zip_member_bounded(zf, "word/document.xml"))
        body = root.find("w:body", ns)
        if body is None:
            return "", 0
        for child in body:
            tag = child.tag.split("}")[-1]
            if tag == "p":
                texts = [t.text for t in child.findall(".//w:t", ns) if t.text]
                line = "".join(texts).strip()
                if line:
                    blocks.append(line)
            elif tag == "tbl":
                rows_txt = []
                for tr in child.findall(".//w:tr", ns):
                    cells = []
                    for tc in tr.findall(".//w:tc", ns):
                        c_txt = "".join(t.text for t in tc.findall(".//w:t", ns) if t.text).strip()
                        cells.append(c_txt)
                    if any(cells):
                        rows_txt.append(" | ".join(cells))
                if rows_txt:
                    blocks.append("\n".join(rows_txt))
    return "\n\n".join(blocks), max(1, len(blocks))


def _sheet_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"sheet(\d+)\.xml$", name)
    return (int(m.group(1)) if m else 10**9, name)


def _extract_xlsx_text(path: str) -> Tuple[str, int]:
    # Pre-validate zip entry sizes before either pandas or native XML parser reads them
    with zipfile.ZipFile(path, "r") as zcheck:
        total_uncompressed = sum(info.file_size for info in zcheck.infolist())
        if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError(
                f".xlsx arşivi açıldığında boyut sınırını aşıyor ({total_uncompressed} > {MAX_ZIP_UNCOMPRESSED_BYTES} bayt)."
            )
    try:
        import pandas as pd

        xls = pd.ExcelFile(path)
        parts = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            parts.append(f"[Sayfa/Sekme: {sheet} | Satır={len(df)}]\n{df.to_string(index=False)}")
        return "\n\n".join(parts), len(xls.sheet_names)
    except Exception:
        # Lightweight native XML fallback for simple xlsx workbooks
        with zipfile.ZipFile(path, "r") as zf:
            shared: List[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                sroot = ET.fromstring(_read_zip_member_bounded(zf, "xl/sharedStrings.xml"))
                for si in sroot:
                    shared.append("".join(el.text or "" for el in si.iter() if el.tag.endswith("t")))
            sheet_names = sorted(
                (n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")),
                key=_sheet_sort_key,
            )
            parts = []
            for idx, sname in enumerate(sheet_names, start=1):
                root = ET.fromstring(_read_zip_member_bounded(zf, sname))
                rows_out = []
                for row in root.iter():
                    if not row.tag.endswith("}row") and row.tag != "row":
                        continue
                    vals = []
                    for c in row:
                        ctype = c.attrib.get("t", "")
                        if ctype == "inlineStr":
                            inline_txt = "".join(
                                el.text or ""
                                for el in c.iter()
                                if el.tag.endswith("}t") or el.tag == "t"
                            )
                            vals.append(inline_txt)
                            continue
                        v_el = next((ch for ch in c if ch.tag.endswith("}v") or ch.tag == "v"), None)
                        if v_el is None or v_el.text is None:
                            vals.append("")
                        elif ctype == "s" and v_el.text.isdigit() and int(v_el.text) < len(shared):
                            vals.append(shared[int(v_el.text)])
                        else:
                            vals.append(v_el.text)
                    if any(vals):
                        rows_out.append(" | ".join(vals))
                if rows_out:
                    parts.append(f"[Sekme {idx}]\n" + "\n".join(rows_out))
            return "\n\n".join(parts), len(sheet_names)


def load_document(raw_path: str) -> LoadedDocument:
    """Loads a local document file into clean text with metadata."""
    cleaned = clean_user_path(raw_path)
    if not cleaned or not os.path.exists(cleaned):
        raise FileNotFoundError(f"Dosya bulunamadı: {cleaned or raw_path}")
    if os.path.isdir(cleaned):
        raise IsADirectoryError(
            f"'{cleaned}' bir klasördür. Lütfen tek bir dosya yolu verin (örn. .pdf, .docx, .txt, .py, .json)."
        )
    file_size = os.path.getsize(cleaned)
    if file_size > MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"Belge boyutu güvenli sınırı aşıyor ({file_size / (1024 * 1024):.1f} MB > {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB): '{os.path.basename(cleaned)}'"
        )

    ext = os.path.splitext(cleaned)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        # Try reading as UTF-8 text if small and not binary
        with open(cleaned, "rb") as bf:
            head = bf.read(2048)
        if b"\x00" in head:
            raise ValueError(
                f"Desteklenmeyen ikili (binary) dosya biçimi: '{ext or 'uzantısız'}'. "
                f"Desteklenenler: .pdf, .docx, .xlsx, .txt, .md, .json, .csv, kod dosyaları."
            )

    sections = 1
    if ext == ".pdf":
        text, sections = _extract_pdf_text(cleaned)
    elif ext == ".docx":
        text, sections = _extract_docx_text(cleaned)
    elif ext == ".xlsx":
        text, sections = _extract_xlsx_text(cleaned)
    elif ext == ".json":
        raw = _read_text_with_fallback(cleaned)
        try:
            obj = json.loads(raw)
            text = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            text = raw
    elif ext == ".jsonl":
        raw = _read_text_with_fallback(cleaned)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        sections = len(lines)
        formatted = []
        for i, ln in enumerate(lines, start=1):
            try:
                formatted.append(f"[Satır {i}] {json.dumps(json.loads(ln), ensure_ascii=False)}")
            except Exception:
                formatted.append(f"[Satır {i}] {ln}")
        text = "\n".join(formatted)
    else:
        text = _read_text_with_fallback(cleaned)
        sections = max(1, text.count("\n\n") + 1)

    text = text.strip()
    if not text:
        raise ValueError(f"'{os.path.basename(cleaned)}' dosyası boş veya okunabilir metin içermiyor (taranmış görsel PDF olabilir).")

    return LoadedDocument(
        name=os.path.basename(cleaned),
        path=os.path.abspath(cleaned),
        ext=ext or ".txt",
        text=text,
        page_or_section_count=sections,
        char_count=len(text),
        word_count=len(text.split()),
    )


def clean_user_path(raw: str) -> str:
    """Normalizes drag-and-drop PowerShell/CMD paths like `& 'C:\\a b\\x.pdf'` or `\"C:\\a\\x.pdf\"`."""
    s = (raw or "").strip()
    if s.startswith("& "):
        s = s[2:].strip()
    if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return os.path.expanduser(s)


def extract_inline_file_paths(user_input: str) -> Tuple[List[str], str]:
    """
    Detects file paths dropped or typed inside `user_input`:
      - PowerShell drag-and-drop: `& 'C:\\path\\file.pdf' bu nedir?`
      - Quoted path: `"C:\\Users\\Burak\\doc.pdf" özetle`
      - Bare Windows/Unix path: `C:\\Users\\Burak\\notes.txt incele`
    Returns `(valid_existing_paths, remaining_query_text)`.
    """
    text = (user_input or "").strip()
    found_paths: List[str] = []
    spans_to_remove: List[Tuple[int, int]] = []

    # 1. Explicit PowerShell `& '...'` or quoted paths `"..."` / `'...'`
    quoted_pat = re.compile(r"""(?:&\s*)?(["'])([a-zA-Z]:[\\/][^"'\n]+|/[^"'\n]+|\.{1,2}[\\/][^"'\n]+)\1""")
    for m in quoted_pat.finditer(text):
        cand = clean_user_path(m.group(2))
        if os.path.isfile(cand):
            found_paths.append(os.path.abspath(cand))
            spans_to_remove.append(m.span())

    # Remove matched spans before searching unquoted paths
    rem_chars = list(text)
    for start, end in spans_to_remove:
        for i in range(start, end):
            rem_chars[i] = " "
    text_after_quoted = "".join(rem_chars)

    # 2. Unquoted Windows drive or POSIX/relative paths without spaces ending in known extensions
    bare_pat = re.compile(
        r"""((?:[a-zA-Z]:[\\/]|(?<![a-zA-Z0-9_])/|(?:\.{1,2}|~)[\\/])[^\s"'<>|*?]+\.[a-zA-Z0-9]{1,8})"""
    )
    bare_spans: List[Tuple[int, int]] = []
    for m in bare_pat.finditer(text_after_quoted):
        cand = clean_user_path(m.group(1).rstrip(".,;:!?"))
        if os.path.isfile(cand):
            abs_c = os.path.abspath(cand)
            if abs_c not in found_paths:
                found_paths.append(abs_c)
            bare_spans.append(m.span())

    rem_chars2 = list(text_after_quoted)
    for start, end in bare_spans:
        for i in range(start, end):
            rem_chars2[i] = " "
    remaining = re.sub(r"\s+", " ", "".join(rem_chars2)).strip()
    return found_paths, remaining


def _select_relevant_chunks(text: str, query: str, budget_chars: int) -> Tuple[str, bool]:
    """
    If `len(text) <= budget_chars`, returns the full document untouched.
    Otherwise keeps the opening header/abstract (~30% budget) + highest query-overlap
    paragraphs/chunks so long PDFs/codebases fit 8 GB VRAM without losing key sections.
    """
    if len(text) <= budget_chars:
        return text, False

    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]
    if len(chunks) <= 1:
        # Split fixed windows if single block
        chunks = [text[i : i + 1200] for i in range(0, len(text), 1100)]

    q_terms = {w for w in re.findall(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ_]{3,}", (query or "").lower())}
    head_budget = int(budget_chars * 0.30)
    selected_indices = set()
    used = 0

    # Always include document beginning (title/intro), capping oversized first chunks
    for idx, ch in enumerate(chunks):
        if not selected_indices and len(ch) > budget_chars:
            chunks[idx] = ch[: max(200, budget_chars - 40)] + " [...]"
            selected_indices.add(idx)
            used += len(chunks[idx]) + 6
            break
        if used + len(ch) > head_budget and selected_indices:
            break
        selected_indices.add(idx)
        used += len(ch) + 6

    # Score remaining chunks by query term hits (TF overlap + numeric/section bonus)
    scored: List[Tuple[float, int]] = []
    for idx, ch in enumerate(chunks):
        if idx in selected_indices:
            continue
        low = ch.lower()
        hits = sum(low.count(t) for t in q_terms) if q_terms else 0
        # Slight boost for section headings / conclusions / tables
        structural = 0.5 if any(k in low for k in ("sonuç", "özet", "conclusion", "abstract", "table", "bulgu", "def ")) else 0.0
        scored.append((hits + structural, idx))

    scored.sort(key=lambda x: (-x[0], x[1]))
    for _, idx in scored:
        ch = chunks[idx]
        if used + len(ch) + 20 > budget_chars:
            continue
        selected_indices.add(idx)
        used += len(ch) + 20

    # Reassemble in original reading order with ellipsis markers where skipped
    ordered = sorted(selected_indices)
    out_parts: List[str] = []
    prev_idx = -1
    for idx in ordered:
        if prev_idx != -1 and idx > prev_idx + 1:
            out_parts.append("[... aradaki bölümler bağlam bütçesi için atlandı ...]")
        out_parts.append(chunks[idx])
        prev_idx = idx
    if ordered and ordered[-1] < len(chunks) - 1:
        out_parts.append("[... belgenin devamı kısaltıldı ...]")

    return "\n\n".join(out_parts), True


def build_document_context(
    docs: List[LoadedDocument],
    query: str = "",
    max_total_chars: int = 16000,
) -> str:
    """
    Formats all attached documents into a single clean context block for the system prompt,
    guaranteeing factual grounding and zero hallucination about unseen parts.
    """
    if not docs:
        return ""

    # Strictly honor max_total_chars across any number of attached documents
    per_doc_budget = max(1, int(max_total_chars) // max(1, len(docs)))
    sections: List[str] = []
    for i, doc in enumerate(docs, start=1):
        excerpt, was_truncated = _select_relevant_chunks(doc.text, query=query, budget_chars=per_doc_budget)
        trunc_note = " | Kısmi Önizleme/Alakalı Kesitler" if was_truncated else " | Tam Metin"
        header = (
            f"=== BELGE {i}: {doc.name} ({doc.ext}, {doc.page_or_section_count} bölüm/sayfa, "
            f"{doc.word_count} kelime{trunc_note}) ==="
        )
        sections.append(f"{header}\n{excerpt}\n=== BELGE {i} SONU ===")

    return (
        "\n\n[EKLİ BELGELER BAĞLAMI]\n"
        "Aşağıdaki belgeler kullanıcı tarafından incelenmen için eklenmiştir. "
        "Soruları yanıtlarken YALNIZCA belgede gördüğün somut verilere dayan; "
        "belgede yer almayan veya kısaltma nedeniyle göremediğin bir bilgiyi kesinlikle uydurma, "
        "eksik kısım varsa bunu dürüstçe belirt.\n\n"
        + "\n\n".join(sections)
    )
