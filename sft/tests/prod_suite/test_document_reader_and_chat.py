"""
SODALS v2.1 — GPU-Free Unit Tests for Document Reader & Chat Document UX
========================================================================
Tests:
  1. Parsing .txt, .md, .json, .jsonl, synthetic .docx, synthetic .xlsx (XML fallback / pandas),
     plus empty-file / missing-file / directory / null-byte binary rejection.
  2. Windows drag-and-drop & PowerShell path normalization (`clean_user_path`,
     `extract_inline_file_paths`: `& '...'`, quoted paths, bare paths + inline queries).
  3. Query-aware chunking in `build_document_context` for >20,000-char documents.
  4. `sodals_serve` document attachment helpers (`_resolve_api_documents`,
     `SodalsNeurosymbolicEngine._prepare_messages_for_mode` with & without system prompts,
     mocked `stream_generate` elevating `SYSTEM1_DIRECT` -> `SYSTEM2_CAUSAL_LOGIC` with
     `rep_penalty == 1.0` and `max_new_tokens >= 1536`, without loading any GPU model).
"""

import json
import os
import sys
import tempfile
import threading
import unittest
import zipfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.dirname(os.path.dirname(TESTS_DIR))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32":
    import smart_app_control_compat  # noqa: F401
from arkhe_neurosymbolic_pipeline import CognitiveMode
from document_reader import (
    LoadedDocument,
    build_document_context,
    clean_user_path,
    extract_inline_file_paths,
    load_document,
)
import sodals_serve


def _create_synthetic_docx(path: str, paragraphs, table_rows=None) -> None:
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body_parts = []
    for p in paragraphs:
        body_parts.append(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>')
    if table_rows:
        tr_xml = []
        for row in table_rows:
            tc_xml = "".join(f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>" for cell in row)
            tr_xml.append(f"<w:tr>{tc_xml}</w:tr>")
        body_parts.append(f'<w:tbl>{"".join(tr_xml)}</w:tbl>')

    doc_xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{ns}"><w:body>{"".join(body_parts)}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))


def _create_synthetic_xlsx_xml(path: str) -> None:
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<si><t>Model</t></si>"
        "<si><t>Score</t></si>"
        "<si><t>SODALS</t></si>"
        "</sst>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>88.5</v></c></row>'
        "</sheetData>"
        "</worksheet>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", shared_xml.encode("utf-8"))
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml.encode("utf-8"))


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        self.last_messages = messages
        self.last_enable_thinking = enable_thinking
        return "<prompt>"

    def encode(self, text, add_special_tokens=False):
        return text.split() or ["tok"]


class TestDocumentReaderAndChatUX(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_text_md_json_jsonl(self):
        txt_path = os.path.join(self.root, "notes.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Arkhe 6 önkoşul denetimi.\n\nİkinci paragraf.")
        doc_txt = load_document(txt_path)
        self.assertEqual(doc_txt.ext, ".txt")
        self.assertIn("Arkhe 6 önkoşul", doc_txt.text)
        self.assertGreaterEqual(doc_txt.word_count, 5)

        md_path = os.path.join(self.root, "readme.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Başlık\n- Madde 1")
        doc_md = load_document(md_path)
        self.assertEqual(doc_md.ext, ".md")
        self.assertIn("# Başlık", doc_md.text)

        json_path = os.path.join(self.root, "audit.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"auroc": 0.886, "coverage": 0.98}, f)
        doc_json = load_document(json_path)
        self.assertIn('"auroc": 0.886', doc_json.text)

        jsonl_path = os.path.join(self.root, "rows.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": 1, "ans": "42"}) + "\n")
            f.write(json.dumps({"id": 2, "ans": "99"}) + "\n")
        doc_jsonl = load_document(jsonl_path)
        self.assertEqual(doc_jsonl.page_or_section_count, 2)
        self.assertIn("[Satır 2]", doc_jsonl.text)

    def test_load_synthetic_docx_and_xlsx(self):
        docx_path = os.path.join(self.root, "rapor.docx")
        _create_synthetic_docx(
            docx_path,
            paragraphs=["SODALS Deney Raporu", "Bulgular özeti burada yer alır."],
            table_rows=[["Metrik", "Değer"], ["Kapsam", "%98"]],
        )
        doc_docx = load_document(docx_path)
        self.assertEqual(doc_docx.ext, ".docx")
        self.assertIn("SODALS Deney Raporu", doc_docx.text)
        self.assertIn("Kapsam | %98", doc_docx.text)

        xlsx_path = os.path.join(self.root, "tablo.xlsx")
        _create_synthetic_xlsx_xml(xlsx_path)
        doc_xlsx = load_document(xlsx_path)
        self.assertEqual(doc_xlsx.ext, ".xlsx")
        self.assertIn("SODALS", doc_xlsx.text)
        self.assertIn("88.5", doc_xlsx.text)

    def test_error_handling_missing_directory_empty_binary(self):
        with self.assertRaises(FileNotFoundError):
            load_document(os.path.join(self.root, "yok.txt"))

        with self.assertRaises(IsADirectoryError):
            load_document(self.root)

        empty_path = os.path.join(self.root, "empty.txt")
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("   \n\t  ")
        with self.assertRaises(ValueError):
            load_document(empty_path)

        bin_path = os.path.join(self.root, "binary.bin")
        with open(bin_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x00binary")
        with self.assertRaises(ValueError):
            load_document(bin_path)

    def test_windows_drag_and_drop_path_normalization_and_inline_extraction(self):
        spaced_dir = os.path.join(self.root, "klasör boşluk")
        os.makedirs(spaced_dir, exist_ok=True)
        doc1 = os.path.join(spaced_dir, "my file.txt")
        doc2 = os.path.join(self.root, "a.txt")
        for p, txt in [(doc1, "Birinci dosya içeriği."), (doc2, "İkinci dosya içeriği.")]:
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)

        self.assertEqual(clean_user_path(f"& '{doc1}'"), doc1)
        self.assertEqual(clean_user_path(f'"{doc2}"'), doc2)

        ps_input = f"& '{doc1}'"
        paths, rem = extract_inline_file_paths(ps_input)
        self.assertEqual(paths, [os.path.abspath(doc1)])
        self.assertEqual(rem, "")

        quoted_input = f'"{doc2}" bu nedir?'
        paths, rem = extract_inline_file_paths(quoted_input)
        self.assertEqual(paths, [os.path.abspath(doc2)])
        self.assertEqual(rem, "bu nedir?")

        bare_input = f"{doc2} özetle"
        paths, rem = extract_inline_file_paths(bare_input)
        self.assertEqual(paths, [os.path.abspath(doc2)])
        self.assertEqual(rem, "özetle")

    def test_query_aware_chunking_for_large_document(self):
        intro = "GİRİŞ BÖLÜMÜ: Bu doküman SODALS mimarisi hakkındadır.\n\n"
        filler_paras = [f"Dolgu paragraf {i}: standart sistem metni ve rutin notlar." * 6 for i in range(65)]
        target_para = "KRİTİK BULGU: episte_floor_99 değerimiz 0.8864 AUROC ile doğrulandı."
        full_text = intro + "\n\n".join(filler_paras[:45] + [target_para] + filler_paras[45:])
        self.assertGreater(len(full_text), 20000)

        doc = LoadedDocument(
            name="uzun_rapor.md",
            path="uzun_rapor.md",
            ext=".md",
            text=full_text,
            page_or_section_count=67,
            char_count=len(full_text),
            word_count=len(full_text.split()),
        )
        ctx = build_document_context([doc], query="episte_floor_99 kaç çıktı?", max_total_chars=6000)
        self.assertIn("Kısmi Önizleme/Alakalı Kesitler", ctx)
        self.assertIn("GİRİŞ BÖLÜMÜ", ctx)
        self.assertIn("episte_floor_99", ctx)
        self.assertLess(len(ctx), 7500)

    def test_sodals_serve_document_integration_without_gpu(self):
        sample_path = os.path.join(self.root, "spec.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("Zorunlu önkoşul: Tüm birim testler GPU gerektirmez.")

        # Security invariant: HTTP API rejects arbitrary host filesystem paths (inline {name, content} ONLY)
        with self.assertRaises(ValueError):
            sodals_serve._resolve_api_documents([sample_path])

        with self.assertRaises(ValueError):
            sodals_serve._resolve_api_documents([{"path": sample_path}])

        resolved = sodals_serve._resolve_api_documents([
            {"name": "spec.txt", "content": "Zorunlu önkoşul: Tüm birim testler GPU gerektirmez."},
            {"name": "inline.md", "content": "Inline markdown özeti."},
        ])
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0].name, "spec.txt")
        self.assertEqual(resolved[1].name, "inline.md")

        # Test _prepare_messages_for_mode with & without existing system prompt
        engine = object.__new__(sodals_serve.SodalsNeurosymbolicEngine)
        engine.lock = threading.Lock()
        engine.tokenizer = _DummyTokenizer()
        engine.total_tokens_generated = 0
        engine.total_requests_served = 0

        msgs_no_sys = [{"role": "user", "content": "merhaba"}]
        prep1 = engine._prepare_messages_for_mode(
            msgs_no_sys, CognitiveMode.SYSTEM2_CAUSAL_LOGIC, attached_docs=resolved, user_query="önkoşul"
        )
        self.assertEqual(prep1[0]["role"], "system")
        self.assertIn("[EKLİ BELGELER BAĞLAMI]", prep1[0]["content"])
        self.assertIn("Inline markdown özeti.", prep1[0]["content"])

        msgs_with_sys = [
            {"role": "system", "content": "Özel sistem yönergesi."},
            {"role": "user", "content": "özetle"},
        ]
        prep2 = engine._prepare_messages_for_mode(
            msgs_with_sys, CognitiveMode.SYSTEM2_CAUSAL_LOGIC, attached_docs=[resolved[0]], user_query="özetle"
        )
        self.assertEqual(len(prep2), 2)
        self.assertTrue(prep2[0]["content"].startswith("Özel sistem yönergesi."))
        self.assertIn("Zorunlu önkoşul", prep2[0]["content"])

        # Test stream_generate elevates SYSTEM1_DIRECT -> SYSTEM2_CAUSAL_LOGIC cleanly when docs attached
        captured = {}

        def _fake_stream_pass(formatted_prompt, use_adapters, max_new_tokens, temperature, top_p, rep_penalty):
            captured["use_adapters"] = use_adapters
            captured["max_new_tokens"] = max_new_tokens
            captured["rep_penalty"] = rep_penalty
            yield "Belge incelendi: "
            yield "önkoşul sağlandı."

        engine._run_streaming_pass = _fake_stream_pass

        packets = list(
            engine.stream_generate(
                [{"role": "user", "content": "merhaba"}],
                max_new_tokens=512,
                temperature=0.6,
                attached_docs=resolved,
            )
        )
        telemetry = next(p for p in packets if p["type"] == "telemetry")
        done = next(p for p in packets if p["type"] == "done")

        self.assertEqual(telemetry["cognitive_mode"], CognitiveMode.SYSTEM2_CAUSAL_LOGIC.value)
        self.assertEqual(telemetry["rep_penalty"], 1.0)
        self.assertIn("Document Analysis (2 belge)", telemetry["domain"])
        self.assertTrue(engine.tokenizer.last_enable_thinking)
        self.assertFalse(captured["use_adapters"])
        self.assertEqual(captured["rep_penalty"], 1.0)
        self.assertGreaterEqual(captured["max_new_tokens"], 1536)
        self.assertEqual(done["cognitive_mode"], CognitiveMode.SYSTEM2_CAUSAL_LOGIC.value)
        self.assertEqual(done["full_text"], "Belge incelendi: önkoşul sağlandı.")


if __name__ == "__main__":
    unittest.main()
