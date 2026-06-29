import json
import requests
import time
import os
import sys
import traceback
import ast
import threading
import multiprocessing
# Bilimsel kütüphaneler (Modelin kullanmasına izin verdiklerimiz)
import numpy as np
import math
import cmath
import scipy
from scipy import optimize, integrate, stats

# ==============================================================================
# KONFİGÜRASYON
# ==============================================================================
SODALS_API_URL = "http://localhost:5500/api/generate"
INPUT_FILE = "problems_dev.jsonl"
OUTPUT_FILE = "sodals_scicode_results.json"
TIMEOUT_GEN = 120  # Modelin düşünme süresi
TIMEOUT_EXEC = 3   # Kodun çalışma süresi (Sonsuz döngü koruması)

# ==============================================================================
# GÜVENLİK MOTORU (SANDBOX LITE)
# ==============================================================================
class SecurityViolation(Exception):
    pass

def scan_for_malware(code_str):
    """
    Kodu çalıştırmadan önce AST (Abstract Syntax Tree) ile tarar.
    Tehlikeli importları ve fonksiyonları engeller.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return # Syntax hatası varsa zaten çalışmaz, Python hata verir.

    banned_imports = {'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'urllib', 'builtins'}
    banned_calls = {'open', 'exec', 'eval', 'compile', 'input'}

    for node in ast.walk(tree):
        # 1. Tehlikeli Importlar
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name.split('.')[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = [node.module.split('.')[0]]
            
            for name in names:
                if name in banned_imports:
                    raise SecurityViolation(f"🚫 Yasaklı kütüphane erişimi: {name}")

        # 2. Tehlikeli Fonksiyon Çağrıları (open(), exec() vb.)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in banned_calls:
                    raise SecurityViolation(f"🚫 Yasaklı fonksiyon çağrısı: {node.func.id}()")
            # os.system gibi çağrılar
            elif isinstance(node.func, ast.Attribute):
                if hasattr(node.func.value, 'id') and node.func.value.id in banned_imports:
                     raise SecurityViolation(f"🚫 Yasaklı modül kullanımı: {node.func.value.id}")

def execute_with_timeout(code, context, timeout):
    """Kodu ayrı bir thread/process içinde zaman sınırlı çalıştırır."""
    result_holder = {'success': False, 'error': 'Timeout'}
    
    def target():
        try:
            exec(code, context)
            result_holder['success'] = True
            result_holder['error'] = None
        except Exception:
            result_holder['error'] = traceback.format_exc()

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        return False, "TIMEOUT: Kod belirtilen sürede bitmedi (Sonsuz döngü?)"
    
    if not result_holder['success']:
        return False, result_holder['error']
    
    return True, "PASSED"

# ==============================================================================
# YARDIMCI FONKSİYONLAR
# ==============================================================================

def extract_code(text):
    text = text.strip()
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[0]
    return text.strip()

def safe_execute(generated_code, test_code):
    # 1. Statik Analiz (AST Scan)
    try:
        scan_for_malware(generated_code)
        scan_for_malware(test_code) # Test kodu da güvenli olmalı
    except SecurityViolation as e:
        return False, str(e)

    # 2. Çalıştırma Ortamı (Namespace)
    # Modelin kullanabileceği güvenli değişkenler
    exec_globals = {
        'np': np, 'numpy': np,
        'scipy': scipy,
        'math': math, 'cmath': cmath,
        'abs': abs, 'sum': sum, 'min': min, 'max': max, 'len': len, 'range': range,
        'int': int, 'float': float, 'complex': complex, 'list': list, 'tuple': tuple,
        'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
        'print': lambda *args: None # Print'i susturuyoruz, logu kirletmesin
    }

    # Kodu birleştir
    full_code = f"{generated_code}\n\n{test_code}"

    # 3. Çalıştır (Zaman Sınırlı)
    return execute_with_timeout(full_code, exec_globals, TIMEOUT_EXEC)

def process_problem(line):
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    prob_id = data.get('problem_id', 'Unknown')
    prob_desc = data.get('problem_description_main', '')
    prob_io = data.get('problem_io', '')
    tests = data.get('general_tests', [])
    
    # Prompt Mühendisliği: Modeli "SciCode" moduna zorluyoruz
    system_prompt = """You are SODALS, an AI Scientist. 
Solve the problem using Python. 
RULES:
1. Use numpy/scipy for calculations.
2. NO 'input()', NO 'open()', NO 'print()'.
3. Output ONLY the raw Python code block."""

    full_prompt = f"PROBLEM:\n{prob_desc}\n\nINPUT/OUTPUT REQUIREMENTS:\n{prob_io}\n\nPYTHON SOLUTION:"

    payload = {
        "prompt": full_prompt,
        "system_prompt": system_prompt,
        "temperature": 0.1,
        "max_tokens": 2048
    }

    start_time = time.time()
    generated_code = ""
    test_result = "SKIPPED"
    error_log = ""
    success = False

    try:
        response = requests.post(SODALS_API_URL, json=payload, timeout=TIMEOUT_GEN)
        if response.status_code == 200:
            raw_content = response.json().get("content", "")
            generated_code = extract_code(raw_content)
        else:
            error_log = f"API Error: {response.status_code}"
    except Exception as e:
        error_log = f"Connection Error: {str(e)}"

    latency = time.time() - start_time

    if generated_code and tests:
        combined_tests = "\n".join(tests)
        is_pass, msg = safe_execute(generated_code, combined_tests)
        success = is_pass
        test_result = "PASSED" if is_pass else "FAILED"
        if not is_pass:
            error_log = msg

    return {
        "id": prob_id,
        "success": success,
        "result": test_result,
        "latency": latency,
        "code": generated_code,
        "error": error_log[-200:] if error_log else ""
    }

# ==============================================================================
# ANA DÖNGÜ
# ==============================================================================
def main():
    print(f"🚀 SODALS SciCode Benchmark (SECURE MODE) Başlatılıyor...")
    print(f"📂 Veri Seti: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ HATA: JSONL dosyası bulunamadı!")
        return

    # Dosya satır sayısını bul
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"📊 Toplam Problem: {total_lines}")
    print("-" * 75)
    print(f"{'ID':<10} | {'DURUM':<10} | {'SÜRE':<8} | {'BİLGİ'}")
    print("-" * 75)

    results = []
    passed_problems = 0
    total_problems = 0
    start_global = time.time()

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            res = process_problem(line)
            if not res: continue

            total_problems += 1
            if res['success']:
                passed_problems += 1
                status_icon = "✅ PASS"
            elif "SecurityViolation" in res['error']:
                status_icon = "🛡️ BLOCKED" # Güvenlik duvarına takıldı
            elif res['result'] == "FAILED":
                status_icon = "❌ FAIL"
            else:
                status_icon = "⚠️ ERR "

            print(f"{res['id']:<10} | {status_icon:<10} | {res['latency']:.1f}s    | {res['error']}")
            
            results.append(res)
            
            # Anlık Kayıt
            if i % 5 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
                    json.dump(results, out, indent=2)

    total_time = time.time() - start_global
    accuracy = (passed_problems / total_problems) * 100 if total_problems > 0 else 0

    print("=" * 75)
    print(f"🏁 TEST TAMAMLANDI")
    print(f"⏱️  Toplam Süre: {total_time/60:.2f} dakika")
    print(f"✅ Başarı Oranı: %{accuracy:.2f} ({passed_problems}/{total_problems})")
    print(f"💾 Sonuçlar: {OUTPUT_FILE}")
    print("=" * 75)

if __name__ == "__main__":
    main()