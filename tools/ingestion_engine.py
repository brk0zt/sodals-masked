import argparse
import subprocess
import os
import sys
import json
import requests
import re
import time

# --- QYPTOS EXTRACTION PROMPT ---
SYSTEM_PROMPT = """You are a Topological Physics Extractor for the SODALS QYPTOS system.
Analyze raw text and extract semantic relationships into this EXACT EBNF format:

"Concept A" RELATION "Concept B" [energy, phase];

RELATION SET (ONLY USE THESE 4):
- CAUSES: direct causality
- MEASURES: sensor/observation
- CORRELATES_PHASE: temporal/phase linkage
- RESONATES_WITH: harmonic coupling

STRICT RULES:
1. You MUST include [energy, phase] for EVERY line. Use [0.5, 0.0] as neutral defaults if not specified.
2. RELATION must be ALL CAPS from the set above. Do not use 'SIMULTANEOUS' or other words.
3. Concepts MUST be in "double quotes".
4. Semicolon ; is REQUIRED at the end of every line.
5. NO natural language. NO markdown. ONLY EBNF code."""

class IngestionEngine:
    def __init__(self, api_type, model_name="yerli-qwen:latest"):
        self.api_type = api_type
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

    def call_llm(self, prompt):
        if self.api_type == "ollama":
            return self._call_ollama(prompt)
        elif self.api_type == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown API type: {self.api_type}")

    def _call_ollama(self, prompt):
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Rotor windings fault causes vibration. A vibration sensor measures this."},
                {"role": "assistant", "content": '"rotor windings fault" CAUSES "vibration" [0.85, 0.0];\n"vibration sensor" MEASURES "vibration" [1.0, 0.0];'},
                {"role": "user", "content": f"EXTRACT TOPOLOGY FROM THIS TEXT (Output ONLY EBNF):\n{prompt}"}
            ],
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=600)  # 10 min timeout for 19GB model
            if response.status_code == 404:
                print(f"[ERROR] Ollama model '{self.model_name}' not found.")
                return ""
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            print(f"[ERROR] Ollama call failed (Chat API): {e}")
            return ""

    def _call_openai(self, prompt):
        if not self.api_key:
            print("[ERROR] OPENAI_API_KEY environment variable not set.")
            return ""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": "gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Rotor windings fault causes vibration. A vibration sensor measures this."},
                {"role": "assistant", "content": '"rotor windings fault" CAUSES "vibration" [0.85, 0.0];\n"vibration sensor" MEASURES "vibration" [1.0, 0.0];'},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[ERROR] OpenAI call failed: {e}")
            return ""

    def clean_output(self, text):
        # Primary Strategy: Use regex to extract all valid triplets
        # Pattern: "concept" RELATION "concept" [energy, phase];
        triple_pattern = r'("(?:[^"]+)"\s+(?:CAUSES|MEASURES|CORRELATES_PHASE|RESONATES_WITH)\s+"(?:[^"]+)"\s*(?:\[[^\]]+\])?;?)'
        matches = re.findall(triple_pattern, text, re.DOTALL)
        
        if not matches:
            return ""

        cleaned_lines = []
        for line in matches:
            line = line.strip()
            # Ensure it ends with a semicolon
            if not line.endswith(";"):
                line += ";"
            
            # Ensure it has [energy, phase]
            if "[" not in line:
                line = line.replace(";", " [0.5, 0.0];")
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines).strip()

    def compile_qmem(self, ebnf_file, qmem_file):
        print(f"[*] Executing Linus Torvalds Factory: {ebnf_file} -> {qmem_file}")
        # Run from root of sodals_core
        cmd = ["cargo", "run", "-p", "sodals_neuro", "--bin", "qyptos_compiler", "--", ebnf_file, qmem_file]
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print("\n" + "\x1b[1;32m" + "="*60)
                print(f"SUCCESS: Topological Cartridge Built in {elapsed:.2f}s")
                print(f"Location: {os.path.abspath(qmem_file)}")
                print("="*60 + "\x1b[0m\n")
                return True
            else:
                print("\n" + "\x1b[1;31m" + "!"*60)
                print("COMPILATION FAILURE: Syntax error in LLM-generated topology.")
                print("-" * 60)
                print(result.stderr)
                print("!"*60 + "\x1b[0m\n")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to execute compiler: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="QYPTOS Topological Ingestion Engine (Linus Edition)")
    parser.add_argument("--input", required=True, help="Path to raw text source")
    parser.add_argument("--output", required=True, help="Output target for .qmem binary")
    parser.add_argument("--api", choices=["ollama", "openai"], default="ollama", help="LLM Backend")
    parser.add_argument("--model", default="yerli-qwen:latest", help="LLM Model (Ollama only)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file {args.input} not found.")
        sys.exit(1)

    print(f"[*] Reading source text: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    engine = IngestionEngine(args.api, args.model)
    print(f"[*] Dispatching to {args.api} for triplet extraction...")
    
    llm_output = engine.call_llm(raw_text)
    
    if not llm_output:
        print("[ERROR] Brain flatline. No output from LLM.")
        sys.exit(1)

    ebnf_content = engine.clean_output(llm_output)
    
    if not ebnf_content:
        print("[ERROR] Extraction failed. LLM output does not contain valid QYPTOS EBNF.")
        print("LLM RAW OUTPUT:")
        print("-" * 20)
        print(llm_output)
        sys.exit(1)

    tmp_file = ".tmp_ingest.txt"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(ebnf_content)

    print(f"[*] Intermediate EBNF emitted to {tmp_file}")
    success = engine.compile_qmem(tmp_file, args.output)
    
    if not success:
        print("[FAIL] Ingestion aborted due to compiler errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
