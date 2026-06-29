import asyncio
import json
import os
import argparse
import re
import time
import subprocess
import random
from typing import List, Dict, Optional
import aiohttp
import fitz  # PyMuPDF
from tqdm.asyncio import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- QYPTOS EBNF EXTRACTION PROMPTS ---

PROMPT_PHASE1_LOCAL = """Analyze raw text and extract causal triplets.
Format exactly: "Concept A" CAUSES "Concept B" [0.8, 0.0];
Output ONLY triplets. Use [0.5, 0.0] as neutral default."""

PROMPT_PHASE2_SYNTHESIS = """You are a Quantum Epistemology Engine. find the g(x) hidden variable.
f(x) Cluster 1: {cluster_1}
f(y) Cluster 2: {cluster_2}
Output ONLY "A" CORRELATES_PHASE "B" [0.5, 3.14]; or "A" RESONATES_WITH "B" [0.5, 0.0];"""

# --- LLM CLIENT (ASYNCHRONOUS + BACKOFF) ---

class LlmClient:
    def __init__(self, api_type: str, model: str, semaphore_limit: int = 5):
        self.api_type = api_type
        self.model = model
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.openai_key = os.getenv("OPENAI_API_KEY")

    async def call(self, system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    if self.api_type == "ollama":
                        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                        return await self._call_ollama(combined)
                    elif self.api_type == "openai":
                        return await self._call_openai(system_prompt, user_prompt)
                except Exception as e:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    if attempt == max_retries - 1:
                        print(f"[RETRY FAILED] Permanent failure after {max_retries} attempts: {e}")
                        return ""
                    await asyncio.sleep(wait_time)
            return ""

    async def _call_ollama(self, prompt) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=600) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("response", "")

    async def _call_openai(self, system, user) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

# --- CORE UTILS ---

class DocumentProcessor:
    @staticmethod
    def extract_and_chunk(path: str, chunk_size_words: int = 1000, overlap_words: int = 150) -> List[str]:
        text = ""
        if path.endswith(".pdf"):
            print(f"[*] Extracting PDF: {path}")
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        
        words = text.split()
        if not words: return []
        
        chunks = []
        if len(words) <= chunk_size_words: return [" ".join(words)]
        
        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i : i + chunk_size_words]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size_words >= len(words): break
        
        print(f"[*] Document split into {len(chunks)} chunks.")
        return chunks

class StateManager:
    def __init__(self, path: str):
        self.path = path
        self.data = {"phase1": [], "phase2": [], "processed_chunks": 0}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.data = json.load(f)
            except: pass

    def checkpoint(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

# --- COGNITIVE SYNTHESIZER ---

class CognitiveSynthesizer:
    def __init__(self, client: LlmClient, state: StateManager):
        self.client = client
        self.state = state

    def clean_ebnf(self, text: str) -> List[str]:
        # Triplet Carver: find (concept) RELATION (concept) [metadata]
        pattern = r'((?:"|\')?[^"\'\r\n]+(?:"|\')?\s+(?:CAUSES|MEASURES|CORRELATES_PHASE|RESONATES_WITH)\s+(?:"|\')?[^"\'\r\n]+(?:"|\')?\s*(?:\[[^\]]+\])?;?)'
        matches = re.findall(pattern, text, re.I | re.DOTALL)
        refined = []
        for m in matches:
            m = m.strip()
            rel_match = re.search(r'(CAUSES|MEASURES|CORRELATES_PHASE|RESONATES_WITH)', m, re.I)
            if not rel_match: continue
            
            rel = rel_match.group(1).upper()
            pre_rel = m[:rel_match.start()].strip()
            post_rel = m[rel_match.end():].strip()
            
            con1 = pre_rel.strip('"\' ')
            metadata = ""
            if "[" in post_rel:
                meta_match = re.search(r'\[.*?\]', post_rel)
                if meta_match:
                    metadata = meta_match.group(0)
                    post_rel = post_rel[:meta_match.start()].strip()
            
            con2 = post_rel.strip('"\' ;')
            if not metadata: metadata = "[0.5, 0.0]"
            
            if con1 and con2 and len(con1) < 200 and len(con2) < 200:
                refined.append(f'"{con1}" {rel} "{con2}" {metadata};')
        return refined

    async def run_phase1(self, chunks: List[str]):
        """f(x): Local Extraction Phase."""
        unprocessed_indices = [i for i in range(len(chunks)) if i >= self.state.data["processed_chunks"]]
        if not unprocessed_indices: return

        print(f"[*] Starting Phase 1: Local f(x) Extraction ({len(unprocessed_indices)} chunks remaining)")
        
        async def process_one(idx, chunk):
            output = await self.client.call(PROMPT_PHASE1_LOCAL, chunk)
            triplets = self.clean_ebnf(output)
            return triplets

        # Process in sub-batches for checkpointing safety
        batch_size = 10
        for i in range(0, len(unprocessed_indices), batch_size):
            batch_idxs = unprocessed_indices[i : i + batch_size]
            tasks = [process_one(idx, chunks[idx]) for idx in batch_idxs]
            results = await tqdm.gather(*tasks, desc=f"f(x) Batch {i//batch_size}")
            
            for res in results:
                self.state.data["phase1"].extend(res)
                self.state.data["processed_chunks"] += 1
            self.state.checkpoint()

    def cluster_triplets(self, triplets: List[str], k: int = 20) -> List[List[str]]:
        """Group triplets into semantic clusters using KMeans."""
        if len(triplets) < k: return [triplets]
        
        # Strip metadata and quotes for vectorization
        cleaned_trips = [re.sub(r'\[.*?\]|;', '', t).replace('"', '') for t in triplets]
        vectorizer = TfidfVectorizer().fit_transform(cleaned_trips)
        
        n_clusters = min(k, len(triplets))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectorizer.toarray())
        
        clusters = [[] for _ in range(n_clusters)]
        for trip, label in zip(triplets, kmeans.labels_):
            clusters[label].append(trip)
        return [c for c in clusters if c]

    async def run_phase2(self):
        """g(x): Cross-Reflection Phase (Rubik's Cube rotation)."""
        if self.state.data["phase2"]: return 
        if not self.state.data["phase1"]: 
            print("[WARNING] No baseline f(x) knowledge found. Skipping Phase 2.")
            return

        clusters = self.cluster_triplets(self.state.data["phase1"], k=20)
        print(f"[*] Starting Phase 2: g(x) Synthesis ({len(clusters)} semantic clusters)")
        
        synth_tasks = []
        cluster_pairs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = "\n".join(clusters[i]), "\n".join(clusters[j])
                cluster_pairs.append((c1, c2))
        
        async def synthesize_one(c1, c2):
            user_msg = PROMPT_PHASE2_SYNTHESIS.format(cluster_1=c1, cluster_2=c2)
            # Combine system prompt into user message for Ollama flexibility
            output = await self.client.call("", user_msg)
            return self.clean_ebnf(output)

        results = await tqdm.gather(*(synthesize_one(p[0], p[1]) for p in cluster_pairs), desc="g(x) Synthesis")
        for res in results:
            self.state.data["phase2"].extend(res)
        self.state.checkpoint()

    def run_phase3_compile(self, output_qmem: str):
        """Final EBNF deduplication and QMEM emission."""
        all_triplets = list(set(self.state.data["phase1"] + self.state.data["phase2"]))
        if not all_triplets:
            print("[ERROR] No intelligence extracted. Check source/model.")
            return

        tmp_txt = ".tmp_synth.txt"
        with open(tmp_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(all_triplets))
        
        print(f"[*] Synthesizer Output: {len(all_triplets)} triplets.")
        cmd = ["cargo", "run", "-p", "sodals_neuro", "--bin", "qyptos_compiler", "--", tmp_txt, output_qmem]
        subprocess.run(cmd, check=True)

async def main():
    parser = argparse.ArgumentParser(description="Cognitive Graph Synthesizer (Production V1.0)")
    parser.add_argument("--input", required=True, help="PDF or Text Source")
    parser.add_argument("--output", required=True, help="Target .qmem binary")
    parser.add_argument("--api", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="yerli-qwen:latest")
    parser.add_argument("--state", default="synth_state.json")
    parser.add_argument("--semaphore", type=int, default=5, help="Async concurrency limit")
    args = parser.parse_args()

    # Orchestration
    chunks = DocumentProcessor.extract_and_chunk(args.input, chunk_size_words=1000, overlap_words=150)
    if not chunks: return

    client = LlmClient(args.api, args.model, semaphore_limit=args.semaphore)
    state = StateManager(args.state)
    synth = CognitiveSynthesizer(client, state)

    await synth.run_phase1(chunks)
    await synth.run_phase2()
    synth.run_phase3_compile(args.output)
    print("\x1b[1;32m[SUCCESS] Cognitive Cartridge Emitted.\x1b[0m")

if __name__ == "__main__":
    asyncio.run(main())
