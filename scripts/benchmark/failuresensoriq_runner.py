"""
FailureSensorIQ Evaluation Runner - Rust/Candle Integration
==========================================================

Evaluation runner for FailureSensorIQ benchmark using Rust Candle model.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from candle_bridge import CandleModelBridge
from assetops_logging_utils import AssetOpsTaskLogger, setup_assetops_logging
from benchmark_logger import save_benchmark_results


class BM25Retriever:
    """
    Lightweight, air-gapped BM25 retriever in pure Python.
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        import math
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(self.tokenize(doc)) for doc in corpus]
        self.avg_doc_len = sum(self.doc_len) / len(corpus) if corpus else 1.0
        
        # Document frequencies and term frequencies
        self.doc_freqs = {}
        self.df = {}
        for i, doc in enumerate(corpus):
            tokens = self.tokenize(doc)
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs[i] = freqs
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1
                
        self.N = len(corpus)
        self.idf = {}
        for t, f in self.df.items():
            self.idf[t] = math.log((self.N - f + 0.5) / (f + 0.5) + 1.0)
            
    def tokenize(self, text: str) -> List[str]:
        import re
        raw_tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stemmed = []
        for t in raw_tokens:
            if t.endswith("ies") and len(t) > 5:
                t = t[:-3] + "y"
            elif t.endswith("es") and len(t) > 4:
                t = t[:-2]
            elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
                t = t[:-1]
            elif t.endswith("ed") and len(t) > 4:
                t = t[:-2]
            elif t.endswith("ing") and len(t) > 5:
                t = t[:-3]
            stemmed.append(t)
        return stemmed
        
    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = self.tokenize(query)
        doc_freq = self.doc_freqs[doc_idx]
        dl = self.doc_len[doc_idx]
        
        score = 0.0
        for t in query_tokens:
            if t in self.idf:
                tf = doc_freq.get(t, 0)
                idf_val = self.idf[t]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                score += idf_val * (numerator / denominator)
        return score

    def retrieve(self, query: str, top_n: int = 5) -> List[str]:
        scores = []
        for i in range(len(self.corpus)):
            s = self.score(query, i)
            if s > 0.0:
                scores.append((self.corpus[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scores[:top_n]]


class FailureSensorIQRunner:
    """
    Production-grade evaluation runner for FailureSensorIQ benchmark.
    """
    
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        log_level: str = "INFO",
        cartridge_path: Optional[str] = None
    ):
        """
        Initialize evaluation runner with Candle model.
        
        Args:
            model_path: Path to Candle-compatible model
            max_new_tokens: Maximum tokens to generate
            log_level: Logging level
            cartridge_path: Path to QMEM cartridge file
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Setup logging
        setup_assetops_logging(log_level)
        self.logger = AssetOpsTaskLogger()
        
        # Initialize Candle model bridge
        self.model = CandleModelBridge(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            cartridge_path=cartridge_path
        )
        
        # Results storage
        self.results = []
        
        # Load clean engineering facts from textbook ontology (for RAG context)
        self.topology_facts = []
        try:
            from pathlib import Path
            topology_file = Path(__file__).parent.parent.parent / "industrial_topology.txt"
            if topology_file.exists():
                with open(topology_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("//"):
                            # Clean relation syntax: "vibration" MEASURES "bearing damage" [0.95, 0.0]; -> vibration measures bearing damage
                            cleaned = line.replace('"', '').replace(';', '')
                            if '[' in cleaned:
                                cleaned = cleaned.split('[')[0].strip()
                            self.topology_facts.append(cleaned)
                logging.info(f"Loaded {len(self.topology_facts)} clean engineering facts for RAG context.")
        except Exception as e:
            logging.warning(f"Failed to load industrial_topology.txt: {e}")
            
        # Initialize BM25 retriever
        self.retriever = BM25Retriever(self.topology_facts) if self.topology_facts else None
            
        logging.info(f"FailureSensorIQRunner initialized: {model_path}, max_tokens: {max_new_tokens}, cartridge: {cartridge_path}")
    
    def run_single_inference(self, question: str, options: List[str], option_ids: List[str], is_multi: bool = False) -> Dict[str, Any]:
        """
        Single-shot inference for FailureSensorIQ multiple choice question.
        
        Args:
            question: The question text
            options: List of option strings
            option_ids: List of option IDs (A, B, C, D, E)
            is_multi: If True, the question expects multiple correct answers
            
        Returns:
            Dictionary with inference results
        """
        # Format prompt for multiple choice (adapts instruction for multi-answer questions)
        prompt = self._format_mcq_prompt(question, options, option_ids, is_multi=is_multi)
        
        example = {
            "task_id": f"fsiq_{len(self.results)}",
            "input_prompt": prompt
        }
        
        logging.info(f"Running Candle inference for FailureSensorIQ question")
        
        # Start task logging
        self.logger.start_task(example["task_id"], prompt, self.model.get_model_info())
        
        # Run inference through Rust Candle
        result = self.model.run_single_inference(example)
        
        # Parse answers depending on single vs multi choice type
        if is_multi:
            parsed_answers = self._parse_multi_mcq_answers(result.get("raw_output", ""), option_ids)
            if not parsed_answers:
                parsed_answers = self._parse_multi_mcq_answers(result.get("parsed_answer", ""), option_ids)
            parsed_answer = ", ".join(parsed_answers) if parsed_answers else ""
        else:
            parsed_answer = self._parse_mcq_answer(result.get("raw_output", ""), option_ids)
            if not parsed_answer:
                parsed_answer = self._parse_mcq_answer(result.get("parsed_answer", ""), option_ids)
            parsed_answers = [parsed_answer] if parsed_answer else []
        
        # Log inference result
        self.logger.log_inference(
            raw_output=result.get("raw_output", ""),
            parsed_answer=parsed_answer,
            token_count=result.get("token_count", 0)
        )
        
        # Complete task
        self.logger.complete_task()
        
        # Add task_id to result for tracking
        result["task_id"] = example["task_id"]
        result["status"] = "completed" if parsed_answers else "failed"
        result["parsed_answer"] = parsed_answer
        result["parsed_answers"] = parsed_answers
        result["question"] = question
        result["options"] = options
        result["option_ids"] = option_ids
        
        self.results.append(result)
        return result
    
    def retrieve_relevant_facts(self, question: str) -> List[str]:
        """Retrieve top 5 matching textbook engineering facts for the question using BM25."""
        if self.retriever:
            return self.retriever.retrieve(question, top_n=5)
        
        # Fallback to simple matching if retriever is not set
        import re
        words = re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
        matched = []
        for fact in self.topology_facts:
            fact_lower = fact.lower()
            match_score = 0
            for w in words:
                if w in fact_lower:
                    match_score += 1
            if match_score > 0:
                matched.append((fact, match_score))
        matched.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matched[:5]]

    def _format_mcq_prompt(self, question: str, options: List[str], option_ids: List[str], is_multi: bool = False) -> str:
        """Format multiple choice question for the model with ChatML, CoT instructions, and BM25 RAG context."""
        allowed = ", ".join(option_ids)
        if is_multi:
            selection_instruction = f"CRITICAL: This question has MULTIPLE correct options. You must analyze the question and select ALL correct options from the list below. Please analyze the question step-by-step. Write down your reasoning first. At the very end of your response, output your final choice(s) inside `<answer>` tags (e.g., `<answer>B, D</answer>`). Only include letters from: {allowed}."
        else:
            selection_instruction = f"CRITICAL: This question has EXACTLY ONE correct option. You must analyze the question and select EXACTLY one option from the list below. Please analyze the question step-by-step. Write down your reasoning first. At the very end of your response, output your final choice inside `<answer>` tags (e.g., `<answer>B</answer>`). Only include a letter from: {allowed}."

        # Retrieve engineering reference facts using BM25
        # 1. Top 3 facts for the question (failure mode / asset)
        question_facts = []
        if self.retriever:
            question_facts = self.retriever.retrieve(question, top_n=3)
        else:
            question_facts = self.retrieve_relevant_facts(question)[:3]

        # 2. Top 1 fact for each option to provide a balanced reference manual
        option_facts = []
        if self.retriever:
            for opt in options:
                if len(opt.strip()) >= 3:
                    opt_facts = self.retriever.retrieve(opt, top_n=1)
                    option_facts.extend(opt_facts)

        # 3. Combine and deduplicate to prevent leakage
        combined_facts = []
        seen = set()
        for f in question_facts + option_facts:
            if f not in seen:
                seen.add(f)
                combined_facts.append(f)

        facts_str = "\n".join(f"- {f}" for f in combined_facts) if combined_facts else "No specific reference available."

        options_text = "\n".join([f"{opt_id}) {opt}" for opt_id, opt in zip(option_ids, options)])

        system_content = "You are an expert industrial systems diagnostic engine specializing in failure modes and sensor relationships."
        
        user_content = f"""Answer the following technical multiple-choice question based on the engineering reference facts provided.
{selection_instruction}

### Reference Engineering Manual:
{facts_str}

### Question:
{question}

### Options:
{options_text}"""

        prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def _strip_thinking_blocks(self, raw_output: str) -> str:
        """Strip thinking blocks from raw model output to prevent false positive matching on thoughts."""
        if not raw_output:
            return ""
        import re
        # Remove any text between thinking tags
        text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<option_analysis>.*?</option_analysis>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Also remove unclosed thinking tags if they exist at the end of text
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thought>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<option_analysis>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    def _parse_mcq_answer(self, raw_output: str, option_ids: List[str]) -> str:
        """
        Robustly parse model output to extract the selected option ID.
        Uses XML tags, regex patterns, or the last line of the output.
        """
        if not raw_output:
            return ""
        
        import re
        
        # 1. Look for XML tags first (with optional opening tag to be robust to model format slip-ups)
        match = re.search(r"(?:<answer>)?\s*([A-E])\s*</answer>", raw_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # 2. Look for the most explicit pattern: "Answer: A"
        match = re.search(r"[Aa]nswer:\s*([A-E])\b", raw_output)
        if match:
            return match.group(1)
            
        # 3. Look for patterns in the tail (last 300 chars) to prevent false positives from earlier reasoning
        tail = raw_output[-300:]
        patterns = [
            r"\b[iI]s\s+([A-E])\b",
            r"\(([A-E])\)",
            r"[Cc]hoice:\s*([A-E])",
            r"[Oo]ption\s+([A-E])"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, tail)
            if match:
                return match.group(1)
        
        # 4. Look for a lone letter at the very end of the output
        match = re.search(r"\b([A-E])\b\s*$", raw_output.strip())
        if match:
            return match.group(1)
            
        # 5. Fallback: Search the last 200 characters of reasoning (avoiding the simple alphabetical first-occurrence loop)
        tail_200 = raw_output[-200:]
        matches = re.findall(r"\b([A-E])\b", tail_200)
        if matches:
            return matches[-1]
        
        return ""

    def _parse_multi_mcq_answers(self, raw_output: str, option_ids: List[str]) -> List[str]:
        """
        Robustly parse model output to extract multiple selected option IDs.
        First strips thinking blocks to avoid false positives from reasoning text.
        """
        if not raw_output:
            return []
        
        import re
        
        # 1. Look for XML tags first (with optional opening tag to prevent format overfitting)
        match = re.search(r"(?:<answer>)?\s*([A-E,\s]+)\s*</answer>", raw_output, re.IGNORECASE)
        if match:
            found = []
            matches = re.findall(r"\b([A-E])\b", match.group(1), re.IGNORECASE)
            for m in matches:
                val = m.upper()
                if val in option_ids and val not in found:
                    found.append(val)
            if found:
                return sorted(found)
        
        # Strip thinking blocks first to prevent false positive letter matches in reasoning
        final_text = self._strip_thinking_blocks(raw_output)
        if not final_text:
            final_text = raw_output
        
        # 2. Look for the "Answer:" pattern in cleaned text (taking the last occurrence)
        pos = final_text.rfind("Answer:")
        if pos == -1:
            pos = final_text.rfind("answer:")
            
        if pos != -1:
            # Search within the answer area following the match
            answer_area = final_text[pos + len("Answer:"):].strip()
            # If the model kept writing, restrict the search area to prevent scanning subsequent reasoning
            if len(answer_area) > 200:
                answer_area = answer_area[:200]
            found = []
            matches = re.findall(r"\b([A-E])\b", answer_area)
            for m in matches:
                val = m
                if val in option_ids and val not in found:
                    found.append(val)
            if found:
                return sorted(found)
        
        # 3. Fallback: Check the last line of the cleaned output
        last_line = final_text.strip().split("\n")[-1] if final_text.strip() else ""
        if last_line:
            found = []
            matches = re.findall(r"\b([A-E])\b", last_line)
            for m in matches:
                val = m
                if val in option_ids and val not in found:
                    found.append(val)
            if found:
                return sorted(found)
                
        # 4. Last resort: Search the last 200 characters of the cleaned text
        tail = final_text[-200:]
        found = []
        matches = re.findall(r"\b([A-E])\b", tail)
        for m in matches:
            val = m
            if val in option_ids and val not in found:
                found.append(val)
                    
        return sorted(found)
    
    def evaluate_dataset(self, dataset_path: str, max_questions: Optional[int] = None, skip_questions: int = 0) -> None:
        """
        Evaluate FailureSensorIQ dataset.
        
        Args:
            dataset_path: Path to JSONL dataset
            max_questions: Maximum number of questions to evaluate (for testing)
            skip_questions: Number of questions to skip and load from cache
        """
        logging.info(f"Starting FailureSensorIQ evaluation: {dataset_path}")
        
        # Load dataset
        questions = self._load_failuresensoriq_dataset(dataset_path, max_questions)
        
        if not questions:
            logging.error("No questions found in dataset")
            return
            
        # Load cached results for skipped questions
        if skip_questions > 0:
            self._load_skipped_cache(skip_questions, questions)
        
        # Evaluate each question
        start_idx = len(self.results)
        for i in range(start_idx, len(questions)):
            question_data = questions[i]
            
            # Detect if this is a multi-answer question based on dataset filename/metadata
            is_multi = "multi" in Path(dataset_path).name.lower()
            
            logging.info(f"Evaluating FailureSensorIQ question {i+1}/{len(questions)} (multi={is_multi})")
            
            result = self.run_single_inference(
                question=question_data["question"],
                options=question_data["options"],
                option_ids=question_data["option_ids"],
                is_multi=is_multi
            )
            
            # Store ground truth for evaluation
            result["correct"] = question_data["correct"]
            result["subject"] = question_data.get("subject", "")
            result["asset_name"] = question_data.get("asset_name", "")
            
            if result["status"] == "error":
                logging.error(f"Question {i+1} failed: {result.get('error')}")
        
        logging.info(f"Completed evaluation of {len(questions)} FailureSensorIQ questions")

    def _load_skipped_cache(self, skip_questions: int, questions: List[Dict[str, Any]]) -> None:
        import os
        from datetime import datetime
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        cached_count = 0
        
        for i in range(skip_questions):
            task_id = f"fsiq_{i}"
            log_path = f"run_logs/{task_id}.json"
            
            loaded = False
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as f:
                        data = json.load(f)
                    
                    # Find the latest entry from today
                    num_entries = len(data)
                    j = num_entries - 3
                    while j >= 0:
                        meta = data[j]
                        inference = data[j+1]
                        timing = data[j+2]
                        
                        timestamp = meta.get("timestamp", "")
                        if timestamp.startswith(today_str):
                            raw_output = inference.get("raw_output", "")
                            parsed_answer = inference.get("parsed_answer", "")
                            parsed_answers = self._parse_multi_mcq_answers(raw_output, ["A", "B", "C", "D", "E"])
                            
                            result = {
                                "task_id": task_id,
                                "status": "completed" if parsed_answers else "failed",
                                "raw_output": raw_output,
                                "parsed_answer": parsed_answer,
                                "parsed_answers": parsed_answers,
                                "token_count": inference.get("token_count", 0),
                                "inference_time": inference.get("inference_time", 0.0),
                                "total_time": timing.get("total_time", 0.0),
                                "correct": questions[i]["correct"],
                                "subject": questions[i].get("subject", ""),
                                "asset_name": questions[i].get("asset_name", "")
                            }
                            self.results.append(result)
                            cached_count += 1
                            loaded = True
                            break
                        j -= 3
                except Exception as e:
                    logging.warning(f"Error loading cache for {task_id}: {e}")
            
            if not loaded:
                logging.warning(f"Could not load cache for {task_id}, inserting dummy failed result")
                result = {
                    "task_id": task_id,
                    "status": "failed",
                    "raw_output": "",
                    "parsed_answer": "",
                    "parsed_answers": [],
                    "token_count": 0,
                    "inference_time": 0.0,
                    "total_time": 0.0,
                    "correct": questions[i]["correct"],
                    "subject": questions[i].get("subject", ""),
                    "asset_name": questions[i].get("asset_name", "")
                }
                self.results.append(result)
                
        logging.info(f"Loaded {cached_count}/{skip_questions} cached results from run_logs")
    
    def _load_failuresensoriq_dataset(self, dataset_path: str, max_questions: Optional[int]) -> List[Dict[str, Any]]:
        """Load FailureSensorIQ JSONL dataset."""
        questions = []
        
        try:
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            question = json.loads(line)
                            questions.append(question)
                            
                            if max_questions and len(questions) >= max_questions:
                                break
                                
                        except json.JSONDecodeError:
                            logging.warning(f"Invalid JSON line {line_num}: {line}")
                            
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_path}: {e}")
            return []
        
        return questions
    
    def save_results(self, output_path: str = "results_failuresensoriq_candle.json") -> None:
        """
        Save evaluation results in FailureSensorIQ format.
        
        Args:
            output_path: Output file path
        """
        if not self.results:
            logging.warning("No results to save")
            return
        
        # Compute accuracy
        correct_answers = 0
        total_answers = 0
        total_f1_score = 0.0
        
        for result in self.results:
            if result.get("status") == "completed":
                total_answers += 1
                
                # Get expected correct option IDs
                expected_correct = [
                    opt_id for opt_id, is_corr in zip(result["option_ids"], result["correct"])
                    if is_corr
                ]
                
                # Get predicted option IDs
                predicted = result.get("parsed_answers", [])
                
                # Calculate strict accuracy
                if set(predicted) == set(expected_correct) and expected_correct:
                    correct_answers += 1
                    
                # Calculate F1-score
                if expected_correct:
                    intersection = set(predicted).intersection(set(expected_correct))
                    precision = len(intersection) / len(predicted) if predicted else 0.0
                    recall = len(intersection) / len(expected_correct)
                    if precision + recall > 0.0:
                        f1 = 2.0 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
                    total_f1_score += f1
        
        accuracy = correct_answers / total_answers if total_answers > 0 else 0.0
        mean_f1_score = total_f1_score / total_answers if total_answers > 0 else 0.0
        
        results_data = {
            "accuracy": accuracy,
            "mean_f1_score": mean_f1_score,
            "correct_answers": correct_answers,
            "total_answers": total_answers,
            "num_tasks": len(self.results),
            "num_completed": sum(1 for r in self.results if r.get("status") == "completed"),
            "completion_rate": sum(1 for r in self.results if r.get("status") == "completed") / len(self.results) if self.results else 0.0,
            "model_type": "rust_candle",
            "config": {
                "model_path": self.model_path,
                "max_new_tokens": self.max_new_tokens,
                "reproducibility_config": self.model.get_model_info()["reproducibility_config"]
            }
        }
        
        # Save results
        try:
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            logging.info(f"Results saved to {output_path}")
            
            # Also save to text files for easy reading
            save_benchmark_results(results_data, "FailureSensorIQ", "run_logs")
            
        except Exception as e:
            logging.error(f"Failed to save results: {e}")


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FailureSensorIQ Candle Evaluation Runner")
    parser.add_argument("--model-path", required=True, help="Path to Candle-compatible model")
    parser.add_argument("--dataset", required=True, help="Path to FailureSensorIQ dataset")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to evaluate")
    parser.add_argument("--skip-questions", type=int, default=0, help="Number of questions to skip and load from cache")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--output", default="results_failuresensoriq_candle.json", help="Output file path")
    parser.add_argument("--cartridge", help="Path to QMEM cartridge")
    
    args = parser.parse_args()
    
    # Run evaluation
    runner = FailureSensorIQRunner(
        model_path=args.model_path,
        max_new_tokens=args.max_tokens,
        log_level=args.log_level,
        cartridge_path=args.cartridge
    )
    
    runner.evaluate_dataset(args.dataset, args.max_questions, args.skip_questions)
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
