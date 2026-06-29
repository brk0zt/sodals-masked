import re
import json
from typing import List

def parse_mcq_answer(raw_output: str) -> str:
    """Robustly parse model output to extract the selected option ID (A-E)."""
    if not raw_output:
        return ""
    
    # 1. Look for explicit pattern: "Answer: A"
    match = re.search(r"Answer:\s*([A-E])", raw_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # 2. Look for common patterns
    patterns = [
        r"is\s+([A-E])\b",
        r"\(([A-E])\)",
        r"Choice:\s*([A-E])",
        r"Option\s+([A-E])"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 3. Lone letter at end
    match = re.search(r"\b([A-E])\b\s*$", raw_output.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    return ""

def test_parsing():
    test_cases = [
        ("The answer is A.", "A"),
        ("I think (B) is correct.", "B"),
        ("Choice: C", "C"),
        ("Option D is the best.", "D"),
        ("After analysis, Answer: E", "E"),
        ("Just B", "B"),
        ("The correct choice is (A)", "A"),
        ("Verdict: Option C seems right.", "C"),
        ("A", "A")
    ]
    
    for inp, exp in test_cases:
        actual = parse_mcq_answer(inp)
        print(f"Input: '{inp}' -> Got: '{actual}' (Expected: '{exp}')")
        assert actual == exp

if __name__ == "__main__":
    test_parsing()
    print("All parsing tests passed!")
