import json
import re

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"

print("Searching transcript.jsonl for original runner code...")
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # We look for steps before step 4547
        if "fn get_match_ratio" in line:
            try:
                data = json.loads(line)
                step = data.get("step_index")
                if step >= 4547:
                    continue # Skip after/at corruption
                print(f"\n--- Found occurrence in Step {step} (Line {i}) ---")
                
                content = str(data.get("content", ""))
                tool_calls = data.get("tool_calls", [])
                
                if "fn get_match_ratio" in content:
                    idx = content.find("fn get_match_ratio")
                    print("Found in content:")
                    print(content[max(0, idx-200):idx+1500])
                    print("="*60)
                
                for tc in tool_calls:
                    tc_str = json.dumps(tc)
                    if "fn get_match_ratio" in tc_str:
                        idx = tc_str.find("fn get_match_ratio")
                        print("Found in tool call:")
                        print(tc_str[max(0, idx-200):idx+1500])
                        print("="*60)
            except Exception as e:
                pass
