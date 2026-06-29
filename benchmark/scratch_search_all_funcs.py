import json

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"

print("Searching transcript.jsonl for function definitions...")
targets = ["fn attach_engine", "fn get_match_ratio", "fn check_match"]
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        for target in targets:
            if target in line:
                try:
                    data = json.loads(line)
                    content = str(data.get("content", ""))
                    tool_calls = data.get("tool_calls", [])
                    step = data.get("step_index")
                    
                    if target in content:
                        idx = content.find(target)
                        print(f"\nFound '{target}' in content of step {step} (Line {i}):")
                        print(content[max(0, idx-100):idx+1200])
                        print("="*50)
                        
                    for tc in tool_calls:
                        tc_str = json.dumps(tc)
                        if target in tc_str:
                            idx = tc_str.find(target)
                            print(f"\nFound '{target}' in tool call of step {step} (Line {i}):")
                            print(tc_str[max(0, idx-100):idx+1200])
                            print("="*50)
                except Exception as e:
                    pass
