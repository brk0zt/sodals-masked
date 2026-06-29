import json

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"

print("Searching transcript.jsonl for set_icl_mode...")
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "set_icl_mode" in line:
            try:
                data = json.loads(line)
                content = str(data.get("content", ""))
                tool_calls = data.get("tool_calls", [])
                
                # Look for set_icl_mode definitions
                if "pub fn set_icl_mode" in content:
                    idx = content.find("pub fn set_icl_mode")
                    print(f"\nFound 'pub fn set_icl_mode' in content of step {data.get('step_index')} (Line {i}):")
                    print(content[idx:idx+1200])
                    print("="*50)
                
                for tc in tool_calls:
                    tc_str = json.dumps(tc)
                    if "pub fn set_icl_mode" in tc_str:
                        idx = tc_str.find("pub fn set_icl_mode")
                        print(f"\nFound 'pub fn set_icl_mode' in tool call of step {data.get('step_index')} (Line {i}):")
                        print(tc_str[idx:idx+1200])
                        print("="*50)
            except Exception as e:
                pass
