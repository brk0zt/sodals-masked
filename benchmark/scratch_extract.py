import json

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"

print("Searching transcript.jsonl for views/edits of oedb_runner.rs...")
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "oedb_runner.rs" in line and "calculate_human_context_score" in line:
            try:
                data = json.loads(line)
                step = data.get("step_index")
                content = str(data.get("content", ""))
                tool_calls = data.get("tool_calls", [])
                
                # Check for view_file outputs
                if "1730:" in content or "1735:" in content or "1740:" in content or "1750:" in content:
                    print(f"\n--- Found View/Edit in Step {step} content (Line {i}) ---")
                    # Let's print the lines between 1730 and 1810 if they are there
                    lines = content.split("\n")
                    for l in lines:
                        if any(l.strip().startswith(str(num) + ":") for num in range(1730, 1810)):
                            print(l)
                            
                for tc in tool_calls:
                    tc_str = json.dumps(tc)
                    if "1730:" in tc_str or "1735:" in tc_str or "1740:" in tc_str or "1750:" in tc_str:
                        print(f"\n--- Found View/Edit in Step {step} tool call (Line {i}) ---")
                        # Let's see if we can find the range
                        print(tc_str[:1000])
            except Exception as e:
                pass
