import json

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"

print("Searching transcript.jsonl for view_file tool calls...")
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "view_file" in line and "oedb_runner.rs" in line:
            try:
                data = json.loads(line)
                step = data.get("step_index")
                tool_calls = data.get("tool_calls", [])
                
                # Check if it has a view_file tool call and see if it's viewing the range we want
                for tc in tool_calls:
                    if tc.get("name") == "view_file":
                        args = tc.get("args", {})
                        start = args.get("StartLine")
                        end = args.get("EndLine")
                        print(f"Step {step} (Line {i}): viewed oedb_runner.rs from line {start} to {end}")
            except Exception as e:
                pass
