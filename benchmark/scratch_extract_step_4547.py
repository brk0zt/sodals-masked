import json

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # We know it is around line 898 (0-indexed)
        if i >= 890 and i <= 910:
            try:
                data = json.loads(line)
                step = data.get("step_index")
                print(f"--- Line {i}, Step {step} ---")
                # Look at tool calls or content
                tool_calls = data.get("tool_calls", [])
                for tc in tool_calls:
                    print("Tool call name:", tc.get("name"))
                    args = tc.get("args", {})
                    # If this is replace_file_content or similar, let's see its arguments
                    if "ReplacementContent" in args:
                        print("ReplacementContent:")
                        print(args["ReplacementContent"])
                    if "TargetContent" in args:
                        print("TargetContent:")
                        print(args["TargetContent"])
                    if "CodeContent" in args:
                        print("CodeContent:")
                        print(args["CodeContent"])
                
                content = data.get("content", "")
                if content:
                    print("Content length:", len(content))
                    if "check_match" in content:
                        print("Content containing check_match:")
                        print(content)
            except Exception as e:
                print(f"Error at line {i}: {e}")
