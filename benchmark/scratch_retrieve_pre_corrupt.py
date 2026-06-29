import json

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"
out_path = r"c:\Users\Burak\Desktop\sodals_core\pre_corrupt_step.txt"

print("Extracting step 4547...")
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 898: # Step 4547
            try:
                data = json.loads(line)
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(f"=== Step {data.get('step_index')} (Line {i}) ===\n")
                    out.write("CONTENT:\n")
                    out.write(data.get("content", ""))
                    out.write("\n\nTOOL_CALLS:\n")
                    out.write(json.dumps(data.get("tool_calls", []), indent=2))
                print(f"Step 4547 successfully extracted to {out_path}!")
                break
            except Exception as e:
                print(f"Error: {e}")
