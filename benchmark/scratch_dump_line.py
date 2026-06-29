import json
import sys

log_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\transcript.jsonl"

line_num = int(sys.argv[1]) if len(sys.argv) > 1 else 414
out_path = r"c:\Users\Burak\Desktop\sodals_core\line_dump.txt"

print(f"Dumping line {line_num}...")
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == line_num:
            try:
                data = json.loads(line)
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(f"=== Line {i} ===\n")
                    out.write("CONTENT:\n")
                    out.write(data.get("content", ""))
                    out.write("\n\nTOOL_CALLS:\n")
                    out.write(json.dumps(data.get("tool_calls", []), indent=2))
                print(f"Line {line_num} successfully dumped to {out_path}!")
                break
            except Exception as e:
                print(f"Error: {e}")
