import os

overview_path = r"C:\Users\Burak\.gemini\antigravity\brain\dd486362-1ce2-44b3-bd11-7710a7cf2bb0\.system_generated\logs\overview.txt"

if os.path.exists(overview_path):
    print("Searching overview.txt...")
    with open(overview_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        if "calculate_human_context_score" in content:
            print("Found in overview.txt!")
            idx = content.find("calculate_human_context_score")
            print(content[max(0, idx-500):idx+1500])
        else:
            print("Not found in overview.txt")
else:
    print("overview.txt does not exist")
