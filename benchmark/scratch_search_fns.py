import os

print("Searching for occurrences of functions in the codebase...")
funcs = ["attach_engine", "get_match_ratio", "check_match", "set_icl_mode"]
for root, dirs, files in os.walk(r"c:\Users\Burak\Desktop\sodals_core"):
    if "target" in root or ".git" in root:
        continue
    for file in files:
        if file.endswith(".rs"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for func in funcs:
                    if func in content:
                        print(f"Found '{func}' in {path}")
                        # print surrounding lines where the call is made
                        lines = content.split("\n")
                        for idx, line in enumerate(lines):
                            if func in line:
                                print(f"  Line {idx+1}: {line}")
