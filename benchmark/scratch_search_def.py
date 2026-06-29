with open(r"c:\Users\Burak\Desktop\sodals_core\src\crates\sodals_kernel\src\oedb_runner.rs", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "fn get_match_ratio" in line or "fn check_match" in line or "fn attach_engine" in line:
            print(f"Line {i+1}: {line.strip()}")
