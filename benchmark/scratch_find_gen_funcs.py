with open(r"c:\Users\Burak\Desktop\sodals_core\src\crates\sodals_kernel\src\bin\eodb_generator.rs", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "fn " in line:
            print(f"Line {i+1}: {line.strip()}")
