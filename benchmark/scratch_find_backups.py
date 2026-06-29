import os

print("Searching for copies or backups of oedb_runner.rs...")
for root, dirs, files in os.walk(r"c:\Users\Burak\Desktop\sodals_core"):
    # skip target, .git
    if "target" in root or ".git" in root:
        continue
    for file in files:
        if "oedb_runner" in file:
            print(os.path.join(root, file))
