import json

scenarios_path = r"c:\Users\Burak\Desktop\sodals_core\oedb_scenarios.json"
try:
    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        scenarios = data.get("scenarios", [])
        
        # Scenario 1 (index 0)
        s1 = scenarios[0]
        # Scenario 16 (index 15)
        s16 = scenarios[15]
        
        with open("health_scenarios_dump.txt", "w", encoding="utf-8") as out:
            out.write("=== Scenario 1 (HEALTH-VISION-L5-001) ===\n")
            out.write(json.dumps(s1, indent=2))
            out.write("\n\n=== Scenario 16 (HEALTHCARE-SGK-L5-001) ===\n")
            out.write(json.dumps(s16, indent=2))
        print("Successfully dumped both scenarios to health_scenarios_dump.txt!")
except Exception as e:
    print("Error:", e)
