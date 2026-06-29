import json

scenarios_path = r"c:\Users\Burak\Desktop\sodals_core\oedb_scenarios.json"
try:
    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        scenarios = data.get("scenarios", [])
        print(f"Total scenarios in file: {len(scenarios)}")
        print("First 20 scenarios:")
        for idx, s in enumerate(scenarios[:20]):
            print(f"  {idx+1}: ID={s.get('scenario_id')}, Name={s.get('name')}, Industry={s.get('industry')}")
except Exception as e:
    print("Error:", e)
