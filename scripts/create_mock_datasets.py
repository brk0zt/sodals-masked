import os
import json

def create_sre():
    base_dir = "sre/Scenario-001"
    os.makedirs(f"{base_dir}/alerts", exist_ok=True)
    os.makedirs(f"{base_dir}/metrics", exist_ok=True)
    
    alert_data = {
        "alertname": "HighLatency",
        "service": "frontend-service",
        "status": "firing"
    }
    with open(f"{base_dir}/alerts/alert1.json", "w") as f:
        json.dump(alert_data, f, indent=2)
        
    metric_data = {
        "metric_name": "latency",
        "value": 150.5
    }
    with open(f"{base_dir}/metrics/metric1.json", "w") as f:
        json.dump(metric_data, f, indent=2)
        
    gt_yaml = """fault:
  - entity:
      name: frontend-service
"""
    with open(f"{base_dir}/ground_truth.yaml", "w") as f:
        f.write(gt_yaml)

def create_finops():
    base_dir = "finops/Scenario-001"
    os.makedirs(base_dir, exist_ok=True)
    
    csv_data = """timestamp,resource_id,cost,usage
2026-06-07T00:00:00,storage-node-05,500.0,95.0
"""
    with open(f"{base_dir}/data.csv", "w") as f:
        f.write(csv_data)
        
    anomaly_data = {
        "anomaly_type": "cost_spike",
        "resource_id": "storage-node-05"
    }
    with open(f"{base_dir}/anomaly.json", "w") as f:
        json.dump(anomaly_data, f, indent=2)
        
    gt_yaml = """resource:
  - resource:
      name: storage-node-05
"""
    with open(f"{base_dir}/ground_truth.yaml", "w") as f:
        f.write(gt_yaml)

if __name__ == "__main__":
    create_sre()
    create_finops()
    print("Mock SRE and FinOps datasets created successfully!")
