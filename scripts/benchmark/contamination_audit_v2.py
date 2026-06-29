#!/usr/bin/env python3
"""
SODALS QMEM Cartridge Contamination Audit v2
==============================================

Detects semantic contamination in QMEM cartridges by comparing:
1. Failure modes from FailureSensorIQ benchmark
2. Sensor-failure_mode pairs from benchmark questions
3. Generated industrial_topology.txt and QMEM ontology

Issues a CONTAMINATION WARNING if benchmark data leaks into QMEM.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Extract benchmark data
# ═══════════════════════════════════════════════════════════════════════════

def extract_benchmark_failures_and_sensors():
    """Extract all failure modes and sensor pairs from FailureSensorIQ."""
    failure_modes = set()
    sensor_failure_pairs = set()
    asset_names = set()
    
    benchmark_file = Path("FailureSensorIQ/failuresensoriq_standard/all.jsonl")
    if not benchmark_file.exists():
        print(f"⚠️  Benchmark file not found: {benchmark_file}")
        return failure_modes, sensor_failure_pairs, asset_names
    
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                question = data.get("question", "")
                options = data.get("options", [])
                asset = data.get("asset_name", "")
                
                # Extract failure mode
                if "failure event" in question.lower():
                    match = re.search(r'failure event\s+([^"]+?)\s+occurs', question, re.IGNORECASE)
                    if match:
                        fail_mode = match.group(1).strip()
                        failure_modes.add(fail_mode)
                        asset_names.add(asset)
                        
                        # Create failure mode + sensor pairs
                        for sensor in options:
                            sensor_clean = sensor.strip().lower()
                            fail_mode_clean = fail_mode.strip().lower()
                            sensor_failure_pairs.add((sensor_clean, fail_mode_clean))
            except Exception as e:
                pass
    
    return failure_modes, sensor_failure_pairs, asset_names

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Extract generated QMEM topology
# ═══════════════════════════════════════════════════════════════════════════

def extract_qmem_ontology():
    """Extract failure modes and sensor-failure pairs from generate_standard_topology.py."""
    qmem_failure_modes = set()
    qmem_sensor_failure_pairs = set()
    
    script_file = Path("scripts/benchmark/generate_standard_topology.py")
    if not script_file.exists():
        print(f"⚠️  Script file not found: {script_file}")
        return qmem_failure_modes, qmem_sensor_failure_pairs
    
    # Parse the Python dictionary from the script
    with open(script_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find standard_ontology dictionary
    try:
        # Extract the dictionary content using regex
        match = re.search(
            r'standard_ontology\s*=\s*\{(.*?)\n\s*\}',
            content,
            re.DOTALL
        )
        if match:
            dict_content = "{" + match.group(1) + "\n}"
            
            # Manually parse key-value pairs
            lines = dict_content.split("\n")
            current_key = None
            
            for line in lines:
                # Match failure mode keys: "failure mode":
                key_match = re.match(r'\s*"([^"]+)"\s*:\s*\[', line)
                if key_match:
                    current_key = key_match.group(1).strip().lower()
                    qmem_failure_modes.add(current_key)
                
                # Match sensor-weight pairs: ("sensor", weight)
                sensor_matches = re.findall(r'\(\s*"([^"]+?)"\s*,\s*([\d.]+)\s*\)', line)
                if sensor_matches and current_key:
                    for sensor, weight in sensor_matches:
                        sensor_clean = sensor.strip().lower()
                        qmem_sensor_failure_pairs.add((sensor_clean, current_key))
    except Exception as e:
        print(f"⚠️  Error parsing QMEM script: {e}")
    
    return qmem_failure_modes, qmem_sensor_failure_pairs

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: Contamination Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_contamination(
    benchmark_failures,
    benchmark_pairs,
    qmem_failures,
    qmem_pairs
):
    """Detect overlaps between benchmark and QMEM data."""
    
    contaminations = {
        "exact_failure_mode_matches": set(),
        "similar_failure_mode_matches": set(),
        "sensor_failure_pair_contamination": set(),
    }
    
    # 1. Exact failure mode matches
    exact_matches = benchmark_failures & qmem_failures
    contaminations["exact_failure_mode_matches"] = exact_matches
    
    # 2. Similar failure modes (>80% string similarity)
    for bf in benchmark_failures:
        for qf in qmem_failures:
            if bf != qf:  # Already handled in exact matches
                similarity = SequenceMatcher(None, bf, qf).ratio()
                if similarity > 0.80:
                    contaminations["similar_failure_mode_matches"].add((bf, qf, similarity))
    
    # 3. Sensor-failure pair contamination
    common_pairs = benchmark_pairs & qmem_pairs
    contaminations["sensor_failure_pair_contamination"] = common_pairs
    
    return contaminations

# ═══════════════════════════════════════════════════════════════════════════
# PART 4: Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_audit_report(contaminations, benchmark_failures, qmem_failures):
    """Print detailed contamination audit report."""
    
    print("\n" + "=" * 80)
    print("SODALS QMEM CARTRIDGE CONTAMINATION AUDIT v2")
    print("=" * 80)
    
    # Summary
    print("\n[SUMMARY]")
    print(f"  Benchmark failure modes: {len(benchmark_failures)}")
    print(f"  QMEM failure modes: {len(qmem_failures)}")
    print(f"  Exact matches: {len(contaminations['exact_failure_mode_matches'])}")
    print(f"  Similarity matches (>80%): {len(contaminations['similar_failure_mode_matches'])}")
    print(f"  Sensor-failure pair overlaps: {len(contaminations['sensor_failure_pair_contamination'])}")
    
    # Exact failure mode matches
    if contaminations["exact_failure_mode_matches"]:
        print("\n[🚨 CRITICAL] EXACT FAILURE MODE MATCHES:")
        for mode in sorted(contaminations["exact_failure_mode_matches"]):
            print(f"  ⚠️  {mode}")
        print("\n  ➜ Action: REMOVE these failure modes from QMEM ontology immediately")
    else:
        print("\n[✅ PASS] No exact failure mode matches found")
    
    # Similar failure modes
    if contaminations["similar_failure_mode_matches"]:
        print("\n[⚠️  WARNING] SIMILAR FAILURE MODES (>80% match):")
        for bf, qf, sim in sorted(contaminations["similar_failure_mode_matches"], key=lambda x: -x[2]):
            print(f"  • Benchmark: '{bf}' ↔ QMEM: '{qf}' (similarity: {sim:.1%})")
        print("\n  ➜ Action: Review and potentially remove similar modes")
    else:
        print("\n[✅ PASS] No similar failure modes detected")
    
    # Sensor-failure pair contamination
    if contaminations["sensor_failure_pair_contamination"]:
        print("\n[🔴 CRITICAL] SENSOR-FAILURE PAIR CONTAMINATION:")
        print(f"  Found {len(contaminations['sensor_failure_pair_contamination'])} overlapping pairs:")
        for i, (sensor, fail_mode) in enumerate(sorted(contaminations["sensor_failure_pair_contamination"])[:10]):
            print(f"    {i+1}. sensor='{sensor}' + failure_mode='{fail_mode}'")
        if len(contaminations["sensor_failure_pair_contamination"]) > 10:
            remaining = len(contaminations["sensor_failure_pair_contamination"]) - 10
            print(f"    ... and {remaining} more pairs")
        print("\n  ➜ Action: These pairs directly appear in benchmark questions!")
    else:
        print("\n[✅ PASS] No sensor-failure pair contamination found")
    
    # Final verdict
    print("\n" + "=" * 80)
    total_contamination = (
        len(contaminations["exact_failure_mode_matches"]) +
        len(contaminations["similar_failure_mode_matches"]) +
        len(contaminations["sensor_failure_pair_contamination"])
    )
    
    if total_contamination == 0:
        print("VERDICT: ✅ CLEAN - No detectable benchmark contamination")
        print("QMEM cartridge is ethically safe for benchmarking")
    else:
        print("VERDICT: 🚨 CONTAMINATED - Benchmark data found in QMEM")
        print(f"Total contamination issues: {total_contamination}")
        print("\nRECOMMENDATION:")
        print("  1. Remove all benchmarked failure modes from ontology")
        print("  2. Keep only general-purpose academic diagnostics")
        print("  3. Re-run this audit until VERDICT: ✅ CLEAN")
    
    print("=" * 80 + "\n")
    
    return total_contamination == 0

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🔍 Extracting benchmark data...")
    benchmark_failures, benchmark_pairs, assets = extract_benchmark_failures_and_sensors()
    print(f"  ✓ {len(benchmark_failures)} unique failure modes")
    print(f"  ✓ {len(benchmark_pairs)} sensor-failure pairs")
    print(f"  ✓ {len(assets)} asset types")
    
    print("\n🔍 Extracting QMEM ontology...")
    qmem_failures, qmem_pairs = extract_qmem_ontology()
    print(f"  ✓ {len(qmem_failures)} failure modes in QMEM")
    print(f"  ✓ {len(qmem_pairs)} sensor-failure pairs in QMEM")
    
    print("\n🔍 Running contamination detection...")
    contaminations = detect_contamination(
        benchmark_failures,
        benchmark_pairs,
        qmem_failures,
        qmem_pairs
    )
    
    # Print detailed report
    is_clean = print_audit_report(contaminations, benchmark_failures, qmem_failures)
    
    # Exit code
    exit(0 if is_clean else 1)
