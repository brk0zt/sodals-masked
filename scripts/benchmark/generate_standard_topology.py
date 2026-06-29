import json
import re
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.parent.parent
    output_txt = root_dir / "industrial_topology.txt"
    
    statements = set()
    

        
    # ETHICALLY CLEAN ONTOLOGY: IEEE/ISO/API Standards Only
    # ⚠️ REMOVED: All 50 failure modes found in FailureSensorIQ benchmark
    # ✅ KEPT: General-purpose industrial diagnostics from academic sources
    #
    # Sources:
    #   - IEEE Std 1415: Guide for Induction Machinery Maintenance Testing
    #   - ISO 13709: Centrifugal pumps for petroleum, chemical and gas industries
    #   - API 617: Axial and Centrifugal Compressors and Expander-Compressors
    #   - API 618: Reciprocating Compressors for Petroleum, Chemical and Gas Industries
    #   - ISO 13373-1: Condition Monitoring and Diagnostics—Vibration Condition Monitoring
    #
    standard_ontology = {
        # ═══════════════════════════════════════════════════════════════════════════════
        # GENERAL-PURPOSE ACADEMIC DIAGNOSTICS (NOT in FailureSensorIQ)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Centrifugal Pump Failure Modes (ISO 13709 / API 610)
        # [Not tested in FailureSensorIQ → Safe for QMEM]
        "pump cavitation": [
            ("acoustic emission", 0.95), ("pressure delta", 0.90), 
            ("vibration", 0.85), ("flow rate", 0.75)
        ],
        "impeller erosion": [
            ("vibration", 0.95), ("flow rate", 0.90), 
            ("discharge pressure", 0.85), ("efficiency loss", 0.80)
        ],
        "pump seal degradation": [
            ("barrier fluid pressure", 0.99), ("leakage sensor", 0.95), 
            ("temperature", 0.80)
        ],
        "suction strainer clogging": [
            ("suction pressure", 0.98), ("flow rate", 0.95), 
            ("motor current", 0.85)
        ],
        
        # Gas Compressor Failure Modes (API 617 / API 618)
        # [Not tested in FailureSensorIQ → Safe for QMEM]
        "valve leakage in reciprocating compressor": [
            ("valve temperature delta", 0.95), ("acoustic emission", 0.90), 
            ("vibration", 0.75)
        ],
        "piston rod packing wear": [
            ("packing temperature", 0.95), ("vent flow", 0.90), 
            ("rod seal condition", 0.85)
        ],
        "compressor surge protection": [
            ("dynamic pressure", 0.99), ("vibration", 0.95), 
            ("flow delta", 0.90)
        ],
        "intercooler fouling": [
            ("temperature delta", 0.98), ("pressure ratio", 0.95), 
            ("cooler effectiveness", 0.90)
        ],
        
        # Wind Turbine Diagnostics (IEC 61400-13)
        # [Not tested in FailureSensorIQ → Safe for QMEM]
        "blade aerodynamic imbalance": [
            ("blade strain gauge", 0.95), ("acoustic emission", 0.90), 
            ("nacelle vibration", 0.85)
        ],
        "blade pitch actuator drift": [
            ("pitch actuator position", 0.98), ("blade load asymmetry", 0.95), 
            ("yaw vibration", 0.85)
        ],
        "gearbox lubrication failure": [
            ("oil particle counter", 0.95), ("oil viscosity", 0.90), 
            ("oil temperature", 0.85)
        ],
        "generator bearing wear (wind turbine)": [
            ("bearing vibration", 0.95), ("winding temperature", 0.90), 
            ("stray flux", 0.80)
        ],
        
        # Hydraulic System Failure Modes (ISO 4413, ISO 1219)
        # [Not tested in FailureSensorIQ → Safe for QMEM]
        "hydraulic fluid contamination": [
            ("particle counter", 0.99), ("water-in-oil sensor", 0.95), 
            ("fluid viscosity", 0.85)
        ],
        "hydraulic pump wear (swashplate)": [
            ("case drain flow", 0.95), ("vibration", 0.90), 
            ("pressure ripple", 0.85)
        ],
        "accumulator gas precharge loss": [
            ("gas pressure", 0.99), ("accumulator surface temperature", 0.90), 
            ("pulsation amplitude", 0.85)
        ],
        "proportional valve spool stiction": [
            ("valve response time", 0.98), ("pressure hysteresis", 0.95), 
            ("flow oscillation", 0.90)
        ],
        
        # Centrifugal Compressor Aerothermodynamics (API 617)
        # [Not tested in FailureSensorIQ → Safe for QMEM]
        "compressor inlet guide vane fouling": [
            ("inlet pressure drop", 0.95), ("compressor efficiency", 0.90), 
            ("inlet temperature", 0.80)
        ],
        "rotor rub/clearance loss": [
            ("dynamic pressure at shroud", 0.98), ("vibration spike", 0.95), 
            ("casing temperature gradient", 0.85)
        ],
        "seal gas leakage path enlargement": [
            ("seal gas temperature", 0.99), ("seal gas flow", 0.95), 
            ("gas temperature at seal pocket", 0.90)
        ],
        
        # Bearing and Rotor Dynamics (ISO 20816, API 670)
        # [General foundation, not specific to FailureSensorIQ scenarios]
        "bearing preload relaxation": [
            ("bearing radial clearance", 0.95), ("bearing temperature rise", 0.90), 
            ("acoustic signature", 0.85)
        ],
        "rotor bow detection": [
            ("radial vibration at 1X speed", 0.98), ("phase shift of orbit", 0.95), 
            ("runout measurement", 0.90)
        ],
        "thermal growth prediction": [
            ("temperature at bearing pedestal", 0.98), ("axial position drift", 0.95), 
            ("expansion model", 0.90)
        ],
    }
    
    for failure_mode, sensors in standard_ontology.items():
        for sensor, weight in sensors:
            clean_sensor = sensor.replace('"', '\\"').strip()
            clean_event = failure_mode.replace('"', '\\"').strip()
            stmt = f'"{clean_sensor}" MEASURES "{clean_event}" [{weight:.2f}, 0.0];'
            statements.add(stmt)
            
    print(f"Generated {len(statements)} unique topological relationships based on independent standards.")
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("// generated standard industrial topology grammar based on ISO/IEEE condition monitoring specs\n\n")
        for stmt in sorted(list(statements)):
            f.write(stmt + "\n")
            
    print(f"Topology saved to: {output_txt}")

if __name__ == "__main__":
    main()
