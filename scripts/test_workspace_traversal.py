"""
Workspace Traversal Test
==========================

Test script for workspace loader and executor.

Usage:
    python test_workspace_traversal.py --workspace-path sodals_workspace
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "crates" / "sodals_mcp" / "src"))

# This is a Python stub for testing the Rust workspace_loader
# In production, this would use PyO3 bindings or call the Rust binary

def test_workspace_structure(workspace_path: str):
    """Test that workspace structure is valid."""
    workspace = Path(workspace_path)
    
    print(f"Testing workspace structure: {workspace}")
    
    # Check Layer 0: AGENT.md
    agent_path = workspace / "AGENT.md"
    if not agent_path.exists():
        print(f"❌ AGENT.md not found")
        return False
    print(f"✅ AGENT.md found")
    
    # Check Layer 1: CONTEXT.md
    context_path = workspace / "CONTEXT.md"
    if not context_path.exists():
        print(f"❌ CONTEXT.md not found")
        return False
    print(f"✅ CONTEXT.md found")
    
    # Check stages
    expected_stages = ["01_perception", "02_embedding", "03_fusion", "04_reasoning", "05_compliance"]
    for stage_name in expected_stages:
        stage_path = workspace / stage_name
        if not stage_path.exists():
            print(f"❌ Stage {stage_name} not found")
            return False
        
        # Check CONTEXT.md in stage
        stage_context = stage_path / "CONTEXT.md"
        if not stage_context.exists():
            print(f"❌ {stage_name}/CONTEXT.md not found")
            return False
        
        # Check references folder
        references_path = stage_path / "references"
        if not references_path.exists():
            print(f"❌ {stage_name}/references not found")
            return False
        
        # Count reference skills
        ref_count = len(list(references_path.glob("*.md")))
        print(f"✅ {stage_name}: {ref_count} reference skills")
    
    return True


def create_sample_input():
    """Create sample input for workspace traversal."""
    return {
        "vision": {
            "images": [
                {
                    "type": "thermal",
                    "data": "base64_encoded_pixels_stub",
                    "metadata": {
                        "timestamp": "2026-06-27T00:00:00Z",
                        "location": "bearing_housing",
                        "resolution": "640x480"
                    }
                }
            ]
        },
        "iot_streams": {
            "channels": [
                {
                    "name": "bearing_temp_celsius",
                    "values": [87.5, 88.2, 89.1],
                    "timestamps": ["2026-06-27T00:00:00Z", "2026-06-27T00:00:05Z", "2026-06-27T00:00:10Z"],
                    "unit": "celsius",
                    "baseline": 75.0,
                    "anomaly_delta": 10.0
                }
            ]
        },
        "documents": [
            {
                "type": "pdf",
                "content": "Maintenance log: Last bearing replacement 2025-03-15",
                "metadata": {
                    "source": "maintenance_log",
                    "date": "2025-03-15"
                }
            }
        ]
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test workspace traversal")
    parser.add_argument("--workspace-path", default="sodals_workspace", help="Path to workspace")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Workspace Structure Test")
    print("="*60)
    
    # Test structure
    structure_valid = test_workspace_structure(args.workspace_path)
    
    if structure_valid:
        print("\n✅ Workspace structure is valid")
        
        # Create sample input
        sample_input = create_sample_input()
        
        print("\nSample input created:")
        print(json.dumps(sample_input, indent=2))
        
        print("\nNote: Actual workspace traversal requires Rust integration.")
        print("The workspace_loader.rs module provides the implementation.")
    else:
        print("\n❌ Workspace structure is invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
