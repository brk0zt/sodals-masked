#!/usr/bin/env python3
"""
ITBench Integration Test Script
============================

Quick test of the evaluation integration system.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from reproducibility import setup_reproducibility
from model_wrapper import LocalModelWrapper
from logging_utils import TaskLogger


def test_integration():
    """Test all integration components."""
    print("Testing ITBench Integration...")
    print("=" * 40)
    
    # Test 1: Reproducibility setup
    print("1. Testing reproducibility setup...")
    setup_reproducibility(42)
    print("   ✓ Seeds configured")
    print("   ✓ CUDA deterministic mode enabled")
    
    # Test 2: Model wrapper initialization
    print("2. Testing model wrapper...")
    try:
        # Note: This would use actual model path in production
        wrapper = LocalModelWrapper(
            model_path="/fake/path/to/model",  # Placeholder
            device="cpu"  # Force CPU for testing
            seed=42
        )
        print("   ✓ Model wrapper initialized")
        print(f"   ✓ Device: {wrapper.device}")
        print(f"   ✓ Seed: {wrapper.seed}")
        
        # Test model info
        info = wrapper.get_model_info()
        print("   ✓ Model info retrieved")
        print(f"   ✓ Reproducibility config: {info['reproducibility_config']}")
        
    except Exception as e:
        print(f"   ✗ Model wrapper failed: {e}")
        return False
    
    # Test 3: Logging system
    print("3. Testing logging system...")
    try:
        logger = TaskLogger("./test_logs")
        print("   ✓ Task logger initialized")
        
        # Test logging
        logger.start_task("test_001", "Test prompt", info)
        logger.log_generation("Test output", "Test action")
        logger.complete_task(0.85)
        print("   ✓ Logging system functional")
        
        # Check log file created
        log_file = Path("./test_logs/test_001.json")
        if log_file.exists():
            print("   ✓ Log file created")
            print(f"   ✓ Log file: {log_file}")
        else:
            print("   ✗ Log file not found")
            return False
            
    except Exception as e:
        print(f"   ✗ Logging system failed: {e}")
        return False
    
    print("\nIntegration Test Results:")
    print("=" * 40)
    print("✓ Reproducibility system: PASS")
    print("✓ Model wrapper: PASS") 
    print("✓ Logging system: PASS")
    print("✓ All components: READY")
    
    print("\nTo use with real model:")
    print("python evaluation_runner.py --model-path /path/to/model --dataset /path/to/data")
    
    return True


def main():
    """Main test entry point."""
    success = test_integration()
    
    if success:
        print("\n🎉 INTEGRATION TEST PASSED")
        print("System is ready for production evaluation!")
        sys.exit(0)
    else:
        print("\n❌ INTEGRATION TEST FAILED")
        print("Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
