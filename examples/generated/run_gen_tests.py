#!/usr/bin/env python3
"""
Run all generated tests for cutile transpiler.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "cutile_typeviz" / "cutile_utils"))


def run_all_tests():
    """Run all test modules and report results."""
    print("=" * 70)
    print("Running Generated Tests for CuTile Transpiler")
    print("=" * 70)

    tests = [
        ("test_vector_add", "Vector Add"),
        ("test_silu", "SiLU Activation"),
        ("test_gelu", "GELU Activation"),
        ("test_reduce", "Reduction Operations"),
        ("test_softmax", "Softmax"),
        ("test_rmsnorm", "RMSNorm"),
        ("test_matmul", "Matrix Multiplication"),
        ("test_num_blocks", "Num Blocks"),
    ]

    results = {}

    for module_name, test_name in tests:
        print(f"\n{'#' * 70}")
        print(f"# {test_name}")
        print(f"{'#' * 70}\n")

        try:
            # Import and run the test module
            module = __import__(module_name)
            passed = module.run_test()
            results[test_name] = passed
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for test_name, result in results.items():
        status = "PASSED ✓" if result else "FAILED ✗"
        print(f"  {test_name}: {status}")

    print("-" * 70)
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
