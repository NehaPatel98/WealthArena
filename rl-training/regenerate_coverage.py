#!/usr/bin/env python3
"""
Regenerate coverage.xml with correct paths for SonarQube
This script fixes the 0% coverage issue by ensuring paths are generated correctly.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_step(step, total, text):
    """Print step information"""
    print(f"[{step}/{total}] {text}")


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{description}")
    print("-" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=False)
    print("-" * 60)
    return result.returncode == 0


def main():
    print_header("Regenerating Coverage for SonarQube")
    
    # Step 1: Verify directory
    print_step(1, 4, "Verifying directory...")
    
    if not Path("pytest.ini").exists():
        print("\n❌ ERROR: pytest.ini not found!")
        print("Please run this script from the wealtharena_rl directory.")
        print(f"Current directory: {os.getcwd()}")
        return 1
    
    print(f"✅ Current directory: {os.getcwd()}")
    
    # Step 2: Delete old coverage file
    print_step(2, 4, "Deleting old coverage.xml...")
    
    coverage_file = Path("coverage.xml")
    if coverage_file.exists():
        coverage_file.unlink()
        print("✅ Old coverage.xml deleted.")
    else:
        print("✅ No old coverage.xml found (OK).")
    
    # Step 3: Install dependencies
    print_step(3, 4, "Installing dependencies...")
    
    if not run_command(
        "pip install pytest pytest-cov --quiet",
        "Installing pytest and pytest-cov..."
    ):
        print("\n⚠️  Warning: Could not install dependencies (may already be installed).")
    else:
        print("✅ Dependencies ready.")
    
    # Step 4: Generate coverage
    print_step(4, 4, "Generating coverage report...")
    
    success = run_command(
        "pytest --cov=. --cov-report=xml --cov-report=term",
        "Running pytest with coverage..."
    )
    
    # Verify results
    print("\n")
    
    if not success:
        print_header("⚠️  WARNING: Tests Failed or Had Issues")
        print("Coverage may still have been generated.")
        print("Check the output above for details.\n")
    
    if coverage_file.exists():
        print_header("✅ SUCCESS! Coverage Generated")
        print(f"Coverage file: {coverage_file.absolute()}\n")
        print("Next steps:")
        print("1. Verify paths in coverage.xml look correct:")
        print("   Windows: findstr /C:\"filename=\" coverage.xml")
        print("   Mac/Linux: grep 'filename=' coverage.xml | head -5")
        print("\n2. Run SonarQube scan")
        print("\n3. Check SonarQube dashboard for coverage percentage")
        print("\nFor more details, see: SONARQUBE_FIX.md\n")
        return 0
    else:
        print_header("❌ ERROR: Coverage Generation Failed")
        print("coverage.xml was not created.\n")
        print("Possible issues:")
        print("- Tests failed (see output above)")
        print("- pytest-cov not installed correctly")
        print("- Configuration issue in pytest.ini or .coveragerc")
        print("\nTry running manually:")
        print("  pytest --cov=. --cov-report=xml -v\n")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

