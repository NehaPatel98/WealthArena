#!/usr/bin/env python3
"""
Convenience script to run tests with coverage for SonarQube
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run pytest with coverage"""
    
    print("=" * 60)
    print("Running Tests with Coverage for SonarQube")
    print("=" * 60)
    print()
    
    # Check if pytest and pytest-cov are installed
    try:
        import pytest
        import pytest_cov
        print("✅ pytest and pytest-cov are installed")
    except ImportError as e:
        print("❌ Missing dependencies. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
        print("✅ Dependencies installed")
    
    print()
    print("Running tests with coverage...")
    print()
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=.",
        "--cov-report=xml",
        "--cov-report=term",
        "-v"
    ]
    
    result = subprocess.run(cmd)
    
    print()
    print("=" * 60)
    
    if result.returncode == 0:
        print("✅ Tests passed!")
        
        # Check if coverage.xml was generated
        if Path("coverage.xml").exists():
            print("✅ coverage.xml generated successfully")
            print()
            print("📁 File location:", Path("coverage.xml").absolute())
            print()
            print("Next steps:")
            print("1. Upload coverage.xml to SonarQube")
            print("2. Commit your code (but NOT coverage.xml)")
        else:
            print("⚠️  Warning: coverage.xml not found")
    else:
        print("❌ Some tests failed")
        print()
        print("Please fix failing tests before generating coverage report")
    
    print("=" * 60)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())

