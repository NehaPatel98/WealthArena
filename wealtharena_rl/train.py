#!/usr/bin/env python3
"""
WealthArena Trading System - Main Training Script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from training.train_agents import main

if __name__ == "__main__":
    main()
