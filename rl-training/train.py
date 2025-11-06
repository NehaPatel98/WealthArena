#!/usr/bin/env python3
"""
WealthArena Trading System - Main Training Script
"""

import sys
from pathlib import Path

# Insert src directory at position 0 for highest priority
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and verify the correct main function
try:
    from training.train_agents import main
    
    # Verify we imported the correct main function
    if not hasattr(main, '__module__') or main.__module__ != 'training.train_agents':
        raise ImportError(
            "Module import conflict: train.py imported wrong main() function. "
            "Expected main.__module__ == 'training.train_agents' but got "
            f"main.__module__ == '{getattr(main, '__module__', 'unknown')}'. "
            "Check that collect_training_data.py is properly renamed and train.py "
            "correctly imports from training.train_agents."
        )
    
except ImportError as e:
    print(f"Failed to import training module: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting WealthArena RL Training...")
    print(f"Imported main from module: {main.__module__}")
    
    try:
        main()
        print("[OK] WealthArena RL Training completed successfully!")
        sys.exit(0)
    except ImportError as e:
        print(f"[FAILED] Import error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except ValueError as e:
        print(f"[FAILED] Value error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except RuntimeError as e:
        print(f"[FAILED] Runtime error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"[FAILED] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
