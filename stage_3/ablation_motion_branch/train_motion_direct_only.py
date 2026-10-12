#!/usr/bin/env python3
"""PyCharm entry for the direct-only motion-branch ablation."""
import sys
from train_motion_branch import main

if __name__ == "__main__":
    sys.argv.extend(["--motion_branch", "direct_only"])
    main()
