#!/usr/bin/env python3
"""PyCharm entry for the complete motion-fusion branch."""
import sys
from train_motion_branch import main

if __name__ == "__main__":
    sys.argv.extend(["--motion_branch", "warp_direct_gate"])
    main()
