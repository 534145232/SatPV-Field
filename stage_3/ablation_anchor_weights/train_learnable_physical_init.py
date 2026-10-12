#!/usr/bin/env python3
"""Train learnable anchor weights initialized from the hand-designed table."""

import sys

from train_learnable_anchor_weights import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--anchor_weight_init", "physical"]
    main()
