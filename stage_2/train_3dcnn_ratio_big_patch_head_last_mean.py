"""Big-patch baseline: temporal head = last + mean."""

import sys

from train_3dcnn_ratio_swr import main


if __name__ == "__main__":
    sys.argv.extend(["--train_mode", "big_patch_5x5", "--temporal_head", "last_mean"])
    main()
