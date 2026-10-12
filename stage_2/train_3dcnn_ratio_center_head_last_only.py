"""Center-only ablation: temporal head = last frame only."""

import sys

from train_3dcnn_ratio_swr import main


if __name__ == "__main__":
    sys.argv.extend(["--train_mode", "center_only", "--temporal_head", "last_only"])
    main()
