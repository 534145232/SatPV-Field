"""PyCharm-friendly launcher for Stage 1 5x5 patch-augmented ratio 3D CNN."""

import sys


from train_3dcnn_ratio_swr import main  # noqa: E402


if __name__ == "__main__":
    if "--train_mode" not in sys.argv:
        sys.argv.extend(["--train_mode", "big_patch_5x5"])
    if "--temporal_head" not in sys.argv:
        sys.argv.extend(["--temporal_head", "conv3d_decoder"])
    main()
