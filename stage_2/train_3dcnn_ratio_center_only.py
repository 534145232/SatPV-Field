"""PyCharm-friendly launcher for the Stage 1 center-only ratio 3D CNN."""

import sys


from train_3dcnn_ratio_swr import main  # noqa: E402


if __name__ == "__main__":
    if "--train_mode" not in sys.argv:
        sys.argv.extend(["--train_mode", "center_only"])
    main()
