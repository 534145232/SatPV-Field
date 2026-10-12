"""PyCharm-friendly residual ConvLSTM V0 launcher."""

import sys

from train_residual_convlstm_correction_swr import main


if __name__ == "__main__":
    sys.argv.extend(["--loss_variant", "v0_full"])
    main()
