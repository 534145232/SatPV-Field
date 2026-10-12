"""PyCharm-friendly residual ConvLSTM V3 launcher."""

import sys

from train_residual_convlstm_correction_swr import main


if __name__ == "__main__":
    sys.argv.extend(["--loss_variant", "v3_no_residual_l1"])
    main()
