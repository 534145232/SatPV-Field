"""PyCharm-friendly residual ConvLSTM V1 launcher."""

import sys

from train_residual_convlstm_correction_swr import main


if __name__ == "__main__":
    sys.argv.extend(["--loss_variant", "v1_no_gradient"])
    main()
