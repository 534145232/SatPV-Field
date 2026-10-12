"""PyCharm-friendly residual ConvLSTM V2 launcher."""

import sys

from train_residual_convlstm_correction_swr import main


if __name__ == "__main__":
    sys.argv.extend(["--loss_variant", "v2_no_center_weight"])
    main()
