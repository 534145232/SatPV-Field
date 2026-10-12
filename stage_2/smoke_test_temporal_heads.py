"""Verify the output shape of each Ratio3DCNN temporal head."""

import torch

from train_3dcnn_ratio_swr import Ratio3DCNN


def main() -> None:
    x = torch.randn(2, 4, 6, 32, 40)
    expected = (2, 3, 32, 40)
    for temporal_head in ("last_mean", "last_only", "flatten", "conv3d_decoder"):
        model = Ratio3DCNN(
            in_channels=6,
            future_frames=3,
            history_frames=4,
            width=8,
            depth=1,
            dropout=0.05,
            ratio_clip=1.6,
            temporal_head=temporal_head,
        )
        with torch.no_grad():
            y = model(x)
        actual = tuple(y.shape)
        if actual != expected:
            raise AssertionError(
                f"{temporal_head}: expected output shape {expected}, got {actual}"
            )
        print(f"{temporal_head}: {actual} OK")


if __name__ == "__main__":
    main()
