"""Print the Stage 1 design without reading raw H8 files."""

from stage1_paths import (
    BIG_FRAME_CACHE_DIR,
    BIG_INDEX_MAIN,
    BIG_MONTHLY_DIR,
    CENTER_FRAME_CACHE_DIR,
    CENTER_INDEX_ALL,
    CENTER_MONTHLY_DIR,
    CENTER_SAMPLE_CACHE_DIR,
)


def main() -> None:
    print("Stage 1 forecast definition")
    print("  history frames: 4")
    print("  future frames : 3")
    print("  valid BJT hours: 06:00-17:00 inclusive")
    print("  data time start: 2022-01-01 07:00:00")
    print("  data time end  : 2023-06-24 16:45:00")
    print("  train end      : 2023-01-13 23:59:59")
    print("  val end        : 2023-03-08 23:59:59")
    print("\nTemporal windows per complete day")
    for start in range(6, 12):
        history = ",".join(f"{hour:02d}" for hour in range(start, start + 4))
        future = ",".join(f"{hour:02d}" for hour in range(start + 4, start + 7))
        print(f"  {history} -> {future}")

    print("\nBig source and center crop")
    print("  big   : lat 18-42, lon 105.5-137.0")
    print("  center: lat 26-34, lon 116.0-126.5")

    print("\nFixed 5x5 patch bank with 50 percent overlap")
    lat_starts = [18, 22, 26, 30, 34]
    lon_starts = [105.5, 110.75, 116.0, 121.25, 126.5]
    for row, lat_min in enumerate(lat_starts):
        for col, lon_min in enumerate(lon_starts):
            label = " CENTER" if row == 2 and col == 2 else ""
            print(
                f"  r{row}c{col}: lat {lat_min}-{lat_min + 8}, "
                f"lon {lon_min}-{lon_min + 10.5}{label}"
            )

    print("\nStage 1 outputs")
    for label, path in (
        ("big monthly", BIG_MONTHLY_DIR),
        ("center monthly", CENTER_MONTHLY_DIR),
        ("big index", BIG_INDEX_MAIN),
        ("center index", CENTER_INDEX_ALL),
        ("big frame cache", BIG_FRAME_CACHE_DIR),
        ("center frame cache", CENTER_FRAME_CACHE_DIR),
        ("center sample cache", CENTER_SAMPLE_CACHE_DIR),
    ):
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
