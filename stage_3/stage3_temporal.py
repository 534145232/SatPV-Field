"""Leakage-safe Stage 3 temporal protocol shared by all cache builders."""

from __future__ import annotations

import argparse

import pandas as pd


PV_HISTORY_FRAMES = 12
PV_FUTURE_FRAMES = 8
PV_FREQ_MINUTES = 15
PV_ISSUE_STRIDE_MINUTES = 60
SWR_HISTORY_FRAMES = 4
SWR_FUTURE_FRAMES = 3


def is_pv_source_daytime(timestamp: pd.Timestamp) -> bool:
    timestamp = pd.Timestamp(timestamp)
    minute = int(timestamp.hour) * 60 + int(timestamp.minute)
    return 7 * 60 <= minute <= 16 * 60 + 45


def build_pv_issue_time_table(args: argparse.Namespace) -> pd.DataFrame:
    """Build hourly PV samples: history excludes T and targets start exactly at T."""
    step = pd.Timedelta(minutes=int(args.freq_minutes))
    issue_step = pd.Timedelta(minutes=int(args.issue_stride_minutes))
    start = pd.Timestamp(args.time_start)
    end = pd.Timestamp(args.time_end)
    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    rows: list[dict[str, object]] = []
    for issue in pd.date_range(start, end, freq=issue_step):
        if int(args.issue_stride_minutes) == 60 and issue.minute != 0:
            continue
        past = [issue - step * (int(args.past_frames) - i) for i in range(int(args.past_frames))]
        future = [issue + step * i for i in range(int(args.future_frames))]
        all_times = past + future
        if not all(is_pv_source_daytime(timestamp) for timestamp in all_times):
            continue
        if past[0].date() != issue.date() or future[-1].date() != issue.date():
            continue
        split = "train" if issue <= train_end else "val" if issue <= val_end else "test"
        rows.append(
            {
                "issue_time": issue,
                "issue_sec": int(issue.timestamp()),
                "time_split": split,
                "past_start": past[0].strftime("%Y-%m-%d %H:%M:%S"),
                "past_end": past[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "future_start": future[0].strftime("%Y-%m-%d %H:%M:%S"),
                "future_end": future[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "past_times": "|".join(timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in past),
                "future_times": "|".join(timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in future),
            }
        )
    result = pd.DataFrame(rows)
    assert_pv_issue_time_table(result, args)
    return result


def swr_history_times(issue: pd.Timestamp, frame_count: int = SWR_HISTORY_FRAMES) -> list[pd.Timestamp]:
    issue = pd.Timestamp(issue).floor("h")
    return [issue - pd.Timedelta(hours=frame_count - index) for index in range(frame_count)]


def swr_future_times(issue: pd.Timestamp, frame_count: int = SWR_FUTURE_FRAMES) -> list[pd.Timestamp]:
    issue = pd.Timestamp(issue).floor("h")
    return [issue + pd.Timedelta(hours=index) for index in range(frame_count)]


def assert_pv_issue_time_table(table: pd.DataFrame, args: argparse.Namespace) -> None:
    if table.empty:
        raise RuntimeError("Stage 3 PV issue-time table is empty.")
    expected_hours = {10, 11, 12, 13, 14, 15}
    actual_hours = set(pd.to_datetime(table["issue_time"]).dt.hour.unique().tolist())
    if actual_hours != expected_hours:
        raise AssertionError(f"Expected PV issue hours {sorted(expected_hours)}, found {sorted(actual_hours)}")
    for _, row in table.iterrows():
        issue = pd.Timestamp(row["issue_time"])
        past = [pd.Timestamp(value) for value in str(row["past_times"]).split("|")]
        future = [pd.Timestamp(value) for value in str(row["future_times"]).split("|")]
        if len(past) != int(args.past_frames) or len(future) != int(args.future_frames):
            raise AssertionError(f"PV frame-count mismatch at {issue}")
        if past[-1] != issue - pd.Timedelta(minutes=int(args.freq_minutes)):
            raise AssertionError(f"PV history must end at T-15min: {issue}")
        if future[0] != issue:
            raise AssertionError(f"PV target must start at T: {issue}")
        if not all(is_pv_source_daytime(timestamp) for timestamp in past + future):
            raise AssertionError(f"PV sample outside 07:00-16:45: {issue}")


def assert_swr_alignment(
    issue: pd.Timestamp,
    history: list[pd.Timestamp],
    future: list[pd.Timestamp],
) -> None:
    issue = pd.Timestamp(issue).floor("h")
    expected_history = swr_history_times(issue)
    expected_future = swr_future_times(issue)
    if [pd.Timestamp(value) for value in history] != expected_history:
        raise AssertionError(f"SWR history mismatch at PV issue {issue}: {history} != {expected_history}")
    if [pd.Timestamp(value) for value in future] != expected_future:
        raise AssertionError(f"SWR future mismatch at PV issue {issue}: {future} != {expected_future}")
