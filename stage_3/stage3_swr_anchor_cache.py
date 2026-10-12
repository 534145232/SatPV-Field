"""Versioned Stage 3 future-SWR anchor-cache helpers."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from stage3_paths import CENTER_H8_SWR_ANCHOR_ACTIVE_MANIFEST, CENTER_H8_SWR_ANCHOR_CACHE_ROOT


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return value or "swr_checkpoint"


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage 2 SWR checkpoint not found: {checkpoint}")
    stat = checkpoint.stat()
    raw_parts = [f"{checkpoint}|{stat.st_size}|{stat.st_mtime_ns}"]
    identity: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
    }
    if checkpoint.suffix.lower() in {".pt", ".pth"}:
        try:
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            identity["model_name"] = str(payload.get("model_name", ""))
            if str(payload.get("loss_variant", "")).strip():
                identity["loss_variant"] = str(payload["loss_variant"])
            if str(payload.get("base_temporal_head", "")).strip():
                identity["base_temporal_head"] = str(payload["base_temporal_head"])
            base_path_text = str(payload.get("base_checkpoint", "")).strip()
            if base_path_text:
                base_path = Path(base_path_text).expanduser().resolve()
                identity["base_checkpoint_path"] = str(base_path)
                if base_path.exists():
                    base_stat = base_path.stat()
                    identity["base_checkpoint_size_bytes"] = int(base_stat.st_size)
                    identity["base_checkpoint_mtime_ns"] = int(base_stat.st_mtime_ns)
                    raw_parts.append(f"{base_path}|{base_stat.st_size}|{base_stat.st_mtime_ns}")
                else:
                    identity["base_checkpoint_missing"] = True
                    raw_parts.append(f"{base_path}|missing")
        except (ImportError, RuntimeError, ValueError):
            # The Stage 3 model loader will emit a detailed error if this file
            # is not a valid supported checkpoint.
            pass
    identity["checkpoint_signature"] = hashlib.sha1("|".join(raw_parts).encode("utf-8")).hexdigest()[:12]
    return identity


def cache_dir_for_checkpoint(
    checkpoint: Path,
    cache_name: str = "",
    cache_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    identity = checkpoint_identity(checkpoint)
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve(), identity
    label = cache_name.strip() or f"{Path(checkpoint).parent.name}_{identity['checkpoint_signature']}"
    return CENTER_H8_SWR_ANCHOR_CACHE_ROOT / _slug(label), identity


def read_cache_info(cache_dir: Path) -> dict[str, Any]:
    path = Path(cache_dir) / "center_h8_swr_anchor_input_cache_meta.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_matches_checkpoint(cache_dir: Path, identity: dict[str, Any]) -> bool:
    info = read_cache_info(cache_dir)
    return (
        str(info.get("stage3_swr_cache_version", {}).get("checkpoint_signature", ""))
        == str(identity["checkpoint_signature"])
    )


def assert_cache_matches_checkpoint(cache_dir: Path, checkpoint: Path) -> None:
    identity = checkpoint_identity(checkpoint)
    if cache_matches_checkpoint(cache_dir, identity):
        return

    # A copied checkpoint receives a new absolute path (and sometimes mtime),
    # while its model bytes remain unchanged.  Older cache signatures included
    # both of those machine-local values.  Accept this relocation only when the
    # recorded file size/name and any recorded model descriptors still agree.
    info = read_cache_info(cache_dir)
    recorded = info.get("stage3_swr_cache_version", {})
    same_size = int(recorded.get("checkpoint_size_bytes", -1)) == int(identity["checkpoint_size_bytes"])
    same_name = Path(str(recorded.get("checkpoint_path", ""))).name == Path(checkpoint).name
    descriptor_keys = ("model_name", "loss_variant", "base_temporal_head")
    descriptors_match = all(
        not str(recorded.get(key, "")).strip()
        or str(recorded.get(key, "")) == str(identity.get(key, ""))
        for key in descriptor_keys
    )
    if same_size and same_name and descriptors_match:
        print(
            "Using relocated Stage 3 future-SWR cache; checkpoint content metadata matches "
            f"the copied checkpoint: {Path(checkpoint).resolve()}",
            flush=True,
        )
        return

    raise RuntimeError(
        "The selected Stage 3 future-SWR cache was not built from the requested checkpoint. "
        f"cache={Path(cache_dir).resolve()} requested_checkpoint={Path(checkpoint).resolve()}. "
        "Rebuild and activate it with run_stage3_build_future_swr_cache.sh --force 1."
    )


def activate_cache(cache_dir: Path, identity: dict[str, Any]) -> None:
    CENTER_H8_SWR_ANCHOR_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "Active Stage 3 future-SWR anchor cache selected for downstream PV-field training.",
        "active_cache_dir": str(Path(cache_dir).resolve()),
        **identity,
    }
    with CENTER_H8_SWR_ANCHOR_ACTIVE_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print("Activated Stage 3 future-SWR cache:", payload["active_cache_dir"], flush=True)


def resolve_active_cache_dir(required: bool = False) -> Path:
    if CENTER_H8_SWR_ANCHOR_ACTIVE_MANIFEST.exists():
        with CENTER_H8_SWR_ANCHOR_ACTIVE_MANIFEST.open("r", encoding="utf-8") as handle:
            cache_dir = Path(json.load(handle)["active_cache_dir"])
        if cache_dir.exists():
            return cache_dir
        # The cache tree may have been copied from another machine while the
        # manifest still contains the old absolute prefix.
        relocated = CENTER_H8_SWR_ANCHOR_CACHE_ROOT / cache_dir.name
        if relocated.exists():
            return relocated
    # Compatibility with the first unversioned Stage 3 implementation.
    if (CENTER_H8_SWR_ANCHOR_CACHE_ROOT / "swr_anchor_64.npy").exists():
        return CENTER_H8_SWR_ANCHOR_CACHE_ROOT
    discovered = sorted(CENTER_H8_SWR_ANCHOR_CACHE_ROOT.glob("*/swr_anchor_64.npy"))
    if len(discovered) == 1:
        return discovered[0].parent
    if required:
        raise FileNotFoundError(
            "No active Stage 3 future-SWR cache. Tried cache root: "
            f"{CENTER_H8_SWR_ANCHOR_CACHE_ROOT}. Set OLD_DATA_ROOT to the directory containing "
            "stage3/, or pass --swr_anchor_cache_dir explicitly."
        )
    return CENTER_H8_SWR_ANCHOR_CACHE_ROOT


def cli_path_or_none(flag: str) -> Path | None:
    values = sys.argv[1:]
    for index, value in enumerate(values):
        if value == flag and index + 1 < len(values):
            return Path(values[index + 1])
        if value.startswith(f"{flag}="):
            return Path(value.split("=", 1)[1])
    return None


def cli_path(flag: str, default: Path) -> Path:
    value = cli_path_or_none(flag)
    if value is not None:
        return value
    return Path(default)
