#!/usr/bin/env python3
"""Audit the repository for common public-release mistakes.

This script intentionally uses only the Python standard library so it can run
in a clean GitHub Actions checkout before scientific dependencies are installed.
"""

from __future__ import annotations

import argparse
import compileall
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".gitignore",
    ".gitattributes",
}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".nc",
    ".csv",
    ".tsv",
    ".parquet",
    ".xlsx",
    ".xls",
    ".h5",
    ".hdf5",
    ".tif",
    ".tiff",
    ".pkl",
    ".joblib",
}
FORBIDDEN_PATH_PATTERNS = {
    "former repository path": re.compile(r"/(?:code|workspace)/Final_3_stage_pipeline"),
    "former legacy-project path": re.compile(r"/code/Final_Project_RegionalSplit"),
    "former server storage path": re.compile(r"/(?:user-storage|2301014)[^\s\"']*"),
    "former dataset owner path": re.compile(r"/data/DayinChen"),
    "Windows user-profile path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"),
    "Bitahub host identifier": re.compile(r"bitahub-[A-Za-z0-9-]+"),
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub classic token": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    "GitHub fine-grained token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
        return [ROOT / name for name in names]
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def is_text_file(path: Path) -> bool:
    return path.name in {".gitignore", ".gitattributes"} or path.suffix.lower() in TEXT_SUFFIXES


def read_text(path: Path, failures: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        failures.append(f"non-UTF-8 text file: {path.relative_to(ROOT)} ({exc})")
        return None


def check_artifacts(files: list[Path], failures: list[str]) -> None:
    for path in files:
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            failures.append(f"tracked data/model artifact: {path.relative_to(ROOT)}")


def check_text(files: list[Path], failures: list[str]) -> None:
    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        if not is_text_file(path):
            continue
        text = read_text(path, failures)
        if text is None:
            continue
        for label, pattern in FORBIDDEN_PATH_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(ROOT)}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"possible {label}: {path.relative_to(ROOT)}")


def check_markdown_links(files: list[Path], failures: list[str]) -> None:
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        text = read_text(path, failures)
        if text is None:
            continue
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            # Markdown titles after a path are outside this repository's style;
            # treating the complete target as a path catches accidental breakage.
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(
                    f"broken Markdown link in {path.relative_to(ROOT)}: {raw_target}"
                )


def check_python_syntax(failures: list[str]) -> None:
    roots = [
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
        "experiments",
        "paper_support",
        "Framework_draw",
        "scripts",
    ]
    for name in roots:
        path = ROOT / name
        if path.exists() and not compileall.compile_dir(path, quiet=1):
            failures.append(f"Python compilation failed under: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    files = tracked_files()
    failures: list[str] = []
    check_artifacts(files, failures)
    check_text(files, failures)
    check_markdown_links(files, failures)
    if not args.skip_compile:
        check_python_syntax(failures)

    if failures:
        print("Public-release audit FAILED:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        raise SystemExit(1)

    print(f"Public-release audit passed ({len(files)} tracked files checked).")


if __name__ == "__main__":
    main()
