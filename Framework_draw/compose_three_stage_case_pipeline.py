#!/usr/bin/env python3
"""Compose LCIG, SAG-PVFF, and FG-MSF case figures into one pipeline panel.

The script does not rerun any model. It reads already-rendered case figures,
standardizes their whitespace and size, and connects them with arrows for a
compact paper-ready overview of the E2F2S workflow.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
CASE_TAG = "idx00529_20230608_120000"
DEFAULT_OUT_DIR = Path("D:/PV_pipeline_figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcig_image", type=Path, default=None, help="Rendered LCIG/SWR-anchor case figure.")
    parser.add_argument("--field_image", type=Path, default=None, help="Rendered regional PV-field case figure.")
    parser.add_argument("--station_image", type=Path, default=None, help="Rendered station-level prediction curve figure.")
    parser.add_argument("--case_tag", type=str, default=CASE_TAG)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_prefix", type=str, default="e2f2s_case_pipeline")
    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--pdf", type=int, default=1)
    parser.add_argument("--png", type=int, default=1)
    parser.add_argument("--svg", type=int, default=0, help="Reserved for compatibility; this raster-composition script writes PNG/PDF.")
    parser.add_argument("--trim", type=int, default=1)
    parser.add_argument("--pad_px", type=int, default=10)
    parser.add_argument("--fig_width", type=float, default=13.6)
    parser.add_argument("--fig_height", type=float, default=4.25)
    parser.add_argument("--left_width", type=float, default=0.34)
    parser.add_argument("--middle_width", type=float, default=0.39)
    parser.add_argument("--right_width", type=float, default=0.20)
    parser.add_argument("--arrow_gap", type=float, default=0.022)
    return parser.parse_args()


def normalized_candidates(path: Path | None, fallback: list[Path]) -> list[Path]:
    if path is not None:
        return [Path(path)]
    return fallback


def default_lcig_candidates(case_tag: str) -> list[Path]:
    return [
        PIPELINE_DIR / "stage_2" / "paper_figures" / "swr_forecast_cases" / f"lcig_swr_anchor_01_{case_tag}.png",
        PIPELINE_DIR / "stage_2" / "paper_figures" / "swr_forecast_cases" / f"lcig_swr_anchor_01_{case_tag}.pdf",
        PIPELINE_DIR / "stage_2" / "paper_figures" / "swr_forecast_cases" / f"case_01_{case_tag}.png",
        PIPELINE_DIR / "stage_2" / "paper_figures" / "swr_forecast_cases" / f"case_01_{case_tag}.pdf",
    ]


def default_field_candidates(case_tag: str) -> list[Path]:
    return [
        PIPELINE_DIR / "stage_3" / "paper_figures" / "pv_field_case_study" / f"case_01_{case_tag}.png",
        PIPELINE_DIR / "stage_3" / "paper_figures" / "pv_field_case_study" / f"case_01_{case_tag}.pdf",
    ]


def default_station_candidates() -> list[Path]:
    return [
        PIPELINE_DIR
        / "stage_4"
        / "runs"
        / "13_stage4_case_comparisons"
        / "stage4_case_01.png",
        SCRIPT_DIR / "stage4_station_prediction_curve_assets" / "sample_05_20230608_120000" / "station_prediction_curves.png",
        SCRIPT_DIR / "stage4_station_prediction_curve_assets" / "sample_05_20230608_120000" / "station_prediction_curves.svg",
    ]


def resolve_existing(candidates: list[Path], label: str) -> Path:
    tried = []
    for path in candidates:
        path = Path(path)
        tried.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {label} image. Tried:\n" + "\n".join(tried))


def load_pdf(path: Path, dpi: int) -> Image.Image:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"PDF input requires PyMuPDF/fitz: {path}") from exc
    doc = fitz.open(str(path))
    try:
        page = doc.load_page(0)
        scale = float(dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


def load_svg(path: Path, dpi: int) -> Image.Image:
    try:
        import cairosvg  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"SVG input requires cairosvg: {path}") from exc
    png = cairosvg.svg2png(url=str(path), dpi=int(dpi))
    return Image.open(io.BytesIO(png)).convert("RGB")


def load_image(path: Path, dpi: int) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, dpi)
    if suffix == ".svg":
        return load_svg(path, dpi)
    return Image.open(path).convert("RGB")


def trim_white_border(image: Image.Image, pad: int = 10, threshold: int = 246) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < int(threshold), axis=2)
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(arr.shape[0], int(ys.max()) + int(pad) + 1)
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(arr.shape[1], int(xs.max()) + int(pad) + 1)
    return image.crop((x0, y0, x1, y1))


def image_to_image(path: Path, dpi: int, trim: bool, pad_px: int) -> Image.Image:
    image = load_image(path, dpi)
    if trim:
        image = trim_white_border(image, pad=int(pad_px))
    return image.convert("RGB")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return int(box[2] - box[0]), int(box[3] - box[1])


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str,
    max_width: int | None = None,
) -> None:
    x, y = xy
    if max_width is None:
        w, _ = text_size(draw, text, fnt)
        draw.text((x - w // 2, y), text, font=fnt, fill=fill)
        return
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if text_size(draw, test, fnt)[0] <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = max(text_size(draw, "Ag", fnt)[1] + 6, int(getattr(fnt, "size", 12) * 1.15))
    for i, line in enumerate(lines):
        w, _ = text_size(draw, line, fnt)
        draw.text((x - w // 2, y + i * line_h), line, font=fnt, fill=fill)


def fit_inside(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(float(width) / max(float(image.width), 1.0), float(height) / max(float(image.height), 1.0))
    new_w = max(1, int(round(float(image.width) * scale)))
    new_h = max(1, int(round(float(image.height) * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def draw_stage_box(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    label: str,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    border = "#cbd5e1"
    draw.rounded_rectangle((x0, y0, x1, y1), radius=12, outline=border, width=2, fill="#ffffff")
    draw.text((x0 + 14, y0 + 12), label, font=fonts["label"], fill="#9b111e")
    draw_centered_text(draw, ((x0 + x1) // 2, y0 + 9), title, fonts["title"], "#0f172a")
    draw_centered_text(draw, ((x0 + x1) // 2, y0 + 50), subtitle, fonts["subtitle"], "#475569", max_width=x1 - x0 - 46)

    image_top = y0 + 115
    image_bottom = y1 - 18
    image_left = x0 + 14
    image_right = x1 - 14
    fitted = fit_inside(image, image_right - image_left, image_bottom - image_top)
    px = image_left + (image_right - image_left - fitted.width) // 2
    py = image_top + (image_bottom - image_top - fitted.height) // 2
    canvas.paste(fitted, (px, py))


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
) -> None:
    x0, y0 = start
    x1, y1 = end
    color = "#334155"
    draw.line((x0, y0, x1, y1), fill=color, width=5)
    head = 20
    draw.polygon([(x1, y1), (x1 - head, y1 - head // 2), (x1 - head, y1 + head // 2)], fill=color)
    w, h = text_size(draw, text, fnt)
    tx = (x0 + x1) // 2 - w // 2
    ty = y0 + 26
    draw.rounded_rectangle((tx - 7, ty - 4, tx + w + 7, ty + h + 6), radius=5, fill="#ffffff")
    draw.text((tx, ty), text, font=fnt, fill="#475569")


def compose(args: argparse.Namespace) -> dict:
    lcig_path = resolve_existing(
        normalized_candidates(args.lcig_image, default_lcig_candidates(str(args.case_tag))),
        "LCIG",
    )
    field_path = resolve_existing(
        normalized_candidates(args.field_image, default_field_candidates(str(args.case_tag))),
        "PV-field",
    )
    station_path = resolve_existing(
        normalized_candidates(args.station_image, default_station_candidates()),
        "station-level",
    )

    images = [
        image_to_image(lcig_path, int(args.dpi), bool(args.trim), int(args.pad_px)),
        image_to_image(field_path, int(args.dpi), bool(args.trim), int(args.pad_px)),
        image_to_image(station_path, int(args.dpi), bool(args.trim), int(args.pad_px)),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(args.dpi)
    width_px = int(round(float(args.fig_width) * dpi))
    height_px = int(round(float(args.fig_height) * dpi))
    canvas = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "label": font(max(18, int(0.031 * dpi)), bold=True),
        "title": font(max(22, int(0.038 * dpi)), bold=True),
        "subtitle": font(max(16, int(0.025 * dpi))),
        "arrow": font(max(15, int(0.023 * dpi))),
        "caption": font(max(15, int(0.024 * dpi))),
    }

    margin_l = int(0.035 * width_px)
    margin_r = int(0.030 * width_px)
    gap_px = int(float(args.arrow_gap) * width_px)
    y0 = int(0.12 * height_px)
    y1 = int(0.83 * height_px)
    available_w = width_px - margin_l - margin_r - 2 * gap_px
    fractions = np.asarray([float(args.left_width), float(args.middle_width), float(args.right_width)], dtype=np.float64)
    fractions = fractions / max(float(fractions.sum()), 1.0e-9)
    panel_ws = [int(round(float(frac) * available_w)) for frac in fractions]
    panel_ws[-1] = available_w - sum(panel_ws[:-1])

    boxes = []
    x = margin_l
    for panel_w in panel_ws:
        boxes.append((x, y0, x + panel_w, y1))
        x += panel_w + gap_px

    draw_stage_box(canvas, draw, images[0], boxes[0], "Stage 1  LCIG", "Large-context SWR forecasting and target-region anchor extraction", "(a)", fonts)
    draw_stage_box(canvas, draw, images[1], boxes[1], "Stage 2  SAG-PVFF", "SWR-anchor-guided regional PV-field forecasting", "(b)", fonts)
    draw_stage_box(canvas, draw, images[2], boxes[2], "Stage 3  FG-MSF", "Field-guided multimodal station-level forecasting", "(c)", fonts)

    arrow_y = (y0 + y1) // 2
    draw_arrow(draw, (boxes[0][2] + 8, arrow_y), (boxes[1][0] - 8, arrow_y), "future SWR anchors", fonts["arrow"])
    draw_arrow(draw, (boxes[1][2] + 8, arrow_y), (boxes[2][0] - 8, arrow_y), "local PV-field patches", fonts["arrow"])

    caption = (
        "Case: 2023-06-08 12:00 BJT. Large-context irradiance guidance is converted into regional "
        "PV-field forecasts and then into target-station predictions."
    )
    draw_centered_text(draw, (width_px // 2, int(0.89 * height_px)), caption, fonts["caption"], "#334155", max_width=int(0.88 * width_px))

    stem = f"{str(args.out_prefix).strip() or 'e2f2s_case_pipeline'}_{str(args.case_tag)}"
    outputs: dict[str, str] = {}
    if int(args.png):
        path = out_dir / f"{stem}.png"
        canvas.save(path)
        outputs["png"] = str(path)
    if int(args.pdf):
        path = out_dir / f"{stem}.pdf"
        canvas.save(path, "PDF", resolution=float(args.dpi))
        outputs["pdf"] = str(path)
    if int(args.svg):
        print("[warn] SVG output is skipped because the composed figure embeds raster case panels.", flush=True)

    meta = {
        "case_tag": str(args.case_tag),
        "inputs": {
            "lcig": str(lcig_path),
            "field": str(field_path),
            "station": str(station_path),
        },
        "outputs": outputs,
    }
    meta_path = out_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["meta"] = str(meta_path)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = compose(args)
    print("==== Three-stage case pipeline figure ====")
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
