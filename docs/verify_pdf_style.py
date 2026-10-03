#!/usr/bin/env python3
"""Render and pixel-compare a CubeFit PDF with the frozen reference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def _render(pdf: Path, out_dir: Path, dpi: int) -> list[Path]:
    if not shutil.which("pdftoppm"):
        raise RuntimeError(
            "pdftoppm was not found. Install poppler-utils first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    _run([
        "pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)])
    return sorted(out_dir.glob("page-*.png"))


def _compare(a_path: Path, b_path: Path, diff_path: Path) -> dict:
    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    if a.size != b.size:
        return {
            "changed": True,
            "size_a": a.size,
            "size_b": b.size,
            "pct_changed": 100.0,
        }
    diff = ImageChops.difference(a, b)
    bbox = diff.getbbox()
    if bbox is None:
        return {"changed": False, "pct_changed": 0.0, "bbox": None}
    mask = diff.convert("L").point(lambda value: 255 if value else 0)
    total = a.size[0] * a.size[1]
    changed = total - mask.histogram()[0]
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff.save(diff_path)
    return {
        "changed": True,
        "pct_changed": 100.0 * changed / total,
        "bbox": list(bbox),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", nargs="?", default="docs/CubeFit.pdf")
    parser.add_argument(
        "--reference", default="docs/CubeFit.reference.pdf")
    parser.add_argument("--out-dir", default="docs/_pdf_style_diff")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    candidate = Path(args.candidate)
    reference = Path(args.reference)
    out_dir = Path(args.out_dir)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if not reference.is_file():
        raise FileNotFoundError(reference)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_pages = _render(reference, tmp / "reference", args.dpi)
        new_pages = _render(candidate, tmp / "candidate", args.dpi)
        count = min(len(ref_pages), len(new_pages))
        pages = []
        for index in range(count):
            result = _compare(
                ref_pages[index], new_pages[index],
                out_dir / f"page-{index + 1:02d}.png")
            result["page"] = index + 1
            pages.append(result)

    summary = {
        "reference": str(reference),
        "candidate": str(candidate),
        "dpi": args.dpi,
        "reference_pages": len(ref_pages),
        "candidate_pages": len(new_pages),
        "pages": pages,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(
        "Pixel differences are expected after content edits. Use this tool "
        "to detect unintended renderer/style changes on unchanged content.")


if __name__ == "__main__":
    main()
