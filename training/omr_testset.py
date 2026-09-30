"""Scores whose page starts are known exactly, to measure a bar counter on.

For each work in data/pdmx/plan.json, its first PDMX transcription held locally is engraved by
MuseScore twice over: as a PDF, and as a measure-position file (.mpos) that gives the page every
bar lands on. So the bar each page starts with is known without anyone counting. The pages are then
rendered with Apple's PDFKit (training/render_pages.swift), clean at 150 dpi, and again made to
look scanned: turned by up to a degree and a quarter, blurred, speckled and thresholded, as an old
scan on IMSLP is.

  python training/omr_testset.py [--limit 60]      # results/omr/<work>/{score.pdf, truth.json, clean/, scan/}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PDMX = ROOT / "data" / "pdmx"
OUT = ROOT / "results" / "omr"
MSCORE = Path(os.environ.get("MSCORE", "/Applications/MuseScore 4.app/Contents/MacOS/mscore"))
RENDER = OUT / "render_pages"
DPI = 150


def works(limit: int) -> list[tuple[str, Path]]:
    """(work, local .mxl) for each work in the plan, its first candidate held here."""
    plan = json.loads((PDMX / "plan.json").read_text())
    held = {p.name: p for p in (PDMX / "mxl").glob("*.mxl")}
    chosen, used = [], set()
    for work, candidates in plan.items():
        for candidate in candidates:
            path = held.get(Path(candidate["mxl"]).name)
            if path is not None and path not in used:  # one transcription can be a candidate for several works
                chosen.append((work, path))
                used.add(path)
                break
    return chosen[:limit]


def export(mxl: Path, target: Path) -> bool:
    """MuseScore 4 sometimes crashes as it quits (a mutex, exit 134), now and then before writing the
    file: a file written is taken whatever the exit, and a missing one is tried again."""
    for _ in range(4):
        target.unlink(missing_ok=True)
        try:
            subprocess.run([str(MSCORE), "-o", str(target), str(mxl)], capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            continue
        if target.exists() and target.stat().st_size > 0:
            return True
    return False


def page_starts(mpos: Path) -> tuple[list[int], int]:
    """The index of the first bar on each page, and how many bars there are."""
    pages = [int(e.get("page")) for e in ET.parse(mpos).getroot().iter("element")]
    starts = [pages.index(p) for p in sorted(set(pages))]
    return starts, len(pages)


def scanned(image: np.ndarray, rng: np.random.Generator, keep: bool = False) -> np.ndarray:
    """A page rendered at twice the resolution made to look like an old scan, then drawn at half
    size as PDFKit draws a scan: IMSLP's are mostly black and white at 300-600 dpi, turned a little
    on the glass, and only grey once reduced."""
    from scipy import ndimage
    angle = rng.uniform(-1.25, 1.25)
    page = 255 - ndimage.rotate(255 - image.astype(np.float32), angle, reshape=False, order=1)
    page = ndimage.gaussian_filter(page, rng.uniform(0.5, 1.0))
    page = page + rng.normal(0, 18, page.shape)
    # Thresholded as a bilevel scan is, at a level that thins or thickens the lines a little
    bilevel = np.where(page < rng.uniform(100, 170), 0.0, 255.0)
    if keep:
        return bilevel.astype(np.uint8)
    h, w = bilevel.shape[0] // 2 * 2, bilevel.shape[1] // 2 * 2
    half = bilevel[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    return half.astype(np.uint8)


def make_scans(folder: Path, seed: int) -> None:
    """Overwrites scan/ (150 dpi, as drawn small) and writes clean300/ and scan300/ (the black and
    white scan at its own resolution), all from a render at 300 dpi."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    fine = folder / "clean300"
    subprocess.run([str(RENDER), str(folder / "score.pdf"), str(fine), str(2 * DPI)], capture_output=True, check=True)
    for name in ("scan", "scan300"):
        (folder / name).mkdir(exist_ok=True)
    for png in sorted(fine.glob("p*.png")):
        full = scanned(np.asarray(Image.open(png).convert("L")), rng, keep=True)
        Image.fromarray(full).save(folder / "scan300" / png.name)
        h, w = full.shape[0] // 2 * 2, full.shape[1] // 2 * 2
        half = full[:h, :w].astype(np.float32).reshape(h // 2, 2, w // 2, 2).mean((1, 3))
        Image.fromarray(half.astype(np.uint8)).save(folder / "scan" / png.name)


def build(job) -> str:
    work, mxl, seed, rescan = job
    folder = OUT / work.replace("/", "_")
    truth = folder / "truth.json"
    if truth.exists():
        if rescan:
            make_scans(folder, seed)
            return f"{work}: scans made again"
        return f"{work}: kept"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "mpos"):
        if not export(mxl, folder / f"score.{suffix}"):
            return f"{work}: MuseScore failed"
    starts, bars = page_starts(folder / "score.mpos")
    pages = int(subprocess.run([str(RENDER), str(folder / "score.pdf"), str(folder / "clean"), str(DPI)],
                               capture_output=True, text=True, check=True).stdout.strip())
    if pages != len(starts):
        return f"{work}: {pages} pages but bars on {len(starts)}"
    make_scans(folder, seed)
    truth.write_text(json.dumps({"work": work, "mxl": str(mxl), "bars": bars, "starts": starts, "dpi": DPI}))
    return f"{work}: {pages} pages, {bars} bars"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rescan", action="store_true", help="make the scanned pages again for works already built")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not RENDER.exists():
        subprocess.run(["swiftc", "-O", "-o", str(RENDER), str(Path(__file__).parent / "render_pages.swift")], check=True)
    jobs = [(work, mxl, k, args.rescan) for k, (work, mxl) in enumerate(works(args.limit))]
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(build, jobs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
