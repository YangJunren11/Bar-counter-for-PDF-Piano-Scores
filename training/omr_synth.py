"""Pages to teach a bar counter with: every score held here, engraved by MuseScore in several house
styles, with the box of every bar on every page.

MuseScore writes the pages as PNGs and, beside them, a measure-position file (.mpos) giving each
bar's box; at -r 150 its units are twelfths of a pixel. The works in results/omr (training/omr_testset.py)
are left out, every transcription of them, so they stay a test the counter never trained on.

  python training/omr_synth.py [--styles 5] [--workers 6]    # data/omr_synth/<style>/<id>-<page>.png, <id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "omr_synth"
# MuseScore 4, where macOS puts it; set MSCORE and MSCORE_TEMPLATE elsewhere
MSCORE = Path(os.environ.get("MSCORE", "/Applications/MuseScore 4.app/Contents/MacOS/mscore"))
TEMPLATE = Path(os.environ.get("MSCORE_TEMPLATE", "/Applications/MuseScore 4.app/Contents/Resources/templates/"
                                                  "01-General/03-Grand_Staff/score_style.mss"))
DPI = 150
UNITS = 12.0  # .mpos units per pixel at -r 150

# House styles: a music font, a staff size, paper, and the weight of barlines and staff lines, so
# that no one engraver's look is all the counter knows
STYLES = {
    "leland": {},
    "bravura": {"musicalSymbolFont": "Bravura", "musicalTextFont": "Bravura Text", "Spatium": "1.6",
                "barWidth": "0.22", "measureSpacing": "1.3"},
    "emmentaler": {"musicalSymbolFont": "Emmentaler", "musicalTextFont": "MScore Text", "Spatium": "1.95",
                   "staffLineWidth": "0.14", "barWidth": "0.16", "showMeasureNumber": "0"},
    "gonville": {"musicalSymbolFont": "Gonville", "musicalTextFont": "Gootville Text", "Spatium": "1.5",
                 "pageWidth": "8.5", "pageHeight": "11", "pagePrintableWidth": "7.5", "minMeasureWidth": "5",
                 "measureSpacing": "1.15"},
    "maestro": {"musicalSymbolFont": "Finale Maestro", "musicalTextFont": "Finale Maestro Text", "Spatium": "1.8",
                "barWidth": "0.25", "staffLineWidth": "0.09", "measureSpacing": "1.7", "showMeasureNumber": "0"},
}
ORIGINAL = "leland"


def sources() -> list[dict]:
    """Every score held here, less every transcription of a work in the test set, with its licence."""
    held = {json.loads(t.read_text())["work"] for t in (ROOT / "results" / "omr").glob("*/truth.json")}
    plan_file = DATA / "pdmx" / "plan.json"
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    licences, excluded = {}, set()
    for work, candidates in plan.items():
        for c in candidates:
            name = Path(c["mxl"]).name
            licences[name] = c.get("licence", "")
            if work in held:
                excluded.add(name)
    found, seen = [], set()
    for path in sorted((DATA / "pdmx" / "mxl").glob("*.mxl")) + sorted((DATA / "pdmx" / "extra").glob("*.mxl")):
        if path.name not in excluded and path.name not in seen:  # some are in both folders
            seen.add(path.name)
            found.append({"id": "pdmx_" + path.stem[:16], "path": str(path), "licence": licences.get(path.name, "pdmx")})
    for path in sorted((DATA / "asap-dataset").glob("**/xml_score.musicxml")):
        work = str(path.parent.relative_to(DATA / "asap-dataset"))
        if work not in held:
            found.append({"id": "asap_" + re.sub(r"\W", "_", work), "path": str(path), "licence": "cc-by-nc-sa (ASAP)"})
    return found


def unlaid(source: dict) -> str:
    """The score without its own layout: MusicXML carries the transcriber's page size, staff size and
    system breaks, and MuseScore keeps them over any style it is given, so every style would break
    the systems at the same bars. Without them, each style lays the music out afresh."""
    target = OUT / "_unlaid" / f"{source['id']}.musicxml"
    if target.exists():
        return str(target)
    path = Path(source["path"])
    if path.suffix == ".mxl":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if not n.startswith("META-INF") and n.endswith((".xml", ".musicxml")))
            text = z.read(name).decode("utf-8", errors="replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<defaults\b.*?</defaults>", "", text, flags=re.S)
    text = re.sub(r"<print\b[^>]*/>", "", text)
    text = re.sub(r"<print\b[^>]*>.*?</print>", "", text, flags=re.S)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target)


def write_style(name: str) -> Path:
    text = TEMPLATE.read_text()
    for tag, value in STYLES[name].items():
        text, n = re.subn(rf"<{tag}>[^<]*</{tag}>", f"<{tag}>{value}</{tag}>", text)
        if not n:
            text = text.replace("<Style>", f"<Style>\n    <{tag}>{value}</{tag}>", 1)
    path = OUT / name / "style.mss"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def boxes(mpos: Path) -> dict[int, list[list[float]]]:
    """Each page's bar boxes, [left, top, right, bottom] in pixels, in the score's order."""
    pages: dict[int, list[list[float]]] = {}
    for e in ET.parse(mpos).getroot().iter("element"):
        x, y, w, h = (float(e.get(k)) / UNITS for k in ("x", "y", "sx", "sy"))
        pages.setdefault(int(e.get("page")), []).append([x, y, x + w, y + h])
    return pages


def run(job) -> list[str]:
    """One MuseScore process for a batch; it may crash as it quits, or on a score it cannot read, so
    what it wrote is kept and what it missed is tried again one at a time."""
    style, batch = job
    folder = OUT / style
    done = []

    def attempt(items):
        jobfile = folder / f"job-{items[0]['id']}.json"
        # The first style keeps the transcriber's own layout, as engraved; the others lay it out afresh
        jobfile.write_text(json.dumps([{"in": s["path"] if style == ORIGINAL else unlaid(s),
                                        "out": [str(folder / f"{s['id']}.png"),
                                                                  str(folder / f"{s['id']}.mpos")]} for s in items]))
        try:
            subprocess.run([str(MSCORE), "-S", str(folder / "style.mss"), "-r", str(DPI), "-j", str(jobfile)],
                           capture_output=True, timeout=120 + 60 * len(items))
        except subprocess.TimeoutExpired:
            pass
        jobfile.unlink(missing_ok=True)

    todo = [s for s in batch if not (folder / f"{s['id']}.json").exists()]
    if todo:
        attempt(todo)
    for s in todo:
        mpos = folder / f"{s['id']}.mpos"
        if not mpos.exists():
            attempt([s])
        if not mpos.exists() or not list(folder.glob(f"{s['id']}-*.png")):
            done.append(f"{style} {s['id']}: failed")
            continue
        pages = boxes(mpos)
        pngs = sorted(folder.glob(f"{s['id']}-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        (folder / f"{s['id']}.json").write_text(json.dumps({
            "source": s, "style": style, "pages": [{"png": p.name, "boxes": pages.get(k, [])} for k, p in enumerate(pngs)]}))
        mpos.unlink()
        done.append(f"{style} {s['id']}: {len(pngs)} pages")
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", type=int, default=len(STYLES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    found = sources()[: args.limit or None]
    print(f"{len(found)} scores", flush=True)
    jobs = []
    for style in list(STYLES)[: args.styles]:
        write_style(style)
        jobs += [(style, found[k:k + args.batch]) for k in range(0, len(found), args.batch)]
    with ProcessPoolExecutor(args.workers) as pool:
        for lines in pool.map(run, jobs):
            for line in lines:
                if "failed" in line:
                    print(line, flush=True)
    for style in list(STYLES)[: args.styles]:
        n = sum(len(json.loads(p.read_text())["pages"]) for p in (OUT / style).glob("*.json"))
        print(f"{style}: {len(list((OUT / style).glob('*.json')))} scores, {n} pages", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
