"""Counting bars on a printed page, to find the bar each page of a PDF starts with.

Many editions print no bar numbers (the Bach-Gesellschaft's, most old Peters and Breitkopf scans on
IMSLP), and where numbers are printed they share the page with fingerings, so reading digits is not
enough. Barlines are on every page of every edition. A page's start bar is the number of barlines
on the pages before it, mapped onto the score's own list of printed bars.

On a page image, grayscale at about 150 dpi:
  1. straighten it: the angle at which dark pixels stack into the sharpest rows is the one at which
     the staff lines are level
  2. staves: rows dark across much of the page, in fives, evenly spaced
  3. barlines in each staff: thin columns dark over the whole staff and not beyond it, unless they
     continue into the next staff (a grand staff's barlines cross the gap between its staves)
  4. systems: staves joined at the left by a line, brace or bracket, or by barlines across the gap
     (some editions and most transcriptions draw barlines staff by staff)
  5. a system's bars: lines running from its top to its bottom where there are any (a piano
     edition's barlines cross the gap between the staves; stems, digits and noteheads never do),
     otherwise lines found in every staff of a grand staff or most of a larger system, less the line
     that opens every system and a start-repeat at the head of one, which end no bar

  python training/count_bars.py [--variant clean|scan|both] [--show WORK]   # scores results/omr/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

OMR = Path(__file__).resolve().parents[1] / "results" / "omr"


@dataclass
class Staff:
    lines: list[float]  # the five line centres, top to bottom
    x0: int
    x1: int
    bars: list[int] = field(default_factory=list)  # barline x positions
    joined_below: bool = False

    @property
    def space(self) -> float:
        return (self.lines[-1] - self.lines[0]) / 4

    @property
    def top(self) -> int:
        return int(round(self.lines[0]))

    @property
    def bottom(self) -> int:
        return int(round(self.lines[-1]))


def dark(image: np.ndarray) -> np.ndarray:
    return image < 200


def straighten(ink: np.ndarray) -> tuple[np.ndarray, float]:
    """Level the staff lines: the shear that makes the row profile sharpest, then that rotation."""
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return ink, 0.0
    xs = xs - ink.shape[1] / 2

    def sharpness(angle: float) -> float:
        rows = np.round(ys - xs * np.tan(np.radians(angle))).astype(int)
        rows -= rows.min()
        return float(np.sum(np.bincount(rows).astype(np.float64) ** 2))

    coarse = np.arange(-2.0, 2.01, 0.1)
    best = coarse[int(np.argmax([sharpness(a) for a in coarse]))]
    fine = np.arange(best - 0.1, best + 0.101, 0.01)
    angle = float(fine[int(np.argmax([sharpness(a) for a in fine]))])
    if abs(angle) < 0.02:
        return ink, angle
    # Turned whichever way makes the rows sharper: the shear's sign and the rotation's are easy to
    # get opposite, and a page turned the wrong way is twice as crooked
    turned = [ndimage.rotate(ink.astype(np.uint8), sign * angle, reshape=False, order=0) > 0 for sign in (1, -1)]
    level = max(turned, key=lambda im: float(np.sum(im.sum(1).astype(np.float64) ** 2)))
    return level, angle


def staff_lines(ink: np.ndarray, broken: bool = False) -> list[float]:
    """Centres of rows holding one long unbroken stroke. A row of beams can be as dark across the
    page as a staff line, but in many short pieces; a clean staff line runs a fifth of the page at
    least."""
    closed = ndimage.binary_closing(ink, np.ones((1, max(3, ink.shape[1] // 400)), bool))
    padded = np.pad(closed, ((0, 0), (1, 1))).astype(np.int8)
    edges = np.diff(padded, axis=1)
    longest = np.zeros(ink.shape[0])
    for r in range(ink.shape[0]):
        starts, ends = np.flatnonzero(edges[r] == 1), np.flatnonzero(edges[r] == -1)
        if len(starts):
            longest[r] = (ends - starts).max()
    # Or dark across much of the page though broken, as a scan's lines are: the staff grouping that
    # follows skips the rows of beams this lets in, which fall between the lines
    darkness = ink.sum(1) / ink.shape[1]
    rows = np.flatnonzero((longest >= 0.2 * ink.shape[1]) | (broken & (darkness >= 0.4)))
    runs: list[list[int]] = []
    for r in rows:
        if runs and r == runs[-1][-1] + 1:
            runs[-1].append(int(r))
        else:
            runs.append([int(r)])
    # A beam lying on a staff line thickens it: the line is where the run is darkest, as a staff line
    # crosses the page and a beam does not
    lines = []
    for run in runs:
        values = darkness[run]
        core = [r for r, v in zip(run, values) if v >= 0.8 * values.max()]
        lines.append(float(np.average(core, weights=darkness[core])))
    return lines


def staves(ink: np.ndarray) -> list[Staff]:
    """The staves on a page, from unbroken lines and from lines dark across the page though broken,
    as a scan's are. Rows closer than a third of a space are one line: a scanned page is seldom
    flat, and one end of a line can sit a few pixels below the other."""
    lines = sorted(set(staff_lines(ink)) | set(staff_lines(ink, broken=True)))
    close = max(3.0, ink.shape[1] / 450)  # about a third of a space at any resolution
    merged: list[list[float]] = []
    for y in lines:
        if merged and y - merged[-1][-1] <= close:
            merged[-1].append(y)
        else:
            merged.append([y])
    return staves_from(ink, [float(np.mean(m)) for m in merged])


def row_darkness(ink: np.ndarray, y: float) -> float:
    r = int(round(y))
    return float(ink[max(r - 1, 0):r + 2].mean(1).max())


def staves_from(ink: np.ndarray, lines: list[float]) -> list[Staff]:
    if len(lines) < 5:
        return []
    # The spacing of the staff lines: the commonest gap between neighbouring lines
    gaps = np.round(np.diff(lines)).astype(int)
    gaps = gaps[(gaps >= 4) & (gaps <= 80)]
    if not len(gaps):
        return []
    mode = float(np.bincount(gaps).argmax())
    exact = np.diff(lines)
    d = float(np.mean(exact[np.abs(exact - mode) <= max(1.5, 0.2 * mode)]))  # to a fraction of a pixel
    tolerance = max(1.6, 0.2 * d)
    used, found = set(), []
    for i, y in enumerate(lines):
        if i in used:
            continue
        picks = [i]
        for j in range(1, 5):
            expected = lines[picks[-1]] + d  # from the line before, so no error adds up
            near = [k for k, z in enumerate(lines) if k not in used and abs(z - expected) <= tolerance]
            if not near:
                break
            picks.append(min(near, key=lambda k: abs(lines[k] - expected)))
        if len(picks) == 5:
            # A staff's five lines are about equally dark; a slur or a beam taken for its first line
            # is far fainter, and the staff starts at the next line instead
            strength = [row_darkness(ink, lines[k]) for k in picks]
            if min(strength) < 0.5 * float(np.median(strength)):
                continue
            used.update(picks)
            found.append([lines[k] for k in picks])
    result = []
    for group in found:
        rows = [int(round(y)) for y in group]
        space = (group[-1] - group[0]) / 4
        # A scan's lines wander a pixel or two across the page; a third of a space either way still
        # cannot reach the next line
        reach = max(1, int(round(0.3 * space)))
        # Where the staff runs: columns where most of its five lines are dark
        band = np.stack([ink[max(r - reach, 0):r + reach + 1].any(0) for r in rows])
        present = band.sum(0) >= 4
        present = ndimage.binary_closing(present, np.ones(max(9, int(2 * space)), bool))
        runs = np.flatnonzero(np.diff(np.concatenate([[0], present.astype(int), [0]])))
        starts, ends = runs[::2], runs[1::2]
        longest = int(np.argmax(ends - starts))
        x0, x1 = int(starts[longest]), int(ends[longest]) - 1
        # Step past a brace, bracket or opening line, which fill the spaces between the lines as
        # the lines themselves do not
        spaces = np.stack([ink[int(round((a + b) / 2))] for a, b in zip(group, group[1:])])
        limit = x0 + int(4 * (group[-1] - group[0]) / 4)
        while x0 < min(x1, limit) and spaces[:, x0].sum() >= 2:
            x0 += 1
        result.append(Staff(group, x0, x1))
    return result


def barlines(ink: np.ndarray, staff: Staff, above_staff: Staff | None,
             below: Staff | None) -> tuple[list[int], list[bool]]:
    """Barline x positions in a staff, and for each whether it continues down to the next staff."""
    d = staff.space
    wide = ndimage.maximum_filter1d(ink, lean(d), axis=1)  # a line may lean a pixel or two either way
    top, bottom = staff.top, staff.bottom
    cover = wide[top:bottom + 1].mean(0)
    # Just outside the staff: where a stem meets its beam or its notehead, and a barline meets paper
    near, far = max(int(round(0.25 * d)), 2), max(int(round(1.0 * d)), 4)
    above = wide[max(top - far, 0):max(top - near, 1)].mean(0)
    under = wide[bottom + near:bottom + far].mean(0)

    def gap_to(other: Staff | None, upper: int, lower: int) -> np.ndarray:
        if other is None or lower - upper > 12 * d or lower <= upper + 1:
            return np.zeros(ink.shape[1], bool)
        return wide[upper + 1:lower].mean(0) >= 0.9

    joined_below = gap_to(below, bottom, below.top if below else 0)
    joined_above = gap_to(above_staff, above_staff.bottom if above_staff else 0, top)
    candidates = cover >= 0.92
    # A barline stops at the staff, unless it carries on into a staff above or below
    candidates &= (above < 0.3) | joined_above
    candidates &= (under < 0.3) | joined_below
    joined = joined_below
    xs = np.flatnonzero(candidates)
    xs = xs[(xs > staff.x0 + OPENING * d) & (xs <= staff.x1 + 2)]
    # One line per run of columns, and a double bar or a repeat sign's thick-and-thin as one
    groups: list[list[int]] = []
    for x in xs:
        if groups and x - groups[-1][-1] <= max(0.9 * d, 3):
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    # Too wide to be a barline (thick-and-thin is under a space): a beam or a filled block
    groups = [g for g in groups if g[-1] - g[0] <= 1.2 * d]
    # A barline stands alone: beside it only the staff lines, where a stem has its notehead and a
    # time signature's digits their other strokes
    groups = [g for g in groups if beside(ink, staff, g[0], g[-1]) <= ISOLATION]
    positions = [int(np.mean(g)) for g in groups]
    return positions, [bool(joined[g].any()) for g in groups]


ISOLATION = 0.15
# The line that opens every system, which ends no bar, can stand a little inside where the staff is
# found to start (past a brace); no barline comes within a space and a half of the start, where the
# clef is
OPENING = 1.5


def lean(d: float) -> int:
    """How far either way a vertical line may wander over a staff, as an odd width in pixels."""
    return 2 * max(1, int(round(0.08 * d))) + 1


def beside(ink: np.ndarray, staff: Staff, left: int, right: int) -> float:
    """How dark the paper is either side of a line, staff lines left out: the darker side's share."""
    d = staff.space
    rows = np.arange(staff.top, staff.bottom + 1)
    off_lines = np.ones(len(rows), bool)
    for y in staff.lines:
        off_lines &= np.abs(rows - y) > max(1.5, 0.15 * d)
    rows = rows[off_lines]
    near, far = max(int(round(0.3 * d)), 2), max(int(round(0.9 * d)), 4)
    sides = []
    for a, b in ((left - far, left - near), (right + near, right + far)):
        a, b = max(a, 0), min(b, ink.shape[1] - 1)
        sides.append(float(ink[rows][:, a:b + 1].mean()) if b >= a else 0.0)
    return max(sides)


def repeat_start(ink: np.ndarray, staff: Staff, x: int) -> bool:
    """A thick line with dots in the two middle spaces just right of it, as a start repeat has. The
    thick line matters: notes just after an ordinary barline can sit in those spaces too."""
    d = staff.space
    rows = slice(staff.top, staff.bottom + 1)
    thick = int((ink[rows, max(x - int(0.4 * d), 0):x + int(0.4 * d) + 1].mean(0) >= 0.9).sum())
    if thick < 0.3 * d:
        return False
    lo, hi = x + int(0.3 * d), x + int(1.6 * d)
    dots = 0
    for upper, lower in ((1, 2), (2, 3)):
        y0, y1 = int(staff.lines[upper] + 0.25 * d), int(staff.lines[lower] - 0.25 * d)
        region = ink[y0:y1 + 1, lo:hi + 1]
        dots += bool(region.mean() > 0.12)
    return dots == 2


def count_page(image: np.ndarray, detail: bool = False) -> int | tuple[int, dict]:
    ink, angle = straighten(dark(image))
    found = staves(ink)
    positions = []
    for k, staff in enumerate(found):
        below = found[k + 1] if k + 1 < len(found) else None
        above = found[k - 1] if k else None
        staff.bars, joined = barlines(ink, staff, above, below)
        positions.append(joined)
    # Systems: a staff joins the one below if barlines cross the gap, or if their barlines agree
    systems: list[list[int]] = []
    for k, staff in enumerate(found):
        if k and joins(found[k - 1], staff, positions[k - 1], ink):
            systems[-1].append(k)
        else:
            systems.append([k])
    total = 0
    per_system = []
    sizes = [len(system) for system in systems]
    for system in systems:
        members = [found[k] for k in system]
        through = through_lines(ink, members) if len(members) > 1 else []
        if through:
            # Barlines drawn from the top of the system to the bottom: nothing else on a page does that
            lines = through
            if repeat_start(ink, members[0], lines[0]) and lines[0] < members[0].x0 + 0.3 * (members[0].x1 - members[0].x0):
                lines = lines[1:]
            per_system.append(len(lines))
            total += len(lines)
            continue
        xs = sorted(x for s in members for x in s.bars)
        merged: list[list[int]] = []
        for x in xs:
            if merged and x - merged[-1][-1] <= max(3, 0.3 * members[0].space):
                merged[-1].append(x)
            else:
                merged.append([x])
        # Both staves of a grand staff; more than half of a larger system
        need = len(members) if len(members) <= 2 else int(np.ceil(0.6 * len(members)))
        lines = [int(np.mean(m)) for m in merged if len(m) >= need]
        if lines and repeat_start(ink, members[0], lines[0]) and lines[0] < members[0].x0 + 0.3 * (members[0].x1 - members[0].x0):
            lines = lines[1:]
        per_system.append(len(lines))
        total += len(lines)
    if detail:
        return total, {"angle": angle, "staves": len(found), "systems": per_system, "sizes": sizes}
    return total


def connected(ink: np.ndarray, upper: Staff, lower: Staff) -> bool:
    """The line at the left edge that joins the staves of a system (and a piano's brace beside it)."""
    if lower.top - upper.bottom > 12 * upper.space or lower.top <= upper.bottom + 1:
        return False
    x = min(upper.x0, lower.x0)
    band = ink[upper.bottom + 1:lower.top, max(x - int(2 * upper.space), 0):x + 4]  # a brace sits further left
    return bool((band.mean(0) >= 0.85).any())


def through_lines(ink: np.ndarray, members: list[Staff]) -> list[int]:
    """Lines that run unbroken from the top of a system's first staff to the bottom of its last."""
    d = members[0].space
    wide = ndimage.maximum_filter1d(ink, lean(d), axis=1)
    cover = wide[members[0].top:members[-1].bottom + 1].mean(0)
    x0, x1 = min(m.x0 for m in members), max(m.x1 for m in members)
    xs = np.flatnonzero(cover >= 0.95)
    xs = xs[(xs > x0 + OPENING * d) & (xs <= x1 + 2)]
    groups: list[list[int]] = []
    for x in xs:
        if groups and x - groups[-1][-1] <= max(0.9 * d, 3):
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    # Wider than a barline elsewhere: a final double bar's thin and thick lines together can span
    # more than a space, and nothing else runs the height of a system
    return [int(np.mean(g)) for g in groups if g[-1] - g[0] <= 2 * d]


def joins(upper: Staff, lower: Staff, joined: list[bool], ink: np.ndarray | None = None) -> bool:
    if ink is not None and connected(ink, upper, lower):
        return True
    # Staves whose barlines merely stand at the same places are not joined: two systems of equal bars
    # would be taken for one
    return bool(upper.bars) and sum(joined) >= max(1, len(upper.bars) // 2)


def evaluate(variant: str, show: str | None) -> list[dict]:
    from PIL import Image
    rows = []
    for truth_path in sorted(OMR.glob("*/truth.json")):
        truth = json.loads(truth_path.read_text())
        folder = truth_path.parent
        pages = sorted((folder / variant).glob("p*.png"))
        if len(pages) != len(truth["starts"]):
            continue
        counts, details = [], []
        for png in pages:
            n, info = count_page(np.asarray(Image.open(png).convert("L")), detail=True)
            counts.append(n)
            details.append(info)
        predicted = list(np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int))
        right = [p == t for p, t in zip(predicted, truth["starts"])]
        rows.append({"work": truth["work"], "pages": len(pages), "right": int(sum(right)),
                     "total": int(sum(counts)), "bars": truth["bars"], "counts": counts,
                     "truth_counts": list(np.diff(truth["starts"] + [truth["bars"]])), "details": details})
        if show and show in truth["work"]:
            print(json.dumps(rows[-1], indent=1, default=int))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="both", help="clean, scan, clean300, scan300, or both (clean and scan)")
    ap.add_argument("--show", default=None)
    args = ap.parse_args()
    for variant in (("clean", "scan") if args.variant == "both" else (args.variant,)):
        rows = evaluate(variant, args.show)
        pages = sum(r["pages"] for r in rows)
        right = sum(r["right"] for r in rows)
        whole = sum(r["right"] == r["pages"] for r in rows)
        adds_up = [r for r in rows if r["total"] == r["bars"]]
        flagged_wrong = sum(1 for r in rows if r["total"] != r["bars"] and r["right"] < r["pages"])
        silent_wrong = sum(1 for r in rows if r["total"] == r["bars"] and r["right"] < r["pages"])
        print(f"{variant}: {len(rows)} scores, {pages} pages. Page starts right: {right}/{pages} ({right / max(pages, 1):.1%}); "
              f"every page right: {whole}/{len(rows)}. Counts add up to the score's bars: {len(adds_up)}/{len(rows)}; "
              f"wrong but flagged by the total: {flagged_wrong}; wrong and not flagged: {silent_wrong}")
        for r in rows:
            if r["right"] < r["pages"]:
                bad = [k + 1 for k, (c, t) in enumerate(zip(r["counts"], r["truth_counts"])) if c != t]
                print(f"   {r['work']:40} {r['right']}/{r['pages']} pages right; counted {r['total']} of {r['bars']} bars; "
                      f"pages miscounted {bad[:8]} (counted {[r['counts'][k - 1] for k in bad[:8]]}, "
                      f"true {[r['truth_counts'][k - 1] for k in bad[:8]]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
