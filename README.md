# Bar Counter for PDF Piano Scores

Counts the bars on each page of a PDF of piano music, so an app can work out which bar every page
starts with instead of asking someone to type them all in. It was written for PageTurner, an iPad app
that follows a pianist through the score and turns the pages; any app that needs to know where the
pages of a PDF fall in a score can use it.

A small network (1.3 million parameters, 2.5 MB in Core ML) marks where each bar on a page ends: one
peak per bar, at the barline halfway down its system. The bars on a page are the peaks. Around the
network, plain Swift measures each page so it reaches the network at the size it was trained on,
makes the counts agree with the score's bar count when they come close, and says which pages are
worth checking. Correcting one page moves every page after it.

```
swift build -c release
swift run -c release barcount model/BarCounter.mlpackage score.pdf --bars 561
swift test                 # 10 tests
```

```
page 1: 28 bars, starts with bar 0
page 2: 35 bars, starts with bar 28
...
page 5: 28 bars (found 29), starts with bar 131, unsure
...
563 bars found, 0.28 s a page (the score has 561)
```

Counting needs Core ML, so a Mac, iPad or iPhone. The rest of the package is plain Swift and
Foundation; the network is also published as PyTorch weights (`model/barcounter.pt`), from which
`training/omr_ml.py` can export it to other runtimes.

## How well it counts

Pages counted exactly, as found:

| Pages | This network | Counting by rules (`training/count_bars.py`) |
|---|---|---|
| Three IMSLP scans that I am currently playing: Bach BWV 868, Beethoven Op. 101 (Breitkopf), Franck's *Prélude, Aria et Final* | **52 of 55** | 30 of 55 |
| AudioLabs v2 real scans held out from training: Beethoven sonatas, the second half of *Winterreise* | **45 of 47** | 9 of 47 |
| MuseScore engravings of 60 test works, clean and made to look scanned | **487 of 498** | — |

The scans' counts are in `results/real/truth.json`, with the IMSLP numbers of the editions; a bar
carried over a page break is counted on the page where it ends. The three pages it gets wrong there:
a system that ends without a barline counted as if it had one (Op. 101, page 13), a thin-and-thick
final barline counted as two bars (Op. 101, page 16), and the brackets that join a chord across both
staves taken for a barline (Franck, page 12).

**What a person has to check.** A page is in doubt when the count before it was a close call: a peak
within 0.1 of the 0.4 threshold, or a bar added or taken away to make the pages agree with the
score's bar count. The page to check is the first after each close call, since its first bar is the
one that call decides; correcting it puts right every page after it. `barcount --truth` walks through
this as a person would:

| Score | Pages | Page starts wrong as counted | Pages asked about | Page starts wrong after checking them |
|---|---|---|---|---|
| Bach BWV 868 | 4 | 0 | 0 | 0 |
| Beethoven Op. 101 | 18 | 5 | 5 | 0 |
| Franck | 33 | 4 | 6 | 0 |

**Speed**, drawing each page included: 0.1–0.3 s a page on an M5 Mac with Core ML free to use the GPU
and Neural Engine. It has not been timed on an iPad yet.

## Using it in an app

```swift
import BarCounter
import CoreML

let counter = try PDFBarCounter(model: Bundle.main.url(forResource: "BarCounter", withExtension: "mlmodelc")!)
let peaks = try counter.peaks(inPDF: pdf) { done, of in true }   // return false to stop
let counts = PageCounts(peaks: peaks, scoreBars: 561)           // made to agree when they come close
var starts = PageStarts(counts: counts)

starts.start(ofPage: 5)          // 131: its first bar, counted from 0; nil for a page with no bars
starts.unsurePages               // [5, 6, 8, 9, 14]: the pages to ask about
starts.set(page: 9, start: 250)  // a correction: the pages after it move, and which did is returned
starts.confirm(page: 5)          // found right as counted
```

Add `BarCounter.mlpackage` to an Xcode target and Xcode compiles it into `BarCounter.mlmodelc`.
`PageCounts.agreesWithScore` and `unrelatedToScore` say whether the PDF looks like the same edition
as the score: more than 5% off, it may be another edition; more than a quarter off, it probably holds
other movements, and its pages are better filled in by hand.

## How it works

Every page is scaled so a staff space (the gap between two staff lines) is 9 pixels,
measured from the page itself: the commonest step from one black run to the next, down the page's
columns. So a page reaches the network at the same size whatever its edition, scan or resolution.
`PDFBarCounter` draws each page twice: at 3 pixels a point to measure it, and again at the size the
network wants, which keeps engraved scores sharp.

The network is a small U-Net, down to 1/32 of the page and back up to 1/4, with tall kernels at
its deepest level, since a barline is known by where it starts and stops across a system. It is
trained as a CenterNet-style heatmap: a Gaussian at the right end of every bar, halfway down its
system, learned with a focal loss.

Core ML runs the network in half precision, whose steps near 1 are
coarse enough for the top of a peak to come out exactly as high as the cell beside it; counted twice,
that was a confident bar too many, so `BarCounting.peaks` takes the first of two equal cells. And the
iOS simulator's GPU gives back nothing but zeros for this network, so `PDFBarCounter` counts on the
CPU there, and falls back to it on any device that does the same.

## Training

The network learned from 14,301 pages:

- 262 scores engraved by MuseScore 4 in five house styles (Leland, Bravura, Emmentaler, Gonville and
  Maestro), with each transcriber's layout taken out of four of them so the same music falls into new
  systems. MuseScore writes each bar's box beside the pages (`training/omr_synth.py`).
- 600 generated pages of text (titles, prefaces, contents) that hold no bars.
- 893 of AudioLabs v2's real scans, annotated by hand, drawn a third of the time.

Every page was degraded at random as it was used — thickened, thinned, blurred, speckled, turned by up
to a degree — so an engraving looks like a scan. 12,000 steps of 8 crops took about two hours on an
M5 Mac's GPU.

```
python3 -m venv .venv && source .venv/bin/activate && pip install -r training/requirements.txt
python training/omr_testset.py                   # results/omr: the 60 test works, engraved and "scanned"
python training/omr_synth.py                     # data/omr_synth: the training engravings
python training/omr_ml.py prepare                # data/omr_ml: every page at 9 pixels a staff space
python training/omr_ml.py train --model data/omr_ml/model.pt
python training/omr_ml.py evaluate --model data/omr_ml/model.pt
python training/omr_ml.py real --model data/omr_ml/model.pt     # the scans in results/real
python training/omr_ml.py export --model data/omr_ml/model.pt   # model/BarCounter.mlpackage
```

What it expects in `data/` (not included):

- `data/pdmx/mxl/*.mxl` (and `extra/`): PDMX scores. `data/pdmx/plan.json` maps each test work to its
  PDMX transcriptions (`{"work": [{"mxl": "…"}]}`), so none of them is trained on.
- `data/asap-dataset/`: a clone of [ASAP](https://github.com/fosfrancesco/asap-dataset).
- `data/audiolabs_v2/`: the AudioLabs v2 measure annotations, from
  [OMR-Datasets](https://apacha.github.io/OMR-Datasets/).
- MuseScore 4, found where macOS installs it or at `$MSCORE` and `$MSCORE_TEMPLATE`.

For `real`, render each scan's pages into `results/real/<name>/p001.png`, `p002.png`, … (for example
with `training/render_pages.swift` at 300 dpi) from the IMSLP editions named in
`results/real/truth.json`.

## Licences

The code is licensed under the [Apache License 2.0](LICENSE). The trained network in `model/` is
licensed under [CC BY-NC-SA 4.0](model/LICENSE.md), because ASAP's scores and the AudioLabs
annotations it learned from are; it may be used and shared, but not commercially. Training on the
PDMX scores and the text pages alone gives a network without those terms. The data is credited in
[NOTICE](NOTICE).
