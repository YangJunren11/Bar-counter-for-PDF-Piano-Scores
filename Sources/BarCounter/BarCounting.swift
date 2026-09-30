import Foundation

/// Counting the bars on each page of a PDF of piano music, so that where every page starts can be
/// filled in rather than typed.
///
/// A small network (`model/BarCounter.mlpackage`, trained by `training/omr_ml.py`) marks where each
/// bar on a page ends: one peak per bar, at the barline halfway down its system. This is everything
/// around the network that does not depend on Apple: measuring a page's staff space so it reaches the
/// network at the size it was trained on, finding the peaks, making the counts agree with the
/// score's bar count, and choosing which pages to ask about (`PageStarts`). `PDFBarCounter` draws the
/// pages and runs the network on Apple's platforms.
///
/// On three scanned editions counted by eye (55 pages), 52 pages were counted exactly; every page
/// that was wrong had a peak near the threshold, which is what marks a page as unsure here.
public enum BarCounting {
    /// Pixels per staff space, as the network sees a page.
    public static let space = 9.0
    /// The network's grid, in pixels of the page it was given.
    public static let stride = 4
    /// A peak at least this high is a bar: the best cut-off on the prototype's test pages.
    public static let threshold: Float = 0.4
    /// A peak this close to the threshold, on either side, makes a page's count a close call.
    public static let doubt: Float = 0.1

    /// The distance between staff lines, in pixels, from runs down the page's columns: from the top
    /// of one staff line to the top of the next is the commonest step between black runs. Nil for a
    /// page with too few lines to tell, such as a title page.
    ///
    /// `gray` is the page row by row from the top, 0 black and 255 white.
    public static func staffSpace(gray: [UInt8], width: Int, height: Int) -> Double? {
        guard width > 0, height > 1, gray.count >= width * height else { return nil }
        var steps = [Int](repeating: 0, count: 160)
        let every = max(1, width / 300)
        var starts: [Int] = []
        for x in Swift.stride(from: 0, to: width, by: every) {
            starts.removeAll(keepingCapacity: true)
            var inked = false
            for y in 0..<height {
                let ink = gray[y * width + x] < 160
                if ink && !inked { starts.append(y) }
                inked = ink
            }
            guard starts.count >= 3 else { continue }
            for k in 1..<starts.count { steps[min(starts[k] - starts[k - 1], 159)] += 1 }
        }
        steps[0] = 0; steps[1] = 0; steps[2] = 0; steps[159] = 0
        guard steps.reduce(0, +) >= 50 else { return nil }
        let mode = steps.indices.max { steps[$0] < steps[$1] }!  // the first, where two are as common
        // To a fraction of a pixel: the steps around the commonest, weighted by how common
        let around = max(mode - 1, 3)..<min(mode + 2, 159)
        let weight = around.reduce(0.0) { $0 + Double(steps[$1]) + 1e-9 }
        return around.reduce(0.0) { $0 + Double($1) * (Double(steps[$1]) + 1e-9) } / weight
    }

    /// How much to scale a page so its staff space is `space` pixels, for a page `width` pixels wide
    /// whose staff space was measured as `measured`. A page without staves is given the size of a
    /// typical printed page, where it will be found to hold no bars.
    public static func scale(forStaffSpace measured: Double?, width: Int) -> Double {
        let scale = measured.map { space / $0 } ?? 1240 / Double(width) * space / 10.3
        return min(max(scale, 0.4), 2.5)
    }

    /// The height of every peak in the network's grid at least `floor` high: a cell that is the
    /// highest within four cells up or down and one across, as one barline's peak can spread down a
    /// tall system. Faint peaks are kept so a page's count can be judged against the threshold.
    ///
    /// Where two cells of one peak are equally high, only the first is taken. Core ML runs the
    /// network in half precision, whose steps near 1 are coarse enough for the top of a peak to tie
    /// with its neighbour: counted twice, that was a bar too many on a page, with nothing to say so.
    public static func peaks(_ grid: [Float], width: Int, height: Int, floor: Float = 0.05) -> [Float] {
        guard width > 0, height > 0, grid.count >= width * height else { return [] }
        var across = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var top = grid[y * width + x]
                if x > 0 { top = max(top, grid[y * width + x - 1]) }
                if x + 1 < width { top = max(top, grid[y * width + x + 1]) }
                across[y * width + x] = top
            }
        }
        var found: [Float] = []
        var taken = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let value = grid[y * width + x]
                guard value >= floor else { continue }
                var top = value
                for dy in max(0, y - 4)...min(height - 1, y + 4) { top = max(top, across[dy * width + x]) }
                guard value == top else { continue }
                // A peak already taken within reach, before this cell, can only be as high: a tie
                var tie = x > 0 && taken[y * width + x - 1]
                for dy in max(0, y - 4)..<y where !tie {
                    for dx in max(0, x - 1)...min(width - 1, x + 1) where taken[dy * width + dx] { tie = true }
                }
                if tie { continue }
                taken[y * width + x] = true
                found.append(value)
            }
        }
        return found
    }
}

/// The bars counted on each page of a PDF, and how sure the count was.
public struct PageCounts: Codable, Equatable {
    /// Bars ending on each page, page 1 first: a bar carried over a page break is counted on the
    /// page where it ends, so a page starts with the bar after the ones counted before it.
    public var bars: [Int]
    /// Whether each page's count was a close call: a peak near the threshold, or a bar added or
    /// taken away to make the counts agree with the score.
    public var unsure: [Bool]
    /// Bars found in the whole PDF, before any were added or taken away.
    public var found: Int
    /// Bars in the score.
    public var scoreBars: Int

    /// How far the bars found may be from the score's before the PDF is taken to be a different
    /// edition, or to hold other movements, rather than miscounted: the counts are then left as found.
    public static func tolerance(scoreBars: Int) -> Int { max(3, scoreBars / 20) }

    /// Counts from every page's peaks (`BarCounting.peaks`), made to agree with a score of
    /// `scoreBars` bars when they come close. The peaks nearest the threshold are the likeliest to be
    /// wrong, so bars missing are taken from the strongest peaks under it, and bars too many from the
    /// weakest over it; the pages they are on are marked unsure.
    public init(peaks pages: [[Float]], scoreBars: Int, threshold: Float = BarCounting.threshold,
                doubt: Float = BarCounting.doubt) {
        var bars = pages.map { page in page.filter { $0 >= threshold }.count }
        var unsure = pages.map { page in page.contains { abs($0 - threshold) < doubt } }
        let found = bars.reduce(0, +)
        let short = scoreBars - found
        if short != 0, abs(short) <= Self.tolerance(scoreBars: scoreBars) {
            let candidates = pages.enumerated().flatMap { page, peaks in peaks.map { (height: $0, page: page) } }
            let chosen = short > 0
                ? candidates.filter { $0.height < threshold }.sorted { $0.height > $1.height }.prefix(short)
                : candidates.filter { $0.height >= threshold }.sorted { $0.height < $1.height }.prefix(-short)
            for peak in chosen {
                bars[peak.page] += short > 0 ? 1 : -1
                unsure[peak.page] = true
            }
        }
        self.bars = bars
        self.unsure = unsure
        self.found = found
        self.scoreBars = scoreBars
    }

    public init(bars: [Int], unsure: [Bool], found: Int, scoreBars: Int) {
        self.bars = bars
        self.unsure = unsure
        self.found = found
        self.scoreBars = scoreBars
    }

    /// The bars found agree with the score's, or came close enough to be made to.
    public var agreesWithScore: Bool { abs(found - scoreBars) <= Self.tolerance(scoreBars: scoreBars) }

    /// So far from the score that the pages cannot be this score's: filling them in would only
    /// have to be undone. A quarter either way.
    public var unrelatedToScore: Bool { abs(found - scoreBars) * 4 > scoreBars }
}

/// Where each page starts, worked out from the bars counted on the pages before it and from the
/// pages whose first bar is known: set or checked by a person. So correcting one page moves every
/// page after it, as far as the next known page; a count that was wrong before a page is wrong for
/// all the pages after it alike.
///
/// Bars are counted from 0, the first bar of the score, whatever the score prints.
public struct PageStarts {
    public let counts: PageCounts
    /// Pages whose first bar is known (1-based page, 0-based bar).
    public private(set) var known: [Int: Int] = [:]

    public init(counts: PageCounts) {
        self.counts = counts
    }

    /// The first bar on a page, or nil for a page with no bars (a title page, a preface) or one that
    /// would start past the last bar.
    public func start(ofPage page: Int) -> Int? {
        if let set = known[page] { return set }
        guard page >= 1, page <= counts.bars.count, counts.bars[page - 1] > 0 else { return nil }
        let from = known.keys.filter { $0 < page }.max()
        let base = from.map { known[$0]! } ?? 0
        let first = from ?? 1
        let start = base + counts.bars[(first - 1)..<(page - 1)].reduce(0, +)
        return start < counts.scoreBars ? start : nil
    }

    /// Set a page's first bar. The pages after it move with it, as far as the next known page.
    /// Returns the pages that moved, and by how many bars each.
    @discardableResult
    public mutating func set(page: Int, start: Int) -> [(page: Int, bars: Int)] {
        let next = known.keys.filter { $0 > page }.min() ?? counts.bars.count + 1
        let after = Array((page + 1)..<max(page + 1, next))
        let before = after.map { self.start(ofPage: $0) }
        known[page] = start
        return zip(after, before).compactMap { page, old in
            guard let old, let new = self.start(ofPage: page), new != old else { return nil }
            return (page, new - old)
        }
    }

    /// A page found right as counted.
    public mutating func confirm(page: Int) {
        if let start = start(ofPage: page) { known[page] = start }
    }

    /// Pages whose first bar is in doubt: the first page with bars after each close call, as its
    /// first bar is the one that call decides. A known page coming first settles the doubt; checking
    /// the page asked about puts right every page after it too.
    public var unsurePages: [Int] {
        var unsure: [Int] = []
        var pending = false
        for page in 1...max(counts.bars.count, 1) {
            if known[page] != nil {
                pending = false
            } else if start(ofPage: page) != nil {
                if pending { unsure.append(page) }
                pending = false
            }
            if page <= counts.unsure.count && counts.unsure[page - 1] { pending = true }
        }
        return unsure
    }
}
