import XCTest
@testable import BarCounter

final class BarCountingTests: XCTestCase {
    /// Staves drawn with a line every 12 pixels: the staff space is 12, whatever the line's weight.
    func testStaffSpaceIsTheStepBetweenLines() {
        let width = 400, height = 600
        var page = [UInt8](repeating: 255, count: width * height)
        for staff in [60, 200, 340, 480] {
            for line in 0..<5 {
                for y in staff + line * 12..<staff + line * 12 + 2 {
                    for x in 0..<width { page[y * width + x] = 0 }
                }
            }
        }
        let space = BarCounting.staffSpace(gray: page, width: width, height: height)
        XCTAssertEqual(space ?? 0, 12, accuracy: 0.05)
        XCTAssertEqual(BarCounting.scale(forStaffSpace: space, width: width), 0.75, accuracy: 0.01)
    }

    func testABlankPageHasNoStaffSpace() {
        XCTAssertNil(BarCounting.staffSpace(gray: [UInt8](repeating: 255, count: 200 * 300), width: 200, height: 300))
    }

    /// One peak per bar, however far a barline's peak spreads down its system; faint ones kept.
    func testPeaksAreLocalMaxima() {
        let width = 10, height = 20
        var grid = [Float](repeating: 0, count: width * height)
        grid[5 * width + 2] = 0.9
        grid[8 * width + 2] = 0.6  // the same barline, lower down: within four cells, so not a bar
        grid[5 * width + 7] = 0.3
        grid[15 * width + 2] = 0.02  // under the floor
        XCTAssertEqual(BarCounting.peaks(grid, width: width, height: height).sorted(), [0.3, 0.9])
    }

    /// Half precision can make the top of one peak two equal cells: one bar, not two.
    func testATiedPeakIsOneBar() {
        let width = 6, height = 12
        var grid = [Float](repeating: 0, count: width * height)
        grid[4 * width + 2] = 0.98
        grid[4 * width + 3] = 0.98
        grid[6 * width + 2] = 0.98
        XCTAssertEqual(BarCounting.peaks(grid, width: width, height: height), [0.98])
    }

    /// Two bars too many: the two weakest over the threshold go, and their pages are marked.
    func testCountsAreMadeToAgreeWithTheScore() {
        let pages: [[Float]] = [[0.9, 0.9, 0.42], [0.95, 0.44, 0.9], [0.9, 0.9, 0.9]]
        let counts = PageCounts(peaks: pages, scoreBars: 7)
        XCTAssertEqual(counts.bars, [2, 2, 3])
        XCTAssertEqual(counts.unsure, [true, true, false])
        XCTAssertEqual(counts.found, 9)
        XCTAssertTrue(counts.agreesWithScore)

        let short = PageCounts(peaks: [[0.9, 0.35], [0.9, 0.9, 0.1]], scoreBars: 4)
        XCTAssertEqual(short.bars, [2, 2])
        XCTAssertEqual(short.unsure, [true, false])
    }

    /// Far from the score's count: another edition, or other movements. Left as found.
    func testCountsFarFromTheScoreAreLeftAlone() {
        let pages = [[Float]](repeating: [0.9, 0.9, 0.9, 0.9, 0.9], count: 10)
        let counts = PageCounts(peaks: pages, scoreBars: 100)
        XCTAssertEqual(counts.bars, [Int](repeating: 5, count: 10))
        XCTAssertFalse(counts.agreesWithScore)
        XCTAssertTrue(counts.unrelatedToScore)
    }

    /// A title page, then pages of 10, 12 and 8 bars: each page starts after the bars before it.
    func testPageStartsFromCounts() {
        let starts = PageStarts(counts: PageCounts(bars: [0, 10, 12, 8], unsure: [false, false, true, false],
                                                   found: 30, scoreBars: 30))
        XCTAssertEqual((1...4).map { starts.start(ofPage: $0) }, [nil, 0, 10, 22])
        XCTAssertEqual(starts.unsurePages, [4])  // page 3's count decides where page 4 starts
    }

    /// Correcting one page moves every page after it, up to a page already known.
    func testACorrectionMovesThePagesAfterIt() {
        var starts = PageStarts(counts: PageCounts(bars: [10, 10, 10, 10, 10], unsure: [true, false, false, false, false],
                                                   found: 50, scoreBars: 60))
        starts.set(page: 5, start: 39)
        XCTAssertEqual((1...5).map { starts.start(ofPage: $0) }, [0, 10, 20, 30, 39])

        let moved = starts.set(page: 2, start: 11)
        XCTAssertEqual((1...5).map { starts.start(ofPage: $0) }, [0, 11, 21, 31, 39])
        XCTAssertEqual(moved.map(\.page), [3, 4])
        XCTAssertEqual(moved.map(\.bars), [1, 1])
        XCTAssertEqual(starts.unsurePages, [])  // page 2 was the one in doubt, and it is known now
    }

    /// Checking a page leaves its bar alone and settles the doubt about it.
    func testConfirmingAPage() {
        var starts = PageStarts(counts: PageCounts(bars: [10, 10, 10], unsure: [true, false, false], found: 30,
                                                   scoreBars: 30))
        XCTAssertEqual(starts.unsurePages, [2])
        starts.confirm(page: 2)
        XCTAssertEqual(starts.unsurePages, [])
        XCTAssertEqual((1...3).map { starts.start(ofPage: $0) }, [0, 10, 20])
    }

    /// A page that would start past the last bar has no start.
    func testPagesPastTheEnd() {
        let starts = PageStarts(counts: PageCounts(bars: [10, 10, 1], unsure: [false, false, false], found: 21,
                                                   scoreBars: 15))
        XCTAssertEqual((1...3).map { starts.start(ofPage: $0) }, [0, 10, nil])
    }
}
