// Counts the bars on each page of a PDF of piano music.
//
//   swift run -c release barcount model/BarCounter.mlpackage score.pdf [--bars 561] [--truth 28,35,...]
//       [--peaks out.json]
//
// A line per page: the bars counted on it, "unsure" when the count was a close call, and the bar it
// starts with, counted from 0. With --bars, the score's bar count, the counts are made to agree with
// it when they come close. With --truth, the bars on each page counted by eye, it also says how many
// pages were counted exactly, and walks through the pages it would ask about as a person would:
// checking each, correcting it when wrong (which moves the pages after it), and moving on.

import BarCounter
import Foundation

#if canImport(CoreML)
import CoreML

var arguments = CommandLine.arguments
func option(_ name: String) -> String? {
    guard let flag = arguments.firstIndex(of: name), flag + 1 < arguments.count else { return nil }
    defer { arguments.removeSubrange(flag...(flag + 1)) }
    return arguments[flag + 1]
}
let truth = option("--truth").map { $0.split(separator: ",").compactMap { Int($0) } }
let scoreBars = option("--bars").flatMap(Int.init) ?? truth?.reduce(0, +)
let peaksOut = option("--peaks")
guard arguments.count == 3 else {
    print("usage: barcount BarCounter.mlpackage score.pdf [--bars N] [--truth a,b,c] [--peaks out.json]")
    exit(2)
}

let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: arguments[1]))
let counter = try PDFBarCounter(model: compiled)
let started = Date()
let peaks = try counter.peaks(inPDF: URL(fileURLWithPath: arguments[2]))
let seconds = Date().timeIntervalSince(started)
if let peaksOut { try JSONEncoder().encode(peaks).write(to: URL(fileURLWithPath: peaksOut)) }

let found = peaks.map { $0.filter { $0 >= BarCounting.threshold }.count }
let counts = PageCounts(peaks: peaks, scoreBars: scoreBars ?? found.reduce(0, +))
var starts = PageStarts(counts: counts)
for (k, bars) in counts.bars.enumerated() {
    var line = "page \(k + 1): \(bars) bars"
    if bars != found[k] { line += " (found \(found[k]))" }
    if let start = starts.start(ofPage: k + 1) { line += ", starts with bar \(start)" }
    if counts.unsure[k] { line += ", unsure" }
    if let truth, k < truth.count, truth[k] != bars { line += ", WRONG: \(truth[k]) by eye" }
    print(line)
}
print(String(format: "%d bars found, %.2f s a page", counts.found, seconds / Double(max(peaks.count, 1)))
      + (scoreBars.map { " (the score has \($0))" } ?? ""))
if !counts.agreesWithScore { print("the bars found are far from the score's: another edition, or other movements?") }

if let truth {
    let exact = zip(counts.bars, truth).filter { $0 == $1 }.count
    print("pages counted exactly: \(exact) of \(truth.count)")
    let trueStart = truth.indices.map { truth[..<$0].reduce(0, +) }
    func wrongStarts() -> [Int] {
        (1...truth.count).filter { page in starts.start(ofPage: page).map { $0 != trueStart[page - 1] } ?? false }
    }
    let before = wrongStarts()
    var checked: [Int] = []
    while let page = starts.unsurePages.first(where: { !checked.contains($0) }) {
        checked.append(page)
        if starts.start(ofPage: page) == trueStart[page - 1] {
            starts.confirm(page: page)
        } else {
            let moved = starts.set(page: page, start: trueStart[page - 1])
            print("  page \(page) corrected; \(moved.count) pages after it moved")
        }
    }
    print("page starts wrong as counted: \(before.count); pages asked about: \(checked.count) \(checked); "
          + "wrong once those are checked: \(wrongStarts().count)")
}
#else
print("barcount needs Core ML")
#endif
