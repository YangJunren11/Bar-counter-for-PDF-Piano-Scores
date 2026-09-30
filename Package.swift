// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "BarCounter",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "BarCounter", targets: ["BarCounter"]),
        .executable(name: "barcount", targets: ["barcount"]),
    ],
    targets: [
        // Plain Swift: the staff space, the peaks, the counts and the page starts. The Core ML part,
        // PDFBarCounter, is compiled only where Core ML is.
        .target(name: "BarCounter"),
        // Counts the bars on a PDF's pages from the command line, on a Mac
        .executableTarget(name: "barcount", dependencies: ["BarCounter"]),
        .testTarget(name: "BarCounterTests", dependencies: ["BarCounter"]),
    ]
)
