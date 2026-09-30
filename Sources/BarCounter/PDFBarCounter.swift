#if canImport(CoreML) && canImport(CoreGraphics)  // Apple's devices; see BarCounting for the rest
import CoreGraphics
import CoreML
import Foundation

/// Counts the bars on the pages of a PDF with the network in `model/BarCounter.mlpackage`.
///
/// Each page is drawn twice: once at 3 pixels a point to measure its staff space, and again at the
/// size where a staff space is 9 pixels, which is how the network was trained to see pages. Drawing
/// again from the PDF rather than shrinking the first picture keeps engraved scores sharp; a scan is
/// resampled from its own pixels either way.
public final class PDFBarCounter {
    private let url: URL
    private var model: MLModel
    private var cpuOnly: Bool

    /// `model` is the compiled network: `BarCounter.mlmodelc` in an app's bundle, or what
    /// `MLModel.compileModel(at:)` makes of `BarCounter.mlpackage`.
    public init(model url: URL) throws {
        self.url = url
        // The simulator's GPU gives back nothing but zeros for this network; its CPU is right
        #if targetEnvironment(simulator)
        cpuOnly = true
        #else
        cpuOnly = false
        #endif
        model = try Self.load(url, cpuOnly: cpuOnly)
    }

    private static func load(_ url: URL, cpuOnly: Bool) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = cpuOnly ? .cpuOnly : .all
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    public enum Failure: Error { case notAPDF, cannotDraw, stopped }

    /// The peaks on every page of a PDF (`BarCounting.peaks`), page 1 first. `progress` is told how
    /// many pages are done and of how many, after each, and stops the count by returning false.
    public func peaks(inPDF url: URL, progress: (Int, Int) -> Bool = { _, _ in true }) throws -> [[Float]] {
        guard let document = CGPDFDocument(url as CFURL), document.numberOfPages > 0 else { throw Failure.notAPDF }
        var pages: [[Float]] = []
        for number in 1...document.numberOfPages {
            try autoreleasepool {
                guard let page = document.page(at: number) else { throw Failure.cannotDraw }
                pages.append(try peaks(on: page))
            }
            guard progress(number, document.numberOfPages) else { throw Failure.stopped }
        }
        return pages
    }

    /// A page as the network is to see it: grey, row by row from the top, 0 black and 255 white,
    /// scaled so a staff space is `BarCounting.space` pixels; and the staff space it was measured at.
    public struct Prepared {
        public let pixels: [UInt8]
        public let width: Int
        public let height: Int
        public let measuredSpace: Double?
    }

    public static func prepare(_ page: CGPDFPage) throws -> Prepared {
        let first = try draw(page, pixelsPerPoint: 3)
        let space = BarCounting.staffSpace(gray: first.pixels, width: first.width, height: first.height)
        let scale = BarCounting.scale(forStaffSpace: space, width: first.width)
        let drawn = try draw(page, pixelsPerPoint: 3 * scale)
        return Prepared(pixels: drawn.pixels, width: drawn.width, height: drawn.height, measuredSpace: space)
    }

    /// The peaks on one page.
    public func peaks(on page: CGPDFPage) throws -> [Float] {
        try peaks(on: Self.prepare(page))
    }

    public func peaks(on page: Prepared) throws -> [Float] {

        // Ink 1 and paper 0, padded with paper to a multiple of 32 each way
        let width = (page.width + 31) / 32 * 32, height = (page.height + 31) / 32 * 32
        let input = try MLMultiArray(shape: [1, 1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let values = input.dataPointer.bindMemory(to: Float.self, capacity: width * height)
        let rowStride = input.strides[2].intValue
        for y in 0..<height {
            for x in 0..<width {
                values[y * rowStride + x] = y < page.height && x < page.width
                    ? 1 - Float(page.pixels[y * page.width + x]) / 255 : 0
            }
        }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["page": input]))
        guard let bars = output.featureValue(for: "bars")?.multiArrayValue else { throw Failure.cannotDraw }
        let grid = MLShapedArray<Float>(converting: bars)
        // Nowhere above zero is no answer at all (an empty page still comes out near 0.02): what the
        // simulator's GPU gives. Should a device ever do the same, its CPU is asked instead.
        if !cpuOnly, (grid.scalars.max() ?? 0) == 0 {
            print("BarCounter: the network gave nothing back; counting on the CPU")
            cpuOnly = true
            model = try Self.load(url, cpuOnly: true)
            return try peaks(on: page)
        }
        let shape = grid.shape
        return BarCounting.peaks(grid.scalars, width: shape[3], height: shape[2])
    }

    /// A page in 8-bit grey, row by row from the top, as it is shown: turned as the PDF says to.
    static func draw(_ page: CGPDFPage, pixelsPerPoint scale: Double) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let box = page.getBoxRect(.cropBox)
        let turn = ((Int(page.rotationAngle) % 360) + 360) % 360
        let (shownWidth, shownHeight) = turn == 90 || turn == 270 ? (box.height, box.width) : (box.width, box.height)
        let width = max(1, Int((shownWidth * scale).rounded())), height = max(1, Int((shownHeight * scale).rounded()))
        var pixels = [UInt8](repeating: 255, count: width * height)
        try pixels.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { throw Failure.cannotDraw }
            context.interpolationQuality = .high
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.scaleBy(x: CGFloat(width) / shownWidth, y: CGFloat(height) / shownHeight)
            // /Rotate turns the page clockwise for showing
            switch turn {
            case 90: context.translateBy(x: 0, y: box.width); context.rotate(by: -.pi / 2)
            case 180: context.translateBy(x: box.width, y: box.height); context.rotate(by: .pi)
            case 270: context.translateBy(x: box.height, y: 0); context.rotate(by: .pi / 2)
            default: break
            }
            context.translateBy(x: -box.minX, y: -box.minY)
            context.clip(to: box)
            context.drawPDFPage(page)
        }
        return (pixels, width, height)
    }
}
#endif
