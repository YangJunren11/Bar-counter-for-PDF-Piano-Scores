// Renders every page of a PDF to an 8-bit grayscale PNG with PDFKit, as an iPad app would see it.
//
//   swiftc -O -o render_pages tools/render_pages.swift
//   render_pages score.pdf out_dir 150        # dots per inch; pages written as p001.png, p002.png, ...
import AppKit
import PDFKit

let arguments = CommandLine.arguments
guard arguments.count >= 3, let document = PDFDocument(url: URL(fileURLWithPath: arguments[1])) else {
    FileHandle.standardError.write("usage: render_pages file.pdf out_dir [dpi]\n".data(using: .utf8)!)
    exit(1)
}
let out = URL(fileURLWithPath: arguments[2], isDirectory: true)
try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
let dpi = arguments.count > 3 ? Double(arguments[3]) ?? 150 : 150

for index in 0..<document.pageCount {
    guard let page = document.page(at: index) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let width = Int((bounds.width / 72 * dpi).rounded()), height = Int((bounds.height / 72 * dpi).rounded())
    guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
    else { continue }
    context.setFillColor(gray: 1, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.scaleBy(x: CGFloat(dpi / 72), y: CGFloat(dpi / 72))
    context.translateBy(x: -bounds.minX, y: -bounds.minY)
    page.draw(with: .mediaBox, to: context)
    guard let image = context.makeImage() else { continue }
    let rep = NSBitmapImageRep(cgImage: image)
    let name = String(format: "p%03d.png", index + 1)
    try rep.representation(using: .png, properties: [:])!.write(to: out.appendingPathComponent(name))
}
print(document.pageCount)
