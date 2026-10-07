import AppKit
import ImageIO
import PDFKit
import Testing
import UniformTypeIdentifiers

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native document image lifecycle", .serialized)
struct NativeDocumentMediaTests {
  func imageFixture() throws -> URL {
    let context = try #require(
      CGContext(
        data: nil, width: 3000, height: 2000, bitsPerComponent: 8,
        bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 3000, height: 2000))
    let image = try #require(context.makeImage())
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Paddock-document-test-\(UUID()).jpg")
    let output = try #require(
      CGImageDestinationCreateWithURL(file as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(output, image, [kCGImagePropertyOrientation: 6] as CFDictionary)
    #expect(CGImageDestinationFinalize(output))
    return file
  }
  @Test func cropGeometryRejectsInvalidRegions() {
    #expect(
      NativeDocumentCrop.rectangle([0, 0, 999, 999], width: 1200, height: 800)
        == CGRect(x: 0, y: 0, width: 1200, height: 800))
    #expect(
      NativeDocumentCrop.rectangle([0, 0, 499.5, 499.5], width: 1200, height: 800)
        == CGRect(x: 0, y: 0, width: 600, height: 400))
    for box in [
      [Double.nan, 0, 2, 3], [0, 0, .infinity, 3], [0, 0, 1000, 2], [4, 5, 2, 3], [0, 0, 0, 2],
      [0, 1, 2],
    ] {
      #expect(NativeDocumentCrop.rectangle(box, width: 10, height: 10) == nil)
    }
  }
  @Test func imageDecodeRespectsEXIFAndPDFDecodeRespectsCropRotation() async throws {
    let file = try imageFixture()
    defer { try? FileManager.default.removeItem(at: file) }
    let decoder = NativeDocumentImageDecoder()
    let image = try await decoder.decode(file, page: nil)
    #expect(image.width == 1024 && image.height == 1536)
    let pdf = try #require(PDFDocument(data: NativePDFDocumentTests().fixture()))
    let page = try #require(pdf.page(at: 196))
    page.rotation = 90
    page.setBounds(CGRect(x: 20, y: 40, width: 200, height: 400), for: .cropBox)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Paddock-document-test-\(UUID()).pdf")
    defer { try? FileManager.default.removeItem(at: url) }
    try #require(pdf.dataRepresentation()).write(to: url)
    let raster = try await decoder.decode(url, page: 197)
    #expect(raster.width * raster.height <= 1_500_000)
    #expect(abs(Double(raster.width) / Double(raster.height) - 2) < 0.01)
    await #expect(throws: (any Error).self) { _ = try await decoder.decode(url, page: 201) }
  }
  @Test @MainActor func compareReadersShareWorkAndCacheIsBoundedAndResettable() async throws {
    let media = StudioDocumentMedia()
    var fetches = 0
    var active = 0
    var peak = 0
    let fetch: @MainActor () async throws -> URL = {
      fetches += 1
      active += 1
      peak = max(peak, active)
      defer { active -= 1 }
      try await Task.sleep(for: .milliseconds(30))
      return try imageFixture()
    }
    let key = StudioDocumentMedia.Key(attachmentID: "same", pdfPage: nil)
    async let a = media.image(key, fetch: fetch)
    async let b = media.image(key, fetch: fetch)
    let (first, second) = try await (a, b)
    #expect(first === second && fetches == 1)
    _ = try await media.image(key, fetch: fetch)
    #expect(fetches == 1)
    for index in 0..<8 {
      _ = try await media.image(.init(attachmentID: "page-\(index)", pdfPage: nil), fetch: fetch)
      #expect(media.retainedBytes <= 16 * 1024 * 1024)
    }
    #expect(peak == 1)
    media.reset()
    #expect(media.retainedBytes == 0)
    _ = try await media.image(key, fetch: fetch)
    #expect(fetches == 10)
  }
  @Test @MainActor func cancelledReadersReleaseAdmissionAndDiscardResults() async throws {
    let media = StudioDocumentMedia()
    var started = false
    let task = Task {
      try await media.image(.init(attachmentID: "cancelled", pdfPage: nil)) {
        started = true
        try await Task.sleep(for: .seconds(30))
        return try imageFixture()
      }
    }
    while !started { await Task.yield() }
    task.cancel()
    do {
      _ = try await task.value
      Issue.record("Cancelled preview returned an image")
    } catch is CancellationError {} catch { Issue.record("\(error)") }
    #expect(media.retainedBytes == 0)
    let image = try await media.image(.init(attachmentID: "next", pdfPage: nil)) {
      try imageFixture()
    }
    #expect(image.height == 1536)
  }
  @Test @MainActor func memoryPressureDoesNotCancelVisibleImageWork() async throws {
    let media = StudioDocumentMedia()
    var pending: CheckedContinuation<Void, Never>?
    let request = Task {
      try await media.image(.init(attachmentID: "visible", pdfPage: nil)) {
        await withCheckedContinuation { pending = $0 }
        return try imageFixture()
      }
    }
    while pending == nil { await Task.yield() }
    media.reclaim()
    #expect(media.retainedBytes == 0)
    pending?.resume()
    let image = try await request.value
    media.reclaim()
    #expect(media.retainedBytes == 0 && image.width > 0)
  }

  @Test @MainActor func cancellingOneCompareReaderDoesNotCancelItsNeighbor() async throws {
    let media = StudioDocumentMedia()
    var fetches = 0
    let fetch: @MainActor () async throws -> URL = {
      fetches += 1
      try await Task.sleep(for: .milliseconds(80))
      return try imageFixture()
    }
    let key = StudioDocumentMedia.Key(attachmentID: "shared", pdfPage: nil)
    let first = Task { try await media.image(key, fetch: fetch) }
    let second = Task { try await media.image(key, fetch: fetch) }
    try await Task.sleep(for: .milliseconds(20))
    first.cancel()
    let surviving = try await second.value
    #expect(surviving.height == 1536 && fetches == 1)
    do {
      _ = try await first.value
      Issue.record("Cancelled reader returned pixels")
    } catch is CancellationError {} catch { Issue.record("\(error)") }
    media.reset()
    #expect(media.retainedBytes == 0)
  }
}
