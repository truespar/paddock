import CoreGraphics
import Foundation
import ImageIO
import PaddockClient

/// One fetch/decode at a time; shared by compare lanes, cancelled when the last
/// visible reader leaves. At most three/16 MiB decoded pages are retained, never
/// a bitmap per region or the whole PDF. Originals remain in attachment storage.
@MainActor public final class StudioDocumentMedia {
  public struct Key: Hashable, Sendable {
    public let attachmentID: String
    public let pdfPage: Int?
    public init(attachmentID: String, pdfPage: Int?) {
      self.attachmentID = attachmentID
      self.pdfPage = pdfPage
    }
  }
  private struct Job {
    let id = UUID()
    let task: Task<CGImage, any Error>
    var readers: Set<UUID>
  }
  private let gate = StudioMediaWorkGate(capacity: 1)
  private let decoder = NativeDocumentImageDecoder()
  private var jobs: [Key: Job] = [:]
  private var cache: [Key: CGImage] = [:]
  private var order: [Key] = []
  private var epoch = 0
  public init() {}
  public func image(_ key: Key, fetch: @escaping @MainActor () async throws -> URL) async throws
    -> CGImage
  {
    try Task.checkCancellation()
    if let image = cache[key] {
      touch(key)
      return image
    }
    let ticket = epoch
    let reader = UUID()
    let job: Job
    if var existing = jobs[key] {
      existing.readers.insert(reader)
      jobs[key] = existing
      job = existing
    } else {
      job = Job(
        task: Task { [gate, decoder] in
          try await gate.acquire()
          defer { Task { await gate.release() } }
          try Task.checkCancellation()
          let file = try await fetch()
          defer { try? FileManager.default.removeItem(at: file) }
          try Task.checkCancellation()
          return try await decoder.decode(file, page: key.pdfPage)
        }, readers: [reader])
      jobs[key] = job
    }
    return try await withTaskCancellationHandler {
      defer { release(reader, key: key, job: job.id) }
      let image = try await job.task.value
      guard ticket == epoch, !Task.isCancelled else { throw CancellationError() }
      cache[key] = image
      touch(key)
      while cache.count > 3
        || cache.values.reduce(0, { $0 + $1.bytesPerRow * $1.height }) > 16 * 1024 * 1024
      {
        guard !order.isEmpty else { break }
        cache[order.removeFirst()] = nil
      }
      return image
    } onCancel: {
      Task { @MainActor [weak self] in self?.release(reader, key: key, job: job.id) }
    }
  }
  private func touch(_ key: Key) {
    order.removeAll { $0 == key }
    order.append(key)
  }
  private func release(_ reader: UUID, key: Key, job: UUID) {
    guard var entry = jobs[key], entry.id == job else { return }
    entry.readers.remove(reader)
    if entry.readers.isEmpty {
      entry.task.cancel()
      jobs[key] = nil
    } else {
      jobs[key] = entry
    }
  }
  public func reset() {
    epoch += 1
    for job in jobs.values { job.task.cancel() }
    jobs.removeAll()
    reclaim()
  }
  /// Memory pressure must not cancel visible readers or discard source files.
  public func reclaim() {
    cache.removeAll(keepingCapacity: false)
    order.removeAll(keepingCapacity: false)
  }
  var retainedBytes: Int { cache.values.reduce(0, { $0 + $1.bytesPerRow * $1.height }) }
}

actor NativeDocumentImageDecoder {
  func decode(_ file: URL, page: Int?) throws -> CGImage {
    try Task.checkCancellation()
    return try autoreleasepool {
      if let number = page {
        guard number > 0, let pdf = CGPDFDocument(file as CFURL),
          !pdf.isEncrypted || pdf.isUnlocked, number <= pdf.numberOfPages,
          let selected = pdf.page(at: number)
        else {
          throw ManagerError.core("The original PDF page could not be opened")
        }
        return try NativePDFDocument.raster(selected, maxPixels: 1_500_000)
      }
      guard
        let source = CGImageSourceCreateWithURL(
          file as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
        let image = CGImageSourceCreateThumbnailAtIndex(
          source, 0,
          [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1536,
            kCGImageSourceShouldCacheImmediately: true,
          ] as CFDictionary)
      else { throw ManagerError.core("The original image could not be opened") }
      try Task.checkCancellation()
      return image
    }
  }
}
