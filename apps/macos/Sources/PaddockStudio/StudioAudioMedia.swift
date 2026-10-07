import Foundation

/// Owns only a downloaded, task-private copy. References from decoders keep it
/// alive through cache eviction; neither eviction nor playback can delete the
/// permanent attachment. A playback hard link gives AVPlayer its own lifetime.
private final class AudioSource: Sendable {
  let url: URL
  let bytes: Int
  init(_ url: URL) {
    self.url = url
    bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
  }
  deinit { try? FileManager.default.removeItem(at: url) }
  func playbackCopy() throws -> URL {
    let destination = FileManager.default.temporaryDirectory
      .appendingPathComponent("Paddock-playback-\(UUID().uuidString).\(url.pathExtension)")
    do { try FileManager.default.linkItem(at: url, to: destination) } catch {
      try FileManager.default.copyItem(at: url, to: destination)
    }
    return destination
  }
}

/// Shared fetch/decode by attachment ID. Limits are explicit: two fetches,
/// one waveform analysis, two/128 MiB cached originals and 32 peak summaries.
/// Previewing a clip never replaces the currently playing AVPlayer item.
@MainActor public final class StudioAudioMedia {
  private let decoder = AudioWaveformDecoder()
  private let fetchGate = StudioMediaWorkGate(capacity: 2)
  private let analysisGate = StudioMediaWorkGate(capacity: 1)
  private var sources: [String: AudioSource] = [:]
  private var sourceOrder: [String] = []
  private struct SourceJob {
    let id = UUID()
    let task: Task<AudioSource, any Error>
  }
  private var sourceJobs: [String: SourceJob] = [:]
  private var peaks: [String: AudioWaveform] = [:]
  private var peakOrder: [String] = []
  private struct Analysis {
    let id: UUID
    let task: Task<AudioWaveform?, Never>
    var readers: Set<UUID>
  }
  private var analyses: [String: Analysis] = [:]
  private var epoch = 0
  public init() {}

  private func source(_ id: String, fetch: @escaping @MainActor () async throws -> URL) async throws
    -> AudioSource
  {
    try Task.checkCancellation()
    if let source = sources[id] {
      touch(id, in: &sourceOrder)
      return source
    }
    let ticket = epoch
    let job: SourceJob
    if let pending = sourceJobs[id] {
      job = pending
    } else {
      job = SourceJob(
        task: Task { [fetchGate] in
          try await fetchGate.acquire()
          defer { Task { await fetchGate.release() } }
          try Task.checkCancellation()
          let file = AudioSource(try await fetch())
          try Task.checkCancellation()
          return file
        })
      sourceJobs[id] = job
    }
    do {
      let file = try await job.task.value
      guard ticket == epoch else { throw CancellationError() }
      if sourceJobs[id]?.id == job.id { sourceJobs[id] = nil }
      sources[id] = file
      touch(id, in: &sourceOrder)
      while sources.count > 2 || sources.values.reduce(0, { $0 + $1.bytes }) > 128 * 1024 * 1024 {
        guard let first = sourceOrder.first else { break }
        sourceOrder.removeFirst()
        sources[first] = nil
      }
      try Task.checkCancellation()
      return file
    } catch {
      if ticket == epoch, sourceJobs[id]?.id == job.id { sourceJobs[id] = nil }
      throw error
    }
  }

  public func playbackCopy(id: String, fetch: @escaping @MainActor () async throws -> URL)
    async throws -> URL
  {
    let file = try await source(id, fetch: fetch)
    return try await Task.detached(priority: .userInitiated) { try file.playbackCopy() }.value
  }

  public func waveform(id: String, fetch: @escaping @MainActor () async throws -> URL) async
    -> AudioWaveform?
  {
    if let value = peaks[id] {
      touch(id, in: &peakOrder)
      return value
    }
    let reader = UUID()
    let ticket = epoch
    let job: Analysis
    if var existing = analyses[id] {
      existing.readers.insert(reader)
      analyses[id] = existing
      job = existing
    } else {
      let task = Task { [weak self, analysisGate, decoder] () -> AudioWaveform? in
        do {
          try await analysisGate.acquire()
          defer { Task { await analysisGate.release() } }
          try Task.checkCancellation()
          guard let self else { return nil }
          let file = try await source(id, fetch: fetch)
          defer { withExtendedLifetime(file) {} }
          return try await decoder.decode(file.url)
        } catch { return nil }  // A waveform failure must never disable playback.
      }
      job = Analysis(id: UUID(), task: task, readers: [reader])
      analyses[id] = job
    }
    return await withTaskCancellationHandler {
      let value = await job.task.value
      releaseReader(reader, clip: id, job: job.id)
      guard ticket == epoch, !Task.isCancelled, let value else { return nil }
      peaks[id] = value
      touch(id, in: &peakOrder)
      while peakOrder.count > 32 { peaks[peakOrder.removeFirst()] = nil }
      return value
    } onCancel: {
      Task { @MainActor [weak self] in self?.releaseReader(reader, clip: id, job: job.id) }
    }
  }

  private func releaseReader(_ reader: UUID, clip: String, job: UUID) {
    guard var entry = analyses[clip], entry.id == job else { return }
    entry.readers.remove(reader)
    if entry.readers.isEmpty {
      entry.task.cancel()
      analyses[clip] = nil
    } else {
      analyses[clip] = entry
    }
  }
  public func reset() {
    epoch += 1
    for job in sourceJobs.values { job.task.cancel() }
    for job in analyses.values { job.task.cancel() }
    sourceJobs.removeAll()
    analyses.removeAll()
    reclaim()
  }
  public func reclaim() {
    // Active decoders/playback hold their own leases on source files.
    sources.removeAll()
    sourceOrder.removeAll()
    peaks.removeAll()
    peakOrder.removeAll()
  }
  private func touch(_ id: String, in order: inout [String]) {
    order.removeAll { $0 == id }
    order.append(id)
  }
}

/// Cancellable admission; queued invisible previews hold neither a downloaded
/// file nor decoded PCM. No sleep/poll loop and no unbounded decode task fanout.
actor StudioMediaWorkGate {
  private var available: Int
  private var queue: [(UUID, CheckedContinuation<Void, any Error>)] = []
  init(capacity: Int) { available = capacity }
  func acquire() async throws {
    try Task.checkCancellation()
    if available > 0 {
      available -= 1
      return
    }
    let id = UUID()
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { queue.append((id, $0)) }
    } onCancel: {
      Task { await self.cancel(id) }
    }
  }
  private func cancel(_ id: UUID) {
    guard let index = queue.firstIndex(where: { $0.0 == id }) else { return }
    queue.remove(at: index).1.resume(throwing: CancellationError())
  }
  func release() {
    if queue.isEmpty { available += 1 } else { queue.removeFirst().1.resume() }
  }
}
