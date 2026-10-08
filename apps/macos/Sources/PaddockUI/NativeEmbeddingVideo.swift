import AVFoundation
import Foundation
import ImageIO
import PaddockConversationCore
import Synchronization
import UniformTypeIdentifiers

/// AVFoundation owns container/codec decoding; Rust owns model preprocessing.
/// Only sampled frames are retained, never a movie-sized Data or frame array.
enum NativeEmbeddingVideo {
  struct Frame: Sendable {
    let timestamp: Double
    let png: Data
  }
  static func sampleTimes(duration: Double) throws -> [Double] {
    guard duration.isFinite, duration > 0, duration < 86400 else {
      throw ConversationFailure.invalid("Choose a finite video shorter than 24 hours.")
    }
    // HF's elected 1 fps, then uniform downsampling to at most 32 frames.
    let count = max(1, Int(duration))
    let limit = min(32, count)
    return (0..<limit).map { i in
      Double(limit == count ? i : i * (count - 1) / (limit - 1))
    }
  }
  static func decode(_ url: URL, byteBudget: Int) async throws -> [Frame] {
    try Task.checkCancellation()
    let asset = AVURLAsset(url: url)
    let duration = try await asset.load(.duration).seconds
    let times = try sampleTimes(duration: duration)
    guard !(try await asset.loadTracks(withMediaType: .video)).isEmpty else {
      throw ConversationFailure.invalid("The file has no video track.")
    }
    let decoder = FrameDecoder(url: url)
    return try await withTaskCancellationHandler {
      defer { decoder.cancel() }
      var result: [Frame] = []
      var total = 0
      for time in times {
        try Task.checkCancellation()
        let decoded = try await decoder.image(
          at: CMTime(seconds: time, preferredTimescale: 60000))
        try Task.checkCancellation()
        let timestamp = decoded.actualTime.seconds
        guard timestamp.isFinite, timestamp >= 0,
          result.last.map({ timestamp > $0.timestamp }) ?? true
        else {
          throw ConversationFailure.invalid("The video's sampled frame timestamps are invalid.")
        }
        let bytes = NSMutableData()
        guard
          let destination = CGImageDestinationCreateWithData(
            bytes, UTType.png.identifier as CFString, 1, nil)
        else { throw ConversationFailure.invalid("Could not encode the video frame.") }
        CGImageDestinationAddImage(destination, decoded.image, nil)
        guard CGImageDestinationFinalize(destination) else {
          throw ConversationFailure.invalid("Could not encode the video frame.")
        }
        total += bytes.length
        guard total <= byteBudget else { throw ConversationFailure.tooLarge }
        result.append(Frame(timestamp: timestamp, png: bytes as Data))
      }
      return result
    } onCancel: {
      decoder.cancel()
    }
  }

  /// Serialize generator access and cancellation without asserting Sendable
  /// conformance on AVFoundation's mutable generator. Exactly one frame is in
  /// flight, and AVFoundation completes its callback even when cancelled.
  private final class FrameDecoder: Sendable {
    private struct State {
      let generator: AVAssetImageGenerator
      var cancelled = false
    }
    private let state: Mutex<State>
    init(url: URL) {
      let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
      generator.appliesPreferredTrackTransform = true
      // Transport bound, not model resize: 32 frames fit the pixel budget.
      generator.maximumSize = CGSize(width: 1024, height: 1024)
      generator.requestedTimeToleranceBefore = .zero
      generator.requestedTimeToleranceAfter = .zero
      state = Mutex(State(generator: generator))
    }
    func image(at time: CMTime) async throws -> (image: CGImage, actualTime: CMTime) {
      try await withCheckedThrowingContinuation { continuation in
        state.withLock { state in
          guard !state.cancelled else {
            continuation.resume(throwing: CancellationError())
            return
          }
          state.generator.generateCGImagesAsynchronously(forTimes: [NSValue(time: time)]) {
            _, image, actualTime, result, error in
            if result == .succeeded, let image {
              continuation.resume(returning: (image, actualTime))
            } else if result == .cancelled {
              continuation.resume(throwing: CancellationError())
            } else {
              continuation.resume(
                throwing: error ?? ConversationFailure.invalid("Could not decode the video frame."))
            }
          }
        }
      }
    }
    func cancel() {
      state.withLock { state in
        state.cancelled = true
        state.generator.cancelAllCGImageGeneration()
      }
    }
  }
}
