@preconcurrency import AVFoundation
import AppKit
import Observation
import PaddockClient
import PaddockConversationCore

@MainActor @Observable final class NativeReadCamera {
  private(set) var capture: NativeReadCameraCapture?
  private(set) var opening = false
  private(set) var live = false
  private(set) var latest: NativeReadsModel.Run?
  private(set) var rate = 0.0
  var error: String?
  var side = 512
  var deviceID = ""
  var devices: [(id: String, name: String)] = []
  @ObservationIgnored private var generation = UUID()
  @ObservationIgnored private var captureGeneration = UUID()
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored private var openingTask: Task<Void, Never>?
  @ObservationIgnored private var inFlight = false
  @ObservationIgnored weak var window: NSWindow?

  func open(resume model: NativeReadsModel? = nil) {
    let resume = live ? model : nil
    close()
    let epoch = captureGeneration
    let resumeEpoch = generation
    opening = true
    openingTask = Task {
      var allowed = AVCaptureDevice.authorizationStatus(for: .video) == .authorized
      if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
        allowed = await AVCaptureDevice.requestAccess(for: .video)
      }
      guard epoch == captureGeneration, !Task.isCancelled else { return }
      guard allowed else {
        error = "Allow Camera access for Paddock in System Settings > Privacy & Security > Camera."
        opening = false
        return
      }
      let source = NativeReadCameraCapture()
      capture = source
      do {
        try await source.start(deviceID: deviceID)
        guard epoch == captureGeneration, !Task.isCancelled else {
          source.stop()
          return
        }
        devices = NativeReadCameraCapture.devices().map { ($0.uniqueID, $0.localizedName) }
      } catch {
        source.stop()
        guard epoch == captureGeneration else { return }
        self.error = error.localizedDescription
        capture = nil
      }
      opening = false
      if let resume, capture != nil, resumeEpoch == generation { start(model: resume) }
    }
  }
  func pause(clear: Bool = false) {
    generation = UUID()
    task?.cancel()
    task = nil
    live = false
    rate = 0
    if clear { latest = nil }
  }
  func close() {
    captureGeneration = UUID()
    pause(clear: true)
    openingTask?.cancel()
    openingTask = nil
    capture?.stop()
    capture = nil
    opening = false
    error = nil
  }
  func start(model: NativeReadsModel) {
    guard !live, !opening, let source = capture, model.canReadCamera else { return }
    startLoop(
      model: model, frame: { try await source.jpeg(longSide: $0) },
      visible: { [weak self] in
        Self.canReadVisibleWindow(
          visible: self?.window?.occlusionState.contains(.visible) == true,
          minimized: self?.window?.isMiniaturized ?? true,
          appHidden: NSApplication.shared.isHidden)
      })
  }
  static func canReadVisibleWindow(visible: Bool, minimized: Bool, appHidden: Bool) -> Bool {
    visible && !minimized && !appHidden
  }
  /// Capture/visibility seams keep cancellation and single-flight tests fully
  /// deterministic without opening a camera or requesting TCC permission.
  func startLoop(
    model: NativeReadsModel,
    frame: @escaping @MainActor (Int) async throws -> Data?,
    visible: @escaping @MainActor () -> Bool
  ) {
    guard !live, model.canReadCamera else { return }
    pause()
    let epoch = generation
    live = true
    error = nil
    task = Task {
      var previous: ContinuousClock.Instant?
      var gap = 0.0
      defer {
        if epoch == generation {
          live = false
          task = nil
        }
      }
      do {
        while epoch == generation, !Task.isCancelled {
          // A background application does not consume the GPU for unseen frames.
          guard visible(), model.canReadCamera, !inFlight else {
            try await Task.sleep(for: .milliseconds(250))
            continue
          }
          let result: NativeReadsModel.Run
          do {
            guard let next = try await nextFrame(model: model, frame: frame) else {
              try await Task.sleep(for: .milliseconds(100))
              continue
            }
            result = next
          } catch is CancellationError {
            if Task.isCancelled || epoch != generation { return }
            continue  // edited questions/read: ask the next frame, never publish stale answers
          }
          guard epoch == generation, !Task.isCancelled else { return }
          latest = result
          let now = ContinuousClock.now
          if let previous {
            let d = previous.duration(to: now)
            let seconds = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
            gap = gap == 0 ? seconds : gap * 0.7 + seconds * 0.3
            rate = 1 / max(gap, 0.001)
          }
          previous = now
        }
      } catch is CancellationError {} catch {
        if epoch == generation { self.error = error.localizedDescription }
      }
    }
  }
  private func nextFrame(
    model: NativeReadsModel,
    frame: @MainActor (Int) async throws -> Data?
  ) async throws -> NativeReadsModel.Run? {
    // This lease survives Stop/Start and model switches until the old transport
    // actually returns, even when it does not cooperate with cancellation.
    inFlight = true
    defer { inFlight = false }
    guard let bytes = try await frame(side) else { return nil }
    try Task.checkCancellation()
    return try await model.readCameraFrame(bytes)
  }
}
