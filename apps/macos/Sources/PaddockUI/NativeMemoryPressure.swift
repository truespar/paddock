import Foundation

/// Kernel notification, not a polling timer. Reclaims only reconstructible UI
/// caches; loaded models, unsaved results, drafts and playback remain untouched.
@MainActor final class NativeMemoryPressure {
  private let source: any DispatchSourceMemoryPressure
  init(reclaim: @escaping @MainActor @Sendable () async -> Void) {
    source = DispatchSource.makeMemoryPressureSource(
      eventMask: [.warning, .critical], queue: .main)
    source.setEventHandler { Task { @MainActor in await reclaim() } }
    source.activate()
  }
  func stop() { source.cancel() }
  deinit { source.cancel() }
}
