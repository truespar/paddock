import Foundation

/// Shared across compare lanes. Admit one new rich-text host per display-sized
/// interval; prioritize visible messages over cold offscreen measurement.
actor NativeMarkdownMounts {
  static let shared = NativeMarkdownMounts()
  private struct Waiter {
    let id: UUID
    let visible: Bool
    let continuation: CheckedContinuation<Void, any Error>
  }
  private var queue: [Waiter] = []
  private var pump: Task<Void, Never>?
  func admit(visible: Bool) async throws {
    try Task.checkCancellation()
    let id = UUID()
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        queue.append(Waiter(id: id, visible: visible, continuation: continuation))
        if pump == nil { pump = Task { await drain() } }
      }
      try Task.checkCancellation()
    } onCancel: {
      Task { await self.cancel(id) }
    }
  }
  private func cancel(_ id: UUID) {
    guard let index = queue.firstIndex(where: { $0.id == id }) else { return }
    queue.remove(at: index).continuation.resume(throwing: CancellationError())
  }
  private func drain() async {
    while !queue.isEmpty {
      let index = queue.firstIndex(where: \.visible) ?? 0
      queue.remove(at: index).continuation.resume()
      try? await Task.sleep(for: .milliseconds(16))
    }
    pump = nil
  }
}
