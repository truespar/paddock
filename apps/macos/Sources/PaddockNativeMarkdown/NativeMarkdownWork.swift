import SwiftUI

/// Retained with the message shell, outside the evictable rich-text subtree.
/// Async attachments finish layout before a height can become a placeholder.
@MainActor @Observable final class NativeMarkdownWork {
  @ObservationIgnored weak var surface: SelectionHostingView?
  var pending = 0
  private var expected = Set<String>()
  private var completed = Set<String>()
  var remaining: Int { pending + expected.subtracting(completed).count }
  func expect(_ sources: Set<String>) {
    expected = sources
    completed.formIntersection(sources)
  }
  func beginMount() { completed.removeAll(keepingCapacity: true) }
  func begin(_ source: String) {
    completed.remove(source)
    pending += 1
  }
  func end(_ source: String, completed: Bool) {
    pending -= 1
    if completed { self.completed.insert(source) }
  }
  func laidOut(_ source: String) { completed.insert(source) }
  var pins = Set<UUID>()
  var pinned: Bool { !pins.isEmpty }
}
extension EnvironmentValues {
  @Entry var nativeMarkdownWork: NativeMarkdownWork? = nil
}
