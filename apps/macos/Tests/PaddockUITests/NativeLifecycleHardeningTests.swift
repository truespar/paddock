import AppKit
import Foundation
import PaddockClient
import PaddockDesign
import Testing

@testable import PaddockNativeMarkdown
@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native lifecycle hardening", .serialized, .timeLimit(.minutes(1)))
struct NativeLifecycleHardeningTests {
  @Test @MainActor func closedReadsRejectsALateHostWithoutPublishingAnError() async throws {
    let manager = LateHost()
    let reads = NativeReadsModel(client: manager)
    let refresh = Task { await reads.refresh() }
    await manager.wait()
    await reads.shutdown()
    try await manager.resolve()
    await refresh.value
    #expect(reads.readers.isEmpty && reads.error == nil && !reads.loading)
    await reads.refresh()
    #expect(await manager.calls == 1)
  }
  @Test @MainActor func shutdownRejectsAnUncooperativeLateHost() async throws {
    let manager = LateHost()
    let workspace = StudioWorkspace(client: manager)
    let start = Task { await workspace.start() }
    await manager.wait()
    let stop = Task { await workspace.shutdown() }
    // shutdown sets its terminal state synchronously before waiting on boot.
    while workspace.restoring { await Task.yield() }
    try await manager.resolve()
    await start.value
    await stop.value
    #expect(!workspace.ready && !workspace.restoring && workspace.state == nil)
    #expect(workspace.runtime == nil && workspace.nativeTransport == nil)
    #expect(workspace.error == nil && !workspace.hasWebViewer)
    await workspace.start()
    #expect(await manager.calls == 1)
  }
  @Test func cancelledMarkdownNeverEntersTheParserOrCache() async throws {
    let worker = NativeMarkdownPreparation()
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      return try await worker.source(
        String(repeating: "<div>hello</div>\n", count: 10000), streaming: false)
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await worker.retainedBytes == 0)
    let text = "<iframe src=\"https://example.com\"></iframe>"
    #expect(try await worker.source(text, streaming: false) == NativeMarkdownPolicy.source(text))
    #expect(await worker.retainedBytes > 0)
    await worker.reclaim()
    #expect(await worker.retainedBytes == 0)
  }
  @Test func markdownCacheIsByteBoundedAndStreamingDoesNotAccumulate() async throws {
    let worker = NativeMarkdownPreparation()
    for i in 0..<20 {
      _ = try await worker.source(
        String(repeating: "text ", count: 30000) + "\(i)", streaming: false)
    }
    #expect(await worker.retainedBytes <= 2 * 1024 * 1024)
    await worker.reclaim()
    _ = try await worker.source("streaming", streaming: true)
    #expect(await worker.retainedBytes == 0)
  }
  @Test func cancellingMountAdmissionDoesNotConsumeFutureSlots() async throws {
    let queue = NativeMarkdownMounts()
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      try await queue.admit(visible: false)
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    try await queue.admit(visible: true)
  }
  @Test @MainActor func delayedDiagramAdmissionCannotAuthorizeAnIncompleteHeight() async throws {
    let worker = NativeMarkdownPreparation()
    let prepared = try await worker.prepared(
      "```mermaid\ngraph LR\nA --> B\n```\n\n```swift\nlet mermaid = 1\n```",
      streaming: false)
    #expect(prepared.diagramSources == ["graph LR\nA --> B"])
    let work = NativeMarkdownWork()
    work.expect(prepared.diagramSources)
    #expect(work.remaining > 0)  // The inline host's task has not started yet.
    let source = try #require(prepared.diagramSources.first)
    work.begin(source)
    work.end(source, completed: false)
    #expect(work.remaining > 0)  // Cancellation is not a completed layout.
    work.begin(source)
    work.end(source, completed: true)
    #expect(work.remaining == 0)
    work.beginMount()
    #expect(work.remaining > 0)  // Previous layout is not a new host's geometry.
    work.laidOut(source)
    #expect(work.remaining == 0)
    work.expect([])
    #expect(work.remaining == 0)
  }
  @Test @MainActor func highContrastHasOpaqueStrongerBordersWithoutRecoloringWarnings() {
    for dark in [false, true] {
      let border = PaddockAppearance.nsColor("border", dark: dark, increasedContrast: true)
      #expect(border.alphaComponent == 1)
      #expect(border != PaddockAppearance.nsColor("border", dark: dark))
      #expect(
        PaddockAppearance.nsColor("caution", dark: dark, increasedContrast: true)
          == PaddockAppearance.nsColor("caution", dark: dark))
    }
  }
}

private actor LateHost: ManagerLoading {
  private var continuation: CheckedContinuation<StudioHost, any Error>?
  private var waiter: CheckedContinuation<Void, Never>?
  private(set) var calls = 0
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.closed }
  func nativeConversationHost() async throws -> StudioHost {
    calls += 1
    return try await withCheckedThrowingContinuation { continuation in
      self.continuation = continuation
      waiter?.resume()
      waiter = nil
    }
  }
  func wait() async {
    if continuation != nil { return }
    await withCheckedContinuation { waiter = $0 }
  }
  func resolve() throws {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43210", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    continuation?.resume(returning: host)
    continuation = nil
  }
}
