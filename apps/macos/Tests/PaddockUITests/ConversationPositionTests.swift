import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Conversation position restoration", .serialized) @MainActor
struct ConversationPositionTests {
  @Test func exactOffsetsSurviveGrowthAboveAndReflowUsesOnlyTheAnchoredMessage() {
    let memory = StudioConversationPosition(id: "chat")
    memory.reading(
      .init(following: false, blockID: "second", index: 1, offset: 200, height: 1000, width: 600))
    let intent = TranscriptScrollIntent(memory: memory)
    intent.retainRows(["first", "second"])
    intent.didLayout("first", frame: CGRect(x: 0, y: 20, width: 600, height: 700))
    intent.didLayout("second", frame: CGRect(x: 0, y: 748, width: 600, height: 1000))
    #expect(intent.restorationY() == 948)
    intent.didLayout("second", frame: CGRect(x: 0, y: 900, width: 600, height: 1200))
    #expect(intent.restorationY() == 1100)
    intent.didLayout("second", frame: CGRect(x: 0, y: 1000, width: 400, height: 1500))
    #expect(intent.restorationY() == 1300)
    intent.reveal()
    #expect(intent.restorationY() == nil)
    intent.capture(y: 1400)
    #expect(memory.value.reading?.offset == 400 && memory.value.reading?.blockID == "second")
    intent.pinned = true
    #expect(memory.value.reading?.following == true && intent.restorationY() == nil)
  }

  @Test func deletedBranchAnchorFallsBackAndRecordsStayIndependent() {
    let first = StudioConversationPosition(id: "a")
    let second = StudioConversationPosition(id: "b")
    first.reading(
      .init(following: false, blockID: "deleted", index: 99, offset: 120, height: 300, width: 600))
    let intent = TranscriptScrollIntent(memory: first)
    intent.retainRows(["replacement"])
    intent.didLayout("replacement", frame: CGRect(x: 0, y: 20, width: 600, height: 400))
    #expect(intent.restorationY() == 140)
    second.reading(.init(following: true))
    #expect(first.value.reading?.following == false)
    #expect(TranscriptScrollIntent(memory: second).pinned)
  }

  @Test func composerRestoresUnicodeSelectionsAndRejectsAChangedDraft() {
    let text = "Hi 🧑🏽‍💻 café next"
    let range = (text as NSString).range(of: "🧑🏽‍💻 café")
    let position = StudioConversationViewState.Composer(selection: range, text: text)
    #expect(position.selection(in: text) == range)
    #expect(position.selection(in: "Different draft") == nil)
    var invalid = position
    invalid.location = 100_000
    invalid.length = 100_000
    #expect(invalid.selection(in: text) == NSRange(location: (text as NSString).length, length: 0))
    let record = StudioConversationPosition(id: "selection")
    record.composer(NSRange(location: NSNotFound, length: Int.max), text: text)
    #expect(
      record.value.composer?.selection(in: text)
        == NSRange(location: (text as NSString).length, length: 0))
  }

  @Test func serializedSavesDrainNewerChangesAndKeepFailuresForRetry() async throws {
    let disk = PositionDisk()
    let store = StudioConversationPositions(
      load: { await disk.get($0) }, save: { try await disk.put($0, $1) })
    try await store.prepare("a")
    let a = store.record("a")
    a.reading(.init(following: false, blockID: "one", offset: 10, height: 400, width: 600))
    let first = Task { try await store.flush() }
    try await wait { await disk.started > 0 }
    a.reading(.init(following: false, blockID: "two", offset: 20, height: 400, width: 600))
    async let second: Void = store.flush()
    try await first.value
    try await second
    #expect(await disk.maximumConcurrent == 1)
    #expect(await disk.get("a")?.reading?.blockID == "two")
    var reportedFailure = false
    var reportedRecovery = false
    store.onError = { _ in reportedFailure = true }
    store.onSaved = { reportedRecovery = true }
    await disk.failWrites(true)
    a.composer(NSRange(location: 3, length: 2), text: "draft text")
    await #expect(throws: (any Error).self) { try await store.flush() }
    #expect(a.revision != a.savedRevision)
    #expect(reportedFailure && !reportedRecovery)
    await disk.failWrites(false)
    try await store.flush()
    #expect(reportedRecovery)
    let reopened = StudioConversationPositions(
      load: { await disk.get($0) }, save: { try await disk.put($0, $1) })
    try await reopened.prepare("a")
    #expect(reopened.record("a").value == a.value)
    await store.close()
    await reopened.close()
  }

  @Test func nativeEditorRestoresSelectionAfterRemountWithoutTakingFocus() async throws {
    let memory = StudioConversationPosition(id: "a")
    let text = "Draft with 🦊 and selection"
    let fixture = mount {
      StudioDraftEditor(text: .constant(text), onSend: {}, position: memory).frame(height: 100)
    }
    let first = try #require(descendants(fixture.root).compactMap { $0 as? DraftTextView }.first)
    let range = (text as NSString).range(of: "🦊 and")
    first.setSelectedRange(range)
    try await wait { memory.value.composer?.selection(in: text) == range }
    fixture.window.close()
    let next = mount {
      StudioDraftEditor(text: .constant(text), onSend: {}, position: memory).frame(height: 100)
    }
    defer { next.window.close() }
    let second = try #require(descendants(next.root).compactMap { $0 as? DraftTextView }.first)
    #expect(second.selectedRange() == range)
    #expect(next.window.firstResponder !== second)
  }

  @Test func actualTranscriptReturnsToTheSameOffsetAndDoesNotJumpOnAppend() async throws {
    let memory = StudioConversationPosition(id: "scroll")
    let model = PositionTranscriptFixture(memory: memory)
    let first = mount { PositionTranscriptView(model: model) }
    try await wait { verticalScroll(first.root)?.documentView?.frame.height ?? 0 > 2500 }
    let scroll = try #require(verticalScroll(first.root))
    var previousHeight: CGFloat = 0
    var stable = 0
    try await wait {
      let height = scroll.documentView?.frame.height ?? 0
      stable = height == previousHeight ? stable + 1 : 0
      previousHeight = height
      return stable >= 15
    }
    scroll.contentView.scroll(to: NSPoint(x: 0, y: 900))
    scroll.reflectScrolledClipView(scroll.contentView)
    try await wait { memory.value.reading?.following == false }
    try await Task.sleep(for: .milliseconds(200))
    let original = scroll.documentVisibleRect.minY
    #expect(abs(original - 900) < 2)
    first.window.close()
    let next = mount { PositionTranscriptView(model: model) }
    defer { next.window.close() }
    try await wait {
      abs((verticalScroll(next.root)?.documentVisibleRect.minY ?? -1000) - original) < 2
    }
    model.count += 1
    try await Task.sleep(for: .milliseconds(250))
    #expect(abs((verticalScroll(next.root)?.documentVisibleRect.minY ?? -1000) - original) < 2)
    #expect(memory.value.reading?.following == false)
  }

  private func wait(_ predicate: () async -> Bool) async throws {
    for _ in 0..<300 {
      if await predicate() { return }
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(await predicate(), "Position restoration did not settle")
  }
  private func descendants(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(descendants)
  }
  private func verticalScroll(_ root: NSView) -> NSScrollView? {
    descendants(root).compactMap { $0 as? NSScrollView }.first { $0.hasVerticalScroller }
  }
  private func mount<Content: View>(@ViewBuilder content: () -> Content) -> (
    window: NSWindow, root: NSView
  ) {
    _ = NSApplication.shared
    let root = NSHostingView(rootView: content())
    let window = NSWindow(
      contentRect: CGRect(x: -12000, y: -12000, width: 760, height: 600), styleMask: [.borderless],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = root
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    root.layoutSubtreeIfNeeded()
    return (window, root)
  }
}

private actor PositionDisk {
  var records: [String: StudioConversationViewState] = [:]
  var fail = false
  var started = 0
  var concurrent = 0
  var maximumConcurrent = 0
  func failWrites(_ value: Bool) { fail = value }
  func get(_ id: String) -> StudioConversationViewState? { records[id] }
  func put(_ id: String, _ value: StudioConversationViewState) async throws {
    started += 1
    concurrent += 1
    maximumConcurrent = max(maximumConcurrent, concurrent)
    defer { concurrent -= 1 }
    try await Task.sleep(for: .milliseconds(30))
    if fail { throw ManagerError.core("Synthetic disk failure") }
    records[id] = value
  }
}

@MainActor @Observable private final class PositionTranscriptFixture {
  let memory: StudioConversationPosition
  var count = 30
  init(memory: StudioConversationPosition) { self.memory = memory }
  var transcript: StudioState.NativeTranscript {
    let messages: [[String: Any]] = (0..<count).map { i in
      [
        "id": "m\(i)", "role": "assistant",
        "text": String(
          repeating: "Message \(i) has text that wraps across several lines. ", count: 25),
        "reasoning": "", "model": "fixture", "streaming": false, "stopped": false, "error": "",
        "incomplete": false,
      ]
    }
    let data = try! JSONSerialization.data(withJSONObject: [
      "available": true, "notice": "", "conversationId": "scroll", "messages": messages,
    ])
    return try! JSONDecoder().decode(StudioState.NativeTranscript.self, from: data)
  }
}
private struct PositionTranscriptView: View {
  @Bindable var model: PositionTranscriptFixture
  var body: some View {
    NativeStudioTranscript(
      transcript: model.transcript, columnWidth: 600, composerHeight: 110,
      composer: AnyView(Color.gray.frame(height: 110)), position: model.memory)
  }
}
