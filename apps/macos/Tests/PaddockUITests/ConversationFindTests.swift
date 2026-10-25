import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockNativeMarkdown
@testable import PaddockUI

@Suite("Native conversation Find", .serialized) @MainActor
struct ConversationFindTests {
  @Test func searchesRenderedMarkdownCodeTablesUnicodeAndHiddenReasoning() async throws {
    let index = ConversationFindIndex()
    let items: [ConversationSelectionItem] = [
      .init(id: "a", text: "A **CAFÉ** and [café](https://invisible.example)", markdown: true),
      .init(id: "b", text: "```swift\nlet café = 42\n```", markdown: true),
      .init(id: "c", text: "| City |\n| --- |\n| Café |", markdown: true),
      .init(id: "d", text: "Hidden café", includeWhenUnmounted: false, markdown: true),
      .init(id: "e", text: "🧑🏽‍💻 cafe\u{301}"),
    ]
    let result = try await index.search(items, query: "cafe")
    #expect(result.matches.map(\.itemID) == ["a", "a", "b", "c", "d", "e"])
    #expect(result.matches[1].occurrence == 1)
    #expect(try await index.search(items, query: "invisible.example").matches.isEmpty)
    #expect(try await index.search(items, query: "**").matches.isEmpty)
    #expect(try await index.search(items, query: "let café = 42").matches.count == 1)
    let last = try #require(result.matches.last)
    #expect((items[4].text as NSString).substring(with: last.range) == "cafe\u{301}")
    #expect(
      try await index.search(
        [
          .init(id: "soft", text: "A **soft**\nwrap", markdown: true)
        ], query: "soft wrap"
      ).matches.count == 1)
  }

  @Test func toolIndexUsesTheDisplayedRepresentationAndFindEscapeClosesImmediately() async throws {
    let index = ConversationFindIndex()
    let items: [ConversationSelectionItem] = [
      .init(
        id: "tool/arguments", text: #"{"path":"target file","count":3}"#,
        presentation: .toolArguments),
      .init(
        id: "tool/output", text: #"[{"type":"text","text":"result phrase"}]"#,
        presentation: .toolOutput),
    ]
    #expect(try await index.search(items, query: "path: target file").matches.count == 1)
    #expect(try await index.search(items, query: "result phrase").matches.count == 1)
    #expect(try await index.search(items, query: "type").matches.isEmpty)
    let find = ConversationFind()
    find.show()
    find.search("keep query")
    let field = NativeSearchField(
      text: Binding(get: { find.query }, set: { find.search($0) }),
      prompt: "Find in conversation", cancelClearsText: false, onCancel: { find.close() })
    let handled = field.makeCoordinator().control(
      NSTextField(), textView: NSTextView(),
      doCommandBy: #selector(NSResponder.cancelOperation(_:)))
    #expect(handled && !find.presented && find.query == "keep query")
  }

  @Test func cancellationIsCooperativeAndResultMemoryIsBounded() async throws {
    let index = ConversationFindIndex()
    let items = [ConversationSelectionItem(id: "a", text: String(repeating: "x ", count: 50_000))]
    let result = try await index.search(items, query: "x")
    #expect(result.matches.count == 10_000 && result.truncated)
    let task = Task {
      try Task.checkCancellation()
      return try await index.search(items, query: "x")
    }
    task.cancel()
    do {
      _ = try await task.value
      Issue.record("Cancelled index completed")
    } catch is CancellationError {} catch { throw error }
  }

  @Test func queryChangesAndBranchReplacementNeverPublishStaleMatches() async throws {
    let find = ConversationFind()
    find.update([.init(id: "old", text: String(repeating: "old ", count: 20_000))])
    find.show()
    find.search("old")
    find.search("new")
    find.update([.init(id: "branch", text: "new new")])
    try await wait { !find.searching }
    #expect(find.matches.map(\.itemID) == ["branch", "branch"])
    find.move(-1)
    #expect(find.position == 1)
    find.move(1)
    #expect(find.position == 0)
    find.update([.init(id: "edited", text: "new")])
    #expect(find.matches.isEmpty && find.current == nil)
    try await wait { !find.searching }
    #expect(find.current?.itemID == "edited")
    find.close()
    #expect(find.current == nil && !find.searching)
    find.update([.init(id: "closed", text: "new")])
    find.show()
    try await wait { !find.searching }
    #expect(find.current?.itemID == "closed")
    find.close()
  }

  @Test func streamingDoesNotStarveSearchOrMoveTheCurrentMatch() async throws {
    let find = ConversationFind()
    find.show()
    find.search("needle")
    for i in 0..<30 {
      find.update([
        .init(id: "answer", text: "needle needle " + String(repeating: "token ", count: i))
      ])
      try await Task.sleep(for: .milliseconds(12))
    }
    #expect(find.matches.count == 2)
    find.move(1)
    let match = find.current
    let revision = find.navigationRevision
    find.update([
      .init(
        id: "answer",
        text: "needle needle " + String(repeating: "token ", count: 29) + "plus needle")
    ])
    try await wait { !find.searching }
    #expect(find.matches.count == 3 && find.current == match)
    #expect(find.navigationRevision == revision)
    find.close()
  }

  @Test func findNextAfterClosingResumesWithoutRetainingTheIndex() async throws {
    let find = ConversationFind()
    find.update([.init(id: "a", text: "needle needle needle")])
    find.show()
    find.search("needle")
    try await wait { !find.searching }
    find.move(1)
    #expect(find.position == 1)
    find.close()
    #expect(find.matches.isEmpty)
    find.move(1)
    try await wait { !find.searching }
    #expect(find.presented && find.position == 2)
    find.close()
    find.move(-1)
    try await wait { !find.searching }
    #expect(find.position == 1)
    find.close()
  }

  @Test func largeHistorySearchDoesNotBlockTheMainActor() async throws {
    let find = ConversationFind()
    find.update(
      (0..<2_000).map {
        .init(id: "\($0)", text: "Message **\($0)** with `code` and a needle.", markdown: true)
      })
    find.show()
    find.search("needle")
    var ticks = 0
    while find.searching && ticks < 600 {
      ticks += 1
      try await Task.sleep(for: .milliseconds(5))
    }
    #expect(find.matches.count == 2_000)
    #expect(ticks > 1 && ticks < 600)
    find.close()
  }

  @Test func textAndCodeFindPreserveSelections() async throws {
    let find = ConversationFind()
    let items = [
      ConversationSelectionItem(
        id: "answer",
        text:
          "Before **needle**\n\nlet needle = 42\n\n```swift\nlet needle = 42\n```\n\nAfter needle",
        markdown: true)
    ]
    let fixture = mount {
      ConversationSelectionSurface(items: items) {
        ScrollView {
          NativeMarkdown(items[0].text).environment(\.conversationTextID, "answer").padding(20)
        }
      }.environment(\.conversationFind, find)
    }
    defer {
      find.close()
      fixture.window.close()
    }
    try await wait { texts(fixture.host).contains { $0.string.contains("Before") } }
    let selection = try #require(
      descendants(fixture.host).compactMap { $0 as? ConversationSelectionHost }.first)
    find.update(items)
    find.show()
    find.search("needle")
    try await wait { find.matches.count == 4 }
    let text = try #require(texts(fixture.host).first { $0.string.contains("Before") })
    text.setSelectedRange(NSRange(location: 0, length: 6))
    try await wait { selection.findTarget(find.matches[2], query: "needle") != nil }
    #expect(selection.findTarget(find.matches[1], query: "needle")?.view === text)
    let target = try #require(selection.findTarget(find.matches[2], query: "needle"))
    #expect((target.view.string as NSString).substring(with: target.range) == "needle")
    find.move(1)
    try await Task.sleep(for: .milliseconds(250))
    #expect(text.selectedRange() == NSRange(location: 0, length: 6))
    #expect(!find.revealFailed)
  }

  @Test func diagramSourceIsRevealedOnlyWhileFindingAndTextKit2RemainsNative() async throws {
    let find = ConversationFind()
    let item = ConversationSelectionItem(
      id: "diagram", text: "```mermaid\ngraph TD\nA[Needle] --> B[Answer]\n```", markdown: true)
    let fixture = mount {
      ConversationSelectionSurface(items: [item]) {
        ScrollView {
          NativeMarkdown(item.text).environment(\.conversationTextID, item.id).padding(20)
        }
      }.environment(\.conversationFind, find)
    }
    defer {
      find.close()
      fixture.window.close()
    }
    find.update([item])
    find.show()
    find.search("Needle")
    try await wait("diagram source opened") {
      texts(fixture.host).contains { $0.string.contains("A[Needle]") }
    }
    let selection = try #require(
      descendants(fixture.host).compactMap { $0 as? ConversationSelectionHost }.first)
    try await wait("diagram source revealed") { visibleTarget(selection, find: find) }
    find.close()
    try await wait("diagram source closed") {
      !texts(fixture.host).contains { $0.string.contains("A[Needle]") }
    }

    let view = NSTextView(usingTextLayoutManager: true)
    view.frame = NSRect(x: 0, y: 0, width: 300, height: 100)
    view.string = "Speech needle"
    view.setSelectedRange(NSRange(location: 0, length: 6))
    let manager = try #require(view.textLayoutManager)
    #expect(ConversationFindReveal.textRect(view, range: NSRange(location: 7, length: 6)) != nil)
    #expect(view.textLayoutManager === manager)
    #expect(view.selectedRange() == NSRange(location: 0, length: 6))
  }

  @Test func actualTranscriptOpensReasoningAndToolResultsWithoutChangingDisclosurePreferences()
    async throws
  {
    let transcript = try JSONDecoder().decode(
      StudioState.NativeTranscript.self,
      from: Data(
        #"""
        {"available":true,"notice":"","conversationId":"c","leafId":"a","messages":[
         {"id":"a","role":"assistant","text":"Final answer","reasoning":"Hidden hypothesis",
          "model":"m","streaming":false,"stopped":false,"error":"","incomplete":false,
          "toolCalls":[{"id":"t","name":"read_file","server":"","arguments":"{}",
            "output":"Tool needle","status":"completed","error":""}]}]}
        """#.utf8))
    let find = ConversationFind()
    let fixture = mount {
      NativeStudioTranscript(
        transcript: transcript, columnWidth: 500, composerHeight: 0, find: find)
    }
    defer {
      find.close()
      fixture.window.close()
    }
    try await wait { texts(fixture.host).contains { $0.string.contains("Final answer") } }
    #expect(!texts(fixture.host).contains { $0.string.contains("Hidden hypothesis") })
    find.show()
    find.search("hypothesis")
    try await wait { texts(fixture.host).contains { $0.string.contains("Hidden hypothesis") } }
    #expect(find.matches.count == 1)
    find.search("Tool needle")
    try await wait { texts(fixture.host).contains { $0.string.contains("Tool needle") } }
    #expect(!texts(fixture.host).contains { $0.string.contains("Hidden hypothesis") })
    find.close()
    try await wait { !texts(fixture.host).contains { $0.string.contains("Tool needle") } }
  }

  @Test func revealFindsOffscreenRowsAndHorizontallyHiddenCompareLanes() async throws {
    var messages: [[String: Any]] = (0..<20).map { index in
      [
        "id": "m\(index)", "role": "assistant",
        "text": String(repeating: "Paragraph \(index). ", count: 40),
        "reasoning": "", "model": "fixture", "streaming": false, "stopped": false, "error": "",
        "incomplete": false,
      ]
    }
    messages[0]["text"] = "An old **needle** before the long conversation."
    for index in 0..<4 {
      messages.append([
        "id": "lane\(index)", "group": "comparison", "role": "assistant",
        "text": index == 3 ? "Last lane needle" : "Other response", "reasoning": "",
        "model": "fixture",
        "streaming": false, "stopped": false, "error": "", "incomplete": false,
      ])
    }
    let data = try JSONSerialization.data(withJSONObject: [
      "available": true, "notice": "", "conversationId": "c", "leafId": "lane3",
      "messages": messages,
    ])
    let transcript = try JSONDecoder().decode(StudioState.NativeTranscript.self, from: data)
    let find = ConversationFind()
    let fixture = mount {
      NativeStudioTranscript(
        transcript: transcript, columnWidth: 500, composerHeight: 0, find: find)
    }
    defer {
      find.close()
      fixture.window.close()
    }
    try await wait { descendants(fixture.host).contains { $0 is ConversationSelectionHost } }
    let selection = try #require(
      descendants(fixture.host).compactMap { $0 as? ConversationSelectionHost }.first)
    find.show()
    find.search("needle")
    try await wait { find.matches.count == 2 }
    try await wait("old row") { visibleTarget(selection, find: find) }
    #expect(!find.revealFailed)
    find.move(1)
    try await wait("last Compare lane") { visibleTarget(selection, find: find) }
    #expect(find.current?.itemID == "lane3/body" && !find.revealFailed)
  }

  private func visibleTarget(_ host: ConversationSelectionHost, find: ConversationFind) -> Bool {
    guard let match = find.current, let target = host.findTarget(match, query: find.query),
      let rect = ConversationFindReveal.textRect(target.view, range: target.range)
    else { return false }
    return target.view.visibleRect.intersects(rect)
  }

  private func wait(_ label: String = "Find", _ condition: () -> Bool) async throws {
    for _ in 0..<250 {
      if condition() { return }
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(condition(), "\(label) did not reach the expected state")
  }
  private func descendants(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(descendants)
  }
  private func texts(_ view: NSView) -> [NSTextView] {
    descendants(view).compactMap { $0 as? NSTextView }
  }
  private func mount<Content: View>(@ViewBuilder _ content: () -> Content) -> (
    host: NSView, window: NSWindow
  ) {
    _ = NSApplication.shared
    let root = NSHostingView(rootView: content())
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 700, height: 600),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = root
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    return (root, window)
  }
}
