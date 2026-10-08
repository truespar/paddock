import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockConversationCore
@testable import PaddockNativeMarkdown
@testable import PaddockStudio
@testable import PaddockUI

@Suite("Conversation window input", .serialized) @MainActor
struct StudioConversationInputTests {
  @Test func wheelInputReachesTranscriptThroughActualConversationChrome() async throws {
    _ = NSApplication.shared
    let workspace = StudioWorkspace(client: NoInputCore())
    // A closed workspace intentionally rejects all later presentations. Keep
    // this fixture live; its client refuses host creation without any I/O.
    defer { Task { await workspace.shutdown() } }
    let descriptor = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: descriptor))
    { _ in }
    var fields = await runtime.presentation()
    fields["revision"] = .number(1)
    fields["conversation"] = .object([
      "id": .string("input-fixture"), "title": .string("Input fixture"),
      "model": .string("fixture"), "messageCount": .number(2),
    ])
    fields["nativeTranscript"] = .object([
      "available": .bool(true), "notice": .string(""),
      "messages": .array([
        message("question", role: "user", text: "Describe this model and compare its features."),
        message("answer", role: "assistant", text: Self.markdown),
      ]),
    ])
    workspace.apply(try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields)))
    await runtime.close()
    let host = NSHostingController(rootView: ConversationInputFixture(workspace: workspace))
    host.sizingOptions = []
    host.safeAreaRegions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 902, height: 741),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView], backing: .buffered,
      defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbar = NSToolbar(identifier: "ConversationInputFixture")
    window.toolbarStyle = .unifiedCompact
    window.contentViewController = host
    window.setContentSize(NSSize(width: 902, height: 741))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await settle(host.view)
    let selection = try #require(find(ConversationSelectionHost.self, in: host.view))
    #expect(
      selection.sizingOptions.isEmpty,
      "The viewport must not constrain its window to the transcript's content size")
    let scroll = try #require(find(NSScrollView.self, in: selection))
    let document = try #require(scroll.documentView)
    for _ in 0..<20 {
      if document.bounds.height > 1500 { break }
      try await settle(host.view)
    }
    #expect(document.bounds.height > 1500)
    scroll.contentView.scroll(to: NSPoint(x: 0, y: 200))
    scroll.reflectScrolledClipView(scroll.contentView)
    for _ in 0..<8 { try await settle(host.view) }
    if let capture = ProcessInfo.processInfo.environment["PADDOCK_SCROLL_CHROME_CAPTURE_DIR"] {
      // Synthetic fixture only. Never drive or capture the user's app/desktop.
      let appearance = window.appearance
      for dark in [false, true] {
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        for _ in 0..<4 { try await settle(host.view) }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        process.arguments = [
          "-x", "-l", String(window.windowNumber),
          URL(fileURLWithPath: capture).appendingPathComponent(
            dark ? "system-scroll-dark.png" : "system-scroll-light.png"
          ).path,
        ]
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
      }
      window.appearance = appearance
      for _ in 0..<4 { try await settle(host.view) }
    }
    #if DEBUG
      let updates = selection.contentUpdates
      let markdownHost = try #require(
        all(SelectionHostingView.self, in: selection).first {
          $0.identifier?.rawValue == "answer/body"
        })
      let markdownUpdates = markdownHost.contentUpdates
    #endif
    for offset in stride(from: 220, through: 600, by: 20) {
      scroll.contentView.scroll(to: NSPoint(x: 0, y: offset))
      scroll.reflectScrolledClipView(scroll.contentView)
      try await settle(host.view)
    }
    #if DEBUG
      #expect(
        selection.contentUpdates - updates == 0,
        "Scrolling unchanged content must not repeatedly replace the transcript's hosting root: \(selection.contentUpdates - updates) updates"
      )
      #expect(
        markdownHost.contentUpdates == markdownUpdates,
        "Wheel ticks must not reinstall an unchanged rich-text hosting root")
    #endif
    // Match the app's accessibility/scrollbar path, not only clip-view jumps.
    // A completed rich response must not oscillate between estimated document
    // heights when its preceding question leaves the viewport.
    let stableHeight = document.bounds.height
    let scroller = try #require(scroll.verticalScroller)
    for fraction in [0.6, 0.25, 0.85, 0.1, 1.0] {
      scroller.setAccessibilityValue(NSNumber(value: fraction))
      for _ in 0..<3 { try await settle(host.view) }
      #expect(abs(document.bounds.height - stableHeight) < 1)
    }
    for width: CGFloat in [902, 1200, 902] {
      window.setContentSize(NSSize(width: width, height: 741))
      try await settle(host.view)
      for _ in 0..<8 { try await settle(host.view) }
      var minimum = scroll.documentVisibleRect.minY
      var reachedEnd = false
      for step in 0..<40 {
        let point = scroll.convert(NSPoint(x: scroll.bounds.midX, y: 300), to: nil)
        let hitPoint = host.view.superview?.convert(point, from: nil) ?? point
        let target = try #require(host.view.hitTest(hitPoint))
        #expect(
          target === selection || target.isDescendant(of: selection),
          "The real composer/chrome must not intercept input over the transcript: \(type(of: target))"
        )
        let press = try #require(
          NSEvent.mouseEvent(
            with: .leftMouseDown, location: point,
            modifierFlags: [], timestamp: 1, windowNumber: window.windowNumber, context: nil,
            eventNumber: step, clickCount: 1, pressure: 1))
        // Exercise layout-time hit queries with a stale press too. Never send
        // that press to the user's app or the system event queue.
        let local = selection.superview?.convert(point, from: nil) ?? point
        #expect(selection.hitTest(local, event: press) != nil)
        let event = try #require(
          CGEvent(
            scrollWheelEvent2Source: nil, units: .pixel,
            wheelCount: 1, wheel1: step < 20 ? 180 : -180, wheel2: 0, wheel3: 0))
        event.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
        event.setIntegerValueField(
          .scrollWheelEventScrollPhase,
          value: step % 20 == 0 ? 1 : (step % 20 == 19 ? 4 : 2))
        // Direct, process-local dispatch. Never post to a system event tap.
        target.scrollWheel(with: try #require(NSEvent(cgEvent: event)))
        try await settle(host.view)
        minimum = min(minimum, scroll.documentVisibleRect.minY)
        // LazyVStack's offscreen estimates legitimately change; test whether
        // input reaches the actual endpoints, not a stale document height.
        if step >= 20,
          document.bounds.height + scroll.contentInsets.bottom - scroll.documentVisibleRect.maxY < 2
        {
          reachedEnd = true
        }
        let viewport = scroll.convert(scroll.bounds, to: nil)
        #expect(abs(viewport.maxY - window.contentLayoutRect.maxY) < 1)
        #expect(
          abs(viewport.minY) < 1, "The composer must still overlay the bottom of the viewport")
      }
      #expect(minimum < 1, "Wheel input must reach the beginning")
      #expect(reachedEnd, "Wheel input must reach the end")
    }
    // Continue the response while pinned, then interrupt it to read earlier
    // text. Both are real production snapshots through the native workspace.
    for part in 0..<8 {
      let reading = part >= 4
      if part == 4 {
        scroll.contentView.scroll(to: NSPoint(x: 0, y: 350))
        scroll.reflectScrolledClipView(scroll.contentView)
        try await settle(host.view)
      }
      let before = scroll.documentVisibleRect.minY
      fields["revision"] = .number(Decimal(part + 2))
      fields["nativeTranscript"] = .object([
        "available": .bool(true), "notice": .string(""),
        "messages": .array([
          message("question", role: "user", text: "Describe this model and compare its features."),
          message(
            "answer", role: "assistant",
            text: Self.markdown
              + String(
                repeating: "\n\nA continued paragraph with **formatting** and `code`.",
                count: part + 1),
            streaming: part != 7),
        ]),
      ])
      workspace.apply(
        try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields)))
      for _ in 0..<3 { try await settle(host.view) }
      if reading {
        #expect(
          abs(scroll.documentVisibleRect.minY - before) < 2,
          "New text must not pull a reader back to the bottom")
      } else {
        #expect(
          abs(
            document.bounds.height + scroll.contentInsets.bottom - scroll.documentVisibleRect.maxY)
            < 2,
          "Pinned reading must follow real streamed content, not height estimates; part \(part), gap \(document.bounds.height + scroll.contentInsets.bottom - scroll.documentVisibleRect.maxY)"
        )
      }
    }
  }

  private func message(_ id: String, role: String, text: String, streaming: Bool = false)
    -> ConversationValue
  {
    .object([
      "id": .string(id), "role": .string(role), "model": .string("fixture"),
      "text": .string(text),
      "reasoning": .string(
        role == "assistant" ? String(repeating: "Reasoning fixture. ", count: 160) : ""),
      "searches": .array(
        role == "assistant"
          ? (0..<2).map { index in
            .object([
              "id": .string("search-\(index)"),
              "query": .string("Local model architecture and performance"),
              "status": .string("completed"), "provider": .string("exa"), "error": .string(""),
              "sources": .array([
                .object([
                  "title": .string("Model reference"), "url": .string("https://example.com/model"),
                ])
              ]),
            ])
          } : []),
      "streaming": .bool(streaming),
      "stopped": .bool(false), "error": .string(""), "incomplete": .bool(false),
    ])
  }
  private func settle(_ view: NSView) async throws {
    for _ in 0..<3 {
      view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(25))
    }
  }
  private func find<T: NSView>(_ type: T.Type, in view: NSView) -> T? {
    (view as? T) ?? view.subviews.lazy.compactMap { find(type, in: $0) }.first
  }
  private func all<T: NSView>(_ type: T.Type, in view: NSView) -> [T] {
    ((view as? T).map { [$0] } ?? []) + view.subviews.flatMap { all(type, in: $0) }
  }
  private static let markdown = """
    Here's a **model overview**, including its architecture, weights and features.

    ## What it is
    A compressed multimodal reasoning model for local inference. Longer paragraphs combine **emphasis**, `inline code`, lists and a table, so that scrolling exercises the full native renderer.

    ## Headline numbers
    | Item | Value |
    | --- | --- |
    | Base model | A local model with the original architecture |
    | Total parameters | 27 billion parameters, including a language backbone, embedding and output projection, and a vision tower |
    | Context length | A large supported context |
    | Modalities | Text and image input, text output |
    | Availability | Local inference on a laptop |
    | Retention | **A comparison** with the original model, including an aggregate score and reasoning-mode evaluation |
    | Footprint | Smaller packed weights with native processing |

    \((1...8).map { "## Section \($0)\n- **First point** with `inline_code` and additional descriptive text about local inference and laptop hardware.\n- Another point with **emphasis** and enough text to wrap over several lines in the narrow chat column.\n\nA concluding paragraph for this section explains the differences and the remaining caveats." }.joined(separator: "\n\n"))
    """
}

private struct ConversationInputFixture: View {
  let workspace: StudioWorkspace
  @State private var draft = StudioDraft()
  var body: some View {
    HStack(spacing: 0) {
      Color.gray.frame(width: 260)
      StudioConversationView(chat: workspace, draft: $draft)
    }.ignoresSafeArea(.container, edges: .top)
  }
}

private struct NoInputCore: ManagerLoading {
  func nativeConversationHost() async throws -> StudioHost {
    throw CancellationError()
  }
  func snapshot() async throws -> ManagerSnapshot {
    Issue.record("The input fixture must not access a backend")
    throw CancellationError()
  }
}
