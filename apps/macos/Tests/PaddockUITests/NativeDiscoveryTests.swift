import AppKit
import Observation
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native discovery and navigation", .serialized) @MainActor
struct NativeDiscoveryTests {
  @Test func settingsIndexHasStableUniqueTargetsAndSearchesControlAliases() {
    let entries = SettingsSearchIndex.entries
    #expect(Set(entries.map(\.id)).count == entries.count)
    #expect(Set(entries.map(\.destination)) == Set(ManagerDestination.allCases))
    #expect(SettingsSearchIndex.search("   ").isEmpty)
    #expect(SettingsSearchIndex.search("missing setting").isEmpty)
    #expect(SettingsSearchIndex.search("OUTPUT tokens").map(\.anchor) == [.replyLength])
    #expect(SettingsSearchIndex.search("dictation").map(\.anchor) == [.microphone])
    #expect(SettingsSearchIndex.search("hotkey").map(\.anchor) == [.shortcut])
    #expect(SettingsSearchIndex.search("OpenRouter").map(\.destination) == [.cloudProviders])
    #expect(Set(SettingsSearchIndex.search("MCP").map(\.destination)) == [.connectors])
    for section in ["maxTokens", "maxToolCalls", "summarize", "microphone", "mapTiles"] {
      #expect(SettingsAnchor.conversationSection(section)?.destination == .general)
    }
    #expect(SettingsAnchor.conversationSection("future") == nil)
  }

  @Test func searchKeyboardNavigationIsBoundedAndRetainsStableIdentity() {
    #expect(SearchNavigation.moved(nil as String?, in: [], by: 1) == nil)
    #expect(SearchNavigation.moved(nil, in: ["a", "b"], by: 1) == "a")
    #expect(SearchNavigation.moved(nil, in: ["a", "b"], by: -1) == "b")
    #expect(SearchNavigation.moved("a", in: ["a", "b"], by: -1) == "a")
    #expect(SearchNavigation.moved("b", in: ["a", "b"], by: 1) == "b")
    #expect(SearchNavigation.moved("removed", in: ["a", "b"], by: 1) == "a")
  }

  @Test func fieldCommandsDoNotStealIMECompositionOrOrdinaryEditing() {
    _ = NSApplication.shared
    var query = "qwen"
    var moves: [Int] = []
    var submits = 0
    var cancels = 0
    let field = NativeSearchField(
      text: Binding(get: { query }, set: { query = $0 }), prompt: "Search",
      onMove: { moves.append($0) }, onSubmit: { submits += 1 }, onCancel: { cancels += 1 })
    let coordinator = field.makeCoordinator()
    let control = NSSearchField()
    let editor = NSTextView()
    let down = #selector(NSResponder.moveDown(_:))
    let enter = #selector(NSResponder.insertNewline(_:))
    let escape = #selector(NSResponder.cancelOperation(_:))
    #expect(coordinator.control(control, textView: editor, doCommandBy: down))
    #expect(coordinator.control(control, textView: editor, doCommandBy: enter))
    #expect(moves == [1] && submits == 1)
    editor.setMarkedText(
      "あ", selectedRange: NSRange(location: 1, length: 0),
      replacementRange: NSRange(location: NSNotFound, length: 0))
    #expect(editor.hasMarkedText())
    for command in [down, enter, escape] {
      #expect(!coordinator.control(control, textView: editor, doCommandBy: command))
    }
    #expect(moves == [1] && submits == 1 && cancels == 0 && query == "qwen")
    editor.unmarkText()
    #expect(
      !coordinator.control(
        control, textView: editor, doCommandBy: #selector(NSResponder.moveLeft(_:))))
    #expect(coordinator.control(control, textView: editor, doCommandBy: escape))
    #expect(query.isEmpty && cancels == 0)
    #expect(coordinator.control(control, textView: editor, doCommandBy: escape))
    #expect(cancels == 1)
  }

  @Test func searchEditorKeepsItsInsetWhenFocusedAndTypedInto() async throws {
    for dark in [false, true] {
      var query = ""
      let component = NativeSearchField(
        text: Binding(get: { query }, set: { query = $0 }), prompt: "Search instances")
      let coordinator = component.makeCoordinator()
      let field = NativeSearchField.Field(frame: NSRect(x: 12, y: 16, width: 296, height: 28))
      field.editor.delegate = coordinator
      field.editor.placeholderString = component.prompt
      field.clearButton.target = coordinator
      field.clearButton.action = #selector(NativeSearchField.Coordinator.clearSearch(_:))
      field.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      field.layer?.backgroundColor = PaddockStyle.nsColor("elevated", dark: dark).cgColor
      field.autoresizingMask = [.width]
      let root = NSView(frame: NSRect(x: 0, y: 0, width: 320, height: 60))
      root.addSubview(field)
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 320, height: 60),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentView = root
      defer { window.close() }
      root.layoutSubtreeIfNeeded()
      let idleBounds = field.editor.frame
      #expect(field.editor.accessibilitySubrole() == .searchField)
      #expect(field.clearButton.isHidden)
      #expect(window.makeFirstResponder(field.editor))
      #expect(field.isEditing && field.layer?.borderWidth == 1)
      let editor = try #require(field.editor.currentEditor() as? NSTextView)
      for width: CGFloat in [320, 200, 480] {
        window.setContentSize(NSSize(width: width, height: 60))
        root.layoutSubtreeIfNeeded()
        let editorBounds = editor.convert(editor.bounds, to: field)
        #expect(editorBounds.minX >= idleBounds.minX - 2)
        #expect(abs(editorBounds.midY - field.bounds.midY) <= 2)
        #expect(editorBounds.maxX <= field.clearButton.frame.minX)
      }
      editor.insertText("Qwen", replacementRange: NSRange(location: NSNotFound, length: 0))
      try await Task.sleep(for: .milliseconds(30))
      #expect(query == "Qwen")
      #expect(field.isEditing && field.layer?.borderWidth == 1)
      #expect(!field.clearButton.isHidden)
      #expect(field.editor.frame.minX == idleBounds.minX)
      try capture(root, name: "search-typing-\(dark ? "dark" : "light")")
      field.clearButton.performClick(nil)
      try await Task.sleep(for: .milliseconds(30))
      #expect(query.isEmpty && field.editor.stringValue.isEmpty)
      #expect(field.clearButton.isHidden)
      #expect(field.editor.currentEditor() != nil, "Clearing retains keyboard focus")
      #expect(field.isEditing && field.layer?.borderWidth == 1)
      try capture(root, name: "search-cleared-\(dark ? "dark" : "light")")
      #expect(window.makeFirstResponder(nil))
      #expect(!field.isEditing && field.layer?.borderWidth == 0)
      #expect(field.editor.frame.minX == idleBounds.minX)
    }
  }

  @Test func sidebarSearchKeepsTheEstablishedChatLayoutInBothAppearances() async throws {
    for dark in [false, true] {
      for width: CGFloat in [200, 300] {
        for query in ["", "Qwen"] {
          let text = Binding.constant(query)
          let original = HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("Search", text: text).textFieldStyle(.plain)
            if !query.isEmpty {
              Button("Clear search", systemImage: "xmark") {}
                .labelStyle(.iconOnly).buttonStyle(.plain)
            }
          }.padding(10).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 8))
          let chat = SidebarSearchField(text: text) {
            TextField("Search", text: text).textFieldStyle(.plain)
          }
          let settings = SidebarSearchField(text: text) {
            NativeSearchField(text: text, prompt: "Search", editorOnly: true)
          }
          let heights = [AnyView(original), AnyView(chat), AnyView(settings)].map { view in
            NSHostingController(rootView: view).sizeThatFits(in: CGSize(width: width, height: 100))
              .height
          }
          #expect(heights.allSatisfy { abs($0 - heights[0]) < 0.1 })
          let window = host(
            VStack(spacing: 12) {
              original
              chat
              settings
            }
            .environment(\.colorScheme, dark ? .dark : .light), width: width, height: 160)
          defer { window.close() }
          try await Task.sleep(for: .milliseconds(40))
          let root = try #require(window.contentView)
          root.layoutSubtreeIfNeeded()
          let inputs = Self.views(root).compactMap { $0 as? NSTextField }
            .filter { $0.placeholderString == "Search" }
          #expect(inputs.count == 3)
          let first = try #require(inputs.first)
          for input in inputs {
            #expect(
              abs(
                input.convert(input.bounds, to: root).minX
                  - first.convert(first.bounds, to: root).minX) < 1)
          }
          let field = try #require(
            Self.views(root).compactMap { $0 as? NativeSearchField.Field }.first)
          #expect(field.clearButton.isHidden && field.layer?.borderWidth == 0)
          #expect(field.layer?.backgroundColor == nil)
          #expect(Self.views(field).compactMap { $0 as? NSImageView }.allSatisfy { $0.isHidden })
          #expect(field.layer?.cornerRadius == 0, "Only the outer search surface may clip corners")
          try capture(
            root,
            name:
              "sidebar-search-\(dark ? "dark" : "light")-\(Int(width))-\(query.isEmpty ? "empty" : "query")"
          )
        }
      }
    }
  }

  @Test func embeddedSidebarEditorKeepsTypingAndCommandsWithoutItsOwnChrome() throws {
    var query = ""
    var moves: [Int] = []
    var submissions = 0
    let component = NativeSearchField(
      text: Binding(get: { query }, set: { query = $0 }), prompt: "Search settings",
      editorOnly: true, onMove: { moves.append($0) }, onSubmit: { submissions += 1 })
    let coordinator = component.makeCoordinator()
    let field = NativeSearchField.Field()
    field.editorOnly = true
    field.editor.delegate = coordinator
    field.frame = NSRect(x: 20, y: 20, width: 180, height: field.intrinsicContentSize.height)
    let root = NSView(frame: NSRect(x: 0, y: 0, width: 220, height: 60))
    root.addSubview(field)
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 220, height: 60),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = root
    defer { window.close() }
    root.layoutSubtreeIfNeeded()
    #expect(window.makeFirstResponder(field.editor))
    let editor = try #require(field.editor.currentEditor() as? NSTextView)
    // AppKit's focus clip view expands the editor frame. Its visible viewport,
    // not the scrolling document bounds, must stay clear of the outer controls.
    let editorBounds = editor.convert(editor.visibleRect, to: field)
    #expect(!editorBounds.isEmpty)
    #expect(editorBounds.minX >= 0 && editorBounds.maxX <= field.bounds.maxX)
    #expect(abs(editorBounds.midY - field.bounds.midY) <= 2)
    editor.insertText("microphone", replacementRange: NSRange(location: NSNotFound, length: 0))
    #expect(query == "microphone")
    #expect(field.layer?.cornerRadius == 0 && field.layer?.borderWidth == 0)
    #expect(field.clearButton.isHidden)
    #expect(
      coordinator.control(
        field.editor, textView: editor, doCommandBy: #selector(NSResponder.moveDown(_:))))
    #expect(
      coordinator.control(
        field.editor, textView: editor, doCommandBy: #selector(NSResponder.insertNewline(_:))))
    #expect(moves == [1] && submissions == 1)
    #expect(window.makeFirstResponder(nil))
  }

  @Test func settingsRevealWaitsForLoadedRowsAndDoesNotStealLaterScrolling() async throws {
    let state = SearchRevealState()
    let window = host(SearchRevealFixture(state: state), width: 640, height: 320)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(80))
    state.loaded = true
    try await Task.sleep(for: .milliseconds(100))
    let root = try #require(window.contentView)
    root.layoutSubtreeIfNeeded()
    let scroll = try #require(Self.views(root).compactMap { $0 as? NSScrollView }.first)
    let marker = try #require(Self.views(root).first { $0 is SearchTargetView })
    func targetVisible() -> Bool {
      scroll.contentView.bounds.contains(marker.convert(marker.bounds, to: scroll.contentView))
    }
    #expect(targetVisible(), "Reveal must wait for asynchronous settings rows")
    scroll.contentView.scroll(to: .zero)
    scroll.reflectScrolledClipView(scroll.contentView)
    state.publication += 1
    try await Task.sleep(for: .milliseconds(60))
    #expect(scroll.contentView.bounds.origin.y == 0, "A refresh must not undo manual scrolling")
    state.request = SettingsSearchRequest(
      entry: .init(destination: .application, anchor: .notifications))
    try await Task.sleep(for: .milliseconds(100))
    #expect(targetVisible(), "Selecting the same result again is a new reveal request")
  }

  @Test func discoverySurfacesFitNarrowWideAndBothAppearancesWithoutMutations() async throws {
    let snapshot = try endpointFixture()
    let client = DiscoveryClient()
    let workspace = WorkspaceModel(client: client)
    for dark in [false, true] {
      for width: CGFloat in [640, 1000] {
        let window = host(
          EndpointsView(workspace: workspace, snapshot: snapshot, onCreate: {})
            .environment(\.colorScheme, dark ? .dark : .light), width: width, height: 650)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(80))
        let root = try #require(window.contentView)
        root.layoutSubtreeIfNeeded()
        let scrolls = Self.views(root).compactMap { $0 as? NSScrollView }
        #expect(scrolls.count == 1)
        #expect((scrolls.first?.documentView?.frame.width ?? 0) <= width)
        for control in Self.views(root).compactMap({ $0 as? NSControl }) {
          guard !control.isHiddenOrHasHiddenAncestor, !control.bounds.isEmpty else { continue }
          let bounds = control.convert(control.bounds, to: root)
          #expect(bounds.minX >= -1 && bounds.maxX <= width + 1)
        }
        try capture(root, name: "instances-\(Int(width))-\(dark ? "dark" : "light")")
      }
      let results = SettingsSearchIndex.search("application")
      let window = host(
        SettingsSearchResults(entries: results, highlighted: results.first?.id, select: { _ in })
          .environment(\.colorScheme, dark ? .dark : .light), width: 200, height: 480)
      defer { window.close() }
      try await Task.sleep(for: .milliseconds(80))
      let root = try #require(window.contentView)
      try capture(root, name: "settings-search-\(dark ? "dark" : "light")")
    }
    #expect(await client.mutations == 0)
  }

  private func host<Content: View>(_ content: Content, width: CGFloat, height: CGFloat) -> NSWindow
  {
    _ = NSApplication.shared
    let controller = NSHostingController(rootView: content)
    controller.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: width, height: height),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = controller
    window.setContentSize(NSSize(width: width, height: height))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    return window
  }
  private static func views(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(views) }
  private func capture(_ view: NSView, name: String) throws {
    guard let folder = ProcessInfo.processInfo.environment["PADDOCK_DISCOVERY_SNAPSHOTS"] else {
      return
    }
    try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
    if let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
      view.cacheDisplay(in: view.bounds, to: bitmap)
      try bitmap.representation(using: .png, properties: [:])?.write(
        to: URL(fileURLWithPath: folder).appending(path: "\(name).png"))
    }
  }

  @Test func instanceSearchRetainsPortsAndTransitionsWithoutMutatingTheFleet() throws {
    let snapshot = try endpointFixture()
    let rows = EndpointRow.rows(snapshot: snapshot, latestJob: nil)
    let row = try #require(rows.first)
    #expect(row.isActive)
    #expect(row.matches(" qwen  12345 "))
    #expect(row.matches("MLX 4bit"))
    #expect(!row.matches("qwen 99999"))
    #expect(
      EndpointRow(port: row.port, runner: nil, configured: row.configured, job: try commandJob())
        .isActive)
    #expect(!EndpointRow(port: 12346, runner: nil, configured: nil, job: nil).isActive)
    #expect(rows.count == 1)
  }

  @Test func compareOrderingRetainsLanesAndSingleListShowsUnavailableModels() throws {
    let selected: Set<String> = ["third", "first", "second"]
    #expect(
      StudioCompareList.orderedSelection(selected, existing: ["second", "first", "removed"]) == [
        "second", "first", "third",
      ])
    let models = try (0..<500).map { index in
      try JSONDecoder().decode(
        StudioState.Model.self,
        from: JSONSerialization.data(withJSONObject: [
          "id": "model-\(index)", "title": "Model \(index)", "provider": "OpenRouter",
          "vendor": "Alibaba",
          "status": index == 499 ? "unreachable" : "ok", "vision": true, "chat": true,
          "audio": false,
        ]))
    }
    #expect(
      StudioCompareList.groups(models, search: "", includeUnavailable: true).flatMap(\.models).count
        == 500)
    #expect(StudioCompareList.groups(models, search: "").flatMap(\.models).count == 499)
    #expect(
      StudioCompareList.groups(models, search: "openrouter 499", includeUnavailable: true).flatMap(
        \.models
      ).map(\.id) == ["model-499"])
    #expect(!StudioCompareList.canApply(["model-499"], models: models))
    #expect(!StudioCompareList.canApply(Set((0..<5).map { "model-\($0)" }), models: models))
  }

  @Test func largeSinglePickerHasABoundedViewportAndKeyboardReveal() async throws {
    let models = try (0..<500).map { index in
      try JSONDecoder().decode(
        StudioState.Model.self,
        from: JSONSerialization.data(withJSONObject: [
          "id": "model-\(index)", "title": "Model \(index)", "provider": "OpenRouter",
          "vendor": "Alibaba",
          "status": "ok", "vision": true, "chat": true, "audio": false,
        ]))
    }
    for dark in [false, true] {
      let state = PickerRevealState()
      let window = host(
        PickerRevealFixture(state: state, models: models)
          .environment(\.colorScheme, dark ? .dark : .light), width: 380, height: 390)
      defer { window.close() }
      try await Task.sleep(for: .milliseconds(100))
      let root = try #require(window.contentView)
      let scroll = try #require(Self.views(root).compactMap { $0 as? NSScrollView }.first)
      #expect(abs(scroll.contentView.bounds.height - 300) < 1)
      #expect((scroll.documentView?.frame.height ?? 0) > 10_000)
      try capture(root, name: "model-picker-\(dark ? "dark" : "light")")
      state.highlight = "model-498"
      try await Task.sleep(for: .milliseconds(100))
      #expect(scroll.contentView.bounds.origin.y > 10_000)
      #expect(state.selected == ["model-0"], "Keyboard navigation never commits selection")
      state.search = "499"
      try await Task.sleep(for: .milliseconds(100))
      #expect(state.selected == ["model-0"], "Filtering never drops a hidden selection")
      #expect((scroll.documentView?.frame.height ?? 0) <= 300)
    }
  }
}

@MainActor @Observable private final class SearchRevealState {
  var loaded = false
  var publication = 0
  var request = SettingsSearchRequest(
    entry: .init(destination: .application, anchor: .notifications))
}
private struct SearchRevealFixture: View {
  @Bindable var state: SearchRevealState
  var body: some View {
    SettingsPage(title: "Application") { _ in
      if state.loaded {
        Color.clear.frame(height: 1000)
        SearchTarget().frame(height: 30).settingsSearchAnchor(.notifications)
        Color.clear.frame(height: 1000)
      }
      Text("Publication \(state.publication)")
    }.environment(\.settingsSearchRequest, state.request)
  }
}
private struct SearchTarget: NSViewRepresentable {
  func makeNSView(context: Context) -> SearchTargetView { SearchTargetView() }
  func updateNSView(_ view: SearchTargetView, context: Context) {}
}
private final class SearchTargetView: NSView {}
@MainActor @Observable private final class PickerRevealState {
  var selected: Set<String> = ["model-0"]
  var highlight: String?
  var search = ""
}
private struct PickerRevealFixture: View {
  @Bindable var state: PickerRevealState
  let models: [StudioState.Model]
  var body: some View {
    VStack(spacing: 12) {
      NativeSearchField(text: $state.search, prompt: "Search models or providers")
      StudioCompareList(
        models: models, selected: $state.selected, search: state.search,
        single: true, highlighted: state.highlight)
    }.padding(16).studioPopoverSurface()
  }
}
private actor DiscoveryClient: ManagerLoading {
  private(set) var mutations = 0
  func snapshot() async throws -> ManagerSnapshot { try endpointFixture() }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    mutations += 1
    throw ManagerError.closed
  }
}
