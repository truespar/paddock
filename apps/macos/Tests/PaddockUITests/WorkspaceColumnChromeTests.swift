import AppKit
import PaddockClient
import PaddockNativeMarkdown
import SwiftUI
import Testing

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Column-owned native window chrome", .serialized) @MainActor
struct WorkspaceColumnChromeTests {
  @Test func flatHeaderSeparatesSharpContentWithoutAnyBackdropBlur() async throws {
    guard #available(macOS 26.0, *) else { return }
    _ = NSApplication.shared
    // A continuous strip must stop at the header, without fading or bleeding
    // into it. Away from the strip there must be just one subtle separator.
    let host = NSHostingController(
      rootView: StudioConversationChrome {
        ConversationSelectionSurface(items: []) {
          ScrollView {
            PaddockStyle.primary.frame(width: 80, height: 2000).frame(maxWidth: .infinity)
          }
          .modifier(
            StudioConversationBars {
              EmptyView()
            } footer: {
              EmptyView()
            }
          )
          .background(PaddockStyle.canvas)
        }
      })
    host.sizingOptions = []
    host.safeAreaRegions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 760, height: 740),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbarStyle = .unifiedCompact
    window.toolbar = NSToolbar(identifier: "continuous-chrome-fixture")
    window.contentViewController = host
    window.setContentSize(NSSize(width: 760, height: 740))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for dark in [false, true] {
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      try await settleChrome(host.view)
      let scroll = try #require(allViews(host.view).compactMap { $0 as? NSScrollView }.first)
      scroll.contentView.scroll(to: NSPoint(x: 0, y: 300))
      scroll.reflectScrolledClipView(scroll.contentView)
      try await settleChrome(host.view)
      let viewport = scroll.convert(scroll.bounds, to: nil)
      #expect(abs(viewport.maxY - window.contentLayoutRect.maxY) < 1)
      #expect(scroll.contentInsets.top == 0)
      if let directory = ProcessInfo.processInfo.environment["PADDOCK_SCROLL_CHROME_CAPTURE_DIR"] {
        let image = try capture(
          window, directory: directory, name: "flat-header-\(dark ? "dark" : "light")")
        let scale = CGFloat(image.pixelsWide) / window.frame.width
        func luminance(_ y: CGFloat, x: CGFloat = 380) throws -> CGFloat {
          let color = try #require(
            image.colorAt(x: Int(x * scale), y: Int(y * scale))?.usingColorSpace(.deviceRGB))
          return (color.redComponent + color.greenComponent + color.blueComponent) / 3
        }
        let canvas = try luminance(100, x: 600)
        let ink = try luminance(100)
        #expect(abs(ink - canvas) > 0.5)
        let edge = window.frame.height - window.contentLayoutRect.maxY
        for y in stride(from: CGFloat(8), through: edge - 3, by: 1) {
          #expect(abs(try luminance(y) - canvas) < 0.02, "Content must not bleed into the header")
        }
        for y in stride(from: edge + 3, through: edge + 32, by: 1) {
          #expect(abs(try luminance(y) - ink) < 0.02, "Content below the header must stay sharp")
        }
        let separator = try stride(from: edge - 2, through: edge + 2, by: 1 / scale).map {
          abs(try luminance($0, x: 600) - canvas)
        }
        #expect((separator.max() ?? 0) > 0.02, "The header should have a visible separator")
        #expect(separator.filter { $0 > 0.02 }.count <= Int(scale) + 1, "Only a single hairline")
        #expect(abs(try luminance(20, x: 600) - canvas) < 0.02)
      }
    }
  }

  @Test func narrowConversationKeepsWindowControlsClearWhileScrolled() async throws {
    _ = NSApplication.shared
    let client = ChromeNoCore()
    let chat = StudioWorkspace(client: client)
    let model = WorkspaceModel(client: client, preparedStudio: chat)
    chat.apply(try artifactState(previews: false))
    model.navigation.sidebarVisible = false
    let host = NSHostingController(rootView: WorkspaceView(model: model))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 760, height: 740),
      styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbarStyle = .unifiedCompact
    window.toolbar = NSToolbar(identifier: "narrow-conversation-chrome")
    window.contentViewController = host
    window.setContentSize(NSSize(width: 760, height: 740))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for (width, sidebar, dark): (CGFloat, Bool, Bool) in [
      (760, false, false), (760, false, true), (902, false, false), (902, false, true),
      (902, true, false), (902, true, true),
    ] {
      model.navigation.sidebarVisible = sidebar
      window.setContentSize(NSSize(width: width, height: 740))
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      for _ in 0..<10 {
        host.view.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(40))
      }
      let text = try #require(
        allViews(host.view).compactMap { $0 as? NSTextView }
          .first { $0.string.hasPrefix("Full-height conversation fixture") })
      let scroll = try #require(text.enclosingScrollView)
      let viewport = scroll.convert(scroll.bounds, to: nil)
      #expect(abs(viewport.maxY - window.contentLayoutRect.maxY) < 1)
      #expect(abs(viewport.minY - host.view.convert(host.view.bounds, to: nil).minY) < 1)
      #expect(scroll.contentInsets.top == 0)
      let controls = try [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton].map {
        try #require(window.standardWindowButton($0))
      }
      for button in controls {
        let point = button.convert(NSPoint(x: button.bounds.midX, y: button.bounds.midY), to: nil)
        let frameView = try #require(window.contentView?.superview)
        let target = frameView.hitTest(frameView.convert(point, from: nil))
        #expect(target === button || target?.isDescendant(of: button) == true)
      }
      scroll.contentView.scroll(to: NSPoint(x: 0, y: -scroll.contentInsets.top))
      scroll.reflectScrolledClipView(scroll.contentView)
      try await settleChrome(host.view)
      let directory = ProcessInfo.processInfo.environment["PADDOCK_SCROLL_CHROME_CAPTURE_DIR"]
      let name =
        "titlebar-\(Int(width))-\(sidebar ? "sidebar" : "no-sidebar")-\(dark ? "dark" : "light")"
      let reference: NSBitmapImageRep?
      if let directory {
        reference = try capture(window, directory: directory, name: name + "-rest")
        try expectCanvasInEmptyHeader(try #require(reference), window: window)
      } else {
        reference = nil
      }
      for offset: CGFloat in [150, 173, 209] {
        scroll.contentView.scroll(to: NSPoint(x: 0, y: offset))
        scroll.reflectScrolledClipView(scroll.contentView)
        try await settleChrome(host.view)
        #expect(abs(scroll.documentVisibleRect.minY - offset) < 1)
        if let directory, let reference {
          let scrolled = try capture(window, directory: directory, name: name + "-\(Int(offset))")
          let titlebar = window.frame.height - window.contentLayoutRect.maxY
          // Compare actual window pixels, not layout insets: the old automatic
          // style passes geometry tests while drawing text over these controls.
          let controlBand = NSRect(x: 8, y: 4, width: width - 16, height: titlebar - 12)
          let changed = try changedFraction(
            reference, scrolled, region: controlBand, window: window)
          #expect(
            changed < 0.01, "Text leaked into the titlebar: \(name), offset \(offset), \(changed)")
          let bodyBand = NSRect(x: sidebar ? 300 : 28, y: titlebar + 20, width: 300, height: 160)
          #expect(
            try changedFraction(reference, scrolled, region: bodyBand, window: window) > 0.03,
            "The visual check must exercise genuinely scrolled text")
        }
      }
    }
    await model.shutdown()
  }

  private func settleChrome(_ view: NSView) async throws {
    for _ in 0..<5 {
      view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(40))
    }
  }

  private func expectCanvasInEmptyHeader(
    _ image: NSBitmapImageRep, window: NSWindow
  ) throws {
    // An opaque/tinted material can hide every glyph and pass the legibility
    // check, but still recolour the header. At rest this empty region must be
    // the actual app canvas, in both appearances, not a system material tint.
    let scale = CGFloat(image.pixelsWide) / window.frame.width
    // Compare rendered pixels in the same capture, not NSColor components:
    // screen capture's display/HDR conversion can differ from the source RGB.
    // The median of this text fixture's body region is its canvas, not its ink.
    let titlebar = window.frame.height - window.contentLayoutRect.height
    var samples: [NSColor] = []
    for y in stride(from: titlebar + 80, through: titlebar + 160, by: 8) {
      for x in stride(from: window.frame.width - 100, through: window.frame.width - 40, by: 8) {
        samples.append(
          try #require(
            image.colorAt(x: Int(x * scale), y: Int(y * scale))?.usingColorSpace(.deviceRGB)))
      }
    }
    let red = samples.map(\.redComponent).sorted()[samples.count / 2]
    let green = samples.map(\.greenComponent).sorted()[samples.count / 2]
    let blue = samples.map(\.blueComponent).sorted()[samples.count / 2]
    for dx: CGFloat in [48, 64, 80] {
      let actual = try #require(
        image.colorAt(x: Int((window.frame.width - dx) * scale), y: Int(20 * scale))?
          .usingColorSpace(.deviceRGB))
      let error = max(
        abs(actual.redComponent - red),
        abs(actual.greenComponent - green),
        abs(actual.blueComponent - blue))
      #expect(error < 0.02, "The empty header must preserve the app canvas colour: \(error)")
    }
  }

  private func capture(_ window: NSWindow, directory: String, name: String) throws
    -> NSBitmapImageRep
  {
    // Only this offscreen synthetic window; no desktop input or user content.
    let url = URL(fileURLWithPath: directory).appendingPathComponent(name + ".png")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    process.arguments = ["-x", "-o", "-l", String(window.windowNumber), url.path]
    try process.run()
    process.waitUntilExit()
    #expect(process.terminationStatus == 0)
    return try #require(NSBitmapImageRep(data: Data(contentsOf: url)))
  }

  private func changedFraction(
    _ before: NSBitmapImageRep, _ after: NSBitmapImageRep,
    region: NSRect, window: NSWindow
  ) throws -> Double {
    #expect(before.pixelsWide == after.pixelsWide && before.pixelsHigh == after.pixelsHigh)
    let scale = CGFloat(before.pixelsWide) / window.frame.width
    #expect(abs(CGFloat(before.pixelsHigh) / scale - window.frame.height) < 1)
    var changed = 0
    var count = 0
    for y in Int(region.minY * scale)..<Int(region.maxY * scale) {
      for x in Int(region.minX * scale)..<Int(region.maxX * scale) {
        let a = try #require(before.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
        let b = try #require(after.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
        let difference = max(
          abs(a.redComponent - b.redComponent),
          abs(a.greenComponent - b.greenComponent), abs(a.blueComponent - b.blueComponent))
        if difference > 0.12 { changed += 1 }
        count += 1
      }
    }
    return Double(changed) / Double(max(1, count))
  }

  @Test func productionConversationUsesFullHeightWithEitherSidebarState() async throws {
    _ = NSApplication.shared
    let client = ChromeNoCore()
    let chat = StudioWorkspace(client: client)
    let model = WorkspaceModel(client: client, preparedStudio: chat)
    chat.apply(try artifactState())
    model.navigation.sidebarVisible = true
    let host = NSHostingController(rootView: WorkspaceView(model: model))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1800, height: 800),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbarStyle = .unifiedCompact
    window.toolbar = NSToolbar(identifier: "production-compare-chrome")
    window.contentViewController = host
    window.setContentSize(NSSize(width: 1800, height: 800))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(150))
    host.view.layoutSubtreeIfNeeded()
    let split = try #require(
      allViews(host.view).compactMap { ($0 as? NSSplitView)?.delegate as? StudioSplitController }
        .first { $0.right })
    for sidebar in [true, false, true] {
      model.navigation.sidebarVisible = sidebar
      for dark in [false, true] {
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        try await Task.sleep(for: .milliseconds(80))
        host.view.layoutSubtreeIfNeeded()
        split.splitView.setPosition(600, ofDividerAt: 0)
        try await Task.sleep(for: .milliseconds(100))
        host.view.layoutSubtreeIfNeeded()
        let text = try #require(
          allViews(split.chatHost.view).compactMap { $0 as? NSTextView }
            .first { $0.string.hasPrefix("Full-height conversation fixture") })
        let scroll = try #require(text.enclosingScrollView)
        let frame = scroll.convert(scroll.bounds, to: host.view)
        let viewport = scroll.convert(scroll.bounds, to: nil)
        #expect(abs(viewport.maxY - window.contentLayoutRect.maxY) < 1)
        #expect(scroll.contentInsets.top == 0)
        #expect(
          abs(frame.maxY - host.view.bounds.height) < 1,
          "The viewport must still reach the bottom behind the composer")
        #expect(
          abs(split.documentHost.view.convert(split.documentHost.view.bounds, to: host.view).minY)
            < 1,
          "The adjacent preview must keep its full-height layout")
        #expect(
          scroll.scrollerInsets.top == StudioConversationSpacing.scrollIndicatorInset
            && scroll.scrollerInsets.bottom == StudioConversationSpacing.scrollIndicatorInset)
        let scroller = try #require(scroll.verticalScroller)
        let track = scroller.convert(scroller.bounds, to: host.view)
        #expect(
          abs(
            track.minY - frame.minY - scroll.contentInsets.top
              - StudioConversationSpacing.scrollIndicatorInset)
            <= 2,
          "The native indicator must clear the top bar: \(track)")
        #expect(
          abs(
            track.maxY
              - (host.view.bounds.height - scroll.contentInsets.bottom
                - StudioConversationSpacing.scrollIndicatorInset))
            <= 2,
          "The native indicator must clear the composer: \(track)")
      }
    }
    // Exercise production headers, not replacement close closures. Closing a
    // neighbour must leave the other preview open, then collapse only the last.
    let previews = try #require(
      allViews(split.documentHost.view).compactMap { $0 as? NSSplitView }
        .first { $0.arrangedSubviews.count == 2 })
    let first = try #require(previews.arrangedSubviews.first)
    try clickClose(in: first, window: window)
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    #expect(chat.presentedArtifacts.map(\.id) == ["artifact-1"])
    #expect(chat.selectedArtifactId == "artifact-1")
    #expect(!split.documentItem.isCollapsed)
    // The one remaining preview uses the regular single-pane layout.
    try clickClose(in: split.documentHost.view, window: window)
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    #expect(chat.selectedArtifactId == nil && chat.presentedArtifacts.isEmpty)
    #expect(split.documentItem.isCollapsed)
    let transcriptText = try #require(
      allViews(split.chatHost.view).compactMap { $0 as? NSTextView }
        .first { $0.string.hasPrefix("Full-height conversation fixture") })
    let transcriptScroll = try #require(transcriptText.enclosingScrollView)
    #expect(
      abs(transcriptScroll.bounds.width - split.chatHost.view.bounds.width) < 1,
      "Closing the last preview must not insert a new rail beside the conversation")
    await model.shutdown()
  }

  private func clickClose(in pane: NSView, window: NSWindow) throws {
    let topInset = max(0, pane.convert(pane.bounds, to: nil).maxY - window.contentLayoutRect.maxY)
    let point = pane.convert(
      NSPoint(
        x: pane.bounds.maxX - 23,
        y: pane.isFlipped ? pane.bounds.minY + topInset + 21 : pane.bounds.maxY - topInset - 21),
      to: nil)
    for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
      window.sendEvent(
        try #require(
          NSEvent.mouseEvent(
            with: type, location: point,
            modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: window.windowNumber, context: nil, eventNumber: 1,
            clickCount: 1, pressure: 1)))
    }
  }

  @Test func fullHeightPreviewHeaderReceivesRealMouseClicks() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: ChromeNoCore())
    for dark in [false, true] {
      let state = ChromeFixtureState()
      let host = NSHostingController(
        rootView: ChromeFixture(state: state, model: model)
          .environment(\.colorScheme, dark ? .dark : .light))
      host.sizingOptions = []
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 1200, height: 700),
        styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
        backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.titleVisibility = .hidden
      window.titlebarAppearsTransparent = true
      window.toolbarStyle = .unifiedCompact
      window.toolbar = NSToolbar(identifier: "column-hit-test")
      window.contentViewController = host
      window.setContentSize(NSSize(width: 1200, height: 700))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for sidebar in [true, false, true] {
        model.navigation.sidebarVisible = sidebar
        for width: CGFloat in [1200, 1000, 1200] {
          window.setContentSize(NSSize(width: width, height: 700))
          try await Task.sleep(for: .milliseconds(100))
          host.view.layoutSubtreeIfNeeded()
          let split = try #require(
            allViews(host.view).compactMap { $0 as? NSSplitView }
              .first { ($0.delegate as? StudioSplitController)?.right == true })
          let splitRect = split.convert(split.bounds, to: host.view)
          #expect(abs(splitRect.minY) < 1)
          #expect(abs(splitRect.height - host.view.bounds.height) < 1)
          let header = state.probe.convert(state.probe.bounds, to: host.view)
          #expect(
            abs(header.minY - (host.view.bounds.height - window.contentLayoutRect.height)) < 1)
          #expect(header.height >= 38 && header.height < 60)
          #expect(window.contentLayoutRect.height < host.view.bounds.height)
          // Last header button is the native Close action. Exercise window event
          // dispatch, not AXPress, which bypasses title-bar hit interception.
          let point = state.probe.convert(
            NSPoint(x: state.probe.bounds.maxX - 23, y: state.probe.bounds.midY), to: nil)
          let before = state.hideCount
          for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            window.sendEvent(
              try #require(
                NSEvent.mouseEvent(
                  with: type, location: point,
                  modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                  windowNumber: window.windowNumber, context: nil, eventNumber: 1,
                  clickCount: 1, pressure: 1)))
          }
          try await Task.sleep(for: .milliseconds(30))
          #expect(
            state.hideCount == before + 1, "Native title bar must not intercept preview controls")
        }
      }
      // Compare introduces HSplitView-owned hosting roots inside our retained
      // split. They must not reapply NSWindow's safe area to each writer pane.
      model.navigation.sidebarVisible = true
      window.setContentSize(NSSize(width: 1700, height: 700))
      state.comparison = true
      try await Task.sleep(for: .milliseconds(100))
      host.view.layoutSubtreeIfNeeded()
      let split = try #require(
        allViews(host.view).compactMap { $0 as? NSSplitView }
          .first { ($0.delegate as? StudioSplitController)?.right == true })
      split.setPosition(600, ofDividerAt: 0)
      try await Task.sleep(for: .milliseconds(100))
      host.view.layoutSubtreeIfNeeded()
      for probe in [state.probe, state.secondProbe] {
        let header = probe.convert(probe.bounds, to: host.view)
        #expect(abs(header.minY - (host.view.bounds.height - window.contentLayoutRect.height)) < 1)
        #expect(header.width >= 340)
        let before = state.hideCount
        let point = probe.convert(NSPoint(x: probe.bounds.maxX - 23, y: probe.bounds.midY), to: nil)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
          window.sendEvent(
            try #require(
              NSEvent.mouseEvent(
                with: type, location: point,
                modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                windowNumber: window.windowNumber, context: nil, eventNumber: 1,
                clickCount: 1, pressure: 1)))
        }
        try await Task.sleep(for: .milliseconds(30))
        #expect(state.hideCount == before + 1)
      }
      state.comparison = false
    }
    await model.shutdown()
  }

  private func find<T: NSView>(_ type: T.Type, in view: NSView) -> T? {
    (view as? T) ?? view.subviews.lazy.compactMap { find(type, in: $0) }.first
  }
  private func allViews(_ root: NSView) -> [NSView] { [root] + root.subviews.flatMap(allViews) }
  private func artifactState(previews: Bool = true) throws -> StudioState {
    let artifacts: [[String: Any]] = (0..<2).map { i in
      [
        "id": "artifact-\(i)", "model": "writer-\(i)", "kind": "html", "title": "Page \(i)",
        "language": "html", "versions": 1, "updatedAt": 1,
      ]
    }
    let fields: [String: Any] = [
      "version": 1, "revision": 1, "history": [], "models": [], "selectedModels": [],
      "busy": false, "loading": false, "error": "", "previewing": true, "unsavedEdits": false,
      "viewport": ["left": 0, "width": 760], "settings": [:], "tools": [],
      "conversation": [
        "id": "chrome", "title": "Chrome fixture", "model": "local", "messageCount": 1,
      ],
      "capabilities": [
        "reasoning": "", "levels": [], "reasoningDefault": "", "reasoningOff": false,
        "preserveThinking": false, "thinkingBudget": false, "webSearch": false, "vision": false,
        "context": 4096, "ocrModes": [], "docParser": false, "pdfRaster": false,
      ],
      "nativeArtifacts": previews ? artifacts : [], "nativeArtifactsPaneOpen": previews,
      "nativeTranscript": [
        "available": true, "notice": "",
        "messages": [
          [
            "id": "message", "role": "assistant", "model": "local",
            "text":
              "Full-height conversation fixture\n\n"
              + (1...30).map {
                "Paragraph \($0). The scrollbar track must span the window, while content clears the composer and window controls."
              }.joined(separator: "\n\n"),
            "reasoning": "", "streaming": false, "stopped": false, "error": "", "incomplete": false,
          ]
        ],
      ],
    ]
    return try JSONDecoder().decode(
      StudioState.self, from: JSONSerialization.data(withJSONObject: fields))
  }
}

@Observable @MainActor private final class ChromeFixtureState {
  var source = false
  var version = 0
  var hideCount = 0
  var comparison = false
  let probe = NSView()
  let secondProbe = NSView()
}

private struct ChromeFixture: View {
  @Bindable var state: ChromeFixtureState
  @Bindable var model: WorkspaceModel
  var body: some View {
    WorkspaceShell(
      navigation: $model.navigation, title: "Preview fixture", onNewChat: {},
      metrics: AnyView(Color.clear.frame(width: 32, height: 32))
    ) {
      VStack {
        Text("Chats")
        Spacer()
      }
    } content: {
      ChromeFixtureContent(state: state)
    }.ignoresSafeArea(.container, edges: .top)
  }
}

private struct ChromeFixtureContent: View {
  @Bindable var state: ChromeFixtureState
  @Environment(\.workspaceLeadingPaneInset) private var topInset
  var body: some View {
    StudioDocumentSplit(open: true, right: true) {
      Color.clear
    } document: {
      Group {
        if state.comparison {
          HSplitView {
            pane(state.probe)
            pane(state.secondProbe)
          }
        } else {
          pane(state.probe)
        }
      }.padding(.top, topInset)
    }
  }
  private func pane(_ probe: NSView) -> some View {
    VStack(spacing: 0) {
      NativeArtifactHeader(
        writer: ("", ""), model: "local", title: "Page",
        source: $state.source, version: $state.version, versions: [],
        dirty: false, saving: false, available: true, copied: false,
        save: {}, revert: {}, copy: {}, download: {}, close: { state.hideCount += 1 }
      )
      .background(ChromeFrameProbe(view: probe))
      Color.clear
    }.frame(minWidth: 340).fullHeightWorkspaceColumn()
  }
}

private struct ChromeFrameProbe: NSViewRepresentable {
  let view: NSView
  func makeNSView(context: Context) -> NSView { view }
  func updateNSView(_ nsView: NSView, context: Context) {}
}

private struct ChromeNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
}
