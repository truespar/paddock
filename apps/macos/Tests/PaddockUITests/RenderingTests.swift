import AppKit
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

/// A native view-layout smoke gate, not a VoiceOver or interaction audit.
/// Optional images are build artifacts, never user screenshots or model data.
@Suite("Native view rendering") @MainActor
struct RenderingTests {
  @Test func workspaceNoticesWrapWithinNeutralCardsInBothThemes() async throws {
    for width in [320, 640, 960] {
      for dark in [false, true] {
        try await render(
          VStack(spacing: 8) {
            WorkspaceNotice(
              message: "Checking the saved endpoint and available memory, then loading the model…",
              kind: .progress)
            WorkspaceNotice(
              message:
                "The model could not start. Check the runner log for its diagnosis. Your saved configuration is retained.",
              kind: .warning, dismiss: {})
          }.padding(16).background(PaddockStyle.canvas),
          name: "workspace-notices-\(width)-\(dark ? "dark" : "light")",
          width: width, dark: dark, minimumHeight: 0, maximumHeight: 300)
      }
    }
  }

  @Test func endpointRowsContainIdentityStatusAndActionsAtNarrowAndWideSizes() async throws {
    let fixture = try endpointFixture()
    let runners = try ManagerWire.decode(
      [RunnerInfo].self,
      from: Data(
        #"[{"port":12345,"pid":42,"status":"ok","model":"qwen3.8-27b","display":"Qwen 3.8 27B with a long checkpoint description","endpoint":"http://127.0.0.1:12345"},{"port":12346,"pid":43,"status":"draining","model":"qwen3.8-27b","display":"Qwen 3.8 27B","endpoint":"http://127.0.0.1:12346"}]"#
          .utf8))
    let saved = try ManagerWire.decode(
      [ConfiguredEndpoint].self,
      from: Data(
        #"[{"port":12345,"model":"qwen3.8-27b","artifact":"mlx-4bit","revision":"revision","running":true,"local_only":true},{"port":12347,"model":"qwen3.8-27b","artifact":"mlx-4bit","display":"Saved Qwen endpoint","revision":"revision","running":false,"local_only":true}]"#
          .utf8))
    let snapshot = ManagerSnapshot(
      identity: fixture.identity, readiness: fixture.readiness,
      catalog: fixture.catalog, runners: runners, servers: saved)
    let workspace = WorkspaceModel(client: ImmediateLoader())
    for width in [420, 640, 960] {
      for dark in [false, true] {
        try await render(
          EndpointsView(workspace: workspace, snapshot: snapshot, onCreate: {}),
          name: "endpoint-rows-\(width)-\(dark)", width: width, dark: dark)
      }
    }
    #expect(
      EndpointRow.rows(snapshot: snapshot, latestJob: nil).map(\.status) == [
        "Running", "Stopping", "Stopped",
      ])
  }

  @Test func dropdownsKeepTheirShapeWithLongValuesAndDisabledState() async throws {
    for dark in [false, true] {
      try await render(
        VStack(alignment: .leading, spacing: 16) {
          Dropdown(title: "Format", value: "All formats") { Text("All formats") }
          Dropdown(title: "Order by", value: "Published · newest first", fillsWidth: true) {
            Text("Published · newest first")
          }
          Dropdown(
            title: "Provider", value: String(repeating: "Long provider name ", count: 8),
            fillsWidth: true
          ) { Text("Provider") }
          Dropdown(title: "Model", value: "Unavailable", fillsWidth: true) { Text("Unavailable") }
            .disabled(true)
          Spacer()
        }.padding(16).frame(maxWidth: .infinity, maxHeight: .infinity)
          .background(PaddockStyle.canvas),
        name: "dropdowns-240-\(dark ? "dark" : "light")", width: 240, dark: dark)
    }
  }

  @Test func providerMarksFitBothAppearances() async throws {
    for dark in [false, true] {
      try await render(
        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 24) {
          ForEach(ProviderArtwork.names.keys.sorted(), id: \.self) { vendor in
            VStack(spacing: 8) {
              HStack {
                ModelAvatar(vendor: vendor, size: 34)
                ModelAvatar(vendor: vendor, size: 44)
              }
              Text(vendor).font(.system(size: 11))
            }
          }
        }.padding(24).frame(maxWidth: .infinity, maxHeight: .infinity)
          .background(PaddockStyle.canvas),
        name: "provider-artwork-\(dark ? "dark" : "light")", width: 640, dark: dark)
    }
  }

  @Test func proposedWidthGateDetectsFixedWidthOverflow() {
    let controller = NSHostingController(rootView: Text("Too wide").frame(width: 900))
    #expect(controller.sizeThatFits(in: CGSize(width: 640, height: 820)).width > 640)
  }
  @Test func nativeComposerFitsNarrowDocumentColumns() async throws {
    for width in [280, 420, 639, 640, 760] {
      for dark in [false, true] {
        let chat = StudioWorkspace(client: ImmediateLoader())
        try await render(
          VStack {
            StudioComposerView(
              chat: chat, draft: .constant(StudioDraft(message: "Explain this document.")))
            Spacer()
          }.padding(8).background(PaddockStyle.canvas),
          name: "native-composer-\(width)-\(dark)", width: width, dark: dark)
        await chat.shutdown()
      }
    }
  }

  @Test func composerPopoverPanelsFitBothAppearances() async throws {
    for dark in [false, true] {
      let chat = StudioWorkspace(client: ImmediateLoader())
      let panels: [(String, Int, AnyView)] = [
        ("thinking", 250, AnyView(StudioReasoningControls(chat: chat))),
        ("tools", 380, AnyView(StudioToolsView(chat: chat))),
        ("instructions", 380, AnyView(StudioInstructionControls(chat: chat))),
        ("sampling", 320, AnyView(StudioSamplingControls(chat: chat))),
        ("context", 320, AnyView(StudioContextControls(chat: chat))),
        ("compare", 380, AnyView(StudioCompareView(chat: chat))),
        ("microphone", 360, AnyView(StudioMicrophoneSettings(chat: chat))),
      ]
      for (name, width, panel) in panels {
        try await render(
          panel.studioPopoverSurface(), name: "popover-\(name)-\(dark)", width: width, dark: dark,
          minimumHeight: 40, maximumHeight: 620)
      }
      await chat.shutdown()
    }
  }

  @Test func workspaceSplitUsesFullHeightBelowTransparentNativeChrome() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: ImmediateLoader())
    var navigation = WorkspaceNavigation()
    navigation.sidebarVisible = true
    let controller = NSHostingController(
      rootView: WorkspaceView(model: model, navigation: navigation))
    // This harness owns its viewport. Do not let NSHostingController's
    // asynchronous preferred-size updates resize it during chrome assertions.
    controller.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1000, height: 800),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbarStyle = .unifiedCompact
    window.toolbar = NSToolbar(identifier: "edge-layout-test")
    window.contentViewController = controller
    window.setContentSize(NSSize(width: 1000, height: 800))
    // AppKit finalizes title-bar geometry only after ordering the window.
    // Keep the fixture offscreen and never make it key.
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(80))
    controller.view.layoutSubtreeIfNeeded()
    // Require a real title-bar inset, otherwise an edge comparison could pass
    // simply because the harness had no native chrome to extend behind.
    #expect(window.contentLayoutRect.height < controller.view.bounds.height)
    // Column backgrounds and dividers fill the window. Only the column that
    // owns native controls reserves their row; preview headers do not inherit it.
    if let split = findSplit(in: controller.view) {
      let frame = split.convert(split.bounds, to: controller.view)
      #expect(abs(frame.minY) < 1)
      #expect(abs(frame.height - controller.view.bounds.height) < 1)
      #expect(abs(frame.minX) < 1)
      #expect(split.arrangedSubviews.last!.frame.minX >= 220 + split.dividerThickness)
    }
    #expect(controller.view.bounds.width == 1000)
    #expect(controller.sizeThatFits(in: CGSize(width: 1000, height: 800)).width <= 1000)
  }

  @Test func catalogSplitStaysBelowTheNativeHeader() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: CatalogLayoutLoader(), cloudClient: CloudFixture())
    await model.start()
    await model.cloud.refresh()
    defer { Task { await model.shutdown() } }
    for destination in [ManagerDestination.models, .cloudProviders] {
      for dark in [false, true] {
        var navigation = WorkspaceNavigation()
        navigation.showManager(destination)
        let controller = NSHostingController(
          rootView: WorkspaceView(model: model, navigation: navigation)
            .environment(\.colorScheme, dark ? .dark : .light))
        controller.sizingOptions = []
        let window = NSWindow(
          contentRect: NSRect(x: 0, y: 0, width: 920, height: 688),
          styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
          backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbarStyle = .unifiedCompact
        window.toolbar = NSToolbar(identifier: "catalog-header-test")
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentViewController = controller
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        for width: CGFloat in [920, 1200, 920] {
          window.setContentSize(NSSize(width: width, height: 688))
          try await Task.sleep(for: .milliseconds(100))
          controller.view.layoutSubtreeIfNeeded()
          let content = controller.view.convert(window.contentLayoutRect, from: nil)
          #expect(content.minY > 0, "Harness must have a real title-bar inset")
          if let split = findSplit(in: controller.view) {
            let frame = split.convert(split.bounds, to: controller.view)
            #expect(
              controller.view.bounds.insetBy(dx: -1, dy: -1).contains(frame),
              "Native workspace divider fills the window; controls clear the toolbar")
          }
          let scrolls = scrollViews(in: controller.view)
          #expect(
            scrolls.count == 3,
            "\(destination), \(width), dark=\(dark): Settings sidebar, list and detail must be mounted"
          )
          let workspaceScrolls = scrolls.filter {
            $0.convert($0.bounds, to: controller.view).minX >= 220
          }
          #expect(
            workspaceScrolls.count == 2, "Catalog list and detail remain separate scroll lanes")
          for scroll in scrolls {
            // A scroll backing view may extend behind chrome, but its content
            // must be inset. Unlike HSplitView, NSScrollView reports that inset.
            var frame = scroll.convert(scroll.bounds, to: controller.view)
            let insets = scroll.contentInsets
            frame.origin.y += insets.top
            frame.size.height -= insets.top + insets.bottom
            #expect(
              content.insetBy(dx: -1, dy: -1).contains(frame),
              "\(destination), \(width), dark=\(dark): scroll \(frame); content \(content)")
            #expect(frame.height > 200, "A clipped or collapsed catalog is not a fix")
          }
        }
      }
    }
  }

  private func scrollViews(in view: NSView) -> [NSScrollView] {
    if let scroll = view as? NSScrollView { return [scroll] }
    return view.subviews.flatMap { scrollViews(in: $0) }
  }

  private func findSplit(in view: NSView) -> NSSplitView? {
    if let split = view as? NSSplitView { return split }
    for child in view.subviews {
      if let split = findSplit(in: child) { return split }
    }
    return nil
  }

  @Test func exportRowsAndDetailFitTheNarrowContentPane() async throws {
    _ = NSApplication.shared
    let snapshot = try await ImmediateLoader().snapshot()
    for dark in [false, true] {
      try await render(
        ModelLibraryView(snapshot: snapshot, format: .mlx),
        name: "mlx-640-\(dark ? "dark" : "light")", width: 640, dark: dark)
      for entry in LibraryCatalog.entries(
        catalog: snapshot.catalog, backend: snapshot.readiness.backend, format: .mlx)
      {
        try await render(
          ModelDetailView(
            model: entry.model, backend: snapshot.readiness.backend,
            canStart: true, onStart: { _, _ in }, artifactID: .constant(entry.initialArtifact)),
          name: "export-\(entry.id)-350-\(dark ? "dark" : "light")",
          width: 350, dark: dark)
      }
    }
  }

  @Test func startupAndReadyLayouts() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: ImmediateLoader())
    try await render(WorkspaceView(model: model), name: "startup", width: 900, dark: false)
    await model.start()
    #expect(model.state == .ready)
    for (width, dark) in [(900, false), (900, true), (1440, false), (1440, true)] {
      try await render(
        WorkspaceView(model: model), name: "workspace-\(width)-\(dark ? "dark" : "light")",
        width: width, dark: dark)
    }
    let snapshot = try #require(model.snapshot)
    try await render(OverviewView(snapshot: snapshot), name: "this-mac", width: 900, dark: false)
    try await render(ModelLibraryView(snapshot: snapshot), name: "models", width: 900, dark: false)
    try await render(
      EndpointsView(workspace: model, snapshot: snapshot, onCreate: {}), name: "endpoints-empty",
      width: 900, dark: true)
    let populated = try endpointFixture()
    let catalogModel = try #require(populated.catalog.models.first)
    try await render(
      ModelDetailView(
        model: catalogModel, backend: populated.readiness.backend, canStart: true,
        onStart: { _, _ in }, artifactID: .constant(nil)),
      name: "model-detail", width: 900, dark: true)
    try await render(
      EndpointsView(workspace: model, snapshot: populated, onCreate: {}), name: "endpoints",
      width: 900, dark: false)
    try await render(
      EndpointsView(
        workspace: model, snapshot: populated, onCreate: {},
        creation: AnyView(StartModelView(snapshot: populated, onBrowse: {}) { _ in false })),
      name: "inline-instance-draft", width: 900, dark: false)
  }

  @Test func bothWorkspaceModesAndEveryDestinationFit() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: ImmediateLoader(), cloudClient: CloudFixture())
    await model.start()
    for dark in [false, true] {
      for destination in StudioDestination.allCases {
        var navigation = WorkspaceNavigation()
        navigation.studio = destination
        navigation.sidebarVisible = true
        try await render(
          WorkspaceView(model: model, navigation: navigation),
          name: "studio-\(destination)-\(dark)",
          width: Int(WorkspaceView.minimumWidth(for: navigation)), dark: dark)
      }
      for destination in ManagerDestination.allCases {
        var navigation = WorkspaceNavigation()
        navigation.showManager(destination)
        try await render(
          WorkspaceView(model: model, navigation: navigation),
          name: "manager-\(destination)-900-\(dark)", width: 900, dark: dark)
      }
      try await render(
        StudioConversationView(
          chat: StudioWorkspace(client: ImmediateLoader()),
          draft: .constant(
            StudioDraft(message: String(repeating: "A longer message draft. ", count: 100)))),
        name: "composer-long-draft-640-\(dark)", width: 640, dark: dark)
      try await render(
        ProjectSetupView(), name: "project-setup-\(dark)", width: 640, dark: dark,
        minimumHeight: 350, maximumHeight: 600)
    }
  }

  @Test func cloudBrowserAndProviderDetailsFitBothAppearances() async throws {
    let model = CloudBrowserModel(client: CloudFixture())
    await model.refresh()
    for dark in [false, true] {
      try await render(
        CloudModelsView(model: model), name: "cloud-browser-640-\(dark)", width: 640, dark: dark)
      let entry = try #require(model.models.first)
      await model.loadProviders(entry.id, force: true)
      #expect(model.selectProvider("provider/us-east-1", for: entry.id))
      try await render(
        CloudModelDetailView(entry: entry, browser: model), name: "cloud-providers-350-\(dark)",
        width: 350, dark: dark)
      let speech = try #require(model.models.first { $0.asr == true })
      await model.loadProviders(speech.id, force: true)
      try await render(
        CloudModelDetailView(entry: speech, browser: model), name: "cloud-audio-350-\(dark)",
        width: 350, dark: dark)
    }
  }

  @Test func cloudTabsSavedPicksAndSparseProvidersFitBothThemes() async throws {
    let picks = (0..<8).map { CloudModelPick(id: "fixture/model-\($0)") }
    let rows = try CloudService.allCases.map { service in
      try CloudWorkflowFixture.row(
        id: service.rawValue,
        base: service == .custom ? "https://fixture.invalid/v1" : service.base,
        kind: service.kind, models: picks)
    }
    let connections = ConnectionsModel(client: CloudWorkflowFixture(rows: rows))
    await connections.refresh()
    let model = CloudBrowserModel(client: CloudFixture())
    await model.refresh()
    for entry in model.models { await model.loadProviders(entry.id, force: true) }
    for row in rows where !row.isOpenRouter { await connections.catalog(for: row).refresh() }
    for dark in [false, true] {
      for service in CloudService.allCases {
        try await render(
          CloudModelsView(model: model, connections: connections, service: service),
          name: "cloud-\(service)-saved-640-\(dark)", width: 640, dark: dark)
      }
      try await render(
        CloudModelsView(model: model, connections: connections),
        name: "cloud-openrouter-saved-1000-\(dark)", width: 1000, dark: dark)
      let empty = ConnectionsModel(client: CloudWorkflowFixture())
      await empty.refresh()
      try await render(
        CloudModelsView(model: model, connections: empty, service: .anthropic),
        name: "cloud-key-gate-640-\(dark)", width: 640, dark: dark)
      let editor = ConnectionEditor(
        client: CloudWorkflowFixture(), connection: nil, openRouter: false,
        pick: CloudModelPick(id: "claude-fixture"), service: .anthropic)
      try await render(
        ConnectionReviewView(editor: editor, onCancel: {}),
        name: "cloud-known-key-572-\(dark)", width: 572, dark: dark, minimumHeight: 100)
    }
    await connections.stop()
    model.stop()
  }

  @Test func compactProviderDetailDoesNotCreateANearlyFullHeightScrollThumb() async throws {
    let model = CloudBrowserModel(client: CloudFixture())
    await model.refresh()
    let entry = try #require(model.models.first)
    await model.loadProviders(entry.id, force: true)
    let controller = NSHostingController(
      rootView: CloudModelDetailView(entry: entry, browser: model))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 350, height: 650),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    controller.sizingOptions = []
    window.contentViewController = controller
    window.setContentSize(NSSize(width: 350, height: 650))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer {
      window.close()
      model.stop()
    }
    try await Task.sleep(for: .milliseconds(100))
    controller.view.layoutSubtreeIfNeeded()
    let scroll = try #require(scrollViews(in: controller.view).first)
    #expect(
      (scroll.documentView?.frame.height ?? .infinity) <= scroll.contentView.bounds.height + 1,
      "Two provider rows plus model metadata should fit without scrolling at 350 × 650")
  }

  @Test func thousandRowCatalogKeepsItsScrollGeometryWhileRecycling() async throws {
    let model = CloudBrowserModel(client: LargeCloudFixture())
    await model.refresh()
    let controller = NSHostingController(
      rootView: CloudModelsView(model: model).frame(width: 640, height: 820))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 640, height: 820),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = controller
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer {
      window.close()
      model.stop()
    }
    try await Task.sleep(for: .milliseconds(100))
    controller.view.layoutSubtreeIfNeeded()
    func scrollViews(_ view: NSView) -> [NSScrollView] {
      (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap(scrollViews)
    }
    let list = try #require(
      scrollViews(controller.view).max {
        ($0.documentView?.frame.height ?? 0) < ($1.documentView?.frame.height ?? 0)
      })
    let height = try #require(list.documentView?.frame.height)
    #expect(list.contentView.bounds.height > 200)
    #expect(controller.view.bounds.width == 640)
    #expect(controller.view.bounds.height == 820)
    #expect(
      abs(height - 78_014) < 1,
      "1,000 fixed 76-point catalog rows, 2-point spacing, 16-point footer")
    for offset in [600.0, 15000, 40000, 65000, 12000, 0] {
      list.contentView.scroll(to: NSPoint(x: 0, y: offset))
      list.reflectScrolledClipView(list.contentView)
      try await Task.sleep(for: .milliseconds(40))
      controller.view.layoutSubtreeIfNeeded()
      let updated = try #require(list.documentView?.frame.height)
      #expect(
        abs(updated - height) < 1,
        "Catalog height changed from \(height) to \(updated) while scrolling")
      #expect(
        abs(list.contentView.bounds.origin.y - offset) < 1,
        "Viewport jumped while rows were recycled")
      #expect(
        (list.verticalScroller?.knobProportion ?? 1) < 0.02,
        "A thousand-row list must not have a page-height thumb")
    }
  }

  private func render<Content: View>(
    _ content: Content, name: String, width: Int, dark: Bool,
    minimumHeight: CGFloat = 650, maximumHeight: CGFloat? = nil
  )
    async throws
  {
    let controller = NSHostingController(
      rootView: content.environment(\.colorScheme, dark ? .dark : .light))
    let view = controller.view
    let size = NSRect(x: 0, y: 0, width: width, height: 820)
    let window = NSWindow(
      contentRect: size, styleMask: [.titled, .resizable, .closable], backing: .buffered,
      defer: false)
    window.isReleasedWhenClosed = false
    window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
    window.contentView = view
    view.frame = size
    defer { window.close() }
    // AppKit-backed lists install their rows on a subsequent main-loop turn.
    // This small render-settling wait is test-only, never a UI reveal delay.
    try await Task.sleep(for: .milliseconds(80))
    view.layoutSubtreeIfNeeded()
    view.displayIfNeeded()
    // NSView.fittingSize asks for intrinsic size, not whether a flexible page
    // accepts this viewport (a max-width 900 page can still fit in 640).
    // Propose the actual available size; the fixed-width regression above
    // verifies this gate still rejects genuine horizontal overflow.
    let measured = controller.sizeThatFits(in: size.size)
    #expect(measured.width <= CGFloat(width), "\(name) width")
    #expect(view.bounds.height >= minimumHeight)
    if let maximumHeight {
      // The harness proposed an 820-point window; preferred-size propagation
      // can keep those bounds for another run-loop turn. A popover's contract
      // is the content's measured height, not that asynchronous window resize.
      #expect(measured.height <= maximumHeight, "\(name): \(measured.height) > \(maximumHeight)")
    }
    if let directory = ProcessInfo.processInfo.environment["PADDOCK_UI_SNAPSHOT_DIR"] {
      let target = URL(fileURLWithPath: directory, isDirectory: true)
      try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
      let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
      view.cacheDisplay(in: view.bounds, to: bitmap)
      let png = try #require(bitmap.representation(using: .png, properties: [:]))
      try png.write(to: target.appending(path: name + ".png"), options: .atomic)
    }
  }
}

private struct CatalogLayoutLoader: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { try endpointFixture() }
}

private struct ImmediateLoader: ManagerLoading {
  func snapshot() async throws
    -> ManagerSnapshot
  {
    if let path = ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] {
      let dir = URL(fileURLWithPath: path, isDirectory: true)
      return try ManagerSnapshot(
        identity: ManagerWire.decode(
          ManagerIdentity.self, from: Data(contentsOf: dir.appending(path: "server.json"))),
        readiness: ManagerWire.decode(
          Readiness.self, from: Data(contentsOf: dir.appending(path: "readiness.json"))),
        catalog: ManagerWire.decode(
          ModelCatalog.self, from: Data(contentsOf: dir.appending(path: "catalog.json"))),
        runners: ManagerWire.decode(
          [RunnerInfo].self, from: Data(contentsOf: dir.appending(path: "runners.json"))))
    }
    return try fixture()
  }
}
