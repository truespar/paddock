import AppKit
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Instance settings search", .serialized) @MainActor
struct SettingsDeepLinkTests {
  @Test func searchUsesInstanceIdentityAndOnlyExposesPresentControls() async throws {
    let client = SettingsSearchClient()
    let workspace = WorkspaceModel(client: client)
    await workspace.refresh()
    let entries = workspace.instanceSettingsEntries
    #expect(Set(entries.map(\.id)).count == entries.count)
    let context = SettingsSearchIndex.search("context", instances: entries).filter {
      $0.port != nil
    }
    #expect(context.count == 2 && context.allSatisfy { $0.anchor == .context })
    #expect(
      SettingsSearchIndex.search("12401 memory budget", instances: entries).map(\.anchor) == [
        .memoryBudget
      ])
    #expect(entries.contains { $0.anchor == .modelLoading })
    #expect(entries.contains { $0.runtimeOption == "prefill_chunk" })
    #expect(entries.allSatisfy { !$0.keywords.contains("do-not-index-values") })

    let snapshot = try await client.snapshot()
    let endpoint = try #require(snapshot.servers?.first)
    let imageCatalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"qwen","display":"Images","capability":["image-generation"],"installed":true,"total_size":1,"artifacts":[]}]}"#
          .utf8))
    let editor = EndpointEditor(
      client: client, endpoint: endpoint, pid: nil, catalog: imageCatalog.models)
    #expect(
      !editor.searchableSettings.contains {
        [.context, .workload, .speculation, .webSearch, .mcpServers].contains($0.anchor)
      })
    #expect(await client.writes == 0)
    _ = await workspace.shutdown()
  }

  @Test func openingResultsRevealsTheRightLensWithoutSavingOrDiscardingDrafts() async throws {
    let client = SettingsSearchClient()
    let workspace = WorkspaceModel(client: client)
    await workspace.refresh()
    func entry(_ anchor: SettingsAnchor, _ port: UInt16 = 12400) throws -> SettingsSearchEntry {
      try #require(
        workspace.instanceSettingsEntries.first { $0.anchor == anchor && $0.port == port })
    }
    #expect(workspace.openSetting(try entry(.loadWait)))
    let editor = try #require(workspace.endpointEditor)
    #expect(workspace.editingEndpoint && editor.advanced && !editor.dirty)
    #expect(workspace.openSetting(try entry(.context)))
    #expect(!editor.advanced && !editor.dirty)
    #expect(workspace.openSetting(try entry(.webSearch)))
    #expect(workspace.systemToolsPort == 12400)
    editor.context = "32768"
    #expect(!workspace.openSetting(try entry(.memoryBudget, 12401)))
    #expect(workspace.endpointEditor === editor && editor.context == "32768")
    #expect(workspace.detailEndpointPort == 12400 && workspace.desktopError != nil)
    #expect(await client.writes == 0)
    _ = await workspace.shutdown()
  }

  @Test func actualEditFormRevealsDeepControlsInBothAppearances() async throws {
    _ = NSApplication.shared
    let client = SettingsSearchClient()
    let workspace = WorkspaceModel(client: client)
    await workspace.refresh()
    for dark in [false, true] {
      for anchor in [SettingsAnchor.context, .memoryBudget, .loadWait] {
        let entry = try #require(workspace.instanceSettingsEntries.first { $0.anchor == anchor })
        #expect(workspace.openSetting(entry))
        let host = NSHostingController(
          rootView:
            EndpointEditView(workspace: workspace, port: entry.port!)
            .environment(\.settingsSearchRequest, SettingsSearchRequest(entry: entry))
            .environment(\.colorScheme, dark ? .dark : .light))
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: 700, height: 480),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        host.sizingOptions = []
        window.contentViewController = host
        window.setContentSize(CGSize(width: 700, height: 480))
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(200))
        let root = try #require(window.contentView)
        let scroll = try #require(views(root).compactMap { $0 as? NSScrollView }.first)
        #expect(scroll.contentView.bounds.minY > 100, "\(anchor) must reveal below the form header")
        #expect((scroll.documentView?.frame.width ?? 0) <= 701)
        if let directory = ProcessInfo.processInfo.environment["PADDOCK_UI_CAPTURE_DIR"] {
          let url = URL(fileURLWithPath: directory)
          try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
          let bitmap = try #require(root.bitmapImageRepForCachingDisplay(in: root.bounds))
          root.cacheDisplay(in: root.bounds, to: bitmap)
          try #require(bitmap.representation(using: .png, properties: [:])).write(
            to: url.appending(path: "settings-\(anchor)-\(dark ? "dark" : "light").png"))
        }
      }
    }
    #expect(await client.writes == 0)
    _ = await workspace.shutdown()
  }

  private func views(_ root: NSView) -> [NSView] { [root] + root.subviews.flatMap(views) }
}

private actor SettingsSearchClient: ManagerLoading {
  var writes = 0
  func snapshot() async throws -> ManagerSnapshot {
    let base = try endpointFixture()
    let settings: [String: Any] = [
      "host": "127.0.0.1", "has_api_key": false, "vision": false, "forensics": false,
      "device": "metal", "max_ctx": 16384, "max_batch": 1, "spec": "on",
      "residency_supported": true, "kv_offload_supported": true,
      "runtime_options": [
        [
          "id": "prefill_chunk", "label": "Prefill chunk size", "group": "Prefill",
          "kind": "integer",
          "minimum": 1, "maximum": 8192, "placeholder": "256", "help": "Prompt processing batch",
          "capability": "",
        ]
      ],
    ]
    let servers = try (12400...12401).map { port in
      try ManagerWire.decode(
        ConfiguredEndpoint.self,
        from: JSONSerialization.data(withJSONObject: [
          "port": port, "model": "qwen", "display": "Same model", "artifact": "mlx",
          "running": false,
          "revision": "r1", "settings": settings,
        ]))
    }
    return ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: base.catalog, runners: [],
      servers: servers)
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    writes += 1
    throw ManagerError.core("Search must not submit model operations")
  }
}
