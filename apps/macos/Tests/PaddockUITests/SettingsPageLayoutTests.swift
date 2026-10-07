import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Consistent settings pages", .serialized) @MainActor
struct SettingsPageLayoutTests {
  @Test func singleColumnPagesShareTheSurfaceWhileCatalogsRemainSplitViews() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appending(path: "Sources/PaddockUI")
    for name in [
      "BenchmarksView", "InsightsView", "ExternalClientsView", "DesktopSettingsView",
      "StudioPreferencesView", "DataStorageView",
    ] {
      let source = try String(contentsOf: root.appending(path: "\(name).swift"), encoding: .utf8)
      #expect(source.contains("SettingsPage(title:"), "\(name) must share the page measure")
      #expect(!source.contains("PaddockScrollView {"), "\(name) must not add a second scroll owner")
    }
    for name in ["ModelLibraryView", "CloudModelsView"] {
      let source = try String(contentsOf: root.appending(path: "\(name).swift"), encoding: .utf8)
      #expect(!source.contains("SettingsPage(title:"))
    }
  }

  @Test func pageStaysCenteredWhenItsScrollbarAppears() async throws {
    for width: CGFloat in [380, 620, 960, 1440] {
      var measured: [CGRect] = []
      for count in [1, 35] {
        try await render(
          SettingsPage(title: "Settings") { _ in
            ForEach(0..<count, id: \.self) { _ in PageLayoutMarker().frame(height: 30) }
          }, width: width, dark: false, name: "measure-\(count)"
        ) { host in
          let marker = try #require(views(host).first { $0 is PageMarkerView })
          measured.append(marker.convert(marker.bounds, to: host))
        }
      }
      let short = try #require(measured.first)
      let long = try #require(measured.last)
      #expect(abs(short.midX - width / 2) <= 1)
      #expect(abs(long.midX - width / 2) <= 1)
      #expect(short.width <= 756 && long.width <= 756)
      #expect(abs(short.minX - long.minX) <= 1)
      #expect(abs(short.width - long.width) <= 1)
    }
  }

  @Test func populatedPagesFitLightDarkNarrowAndWideWithoutSideEffects() async throws {
    let client = SettingsLayoutClient()
    let runner = try Self.runner()
    let endpoint = try Self.endpoint()
    for dark in [false, true] {
      for width: CGFloat in [380, 620, 960, 1440] {
        let benchmarks = BenchmarksModel(client: client)
        try await render(
          BenchmarksView(model: benchmarks, runners: [runner]), width: width, dark: dark,
          name: "benchmarks"
        ) { _ in }
        try await render(
          ExternalClientsView(client: client, runners: [runner]), width: width, dark: dark,
          name: "clients"
        ) { _ in }
        for page in InsightPage.allCases {
          let insights = InsightsModel(client: client)
          insights.live = false
          insights.page = page
          insights.port = endpoint.port
          try await render(
            InsightsView(model: insights, endpoints: [endpoint]), width: width, dark: dark,
            name: "insights-\(page.id)"
          ) { _ in }
          #expect(insights.sampledAt != nil && insights.error == nil)
        }
      }
    }
    #expect(await client.mutations == 0)
  }

  @Test func allClientFormatsAndCredentialReceiptsStayBounded() async throws {
    let setup = try Self.setup()
    for format in ExternalClient.allCases {
      for width: CGFloat in [380, 620, 960] {
        try await render(
          SettingsPage(title: "Client setup") { stacked in
            ClientConfigurationSection(
              setup: setup, format: .constant(format), stacked: stacked, exporting: false,
              exportedPath: "/Volumes/Models/Local models and credentials/paddock-11543.sh",
              copy: { _ in Issue.record("Layout must not change the clipboard") },
              export: { Issue.record("Layout must not export credentials") })
          }, width: width, dark: true, name: "configuration-\(format.id)"
        ) { _ in }
      }
    }
  }

  @Test func emptyAndFailureStatesKeepTheSamePageBounds() async throws {
    for failure in [false, true] {
      let client = SettingsLayoutClient(failure: failure)
      let insights = InsightsModel(client: client)
      insights.live = false
      try await render(
        InsightsView(model: insights, endpoints: []), width: 380, dark: false,
        name: failure ? "insights-error" : "insights-empty"
      ) { _ in }
      try await render(
        ExternalClientsView(client: client, runners: failure ? [try Self.runner()] : []),
        width: 380, dark: false, name: failure ? "clients-error" : "clients-empty"
      ) { _ in }
      #expect(await client.mutations == 0)
    }
  }

  private func render<Content: View>(
    _ content: Content, width: CGFloat, dark: Bool, name: String,
    check: (NSView) throws -> Void
  ) async throws {
    _ = NSApplication.shared
    let controller = NSHostingController(
      rootView: content.environment(\.colorScheme, dark ? .dark : .light))
    controller.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: width, height: 780),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
    window.contentViewController = controller
    window.setContentSize(NSSize(width: width, height: 780))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(100))
    let host = controller.view
    host.layoutSubtreeIfNeeded()
    #expect(abs(host.frame.width - width) < 1)
    let scrolls = views(host).compactMap { $0 as? NSScrollView }
    #expect(scrolls.count == 1, "\(name) must have one vertical scroll owner")
    let scroll = try #require(scrolls.first)
    #expect((scroll.documentView?.frame.width ?? 0) <= scroll.contentSize.width + 1)
    try check(host)
    for control in views(host).filter({ $0 is NSControl }) {
      // AppKit parks an autohidden scroller at x=-17 with zero height.
      // That is not a rendered control escaping the form.
      guard !control.isHiddenOrHasHiddenAncestor, !control.bounds.isEmpty else { continue }
      let frame = control.convert(control.bounds, to: host)
      #expect(frame.minX >= -1 && frame.maxX <= width + 1, "\(name): control overflow \(frame)")
    }
    if let folder = ProcessInfo.processInfo.environment["PADDOCK_SETTINGS_LAYOUT_SNAPSHOTS"] {
      try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
      if let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
        host.cacheDisplay(in: host.bounds, to: bitmap)
        try bitmap.representation(using: .png, properties: [:])?.write(
          to: URL(fileURLWithPath: folder).appending(
            path: "\(name)-\(Int(width))-\(dark ? "dark" : "light").png"))
      }
    }
    // Only this offscreen test scroll view is touched; never global mouse/keys.
    if let document = scroll.documentView {
      scroll.contentView.scroll(
        to: NSPoint(x: 0, y: max(0, document.frame.height - scroll.contentSize.height)))
      scroll.reflectScrolledClipView(scroll.contentView)
      host.layoutSubtreeIfNeeded()
      #expect(document.frame.width <= scroll.contentSize.width + 1)
    }
  }

  private func views(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(views) }
  private static let modelName = "Qwen3.8-Flash-Next-MLX-4bit-long-instance-name-for-layout"
  private static func runner() throws -> RunnerInfo {
    try ManagerWire.decode(
      RunnerInfo.self,
      from: Data(
        """
        {"port":11543,"pid":42,"status":"ok","model":"\(modelName)","endpoint":"http://127.0.0.1:11543","in_flight":0}
        """.utf8))
  }
  private static func endpoint() throws -> ConfiguredEndpoint {
    try ManagerWire.decode(
      ConfiguredEndpoint.self,
      from: Data(
        """
        {"port":11543,"model":"\(modelName)","artifact":"mlx-4bit","running":true,"revision":"fixture",
        "settings":{"host":"127.0.0.1","max_ctx":32768,"max_batch":1,"has_api_key":true,"vision":true,"forensics":false,"device":"metal"}}
        """.utf8))
  }
  private static func setup() throws -> LocalClientSetup {
    try ManagerWire.decode(
      LocalClientSetup.self,
      from: Data(
        """
        {"base_url":"http://127.0.0.1:11543/v1","model":"\(modelName)","has_key":true}
        """.utf8))
  }
}

private struct PageLayoutMarker: NSViewRepresentable {
  func makeNSView(context: Context) -> PageMarkerView { PageMarkerView() }
  func updateNSView(_ view: PageMarkerView, context: Context) {}
}
private final class PageMarkerView: NSView {}

private actor SettingsLayoutClient: ManagerLoading {
  let failure: Bool
  private(set) var mutations = 0
  init(failure: Bool = false) { self.failure = failure }
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.closed }
  func maintenance(_ command: MaintenanceCommand) async throws -> MaintenanceReply {
    if failure {
      throw ManagerError.core("The endpoint could not be reached. Check its instance settings.")
    }
    let payload: String
    switch command {
    case .clientInfo:
      payload =
        #"{"base_url":"http://127.0.0.1:11543/v1","model":"Qwen3.8-Flash-Next-MLX-4bit-long-instance-name-for-layout","has_key":true}"#
    case .benchmarkHistory: payload = #"{"reports":[]}"#
    case .usage(_, _, let port):
      payload =
        port == nil
        ? #"{"grain_ms":60000,"now_ms":1791100800000,"buckets":[],"gaps":[],"generations":[],"web":[]}"#
        : #"{"grain_ms":60000,"now_ms":1791100800000,"buckets":[{"t":1791100800000,"port":11543,"requests":1000,"errors_4xx":0,"errors_5xx":0,"disconnects":0,"input_tokens":123456,"output_tokens":12345,"cached_tokens":8000,"duration_ms_sum":10000,"spec_drafted":0,"spec_accepted":0}],"gaps":[{"id":1,"port":11543,"from_ts_ms":1791100000000,"to_ts_ms":1791100800000,"cause":"Asleep"}],"generations":[],"web":[{"provider":"exa","requests":10,"credits":10,"microdollars":1000}]}"#
    case .activity:
      payload =
        #"{"events":[{"port":11543,"ts_ms":1791100800000,"seq":1,"gen_ai.response.model":"Qwen3.8-Flash-Next-MLX-4bit-long-instance-name-for-layout","status":200,"gen_ai.usage.output_tokens":128}]}"#
    case .cache:
      payload =
        #"{"servers":[{"port":11543,"model":"Qwen3.8-Flash-Next-MLX-4bit-long-instance-name-for-layout","tier":{"lookups":100,"hits":75,"ram_ready":536870912,"ram_capacity":1073741824,"disk_ready":1073741824,"disk_capacity":8589934592}}]}"#
    case .close: payload = "{}"
    default:
      mutations += 1
      throw ManagerError.closed
    }
    return try ManagerWire.decode(
      MaintenanceReply.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": "layout", "state": "complete", "payload": payload,
      ]))
  }
}
