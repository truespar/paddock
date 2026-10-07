import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Benchmark settings layout", .serialized) @MainActor
struct BenchmarkLayoutTests {
  @Test func nativeSegmentsUpdateBindingsAndRespectDisabledState() async throws {
    let model = BenchmarksModel(client: BenchmarkLayoutFixture())
    try await render(
      PaddockScrollView {
        VStack {
          BenchmarkSegments(
            title: "Prompt", options: [(false, "Short"), (true, "Long")],
            selection: Binding(get: { model.long }, set: { model.long = $0 }))
          BenchmarkSegments(
            title: "Concurrency", options: [(1, "1 request"), (4, "4 requests")],
            selection: Binding(get: { model.concurrency }, set: { model.concurrency = $0 }))
          BenchmarkSegments(
            title: "Disabled", options: [(false, "Short"), (true, "Long")],
            selection: Binding(
              get: { false }, set: { _ in Issue.record("Disabled control changed a value") })
          )
          .disabled(true)
        }.padding(20)
      }, width: 380, dark: false, name: "bindings"
    ) { host in
      let controls = allViews(host).compactMap { $0 as? NSSegmentedControl }
      #expect(controls.count == 3)
      for control in controls {
        #expect(control.segmentDistribution == .fillEqually)
        control.selectedSegment = 1
        #expect(control.sendAction(control.action, to: control.target))
      }
      #expect(model.long && model.concurrency == 4)
      #expect(controls.filter { !$0.isEnabled }.count == 1)
    }
  }

  @Test func controlsShareOneColumnWithoutDuplicatePickerLabelsOrHorizontalOverflow() async throws {
    let fixture = BenchmarkLayoutFixture()
    let model = BenchmarksModel(client: fixture)
    let runner = try Self.runner()
    for dark in [false, true] {
      for width: CGFloat in [380, 580, 620, 960] {
        try await render(
          BenchmarksView(model: model, runners: [runner]), width: width, dark: dark,
          name: "setup"
        ) { host in
          let controls = allViews(host).compactMap { $0 as? NSSegmentedControl }
          #expect(controls.count == 2)
          let frames = controls.map { $0.convert($0.bounds, to: host) }.sorted { $0.minY < $1.minY }
          if frames.count == 2 {
            #expect(abs(frames[0].minX - frames[1].minX) <= 1)
            #expect(abs(frames[0].width - frames[1].width) <= 1)
            #expect(frames.allSatisfy { $0.width >= 220 && $0.height <= 32 })
            #expect(frames[1].minY - frames[0].minY <= (width < 600 ? 78 : 50))
          }
        }
      }
    }
    #expect(await fixture.runs == 0, "Opening/resizing settings must never run inference")
  }

  @Test func emptyBusyAndFailedStatesFitWithoutChangingTheFormGeometry() async throws {
    let fixture = BenchmarkLayoutFixture()
    let model = BenchmarksModel(client: fixture)
    let runner = try Self.runner(inFlight: 2)
    for rows in [[RunnerInfo](), [runner]] {
      try await render(
        BenchmarksView(model: model, runners: rows), width: 380, dark: true,
        name: rows.isEmpty ? "empty" : "busy"
      ) { _ in }
    }
    await fixture.failHistory()
    await model.load()
    #expect(model.error != nil)
    try await render(
      BenchmarksView(model: model, runners: [runner]), width: 380, dark: false,
      name: "error"
    ) { _ in }
    #expect(await fixture.runs == 0)
  }

  @Test func reportCardsAndExpandedDetailsFitInBothThemes() async throws {
    let report = try ManagerWire.decode(BenchmarkReport.self, from: Data(Self.report.utf8))
    for dark in [false, true] {
      for width: CGFloat in [380, 620, 960] {
        try await render(
          PaddockScrollView {
            VStack(spacing: 20) {
              BenchmarkResultCard(report: report, stacked: width < 600) {
                Issue.record("Measuring layout must not export")
              }
              BenchmarkMeasurementDetails(report: report).padding(18)
            }.padding(20).frame(maxWidth: 820).frame(maxWidth: .infinity)
          }.font(.system(size: 13)).buttonStyle(FlatButtonStyle()).background(PaddockStyle.canvas),
          width: width, dark: dark, name: "results"
        ) { _ in }
      }
    }
  }

  @Test func cancellingRemainsAvailableWhenTheRunnerDisappears() async throws {
    let fixture = BenchmarkLayoutFixture()
    let model = BenchmarksModel(client: fixture)
    model.start(try Self.runner())
    #expect(model.running)
    try await render(
      BenchmarksView(model: model, runners: []), width: 380, dark: true,
      name: "running"
    ) { host in
      #expect(allViews(host).contains { $0 is NSProgressIndicator })
    }
    model.cancel()
    for _ in 0..<30 {
      if !model.running { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(!model.running && model.cancelled)
    #expect(await fixture.runs == 1)
  }

  private func render<Content: View>(
    _ content: Content, width: CGFloat, dark: Bool, name: String,
    check: (NSView) throws -> Void
  ) async throws {
    _ = NSApplication.shared
    let host = NSHostingController(
      rootView: content.environment(\.colorScheme, dark ? .dark : .light))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: width, height: 950),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
    window.contentViewController = host
    window.setContentSize(NSSize(width: width, height: 950))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(80))
    host.view.layoutSubtreeIfNeeded()
    #expect(abs(host.view.frame.width - width) < 1)
    let scroll = try #require(allViews(host.view).compactMap { $0 as? NSScrollView }.first)
    #expect((scroll.documentView?.frame.width ?? 0) <= width + 1)
    for control in allViews(host.view).filter({ $0 is NSControl }) {
      guard !control.isHiddenOrHasHiddenAncestor, !control.bounds.isEmpty else { continue }
      let frame = control.convert(control.bounds, to: host.view)
      #expect(frame.minX >= -1 && frame.maxX <= width + 1, "Control overflow: \(frame)")
    }
    try check(host.view)
    if let folder = ProcessInfo.processInfo.environment["PADDOCK_BENCHMARK_SNAPSHOTS"],
      let bitmap = host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds)
    {
      host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
      try bitmap.representation(using: .png, properties: [:])?.write(
        to: URL(fileURLWithPath: folder).appending(
          path: "\(name)-\(Int(width))-\(dark ? "dark" : "light").png"))
    }
  }
  private func allViews(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(allViews) }
  private static func runner(inFlight: Int = 0) throws -> RunnerInfo {
    try ManagerWire.decode(
      RunnerInfo.self,
      from: JSONSerialization.data(withJSONObject: [
        "port": 11543, "pid": 42, "status": "ok", "model": "diffusiongemma",
        "display": "DiffusionGemma 26B A4B — a long instance name that must not expand the page",
        "endpoint": "http://127.0.0.1:11543", "in_flight": inFlight,
      ]))
  }
  private static let report = #"""
    {"id":"layout-fixture","model":"DiffusionGemma 26B A4B — synthetic layout fixture, not a measured result",
    "created_at_ms":1790330400000,"concurrency":4,"prompt_words":2048,"trials":3,"warmups":1,
    "output_limit":128,"aggregate_output_tok_s":1234.56,"wall_seconds":123.45,
    "ttft_median_ms":12345,"stream_event_gap_p99_ms":null,"output_tokens":1536,
    "max_ctx":32768,"max_batch":4,"cache_policy":"Synthetic layout fixture. Unique first-content prompts; existing cache remains unchanged.",
    "runner_version":"0.1.9","samples":[{"input_tokens":2500,"output_tokens":128,"cached_tokens":8,
    "ttft_ms":12000,"duration_ms":25000,"finish_reason":"length"}]}
    """#
}

private actor BenchmarkLayoutFixture: ManagerLoading {
  private(set) var runs = 0
  private var failed = false
  func failHistory() { failed = true }
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.closed }
  func maintenance(_ command: MaintenanceCommand) async throws -> MaintenanceReply {
    switch command {
    case .benchmarkHistory:
      if failed {
        throw ManagerError.core(
          "Synthetic history failure with a longer message that must wrap inside the card.")
      }
      return try reply("history", state: "complete", payload: #"{"reports":[]}"#)
    case .benchmark:
      runs += 1
      return try reply("run", state: "running")
    case .poll: return try reply("run", state: "running")
    case .close: return try reply("closed", state: "complete")
    default:
      Issue.record("Unexpected management operation during layout")
      throw ManagerError.closed
    }
  }
  private func reply(_ id: String, state: String, payload: String = "{}") throws -> MaintenanceReply
  {
    try ManagerWire.decode(
      MaintenanceReply.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": id, "state": state, "payload": payload,
      ]))
  }
}
