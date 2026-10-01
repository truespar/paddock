import AppKit
import Foundation
import PaddockClient
import PaddockConversationCore
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native Tables workspace", .serialized) @MainActor
struct NativeTablesTests {
  private let csv = "size,city,label\n1,a,yes\n2,b,no\n3,c,\n"
  private func json(_ text: String) throws -> ConversationValue {
    try JSONDecoder().decode(ConversationValue.self, from: Data(text.utf8))
  }
  private func model() -> NativeTablesModel {
    let m = NativeTablesModel(client: NativeManager())
    m.api = { path, _, _, _ in
      if path == "api/runners" {
        return try json(
          #"[{"port":1234,"pid":10,"status":"ok","tabular":"kumo-tabular-small-classification","display":"Kumo Tabular","vendor":"NVIDIA"},{"port":1235,"pid":11,"status":"ok","model":"chat"}]"#
        )
      }
      if path.hasSuffix("/server") {
        return try json(
          #"{"tabular":{"capabilities":{"task":"classification","max_estimators":16,"default_estimators":8}}}"#
        )
      }
      return try json(
        #"{"task":"classification","classes":["yes","no"],"num_estimators":8,"predictions":[{"class":0,"label":"yes","probabilities":[0.75,0.25]}],"usage":{"gpu_ms":3,"elapsed_ms":5}}"#
      )
    }
    return m
  }
  @Test func discoversOnlyTableModelsAndKeepsVariantNames() async throws {
    let m = model()
    await m.refresh()
    #expect(m.predictors.count == 1 && m.port == 1234)
    #expect(m.current?.title == "Kumo Tabular · small classification")
    #expect(m.current?.vendor == "NVIDIA" && m.current?.limits.task == .classification)
    let runner = try JSONDecoder().decode(
      RunnerInfo.self,
      from: Data(
        #"{"port":1234,"pid":10,"status":"ok","tabular":"kumo-small-classification","endpoint":"http://127.0.0.1:1234"}"#
          .utf8))
    #expect(
      runner.title == "kumo-small-classification" && runner.studioActionTitle == "Open Tables")
    #expect(runner.studioActionSymbol == "tablecells")
    var nav = WorkspaceNavigation()
    nav.studio = .tables
    nav.returnToChat()
    #expect(nav.studio == .chats)
  }
  @Test func predictsViaRelayAndExportsTheImmutableInput() async throws {
    let m = model()
    await m.refresh()
    m.setSource(csv)
    await m.settle()
    #expect(m.canRun && m.plan?.queryRows == [2])
    #expect(m.columns[2].missing == 1 && m.columns[1].sample == "a, b, c")
    let previous = m.api
    m.api = { path, method, body, query in
      if path.hasSuffix("/predictions") {
        #expect(path == "api/runners/1234/v1/tabular/predictions" && method == "POST")
        #expect(body?["preprocessing"] == .string("sdm_v1"))
        #expect(body?["query"] == .array([.array([.number(3), .string("c")])]))
        #expect(body?["model"] == .string("kumo-tabular-small-classification"))
      }
      return try await previous(path, method, body, query)
    }
    m.run()
    await m.settle()
    #expect(m.error == nil && !m.busy && !m.stale)
    let original = try #require(m.result?.csv)
    #expect(original == "size,city,label,confidence\n1,a,yes,\n2,b,no,\n3,c,yes,0.7500\n")
    #expect(m.result?.header.suffix(3) == ["Confidence", "P(yes)", "P(no)"])
    await m.refresh()
    #expect(!m.stale)  // status polls don't dirty the input or reset results
    m.setSource(csv.replacingOccurrences(of: "3,c,", with: "100,c,"))
    await m.settle()
    #expect(m.stale && m.result?.csv == original)
    m.spec?.use[0] = false
    await m.settle()
    #expect(m.plan?.features == [1])
  }
  @Test func cancelledReplyCannotRepopulateResults() async throws {
    let m = model()
    await m.refresh()
    m.setSource(csv)
    await m.settle()
    let previous = m.api
    var release: CheckedContinuation<Void, Never>?
    m.api = { path, method, body, query in
      if path.hasSuffix("/predictions") { await withCheckedContinuation { release = $0 } }
      return try await previous(path, method, body, query)
    }
    m.run()
    for _ in 0..<100 where release == nil { await Task.yield() }
    #expect(release != nil && m.busy)
    m.cancel()
    m.setSource("")
    release?.resume()
    await m.settle()
    #expect(m.result == nil && m.error == nil && !m.busy && !m.canRun)
  }
  @Test func latestParseWinsAndSameHeaderKeepsColumnChoices() async throws {
    let m = model()
    await m.refresh()
    m.setSource(csv)
    m.setSource("amount,kind,result\n10,a,yes\n20,b,no\n30,c,\n")
    await m.settle()
    #expect(m.table?.header.first == "amount")
    m.spec?.use[1] = false
    await m.settle()
    m.setSource("amount,kind,result\n10,a,yes\n20,b,no\n40,c,\n")
    await m.settle()
    #expect(m.spec?.use == [true, false, true] && m.plan?.features == [0])
    m.seed = -1
    await m.settle()
    #expect(!m.canRun && m.validation != nil)
    m.seed = 0
    m.estimators = 17
    await m.settle()
    #expect(!m.canRun)
    m.setSource("a,b\n1,\"broken")
    await m.settle()
    #expect(m.table == nil && !m.canRun && m.validation != nil)
  }
  @Test func catalogRoutesTableWeightsWithoutChatOrMLXSettings() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let modelWorkload = try String(
      contentsOf: root.appending(path: "Sources/PaddockUI/EndpointModelWorkload.swift"),
      encoding: .utf8)
    #expect(modelWorkload.contains("!editor.capabilities.contains(\"tabular\")"))
    let catalog: CatalogModel = try JSONDecoder().decode(
      CatalogModel.self,
      from: Data(
        #"{"id":"kumo","display":"Kumo","family":"kumo","installed":true,"totalSize":100,"capability":["tabular"],"artifacts":[{"id":"small","kind":"weights","format":"safetensors","label":"Small","installed":true,"totalSize":100,"runtime":{"backends":["metal"],"capability":["tabular"]}}]}"#
          .utf8))
    #expect(ModelStartPurpose.tables.weights(catalog, backend: "metal").count == 1)
    #expect(ModelStartPurpose.speech.weights(catalog, backend: "metal").isEmpty)
  }
  @Test func layoutsFitBothThemesWithoutShowingWindows() async throws {
    let m = model()
    await m.refresh()
    m.setSource(csv)
    await m.settle()
    m.run()
    await m.settle()
    for width in [620.0, 1100.0] {
      for dark in [false, true] {
        let host = NSHostingController(
          rootView: NativeTablesView(model: m, onStart: {})
            .environment(\.colorScheme, dark ? .dark : .light))
        host.sizingOptions = []
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: width, height: 1100),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentViewController = host
        window.setContentSize(NSSize(width: width, height: 1100))
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(150))
        host.view.layoutSubtreeIfNeeded()
        #expect(abs(host.view.frame.width - width) < 1)
        if let folder = ProcessInfo.processInfo.environment["PADDOCK_TABLES_SNAPSHOTS"],
          let bitmap = host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds)
        {
          host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
          try bitmap.representation(using: .png, properties: [:])?.write(
            to: URL(fileURLWithPath: folder).appending(
              path: "tables-\(Int(width))-\(dark ? "dark" : "light").png"))
        }
      }
    }
  }
}
