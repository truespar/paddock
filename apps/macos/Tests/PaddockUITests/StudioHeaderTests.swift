import AppKit
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Studio header placement", .serialized) @MainActor
struct StudioHeaderTests {
  @Test func chromeHasSettingsInsteadOfAWorkspaceSwitch() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let toolbar = try String(
      contentsOf: root.appending(path: "Sources/PaddockUI/WorkspaceShell.swift"), encoding: .utf8)
    #expect(!toolbar.contains("WorkspaceModeMenu") && !toolbar.contains("workspace-switcher"))
    #expect(toolbar.contains("navigation.showSettings()") && toolbar.contains("Back to chat"))
    let app = try String(
      contentsOf: root.appending(path: "Sources/PaddockMac/PaddockMacApp.swift"), encoding: .utf8)
    #expect(!app.contains("Open Manager") && !app.contains("Open Studio"))
    #expect(app.contains("CommandGroup(replacing: .appSettings)"))
    #expect(app.contains("lifecycle.open(.settings)"))
  }

  @Test func menuBarWorkspaceSwitchesPreserveDestinationsAndDrafts() {
    let model = WorkspaceModel(client: HeaderNavigationLoader())
    model.draft.message = "Keep this unsent question"
    model.navigation.studio = .prompts
    model.navigation.manager = .downloads
    model.navigation.sidebarVisible = true
    model.selectWorkspace(.manager)
    #expect(model.navigation.mode == .manager && model.navigation.manager == .downloads)
    model.navigation.sidebarVisible = false
    model.selectWorkspace(.studio)
    #expect(model.navigation.mode == .studio && model.navigation.studio == .prompts)
    #expect(model.navigation.sidebarVisible)
    #expect(model.draft.message == "Keep this unsent question")
    model.selectWorkspace(.manager)
    #expect(!model.navigation.sidebarVisible)
    model.desktopTransition = true
    model.selectWorkspace(.studio)
    #expect(model.navigation.mode == .manager)
  }

  @Test func decodesTheSharedWebFixture() throws {
    let header = try decode(fixture())
    #expect(header.current?.label == "Qwen 3.8 27B")
    #expect(header.current?.hint == "12481")
    #expect(header.current?.vendor == "Alibaba")
    #expect(header.isVision && header.specLabel == "MTP" && !header.comparing)
    #expect(header.pickerOptions[1].hint == "OpenRouter")
  }

  @Test func missingTargetIsNotSilentlyReplacedByAnotherModel() throws {
    var value = try fixture()
    value["currentModel"] = "stopped"
    #expect(try decode(value).current == nil)
    var options = try #require(value["pickerOptions"] as? [[String: Any]])
    options.insert(
      [
        "value": "stopped", "label": "Stopped model", "hint": "not running",
        "vendor": "IBM", "title": "stopped - not running", "available": false,
      ], at: 0)
    value["pickerOptions"] = options
    let stopped = try decode(value)
    #expect(stopped.current?.label == "Stopped model")
    #expect(stopped.current?.available == false)
  }

  @Test func singleCompareAndStoppedHeaderFitWithoutMakingSelections() throws {
    _ = NSApplication.shared
    let base = try fixture()
    var long = base
    var options = try #require(base["pickerOptions"] as? [[String: Any]])
    options[0]["label"] = String(repeating: "Long model name ", count: 8)
    options[0]["available"] = false
    long["pickerOptions"] = options
    var compare = base
    compare["comparing"] = true
    compare["compareLanes"] = (0..<4).map { index in
      [
        "id": "lane-\(index)", "label": "Qwen 3.8 27B lane \(index)", "vendor": "Alibaba",
        "spec": "MTP",
      ]
    }
    var selections: [String] = []
    for (name, value) in [("single", base), ("stopped-long", long), ("compare", compare)] {
      for dark in [false, true] {
        for width in [200, 280] {
          let header = try decode(value)
          let host = NSHostingController(
            rootView: StudioHeaderModelContent(header: header, enabled: name != "stopped-long") {
              selections.append($0)
            }.environment(\.colorScheme, dark ? .dark : .light)
          )
          let size = CGSize(width: width, height: 34)
          host.view.frame = NSRect(origin: .zero, size: size)
          host.view.layoutSubtreeIfNeeded()
          let fit = host.sizeThatFits(in: size)
          #expect(fit.width <= size.width, "\(name) \(dark) \(width): \(fit)")
          #expect(fit.height <= size.height, "\(name) \(dark) \(width): \(fit)")
        }
      }
    }
    #expect(selections.isEmpty, "Rendering a target cannot retarget a conversation")
  }

  private func fixture() throws -> [String: Any] {
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    return try #require(
      JSONSerialization.jsonObject(
        with: Data(
          contentsOf: root.appending(path: "studio/src/lib/studio-model-header.fixture.json")))
        as? [String: Any])
  }

  private func decode(_ value: [String: Any]) throws -> StudioModelHeader {
    try JSONDecoder().decode(
      StudioModelHeader.self, from: JSONSerialization.data(withJSONObject: value))
  }
}

private struct HeaderNavigationLoader: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    Issue.record("Workspace switching must not start or refresh a backend")
    throw ManagerError.core("Unexpected backend access")
  }
}
