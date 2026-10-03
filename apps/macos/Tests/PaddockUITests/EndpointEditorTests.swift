import AppKit
import Foundation
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Acknowledged native endpoint editing", .timeLimit(.minutes(1))) @MainActor
struct EndpointEditorTests {
  @Test func quantizedClefShowsItsArtifactPrecisionWithoutChatOrVisionControls() throws {
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"clef","display":"Clef","capability":["decision","vision"],"installed":true,"total_size":30000000000,"artifacts":[{"id":"q8","kind":"weights","format":"gguf","label":"GGUF Q8_0","quant":"Q8_0","installed":true,"total_size":28732215264,"backend_supported":true,"runtime":{"checkpoint_dir":false,"kv_cache_dtype":"auto","capability":["decision"],"companions":[]}},{"id":"mlx8","kind":"weights","format":"safetensors","label":"MLX 8-bit","quant":"MLX-AFFINE-8-G64","installed":true,"total_size":29800000000,"backend_supported":true,"runtime":{"checkpoint_dir":true,"embedded_vision":false,"kv_cache_dtype":"auto","capability":["decision"],"companions":[]}}]}]}"#
          .utf8))
    for (id, label) in [("q8", "GGUF · Q8_0"), ("mlx8", "MLX · 8-bit")] {
      let editor = EndpointEditor(
        client: EndpointEditFixture(), endpoint: try editEndpoint(), pid: nil)
      editor.catalog = catalog.models
      editor.modelID = "clef"
      editor.artifactID = id
      editor.selectArtifact(try #require(editor.selectedArtifact), newModel: true)
      #expect(editor.checkpointLabel == label)
      #expect(editor.capabilities == ["decision"] && editor.kvDtype == "auto")
      #expect(!editor.canSpeculate && !editor.canTools && !editor.embeddedVision)
    }
  }
  @Test func layaUsesOfficialF16CheckpointAndNoChatFeatures() throws {
    let editor = EndpointEditor(
      client: EndpointEditFixture(), endpoint: try editEndpoint(), pid: nil)
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"laya","display":"Laya","capability":["decision"],"installed":true,"total_size":2370000000,"artifacts":[{"id":"f16","kind":"weights","format":"safetensors","label":"Official F16 checkpoints (all three)","installed":true,"total_size":2370000000,"backend_supported":true,"runtime":{"checkpoint_dir":true,"kv_cache_dtype":"auto","default_max_batch":1}}]}]}"#
          .utf8))
    editor.catalog = catalog.models
    editor.modelID = "laya"
    editor.artifactID = "f16"
    editor.selectArtifact(try #require(editor.selectedArtifact), newModel: true)
    #expect(editor.checkpointLabel == "F16 checkpoint" && !editor.isMLX)
    #expect(editor.kvDtype == "auto" && editor.capabilities == ["decision"])
    #expect(!editor.canSpeculate && !editor.canTools && !editor.embeddedVision)
  }
  @Test func residencyEditsAreValidatedAndDoNotChangeServingGeometry() throws {
    let data = try JSONSerialization.data(withJSONObject: [
      "port": 12345, "model": "kb-whisper-large", "running": true, "revision": "before",
      "runtime_state": [
        "pid": 42, "restart_required": false, "residency_live": true,
        "changed": [], "max_ctx": 448, "max_batch": 1,
      ],
      "settings": [
        "host": "127.0.0.1", "max_ctx": 448, "max_batch": 1,
        "has_api_key": true, "vision": false, "forensics": false, "device": "metal",
        "residency_supported": true,
      ],
    ])
    let endpoint = try ManagerWire.decode(ConfiguredEndpoint.self, from: data)
    let editor = EndpointEditor(client: EndpointEditFixture(), endpoint: endpoint, pid: 42)
    #expect(!editor.dirty && editor.residencySupported)
    editor.loadOnDemand = true
    editor.unloadIdleSeconds = "0"
    #expect(editor.onlyResidencyChanges && editor.validation == nil)
    #expect(editor.runtimeState?.residencyLive == true)
    editor.unloadIdleSeconds = "-1"
    #expect(editor.validation != nil)
    editor.reset()
    #expect(!editor.dirty && editor.unloadIdleSeconds.isEmpty)
  }
  @Test func pendingSavedSettingsCanRestartWithoutInventingAnEdit() async throws {
    let client = EndpointEditFixture()
    let data = try JSONSerialization.data(withJSONObject: [
      "port": 12345, "model": "qwen3.8-27b", "running": true, "revision": "before",
      "runtime_state": [
        "pid": 42, "restart_required": true, "changed": ["Context"], "max_ctx": 4096,
        "max_batch": 1,
      ],
      "settings": [
        "host": "127.0.0.1", "max_ctx": 8192, "max_batch": 1, "has_api_key": true, "vision": false,
        "forensics": false, "device": "metal",
      ],
    ])
    let endpoint = try ManagerWire.decode(ConfiguredEndpoint.self, from: data)
    let editor = EndpointEditor(client: client, endpoint: endpoint, pid: 42)
    #expect(editor.restartRequired && !editor.dirty && editor.changes.isEmpty)
    editor.save(.restart)
    await editor.settle()
    #expect(await client.mutations == 1)
    let replaced = EndpointEditor(client: client, endpoint: endpoint, pid: 99)
    #expect(
      !replaced.restartRequired, "A stale process report cannot describe a replacement runner")
  }
  @Test func webWorkloadAndSpeculationChoicesAreUsedWithoutDirtyingAliases() throws {
    #expect(EndpointEditor.workloads.map(\.batch) == [1, 4, 16])
    #expect(EndpointEditor.speculationChoices.map(\.0) == ["on", "off", "adaptive"])
    #expect(EndpointEditor.specPolicy("auto") == "adaptive")
    #expect(EndpointEditor.specPolicy("ladder") == "on")
    for value in ["false", "no", "none", "0", "off"] {
      #expect(EndpointEditor.specPolicy(value) == "off")
    }
    let editor = EndpointEditor(
      client: EndpointEditFixture(), endpoint: try editEndpoint(), pid: nil)
    #expect(!editor.dirty)
    editor.memoryLimit = "nan"
    #expect(editor.dirty && editor.validation != nil)
    #expect(editor.memoryBudgetMiB == nil)
    editor.memoryLimit = "8"
    #expect(editor.memoryBudgetMiB == 8192)
    editor.generateKey()
    #expect(editor.replacementKey.hasPrefix("pd-") && editor.replacementKey.count == 35)
  }
  @Test func unchangedSettingsSendNothingAndNullMeansDefault() throws {
    let editor = EndpointEditor(
      client: EndpointEditFixture(), endpoint: try editEndpoint(), pid: 42)
    #expect(!editor.dirty && editor.validation == nil)
    editor.context = ""
    #expect(editor.dirty)
    let data = try JSONEncoder().encode(editor.changes)
    let values = try #require(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    #expect(values.count == 1 && values[0]["field"] as? String == "max_ctx")
    #expect(values[0]["value"] is NSNull)
    #expect(!String(decoding: data, as: UTF8.self).contains("api_key"))
  }

  @Test func validationDoesNotTurnInvalidInputIntoDefault() throws {
    let editor = EndpointEditor(
      client: EndpointEditFixture(), endpoint: try editEndpoint(), pid: nil)
    editor.context = "abc"
    #expect(editor.validation != nil)
    editor.context = "8192"
    editor.replacementKey = "short"
    #expect(editor.validation != nil)
    editor.replacementKey = "synthetic-replacement-key"
    #expect(editor.validation == nil)
    editor.host = "0.0.0.0"
    editor.save(.defer)
    #expect(editor.error?.contains("Confirm network") == true)
    #expect(!editor.saving)
  }

  @Test func successIsAcknowledgedBeforeDraftAndSecretAreCleared() async throws {
    let client = EndpointEditFixture()
    let editor = EndpointEditor(client: client, endpoint: try editEndpoint(), pid: 42)
    editor.context = "8192"
    editor.replacementKey = "synthetic-replacement-key"
    editor.save(.defer)
    #expect(editor.saving && editor.dirty)
    await editor.settle()
    #expect(await client.mutations == 1)
    #expect(!editor.saving && !editor.dirty)
    #expect(editor.context == "8192" && editor.replacementKey.isEmpty)
    #expect(editor.message == "Saved for next start")
  }

  @Test func failedOperationPreservesDraftAndReviewedRevision() async throws {
    let client = EndpointEditFixture()
    await client.configure(failure: true)
    let editor = EndpointEditor(client: client, endpoint: try editEndpoint(), pid: 42)
    editor.context = "8192"
    editor.replacementKey = "synthetic-replacement-key"
    editor.save(.restart)
    await editor.settle()
    #expect(editor.error == "Stale configuration")
    #expect(editor.dirty && editor.replacementKey == "synthetic-replacement-key")
    #expect(editor.endpoint.revision == "before" && !editor.saving)
  }

  @Test func lostPollKeepsReceiptAndRetryDoesNotRepeatMutation() async throws {
    let client = EndpointEditFixture()
    await client.configure(pollFailure: true)
    let editor = EndpointEditor(client: client, endpoint: try editEndpoint(), pid: 42)
    editor.context = "8192"
    editor.save(.defer)
    await editor.settle()
    #expect(editor.pending?.id == 7 && editor.saving)
    editor.save(.defer)
    #expect(await client.mutations == 1)
    await client.configure(pollFailure: false)
    editor.retryStatus()
    await editor.settle()
    #expect(await client.mutations == 1)
    #expect(!editor.saving && !editor.dirty && editor.error == nil)
  }

  @Test func stoppedEndpointCannotRestartAndRemovalNeedsCleanDraft() async throws {
    let client = EndpointEditFixture()
    let editor = EndpointEditor(client: client, endpoint: try editEndpoint(), pid: nil)
    editor.context = "8192"
    editor.save(.restart)
    #expect(!editor.saving && editor.error != nil)
    editor.remove()
    #expect(await client.mutations == 0)
    editor.reset()
    editor.remove()
    await editor.settle()
    #expect(editor.removed && !editor.saving)
  }

  @Test func navigatingToAnotherEndpointNeverDropsAnUnsentDraft() async throws {
    let client = EndpointEditFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let snapshot = try #require(workspace.snapshot)
    let row = try #require(EndpointRow.rows(snapshot: snapshot, latestJob: nil).first)
    workspace.openEndpoint(row)
    workspace.endpointEditor?.context = "32768"
    let second = EndpointRow(port: 13494, runner: nil, configured: nil, job: nil)
    workspace.openEndpoint(second)
    #expect(workspace.detailEndpointPort == row.port)
    #expect(workspace.endpointEditor?.context == "32768")
    #expect(workspace.studioNeedsQuitConfirmation && workspace.desktopError != nil)
  }

  @Test func settingsRenderOffscreenInBothThemes() throws {
    for dark in [false, true] {
      for width: CGFloat in [640, 820] {
        let editor = EndpointEditor(
          client: EndpointEditFixture(), endpoint: try editEndpoint(), pid: 42)
        let view = NSHostingView(
          rootView: EndpointSettingsView(editor: editor, canMutate: true, onTools: {})
            .font(.system(size: 12)).padding(24).frame(width: width)
            .background(PaddockStyle.canvas).environment(\.colorScheme, dark ? .dark : .light))
        view.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        let size = view.fittingSize
        #expect(size.width == width && size.height > 400 && size.height < 2200)
        view.frame = NSRect(origin: .zero, size: size)
        view.layoutSubtreeIfNeeded()
        if let path = ProcessInfo.processInfo.environment["PADDOCK_ENDPOINT_SNAPSHOTS"],
          let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds)
        {
          view.cacheDisplay(in: view.bounds, to: bitmap)
          let directory = URL(fileURLWithPath: path)
          try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
          try bitmap.representation(using: .png, properties: [:])?.write(
            to: directory.appending(path: "endpoint-\(dark ? "dark" : "light")-\(Int(width)).png"))
        }
      }
    }
  }

  @Test func detailAndEditAreSeparateWithoutLosingTheDraftOrMutatingOnNavigation() async throws {
    let client = EndpointEditFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let snapshot = try #require(workspace.snapshot)
    let row = try #require(EndpointRow.rows(snapshot: snapshot, latestJob: nil).first)
    workspace.openEndpoint(row)
    #expect(!workspace.editingEndpoint)
    workspace.editEndpoint()
    #expect(workspace.editingEndpoint)
    workspace.endpointEditor?.context = "32768"
    workspace.editingEndpoint = false
    workspace.navigation.mode = .studio
    workspace.navigation.mode = .manager
    workspace.editEndpoint()
    #expect(workspace.editingEndpoint && workspace.endpointEditor?.context == "32768")
    #expect(workspace.studioNeedsQuitConfirmation)
    #expect(await client.mutations == 0)
  }

  @Test func liveAndConfigurationCardsFitNarrowAndWideOffscreen() throws {
    let row = EndpointRow(port: 12345, runner: nil, configured: try editEndpoint(), job: nil)
    for dark in [false, true] {
      for width: CGFloat in [584, 764] {
        let host = NSHostingView(
          rootView: EndpointSummaryView(row: row)
            .frame(width: width).environment(\.colorScheme, dark ? .dark : .light))
        #expect(host.fittingSize.width == width && host.fittingSize.height > 100)
      }
    }
  }
}

private func editEndpoint(context: Int = 4096, revision: String = "before") throws
  -> ConfiguredEndpoint
{
  try ManagerWire.decode(
    ConfiguredEndpoint.self,
    from: JSONSerialization.data(withJSONObject: [
      "port": 12345, "model": "qwen3.8-27b", "running": false, "revision": revision,
      "settings": [
        "host": "127.0.0.1", "max_ctx": context, "max_batch": 4, "has_api_key": true,
        "vision": false, "forensics": false, "device": "metal",
      ],
    ]))
}

private actor EndpointEditFixture: ManagerLoading {
  private(set) var mutations = 0
  private var failure = false
  private var pollFailure = false
  private var action = "save"
  func configure(failure: Bool = false, pollFailure: Bool = false) {
    self.failure = failure
    self.pollFailure = pollFailure
  }
  func snapshot() async throws -> ManagerSnapshot {
    let base = try endpointFixture()
    return ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: base.catalog,
      runners: [],
      servers: [
        try editEndpoint(
          context: mutations == 0 || failure ? 4096 : 8192,
          revision: mutations == 0 ? "before" : "after")
      ])
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    if case .poll = command {
      if pollFailure { throw ManagerError.core("Synthetic status failure") }
      return try receipt(failure ? "failed" : "succeeded")
    }
    mutations += 1
    if case .remove = command { action = "remove" }
    return try receipt("running")
  }
  private func receipt(_ state: String) throws -> ManagementJob {
    try ManagerWire.decode(
      ManagementJob.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": 7, "port": 12345, "action": action, "state": state,
        "message": failure ? "Stale configuration" : "Saved for next start",
      ]))
  }
}
