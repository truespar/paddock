import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Native management presentation") @MainActor
struct ManagementUITests {
  @Test func backupOwnsItsReceiptAcrossNavigationAndRejectsDoubleSubmission() async throws {
    let client = BackupReceiptFixture()
    let workspace = WorkspaceModel(client: client)
    let path = URL(fileURLWithPath: "/tmp/fixture-only-not-written.sqlite")
    workspace.dataStorage.export(to: path)
    workspace.navigation.showManager(.models)
    workspace.navigation.returnToChat()
    workspace.dataStorage.export(to: URL(fileURLWithPath: "/tmp/not-a-second-export.sqlite"))
    #expect(workspace.maintenanceInProgress)
    await workspace.dataStorage.settle()
    #expect(!workspace.maintenanceInProgress)
    #expect(workspace.dataStorage.backupURL == path)
    #expect(workspace.dataStorage.receipt?.bytes == 1024)
    #expect(await client.exports == 1)
    #expect(await client.closed)
  }
  @Test func servingProfileChangesOnlyTheReviewedDraftAndMatchingArtifact() throws {
    let endpoint = try ManagerWire.decode(
      ConfiguredEndpoint.self,
      from: Data(
        #"{"port":11540,"model":"qwen3.8-27b","artifact":"mlx-4bit","running":false,"revision":"reviewed","settings":{"host":"127.0.0.1","max_ctx":4096,"max_batch":1,"has_api_key":true,"vision":true,"forensics":false,"device":"metal"}}"#
          .utf8))
    let profile = try ManagerWire.decode(
      ModelProfile.self,
      from: Data(
        #"{"id":"profile","name":"Long context","model":"qwen3.8-27b","artifact":"mlx-4bit","settings":{"max_ctx":8192,"max_batch":1,"runtime_options":[]}}"#
          .utf8))
    let editor = EndpointEditor(client: ProfileDraftFixture(), endpoint: endpoint, pid: nil)
    editor.useProfile(profile)
    #expect(editor.context == "8192" && editor.dirty)
    #expect(editor.host == "127.0.0.1" && editor.vision)
    #expect(editor.replacementKey.isEmpty && editor.pid == nil && !editor.saving)
    editor.context = "16384"
    editor.useProfile(profile)
    #expect(editor.context == "16384", "A profile must not overwrite an unsaved draft")
    editor.reset()
    editor.artifactID = "different-artifact"
    editor.useProfile(profile)
    #expect(editor.context == "4096", "A profile must not cross checkpoint identities")
  }
  @Test func clientExamplesUseActualServedIdentityAndEnvironmentKeys() throws {
    let setup = try ManagerWire.decode(
      LocalClientSetup.self,
      from: Data(
        #"{"base_url":"http://127.0.0.1:11540/v1","model":"test'$(bad)","has_key":true}"#.utf8))
    let raw = ExternalClient.openCode.configuration(setup)
    let config = try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
    let provider = (config?["provider"] as? [String: Any])?["paddock"] as? [String: Any]
    #expect((provider?["options"] as? [String: String])?["apiKey"] == "{env:PADDOCK_API_KEY}")
    #expect(raw.contains(setup.model))
    let shell = ExternalClient.responses.configuration(setup)
    #expect(shell.contains("$PADDOCK_API_KEY"))
    #expect(shell.contains("test'\\''$(bad)"))
    #expect(shell.contains("/responses"))
  }
  @Test func updaterRequiresPinnedHTTPSFeedAndRealPublicKey() {
    let valid: [String: Any] = [
      "SUFeedURL": "https://github.com/truespar/paddock/releases/latest/download/appcast.xml",
      "SUPublicEDKey": Data(repeating: 1, count: 32).base64EncodedString(),
      "SURequireSignedFeed": true, "SUVerifyUpdateBeforeExtraction": true,
    ]
    #expect(DesktopUpdater.validConfiguration(valid))
    for url in [
      "http://github.com/truespar/paddock/releases/latest/download/appcast.xml",
      "https://example.com/appcast.xml",
      "https://github.com/other/repo/releases/latest/download/appcast.xml",
    ] {
      var wrong = valid
      wrong["SUFeedURL"] = url
      #expect(!DesktopUpdater.validConfiguration(wrong))
    }
    var wrong = valid
    wrong["SUPublicEDKey"] = "placeholder"
    #expect(!DesktopUpdater.validConfiguration(wrong))
    #expect(!DesktopUpdater.validConfiguration([:]))
  }
}

private struct ProfileDraftFixture: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    Issue.record("Applying a profile must not perform management operations")
    throw ManagerError.closed
  }
}

private actor BackupReceiptFixture: ManagerLoading {
  private(set) var exports = 0
  private(set) var closed = false
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.closed }
  func maintenance(_ command: MaintenanceCommand) async throws -> MaintenanceReply {
    switch command {
    case .backup: exports += 1
    case .close: closed = true
    default: break
    }
    return try ManagerWire.decode(
      MaintenanceReply.self,
      from: Data(#"{"id":"backup","state":"complete","payload":"{\"bytes\":1024}"}"#.utf8))
  }
}
