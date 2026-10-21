import AppKit
import Foundation
import PaddockClient
import PaddockConversationCore
import PaddockStudio
import Testing

@testable import PaddockUI

@Suite("OS integration policy", .serialized, .timeLimit(.minutes(2))) @MainActor
struct DesktopTests {
  @Test func readerEndpointOpensReadsAndPreservesBothDrafts() async throws {
    let model = WorkspaceModel(client: DesktopReaderCore())
    model.draft.message = "Keep the chat draft"
    model.reads.api = { path, _, _, _ in
      let json: String
      if path == "api/runners" {
        json = #"[{"port":12345,"reader":"laya"}]"#
      } else if path.hasSuffix("/server") {
        json =
          #"{"structured_read":{"backend":"laya","max_samples":1,"checkpoints":[{"name":"english"}]}}"#
      } else {
        json = "[]"
      }
      return try JSONDecoder().decode(ConversationValue.self, from: Data(json.utf8))
    }
    await model.refresh()
    await model.handleDesktopRequest(.init(.chat(port: 12345)))
    #expect(model.navigation.studio == .reads && model.reads.port == 12345)
    #expect(model.desktopError == nil && model.draft.message == "Keep the chat draft")
    model.reads.draft.state = "Keep the read draft"
    model.reads.port = 12346
    await model.handleDesktopRequest(.init(.chat(port: 12345)))
    #expect(model.reads.port == 12346 && model.reads.draft.state == "Keep the read draft")
    #expect(model.desktopError?.contains("Nothing was discarded") == true)
  }
  @Test func inventoryRestorationAndNormalPollingAreSilent() throws {
    var tracker = DesktopEventTracker()
    let snapshot = try endpointFixture()
    #expect(tracker.management(snapshot).isEmpty)
    for _ in 0..<100 { #expect(tracker.management(snapshot).isEmpty) }
  }
  @Test func readinessRequiresHealthNotAPID() throws {
    var tracker = DesktopEventTracker()
    #expect(tracker.management(try fleet(status: nil)).isEmpty)
    #expect(tracker.management(try fleet(status: "unreachable")).isEmpty)
    let ready = tracker.management(try fleet(status: "ok"))
    #expect(ready.map(\.kind) == [.modelReady])
    #expect(ready.first?.route == .chat(port: 12345))
    #expect(tracker.management(try fleet(status: "ok")).isEmpty)
    #expect(tracker.management(try fleet(status: "ok", pid: 43)).map(\.kind) == [.modelReady])
    #expect(tracker.management(try fleet(status: "ok", pid: 43)).isEmpty)
  }
  @Test func outageNeedsTwoSamplesAndRecoveryRearmsIt() throws {
    for status: String? in [nil, "unreachable"] {
      var tracker = DesktopEventTracker()
      _ = tracker.management(try fleet(status: "ok"))
      #expect(tracker.management(try fleet(status: status)).isEmpty)
      #expect(tracker.management(try fleet(status: status)).map(\.kind) == [.modelIssue])
      for _ in 0..<10 { #expect(tracker.management(try fleet(status: status)).isEmpty) }
      #expect(tracker.management(try fleet(status: "ok")).map(\.kind) == [.modelReady])
    }
  }
  @Test func plannedStopsDoNotBecomeCrashNotifications() throws {
    var tracker = DesktopEventTracker()
    _ = tracker.management(try fleet(status: "ok"))
    #expect(tracker.management(try fleet(status: "ok", job: "running")).isEmpty)
    #expect(tracker.management(try fleet(status: nil, job: "succeeded")).isEmpty)
    #expect(tracker.management(try fleet(status: nil, job: "succeeded")).isEmpty)
  }
  @Test func failedOperationsNotifyOnceWithoutExposingErrorBodies() throws {
    var tracker = DesktopEventTracker()
    _ = tracker.management(try fleet(status: "ok", job: "running"))
    let events = tracker.management(try fleet(status: "ok", job: "failed"))
    #expect(events.count == 1)
    #expect(events[0].kind == .modelIssue)
    #expect(!events[0].body.contains("sensitive"))
    #expect(tracker.management(try fleet(status: "ok", job: "failed")).isEmpty)
  }
  @Test func historicalRepliesAreNotNewNotifications() throws {
    var tracker = DesktopEventTracker()
    #expect(
      tracker.studio(conversationID: "chat", busy: false, activity: try activity("completed"))
        .isEmpty)
    #expect(
      tracker.studio(conversationID: "other-chat", busy: false, activity: try activity("completed"))
        .isEmpty)
  }
  @Test func terminalNotificationsAreDeduplicatedAndRespectCancellation() throws {
    for state in ["completed", "failed", "incomplete", "stopped"] {
      var tracker = DesktopEventTracker()
      #expect(
        tracker.studio(conversationID: "chat", busy: true, activity: try activity("streaming"))
          .isEmpty)
      let result = tracker.studio(
        conversationID: "chat", busy: false, activity: try activity(state))
      #expect(
        result.map(\.kind)
          == (state == "stopped" ? [] : [state == "completed" ? .replyReady : .replyIssue]))
      #expect(
        tracker.studio(conversationID: "chat", busy: false, activity: try activity(state)).isEmpty)
    }
  }
  @Test func fastRepliesDoNotNeedAnObservedStreamingFrame() throws {
    var tracker = DesktopEventTracker()
    _ = tracker.studio(conversationID: nil, busy: false, activity: try activity("completed"))
    let terminal = try activity("completed", turn: "fast-request")
    #expect(
      tracker.studio(conversationID: "chat", busy: false, activity: terminal).map(\.kind) == [
        .replyReady
      ])
    #expect(tracker.studio(conversationID: "chat", busy: false, activity: terminal).isEmpty)
    #expect(tracker.studio(conversationID: "other-chat", busy: false, activity: terminal).isEmpty)
  }
  @Test func approvalNotificationsNeverApproveAnything() throws {
    var tracker = DesktopEventTracker()
    let projection = try activity("streaming", approvals: ["approval-a"])
    let result = tracker.studio(conversationID: "chat", busy: true, activity: projection)
    #expect(result.map(\.kind) == [.approval])
    #expect(result.first?.route == .conversation("chat"))
    #expect(tracker.studio(conversationID: "chat", busy: true, activity: projection).isEmpty)
  }
  @Test func notificationRoutesAreAllowlistedDataOnly() {
    #expect(DesktopNotifications.route(["destination": "manager"]) == .manager)
    #expect(
      DesktopNotifications.route(["destination": "model", "port": "12345"]) == .chat(port: 12345))
    #expect(
      DesktopNotifications.route(["destination": "conversation", "id": "chat-123"])
        == .conversation("chat-123"))
    for data: [String: Any] in [
      ["destination": "stop", "port": "12345"], ["destination": "model", "port": "65536"],
      ["destination": "model", "port": "80"], ["destination": "conversation", "id": "../secrets"],
      ["destination": "conversation", "id": String(repeating: "a", count: 129)],
      ["destination": "https://evil.example"], ["destination": "conversation", "id": ""],
    ] { #expect(DesktopNotifications.route(data) == nil) }
  }
  @Test func quickStartRequiresStoppedLocalRevision() throws {
    let base = try endpointFixture()
    #expect(!EndpointRow.rows(snapshot: base, latestJob: nil)[0].canQuickStart)
    for (running, local, revision, allowed) in [
      (false, true, "r", true), (true, true, "r", false), (false, false, "r", false),
      (false, true, "", false),
    ] {
      let endpoint = try ManagerWire.decode(
        ConfiguredEndpoint.self,
        from: JSONSerialization.data(withJSONObject: [
          "port": 12345, "running": running, "local_only": local,
          "revision": revision.isEmpty ? NSNull() : revision as Any,
        ]))
      let value = ManagerSnapshot(
        identity: base.identity, readiness: base.readiness, catalog: base.catalog, runners: [],
        servers: [endpoint])
      #expect(EndpointRow.rows(snapshot: value, latestJob: nil)[0].canQuickStart == allowed)
    }
  }
  @Test func osNavigationPreservesUnsentDraftAndModeState() async {
    let model = WorkspaceModel(client: DesktopNoCore())
    model.draft.message = "Keep this private draft"
    await model.handleDesktopRequest(DesktopRequest(.manager))
    #expect(model.navigation.mode == .manager)
    #expect(model.draft.hasContent)
    await model.handleDesktopRequest(DesktopRequest(.newChat))
    #expect(model.draft.message == "Keep this private draft")
    // Navigation is now allowed with an unsent draft, but a missing core must
    // still leave that draft untouched and surface its connection failure.
    #expect(model.desktopError == "The native conversation service is unavailable.")
    #expect(!model.desktopTransition)
  }
  @Test func anotherDesktopActionClearsThePreviousReadinessError() async {
    let model = WorkspaceModel(client: DesktopNoCore())
    model.desktopError = "This model is not ready for chat. Check Settings > Instances."
    await model.handleDesktopRequest(DesktopRequest(.studio))
    #expect(model.desktopError == nil)
    #expect(model.navigation.mode == .studio)
  }
  @Test func quickQuestionKeepsBothDraftsOnConflict() async {
    let model = WorkspaceModel(client: DesktopNoCore())
    model.draft.message = "Main draft"
    let quick = QuickQuestionModel()
    quick.text = "Quick draft"
    var opened = false
    #expect(!(await quick.handoff(to: model) { opened = true }))
    #expect(!opened)
    #expect(quick.text == "Quick draft")
    #expect(model.draft.message == "Main draft")
    #expect(quick.error != nil)
    model.draft.message = ""
    model.quitting = true
    #expect(!(await quick.handoff(to: model) { opened = true }))
    #expect(!opened && quick.text == "Quick draft")
  }
  @Test func quickAttachmentsAreBoundedAndRejectNetworkURLs() {
    let quick = QuickQuestionModel()
    quick.addFiles([URL(string: "https://example.com/document.pdf")!])
    #expect(quick.attachments.isEmpty)
    quick.addFiles((0..<33).map { URL(fileURLWithPath: "/tmp/synthetic-only-\($0).pdf") })
    #expect(quick.attachments.isEmpty)
    quick.addImage(Data(repeating: 0, count: 16 * 1024 * 1024 + 1), png: true)
    #expect(quick.attachments.isEmpty)
    quick.addFiles([URL(fileURLWithPath: "/tmp/synthetic-only.pdf")])
    #expect(quick.attachments.count == 1)
    #expect(quick.hasContent)
  }
  @Test func quitNeverStopsReplacementIdentity() async throws {
    let approved = try endpointFixture().runners
    let client = QuitFixture(replacePID: true)
    let model = WorkspaceModel(client: client)
    await model.start()
    #expect(!(await model.stopForQuit(approved)))
    #expect(await client.stops == 0)
    #expect(model.desktopError?.contains("replaced") == true)
  }
  @Test func quitWaitsForAcknowledgedStopAndKeepsOpenOnFailure() async throws {
    for succeeds in [true, false] {
      let client = QuitFixture(succeeds: succeeds)
      let model = WorkspaceModel(client: client)
      await model.start()
      model.quitting = true
      #expect(!model.canSubmit)
      #expect(await model.stopForQuit(model.managedRunners) == succeeds)
      #expect(await client.stops == 1)
      #expect(model.state != .stopped)  // Only the application delegate closes core.
      if !succeeds { #expect(model.desktopError != nil) }
    }
  }
  @Test func shortcutValidationDoesNotRegisterAKey() throws {
    let hotKey = QuestionHotKey()
    try hotKey.register("off") {}
    #expect(throws: DesktopShortcutError.self) { try hotKey.register("arbitrary-keylogger") {} }
  }
}

private struct DesktopNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    throw ManagerError.core("Synthetic unavailable core")
  }
}
private struct DesktopReaderCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    let base = try endpointFixture()
    return try ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: base.catalog,
      runners: ManagerWire.decode(
        [RunnerInfo].self,
        from: Data(
          #"[{"port":12345,"pid":42,"status":"ok","reader":"laya","endpoint":"http://127.0.0.1:12345"}]"#
            .utf8)),
      servers: [])
  }
}
private actor QuitFixture: ManagerLoading {
  var replacePID: Bool
  var succeeds: Bool
  var stops = 0
  init(replacePID: Bool = false, succeeds: Bool = true) {
    self.replacePID = replacePID
    self.succeeds = succeeds
  }
  func snapshot() async throws -> ManagerSnapshot {
    try fleet(
      status: stops > 0 && succeeds ? nil : "ok",
      job: stops > 0 ? (succeeds ? "succeeded" : "failed") : nil, pid: replacePID ? 43 : 42)
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    stops += 1
    return try ManagerWire.decode(
      ManagementJob.self,
      from: Data(
        #"{"id":1,"port":12345,"action":"stop","state":"running","message":"Draining"}"#.utf8))
  }
}
private func fleet(status: String?, job: String? = nil, pid: Int = 42) throws -> ManagerSnapshot {
  let base = try endpointFixture()
  let runners: [RunnerInfo] =
    try status.map {
      [
        try ManagerWire.decode(
          RunnerInfo.self,
          from: JSONSerialization.data(withJSONObject: [
            "port": 12345, "pid": pid, "status": $0, "endpoint": "http://127.0.0.1:12345",
          ]))
      ]
    } ?? []
  let jobs: [ManagementJob] =
    try job.map {
      [
        try ManagerWire.decode(
          ManagementJob.self,
          from: JSONSerialization.data(withJSONObject: [
            "id": 1, "port": 12345, "action": "stop", "state": $0,
            "message": "sensitive backend details",
          ]))
      ]
    } ?? []
  return ManagerSnapshot(
    identity: base.identity, readiness: base.readiness, catalog: base.catalog, runners: runners,
    jobs: jobs)
}
private func activity(_ state: String, approvals: [String] = [], turn: String? = nil) throws
  -> StudioState.Activity
{
  var object: [String: Any] = [
    "replies": [["id": "reply", "state": state]], "approvals": approvals,
  ]
  if let turn { object["completedTurn"] = ["id": turn, "conversationId": "chat", "state": state] }
  return try JSONDecoder().decode(
    StudioState.Activity.self, from: JSONSerialization.data(withJSONObject: object))
}
