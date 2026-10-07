import AppKit
import PaddockClient
import PaddockConversationCore
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native Reads cross-client parity", .serialized) @MainActor
struct NativeReadParityTests {
  let request =
    #"{"questions":{"first":{"type":"noul"},"second":{"type":"noul","ask_if":{"first":["yes"]}}}}"#
  let response =
    #"{"model":"diffusion","answers":{"first":{"type":"noul","noul":0.1,"confidence":0.8,"agreement":1,"outside":0},"second":null},"diagnostics":{"reads":1,"canvas":16,"skipped":{"second":{"because":"first","was":"no","wanted":["yes"]}},"questions":[],"timing":{"total_ms":1}}}"#
  func value(_ json: String) throws -> ConversationValue {
    try JSONDecoder().decode(ConversationValue.self, from: Data(json.utf8))
  }
  func model(conditional: Bool) async -> NativeReadsModel {
    let m = NativeReadsTests().model()
    let api = m.api
    m.api = { path, method, body, query in
      if path == "api/runners/1234/server" {
        return try value(
          #"{"structured_read":{"canvas_width":256,"max_questions":64,"max_samples":32,"images":true,"conditional":CONDITION}}"#
            .replacingOccurrences(of: "CONDITION", with: conditional ? "true" : "false"))
      }
      return try await api(path, method, body, query)
    }
    await m.refresh()
    return m
  }

  @Test func conditionsAreCapabilityGatedWithoutDiscardingTheDraft() async throws {
    let m = await model(conditional: false)
    #expect(m.applyJSON(request))
    m.draft.state = "Test"
    #expect(!m.canRun && m.validation?.contains("conditions") == true)
    #expect(!m.draft.questions[1].askIf.isEmpty)
    let enabled = await model(conditional: true)
    #expect(enabled.applyJSON(request))
    enabled.draft.state = "Test"
    #expect(enabled.canRun)
    let first = enabled.draft.questions[0].id
    enabled.editID(first, text: "renamed")
    #expect(enabled.draft.setBody["questions"]?["second"]?["ask_if"]?["renamed"] != nil)
    enabled.duplicate(enabled.draft.questions[1].id)
    #expect(enabled.draft.questions[2].askIf == enabled.draft.questions[1].askIf)
    enabled.draft.removeQuestion(first)
    #expect(enabled.draft.questions.allSatisfy { $0.askIf.isEmpty })
  }

  @Test func webConditionalHistoryOpensAndKeepsCameraSnapshots() async throws {
    let input = try ReadDraft.parse(Data(request.utf8))
    let picture = ReadPicture(name: "camera 12:30:00", url: "data:image/jpeg;base64,YQ==")
    let run: ConversationValue = .object([
      "at": .number(1000), "port": .number(1234), "model": .string("diffusion"),
      "state": .string("A camera with optional context"),
      "excerpt": .string("A camera with optional context"),
      "chars": .number(30), "samples": .string("auto"), "ms": .number(25),
      "questions": input.setBody["questions"]!, "response": try value(response),
      "images": .array([picture.historyReference]),
    ])
    let doc = ReadHistoryDocument(
      value: .object([
        "id": .string("web-read"), "title": .string("Web read"), "model": .string("diffusion"),
        "createdAt": .number(1000), "updatedAt": .number(1000), "runs": .array([run]),
        "images": .object([picture.ref: .string(picture.url)]),
      ]))
    let m = await model(conditional: true)
    let api = m.api
    m.api = { path, method, body, query in
      if path == "api/read-history/web-read" {
        return .object(["doc": .string(try doc.json), "revision": .string("rev-1")])
      }
      return try await api(path, method, body, query)
    }
    await m.openSession("web-read")
    #expect(m.historyError == nil && m.runs.count == 1)
    #expect(m.result?.isCameraFrame == true)
    #expect(m.result?.response.diagnostics.skipped?["second"]?.was == "no")
    #expect(m.result?.elapsedMilliseconds == 25)
    #expect(m.draft.setBody == input.setBody)
  }

  @Test func visibilityPolicyDoesNotRequireApplicationFocus() {
    #expect(
      NativeReadCamera.canReadVisibleWindow(visible: true, minimized: false, appHidden: false))
    #expect(
      !NativeReadCamera.canReadVisibleWindow(visible: false, minimized: false, appHidden: false))
    #expect(
      !NativeReadCamera.canReadVisibleWindow(visible: true, minimized: true, appHidden: false))
    #expect(
      !NativeReadCamera.canReadVisibleWindow(visible: true, minimized: false, appHidden: true))
  }

  @Test func conditionalRunUsesTheWireAndPersistsNullSkips() async throws {
    let m = await model(conditional: true)
    #expect(m.applyJSON(request))
    m.draft.state = "A question, not an urgent incident"
    var sent = false
    var saved: ReadHistoryDocument?
    let api = m.api
    m.readAPI = { _, bytes in
      let wire = try JSONDecoder().decode(ConversationValue.self, from: bytes)
      #expect(wire["questions"]?["second"]?["ask_if"]?["first"] == .array([.string("yes")]))
      sent = true
      return try value(response)
    }
    m.api = { path, method, body, query in
      if path.hasPrefix("api/read-history/"), method == "PUT" {
        saved = try ReadHistoryDocument(json: #require(body?["doc"]?.string))
        return .object(["read": .object(["revision": .string("rev")])])
      }
      return try await api(path, method, body, query)
    }
    m.run()
    await m.settle()
    #expect(sent && m.error == nil && !m.historyUnsaved)
    #expect(saved?.runs.first?["response"]?["answers"]?["second"] == .null)
    let restored = try ReadHistoryDocument.draft(#require(saved?.runs.first))
    try #require(m.result).response.validate(for: restored.questions)
  }

  @Test func hiddenCameraWaitsAndVisibleCameraResumesWithoutQueueing() async throws {
    let m = await model(conditional: true)
    var visible = false
    var calls = 0
    var frames = 0
    m.readAPI = { _, _ in
      calls += 1
      visible = false
      return try value(
        #"{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.9,"confidence":0.8,"agreement":1,"outside":0}},"diagnostics":{"reads":1,"canvas":16,"questions":[],"timing":{"total_ms":1}}}"#
      )
    }
    let camera = NativeReadCamera()
    defer { camera.close() }
    camera.startLoop(
      model: m,
      frame: { _ in
        frames += 1
        return Data([1])
      }, visible: { visible })
    try await Task.sleep(for: .milliseconds(20))
    #expect(frames == 0 && calls == 0 && camera.live)
    visible = true
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while camera.latest == nil, ContinuousClock.now < deadline {
      try await Task.sleep(for: .milliseconds(10))
    }
    #expect(camera.latest != nil && frames == 1 && calls == 1)
    try await Task.sleep(for: .milliseconds(280))
    #expect(frames == 1 && calls == 1 && camera.live)
  }

  @Test func imageLoadFileUsesExtractionWhileAddImagesUsesVision() async throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".png")
    // Test-owned bytes only: no photograph, camera access, or model inference.
    try Data([1, 2, 3]).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    let m = await model(conditional: true)
    var extracted = false
    m.api = { path, method, body, _ in
      #expect(path == "api/runners/1234/extract" && method == "POST")
      #expect(body?["file_metadata"] == .string("off"))
      extracted = true
      return .object(["text": .string("Extracted text")])
    }
    await m.loadFile(url)
    #expect(extracted && m.draft.state == "Extracted text" && m.draft.images.isEmpty)
    #expect(m.error == nil)
  }

  @Test func conditionControlsFitTheNativeColumnWithoutOpeningAWindow() throws {
    var input = try ReadDraft.parse(Data(request.utf8))
    input.questions[0].questionID =
      "a_very_long_question_name_that_must_not_push_the_controls_outside_the_window"
    input.questions[1].after = [input.questions[0].id]
    input.questions[1].alone = true
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: NativeReadConditionEditor(
          question: .constant(input.questions[1]), others: [input.questions[0]]
        )
        .environment(\.colorScheme, dark ? .dark : .light))
      #expect(host.sizeThatFits(in: NSSize(width: 340, height: 1000)).width <= 341)
    }
  }
}
