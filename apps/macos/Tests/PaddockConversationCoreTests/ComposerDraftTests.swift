import Foundation
import PaddockClient
import Testing

@testable import PaddockConversationCore

extension RuntimeTests {
  private func draftInput(_ runtime: NativeStudioRuntime, text: String, attachments: [V] = [])
    async throws -> O
  {
    [
      "conversationId": .string(try #require(await runtime.currentFields()?["id"]?.string)),
      "text": .string(text), "attachments": .array(attachments),
    ]
  }

  @Test func composerDraftRestoresTextModelsAndExactPDFChoicesWithoutCopyingTranscript()
    async throws
  {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    let file: O = [
      "id": .string("pdf"), "name": .string("Notes.pdf"), "mime": .string("application/pdf"),
      "size": .number(10), "pages": .number(20),
    ]
    _ = try await runtime.command("stage", file)
    let display = file.merging([
      "firstPage": .string(" 2 "), "lastPage": .string("7"), "textOnly": .bool(true),
      "detail": .string("high"),
    ]) { _, new in new }
    let input = try await draftInput(
      runtime, text: "Keep this 👋\ntext", attachments: [.object(display)])
    let id = try #require(input["conversationId"]?.string)
    _ = try await runtime.command("saveDraft", input)
    #expect(RuntimeProtocol.state.withLock { $0.documents.isEmpty })
    #expect(await runtime.presentation()["history"]?.array?.first?["composerDraft"] == .bool(true))
    _ = try await runtime.command("newChat")
    #expect(await runtime.presentation()["composerDraft"]?["text"] == .string(""))
    _ = try await runtime.command("open", ["id": .string(id)])
    let reopened = await runtime.presentation()
    #expect(reopened["composerDraft"]?["text"] == input["text"])
    #expect(reopened["composerDraft"]?["attachments"] == .array([.object(display)]))
    #expect(reopened["selectedModels"]?.array?.count == 2)
    #expect(await runtime.staged["pdf"]?["attachmentId"] == .string("pdf"))
    await runtime.close()
  }

  @Test func staleEditorAndSaveFailuresNeverReplaceAnotherConversation() async throws {
    let runtime = try await fixture()
    let first = try await draftInput(runtime, text: "Keep me")
    RuntimeProtocol.state.withLock { $0.failDraftSave = true }
    await #expect(throws: (any Error).self) { try await runtime.command("saveDraft", first) }
    #expect(RuntimeProtocol.state.withLock { $0.drafts.isEmpty })
    RuntimeProtocol.state.withLock { $0.failDraftSave = false }
    _ = try await runtime.command("saveDraft", first)
    _ = try await runtime.command("newChat")
    let second = try await draftInput(runtime, text: "Other draft")
    await #expect(throws: (any Error).self) { try await runtime.command("saveDraft", first) }
    _ = try await runtime.command("saveDraft", second)
    #expect(RuntimeProtocol.state.withLock { $0.drafts.count == 2 })
    #expect(await runtime.presentation()["composerDraft"]?["text"] == .string("Other draft"))
    await runtime.close()
  }

  @Test func failedAdmissionKeepsDurableDraftAndAcceptedTurnConsumesIt() async throws {
    let runtime = try await fixture()
    let input = try await draftInput(runtime, text: "Hello")
    _ = try await runtime.command("saveDraft", input)
    RuntimeProtocol.state.withLock { $0.failSave = true }
    await #expect(throws: (any Error).self) {
      try await runtime.command("send", ["text": .string("Hello")])
    }
    #expect(RuntimeProtocol.state.withLock { $0.drafts.count == 1 && $0.documents.isEmpty })
    RuntimeProtocol.state.withLock { $0.failSave = false }
    _ = try await runtime.command("send", ["text": .string("Hello")])
    _ = try await settled(runtime)
    #expect(RuntimeProtocol.state.withLock { $0.drafts.isEmpty && $0.documents.count == 1 })
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": input["conversationId"]!])
    #expect(await runtime.presentation()["composerDraft"]?["text"] == .string(""))
    await runtime.close()
  }

  @Test func restartingRestoresMostRecentDraftAndClearingItKeepsMessages() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("First turn")])
    _ = try await settled(runtime)
    let input = try await draftInput(runtime, text: "Unsent follow-up")
    _ = try await runtime.command("saveDraft", input)
    await runtime.close()
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43210", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [RuntimeProtocol.self]
    let restarted = NativeStudioRuntime(
      transport: try NativeConversationTransport(host: host, configuration: configuration)
    ) { _ in }
    try await restarted.start()
    #expect(await restarted.presentation()["composerDraft"]?["text"] == .string("Unsent follow-up"))
    #expect(await restarted.presentation()["conversation"]?["messageCount"] == .number(2))
    let cleared = try await draftInput(restarted, text: "")
    _ = try await restarted.command("saveDraft", cleared)
    #expect(RuntimeProtocol.state.withLock { $0.drafts.isEmpty && $0.documents.count == 1 })
    await restarted.close()
  }
}
