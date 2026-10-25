import Foundation
import PaddockClient
import PaddockConversationCore

extension NativeCoreCheck {
  static func composerDraftChecks(
    runtime: NativeStudioRuntime, transport: NativeConversationTransport, host: StudioHost
  ) async throws {
    typealias V = ConversationValue
    _ = try await transport.bytes(
      "api/attachments/draft-pdf", method: "PUT",
      body: Data("%PDF-synthetic-draft-only".utf8), contentType: "application/pdf",
      query: ["name": "Notes.pdf"])
    let metadata: [String: V] = [
      "id": .string("draft-pdf"), "name": .string("Notes.pdf"), "mime": .string("application/pdf"),
      "size": .number(25), "pages": .number(20),
    ]
    _ = try await runtime.command("stage", metadata)
    let display = metadata.merging([
      "phase": .string("Ready"), "detail": .string("auto"), "textOnly": .bool(true),
      "firstPage": .string("2"), "lastPage": .string("7"),
    ]) { _, new in new }
    _ = try await runtime.command(
      "saveDraft",
      [
        "conversationId": .string("native-fixture"), "text": .string("Keep this follow-up 🦊"),
        "attachments": .array([.object(display)]),
      ])
    let saved = try await transport.api("api/conversation-drafts/native-fixture")
    let viewState: V = .object([
      "version": .number(1),
      "reading": .object([
        "following": .bool(false), "blockID": .string("answer"),
        "index": .number(1), "offset": .number(244), "height": .number(1200), "width": .number(760),
      ]),
      "composer": .object([
        "location": .number(3), "length": .number(5),
        "digest": .string(String(repeating: "a", count: 64)),
      ]),
    ])
    let positionPath = "api/conversations/native-fixture/view-state/macos"
    try require(
      try await transport.api(positionPath, method: "PUT", body: viewState)["saved"] == .bool(true),
      "reading position stored independently in SQLite")
    let fresh = try NativeConversationTransport(host: host)
    try require(
      try await fresh.api(positionPath) == viewState,
      "fresh native transport restores exact reading and selection state")
    try require(
      try await fresh.api("api/conversations/native-fixture/view-state/web") == .null,
      "web and native view state are independent")
    await fresh.close()
    try require(saved["text"] == .string("Keep this follow-up 🦊"), "draft text stored in SQLite")
    try require(
      saved["attachments"]?.array?.first?["display"]?["lastPage"] == .string("7"),
      "draft page choices stored")
    try require(
      saved["conversation"]?["messages"]?.array?.isEmpty == true,
      "draft does not copy the transcript")
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": .string("native-fixture")])
    // A second native coordinator uses a fresh URLSession, but the same Rust
    // SQLite store. No model request, user database or window is involved.
    let capture = DraftProjection()
    let restarted = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) {
      await capture.receive($0)
    }
    try await restarted.start()
    let restored = await capture.value
    try require(restored["composerDraft"]?["text"] == saved["text"], "restart resumes the draft")
    try require(
      restored["composerDraft"]?["attachments"] == .array([.object(display)]),
      "restart restores exact attachment choices")
    _ = try await restarted.command(
      "saveDraft",
      [
        "conversationId": .string("native-fixture"), "text": .string(""), "attachments": .array([]),
      ])
    try require(
      try await transport.api("api/conversation-drafts/native-fixture") == .null,
      "clearing removes only the draft")
    try require(
      try await transport.loadConversation("native-fixture").messages.count == 3,
      "clearing preserves all message branches")
    try require(
      try await transport.api(positionPath) == viewState,
      "clearing a follow-up draft retains the conversation reading position")
    await restarted.close()
    print(
      "PASS: native composer drafts, PDF selections, restart restoration and empty-draft cleanup through real Rust/SQLite"
    )
  }
}

private actor DraftProjection {
  var value: [String: ConversationValue] = [:]
  func receive(_ value: [String: ConversationValue]) { self.value = value }
}
