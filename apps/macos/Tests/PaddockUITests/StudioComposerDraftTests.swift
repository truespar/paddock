import Foundation
import PaddockClient
import Synchronization
import Testing

@testable import PaddockConversationCore
@testable import PaddockStudio
@testable import PaddockUI

@Suite("Conversation-owned native drafts", .serialized) @MainActor
struct StudioComposerDraftTests {
  private func workspace() async throws -> StudioWorkspace {
    DraftProtocol.state.withLock { $0 = .init() }
    let workspace = StudioWorkspace(client: DraftNoCore())
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43218", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [DraftProtocol.self]
    let runtime = NativeStudioRuntime(
      transport: try NativeConversationTransport(host: host, configuration: configuration)
    ) { fields in
      if let value = try? JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields))
      {
        await workspace.apply(value)
      }
    }
    workspace.runtime = runtime
    try await runtime.start()
    await workspace.start()
    #expect(workspace.ready)
    return workspace
  }

  @Test func simultaneousFlushesDrainLatestTypingExactlyOnceBeforeNavigation() async throws {
    let chat = try await workspace()
    let id = try #require(chat.composerIdentity)
    chat.composerText = "First edit"
    let first = Task { try await chat.flushComposerDraft() }
    for _ in 0..<100 {
      if DraftProtocol.state.withLock({ $0.writes > 0 }) { break }
      try await Task.sleep(for: .milliseconds(5))
    }
    chat.composerText = "Latest edit"
    async let second: Void = chat.flushComposerDraft()
    try await first.value
    try await second
    await chat.runtime?.publishNow()
    #expect(
      StudioConversationView(chat: chat, draft: .constant(StudioDraft(message: chat.composerText)))
        .welcome)
    #expect(DraftProtocol.state.withLock { $0.writes == 2 })
    #expect(DraftProtocol.state.withLock { $0.drafts[id]?["text"] == .string("Latest edit") })
    try await chat.command("newChat")
    #expect(chat.composerText.isEmpty && chat.composerIdentity != id)
    try await chat.command("open", ["id": .string(id)])
    #expect(chat.composerText == "Latest edit")
    await chat.shutdown()
  }

  @Test func failedSaveBlocksNavigationAndShutdownDrainsAcceptedEdits() async throws {
    let chat = try await workspace()
    let id = try #require(chat.composerIdentity)
    chat.composerText = "Never discard this"
    DraftProtocol.state.withLock { $0.fail = true }
    await #expect(throws: (any Error).self) { try await chat.command("newChat") }
    #expect(chat.composerText == "Never discard this" && chat.composerIdentity == id)
    #expect(!chat.switchingConversation)
    DraftProtocol.state.withLock { $0.fail = false }
    chat.attachments = [
      StudioAttachment(
        id: "unfinished", name: "File.pdf", mime: "application/pdf", size: 1, phase: "Failed")
    ]
    await chat.shutdown()
    #expect(!chat.ready)
    #expect(
      DraftProtocol.state.withLock { $0.drafts[id]?["text"] == .string("Never discard this") })
    #expect(DraftProtocol.state.withLock { $0.drafts[id]?["attachments"] == .array([]) })
  }
  private func draft(_ id: String, text: String, attachments: [StudioAttachment] = []) throws
    -> StudioState.ComposerDraft
  {
    let files = try JSONDecoder().decode(
      [StudioValue].self, from: JSONEncoder().encode(attachments))
    return try JSONDecoder().decode(
      StudioState.ComposerDraft.self,
      from: JSONEncoder().encode(
        StudioValue.object([
          "id": .string(id), "text": .string(text), "attachments": .array(files),
        ])))
  }

  @Test func restoreIsAnIdentityBoundaryNotAnEchoThatReplacesNewTyping() throws {
    let model = WorkspaceModel(client: DraftNoCore())
    model.chat.restoreComposer(try draft("first", text: "Stored first"))
    model.draft.message = "Newer typing"
    let edit = model.chat.composerEditRevision
    model.chat.restoreComposer(try draft("first", text: "Stale projection"))
    #expect(model.draft.message == "Newer typing")
    #expect(model.chat.composerEditRevision == edit)
    model.chat.restoreComposer(try draft("second", text: "Second chat"))
    #expect(model.draft.message == "Second chat")
    #expect(model.chat.composerSavedRevision == model.chat.composerEditRevision)
    #expect(model.chat.canNavigateConversation)
  }

  @Test func attachmentChoicesRoundTripIncludingInvalidIntermediateEdits() throws {
    let model = WorkspaceModel(client: DraftNoCore())
    var pdf = StudioAttachment(
      id: "pdf", name: "Plan.pdf", mime: "application/pdf", size: 128, phase: "Ready")
    pdf.pages = 24
    pdf.firstPage = "2"
    pdf.lastPage = "7x"
    pdf.textOnly = true
    pdf.detail = "high"
    model.chat.restoreComposer(try draft("pdf-chat", text: "Review", attachments: [pdf]))
    let restored = try #require(model.chat.attachments.first)
    #expect(restored.firstPage == "2" && restored.lastPage == "7x")
    #expect(restored.textOnly && restored.detail == "high" && restored.pages == 24)
    #expect(restored.selectionError != nil)
    // Invalid intermediate input must be saved, not silently clamped or lost.
    #expect(model.chat.canNavigateConversation)
    model.chat.attachments[0].lastPage = "7"
    #expect(model.chat.composerEditRevision > model.chat.composerSavedRevision)
    model.chat.restoreComposer(try draft("other", text: ""))
    #expect(model.chat.attachments.isEmpty)
  }

  @Test func failedFilesAndActiveOperationsStillBlockNavigation() {
    let model = WorkspaceModel(client: DraftNoCore())
    model.chat.attachments = [
      StudioAttachment(id: "pending", name: "File", mime: "text/plain", size: 0, phase: "Uploading")
    ]
    #expect(!model.chat.canNavigateConversation)
    model.chat.attachments[0].phase = "Failed"
    #expect(!model.chat.canNavigateConversation)
    model.chat.attachments = []
    model.chat.microphoneStarting = true
    #expect(!model.chat.canNavigateConversation)
  }
}

private struct DraftNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    throw ManagerError.core("Synthetic offline core")
  }
}

private final class DraftProtocol: URLProtocol, @unchecked Sendable {
  typealias V = ConversationValue
  struct State: Sendable {
    var drafts: [String: V] = [:]
    var writes = 0
    var fail = false
  }
  static let state = Mutex(State())
  override class func canInit(with request: URLRequest) -> Bool { request.url?.port == 43218 }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let path = request.url!.path
    var bytes = request.httpBody ?? Data()
    if bytes.isEmpty, let stream = request.httpBodyStream {
      stream.open()
      defer { stream.close() }
      var buffer = [UInt8](repeating: 0, count: 4096)
      while stream.hasBytesAvailable {
        let count = stream.read(&buffer, maxLength: buffer.count)
        if count <= 0 { break }
        bytes.append(contentsOf: buffer.prefix(count))
      }
    }
    let body = (try? JSONDecoder().decode(V.self, from: bytes)) ?? .null
    let reply: (Int, V) = Self.state.withLock { state in
      if path == "/api/conversation-drafts" {
        return (
          200,
          .array(
            state.drafts.map { id, value in
              .object([
                "id": .string(id), "title": value["conversation"]?["title"] ?? .string("Draft"),
                "model": .string("default"), "updatedAt": .number(1), "composerDraft": .bool(true),
              ])
            })
        )
      }
      if path.hasPrefix("/api/conversation-drafts/") {
        let id = request.url!.lastPathComponent
        if request.httpMethod == "PUT" {
          state.writes += 1
          if state.fail { return (503, .null) }
          guard (state.drafts[id]?["revision"]?.string ?? "") == body["revision"]?.string else {
            return (409, .null)
          }
          var fields = body.object!
          fields["revision"] = .string(UUID().uuidString)
          state.drafts[id] = .object(fields)
          return (200, .object(["revision": fields["revision"]!]))
        }
        return (200, state.drafts[id] ?? .null)
      }
      if path.hasPrefix("/api/conversations/") { return (404, .null) }
      return (200, path == "/api/settings" ? .object([:]) : .array([]))
    }
    DispatchQueue.global().asyncAfter(deadline: .now() + (request.httpMethod == "PUT" ? 0.1 : 0)) {
      [self] in
      client?.urlProtocol(
        self,
        didReceive: HTTPURLResponse(
          url: request.url!, statusCode: reply.0, httpVersion: nil,
          headerFields: ["Content-Type": "application/json"])!, cacheStoragePolicy: .notAllowed)
      client?.urlProtocol(self, didLoad: (try? JSONEncoder().encode(reply.1)) ?? Data())
      client?.urlProtocolDidFinishLoading(self)
    }
  }
  override func stopLoading() {}
}
