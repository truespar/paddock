import Foundation
import PaddockClient
import Synchronization
import Testing

@testable import PaddockConversationCore
@testable import PaddockStudio

@Suite("Independent native conversation sessions", .serialized, .timeLimit(.minutes(1)))
@MainActor struct StudioSessionTests {
  private func workspace() async throws -> StudioWorkspace {
    SessionProtocol.state.withLock { $0 = .init() }
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let chat = StudioWorkspace(client: SessionNoCore())
    chat.transportFactory = {
      let config = URLSessionConfiguration.ephemeral
      config.protocolClasses = [SessionProtocol.self]
      return try NativeConversationTransport(host: host, configuration: config)
    }
    let session = try chat.makeSession()
    chat.selectSession(session)
    try await session.runtime.start()
    await chat.start()
    #expect(chat.ready)
    return chat
  }
  private func wait(_ predicate: () -> Bool) async throws {
    for _ in 0..<500 {
      if predicate() { return }
      try await Task.sleep(for: .milliseconds(10))
    }
    throw ManagerError.core("Session fixture timed out")
  }
  private func send(_ text: String, chat: StudioWorkspace) async throws -> String {
    let id = try #require(chat.composerIdentity)
    chat.composerText = text
    #expect(await chat.send(text))
    try await wait { SessionProtocol.state.withLock { $0.pending[text] != nil } }
    await chat.runtime?.publishNow()
    #expect(chat.busy && chat.canNavigateConversation)
    return id
  }

  @Test func switchingKeepsStreamsSeparateAndReattachesWithoutReloading() async throws {
    let chat = try await workspace()
    let first = try await send("first", chat: chat)
    let firstCore = try #require(chat.runtime)
    try await chat.command("newChat")
    let second = try await send("second", chat: chat)
    let secondCore = try #require(chat.runtime)
    #expect(firstCore !== secondCore && chat.hasBackgroundWork)
    try await chat.command("open", ["id": .string(first)])
    #expect(chat.runtime === firstCore && chat.conversation?.id == first)
    #expect(chat.state?.nativeTranscript?.messages.last?.streaming == true)
    await SessionProtocol.finish("second", answer: "Only the second reply")
    try await wait { SessionProtocol.saved(second, contains: "Only the second reply") }
    #expect(chat.conversation?.id == first)
    #expect(chat.state?.nativeTranscript?.messages.last?.text != "Only the second reply")
    await SessionProtocol.finish("first", answer: "Only the first reply")
    try await wait { !chat.busy && SessionProtocol.saved(first, contains: "Only the first reply") }
    #expect(chat.state?.nativeTranscript?.messages.last?.text == "Only the first reply")
    #expect(await chat.shutdown())
  }

  @Test func stopTargetsSelectedSessionAndBackgroundHistoryCannotBeDeleted() async throws {
    let chat = try await workspace()
    let first = try await send("first", chat: chat)
    try await chat.command("newChat")
    let second = try await send("second", chat: chat)
    await #expect(throws: (any Error).self) {
      try await chat.command("deleteChats", ["ids": .array([.string(first)])])
    }
    try await chat.command("stop")
    try await wait { !chat.busy }
    #expect(SessionProtocol.state.withLock { $0.pending["first"] != nil })
    #expect(
      SessionProtocol.state.withLock {
        $0.documents[second]?["messages"]?.array?.last?["stopped"] == .bool(true)
      })
    await SessionProtocol.finish("first", answer: "Still running in background")
    try await wait { SessionProtocol.saved(first, contains: "Still running in background") }
    #expect(await chat.shutdown())
  }

  @Test func failedBackgroundSaveRetainsRuntimeAndBlocksQuitUntilRetry() async throws {
    let chat = try await workspace()
    let first = try await send("first", chat: chat)
    let firstCore = try #require(chat.runtime)
    try await chat.command("newChat")
    SessionProtocol.state.withLock { $0.failSave = true }
    await SessionProtocol.finish("first", answer: "Must survive a disk error")
    try await wait {
      chat.sessions.values.contains { $0.state?.error.contains("could not be saved") == true }
    }
    #expect(await chat.shutdown() == false)
    #expect(chat.ready && chat.sessions.values.contains { $0.runtime === firstCore })
    try await chat.command("open", ["id": .string(first)])
    #expect(chat.state?.nativeTranscript?.messages.last?.text == "Must survive a disk error")
    SessionProtocol.state.withLock { $0.failSave = false }
    await chat.perform("retrySave")
    await chat.runtime?.publishNow()
    #expect(chat.state?.saveFailed == false && chat.error == nil)
    #expect(await chat.shutdown())
    #expect(SessionProtocol.saved(first, contains: "Must survive a disk error"))
  }

  @Test func failedDestinationKeepsActiveStreamAndDraft() async throws {
    let chat = try await workspace()
    let first = try await send("first", chat: chat)
    let core = chat.runtime
    await #expect(throws: (any Error).self) {
      try await chat.command("open", ["id": .string("missing")])
    }
    #expect(chat.runtime === core && chat.conversation?.id == first)
    #expect(!chat.switchingConversation && chat.sessions.count == 1)
    #expect(SessionProtocol.state.withLock { $0.pending["first"] != nil })
    #expect(await chat.shutdown())
  }

  @Test func idleNavigationReleasesOldRuntimeAndRetainsDrafts() async throws {
    let chat = try await workspace()
    let id = try #require(chat.composerIdentity)
    chat.composerText = "An unfinished question"
    try await chat.command("newChat")
    try await wait { chat.sessions.count == 1 }
    try await chat.command("open", ["id": .string(id)])
    #expect(chat.composerText == "An unfinished question")
    #expect(await chat.shutdown())
  }
  @Test func compareLanesStayTogetherWhileAnotherChatRuns() async throws {
    let chat = try await workspace()
    try await chat.command("models", ["ids": .array([.string("local"), .string("peer")])])
    let compared = try await send("compare", chat: chat)
    try await wait { SessionProtocol.state.withLock { $0.pending["compare-peer"] != nil } }
    try await chat.command("newChat")
    let foreground = try #require(chat.composerIdentity)
    await SessionProtocol.finish("compare", answer: "First lane")
    #expect(chat.composerIdentity == foreground)
    try await chat.command("open", ["id": .string(compared)])
    #expect(chat.busy)
    await SessionProtocol.finish("compare-peer", answer: "Second lane")
    try await wait { !chat.busy }
    let replies = chat.state?.nativeTranscript?.messages.filter { $0.role == "assistant" } ?? []
    #expect(Set(replies.map(\.text)) == ["First lane", "Second lane"])
    #expect(replies.allSatisfy { !$0.streaming && !$0.stopped })
    #expect(await chat.shutdown())
  }

  @Test func searchFilterAndBusyRowsSurviveSwitching() async throws {
    let chat = try await workspace()
    let first = try await send("first", chat: chat)
    try await chat.command(
      "historyFilter", ["search": .string("first"), "sort": .string("oldest")])
    await chat.runtime?.publishNow()
    try await chat.command("newChat")
    await chat.runtime?.publishNow()
    #expect(chat.state?.library?.search == "first" && chat.state?.library?.sort == "oldest")
    #expect(chat.state?.library?.rows.first { $0.id == first }?.busy == true)
    #expect(await chat.shutdown())
  }

  @Test func activeSessionLimitDoesNotDiscardFifthDraft() async throws {
    let chat = try await workspace()
    for index in 0..<4 {
      _ = try await send("turn-\(index)", chat: chat)
      try await chat.command("newChat")
    }
    chat.composerText = "Fifth draft stays here"
    #expect(await chat.send(chat.composerText) == false)
    #expect(chat.composerText == "Fifth draft stays here")
    #expect(SessionProtocol.state.withLock { $0.pending.count == 4 })
    #expect(await chat.shutdown())
    #expect(
      SessionProtocol.state.withLock {
        $0.drafts.values.contains { $0["text"] == .string("Fifth draft stays here") }
      })
  }

  @Test func suspendedCommandCannotBeReleasedOrDetached() async throws {
    let chat = try await workspace()
    let core = try #require(chat.runtime)
    SessionProtocol.state.withLock { $0.holdHistory = true }
    let refresh = Task { try await core.command("refresh") }
    try await wait { SessionProtocol.state.withLock { $0.pending["_history"] != nil } }
    #expect(await core.canLeaveConversation() == false)
    #expect(await core.canReleaseConversation() == false)
    await SessionProtocol.releaseHistory()
    _ = try await refresh.value
    #expect(await core.canLeaveConversation())
    #expect(await chat.shutdown())
  }
}

private struct SessionNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.core("Fixture") }
}

private final class SessionProtocol: URLProtocol, @unchecked Sendable {
  typealias V = ConversationValue
  struct State: Sendable {
    var documents: [String: V] = [:]
    var drafts: [String: V] = [:]
    var pending: [String: SessionProtocol] = [:]
    var failSave = false
    var holdHistory = false
  }
  static let state = Mutex(State())
  static let queue = DispatchQueue(label: "paddock.sessions.fixture")
  override class func canInit(with request: URLRequest) -> Bool { request.url?.port == 43219 }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  static func saved(_ id: String, contains text: String) -> Bool {
    state.withLock {
      $0.documents[id]?["messages"]?.array?.last?["content"]?.array?.first?["text"]?.string == text
    }
  }
  static func releaseHistory() async {
    await withCheckedContinuation { continuation in
      queue.async {
        let pair = state.withLock { state in
          state.holdHistory = false
          return (state.pending.removeValue(forKey: "_history"), Array(state.documents.values))
        }
        if let request = pair.0 {
          request.respond(200, mime: "application/json")
          request.client?.urlProtocol(request, didLoad: try! JSONEncoder().encode(V.array(pair.1)))
          request.client?.urlProtocolDidFinishLoading(request)
        }
        continuation.resume()
      }
    }
  }
  static func finish(_ key: String, answer: String) async {
    await withCheckedContinuation { continuation in
      queue.async {
        if let request = state.withLock({ $0.pending.removeValue(forKey: key) }) {
          let event: V = .object([
            "type": .string("response.completed"),
            "response": .object([
              "status": .string("completed"),
              "output": .array([
                .object([
                  "type": .string("message"),
                  "content": .array([
                    .object([
                      "type": .string("output_text"), "text": .string(answer),
                    ])
                  ]),
                ])
              ]),
            ]),
          ])
          request.client?.urlProtocol(
            request,
            didLoad: Data(
              ("data: " + String(decoding: try! JSONEncoder().encode(event), as: UTF8.self) + "\n\n")
                .utf8))
          request.client?.urlProtocolDidFinishLoading(request)
        }
        continuation.resume()
      }
    }
  }
  override func startLoading() {
    Self.queue.async { [self] in
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
      let path = request.url!.path
      if path == "/api/conversations",
        Self.state.withLock({ state in
          guard state.holdHistory else { return false }
          state.pending["_history"] = self
          return true
        })
      {
        return
      }
      if path.hasSuffix("/responses") {
        var key =
          body["input"]?.array?.flatMap { $0["content"]?.array ?? [] }.compactMap {
            $0["text"]?.string
          }.last ?? "unknown"
        if body["model"]?.string == "peer" { key += "-peer" }
        Self.state.withLock { $0.pending[key] = self }
        respond(200, mime: "text/event-stream")
        client?.urlProtocol(
          self,
          didLoad: Data(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n".utf8))
        return
      }
      let reply: (Int, V) = Self.state.withLock { state in
        if path == "/api/settings" {
          return (
            200,
            .object([
              "macos_studio_preferences": .object([
                "pk_auto_title": .string("false"), "pk_summarize": .string("false"),
              ])
            ])
          )
        }
        if path == "/api/runners" {
          return (
            200,
            .array(
              ["local", "peer"].enumerated().map { index, name in
                .object([
                  "model": .string(name), "port": .number(Decimal(12481 + index)),
                  "status": .string("ok"),
                ])
              })
          )
        }
        if path.hasSuffix("/server") { return (200, .object(["max_ctx": .number(32768)])) }
        if path == "/api/conversations" { return (200, .array(Array(state.documents.values))) }
        if path == "/api/conversation-drafts" {
          return (
            200,
            .array(
              state.drafts.values.compactMap { value in
                guard var row = value["conversation"]?.object else { return nil }
                row["composerDraft"] = .bool(true)
                return .object(row)
              })
          )
        }
        if path.hasPrefix("/api/conversation-drafts/") {
          let id = request.url!.lastPathComponent
          if request.httpMethod == "PUT" {
            if state.failSave { return (503, .null) }
            guard (state.drafts[id]?["revision"]?.string ?? "") == body["revision"]?.string else {
              return (409, .null)
            }
            var value = body.object!
            value["revision"] = .string(UUID().uuidString)
            state.drafts[id] = .object(value)
            return (200, .object(["revision": value["revision"]!]))
          }
          return (200, state.drafts[id] ?? .null)
        }
        if path.hasPrefix("/api/conversations/") {
          let id = request.url!.lastPathComponent
          if request.httpMethod == "PUT" {
            if state.failSave { return (503, .null) }
            state.documents[id] = body
            if request.value(forHTTPHeaderField: "X-Paddock-Consume-Draft") != nil {
              state.drafts[id] = nil
            }
            return (200, .object([:]))
          }
          return (state.documents[id] == nil ? 404 : 200, state.documents[id] ?? .null)
        }
        return (200, .array([]))
      }
      respond(reply.0, mime: "application/json")
      client?.urlProtocol(self, didLoad: try! JSONEncoder().encode(reply.1))
      client?.urlProtocolDidFinishLoading(self)
    }
  }
  private func respond(_ status: Int, mime: String) {
    client?.urlProtocol(
      self,
      didReceive: HTTPURLResponse(
        url: request.url!, statusCode: status, httpVersion: nil,
        headerFields: ["Content-Type": mime])!, cacheStoragePolicy: .notAllowed)
  }
  override func stopLoading() {
    Self.queue.async { [self] in
      Self.state.withLock { state in state.pending = state.pending.filter { $0.value !== self } }
    }
  }
}
