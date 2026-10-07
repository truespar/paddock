import Foundation
import PaddockClient
import Testing

@testable import PaddockConversationCore

@Suite("Native presentation and history budgets")
struct NativeHardeningTests {
  @Test func coalescedPublicationDoesNotCancelItsOwnDecoder() async throws {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43210", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let capture = PublicationCapture()
    let transport = try NativeConversationTransport(host: host)
    let runtime = NativeStudioRuntime(transport: transport) { _ in
      await capture.receive(cancelled: Task.isCancelled)
    }
    try await runtime.newDocument()
    await runtime.schedulePublish()
    #expect(await capture.next() == false)
    await runtime.close()
  }
  struct Numbers: Codable, Equatable {
    let signed: Int64
    let unsigned: UInt64
    let decimal: Decimal
    let floating: Double
    let absent: String?
    let booleans: [Bool]
    let unicode: String
  }
  @Test func directDecodeMatchesJSONWithoutRoundingIntegers() throws {
    let expected = Numbers(
      signed: .min, unsigned: .max,
      decimal: Decimal(string: "123456789.123456789")!, floating: 1.25, absent: nil,
      booleans: [true, false], unicode: "å 🦊\n\u{0}")
    let data = try JSONEncoder().encode(expected)
    let raw = try JSONDecoder().decode(ConversationValue.self, from: data)
    #expect(try ConversationValueDecoder().decode(Numbers.self, from: raw) == expected)
    for invalid: ConversationValue in [.bool(true), .string("1"), .number(1.5), .number(-1)] {
      #expect(throws: (any Error).self) {
        try ConversationValueDecoder().decode(UInt64.self, from: invalid)
      }
    }
  }
  private struct Projection: Decodable {
    let nativeTranscript: Transcript
    struct Transcript: Decodable { let messages: [Message] }
    final class Message: Decodable {
      let id: String
      let text: String
    }
  }
  @Test func largePresentationDecodeCost() throws {
    var messages: [ConversationValue] = (0..<512).map {
      .object([
        "id": .string("m\($0)"),
        "text": .string(String(repeating: "Markdown body text. ", count: 200)),
      ])
    }
    let cache = ConversationDecodingCache()
    var cached: [Double] = []
    var json: [Double] = []
    func milliseconds(_ duration: Duration) -> Double {
      Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
    }
    for index in 0..<12 {
      messages[511] = .object(["id": .string("m511"), "text": .string("Streaming delta \(index)")])
      let raw = ConversationValue.object([
        "nativeTranscript": .object(["messages": .array(messages)])
      ])
      let started = ContinuousClock.now
      let direct = try cache.decode(Projection.self, from: raw)
      let middle = ContinuousClock.now
      let roundtrip = try JSONDecoder().decode(Projection.self, from: JSONEncoder().encode(raw))
      let ended = ContinuousClock.now
      #expect(
        direct.nativeTranscript.messages.last?.text
          == roundtrip.nativeTranscript.messages.last?.text)
      if index > 0 {
        cached.append(milliseconds(started.duration(to: middle)))
        json.append(milliseconds(middle.duration(to: ended)))
      }
    }
    print(
      "512-message presentation median ms: cached=\(cached.sorted()[5]), JSON roundtrip=\(json.sorted()[5])"
    )
  }
  @Test func incrementalProjectionReusesUnchangedMessagesAndDropsOldBranches() throws {
    func value(_ text: String, ids: [String] = ["a", "b"]) -> ConversationValue {
      .object([
        "nativeTranscript": .object([
          "messages": .array(
            ids.map {
              .object(["id": .string($0), "text": .string($0 == "b" ? text : "stable")])
            })
        ])
      ])
    }
    let cache = ConversationDecodingCache()
    let first = try cache.decode(Projection.self, from: value("one"))
    let unchanged = try cache.decode(Projection.self, from: value("one"))
    let second = try cache.decode(Projection.self, from: value("two"))
    #expect(first.nativeTranscript.messages[0] === second.nativeTranscript.messages[0])
    #expect(first.nativeTranscript.messages[1] === unchanged.nativeTranscript.messages[1])
    #expect(first.nativeTranscript.messages[1] !== second.nativeTranscript.messages[1])
    _ = try cache.decode(Projection.self, from: value("", ids: []))
    let reopened = try cache.decode(Projection.self, from: value("one"))
    #expect(first.nativeTranscript.messages[0] !== reopened.nativeTranscript.messages[0])
    cache.reclaim()
  }
  @Test func historyRollsOverWithoutDeletingThePreviousRuns() throws {
    var doc: ReadHistoryDocument?
    for index in 0..<20 {
      let next = try ReadHistoryAdmission.append(
        .object(["at": .number(Decimal(index))]),
        pictures: [], title: "Read", to: doc)
      #expect(!next.rolledOver)
      doc = next.document
    }
    let full = try #require(doc)
    let next = try ReadHistoryAdmission.append(
      .object(["at": .number(21)]),
      pictures: [], title: "Read", to: full)
    #expect(next.rolledOver && next.document.id != full.id)
    #expect(full.runs.count == 20 && next.document.runs.count == 1)
  }
  @Test func byteBudgetRollsOverAndRetainsOversizedResultForExport() throws {
    let run = ConversationValue.object(["state": .string(String(repeating: "x", count: 1024))])
    let first = try ReadHistoryAdmission.append(
      run, pictures: [], title: "Read", to: nil,
      byteLimit: 1500
    ).document
    let next = try ReadHistoryAdmission.append(
      run, pictures: [], title: "Read", to: first,
      byteLimit: 1500)
    #expect(next.rolledOver && first.runs.count == 1)
    do {
      _ = try ReadHistoryAdmission.append(
        run, pictures: [], title: "Read", to: first, byteLimit: 100)
      Issue.record("An oversized result must not be admitted")
    } catch let overflow as ReadHistoryAdmission.Overflow {
      #expect(overflow.document.runs == [run])
    }
  }
}

private actor PublicationCapture {
  private var received: Bool?
  private var waiting: CheckedContinuation<Bool, Never>?
  func receive(cancelled: Bool) {
    received = cancelled
    waiting?.resume(returning: cancelled)
    waiting = nil
  }
  func next() async -> Bool {
    if let received { return received }
    return await withCheckedContinuation { waiting = $0 }
  }
}
