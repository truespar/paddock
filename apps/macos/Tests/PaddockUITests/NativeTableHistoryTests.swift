import Foundation
import PaddockClient
import PaddockConversationCore
import Testing

@testable import PaddockUI

@Suite("Durable Tables history", .serialized) @MainActor
struct NativeTableHistoryTests {
  let source = "size,city,label\n1,a,yes\n2,b,no\n3,c,\n"
  func input() throws -> TableInput {
    let table = try TabularTable.parse(source)
    return TableInput(
      source: source, fileName: "Example.csv", spec: TabularSpec(table: table),
      model: "kumo", port: 1234, estimators: 8, seed: 0)
  }
  func raw<T: Encodable>(_ value: T) throws -> ConversationValue {
    try JSONDecoder().decode(ConversationValue.self, from: JSONEncoder().encode(value))
  }
  func response() throws -> ConversationValue {
    try JSONDecoder().decode(
      ConversationValue.self,
      from: Data(
        #"{"task":"classification","classes":["yes","no"],"predictions":[{"class":0,"label":"yes","probabilities":[0.8,0.2]}]}"#
          .utf8))
  }
  func summary(_ doc: ConversationValue, revision: String) -> ConversationValue {
    .object([
      "id": doc["id"]!, "title": doc["title"]!, "model": doc["model"]!,
      "createdAt": doc["createdAt"]!, "updatedAt": doc["updatedAt"]!,
      "runs": .number(Decimal(doc["runs"]!.array!.count)), "revision": .string(revision),
    ])
  }
  @Test func wireContractUsesWebTypesAndDatasetHash() throws {
    let input = try input()
    #expect(input.dataset == "a47678d9564a8102e8467fd45a06b9e753a5450ed76f3af163a15eea89c9f3c0")
    let wire = try raw(input)
    #expect(
      wire["spec"]?["types"]
        == .array([.string("numerical"), .string("categorical"), .string("categorical")]))
    let decoded = try JSONDecoder().decode(TableInput.self, from: JSONEncoder().encode(wire))
    #expect(decoded == input)
  }
  @Test func runSnapshotsSurviveReopenRenameAndStoppedModel() async throws {
    let history = NativeTableHistory()
    var documents: [String: ConversationValue] = [:]
    var rows: [String: ConversationValue] = [:]
    var revision = 0
    let api: NativeTablesModel.API = { path, method, body, query in
      #expect(!path.contains("?"))
      if method == "PUT" || method == "DELETE" { #expect(query["revision"] != nil) }
      let id = String(
        path.split(separator: "/").last!.split(separator: "?", omittingEmptySubsequences: false)[0])
      if method == "PUT", let body {
        revision += 1
        documents[id] = body
        let row = summary(body, revision: "r\(revision)")
        rows[id] = row
        return row
      }
      if path == "api/table-history" { return .array(Array(rows.values)) }
      if method == "DELETE" {
        rows[id] = nil
        documents[id] = nil
        return .null
      }
      let text = String(decoding: try JSONEncoder().encode(documents[id]!), as: UTF8.self)
      return .object(["doc": .string(text), "revision": rows[id]!["revision"]!])
    }
    var input = try input()
    let first = TableRun(input: input, task: .classification, ms: 3, response: try response())
    #expect(await history.save(input: input, source: source, run: first, api: api))
    input.seed = 1
    #expect(
      await history.save(
        input: input, source: source,
        run: TableRun(input: input, task: .classification, ms: 4, response: try response()),
        api: api))
    let id = try #require(history.document?.id)
    #expect(history.document?.datasets.count == 1 && history.document?.runs.count == 2)
    #expect(history.document?.runs.first?.input.seed == 0)
    #expect(!history.blocked)
    history.reset()
    let reopened = await history.open(id, api: api)
    #expect(reopened?.runs.count == 2 && reopened?.draft.seed == 1)
    await history.rename(id, title: "Renamed", api: api)
    #expect(history.document?.title == "Renamed" && history.document?.runs.count == 2)
    let row = try #require(history.sessions.first)
    #expect(await history.remove(row, api: api))
    #expect(history.document == nil && history.sessions.isEmpty)
  }
  @Test func failedAcknowledgementRetriesIdenticalPayloadAndRetainsResults() async throws {
    let history = NativeTableHistory()
    var calls: [ConversationValue] = []
    let api: NativeTablesModel.API = { _, _, body, _ in
      let body = try #require(body)
      calls.append(body)
      if calls.count == 1 { throw ConversationFailure.http(503) }
      return summary(body, revision: "r\(calls.count)")
    }
    let input = try input()
    let run = TableRun(input: input, task: .classification, ms: 3, response: try response())
    #expect(await !history.save(input: input, source: source, run: run, api: api))
    #expect(history.document?.runs.count == 1 && history.sessions.isEmpty)
    #expect(await history.save(input: input, source: source, api: api))
    #expect(calls[0] == calls[1])
    #expect(history.document?.runs.count == 1 && history.error == nil)
  }
}
