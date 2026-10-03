import Foundation

/// Shared /api/read-history schema. Keep explicit question/choice order when
/// passing through Swift dictionaries; the web client observes the same order.
public struct ReadHistoryDocument: Sendable {
  public var value: ConversationValue
  public init(value: ConversationValue) { self.value = value }
  public init(json: String) throws {
    let data = Data(json.utf8)
    guard data.count <= 16 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
    var root = try JSONDecoder().decode(ConversationValue.self, from: data)
    guard var object = root.object, let id = root["id"]?.string,
      ConversationDocument.validID(id), let runs = root["runs"]?.array, runs.count <= 20
    else { throw ConversationFailure.invalid("Invalid read history document.") }
    let index = try ReadJSONOrder(data)
    object["runs"] = .array(
      try runs.enumerated().map { i, run in
        guard var fields = run.object, let questions = run["questions"]?.object else {
          throw ConversationFailure.invalid("Invalid saved read.")
        }
        if fields["questionOrder"] == nil {
          let path = ["runs", String(i), "questions"]
          fields["questionOrder"] = .array(
            (index.keys[path] ?? questions.keys.sorted()).map { id in
              .array(
                ([id]
                  + (questions[id]?["type"]?.string == "choice"
                    ? index.keys[path + [id, "criteria"]] ?? [] : []))
                  .map(ConversationValue.string))
            })
        }
        return .object(fields)
      })
    root = .object(object)
    value = root
  }
  public var id: String { value["id"]?.string ?? "" }
  public var runs: [ConversationValue] { value["runs"]?.array ?? [] }
  public var json: String { get throws { try ReadDraft.json(value) } }
  public static func draft(_ run: ConversationValue) throws -> ReadDraft {
    var result = try ReadDraft.parse(
      JSONEncoder().encode(
        ConversationValue.object([
          "questions": run["questions"] ?? .object([:]),
          "samples": run["samples"] ?? .string("auto"),
          "steps": run["steps"] ?? .number(1),
          "think": run["think"] ?? .number(0),
          "model": run["checkpoint"] ?? .null,
        ])), maxQuestions: 1024)
    if let order = run["questionOrder"]?.array {
      let original = result.questions
      let ids = order.compactMap { $0.array?.first?.string }
      guard Set(ids) == Set(original.map(\.questionID)), ids.count == original.count else {
        throw ConversationFailure.invalid("Invalid question order in read history.")
      }
      result.questions = try order.map { row in
        let keys = row.array?.compactMap(\.string) ?? []
        var q = original.first { $0.questionID == keys.first }!
        if q.kind == .choice {
          let names = Array(keys.dropFirst())
          guard Set(names) == Set(q.options.map(\.name)), names.count == q.options.count else {
            throw ConversationFailure.invalid("Invalid choice order in read history.")
          }
          let options = q.options
          q.options = names.map { name in options.first { $0.name == name }! }
        }
        return q
      }
    }
    result.state = run["state"]?.string ?? ""
    return result
  }
}
