import Foundation

extension ReadQuestion {
  /// References use stable editor identities, never the editable prompt IDs.
  public struct Condition: Codable, Equatable, Sendable {
    public var question: UUID
    public var answers: [String]
    public init(question: UUID, answers: [String] = []) {
      self.question = question
      self.answers = answers
    }
  }
  public var answerNames: [String] {
    switch kind {
    case .noul: ["yes", "no"]
    case .choice: options.map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines) }
    case .score: levels.map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines) }
    }
  }
  public var dependencies: Set<UUID> { Set(after + askIf.map(\.question)) }
  public var conditional: Bool { alone || !dependencies.isEmpty }

  // The former local run format has no conditions; keep it decodable too.
  private enum CodingKeys: String, CodingKey {
    case id, questionID, idTouched, kind, instructions, yesMeans, noMeans, options, levels
    case askIf, after, alone
  }
  public init(from decoder: any Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    self.init(
      questionID: try c.decode(String.self, forKey: .questionID),
      kind: try c.decode(Kind.self, forKey: .kind))
    id = try c.decode(UUID.self, forKey: .id)
    idTouched = try c.decodeIfPresent(Bool.self, forKey: .idTouched) ?? false
    instructions = try c.decodeIfPresent(String.self, forKey: .instructions) ?? ""
    yesMeans = try c.decodeIfPresent(String.self, forKey: .yesMeans) ?? ""
    noMeans = try c.decodeIfPresent(String.self, forKey: .noMeans) ?? ""
    options = try c.decodeIfPresent([Option].self, forKey: .options) ?? options
    levels = try c.decodeIfPresent([Option].self, forKey: .levels) ?? levels
    askIf = try c.decodeIfPresent([Condition].self, forKey: .askIf) ?? []
    after = try c.decodeIfPresent([UUID].self, forKey: .after) ?? []
    alone = try c.decodeIfPresent(Bool.self, forKey: .alone) ?? false
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(id, forKey: .id)
    try c.encode(questionID, forKey: .questionID)
    try c.encode(idTouched, forKey: .idTouched)
    try c.encode(kind, forKey: .kind)
    try c.encode(instructions, forKey: .instructions)
    try c.encode(yesMeans, forKey: .yesMeans)
    try c.encode(noMeans, forKey: .noMeans)
    try c.encode(options, forKey: .options)
    try c.encode(levels, forKey: .levels)
    try c.encode(askIf, forKey: .askIf)
    try c.encode(after, forKey: .after)
    try c.encode(alone, forKey: .alone)
  }
}

extension ReadDraft {
  public var conditionValidation: String? {
    let rows = Dictionary(questions.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
    for q in questions {
      for id in q.dependencies {
        guard let other = rows[id] else {
          return "\(q.questionID): a condition names a missing question."
        }
        if other.id == q.id { return "\(q.questionID): a question cannot wait on itself." }
      }
      if Set(q.askIf.map(\.question)).count != q.askIf.count {
        return "\(q.questionID): combine conditions on the same question."
      }
      for condition in q.askIf {
        guard let other = rows[condition.question] else { continue }
        if condition.answers.isEmpty {
          return "\(q.questionID): pick the answers of \(other.questionID) to ask on."
        }
        if let invalid = condition.answers.first(where: { !other.answerNames.contains($0) }) {
          return "\(q.questionID): \(other.questionID) cannot answer \(invalid)."
        }
      }
    }
    var visiting: Set<UUID> = []
    var done: Set<UUID> = []
    func cyclic(_ id: UUID) -> Bool {
      if visiting.contains(id) { return true }
      if done.contains(id) { return false }
      visiting.insert(id)
      for dependency in rows[id]?.dependencies ?? [] {
        if cyclic(dependency) { return true }
      }
      visiting.remove(id)
      done.insert(id)
      return false
    }
    return questions.contains { cyclic($0.id) }
      ? "These questions wait on each other. Remove the circular dependency." : nil
  }

  mutating func importConditions(_ map: [String: ConversationValue]) throws {
    let ids = Dictionary(uniqueKeysWithValues: questions.map { ($0.questionID, $0.id) })
    func resolve(_ name: String) throws -> UUID {
      guard let id = ids[name] else {
        throw ConversationFailure.invalid("A condition names missing question \(name).")
      }
      return id
    }
    func strings(_ value: ConversationValue) throws -> [String] {
      guard let list = value.array, list.allSatisfy({ $0.string != nil }) else {
        throw ConversationFailure.invalid("Conditions require arrays of names.")
      }
      return list.compactMap(\.string)
    }
    for index in questions.indices {
      let raw = map[questions[index].questionID]!
      if let alone = raw["alone"] {
        guard alone == .bool(true) || alone == .bool(false) else {
          throw ConversationFailure.invalid("alone must be true or false.")
        }
        questions[index].alone = alone == .bool(true)
      }
      if let after = raw["depends_on"] {
        var seen: Set<UUID> = []
        questions[index].after = try strings(after).map(resolve).filter { seen.insert($0).inserted }
      }
      if let conditions = raw["ask_if"] {
        guard let entries = conditions.object else {
          throw ConversationFailure.invalid("ask_if must map questions to answers.")
        }
        questions[index].askIf = try entries.keys.sorted().map { name in
          let entry = entries[name]!
          let answers = try entry.string.map { [$0] } ?? strings(entry)
          return try .init(question: resolve(name), answers: answers)
        }
      }
    }
  }

  public mutating func removeQuestion(_ id: UUID) {
    questions.removeAll { $0.id == id }
    for i in questions.indices {
      questions[i].askIf.removeAll { $0.question == id }
      questions[i].after.removeAll { $0 == id }
    }
  }
  public mutating func followAnswerRename(_ id: UUID, before: [String], after: [String]) {
    guard before.count == after.count else { return }
    for i in questions.indices {
      for j in questions[i].askIf.indices where questions[i].askIf[j].question == id {
        questions[i].askIf[j].answers = questions[i].askIf[j].answers.map { value in
          before.firstIndex(of: value).map { after[$0] } ?? value
        }
      }
    }
  }
}
