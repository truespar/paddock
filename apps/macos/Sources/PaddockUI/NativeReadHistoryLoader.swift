import Foundation
import PaddockConversationCore

/// Hydration is serialized off the UI thread. SQLite remains the durable owner;
/// at most one bounded document is parsed/validated by this workspace at a time.
actor NativeReadHistoryLoader {
  struct Loaded: Sendable {
    let document: ReadHistoryDocument
    let runs: [NativeReadsModel.Run]
    let draft: ReadDraft?
  }
  func load(_ json: String) throws -> Loaded {
    try Task.checkCancellation()
    let doc = try ReadHistoryDocument(json: json)
    return try hydrate(doc)
  }
  func hydrate(_ doc: ReadHistoryDocument) throws -> Loaded {
    try Task.checkCancellation()
    var restored: [NativeReadsModel.Run] = []
    var last: ReadDraft?
    let decoder = ConversationValueDecoder()
    for value in doc.runs {
      try Task.checkCancellation()
      var input = try ReadHistoryDocument.draft(value)
      input.images = try ReadPicture.restore(value, table: doc.value["images"])
      let raw = value["response"] ?? .null
      let response = try decoder.decode(ReadResponse.self, from: raw)
      try response.validate(for: input.questions)
      guard let p = value["port"]?.integer, let port = UInt16(exactly: p), port > 0 else {
        throw ConversationFailure.invalid("Invalid saved reader port.")
      }
      var run = NativeReadsModel.Run(
        fingerprint: "", excerpt: value["excerpt"]?.string ?? "",
        characters: value["chars"]?.integer ?? 0, questions: input.questions, raw: raw,
        port: port,
        elapsedMilliseconds: try decoder.decode(Double.self, from: value["ms"] ?? .number(0)),
        response: response, state: value["stateMissing"] == .bool(true) ? nil : input.state,
        fileName: value["fileName"]?.string ?? "", samples: input.samples,
        pictures: input.images, steps: input.steps, think: input.think, checkpoint: input.checkpoint
      )
      run.id = value["id"]?.string.flatMap(UUID.init(uuidString:)) ?? UUID()
      run.at = Date(
        timeIntervalSince1970: try decoder.decode(Double.self, from: value["at"] ?? .number(0))
          / 1000)
      restored.append(run)
      last = input
    }
    try Task.checkCancellation()
    return Loaded(document: doc, runs: restored, draft: last)
  }
}
