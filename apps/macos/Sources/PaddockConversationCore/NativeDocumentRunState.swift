import Foundation

/// Bounded per-page reduction. Responses sequence numbers restart on every
/// page; a terminal event is mandatory and partial output is never called Done.
struct NativeDocumentRunState: Sendable {
  typealias V = ConversationValue
  typealias O = [String: V]
  let sourceID: String
  var pages: [O]
  var current: Int?
  var response = ResponseAccumulator(maximumBytes: 4 * 1024 * 1024)
  var confidence = NativeOCRConfidence()
  var firstToken: Double?
  var usageRows: [O] = []
  var missingUsage = false

  mutating func begin(_ index: Int) {
    current = index
    pages[index]["state"] = .string("reading")
    response = ResponseAccumulator(maximumBytes: 4 * 1024 * 1024)
    confidence = NativeOCRConfidence()
  }
  mutating func receive(_ event: O, elapsed: Double) throws {
    try response.apply(event)
    if event["type"]?.string == "response.output_text.delta" {
      if firstToken == nil, event["delta"]?.string?.isEmpty == false { firstToken = elapsed }
      confidence.append(event["logprobs"]?.array ?? [])
    }
  }
  mutating func flush() throws {
    guard let current else { return }
    let text = response.text
    let otherBytes = pages.enumerated().reduce(0) { sum, page in
      sum + (page.offset == current ? 0 : (page.element["text"]?.string ?? "").utf8.count)
    }
    guard text.utf8.count <= 8 * 1024 * 1024 - otherBytes else {
      throw ConversationFailure.tooLarge
    }
    pages[current]["text"] = .string(text)
  }
  mutating func finish() throws {
    try response.requireTerminal()
    try flush()
    guard let current else { throw ConversationFailure.stale }
    let terminal = response.terminal ?? [:]
    pages[current]["state"] = .string(
      response.status == "failed" ? "error" : response.status == "incomplete" ? "review" : "done")
    if response.status == "failed" {
      pages[current]["note"] = .string(response.failure ?? "The model failed to read this page")
    } else if response.status == "incomplete" {
      pages[current]["note"] = .string(
        terminal["incomplete_details"]?["reason"]?.string == "max_output_tokens"
          ? "Cut off at the token limit" : "The model returned an incomplete page")
    }
    if !response.reasoning.isEmpty { pages[current]["reasoning"] = .string(response.reasoning) }
    if let ocr = terminal["ocr"]?.object {
      pages[current]["ocr"] = NativeOCRMetadata.stored(.object(ocr))
      pages[current]["regions"] = .array(Self.regions(ocr["regions"]?.array ?? []))
    }
    if let words = confidence.words(matching: response.text) {
      pages[current]["words"] = .array(words)
    }
    if let usage = terminal["usage"]?.object,
      Self.validUsage(usage)
    {
      usageRows.append(usage)
      pages[current]["usage"] = .object(usage)
    } else {
      missingUsage = true
    }
    self.current = nil
  }
  mutating func fail(_ note: String, index: Int) {
    try? flush()
    pages[index]["state"] = .string("error")
    pages[index]["note"] = .string(note)
    missingUsage = true
    current = nil
  }
  mutating func stopRemaining(_ note: String) {
    try? flush()
    for i in pages.indices where ["queued", "reading"].contains(pages[i]["state"]?.string ?? "") {
      pages[i]["state"] = .string("error")
      pages[i]["note"] = .string(note)
    }
    missingUsage = true
    current = nil
  }
  var snapshot: V {
    .object(["sourceId": .string(sourceID), "pages": .array(pages.map(V.object))])
  }
  var text: String {
    pages.enumerated().map { i, page in
      let title = pages.count > 1 ? "## Page \(page["page"]?.integer ?? i + 1)\n\n" : ""
      return title + (page["text"]?.string ?? "")
    }.joined(separator: "\n\n")
  }
  func usage(seconds: Double) -> O? {
    // A total assembled from only the successful pages would look complete.
    // Retain their usage on the pages, but don't fabricate a whole-run total.
    guard !missingUsage, !pages.isEmpty, usageRows.count == pages.count,
      seconds.isFinite, seconds >= 0
    else { return nil }
    let input = usageRows.reduce(0) { $0 + ($1["input_tokens"]?.integer ?? 0) }
    let output = usageRows.reduce(0) { $0 + ($1["output_tokens"]?.integer ?? 0) }
    let reasoning = usageRows.reduce(0) {
      $0 + ($1["output_tokens_details"]?["reasoning_tokens"]?.integer ?? 0)
    }
    var result: O = [
      "promptTokens": .number(Decimal(input)),
      "completionTokens": .number(Decimal(max(0, output - reasoning))),
      "reasoningTokens": .number(Decimal(reasoning)), "ms": .number(Decimal(seconds * 1000)),
      "timingSource": .string("end-to-end"),
    ]
    if let firstToken { result["ttftMs"] = .number(Decimal(firstToken * 1000)) }
    if usageRows.allSatisfy({ $0["cost"]?.double.map { $0.isFinite && $0 >= 0 } == true }) {
      let cost = usageRows.reduce(0.0) { $0 + ($1["cost"]?.double ?? 0) }
      if cost.isFinite { result["costUsd"] = .number(Decimal(cost)) }
    }
    return NativeResponseMetrics.presentation(result)
  }
  private static func validUsage(_ usage: O) -> Bool {
    // Bound untrusted per-page counts before summing a run of up to 64 pages.
    guard let input = usage["input_tokens"]?.integer, let output = usage["output_tokens"]?.integer,
      (0...Int.max / 64).contains(input), (0...Int.max / 64).contains(output)
    else { return false }
    let reasoning = usage["output_tokens_details"]?["reasoning_tokens"]
    return reasoning == nil || reasoning?.integer.map { (0...output).contains($0) } == true
  }
  static func regions(_ values: [V]) -> [V] {
    values.prefix(4096).compactMap { value in
      guard let label = value["label"]?.string else { return nil }
      func coordinates(_ key: String, count: Int) -> V {
        .array(
          (value[key]?.array ?? []).prefix(4096).filter { row in
            guard let values = row.array, values.count == count else { return false }
            return values.allSatisfy {
              $0.double.map { $0.isFinite && $0 >= 0 && $0 <= 999 } == true
            }
          })
      }
      var region: O = [
        "label": .string(label), "text": .string(value["text"]?.string ?? ""),
        "boxes": coordinates("boxes", count: 4), "quads": coordinates("quads", count: 8),
      ]
      if value["continues"]?.bool == true { region["continues"] = .bool(true) }
      return .object(region)
    }
  }
  static func recover(_ message: O) -> O {
    guard var run = message["docRun"]?.object, var pages = run["pages"]?.array else {
      return message
    }
    var changed = false
    for i in pages.indices where ["queued", "reading"].contains(pages[i]["state"]?.string ?? "") {
      guard var page = pages[i].object else { continue }
      page["state"] = .string("error")
      page["note"] = .string("Interrupted")
      pages[i] = .object(page)
      changed = true
    }
    guard changed else { return message }
    var value = message
    run["pages"] = .array(pages)
    value["docRun"] = .object(run)
    value["streaming"] = .bool(false)
    value["stopped"] = .bool(true)
    return value
  }
}

/// Same exp(mean token logprob) and whitespace folding as web ocr.ts.
/// Scores are only shown when their token text matches the final extraction.
struct NativeOCRConfidence: Sendable {
  typealias V = ConversationValue
  var entries: [V] = []
  var tokenText = ""
  var tokenBytes = 0
  var invalid = false
  mutating func append(_ values: [V]) {
    guard !invalid else { return }
    for entry in values {
      guard let token = entry["token"]?.string, let score = entry["logprob"]?.double,
        score.isFinite, score <= 0, entries.count < 65_536,
        tokenBytes + token.utf8.count <= 4 * 1024 * 1024
      else {
        invalid = true
        entries = []
        tokenText = ""
        tokenBytes = 0
        return
      }
      tokenText += token
      tokenBytes += token.utf8.count
      entries.append(entry)
    }
  }
  func words(matching text: String) -> [V]? {
    guard !invalid, !entries.isEmpty, tokenText == text else { return nil }
    var out: [V] = []
    var word = ""
    var sum = 0.0
    var n = 0
    func flush() {
      if !word.isEmpty {
        out.append(
          .object(["w": .string(word), "c": .number(Decimal(exp(sum / Double(max(1, n)))))]))
      }
      word = ""
      sum = 0
      n = 0
    }
    for entry in entries {
      let token = entry["token"]!.string!
      if token.hasPrefix("<|"), token.hasSuffix("|>") { continue }
      var segment = ""
      func consume() {
        if !segment.isEmpty {
          word += segment
          sum += entry["logprob"]!.double!
          n += 1
          segment = ""
        }
      }
      for character in token {
        if character.isWhitespace {
          consume()
          flush()
        } else {
          segment.append(character)
        }
      }
      consume()
    }
    flush()
    return out
  }
}
