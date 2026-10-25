import Foundation

extension NativeStudioRuntime {
  func contextPresentation() -> V {
    guard let doc = document, let model = selected.first else { return .null }
    let plan = contextPlan(doc, modelID: model)
    guard plan.from > 0 || compactionTask != nil || !compactionNotice.isEmpty else { return .null }
    let summary = plan.summary ?? plan.item?["encrypted_content"]?.string ?? ""
    let path = doc.activeMessages
    return .object([
      "before": plan.from > 0 && plan.from < path.count ? path[plan.from]["id"]! : .null,
      "title": .string(
        plan.from == 0
          ? ""
          : summary.isEmpty
            ? "Earlier messages are outside the model’s context" : "Earlier messages summarized"),
      "summary": .string(summary), "working": .bool(compactionTask != nil),
      "error": .string(compactionNotice),
    ])
  }
  func cancelCompaction() {
    compactionEpoch += 1
    compactionTask?.cancel()
    compactionTask = nil
    compactionNotice = ""
  }
  func contextPlan(_ doc: ConversationDocument, modelID: String, continuing: Bool = false)
    -> NativeContextPlan
  {
    let comparison = (doc.fields["compareModels"]?.array ?? []).count > 1
    let window = comparison ? contextLimit : capability(modelID)["max_ctx"]?.integer ?? 0
    return NativeContextPlan.resolve(
      doc, context: window,
      reply: NativeContextPlan.reserve(
        maxTokens, context: window,
        outputCeiling: models.first { $0["id"]?.string == modelID }?["endpoint"] != nil
          ? capability(modelID)["default_max_output_tokens"]?.integer : nil),
      summarize: preferenceBool("summarize", fallback: true) && !comparison,
      server: models.first { $0["id"]?.string == modelID }?["endpoint"] == nil && !comparison
        && !continuing)
  }
  static let summaryKeys = ["summary", "summaryCount", "summaryLastId", "summaryModel"]
  static let summaryInstructions =
    "Summarize the conversation transcript below into a compact brief for continuing "
    + "the same conversation later. Keep: what the user is trying to do, decisions and "
    + "facts established, names/numbers/identifiers, open questions, and the current "
    + "state. Write plain prose, no preamble, at most 400 words."

  /// At most one optional job on the active conversation. Foreground commands
  /// cancel it. The result merges only summary fields into the CURRENT document,
  /// never saves a snapshot over later titles, branches, preferences or turns.
  func scheduleCompaction() {
    guard compactionTask == nil, !closed, !preparingQuit, !busy, !draft, selected.count == 1,
      preferenceBool("summarize", fallback: true), let doc = document,
      let modelID = selected.first, canChat(modelID),
      capability(modelID)["document_parser"]?.bool != true,
      let row = try? model(modelID), row["endpoint"] != nil,
      let context = capability(modelID)["max_ctx"]?.integer, context > 2688,
      doc.activeMessages.last?["error"] == nil, doc.activeMessages.last?["stopped"]?.bool != true,
      doc.activeMessages.last?["incomplete"] == nil, doc.activeMessages.last?["docRun"] == nil,
      doc.activeMessages.last?["transcript"] == nil
    else { return }
    let count = NativeContextPlan.compactionTarget(
      doc, context: context,
      reply: NativeContextPlan.reserve(
        maxTokens, context: context,
        outputCeiling: capability(modelID)["default_max_output_tokens"]?.integer))
    guard count > 0 else { return }
    let epoch = compactionEpoch
    let cap = capability(modelID)
    let timeout = compactionTimeout
    compactionNotice = ""
    compactionTask = Task { [weak self] in
      guard let self else { return }
      do {
        let text = await Task.detached(priority: .utility) {
          Self.summaryInput(doc, count: count, context: context, model: modelID)
        }.value
        try Task.checkCancellation()
        guard !text.isEmpty else { throw ConversationFailure.invalid("No text to summarize") }
        var body: O = [
          "model": row["wireModel"] ?? .string(modelID), "stream": .bool(true),
          "instructions": .string(Self.summaryInstructions), "input": .string(text),
          "max_output_tokens": .number(640), "temperature": .number(0),
        ]
        if cap["reasoning"]?.string == "effort" {
          body["reasoning"] = .object([
            "effort": .string(cap["reasoning_off"]?.bool == true ? "none" : "low")
          ])
        } else if cap["reasoning"]?.string == "toggle" {
          body["chat_template_kwargs"] = .object(["enable_thinking": .bool(false)])
        }
        let endpoint = try await self.endpoint(row)
        let transport = self.transport
        let request = body
        let result = try await withThrowingTaskGroup(of: ResponseAccumulator.self) { group in
          group.addTask {
            try await transport.responses(
              endpoint: endpoint, body: request, maximumBytes: 64 * 1024
            ) { _ in }
          }
          group.addTask {
            try await Task.sleep(for: timeout)
            throw ConversationFailure.invalid("Conversation summarization timed out")
          }
          defer { group.cancelAll() }
          return try await group.next()!
        }
        try Task.checkCancellation()
        let summary = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard result.status == "completed", !summary.isEmpty, summary.utf8.count <= 64 * 1024 else {
          throw ConversationFailure.invalid("The model did not complete a summary")
        }
        try await self.acceptSummary(summary, source: doc, count: count, epoch: epoch)
      } catch {
        await self.compactionFailed(
          epoch: epoch, cancelled: Task.isCancelled || error is CancellationError)
      }
      await self.compactionFinished(epoch: epoch)
    }
    schedulePublish()
  }
  nonisolated static func summaryInput(
    _ doc: ConversationDocument, count: Int, context: Int, model: String
  ) -> String {
    guard context > 2688 else { return "" }
    let cap = min(64 * 1024, context - 2688) * 4
    let prior = NativeContextPlan.summaryValid(doc) ? doc.fields["summaryCount"]?.integer ?? 0 : 0
    var lines: [String] = []
    var size = 0
    for message in doc.activeMessages.prefix(count).dropFirst(prior).reversed() {
      if message["role"]?.string == "assistant", message["group"] != nil,
        message["model"]?.string != model
      {
        continue
      }
      var notes: [String] = []
      for part in message["content"]?.array ?? [] {
        switch part["type"]?.string {
        case "image": notes.append("[image attached]")
        case "file": notes.append("[file: \(part["name"]?.string ?? "document")]")
        case "audio": notes.append("[audio: \(part["name"]?.string ?? "recording")]")
        default: break
        }
      }
      for tool in message["toolCalls"]?.array ?? [] {
        notes.append("[used tool \(tool["name"]?.string ?? "tool")]")
      }
      let text = ConversationDocument.text(message).trimmingCharacters(in: .whitespacesAndNewlines)
      guard !text.isEmpty || !notes.isEmpty else { continue }
      let line =
        "\(message["role"]?.string == "user" ? "User" : "Assistant"): \((notes + [text]).joined(separator: " ").trimmingCharacters(in: .whitespaces))"
      lines.append(String(line.suffix(max(0, cap - size))))
      size += line.count + 2
      if size >= cap { break }
    }
    let priorBlock =
      prior > 0
      ? "Summary of the conversation so far:\n\(doc.fields["summary"]?.string ?? "")\n\n" : ""
    return String((priorBlock + lines.reversed().joined(separator: "\n\n")).suffix(cap))
  }
  func acceptSummary(_ summary: String, source: ConversationDocument, count: Int, epoch: Int)
    async throws
  {
    guard epoch == compactionEpoch, !closed, !busy, let current = document,
      current.id == source.id, current.fields["model"] == source.fields["model"],
      selected.count == 1, preferenceBool("summarize", fallback: true),
      Array(current.activeMessages.prefix(count)) == Array(source.activeMessages.prefix(count)),
      Self.summaryKeys.allSatisfy({ current.fields[$0] == source.fields[$0] })
    else { return }
    let patch: O = [
      "summary": .string(summary), "summaryCount": .number(Decimal(count)),
      "summaryLastId": source.activeMessages[count - 1]["id"]!,
      "summaryModel": source.fields["model"]!,
    ]
    try change { $0.merge(patch) { _, new in new } }
    do { try await persist() } catch {
      if document?.id == current.id, patch.allSatisfy({ document?.fields[$0.key] == $0.value }) {
        try? change { fields in for key in Self.summaryKeys { fields[key] = current.fields[key] } }
      }
      throw error
    }
  }
  func compactionFailed(epoch: Int, cancelled: Bool) {
    if epoch == compactionEpoch && !cancelled {
      compactionNotice =
        "Earlier messages could not be summarized. The next reply will use the context that fits."
    }
  }
  func compactionFinished(epoch: Int) {
    guard epoch == compactionEpoch else { return }
    compactionTask = nil
    schedulePublish()
  }
}
