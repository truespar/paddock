import Foundation

extension NativeStudioRuntime {
  func admit(
    _ p: O, requestID: String,
    action: (ConversationMessageAction, ResolvedConversationAction)? = nil
  ) async throws -> O {
    cancelCompaction()
    titleTask?.cancel()
    titleTask = nil
    titleEpoch += 1
    guard let before = document else {
      throw ConversationFailure.invalid("Open a conversation first")
    }
    let wasDraft = draft
    let text = try Self.text(p["text"] ?? .string(""), limit: 128 * 1024)
    let issue = inputIssue()
    if action == nil, !issue.isEmpty { throw ConversationFailure.invalid(issue) }
    var parts: [V] = []
    if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      parts.append(.object(["type": .string("text"), "text": .string(text)]))
    }
    let choices = p["attachments"]?.array ?? []
    guard choices.count <= 32 else { throw ConversationFailure.invalid("Too many attachments") }
    var ids = Set<String>()
    for choice in choices {
      let id = try Self.id(choice["id"])
      guard ids.insert(id).inserted, var part = staged[id] else {
        throw ConversationFailure.invalid("An attachment is no longer staged")
      }
      if let detail = choice["detail"]?.string {
        guard ["auto", "low", "high"].contains(detail) else {
          throw ConversationFailure.invalid("Invalid image detail")
        }
        part["detail"] = .string(detail)
      }
      if choice["text"]?.bool == true { part["pdfMode"] = .string("text") }
      let first = choice["from"]?.integer
      let last = choice["to"]?.integer
      if first != nil || last != nil {
        guard first ?? 1 > 0, last ?? Int.max >= first ?? 1,
          last ?? first ?? 1 <= part["pages"]?.integer ?? Int.max
        else { throw ConversationFailure.invalid("Invalid page range") }
        part["pageRange"] = .string("\(first ?? 1)-\(last.map(String.init) ?? "")")
      }
      if part["type"]?.string == "audio" { part["language"] = audioPresentation()["language"] }
      parts.append(.object(part))
    }
    var laneIDs = selected
    var parent = before.tipID
    var appendID: String?
    if let (kind, resolved) = action {
      switch kind {
      case .edit:
        parts = resolved.parts
        parent = resolved.message["parentId"]?.string
      case .retry: parent = resolved.message["parentId"]?.string
      case .continueReply:
        appendID = resolved.message["id"]?.string
        laneIDs = [resolved.message["model"]?.string ?? before.fields["model"]!.string!]
      case .branch: throw ConversationFailure.invalid("A branch action cannot generate")
      }
    }
    guard !laneIDs.isEmpty else { throw ConversationFailure.invalid("Choose a model") }
    for id in laneIDs { _ = try model(id) }
    if parts.isEmpty, action == nil,
      laneIDs.contains(where: { capability($0)["document_parser"]?.bool == true }),
      before.activeMessages.contains(where: { !NativeDocumentPlan.rasterParts($0).isEmpty })
    {
      parts = [.object(["type": .string("text"), "text": .string("")])]
    }
    guard !parts.isEmpty || action != nil else {
      throw ConversationFailure.invalid("Write a message or attach a file")
    }
    stopRequested = false
    var messages = before.messages
    if !parts.isEmpty {
      let uid = UUID().uuidString
      messages.append([
        "id": .string(uid), "role": .string("user"), "parentId": parent.map(V.string) ?? .null,
        "content": .array(parts), "createdAt": Self.now,
      ])
      parent = uid
    }
    let group = laneIDs.count > 1 ? UUID().uuidString : nil
    var laneMessages: [(String, String)] = []
    for modelID in laneIDs {
      let mid = appendID ?? UUID().uuidString
      if let appendID, let i = messages.firstIndex(where: { $0["id"]?.string == appendID }) {
        continuationPrefixes[mid] = ConversationDocument.text(messages[i])
        messages[i]["streaming"] = .bool(true)
        // Preserve Continue when preflight refuses before a new terminal.
        messages[i]["error"] = nil
        messages[i]["stopped"] = nil
      } else {
        var row: O = [
          "id": .string(mid), "parentId": parent.map(V.string) ?? .null,
          "role": .string("assistant"), "model": .string(modelID), "content": .array([]),
          "createdAt": Self.now, "streaming": .bool(true),
        ]
        if let group { row["group"] = .string(group) }
        messages.append(row)
      }
      laneMessages.append((modelID, mid))
    }
    var fields = before.fields
    fields["messages"] = .array(messages.map(V.object))
    fields["leafId"] = .string(laneMessages[0].1)
    fields["updatedAt"] = Self.now
    if wasDraft {
      fields["title"] = .string(
        String(
          (text.isEmpty ? parts.first?["name"]?.string ?? "New conversation" : text).prefix(100)
        ).replacingOccurrences(of: "\n", with: " "))
    }
    var admitted = try ConversationDocument(fields: fields)
    for modelID in laneIDs {
      if canImagine(modelID), let mid = laneMessages.first(where: { $0.0 == modelID })?.1,
        let message = admitted.messages.first(where: { $0["id"]?.string == mid })
      {
        _ = try NativeImageGeneration.plan(
          document: admitted, message: message,
          caps: capability(modelID)["image_generation"]?.object ?? [:])
        continue
      }
      if let plan = NativeDocumentPlan.make(document: admitted, capability: capability(modelID)) {
        fields["activeDocId"] = .string(plan.sourceID)
        for part in plan.parts { try validateImageBudget([.object(part)], models: [modelID]) }
      } else {
        try validateImageBudget(parts, models: [modelID])
      }
    }
    admitted = try ConversationDocument(fields: fields)
    let graph = admitted.activeMessages.flatMap { $0["content"]?.array ?? [] }.last {
      $0["type"]?.string == "graph"
    }
    if let graph {
      graphArtifact = nil
      graphVisible = true
      await emit()
      graphGrounding = try await prepareGraph([
        "conversationId": .string(admitted.id), "graphSource": graph, "visibleGraph": .bool(true),
        "graphHistory": .array(Self.graphHistory(admitted)),
      ])
      guard !stopRequested else {
        throw ConversationFailure.invalid("Stopped before model execution; your draft is kept")
      }
    } else {
      graphGrounding = ""
    }
    try await save(
      admitted,
      consumingDraft: action == nil && !composerDraftRevision.isEmpty ? composerDraftRevision : nil)
    if action == nil {
      composerDraftRevision = ""
      composerDraftAttachments = []
    }
    // Stop may arrive while SQLite admission is awaiting its acknowledgement.
    document = admitted
    draft = false
    if stopRequested {
      for (_, mid) in laneMessages {
        try updateMessage(mid) {
          $0["streaming"] = .bool(false)
          $0["stopped"] = .bool(true)
        }
      }
      try await persist()
    } else {
      for (modelID, mid) in laneMessages {
        if !canImagine(modelID) { reducers[mid] = ResponseAccumulator() }
        starts[mid] = .now
        responseMetrics[mid] = NativeResponseMetrics()
        tasks[mid] = Task { [weak self] in
          await self?.run(
            modelID: modelID, messageID: mid, requestID: requestID, continuing: appendID != nil)
        }
      }
    }
    for id in ids { staged[id] = nil }
    if action == nil { draftText = "" }
    previewPart = nil
    documentPreview = nil
    return ["accepted": .bool(true), "requestId": .string(requestID)]
  }
  func messageAction(_ p: O, requestID: String) async throws -> O {
    guard var doc = document else { throw ConversationFailure.stale }
    let target = ConversationActionTarget(
      conversationID: try Self.id(p["conversationId"]), leafID: try Self.id(p["leafId"]),
      messageID: try Self.id(p["messageId"]))
    let action: ConversationMessageAction
    switch p["action"]?.string {
    case "edit":
      action = .edit(
        text: try Self.text(p["text"], limit: 128 * 1024),
        originalText: try Self.text(p["originalText"], limit: 128 * 1024))
    case "retry": action = .retry
    case "continue": action = .continueReply
    case "branch": action = .branch(targetID: try Self.id(p["targetId"]))
    default: throw ConversationFailure.invalid("Unknown message action")
    }
    let resolved = try doc.resolve(target, action: action)
    if case .branch = action {
      guard doc.stepSibling(of: target.messageID, delta: resolved.branchDelta) else {
        throw ConversationFailure.stale
      }
      try await save(doc)
      document = doc
      return ["accepted": .bool(true)]
    }
    return try await admit([:], requestID: requestID, action: (action, resolved))
  }
  func requestBody(modelID: String, messageID: String, continuing: Bool) async throws -> O {
    guard let document else { throw ConversationFailure.stale }
    let model = try model(modelID)
    let cap = capability(modelID)
    let fields = document.fields
    if !(fields["connectorIds"]?.array ?? []).isEmpty {
      connectors = try await transport.api("api/connectors").array?.compactMap(\.object) ?? []
    }
    let params = fields["params"]?.object ?? Self.defaultParams
    var input: [V] = []
    var originals: [String: String] = [:]
    let plan = contextPlan(document, modelID: modelID, continuing: continuing)
    let path = document.activeMessages.dropFirst(plan.from)
    if let item = plan.item { input.append(.object(item)) }
    for message in path {
      let role = message["role"]?.string ?? "user"
      if message["id"]?.string == messageID && !continuing { continue }
      if role == "assistant", message["group"] != nil, message["model"]?.string != modelID {
        continue
      }
      if let lane = message["lane"]?.string, lane != modelID { continue }
      if message["error"]?.string != nil { continue }
      if role == "assistant", !continuing, params["preserveThinking"]?.bool == true,
        cap["reasoning_preserve"]?.bool == true,
        let thought = message["reasoning"]?.string, !thought.isEmpty
      {
        input.append(
          .object([
            "type": .string("reasoning"), "summary": .array([]),
            "content": .array([
              .object(["type": .string("reasoning_text"), "text": .string(thought)])
            ]),
          ]))
      }
      var content: [V] = []
      for part in message["content"]?.array ?? [] {
        let type = part["type"]?.string ?? ""
        if type == "text" {
          content.append(
            .object([
              "type": .string(role == "assistant" ? "output_text" : "input_text"),
              "text": part["text"] ?? .string(""),
            ]))
        } else if type == "audio" || type == "graph" {
          content.append(
            .object([
              "type": .string("input_text"),
              "text": .string("[\(type): \(part["name"]?.string ?? "attachment")]"),
            ]))
        } else if type == "file" || type == "image" && cap["vision"]?.bool == true {
          let aid = part["attachmentId"]?.string ?? ""
          if type == "file", part["mime"]?.string == "application/pdf",
            part["pdfMode"]?.string != "text",
            cap["vision"]?.bool == true, cap["pdf"]?["raster"]?.bool != true,
            model["endpoint"] == nil
          {
            _ = try Self.id(.string(aid))
            let bytes = try await transport.bytes(
              "api/attachments/\(aid)", maximum: 100 * 1024 * 1024)
            content.append(contentsOf: try await rasterPDF(bytes, part.object!))
            continue
          }
          var source = originals[aid] ?? part["modelUrl"]?.string ?? part["dataUrl"]?.string
          if source == nil, ConversationDocument.validID(aid) {
            let bytes = try await transport.bytes(
              "api/attachments/\(aid)", maximum: 100 * 1024 * 1024)
            source =
              "data:\(part["mime"]?.string ?? "application/octet-stream");base64,\(bytes.base64EncodedString())"
            originals[aid] = source
          }
          guard let source else {
            throw ConversationFailure.invalid("The original attachment is missing")
          }
          var apiPart: O =
            type == "image"
            ? [
              "type": .string("input_image"), "image_url": .string(source),
              "detail": part["detail"] ?? .string("auto"),
            ]
            : [
              "type": .string("input_file"), "filename": part["name"] ?? .string("document"),
              "file_data": .string(source),
            ]
          apiPart["pages"] = part["pageRange"]
          apiPart["pdf_mode"] = part["pdfMode"]
          content.append(.object(apiPart))
        }
      }
      if !content.isEmpty {
        input.append(
          .object(["type": .string("message"), "role": .string(role), "content": .array(content)]))
      }
    }
    if continuing {
      input.append(
        .object([
          "type": .string("message"), "role": .string("user"),
          "content": .string(
            "Continue your previous reply from exactly where it left off. Do not repeat any text."),
        ]))
    }
    var body: O = [
      "model": model["wireModel"] ?? .string(modelID), "input": .array(input),
      "stream": .bool(true),
    ]
    for (local, wire) in [
      "temperature": "temperature", "topP": "top_p", "topK": "top_k", "minP": "min_p",
      "presencePenalty": "presence_penalty", "frequencyPenalty": "frequency_penalty",
      "repeatPenalty": "repeat_penalty", "seed": "seed",
    ] {
      if let value = params[local], value != .null { body[wire] = value }
    }
    let date = Date().formatted(.iso8601.year().month().day().dateSeparator(.dash))
    body["instructions"] = .string(
      [
        "Today's date: \(date) (user timezone: \(TimeZone.current.identifier)).",
        fields["systemPrompt"]?.string ?? "", graphGrounding,
        plan.summary.map(NativeContextPlan.summaryBlock) ?? "",
      ].filter { !$0.isEmpty }.joined(separator: "\n\n"))
    let style = cap["reasoning"]?.string
    if style == "effort" {
      body["reasoning"] = .object([
        "effort": params["thinking"]?.bool == false && cap["reasoning_off"]?.bool == true
          ? .string("none")
          : (params["reasoningEffort"]?.string?.isEmpty == false
            ? params["reasoningEffort"]! : cap["reasoning_default"] ?? .string("low"))
      ])
    } else if style == "toggle" {
      body["chat_template_kwargs"] = .object([
        "enable_thinking": .bool(params["thinking"]?.bool != false)
      ])
    }
    if cap["thinking_budget"]?.bool == true, let budget = params["thinkingBudget"], budget != .null
    {
      var reasoning = body["reasoning"]?.object ?? [:]
      reasoning["max_tokens"] = budget
      body["reasoning"] = .object(reasoning)
    }
    if params["preserveThinking"]?.bool == true, cap["reasoning_preserve"]?.bool == true {
      var kwargs = body["chat_template_kwargs"]?.object ?? [:]
      kwargs["preserve_thinking"] = .bool(true)
      body["chat_template_kwargs"] = .object(kwargs)
    }
    if plan.threshold > 0 {
      body["context_management"] = .array([
        .object([
          "type": .string("compaction"),
          "compact_threshold": .number(Decimal(plan.threshold)),
        ])
      ])
      body["truncation"] = .string("auto")
    }
    let extraction = NativeDocumentRequestOptions.apply(
      fields: fields, capability: cap, input: input, body: &body)
    if !extraction {
      let tools = await requestTools(modelID: modelID)
      if !tools.isEmpty { body["tools"] = .array(tools) }
    }
    if body["tools"]?.array?.isEmpty == false,
      let max = preferences["pk_max_tool_calls"]?.string.flatMap(Int.init), max > 0
    {
      body["max_tool_calls"] = .number(Decimal(max))
    }
    // Compare may trim shared history to the smallest window, but each lane's
    // output budget belongs to THAT model, not to its smallest neighbour.
    let window = cap["max_ctx"]?.integer ?? 0
    let ceiling = model["endpoint"] == nil ? nil : cap["default_max_output_tokens"]?.integer
    let prompt = NativeContextPlan.replyPrompt(
      document, body: body, model: modelID, pending: messageID)
    let limit = NativeReplyBudget.resolve(
      requested: maxTokens.integer, cloud: model["endpoint"] != nil, context: window,
      prompt: prompt.estimated, outputCeiling: ceiling, exact: prompt.exact)
    // 0 is the budget's refusal: under a useful reply's worth of room a send
    // would buy a fragment, and the provider's own refusal reads worse than ours.
    if let limit, limit <= 0 {
      throw ConversationFailure.invalid(
        "No room left for a reply: this conversation fills the model's \(window)-token "
          + "context window. Start a new chat, or shorten this one.")
    }
    body["max_output_tokens"] = limit.map { .number(Decimal($0)) }
    return body
  }
  func run(modelID: String, messageID: String, requestID: String, continuing: Bool) async {
    do {
      try Task.checkCancellation()
      let audio = document?.activeMessages.last(where: { $0["role"]?.string == "user" })?[
        "content"]?.array?.first { $0["type"]?.string == "audio" }?.object
      if canImagine(modelID) {
        guard !continuing else {
          throw ConversationFailure.invalid("Retry an image to make another picture")
        }
        try await generateImages(modelID: modelID, messageID: messageID)
      } else if let audio, canAudio(modelID) {
        try await transcribe(audio, modelID: modelID, messageID: messageID)
      } else if !continuing, try await runDocument(modelID: modelID, messageID: messageID) {
        // The per-page path owns its reductions, persistence and usage totals.
      } else {
        let body = try await requestBody(
          modelID: modelID, messageID: messageID, continuing: continuing)
        try Task.checkCancellation()
        let endpoint = try endpoint(model(modelID))
        let run = runSnapshot(modelID: modelID, body: body)
        try updateMessage(messageID) {
          if !continuing || $0["run"] == nil { $0["run"] = .object(run) }
        }
        let result = try await responseWithContextRetry(
          endpoint: endpoint, body: body,
          modelID: modelID, messageID: messageID, continuing: continuing)
        reducers[messageID] = result
        flushResponses()
        try updateMessage(messageID) { message in
          message["response"] = result.terminal.map(V.object)
          message["ocr"] = NativeOCRMetadata.stored(result.terminal?["ocr"])
          if result.status == "completed" { message["incomplete"] = nil }
          if continuing, var run = message["run"]?.object {
            run["nativeContext"] = nil
            message["run"] = .object(run)
          }
          if result.status == "incomplete" {
            message["incomplete"] = .string(
              result.terminal?["incomplete_details"]?["reason"]?.string == "max_output_tokens"
                ? "length" : "incomplete")
          }
          if result.status == "failed" {
            if let error = result.terminal?["error"]?.object,
              error["code"] != nil || error["metadata"] != nil
            {
              message["error"] = .string(
                (try? JSONEncoder().encode(error)).map { String(decoding: $0, as: UTF8.self) }
                  ?? result.failure ?? "Model generation failed")
            } else {
              message["error"] = .string(result.failure ?? "Model generation failed")
            }
          }
          let seconds = starts[messageID].map { Self.seconds($0.duration(to: .now)) } ?? 0
          message["usage"] = responseMetrics[messageID]?.usage(
            result.terminal, seconds: seconds, previous: continuing ? message["usage"]?.object : nil
          ).map(V.object)
        }
      }
    } catch {
      flushResponses()
      let cancelled = Task.isCancelled || error is CancellationError
      try? updateMessage(messageID) { message in
        if cancelled {
          message["stopped"] = .bool(true)
        } else {
          message["error"] = .string(error.localizedDescription)
        }
      }
    }
    try? updateMessage(messageID) { $0["streaming"] = .bool(false) }
    imagePreviews[messageID] = nil
    reducers[messageID] = nil
    starts[messageID] = nil
    continuationPrefixes[messageID] = nil
    responseMetrics[messageID] = nil
    requestEvents.remove(messageID)
    // Cancellation of a stream does not cancel saving its partial response.
    do { try await persist() } catch {
      self.error = "The reply is in memory but could not be saved: \(error.localizedDescription)"
    }
    tasks[messageID] = nil
    if tasks.isEmpty, let document {
      let replies = document.activeMessages.filter { $0["role"]?.string == "assistant" }
      completedTurn = .object([
        "id": .string(requestID), "conversationId": .string(document.id),
        "state": .string(
          saveFailed || replies.contains { $0["error"] != nil }
            ? "failed" : stopRequested ? "stopped" : "completed"),
      ])
      try? await refreshArtifacts()
      if !preparingQuit, canChat(modelID), preferenceBool("auto_title", fallback: true),
        document.fields["titleSource"] == nil,
        document.activeMessages.filter({ $0["role"]?.string == "user" }).count == 1,
        !replies.contains(where: { $0["error"] != nil || $0["stopped"]?.bool == true })
      {
        titleEpoch += 1
        let epoch = titleEpoch
        titleTask = Task { [weak self] in
          try? await self?.generateTitle(document.id, automatic: true)
          await self?.titleFinished(epoch)
        }
      }
    }
    if tasks.isEmpty { scheduleCompaction() }
    schedulePublish()
  }
  private func titleFinished(_ epoch: Int) {
    guard titleEpoch == epoch else { return }
    titleTask = nil
    schedulePublish()
  }
  func receive(_ event: O, messageID: String) async throws {
    try Task.checkCancellation()
    requestEvents.insert(messageID)
    guard reducers[messageID] != nil else { throw ConversationFailure.stale }
    try reducers[messageID]!.apply(event)
    if let start = starts[messageID] {
      responseMetrics[messageID]?.observe(event, seconds: Self.seconds(start.duration(to: .now)))
    }
    if let item = event["item"]?.object {
      try applyOutputItem(
        item, messageID: messageID, done: event["type"]?.string == "response.output_item.done")
    }
    if event["type"]?.string == "response.output_item.done", let item = event["item"],
      item["type"]?.string == "mcp_call", Self.toolArtifactID(item) != nil
    {
      scheduleArtifactRefresh()
    }
    if let output = event["response"]?["output"]?.array {
      for item in output {
        if let item = item.object { try applyOutputItem(item, messageID: messageID, done: true) }
      }
    }
    if ContinuousClock.now - lastCheckpoint >= .seconds(2) {
      flushResponses()
      lastCheckpoint = .now
      do { try await persist() } catch {
        self.error = "Could not checkpoint the current reply: \(error.localizedDescription)"
      }
    }
    schedulePublish()
  }
  func flushResponses() {
    flushDocumentRuns()
    for (id, response) in reducers {
      let prefix = continuationPrefixes[id] ?? ""
      try? updateMessage(id) {
        $0["content"] = .array([
          .object(["type": .string("text"), "text": .string(prefix + response.text)])
        ])
        $0["reasoning"] = .string(response.reasoning)
      }
    }
  }
  static func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
  }
  func generateTitle(_ id: String, automatic: Bool = false) async throws {
    var doc = id == document?.id ? document! : try await transport.loadConversation(id)
    let modelID = doc.fields["model"]?.string ?? ""
    let model = try model(modelID)
    let transcript = doc.activeMessages.map {
      "\($0["role"]?.string ?? ""): \(ConversationDocument.text($0))"
    }.joined(separator: "\n")
    guard !transcript.isEmpty else {
      throw ConversationFailure.invalid("Send a message before generating a title")
    }
    let result = try await transport.responses(
      endpoint: endpoint(model),
      body: [
        "model": model["wireModel"] ?? .string(modelID), "stream": .bool(true),
        "max_output_tokens": .number(128),
        "instructions": .string(
          "Return only a concise conversation title on one line, without quotes. Do not answer the conversation."
        ), "input": .string(String(transcript.prefix(12000))),
      ]
    ) { _ in }
    try Task.checkCancellation()
    if automatic {
      guard document?.id == id, document?.fields["updatedAt"] == doc.fields["updatedAt"],
        document?.fields["title"] == doc.fields["title"], !mutating
      else { return }
    }
    guard result.status == "completed", let title = result.text.split(separator: "\n").first,
      !title.isEmpty
    else { throw ConversationFailure.invalid("The model did not return a title") }
    try doc.rename(
      String(title.prefix(512)).trimmingCharacters(in: CharacterSet(charactersIn: "\"")))
    if automatic {
      var fields = doc.fields
      fields["titleSource"] = .string("model")
      fields["titleModel"] = .string(modelID)
      doc = try .init(fields: fields)
    }
    try await save(doc)
    if document?.id == id { document = doc }
    schedulePublish()
  }
}
