import Foundation

extension NativeStudioRuntime {
  public func command(_ kind: String, _ p: O = [:], id: String = UUID().uuidString) async throws
    -> O
  {
    guard !closed else { throw ConversationFailure.closed }
    if let receipt = receipts[id] { return receipt }
    // Read/display commands and Stop remain responsive during generation.
    let concurrent = [
      "draft", "stop", "refresh", "artifactsPane", "tools", "toolQuery", "historyFilter", "preview",
      "openDocument", "closePreview", "graphPanel", "graphArtifact", "toolApproval",
      "preferencesGet", "promptList", "promptGet", "instructionsGet", "transcriptExport",
      "composerSize", "renderer", "quote",
    ]
    if !concurrent.contains(kind) {
      try idle()
      cancelCompaction()
      mutating = true
    }
    let ownsMutation = !concurrent.contains(kind)
    defer {
      if ownsMutation { mutating = false }
      schedulePublish()
    }
    var result: O = [:]
    do {
      switch kind {
      case "refresh":
        try await refreshModels()
        history = try await transport.listConversations()
        try await refreshArtifacts()
      case "draft": draftText = try Self.text(p["text"], limit: 128 * 1024)
      case "composerSize", "renderer": break  // Layout/rendering is wholly native.
      case "quote": result["text"] = .string("")  // The selected TextKit view owns selection.
      case "newChat":
        try newDocument()
        documentDisplay = NativeOCRDisplayCache()
      case "openEndpoint":
        // Resolve against the freshly fetched fleet in this actor, not the
        // SwiftUI projection (which is deliberately coalesced across frames).
        // Validate before replacing the conversation so a failed readiness
        // check cannot discard the current document or staged attachments.
        guard let port = p["port"]?.integer, (1...65535).contains(port),
          p["port"]?.double == Double(port)
        else { throw ConversationFailure.invalid("Invalid model endpoint") }
        try await refreshModels()
        guard let target = models.first(where: { $0["port"]?.integer == port }),
          let modelID = target["id"]?.string, target["status"]?.string == "ok"
        else {
          throw ConversationFailure.invalid(
            "This model is not ready. Check Settings > Instances.")
        }
        guard canChat(modelID) || canAudio(modelID) || canImagine(modelID) else {
          throw ConversationFailure.invalid(
            target["kind"]?.string == "image"
              ? "Could not load this model’s image capabilities. Try opening Studio again."
              : "This endpoint does not support chat, transcription or image generation.")
        }
        try newDocument()
        try change { $0["model"] = .string(modelID) }
        documentDisplay = NativeOCRDisplayCache()
      case "open":
        documentSelectionRevision += 1
        graphSelectionRevision += 1
        let id = try Self.id(p["id"])
        var loaded = try await transport.loadConversation(id)
        loaded.migrateTree()
        // An app/process interruption must never leave old rows spinning forever.
        var fields = loaded.fields
        fields["messages"] = .array(
          loaded.messages.map { m in
            var value = m
            if value["streaming"]?.bool == true {
              value["streaming"] = .bool(false)
              value["stopped"] = .bool(true)
            }
            return .object(NativeDocumentRunState.recover(value))
          })
        document = try .init(fields: fields)
        documentDisplay = NativeOCRDisplayCache()
        draft = false
        staged = [:]
        previewPart = nil
        documentPreview = nil
        graphVisible = false
        graphArtifact = nil
        graphGrounding = ""
        try await refreshArtifacts()
      case "models":
        let ids = p["ids"]?.array?.compactMap(\.string) ?? []
        guard (1...4).contains(ids.count), Set(ids).count == ids.count else {
          throw ConversationFailure.invalid("Choose one to four distinct models")
        }
        for id in ids { _ = try model(id) }
        guard ids.allSatisfy(canChat) || ids.allSatisfy(canAudio) || ids.allSatisfy(canImagine)
        else {
          throw ConversationFailure.invalid(
            "Compare models must share chat, speech or image generation")
        }
        let before = document
        try change {
          $0["model"] = .string(ids[0])
          $0["compareModels"] = ids.count > 1 ? .array(ids.map(V.string)) : nil
        }
        do { try await persist() } catch {
          document = before
          throw error
        }
      case "settings", "samplerDefaults":
        let before = document
        try change { fields in
          if kind == "samplerDefaults" {
            fields["params"] = .object(Self.defaultParams)
            return
          }
          for (key, value) in p {
            switch key {
            case "imageParams":
              if value == .null {
                fields[key] = nil
                continue
              }
              guard let patch = value.object else {
                throw ConversationFailure.invalid("Invalid image settings")
              }
              let merged = (fields[key]?.object ?? NativeImageGeneration.defaults).merging(patch) {
                _, new in new
              }
              try NativeImageGeneration.validate(merged)
              fields[key] = .object(merged)
            case "params":
              guard let patch = value.object else {
                throw ConversationFailure.invalid("Invalid sampling settings")
              }
              try Self.validateSampling(patch)
              fields[key] = .object(
                (fields[key]?.object ?? Self.defaultParams).merging(patch) { _, new in new })
            case "systemPrompt": fields[key] = .string(try Self.text(value, limit: 128 * 1024))
            case "webSearchEnabled", "forensicsEnabled", "fileMetadataEnabled", "ocrRegions":
              guard value.bool != nil else {
                throw ConversationFailure.invalid("Invalid preference switch")
              }
              fields[key] = value
            case "ocrMode", "audioLanguage": fields[key] = .string(try Self.text(value, limit: 128))
            case "toolSelection", "connectorIds": fields[key] = value
            default: throw ConversationFailure.invalid("Unknown conversation setting: \(key)")
            }
          }
        }
        do { try await persist() } catch {
          document = before
          throw error
        }
      case "stage":
        let attachmentID = try Self.id(p["id"])
        guard staged.count < 32 || staged[attachmentID] != nil else {
          throw ConversationFailure.invalid("Attach at most 32 files per turn")
        }
        let part = try Self.attachment(p)
        staged[attachmentID] = part
        result["attachment"] = .object(part)
      case "removeAttachment":
        let attachmentID = try Self.id(p["id"])
        staged[attachmentID] = nil
        if previewPart?["attachmentId"]?.string == attachmentID {
          documentSelectionRevision += 1
          previewPart = nil
          documentPreview = nil
        }
      case "preview", "openDocument":
        let conversationID = document?.id
        documentSelectionRevision += 1
        let selection = documentSelectionRevision
        let part: O?
        var selectedSource: String?
        if kind == "preview" {
          part = staged[try Self.id(p["id"])]
        } else {
          let mid = try Self.id(p["messageId"])
          let aid = try Self.id(p["attachmentId"])
          var selected =
            document?.activeMessages.first { $0["id"]?.string == mid }?["content"]?.array?
            .enumerated().first { index, value in
              value["attachmentId"]?.string == aid
                || NativeImageGeneration.pictureID(
                  value.object ?? [:], messageID: mid, index: index) == aid
            }?.element.object
          // Only the transient preview descriptor needs an ID for an inline
          // picture. The persisted web-compatible part remains unchanged.
          if selected?["attachmentId"]?.string != aid {
            selected?["attachmentId"] = .string(aid)
          }
          part = selected
          selectedSource = mid
        }
        guard let part, Self.documentBadge(part) != nil || Self.audioBadge(part) != nil else {
          throw ConversationFailure.invalid("This attachment has no preview")
        }
        if let selectedSource, part["type"]?.string != "audio" {
          let previous = document?.fields["activeDocId"]
          try change { $0["activeDocId"] = .string(selectedSource) }
          do { try await persist() } catch {
            if document?.id == conversationID,
              documentSelectionRevision == selection,
              document?.fields["activeDocId"] == .string(selectedSource)
            {
              try? change { $0["activeDocId"] = previous }
            }
            throw error
          }
        }
        guard document?.id == conversationID, documentSelectionRevision == selection else {
          throw ConversationFailure.stale
        }
        previewPart = part
        documentPreview = [
          "id": document?.fields["id"] ?? .null,
          "title": document?.fields["title"] ?? .string("Document"),
        ]
        let attachmentID = part["attachmentId"]!
        documentPreview?["messages"] = .array([
          .object([
            "id": attachmentID, "parentId": .null, "role": .string("user"),
            "content": .array([.object(part)]), "createdAt": Self.now,
          ])
        ])
        documentPreview?["leafId"] = attachmentID
        documentPreview?["activeDocId"] = attachmentID
      case "closePreview":
        documentSelectionRevision += 1
        previewPart = nil
        documentPreview = nil
      case "graphPanel":
        graphSelectionRevision += 1
        graphVisible = p["open"]?.bool == true
        // This command selects the attached graph, not a previously opened
        // generated artifact. The independent left document stays open.
        graphArtifact = nil
      case "artifactsPane":
        guard let open = p["open"]?.bool, !draft, let cid = document?.id,
          p["conversationId"]?.string == cid
        else { throw ConversationFailure.stale }
        let previous = document?.fields["artifactsPaneOpen"]
        let previousGraph = graphVisible
        try change { $0["artifactsPaneOpen"] = .bool(open) }
        if open {
          graphSelectionRevision += 1
          graphVisible = false
        }
        do { try await persist() } catch {
          if document?.id == cid {
            try? change { $0["artifactsPaneOpen"] = previous }
            graphVisible = previousGraph
          }
          throw error
        }
      case "graphArtifact":
        let conversationID = document?.id
        graphSelectionRevision += 1
        let selection = graphSelectionRevision
        let aid = try Self.id(p["id"])
        guard
          let meta = artifacts.first(where: {
            $0["id"]?.string == aid && $0["kind"]?.string == "graph"
          })
        else { throw ConversationFailure.stale }
        let body = try await transport.bytes(
          "api/artifacts/\(aid)/content", maximum: 4 * 1024 * 1024)
        guard document?.id == conversationID, graphSelectionRevision == selection else {
          throw ConversationFailure.stale
        }
        graphArtifact = [
          "id": .string(aid), "title": meta["title"] ?? .string("Graph"),
          "body": .string(String(decoding: body, as: UTF8.self)),
        ]
        graphVisible = true
      case "send": result = try await admit(p, requestID: id)
      case "stop":
        stopRequested = true
        for task in tasks.values { task.cancel() }
      case "messageAction": result = try await messageAction(p, requestID: id)
      case "renameChat", "pinChat":
        let cid = try Self.id(p["id"])
        var doc = cid == document?.id ? document! : try await transport.loadConversation(cid)
        if kind == "renameChat" {
          try doc.rename(try Self.text(p["title"], limit: 512))
        } else {
          doc.setPinned(doc.fields["pinned"]?.bool != true)
        }
        try await save(doc)
        if cid == document?.id { document = doc }
      case "deleteChats":
        guard let ids = p["ids"]?.array, !ids.isEmpty, ids.count <= 100 else {
          throw ConversationFailure.invalid("Select conversations to delete")
        }
        for value in ids {
          let cid = try Self.id(value)
          _ = try await transport.api("api/conversations/\(cid)", method: "DELETE")
          history.removeAll { $0["id"]?.string == cid }
          if document?.id == cid { try newDocument() }
        }
      case "generateTitle": try await generateTitle(try Self.id(p["id"]))
      case "historyFilter":
        search = try Self.text(p["search"] ?? .string(""), limit: 512)
        sort = p["sort"]?.string ?? "newest"
        page = (p["page"]?.integer ?? 0) + 1
      case "preferencesGet": result["preferences"] = .object(preferencePresentation())
      case "preferencesSave": result = try await savePreferences(p)
      case "autoTitle":
        result = try await savePreferences([
          "changes": .object(["autoTitle": p["enabled"] ?? .null]),
          "expected": .object(["autoTitle": .bool(preferenceBool("auto_title", fallback: true))]),
        ])
      case "transcriptMarks":
        result = try await savePreferences([
          "changes": .object(["markUnsure": p["enabled"] ?? .null]),
          "expected": .object(["markUnsure": .bool(preferenceBool("mark_unsure", fallback: true))]),
        ])
      case "promptList", "promptGet", "promptSave", "promptDelete":
        result = try await promptCommand(kind, p)
      case "instructionsGet":
        try await refreshTools()
        result["instructions"] = .object([
          "conversationId": .string(document!.id),
          "body": document!.fields["systemPrompt"] ?? .string(""),
          "blocks": .array(
            toolGroups.compactMap { group in
              guard let text = group["instructions"]?.string, !text.isEmpty else { return nil }
              return .object([
                "label": group["label"] ?? .string("Tool instructions"), "text": .string(text),
              ])
            }),
        ])
      case "instructionsApply":
        guard p["conversationId"]?.string == document?.id,
          p["expected"] == document?.fields["systemPrompt"]
        else { throw ConversationFailure.stale }
        let body = try Self.text(p["body"], limit: 128 * 1024)
        let before = document
        try change { $0["systemPrompt"] = .string(body) }
        do { try await persist() } catch {
          document = before
          throw error
        }
      case "tools": try await refreshTools()
      case "toolQuery": toolQuery = try Self.text(p["query"], limit: 256)
      case "toolPicker": try await changeTools(p)
      case "toolApproval": try await approveTool(p)
      case "transcriptExport": result = try transcriptExport(p)
      case "identifySpeakers": try await identifySpeakers(p)
      default: throw ConversationFailure.invalid("Unsupported native command: \(kind)")
      }
      error = ""
      receipts[id] = result
      receiptOrder.append(id)
      if receiptOrder.count > 128 { receipts[receiptOrder.removeFirst()] = nil }
      return result
    } catch {
      self.error = error.localizedDescription
      throw error
    }
  }
  static func text(_ value: V?, limit: Int = 1024) throws -> String {
    guard let text = value?.string, text.utf8.count <= limit else {
      throw ConversationFailure.invalid("Invalid or oversized text field")
    }
    return text
  }
  static func id(_ value: V?) throws -> String {
    let id = try text(value, limit: 128)
    guard ConversationDocument.validID(id) else {
      throw ConversationFailure.invalid("Invalid identity")
    }
    return id
  }
  static func validateSampling(_ patch: O) throws {
    let ranges: [String: ClosedRange<Double>] = [
      "temperature": 0...2, "topP": 0...1, "topK": 0...100000,
      "minP": 0...1, "presencePenalty": -2...2, "frequencyPenalty": -2...2,
      "repeatPenalty": 0...100,
      "seed": -9_007_199_254_740_991...9_007_199_254_740_991, "thinkingBudget": 0...1_048_576,
    ]
    for (key, value) in patch {
      if let range = ranges[key] {
        guard value == .null || value.double.map({ range.contains($0) }) == true else {
          throw ConversationFailure.invalid("Invalid \(key)")
        }
      } else if ["thinking", "preserveThinking"].contains(key) {
        guard value.bool != nil else {
          throw ConversationFailure.invalid("Invalid reasoning switch")
        }
      } else if key == "reasoningEffort" {
        _ = try text(value, limit: 32)
      } else if key == "stop", let values = value.array, values.count <= 16 {
        for value in values { _ = try text(value, limit: 1024) }
      } else {
        throw ConversationFailure.invalid("Unknown sampling parameter")
      }
    }
  }
  static func attachment(_ metadata: O) throws -> O {
    let id = try id(metadata["id"])
    let name = try text(metadata["name"])
    let mime = try text(metadata["mime"], limit: 128)
    guard let size = metadata["size"]?.integer, (0...100 * 1024 * 1024).contains(size),
      !mime.contains("\n"), !mime.contains("\r")
    else { throw ConversationFailure.invalid("Invalid attachment metadata") }
    let kind =
      name.lowercased().hasSuffix(".tvdb")
      ? "graph"
      : mime.hasPrefix("audio/") || name.lowercased().hasSuffix(".webm")
        ? "audio" : mime.hasPrefix("image/") ? "image" : "file"
    var part: O = [
      "type": .string(kind), "attachmentId": .string(id), "name": .string(name),
      "mime": .string(mime), "size": .number(Decimal(size)),
    ]
    for key in ["pages", "width", "height", "thumbUrl", "durationS"] { part[key] = metadata[key] }
    if kind == "image" { part["detail"] = .string("auto") }
    return part
  }
  func refreshArtifacts() async throws {
    artifactRefreshEpoch += 1
    let epoch = artifactRefreshEpoch
    guard let document, !draft else {
      artifacts = []
      return
    }
    let value = try await transport.api("api/conversations/\(document.id)/artifacts")
    guard !closed, epoch == artifactRefreshEpoch, self.document?.id == document.id, !self.draft
    else { return }
    artifacts = value["artifacts"]?.array ?? value.array ?? []
  }
  func scheduleArtifactRefresh() {
    guard artifactRefreshTask == nil else { return }
    let conversation = document?.id
    artifactRefreshTask = Task { [weak self] in
      do { try await Task.sleep(for: .milliseconds(100)) } catch { return }
      guard let self else { return }
      await self.refreshPublishedArtifacts(conversation: conversation)
    }
  }
  private func refreshPublishedArtifacts(conversation: String?) async {
    artifactRefreshTask = nil
    guard !closed, document?.id == conversation else { return }
    try? await refreshArtifacts()
    schedulePublish()
  }
}
