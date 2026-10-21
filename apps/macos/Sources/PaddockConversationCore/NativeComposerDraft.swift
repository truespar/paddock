import Foundation

extension NativeStudioRuntime {
  func loadHistory() async throws {
    let saved = try await transport.listConversations()
    let drafts =
      try await transport.api("api/conversation-drafts").array?.compactMap(\.object) ?? []
    let ids = Set(drafts.compactMap { $0["id"]?.string })
    latestComposerDraftID =
      drafts.max {
        ($0["updatedAt"]?.double ?? 0) < ($1["updatedAt"]?.double ?? 0)
      }?["id"]?.string
    history = saved.map { row in
      var row = row
      row["composerDraft"] = .bool(row["id"]?.string.map(ids.contains) == true)
      return row
    }
    let savedIDs = Set(saved.compactMap { $0["id"]?.string })
    history += drafts.filter { $0["id"]?.string.map { !savedIDs.contains($0) } == true }
  }

  /// The caller serializes acknowledged saves and drains them before navigation
  /// or admission. The ID rejects an old editor callback after a chat switch;
  /// the SQLite revision also rejects writes from another client/window.
  func saveComposerDraft(_ input: O) async throws -> O {
    guard let document, input["conversationId"]?.string == document.id else {
      throw ConversationFailure.stale
    }
    let text = try Self.text(input["text"], limit: 128 * 1024)
    guard let displays = input["attachments"]?.array, displays.count <= 32 else {
      throw ConversationFailure.invalid("Invalid draft attachments")
    }
    var ids = Set<String>()
    let attachments: [V] = try displays.map { display in
      let id = try Self.id(display["id"])
      guard ids.insert(id).inserted, let part = staged[id] else {
        throw ConversationFailure.invalid("Wait for all attachments to finish uploading")
      }
      return .object(["display": display, "part": .object(part)])
    }
    // Store configuration only, never copy the existing transcript into drafts.
    var seed = document.fields
    seed["messages"] = .array([])
    seed["leafId"] = .null
    if draft {
      let title = text.trimmingCharacters(in: .whitespacesAndNewlines)
      seed["title"] = .string(
        String(
          (title.isEmpty ? displays.first?["name"]?.string ?? "New conversation" : title).prefix(
            100)))
    }
    let reply = try await transport.api(
      "api/conversation-drafts/\(document.id)", method: "PUT",
      body: .object([
        "revision": .string(composerDraftRevision), "text": .string(text),
        "attachments": .array(attachments), "conversation": .object(seed),
      ]))
    guard self.document?.id == document.id, let revision = reply["revision"]?.string else {
      throw ConversationFailure.stale
    }
    composerDraftRevision = revision
    composerDraftAttachments = displays
    draftText = text
    if draft { self.document = try .init(fields: seed) }
    // Update just this summary, without refetching every transcript or draft.
    if let index = history.firstIndex(where: { $0["id"]?.string == document.id }) {
      if draft {
        history.remove(at: index)
      } else {
        history[index]["composerDraft"] = .bool(!revision.isEmpty)
      }
    }
    if draft, !revision.isEmpty {
      history.append([
        "id": .string(document.id), "title": seed["title"]!, "model": seed["model"]!,
        "updatedAt": Self.now, "composerDraft": .bool(true),
      ])
    }
    return ["revision": .string(revision)]
  }

  func draftableConversation(_ id: String, savedDraft: O?) async throws -> (
    document: ConversationDocument, unsent: Bool
  ) {
    do { return (try await transport.loadConversation(id), false) } catch ConversationFailure.http(
      404)
    {
      guard let fields = savedDraft?["conversation"]?.object else {
        throw ConversationFailure.http(404)
      }
      let value = try ConversationDocument(fields: fields)
      guard value.id == id, value.messages.isEmpty else { throw ConversationFailure.stale }
      return (value, true)
    }
  }

  func openConversationAndDraft(_ id: String) async throws {
    let savedDraft = try await transport.api("api/conversation-drafts/\(id)").object
    let source = try await draftableConversation(id, savedDraft: savedDraft)
    var loaded = source.document
    loaded.migrateTree()
    var fields = loaded.fields
    fields["messages"] = .array(
      loaded.messages.map { raw in
        var value = raw
        if value["streaming"]?.bool == true {
          value["streaming"] = .bool(false)
          value["stopped"] = .bool(true)
        }
        return .object(NativeDocumentRunState.recover(value))
      })
    let entries = savedDraft?["attachments"]?.array ?? []
    var restored: [String: O] = [:]
    for entry in entries {
      let id = try Self.id(entry["display"]?["id"])
      guard let part = entry["part"]?.object, part["attachmentId"]?.string == id,
        restored[id] == nil
      else { throw ConversationFailure.invalid("Invalid saved attachment") }
      restored[id] = part
    }
    // Replace active state only after every load/validation succeeds.
    document = try .init(fields: fields)
    draft = source.unsent
    composerDraftRevision = savedDraft?["revision"]?.string ?? ""
    composerDraftAttachments = entries.compactMap { $0["display"] }
    draftText = savedDraft?["text"]?.string ?? ""
    staged = restored
    documentSelectionRevision += 1
    graphSelectionRevision += 1
    documentDisplay = NativeOCRDisplayCache()
    previewPart = nil
    documentPreview = nil
    graphVisible = false
    graphArtifact = nil
    graphGrounding = ""
    try await refreshArtifacts()
  }
}
