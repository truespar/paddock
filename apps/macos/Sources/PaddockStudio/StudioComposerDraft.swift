import Foundation
import PaddockClient
import PaddockConversationCore

extension StudioWorkspace {
  public var canNavigateConversation: Bool {
    !preparingQuit && !switchingConversation && !sending && !messageMutation && !microphoneBusy
      && (state?.canNavigateConversation ?? !busy)
      && !uploading && !hasMessageEdit && attachments.allSatisfy(\.ready) && !hasArtifactEdits
  }

  func composerChanged() {
    guard !applyingComposer else { return }
    composerEditRevision &+= 1
    composerDebounce?.cancel()
    guard ready, composerIdentity != nil else { return }
    composerDebounce = Task { [weak self] in
      do { try await Task.sleep(for: .milliseconds(600)) } catch { return }
      guard let self, !self.busy, !self.uploading else { return }
      self.composerDebounce = nil
      do { try await self.flushComposerDraft() } catch { self.error = error.localizedDescription }
    }
  }

  /// A navigation/send boundary drains in-flight saves and newer edits. An
  /// error leaves the editor and attachments untouched. Never rewrite a transcript.
  public func flushComposerDraft() async throws {
    try await flushComposerDraft(allowClosed: false)
  }

  func flushComposerDraft(allowClosed: Bool) async throws {
    composerDebounce?.cancel()
    composerDebounce = nil
    guard ready || allowClosed, let identity = composerIdentity, let runtime else { return }
    while composerSavedRevision != composerEditRevision {
      if let pending = composerSave {
        try await pending.value
        guard composerIdentity == identity else { return }
        continue
      }
      guard allowClosed || (!uploading && attachments.allSatisfy(\.ready)) else {
        throw ManagerError.core("Finish or remove the pending attachment before leaving this chat.")
      }
      let revision = composerEditRevision
      // Explicit quit may cancel an unfinished upload, but must still retain
      // the typed text and every attachment already safely owned by Rust.
      let retained = allowClosed ? attachments.filter(\.ready) : attachments
      let displays = try JSONDecoder().decode(
        [StudioValue].self, from: JSONEncoder().encode(retained))
      let payload: [String: StudioValue] = [
        "conversationId": .string(identity), "text": .string(composerText),
        "attachments": .array(displays),
      ]
      let native = try JSONDecoder().decode(
        [String: ConversationValue].self, from: JSONEncoder().encode(payload))
      let task = Task { [weak self] in
        guard let self else { return }
        defer { self.composerSave = nil }
        _ = try await runtime.command("saveDraft", native)
        guard self.composerIdentity == identity else { return }
        self.composerSavedRevision = revision
        self.conversationPositions.resave(identity)
      }
      composerSave = task
      try await task.value
      guard composerIdentity == identity else { return }
    }
  }

  func restoreComposer(_ draft: StudioState.ComposerDraft?) {
    guard let draft, draft.id != composerIdentity else { return }
    composerDebounce?.cancel()
    composerDebounce = nil
    applyingComposer = true
    composerIdentity = draft.id
    composerText = draft.text
    attachments = draft.attachments
    composerEditRevision = 0
    composerSavedRevision = 0
    applyingComposer = false
  }
}
