import Foundation
import Observation
import PaddockClient
import PaddockConversationCore

/// One stream/writer/transport per retained conversation, never one mutable
/// transcript shared by unrelated requests. Completed background sessions are
/// released only after their writers are durable; SQLite owns idle history.
@MainActor @Observable final class StudioConversationSession {
  let id: UUID
  let runtime: NativeStudioRuntime
  let transport: NativeConversationTransport
  let decoder = StudioPresentationDecoder()
  var state: StudioState?
  var historyRow: [String: ConversationValue]?
  var revision = 0
  init(id: UUID, runtime: NativeStudioRuntime, transport: NativeConversationTransport) {
    self.id = id
    self.runtime = runtime
    self.transport = transport
  }
}

extension StudioWorkspace {
  public var hasBackgroundWork: Bool {
    sessions.values.contains { $0.id != activeSession && $0.state?.busy == true }
  }

  func makeSession() throws -> StudioConversationSession {
    guard let transportFactory else { throw ManagerError.core("Native Studio is not ready") }
    let transport = try transportFactory()
    let id = UUID()
    let core = NativeStudioRuntime(
      transport: transport,
      prepareGraph: { [weak self] fields in
        guard let self else { throw CancellationError() }
        return try await self.prepareGraph(fields)
      },
      rasterPDF: { bytes, part in try await NativePDFInput.shared.parts(bytes, metadata: part) },
      openPDF: { bytes in try await NativePDFInput.shared.source(bytes) },
      publish: { [weak self] fields in await self?.receiveSession(fields, id: id) })
    let session = StudioConversationSession(id: id, runtime: core, transport: transport)
    sessions[id] = session
    return session
  }

  func receiveSession(_ fields: [String: ConversationValue], id: UUID) async {
    guard !closed, let session = sessions[id], let revision = fields["revision"]?.integer,
      revision > session.revision
    else { return }
    do {
      let value = try await session.decoder.decode(fields)
      if id == activeSession, let cid = value.composerDraft?.id {
        try await conversationPositions.prepare(cid)
      }
      guard !closed, sessions[id] === session, revision > session.revision else { return }
      session.revision = revision
      session.state = value
      session.historyRow = fields["sessionSummary"]?.object
      if id == activeSession {
        apply(value)
      } else {
        // Activity keeps its source conversation identity; never apply a hidden
        // transcript, error, attachment or artifact selection to the foreground.
        onPresentation?(value)
      }
      await synchronizeSessionHistory()
      if id != activeSession, !switchingConversation, !preparingQuit {
        await releaseIdleSession(id)
      }
    } catch {
      if id == activeSession, !closed, !(error is CancellationError) {
        self.error = "Invalid native presentation: \(error)"
      }
    }
  }

  func synchronizeSessionHistory() async {
    let rows = sessions.values.compactMap(\.historyRow)
    let busy = Set(
      sessions.values.compactMap { session -> String? in
        session.state?.busy == true ? session.state?.conversation?.id : nil
      })
    await runtime?.mergeSessionHistory(rows, busy: busy)
  }

  func releaseIdleSession(_ id: UUID) async {
    guard id != activeSession, let session = sessions[id],
      await session.runtime.canReleaseConversation(),
      id != activeSession, !switchingConversation, !preparingQuit,
      sessions[id] === session
    else { return }
    sessions[id] = nil
    await session.runtime.close()
    await session.decoder.close()
    await synchronizeSessionHistory()
  }

  func selectSession(_ session: StudioConversationSession) {
    activeSession = session.id
    runtime = session.runtime
    nativeTransport = session.transport
    receivedRevision = 0
    error = nil
  }

  func navigateSession(_ kind: String, _ payload: [String: ConversationValue]) async throws {
    guard let previous = runtime, await previous.canLeaveConversation() else {
      throw ManagerError.core("Finish the current operation before switching conversations.")
    }
    let filter = state?.library
    let target = kind == "open" ? payload["id"]?.string : nil
    if target == composerIdentity { return }
    let retained = sessions.values.first { $0.state?.composerDraft?.id == target && target != nil }
    let session: StudioConversationSession
    if let retained {
      session = retained
      // Reattach the live actor, never hydrate a streaming SQLite checkpoint as
      // a stopped conversation. Refresh only fleet/history/artifact metadata.
      _ = try? await session.runtime.command("refresh")
    } else {
      guard sessions.count < 8 else {
        throw ManagerError.core(
          "Finish a response or resolve its save error before opening another chat.")
      }
      session = try makeSession()
      // Bootstrap a candidate without exposing its intermediate empty document.
      await session.runtime.setForeground(false)
      do {
        try await session.runtime.start(resumeDraft: false)
        if kind != "newChat" {
          _ = try await session.runtime.command(kind, payload)
        } else if let model = state?.selectedModels.first,
          session.state?.models.contains(where: { $0.id == model && $0.status == "ok" }) == true
        {
          _ = try await session.runtime.command("models", ["ids": .array([.string(model)])])
        }
      } catch {
        sessions[session.id] = nil
        await session.runtime.close()
        await session.decoder.close()
        throw error
      }
    }
    guard !closed else { throw CancellationError() }
    if let cid = session.state?.composerDraft?.id {
      try await conversationPositions.prepare(cid)
    }
    if let filter {
      _ = try await session.runtime.command(
        "historyFilter",
        [
          "search": .string(filter.search), "sort": .string(filter.sort),
          "page": .number(Decimal(filter.page)),
        ])
    }
    closeViewers()
    await previous.setForeground(false)
    selectSession(session)
    await session.runtime.setForeground(true)
    // Await a full projection before acknowledging navigation to the composer.
    await session.runtime.publishNow()
    await synchronizeSessionHistory()
  }

  func protectSessionMutation(_ kind: String, _ payload: [String: ConversationValue]) throws {
    guard ["deleteChats", "renameChat", "pinChat", "generateTitle"].contains(kind) else { return }
    let ids = Set(
      payload["ids"]?.array?.compactMap(\.string) ?? [payload["id"]?.string].compactMap { $0 })
    for session in sessions.values where session.id != activeSession {
      if let cid = session.state?.composerDraft?.id, ids.contains(cid) {
        throw ManagerError.core(
          "Open this conversation and finish its response before changing it.")
      }
    }
  }
}
