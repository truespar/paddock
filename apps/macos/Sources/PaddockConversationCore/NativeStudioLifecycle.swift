import Foundation

extension NativeStudioRuntime {
  /// Selection never changes a running actor's document. The app retains this
  /// actor and its transport until every writer is settled and durable.
  public func canLeaveConversation() -> Bool {
    !closed && !preparingQuit && !mutating && commandCount == 0
      && (audio["phase"]?.string ?? "idle") == "idle"
      // Traverse's MCP bridge is still owned by the visible viewer.
      && (tasks.isEmpty || graphGrounding.isEmpty)
  }

  public func canReleaseConversation() -> Bool {
    !busy && commandCount == 0 && !saveFailed && titleTask == nil && compactionTask == nil
  }

  public func setForeground(_ value: Bool) {
    foreground = value
    if value {
      beginPolling()
    } else {
      polling?.cancel()
      polling = nil
      reclaimPresentationCaches()
    }
  }

  /// A failed save must leave the runtime alive for retry; closing its transport
  /// first would turn a recoverable SQLite failure into lost in-memory work.
  public func prepareForQuit() async throws {
    guard !closed else { return }
    preparingQuit = true
    defer {
      preparingQuit = false
      schedulePublish()
    }
    while commandCount > 0 { try await Task.sleep(for: .milliseconds(5)) }
    let summary = compactionTask
    let title = titleTask
    cancelCompaction()
    title?.cancel()
    stopRequested = true
    let running = Array(tasks.values)
    for task in running { task.cancel() }
    for task in running { await task.value }
    await summary?.value
    await title?.value
    titleTask = nil
    _ = try? await writeTail?.value
    flushResponses()
    try await persist()
    saveFailed = false
    await emit()
  }

  public func mergeSessionHistory(_ rows: [O], busy: Set<String>) {
    var changed = activeConversationIDs != busy
    activeConversationIDs = busy
    for row in rows {
      guard let id = row["id"]?.string, id != document?.id else { continue }
      if let index = history.firstIndex(where: { $0["id"]?.string == id }) {
        if history[index] != row {
          history[index] = row
          changed = true
        }
      } else {
        history.append(row)
        changed = true
      }
    }
    if changed { schedulePublish() }
  }
}
