import Foundation
import Observation
import PaddockConversationCore

@MainActor @Observable final class NativeTableHistory {
  typealias API = NativeTablesModel.API
  private(set) var sessions: [TableSummary] = []
  private(set) var document: TableSession?
  private(set) var revision = ""
  private(set) var loaded = false
  private(set) var saving = false
  private(set) var navigating = false
  private(set) var error: String?
  private var generation = 0
  private var write: Task<Bool, Never>?
  private var writeID = UUID()
  private var pending: (TableSession, String)?
  var blocked: Bool { saving || navigating }

  func refresh(api: API) async {
    let attempt = generation
    do {
      let raw = try await api("api/table-history", "GET", nil, [:])
      let rows = try await Task.detached {
        try JSONDecoder().decode([TableSummary].self, from: JSONEncoder().encode(raw))
      }.value
      if attempt == generation {
        sessions = rows
        loaded = true
        if pending == nil { error = nil }
      }
    } catch {
      self.error = error.localizedDescription
      loaded = true
    }
  }

  /// Queue writes so a run, a navigation flush and autosave share one revision.
  /// Failed writes retain the full draft/run in memory, never just a list row.
  func save(input: TableInput, source: String, run: TableRun? = nil, api: @escaping API) async
    -> Bool
  {
    let previous = write
    let attempt = UUID()
    writeID = attempt
    saving = true
    let task = Task { [self] in
      if let previous { _ = await previous.value }
      // Retry the *identical* payload first: the server may have committed it
      // before the acknowledgement was lost. Never rebase a stale client.
      var acknowledged = true
      if let pending { acknowledged = await persist(pending.0, expected: pending.1, api: api) }
      var doc = document ?? TableSession(input: input, source: source)
      doc.draft = input
      doc.model = input.model
      doc.datasets[input.dataset] = source
      if let run, !doc.runs.contains(where: { $0.id == run.id }) { doc.runs.append(run) }
      // Drop superseded drafts, but never a dataset referenced by a saved run.
      let retained = Set(doc.runs.map { $0.input.dataset } + [input.dataset])
      doc.datasets = doc.datasets.filter { retained.contains($0.key) }
      doc.updatedAt = max(doc.updatedAt + 1, Int(Date().timeIntervalSince1970 * 1000))
      document = doc
      guard acknowledged else { return false }
      return await persist(doc, expected: revision, api: api)
    }
    write = task
    let result = await task.value
    // Tasks can queue while this one is awaiting the service. A generation of
    // UI writes is serialized; callers block navigation until flush completes.
    if writeID == attempt {
      write = nil
      saving = false
    }
    return result
  }

  func flush() async -> Bool {
    guard let task = write else { return pending == nil }
    let attempt = writeID
    let ok = await task.value
    if writeID == attempt {
      write = nil
      saving = false
    }
    return ok
  }

  private func persist(_ doc: TableSession, expected: String, api: API) async -> Bool {
    pending = (doc, expected)
    do {
      let raw = try await Task.detached {
        let data = try JSONEncoder().encode(doc)
        guard data.count <= 64 * 1024 * 1024 else {
          throw ConversationFailure.invalid(
            "Start a new table; this session exceeds 64 MiB. Your results are kept.")
        }
        return try JSONDecoder().decode(ConversationValue.self, from: data)
      }.value
      let path = "api/table-history/\(doc.id)"
      let reply = try await api(path, "PUT", raw, ["revision": expected])
      let row = try JSONDecoder().decode(TableSummary.self, from: JSONEncoder().encode(reply))
      revision = row.revision
      pending = nil
      generation += 1
      sessions = ([row] + sessions.filter { $0.id != row.id }).sorted {
        $0.updatedAt > $1.updatedAt
      }
      error = nil
      return true
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }

  func open(_ id: String, api: API) async -> TableSession? {
    guard await flush(), !navigating else { return nil }
    navigating = true
    defer { navigating = false }
    do {
      let raw = try await api("api/table-history/\(id)", "GET", nil, [:])
      guard let text = raw["doc"]?.string, let revision = raw["revision"]?.string else {
        throw ConversationFailure.invalid("Invalid table session")
      }
      let doc = try await Task.detached {
        let doc = try JSONDecoder().decode(TableSession.self, from: Data(text.utf8))
        guard doc.version == 1, doc.id == id, doc.datasets[doc.draft.dataset] != nil else {
          throw ConversationFailure.invalid("Unsupported table session")
        }
        return doc
      }.value
      generation += 1
      document = doc
      self.revision = revision
      error = nil
      return doc
    } catch {
      self.error = error.localizedDescription
      return nil
    }
  }

  func reset() {
    generation += 1
    document = nil
    revision = ""
    pending = nil
    error = nil
  }
  func fork() {
    guard !blocked, var doc = document else { return }
    doc.id = UUID().uuidString
    doc.title = String(doc.title.prefix(200)) + " copy"
    doc.createdAt = Int(Date().timeIntervalSince1970 * 1000)
    doc.updatedAt = doc.createdAt
    document = doc
    revision = ""
    pending = nil
    error = nil
  }
  func rename(_ id: String, title: String, api: API) async {
    guard !blocked, error == nil, !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { return }
    navigating = true
    defer { navigating = false }
    do {
      let raw = try await api("api/table-history/\(id)", "GET", nil, [:])
      guard let text = raw["doc"]?.string, let expected = raw["revision"]?.string else { return }
      var doc = try await Task.detached {
        try JSONDecoder().decode(TableSession.self, from: Data(text.utf8))
      }.value
      doc.title = title.trimmingCharacters(in: .whitespacesAndNewlines)
      let active = document
      let activeRevision = revision
      let ok = await persist(doc, expected: expected, api: api)
      if active?.id != id || !ok {
        document = active
        revision = activeRevision
      } else {
        document = doc
      }
      // A failed rename is retried explicitly, never as another table's autosave.
      if !ok { pending = nil }
    } catch { self.error = error.localizedDescription }
  }
  func remove(_ row: TableSummary, api: API) async -> Bool {
    guard !blocked else { return false }
    navigating = true
    defer { navigating = false }
    do {
      _ = try await api("api/table-history/\(row.id)", "DELETE", nil, ["revision": row.revision])
      generation += 1
      sessions.removeAll { $0.id == row.id }
      if document?.id == row.id { reset() }
      return true
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }
}
