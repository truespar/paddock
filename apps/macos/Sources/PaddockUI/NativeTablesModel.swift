import Foundation
import Observation
import PaddockClient
import PaddockConversationCore

private actor TablesConnection {
  let client: any ManagerLoading
  var transport: NativeConversationTransport?
  init(client: any ManagerLoading) { self.client = client }
  func call(_ path: String, _ method: String, _ body: ConversationValue?, _ query: [String: String])
    async throws
    -> ConversationValue
  {
    if transport == nil {
      transport = try NativeConversationTransport(host: await client.nativeConversationHost())
    }
    // 1,024 regression rows carry 999 quantiles each; the chat transport's
    // 16 MiB default can reject an otherwise valid result. Decode off-main.
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try await transport!.bytes(
      path, method: method,
      body: body.map { try encoder.encode($0) },
      query: query,
      maximum: (path.hasPrefix("api/table-history") ? 128 : 64) * 1024 * 1024)
    return data.isEmpty ? .null : try JSONDecoder().decode(ConversationValue.self, from: data)
  }
}

/// Durable table workspace backed by the same SQLite sessions as web Tables.
@MainActor @Observable final class NativeTablesModel {
  typealias API =
    @MainActor (String, String, ConversationValue?, [String: String]) async throws ->
    ConversationValue
  struct Predictor: Identifiable, Equatable, Sendable {
    let port: UInt16
    let pid: Int
    let model: String
    let title: String
    let vendor: String?
    let limits: TabularLimits
    var id: String { "\(port):\(pid)" }
  }
  struct Result: Sendable {
    let id = UUID()
    let draft: UUID
    let endpoint: String
    let title: String
    let task: TabularTask
    let csv: String
    let header: [String]
    let rows: [[String]]
    let response: TabularResults
    let milliseconds: Double
  }
  private(set) var predictors: [Predictor] = []
  var port: UInt16 = 0 {
    didSet {
      guard port != oldValue else { return }
      if history.document == nil && !restoring {
        estimators = current?.limits.defaultEstimators ?? 8
      }
      prepare()
    }
  }
  private(set) var source = ""
  private(set) var fileName = ""
  private(set) var table: TabularTable?
  private(set) var columns: [Column] = []
  struct Column: Identifiable, Sendable {
    let id: Int
    let name: String
    let sample: String
    let missing: Int
  }
  var spec: TabularSpec? { didSet { if spec != oldValue { prepare() } } }
  var estimators = 8 { didSet { if estimators != oldValue { prepare() } } }
  var seed = 0 { didSet { if seed != oldValue { prepare() } } }
  private(set) var plan: TabularPlan?
  private(set) var request: ConversationValue?
  private(set) var curl = ""
  private(set) var curlPreview = ""
  private(set) var validation: String?
  private(set) var error: String?
  private(set) var discoveryError: String?
  private(set) var result: Result?
  private(set) var loading = false
  private(set) var importing = false
  private(set) var preparing = false
  private(set) var busy = false
  private(set) var tableVersion = UUID()
  let history = NativeTableHistory()
  private(set) var selectedRun: String?
  private(set) var restoring = false
  @ObservationIgnored private var autosaveTask: Task<Void, Never>?
  private var savedDraft: UUID?
  private var presentationID = UUID()
  private var navigatingSession = false
  var historyNavigationBlocked: Bool {
    busy || importing || restoring || navigatingSession || history.blocked
  }
  var sessionTitle: String {
    history.document?.title ?? (fileName.isEmpty ? "New table" : fileName)
  }
  @ObservationIgnored var api: API
  @ObservationIgnored private var parseTask: Task<Void, Never>?
  @ObservationIgnored private var planTask: Task<Void, Never>?
  @ObservationIgnored private var runTask: Task<Void, Never>?
  private var draftID = UUID()
  private var importID = UUID()
  private var runID = UUID()
  var current: Predictor? { predictors.first { $0.port == port } }
  var hasWork: Bool { !source.isEmpty || busy || importing }
  var canRun: Bool {
    current != nil && request != nil && !busy && !importing && !preparing && !restoring
  }
  var stale: Bool {
    result.map { result in
      result.draft != draftID || (current.map { result.endpoint != $0.id } ?? false)
    } ?? false
  }

  init(client: any ManagerLoading) {
    let connection = TablesConnection(client: client)
    api = { path, method, body, query in try await connection.call(path, method, body, query) }
  }
  func refresh() async {
    guard !loading else { return }
    loading = true
    defer { loading = false }
    do {
      let fleet = try await api("api/runners", "GET", nil, [:]).array ?? []
      let previous = current
      var next: [Predictor] = []
      var failures: [String] = []
      for runner in fleet {
        guard let id = runner["tabular"]?.string,
          let n = runner["port"]?.integer, let port = UInt16(exactly: n), port > 0,
          runner["status"]?.string == "ok"
        else { continue }
        try Task.checkCancellation()
        do {
          let server = try await api("api/runners/\(port)/server", "GET", nil, [:])
          let limits = try TabularLimits(server: server)
          next.append(
            Predictor(
              port: port, pid: runner["pid"]?.integer ?? 0, model: id,
              title: Self.variantTitle(id: id, display: runner["display"]?.string),
              vendor: runner["vendor"]?.string ?? "NVIDIA", limits: limits))
        } catch is CancellationError { throw CancellationError() } catch {
          failures.append("\(id): \(error.localizedDescription)")
        }
      }
      try Task.checkCancellation()
      predictors = next
      if !next.contains(where: { $0.port == port })
        || (previous != nil && current?.model != previous?.model)
      {
        if let saved = history.document?.draft.model, !saved.isEmpty {
          port = next.first { $0.model == saved }?.port ?? 0
        } else {
          port = next.first?.port ?? 0
        }
      }
      if previous != current { prepare() }
      discoveryError = failures.isEmpty ? nil : failures.joined(separator: "\n")
    } catch is CancellationError {} catch { discoveryError = error.localizedDescription }
  }
  static func variantTitle(id: String, display: String?) -> String {
    guard let display else { return id.replacingOccurrences(of: "-", with: " ") }
    let slug = display.lowercased().replacingOccurrences(
      of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
    let suffix = id.lowercased().hasPrefix(slug + "-") ? String(id.dropFirst(slug.count + 1)) : ""
    return suffix.isEmpty
      ? display : display + " · " + suffix.replacingOccurrences(of: "-", with: " ")
  }
  func setSource(_ text: String, name: String = "") {
    guard !busy else { return }
    parseTask?.cancel()
    planTask?.cancel()
    importID = UUID()
    let attempt = importID
    source = text
    fileName = name
    draftID = UUID()
    request = nil
    plan = nil
    curl = ""
    curlPreview = ""
    error = nil
    validation = nil
    importing = !text.isEmpty
    preparing = false
    guard !text.isEmpty else {
      table = nil
      spec = nil
      result = nil
      columns = []
      return
    }
    parseTask = Task { [weak self] in
      do {
        try await Task.sleep(for: .milliseconds(150))
        let job = Task.detached(priority: .userInitiated) {
          let table = try TabularTable.parse(text)
          let columns = table.header.indices.map { j in
            var sample: [String] = []
            var missing = 0
            for row in table.rows {
              if TabularTable.isMissing(row[j]) {
                missing += 1
              } else if sample.count < 4 && !sample.contains(row[j]) {
                sample.append(row[j])
              }
            }
            return Column(
              id: j, name: table.header[j],
              sample: String(sample.joined(separator: ", ").prefix(512)), missing: missing)
          }
          return (table, TabularSpec(table: table), columns)
        }
        let (table, spec, columns) = try await withTaskCancellationHandler(
          operation: { try await job.value }, onCancel: { job.cancel() })
        try Task.checkCancellation()
        guard let self, self.importID == attempt else { return }
        let sameHeader = self.table?.header == table.header
        self.table = table
        self.columns = columns
        self.tableVersion = UUID()
        self.importing = false
        if !sameHeader || self.spec == nil { self.spec = spec }
        self.prepare()
      } catch is CancellationError {} catch {
        guard let self, self.importID == attempt else { return }
        self.table = nil
        self.importing = false
        self.validation = error.localizedDescription
      }
    }
  }
  func prepare() {
    planTask?.cancel()
    draftID = UUID()
    scheduleSave()
    let attempt = draftID
    plan = nil
    request = nil
    curl = ""
    curlPreview = ""
    validation = nil
    preparing = false
    guard !importing, let table, let spec, let current else { return }
    let estimators = estimators
    let seed = seed
    preparing = true
    planTask = Task { [weak self] in
      do {
        let job = Task.detached(priority: .userInitiated) {
          let plan = try table.plan(
            spec: spec, limits: current.limits, estimators: estimators, seed: seed)
          let curl = try plan.curl(port: current.port, model: current.model)
          let preview =
            String(curl.prefix(3000)) + (curl.utf8.count > 3000 ? "\n… Preview shortened" : "")
          return (plan, try plan.request(model: current.model), curl, preview)
        }
        let (plan, request, curl, preview) = try await withTaskCancellationHandler(
          operation: { try await job.value }, onCancel: { job.cancel() })
        try Task.checkCancellation()
        guard let self, self.draftID == attempt else { return }
        self.plan = plan
        self.request = request
        self.curl = curl
        self.curlPreview = preview
        self.preparing = false
      } catch is CancellationError {} catch {
        guard let self, self.draftID == attempt else { return }
        self.preparing = false
        self.validation = error.localizedDescription
      }
    }
  }
  func loadExample() {
    setSource((current?.limits.task ?? .classification).example, name: "Example.csv")
  }
  func importFile(_ url: URL) async {
    guard !busy, !importing else { return }
    let attempt = importID
    importing = true
    defer {
      if importID == attempt {
        importing = false
        prepare()
      }
    }
    let allowed = url.startAccessingSecurityScopedResource()
    defer { if allowed { url.stopAccessingSecurityScopedResource() } }
    do {
      let task = Task.detached(priority: .userInitiated) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: TabularTable.maximumBytes + 1) ?? Data()
        guard data.count <= TabularTable.maximumBytes else {
          throw ConversationFailure.invalid("Open a table smaller than 8 MiB.")
        }
        guard let text = String(data: data, encoding: .utf8) else {
          throw ConversationFailure.invalid(
            "Save this table as UTF-8 CSV or TSV before opening it.")
        }
        return text
      }
      let text = try await withTaskCancellationHandler(
        operation: { try await task.value }, onCancel: { task.cancel() })
      try Task.checkCancellation()
      guard importID == attempt, !busy else { return }
      setSource(text, name: url.lastPathComponent)
    } catch is CancellationError {} catch { self.error = error.localizedDescription }
  }
  func report(_ error: any Error) { self.error = error.localizedDescription }
  func run() {
    guard canRun, let current, let table, let spec, let plan, let request else { return }
    let draft = draftID
    let source = source
    let name = fileName
    let estimators = estimators
    let seed = seed
    runID = UUID()
    let attempt = runID
    busy = true
    error = nil
    runTask = Task { [weak self] in
      guard let self else { return }
      let start = ContinuousClock.now
      defer { if self.runID == attempt { self.busy = false } }
      do {
        let raw = try await api(
          "api/runners/\(current.port)/v1/tabular/predictions", "POST", request, [:])
        try Task.checkCancellation()
        let elapsed = start.duration(to: .now)
        let ms =
          Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
        let job = Task.detached(priority: .userInitiated) {
          let response = try TabularResults(response: raw, plan: plan)
          let features = Array(plan.features.prefix(4))
          let header =
            ["Row"] + features.map { table.header[$0] } + [table.header[spec.target]]
            + (plan.task == .classification
              ? ["Confidence"] + response.classes.map { "P(\($0))" } : ["p10", "p90"])
          let rows = response.rows.map { r in
            [String(r.row + 1)] + features.map { table.rows[r.row][$0] } + [r.value]
              + (plan.task == .classification
                ? ([r.confidence ?? 0] + r.probabilities).map { String(format: "%.1f%%", $0 * 100) }
                : [r.low, r.high].map { $0.map(TabularResults.plain) ?? "" })
          }
          return Result(
            draft: draft, endpoint: current.id, title: current.title, task: plan.task,
            csv: response.csv(table: table, spec: spec), header: header, rows: rows,
            response: response, milliseconds: ms)
        }
        let result = try await withTaskCancellationHandler(
          operation: { try await job.value }, onCancel: { job.cancel() })
        try Task.checkCancellation()
        guard self.runID == attempt else { return }
        self.result = result
        let input = await Task.detached {
          TableInput(
            source: source, fileName: name, spec: spec, model: current.model,
            port: current.port, estimators: estimators, seed: seed)
        }.value
        let run = TableRun(input: input, task: plan.task, ms: ms, response: raw)
        self.selectedRun = run.id
        autosaveTask?.cancel()
        if await history.save(input: input, source: source, run: run, api: api) {
          savedDraft = draft
        }
      } catch is CancellationError {} catch {
        if self.runID == attempt { self.error = error.localizedDescription }
      }
    }
  }
  func cancel() {
    runTask?.cancel()
    runID = UUID()
    busy = false
  }
  func prepareForQuit() async -> Bool {
    autosaveTask?.cancel()
    cancel()
    await settle()
    return await saveSession()
  }
  func saveCopy() async {
    guard !historyNavigationBlocked else { return }
    history.fork()
    savedDraft = nil
    _ = await saveSession()
  }
  func settle() async {
    await parseTask?.value
    await planTask?.value
    await runTask?.value
  }

  private func scheduleSave() {
    guard !restoring else { return }
    autosaveTask?.cancel()
    autosaveTask = Task { [weak self] in
      do { try await Task.sleep(for: .milliseconds(800)) } catch { return }
      guard let self, !busy else { return }
      _ = await saveSession()
    }
  }
  @discardableResult func saveSession() async -> Bool {
    guard !restoring else { return false }
    await parseTask?.value
    if source.isEmpty && history.document == nil { return true }
    if savedDraft == draftID { return await history.flush() }
    let attempt = draftID
    let source = source
    let name = fileName
    let spec = spec
    let model = current?.model ?? history.document?.draft.model ?? ""
    let port = port
    let estimators = estimators
    let seed = seed
    let input = await Task.detached {
      TableInput(
        source: source, fileName: name, spec: spec, model: model,
        port: port, estimators: estimators, seed: seed)
    }.value
    let ok = await history.save(input: input, source: source, api: api)
    if ok { savedDraft = attempt }
    return ok
  }
  func newSession() async {
    guard !historyNavigationBlocked else { return }
    navigatingSession = true
    defer { navigatingSession = false }
    autosaveTask?.cancel()
    guard await saveSession() else { return }
    clearSession()
  }
  private func clearSession() {
    restoring = true
    history.reset()
    setSource("")
    selectedRun = nil
    result = nil
    savedDraft = nil
    restoring = false
  }
  func openSession(_ id: String) async {
    guard !historyNavigationBlocked, history.document?.id != id else { return }
    navigatingSession = true
    defer { navigatingSession = false }
    autosaveTask?.cancel()
    guard await saveSession(), let doc = await history.open(id, api: api) else { return }
    restoring = true
    await restoreInput(doc.draft, source: doc.datasets[doc.draft.dataset] ?? "")
    result = nil
    selectedRun = nil
    if let last = doc.runs.last { await selectRun(last) }
    savedDraft = draftID
    restoring = false
  }
  private func restoreInput(_ input: TableInput, source: String) async {
    setSource(source, name: input.fileName)
    await parseTask?.value
    // Never silently send a restored session to a different model that reused its port.
    port =
      predictors.first { $0.model == input.model && $0.port == input.port }?.port
      ?? predictors.first { $0.model == input.model }?.port ?? 0
    spec = input.spec
    estimators = input.estimators
    seed = input.seed
    await planTask?.value
  }
  func selectRun(_ run: TableRun) async {
    guard let source = history.document?.datasets[run.input.dataset], let spec = run.input.spec
    else { return }
    let attempt = UUID()
    presentationID = attempt
    let session = history.document?.id
    do {
      let draft = draftID
      let endpoint = current?.model == run.input.model ? current?.id ?? "" : ""
      let matches =
        source == self.source && spec == self.spec && seed == run.input.seed
        && estimators == run.input.estimators
        && (current == nil || current?.model == run.input.model)
      let result = try await Task.detached {
        let table = try TabularTable.parse(source)
        let plan = try table.plan(
          spec: spec, limits: TabularLimits(task: run.task),
          estimators: run.input.estimators, seed: run.input.seed)
        let response = try TabularResults(response: run.response, plan: plan)
        let features = Array(plan.features.prefix(4))
        let header =
          ["Row"] + features.map { table.header[$0] } + [table.header[spec.target]]
          + (run.task == .classification
            ? ["Confidence"] + response.classes.map { "P(\($0))" } : ["p10", "p90"])
        let rows = response.rows.map { r in
          [String(r.row + 1)] + features.map { table.rows[r.row][$0] } + [r.value]
            + (run.task == .classification
              ? ([r.confidence ?? 0] + r.probabilities).map { String(format: "%.1f%%", $0 * 100) }
              : [r.low, r.high].map { $0.map(TabularResults.plain) ?? "" })
        }
        return Result(
          draft: matches ? draft : UUID(), endpoint: endpoint, title: run.input.model,
          task: run.task, csv: response.csv(table: table, spec: spec), header: header,
          rows: rows, response: response, milliseconds: run.ms)
      }.value
      guard presentationID == attempt, history.document?.id == session else { return }
      self.result = result
      selectedRun = run.id
    } catch { self.error = error.localizedDescription }
  }
  func restoreRun(_ run: TableRun) async {
    guard !historyNavigationBlocked, let source = history.document?.datasets[run.input.dataset]
    else { return }
    restoring = true
    await restoreInput(run.input, source: source)
    await selectRun(run)
    restoring = false
    scheduleSave()
  }
  func removeSession(_ row: TableSummary) async {
    guard !historyNavigationBlocked else { return }
    let active = history.document?.id == row.id
    autosaveTask?.cancel()
    if await history.remove(row, api: api), active { clearSession() }
  }
  func renameSession(_ id: String, title: String) async {
    guard !historyNavigationBlocked else { return }
    navigatingSession = true
    defer { navigatingSession = false }
    autosaveTask?.cancel()
    guard await saveSession() else { return }
    await history.rename(id, title: title, api: api)
  }
}
