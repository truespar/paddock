import AppKit
import CryptoKit
import Foundation
import Observation
import PaddockClient
import PaddockConversationCore
import UniformTypeIdentifiers

private actor ReadsConnection {
  let client: any ManagerLoading
  var transport: NativeConversationTransport?
  private var closed = false
  init(client: any ManagerLoading) { self.client = client }
  private func connection() async throws -> NativeConversationTransport {
    guard !closed else { throw ConversationFailure.closed }
    if let transport { return transport }
    let host = try await client.nativeConversationHost()
    try Task.checkCancellation()
    guard !closed else { throw ConversationFailure.closed }
    let next = try NativeConversationTransport(host: host)
    transport = next
    return next
  }
  func close() async {
    closed = true
    await transport?.close()
    transport = nil
  }
  func read(_ port: UInt16, _ body: Data) async throws -> ConversationValue {
    let transport = try await connection()
    let bytes = try await transport.bytes(
      "api/runners/\(port)/v1/systemone", method: "POST", body: body)
    return try JSONDecoder().decode(ConversationValue.self, from: bytes)
  }
  func call(_ path: String, _ method: String, _ body: ConversationValue?, _ query: [String: String])
    async throws -> ConversationValue
  {
    let transport = try await connection()
    if path.hasPrefix("api/read-history/") && method == "GET" {
      // A 16 MiB document expands when carried as a JSON string envelope.
      let data = try await transport.bytes(
        path, method: method, query: query, maximum: 40 * 1024 * 1024)
      return try JSONDecoder().decode(ConversationValue.self, from: data)
    }
    return try await transport.api(path, method: method, body: body, query: query)
  }
}

@MainActor @Observable final class NativeReadsModel {
  typealias API =
    @MainActor (String, String, ConversationValue?, [String: String]) async throws ->
    ConversationValue
  struct Reader: Identifiable, Equatable {
    let port: UInt16
    let model: String
    let title: String
    let vendor: String?
    let maxQuestions: Int
    let maxSamples: Int
    let types: [String]
    var images = false
    var maxSteps = 1
    var think = false
    var backend: String?
    var checkpoints: [String] = []
    var conditional = false
    var id: UInt16 { port }
  }
  struct SavedSet: Decodable, Identifiable, Equatable {
    let id: String
    let name: String
    let body: String
    let revision: String
  }
  struct Run: Identifiable, Codable, Sendable {
    var id = UUID()
    var at = Date()
    let fingerprint: String
    let excerpt: String
    let characters: Int
    let questions: [ReadQuestion]
    let raw: ConversationValue
    let port: UInt16
    let elapsedMilliseconds: Double
    let response: ReadResponse
    var state: String?
    var fileName = ""
    var samples = 0
    var pictures: [ReadPicture] = []
    var steps = 1
    var think = 0
    var checkpoint: String?
    var isCameraFrame: Bool {
      pictures.first?.name.hasPrefix("camera") == true || excerpt == "camera frame"
    }
    enum CodingKeys: String, CodingKey {
      case id, at, fingerprint, excerpt, characters, questions, raw, port, elapsedMilliseconds,
        state, fileName, samples, pictures, steps, think, checkpoint
    }
    var retainedBytes: Int {
      // Never base64-encode multi-megabyte pictures on the main actor during
      // memory-pressure trimming. Count their existing storage directly.
      return raw.estimatedRetainedBytes * 2 + (state?.utf8.count ?? 0) + excerpt.utf8.count
        + questions.reduce(0) { $0 + $1.instructions.utf8.count + 256 }
        + pictures.reduce(0) { $0 + $1.url.utf8.count + $1.name.utf8.count }
    }
    nonisolated static func fingerprint(_ bytes: Data) -> String {
      SHA256.hash(data: bytes)
        .map { String(format: "%02x", $0) }.joined()
    }
  }
  var draft = ReadDraft()
  var port: UInt16 = 0
  var setName = ""
  var jsonText = ""
  var fileName = ""
  private(set) var readers: [Reader] = []
  private(set) var sets: [SavedSet] = []
  private(set) var selectedSet: SavedSet?
  private(set) var runs: [Run] = []
  var selectedRun: UUID?
  private(set) var loading = false
  private(set) var busy = false
  private(set) var saving = false
  private(set) var importing = false
  @ObservationIgnored private var imageImportTask: Task<Void, Never>?
  var error: String?
  var stateError: String?
  var questionsError: String?
  var rowErrors: [String: String] = [:]
  var historyError: String?
  struct Session: Decodable, Identifiable {
    let id: String
    let title: String
    let model: String
    let runs: Int
    let updatedAt: Double
  }
  private(set) var sessions: [Session] = []
  private(set) var historyLoaded = false
  private(set) var historyListError: String?
  private var historyGeneration = 0
  private(set) var activeSession: ReadHistoryDocument?
  private var sessionRevision = ""
  private var sessionEpoch = 0
  private var closed = false
  @ObservationIgnored private let historyLoader = NativeReadHistoryLoader()
  private(set) var openingSession = false
  private(set) var historyUnsaved = false
  @ObservationIgnored var api: API
  // Inference takes authored JSON bytes, never an unordered dictionary.
  @ObservationIgnored var readAPI: @MainActor (UInt16, Data) async throws -> ConversationValue
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored private var closeConnection: () async -> Void
  @ObservationIgnored private var refreshError: String?
  @ObservationIgnored private var latestRequest: ConversationValue?
  @ObservationIgnored private var latestOrdering: [[String]]?
  @ObservationIgnored private var latestRunID: UUID?
  @ObservationIgnored private var setsEpoch = 0
  @ObservationIgnored private var originalBody = ReadDraft().setBody
  @ObservationIgnored private var originalOrdering = ReadDraft().ordering
  @ObservationIgnored private var originalJSON = ""
  var hasUnappliedJSON: Bool { jsonText != originalJSON }
  var current: Reader? { readers.first { $0.port == port } }
  // A bundle id named "laya" is also upstream's explicit English alias.
  // Automatic routing must not accidentally force English for non-English input.
  var requestModel: String {
    current?.backend == "laya" ? (draft.checkpoint ?? "") : (current?.model ?? "")
  }
  var result: Run? { runs.first { $0.id == selectedRun } ?? runs.first }
  var dirty: Bool {
    draft.setBody != originalBody || draft.ordering != originalOrdering
      || setName != (selectedSet?.name ?? "") || hasUnappliedJSON
  }
  var hasWork: Bool {
    dirty || busy || saving || importing || historyUnsaved || !draft.state.isEmpty
      || !draft.images.isEmpty
  }
  var historyNavigationBlocked: Bool { closed || busy || saving || importing || openingSession }
  /// A completed, saved read can be reopened without a discard prompt. Only
  /// input edits or results that have not reached SQLite need confirmation.
  var unsavedRead: Bool {
    if historyUnsaved || hasUnappliedJSON { return true }
    guard activeSession != nil, let last = runs.first else { return hasWork }
    var saved = ReadDraft()
    saved.state = last.state ?? ""
    saved.questions = last.questions
    saved.samples = last.samples
    saved.checkpoint = last.checkpoint
    saved.images = last.pictures
    saved.steps = last.steps
    saved.think = last.think
    return draft.state != saved.state || draft.setBody != saved.setBody
      || draft.images != saved.images
      || draft.ordering != saved.ordering || fileName != last.fileName
      || setName != (selectedSet?.name ?? "")
  }
  func visibleSessions(search: String) -> [Session] {
    let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
    return sessions.filter { query.isEmpty || $0.title.localizedStandardContains(query) }
      .sorted { $0.updatedAt == $1.updatedAt ? $0.id < $1.id : $0.updatedAt > $1.updatedAt }
  }
  var validation: String? {
    if draft.questions.contains(where: \.conditional), current?.conditional != true {
      return "This model reads every question at once; choose a reader that supports conditions."
    }
    if let checkpoint = draft.checkpoint, current?.checkpoints.contains(checkpoint) != true {
      return
        "The selected checkpoint is not available on this reader. Choose automatic routing or a Laya instance."
    }
    if !draft.images.isEmpty && current?.images != true {
      return "This model reads text only. Remove the images or start it with vision."
    }
    if draft.steps > (current?.maxSteps ?? 1) {
      return "This runner does not support the selected step count."
    }
    if draft.think > 0 && current?.think != true {
      return "This runner does not support a thought before reading."
    }
    return draft.validation(
      maxQuestions: current?.maxQuestions ?? 64, maxSamples: current?.maxSamples ?? 32)
      ?? (draft.questions.contains {
        !(current?.types ?? ["noul", "choice", "score"]).contains($0.kind.rawValue)
      }
        ? "This runner does not support one of these question types." : nil)
  }
  var canRun: Bool {
    current != nil && !historyNavigationBlocked && !historyUnsaved
      && validation == nil
      && (!draft.state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        || !draft.images.isEmpty)
  }
  var stale: Bool {
    guard let result else { return false }
    guard result.id == latestRunID else { return false }
    return result.port != port || latestRequest != draft.request(model: requestModel)
      || latestOrdering != draft.ordering
  }
  var previousRead: Bool { result != nil && result?.id != latestRunID }
  init(client: any ManagerLoading) {
    let connection = ReadsConnection(client: client)
    api = { path, method, body, query in try await connection.call(path, method, body, query) }
    readAPI = { port, body in try await connection.read(port, body) }
    closeConnection = { await connection.close() }
  }
  private func decode<T: Decodable>(_ type: T.Type, _ v: ConversationValue) throws -> T {
    try ConversationValueDecoder().decode(type, from: v)
  }
  private func checkOpen() throws {
    try Task.checkCancellation()
    guard !closed else { throw ConversationFailure.closed }
  }
  func refresh() async {
    guard !closed, !loading else { return }
    loading = true
    defer { loading = false }
    do {
      let fleet = try await api("api/runners", "GET", nil, [:]).array ?? []
      try checkOpen()
      var next: [Reader] = []
      for runner in fleet {
        try Task.checkCancellation()
        guard let number = runner["port"]?.integer, let port = UInt16(exactly: number), port > 0,
          let id = runner["reader"]?.string ?? runner["model"]?.string
        else { continue }
        do {
          let info = try await api("api/runners/\(port)/server", "GET", nil, [:])
          try checkOpen()
          guard let caps = info["structured_read"],
            (caps["canvas_width"]?.integer ?? 0) > 0
              || ["laya", "clef"].contains(caps["backend"]?.string ?? "")
          else {
            continue
          }
          next.append(
            Reader(
              port: port, model: id, title: runner["display"]?.string ?? id,
              vendor: runner["vendor"]?.string,
              maxQuestions: min(1024, max(1, caps["max_questions"]?.integer ?? 64)),
              maxSamples: min(32, max(1, caps["max_samples"]?.integer ?? 32)),
              types: caps["types"]?.array?.compactMap(\.string) ?? ["noul", "choice", "score"],
              images: caps["images"] == .bool(true),
              maxSteps: min(8, max(1, caps["max_steps"]?.integer ?? 1)),
              think: caps["think"] == .bool(true),
              backend: caps["backend"]?.string,
              checkpoints: caps["checkpoints"]?.array?.compactMap { $0["name"]?.string } ?? [],
              conditional: caps["conditional"] == .bool(true)))
        } catch is CancellationError { throw CancellationError() } catch {
          if let previous = readers.first(where: { $0.port == port && $0.model == id }) {
            next.append(previous)
          }
        }
      }
      try checkOpen()
      readers = next
      if !readers.contains(where: { $0.port == port }) { port = readers.first?.port ?? 0 }
      let epoch = setsEpoch
      let saved = try await api("api/reads", "GET", nil, [:])
      try checkOpen()
      if epoch == setsEpoch { sets = try decode([SavedSet].self, saved) }
      await refreshHistory()
      try checkOpen()
      if error == refreshError { error = nil }
      refreshError = nil
    } catch is CancellationError {} catch {
      guard !closed else { return }
      refreshError = error.localizedDescription
      self.error = refreshError
    }
  }
  func add(_ kind: ReadQuestion.Kind) {
    guard draft.questions.count < (current?.maxQuestions ?? 64) else { return }
    var n = 1
    while draft.questions.contains(where: { $0.questionID == "q\(n)" }) { n += 1 }
    draft.questions.append(.init(questionID: "q\(n)", kind: kind))
  }
  func editID(_ id: UUID, text: String) {
    guard let index = draft.questions.firstIndex(where: { $0.id == id }) else { return }
    rowErrors[draft.questions[index].questionID] = nil
    draft.questions[index].idTouched = !text.isEmpty
    draft.questions[index].questionID =
      text.isEmpty
      ? ReadQuestion.derivedID(
        draft.questions[index].instructions,
        taken: draft.questions.filter { $0.id != id }.map(\.questionID))
      : ReadQuestion.cleanID(text)
  }
  func editInstructions(_ id: UUID, text: String) {
    guard let index = draft.questions.firstIndex(where: { $0.id == id }) else { return }
    rowErrors[draft.questions[index].questionID] = nil
    draft.questions[index].instructions = text
    if !draft.questions[index].idTouched {
      draft.questions[index].questionID = ReadQuestion.derivedID(
        text,
        taken: draft.questions.filter { $0.id != id }.map(\.questionID))
    }
  }
  func duplicate(_ id: UUID) {
    guard let row = draft.questions.first(where: { $0.id == id }),
      draft.questions.count < (current?.maxQuestions ?? 64)
    else { return }
    var copy = row
    copy.id = UUID()
    var n = 2
    while draft.questions.contains(where: { $0.questionID == "\(row.questionID)_\(n)" }) { n += 1 }
    copy.questionID = "\(row.questionID)_\(n)"
    copy.idTouched = true
    let index = draft.questions.firstIndex { $0.id == id }!
    draft.questions.insert(copy, at: index + 1)
  }
  func move(_ id: UUID, by delta: Int) {
    guard let i = draft.questions.firstIndex(where: { $0.id == id }),
      draft.questions.indices.contains(i + delta)
    else { return }
    draft.questions.swapAt(i, i + delta)
  }
  @discardableResult func applyJSON(_ text: String) -> Bool {
    do {
      var parsed = try ReadDraft.parse(Data(text.utf8), maxQuestions: current?.maxQuestions ?? 64)
      // The questions tab and saved sets omit state; keep the loaded document.
      if try JSONDecoder().decode(ConversationValue.self, from: Data(text.utf8))["state"] == nil {
        parsed.state = draft.state
      }
      parsed.images = draft.images
      draft = parsed
      originalJSON = (try? draft.orderedJSON()) ?? ""
      jsonText = originalJSON
      error = nil
      questionsError = nil
      rowErrors = [:]
      return true
    } catch {
      questionsError = error.localizedDescription
      return false
    }
  }
  func beginJSON() {
    guard !hasUnappliedJSON else { return }
    originalJSON = (try? draft.orderedJSON()) ?? ""
    jsonText = originalJSON
  }
  func reset() {
    guard !busy && !saving && !importing else { return }
    draft = ReadDraft()
    jsonText = ""
    originalJSON = ""
    beginJSON()
    originalBody = draft.setBody
    originalOrdering = draft.ordering
    selectedSet = nil
    setName = ""
    fileName = ""
    sessionEpoch += 1
    activeSession = nil
    sessionRevision = ""
    openingSession = false
    historyUnsaved = false
    runs = []
    latestRequest = nil
    latestOrdering = nil
    latestRunID = nil
    selectedRun = nil
    error = nil
    clearRunErrors()
  }
  func open(_ set: SavedSet) {
    guard !busy && !saving && !importing else { return }
    do {
      var parsed = try ReadDraft.parse(
        Data(set.body.utf8), maxQuestions: current?.maxQuestions ?? 1024)
      parsed.state = draft.state
      parsed.images = draft.images
      draft = parsed
      originalBody = parsed.setBody
      originalOrdering = parsed.ordering
      jsonText = ""
      originalJSON = ""
      beginJSON()
      selectedSet = set
      setName = set.name
      error = nil
      clearRunErrors()
    } catch { self.error = error.localizedDescription }
  }
  func save(asNew: Bool = false) async {
    guard !saving, !busy, validation == nil else { return }
    guard !hasUnappliedJSON else {
      error = "Apply the JSON questions before saving."
      return
    }
    let name = setName.trimmingCharacters(in: .whitespacesAndNewlines)
    let submittedName = setName
    guard !name.isEmpty && name.utf8.count <= 512 else {
      error = "Enter a set name up to 512 bytes."
      return
    }
    saving = true
    defer { saving = false }
    do {
      let body = draft.setBody
      let ordering = draft.ordering
      let serialized = try draft.orderedJSON()
      guard serialized.utf8.count <= 512 * 1024 else { throw ConversationFailure.tooLarge }
      let record: ConversationValue = .object([
        "id": .string(asNew ? UUID().uuidString : selectedSet?.id ?? UUID().uuidString),
        "name": .string(name), "body": .string(serialized),
        "revision": .string(asNew ? "" : selectedSet?.revision ?? ""),
      ])
      let reply = try await api("api/reads", "POST", record, [:])
      guard let value = reply["set"] else {
        throw ConversationFailure.invalid("The saved set was not acknowledged.")
      }
      let saved = try decode(SavedSet.self, value)
      setsEpoch += 1
      sets = [saved] + sets.filter { $0.id != saved.id }
      selectedSet = saved
      originalBody = body
      originalOrdering = ordering
      if setName == submittedName { setName = saved.name }
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func remove() async {
    guard !saving, !busy, let set = selectedSet else { return }
    saving = true
    defer { saving = false }
    do {
      guard ConversationDocument.validID(set.id) else {
        throw ConversationFailure.invalid("Invalid saved set identity.")
      }
      _ = try await api("api/reads/\(set.id)", "DELETE", nil, ["revision": set.revision])
      setsEpoch += 1
      sets.removeAll { $0.id == set.id }
      selectedSet = nil
      setName = ""
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func run(applyJSON: Bool = false) {
    if applyJSON, hasUnappliedJSON, !self.applyJSON(jsonText) { return }
    guard canRun, let reader = current else {
      if let validation { questionsError = validation }
      return
    }
    let request = draft.request(model: requestModel)
    let input = draft
    let name = requestModel
    guard draft.state.utf8.count <= 4 * 1024 * 1024 else {
      stateError = "The text exceeds the 4 MiB reading limit. Use a smaller section."
      return
    }
    let questions = draft.questions
    let submittedFileName = fileName
    let submittedSamples = draft.samples
    let submittedCheckpoint = draft.checkpoint
    let submittedPictures = draft.images
    let submittedSteps = draft.steps
    let submittedThink = draft.think
    busy = true
    error = nil
    clearRunErrors()
    task = Task {
      defer {
        busy = false
        trimHistory()
      }
      do {
        let started = ContinuousClock.now
        let bytes = try await Task.detached(priority: .userInitiated) {
          try input.requestData(model: name)
        }.value
        guard bytes.count <= ReadHistoryAdmission.requestLimit else {
          throw ConversationFailure.invalid(
            "This read is too large to save safely. Use fewer images or a smaller text section.")
        }
        try Task.checkCancellation()
        let raw = try await readAPI(reader.port, bytes)
        try Task.checkCancellation()
        let response = try decode(ReadResponse.self, raw)
        try response.validate(for: questions)
        let elapsed = started.duration(to: .now)
        let milliseconds =
          Double(elapsed.components.seconds) * 1000
          + Double(elapsed.components.attoseconds) / 1e15
        let fingerprint = await Task.detached(priority: .utility) { Run.fingerprint(bytes) }.value
        try Task.checkCancellation()
        let result = Run(
          fingerprint: fingerprint,
          excerpt: String(
            (request["state"]?.string ?? "").split(whereSeparator: \.isWhitespace)
              .joined(separator: " ").prefix(120)),
          characters: request["state"]?.string?.count ?? 0,
          questions: questions, raw: raw, port: reader.port, elapsedMilliseconds: milliseconds,
          response: response, state: request["state"]?.string,
          fileName: submittedFileName, samples: submittedSamples,
          pictures: submittedPictures, steps: submittedSteps, think: submittedThink,
          checkpoint: submittedCheckpoint)
        latestRequest = request
        latestOrdering = input.ordering
        latestRunID = result.id
        runs.insert(result, at: 0)
        selectedRun = result.id
        await keepRun(result)
      } catch is CancellationError {} catch { routeError(error.localizedDescription) }
    }
  }
  func cancel() { task?.cancel() }
  var canReadCamera: Bool {
    current?.images == true && !historyNavigationBlocked && !historyUnsaved && validation == nil
      && draft.state.utf8.count <= 4 * 1024 * 1024
  }
  /// Same transport and response validation as a text read, but no automatic
  /// history write. Freeze all metadata before awaiting the runner; stale
  /// results never migrate to a newly selected model/read/question set.
  func readCameraFrame(_ jpeg: Data) async throws -> Run {
    guard canReadCamera, let reader = current else {
      throw ConversationFailure.invalid(validation ?? "Choose a reader with vision.")
    }
    let epoch = sessionEpoch
    let original = draft
    var input = draft
    input.images = [
      ReadPicture(name: "camera frame", url: "data:image/jpeg;base64," + jpeg.base64EncodedString())
    ]
    let name = requestModel
    let bytes = try await Task.detached(priority: .userInitiated) {
      try input.requestData(model: name)
    }.value
    try Task.checkCancellation()
    guard epoch == sessionEpoch, port == reader.port, original == draft else {
      throw CancellationError()
    }
    let started = ContinuousClock.now
    let raw = try await readAPI(reader.port, bytes)
    try Task.checkCancellation()
    guard epoch == sessionEpoch, port == reader.port, original == draft else {
      throw CancellationError()
    }
    let response = try decode(ReadResponse.self, raw)
    try response.validate(for: input.questions)
    let duration = started.duration(to: .now)
    let fingerprint = await Task.detached(priority: .utility) { Run.fingerprint(bytes) }.value
    try Task.checkCancellation()
    guard epoch == sessionEpoch, port == reader.port, original == draft else {
      throw CancellationError()
    }
    return Run(
      fingerprint: fingerprint, excerpt: "camera frame", characters: input.state.count,
      questions: input.questions, raw: raw, port: reader.port,
      elapsedMilliseconds: Double(duration.components.seconds) * 1000 + Double(
        duration.components.attoseconds) / 1e15,
      response: response, state: input.state, samples: input.samples, pictures: input.images,
      steps: input.steps, think: input.think, checkpoint: input.checkpoint)
  }
  func keepCameraFrame(_ frame: Run) async {
    guard !historyNavigationBlocked, !historyUnsaved else { return }
    var saved = frame
    saved.id = UUID()
    runs.insert(saved, at: 0)
    selectedRun = saved.id
    await keepRun(saved)
  }
  func runExample() {
    guard current != nil, !busy, !saving, !importing, !openingSession else { return }
    do {
      let example = try ReadDraft.example
      reset()
      draft = example
      beginJSON()
      run()
    } catch { self.error = error.localizedDescription }
  }
  // Bound in-memory results. Durable history is the shared SQLite document.
  func trimHistory(maxBytes: Int = 16 * 1024 * 1024) {
    // Never reclaim unacknowledged results, or the one the user is reading.
    guard !historyUnsaved, !busy, !saving, !openingSession else { return }
    var used = 0
    let selected = result?.id
    var keep = Set<UUID>()
    for run in runs where run.id == selected || run.id == latestRunID {
      keep.insert(run.id)
      used += run.retainedBytes
    }
    for run in runs where !keep.contains(run.id) {
      let bytes = run.retainedBytes
      if used + bytes <= maxBytes {
        keep.insert(run.id)
        used += bytes
      }
    }
    runs = runs.filter { keep.contains($0.id) }
    if let latestRunID, !keep.contains(latestRunID) {
      latestRequest = nil
      latestOrdering = nil
      self.latestRunID = nil
    }
  }
  func settle() async {
    await task?.value
    await imageImportTask?.value
  }
  var hasEvictedRuns: Bool { (activeSession?.runs.count ?? 0) > runs.count && !historyUnsaved }
  func reloadCachedRuns() async {
    guard !closed, !historyNavigationBlocked, !historyUnsaved, let doc = activeSession else {
      return
    }
    let epoch = sessionEpoch
    openingSession = true
    defer { if epoch == sessionEpoch { openingSession = false } }
    do {
      let loaded = try await historyLoader.hydrate(doc)
      guard !closed, epoch == sessionEpoch, activeSession?.id == doc.id, !Task.isCancelled else {
        return
      }
      runs = loaded.runs.reversed()
      // This is cache refill, not navigation: keep the draft and selection.
    } catch is CancellationError {} catch {
      guard !closed, epoch == sessionEpoch else { return }
      historyError = error.localizedDescription
    }
  }
  func shutdown() async {
    guard !closed else { return }
    closed = true
    task?.cancel()
    imageImportTask?.cancel()
    sessionEpoch += 1
    await closeConnection()
    await settle()
  }
  func clearRunErrors() {
    stateError = nil
    questionsError = nil
    rowErrors = [:]
  }
  func routeError(_ message: String) {
    if message.hasPrefix("question \""),
      let expression = try? NSRegularExpression(pattern: #"^question \"((?:[^\"\\]|\\.)+)\":"#),
      let match = expression.firstMatch(
        in: message, range: NSRange(message.startIndex..., in: message)),
      let range = Range(match.range(at: 1), in: message)
    {
      let raw = String(message[range])
      let id = (try? JSONDecoder().decode(String.self, from: Data(("\"" + raw + "\"").utf8))) ?? raw
      if draft.questions.contains(where: { $0.questionID == id }) {
        rowErrors[id] = message
      } else {
        error = message
      }
    } else if message.hasPrefix("state:") || message.contains("the window is") {
      stateError = message
    } else if message.hasPrefix("images:") {
      stateError = message
    } else if message.hasPrefix("questions:") || message.hasPrefix("samples:")
      || message.contains("the answer template needs")
    {
      questionsError = message
    } else {
      error = message
    }
  }
  func refreshHistory() async {
    guard !closed else { return }
    let epoch = sessionEpoch
    historyGeneration += 1
    let generation = historyGeneration
    do {
      let rows = try await api("api/read-history", "GET", nil, [:])
      guard epoch == sessionEpoch, generation == historyGeneration else { return }
      sessions = try decode([Session].self, rows)
      historyLoaded = true
      historyListError = nil
    } catch is CancellationError {} catch {
      guard epoch == sessionEpoch, generation == historyGeneration else { return }
      historyLoaded = true
      historyListError = "Read history could not be loaded: \(error.localizedDescription)"
    }
  }

  private func keepRun(_ run: Run) async {
    guard !closed else { return }
    let at = (run.at.timeIntervalSince1970 * 1000).rounded()
    var fields: [String: ConversationValue] = [
      "id": .string(run.id.uuidString), "at": .number(Decimal(at)),
      "model": .string(run.response.model),
      "port": .number(Decimal(run.port)), "excerpt": .string(run.excerpt),
      "chars": .number(Decimal(run.characters)), "state": .string(run.state ?? ""),
      "fileName": .string(run.fileName),
      "questions": .object(
        Dictionary(
          run.questions.map { ($0.questionID, $0.wire(in: run.questions)) },
          uniquingKeysWith: { _, b in b })),
      "questionOrder": .array(
        run.questions.map { q in
          .array(
            ([q.questionID]
              + (q.kind == .choice
                ? q.options.map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines) } : []))
              .map(ConversationValue.string))
        }),
      "samples": run.samples == 0 ? .string("auto") : .number(Decimal(run.samples)),
      "response": run.raw, "ms": .number(Decimal(run.elapsedMilliseconds)),
    ]
    if !run.pictures.isEmpty { fields["images"] = .array(run.pictures.map(\.historyReference)) }
    if run.steps > 1 { fields["steps"] = .number(Decimal(run.steps)) }
    if run.think > 0 { fields["think"] = .number(Decimal(run.think)) }
    if let checkpoint = run.checkpoint { fields["checkpoint"] = .string(checkpoint) }
    let value = ConversationValue.object(fields)
    let title =
      run.fileName.isEmpty
      ? String(
        (run.state ?? "").split(separator: "\n").first.map(String.init)?.prefix(60)
          ?? run.pictures.first?.name.prefix(60) ?? "Untitled read")
      : run.fileName
    let previous = activeSession
    do {
      let admission = try await Task.detached(priority: .utility) {
        try ReadHistoryAdmission.append(value, pictures: run.pictures, title: title, to: previous)
      }.value
      guard !closed else { return }
      if admission.rolledOver {
        sessionRevision = ""
        // Older results remain in their original SQLite history entry.
        runs = [run]
      }
      activeSession = admission.document
      historyUnsaved = true
      await saveHistory()
    } catch let overflow as ReadHistoryAdmission.Overflow {
      guard !closed else { return }
      activeSession = overflow.document
      sessionRevision = ""
      historyUnsaved = true
      historyError =
        "This result is retained in memory but exceeds the history limit. Export it before starting another read."
    } catch {
      guard !closed else { return }
      historyUnsaved = true
      historyError = error.localizedDescription
    }
  }

  func saveHistory() async {
    guard !closed, !saving, let doc = activeSession else { return }
    saving = true
    defer {
      saving = false
      trimHistory()
    }
    let epoch = sessionEpoch
    do {
      let json = try await Task.detached(priority: .utility) { try doc.json }.value
      try checkOpen()
      guard json.utf8.count <= 16 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
      let reply = try await api(
        "api/read-history/\(doc.id)", "PUT",
        .object(["doc": .string(json)]), ["envelope": "true", "revision": sessionRevision])
      guard epoch == sessionEpoch else { return }
      guard let revision = reply["read"]?["revision"]?.string else {
        throw ConversationFailure.invalid("Read history was not acknowledged.")
      }
      sessionRevision = revision
      historyUnsaved = false
      historyError = nil
      await refreshHistory()
    } catch is CancellationError {} catch {
      guard !closed, epoch == sessionEpoch else { return }
      historyError =
        "The result is available, but history could not be saved: \(error.localizedDescription)"
    }
  }

  func openSession(_ id: String) async {
    guard !closed, !busy, !saving, !importing, ConversationDocument.validID(id) else { return }
    sessionEpoch += 1
    let epoch = sessionEpoch
    let before = draft
    openingSession = true
    defer { if epoch == sessionEpoch { openingSession = false } }
    do {
      let snapshot = try await api("api/read-history/\(id)", "GET", nil, ["envelope": "true"])
      guard let text = snapshot["doc"]?.string, let revision = snapshot["revision"]?.string else {
        throw ConversationFailure.invalid("Invalid read history response.")
      }
      let loaded = try await historyLoader.load(text)
      let doc = loaded.document
      guard doc.id == id else {
        throw ConversationFailure.invalid("The read identity does not match its address.")
      }
      guard !closed, epoch == sessionEpoch, draft == before, !Task.isCancelled else { return }
      if let input = loaded.draft, let last = loaded.runs.last {
        draft = input
        fileName = last.fileName
        if readers.contains(where: { $0.port == last.port }) { port = last.port }
      }
      activeSession = doc
      sessionRevision = revision
      runs = loaded.runs.reversed()
      selectedRun = runs.first?.id
      latestRunID = runs.first?.id
      latestRequest = draft.request(
        model: runs.first?.response.diagnostics.backend == "laya"
          ? "" : (runs.first?.response.model ?? ""))
      latestOrdering = draft.ordering
      selectedSet = nil
      setName = ""
      jsonText = ""
      originalJSON = ""
      beginJSON()
      originalBody = draft.setBody
      originalOrdering = draft.ordering
      historyUnsaved = false
      historyError = nil
      clearRunErrors()
    } catch is CancellationError {} catch {
      guard !closed, epoch == sessionEpoch else { return }
      historyError = error.localizedDescription
    }
  }

  func clearHistory() async {
    guard let id = activeSession?.id else { return }
    await removeSession(id)
  }

  func renameSession(_ id: String, title: String) async {
    let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !historyNavigationBlocked, ConversationDocument.validID(id), !title.isEmpty else {
      return
    }
    guard title.utf8.count <= 512 else {
      historyError = "Use a shorter read title (at most 512 UTF-8 bytes)."
      return
    }
    guard id != activeSession?.id || !historyUnsaved else {
      historyError = "Save this read before renaming it."
      return
    }
    saving = true
    defer { saving = false }
    do {
      let snapshot = try await sessionSnapshot(id)
      guard id != activeSession?.id || snapshot.revision == sessionRevision else {
        throw ConversationFailure.invalid("This read changed elsewhere. Reopen it before renaming.")
      }
      var fields = snapshot.doc.value.object ?? [:]
      fields["title"] = .string(title)
      let renamed = ReadHistoryDocument(value: .object(fields))
      let json = try await Task.detached(priority: .utility) { try renamed.json }.value
      let reply = try await api(
        "api/read-history/\(id)", "PUT", .object(["doc": .string(json)]),
        ["envelope": "true", "revision": snapshot.revision])
      guard let revision = reply["read"]?["revision"]?.string else {
        throw ConversationFailure.invalid("Read history was not acknowledged.")
      }
      historyGeneration += 1
      sessions = sessions.map { row in
        row.id == id
          ? Session(
            id: id, title: title, model: row.model, runs: row.runs, updatedAt: row.updatedAt)
          : row
      }
      if activeSession?.id == id {
        activeSession = renamed
        sessionRevision = revision
      }
      historyError = nil
    } catch { historyError = error.localizedDescription }
  }

  func removeSession(_ id: String) async {
    guard !historyNavigationBlocked, ConversationDocument.validID(id) else { return }
    saving = true
    defer { saving = false }
    do {
      let revision: String
      if activeSession?.id == id {
        revision = sessionRevision
      } else {
        revision = try await sessionSnapshot(id).revision
      }
      _ = try await api("api/read-history/\(id)", "DELETE", nil, ["revision": revision])
      historyGeneration += 1
      sessions.removeAll { $0.id == id }
      if activeSession?.id == id {
        // The confirmed deletion also clears its editor, like the web sidebar.
        saving = false
        reset()
      }
      historyError = nil
    } catch { historyError = error.localizedDescription }
  }

  private func sessionSnapshot(_ id: String) async throws -> (
    doc: ReadHistoryDocument, revision: String
  ) {
    let snapshot = try await api("api/read-history/\(id)", "GET", nil, ["envelope": "true"])
    guard let text = snapshot["doc"]?.string, let revision = snapshot["revision"]?.string else {
      throw ConversationFailure.invalid("Invalid read history response.")
    }
    let doc = try await Task.detached(priority: .utility) { try ReadHistoryDocument(json: text) }
      .value
    guard doc.id == id else {
      throw ConversationFailure.invalid("The read identity does not match its address.")
    }
    return (doc, revision)
  }
  func addPictures(_ urls: [URL]) async {
    await beginPictures(urls.map(NativeReadPictures.Source.file))?.value
  }

  /// Synchronous routing reserves the import before AppKit returns from Paste,
  /// preventing repeated commands from starting overlapping decode batches.
  @discardableResult func pastePictures(_ board: NSPasteboard) -> Bool {
    guard NativeReadClipboard.containsImages(board) else { return false }
    guard canImportPictures else { return true }
    do {
      let sources = try NativeReadClipboard.snapshot(board, remaining: 16 - draft.images.count)
      _ = beginPictures(sources)
    } catch { stateError = error.localizedDescription }
    return true
  }

  private var canImportPictures: Bool {
    guard !historyNavigationBlocked else {
      stateError = "Wait for the current operation before attaching images."
      return false
    }
    guard current?.images == true else {
      stateError = "Start this model with vision to read images."
      return false
    }
    return true
  }

  private func beginPictures(_ sources: [NativeReadPictures.Source]) -> Task<Void, Never>? {
    guard !sources.isEmpty, canImportPictures else { return nil }
    guard sources.count + draft.images.count <= 16 else {
      stateError = "A read takes up to 16 images."
      return nil
    }
    importing = true
    let work = Task {
      defer {
        importing = false
        imageImportTask = nil
      }
      do {
        var added: [ReadPicture] = []
        var bytes = draft.images.reduce(0) { $0 + $1.url.utf8.count }
        for source in sources {
          try Task.checkCancellation()
          let picture = try await Task.detached(priority: .userInitiated) {
            try source.prepare()
          }.value
          bytes += picture.url.utf8.count
          guard bytes <= 8 * 1024 * 1024 else {
            throw ConversationFailure.invalid(
              "These images exceed the read's 8 MiB storage budget. Choose fewer images.")
          }
          added.append(picture)
        }
        try Task.checkCancellation()
        draft.images += added
        stateError = nil
      } catch is CancellationError {} catch { stateError = error.localizedDescription }
    }
    imageImportTask = work
    return work
  }
  func loadFile(_ url: URL, asJSON: Bool = false) async {
    // "Load a file" is extraction, like web Reads. Images reach the vision
    // attachment path only through Add images, paste, or a vision-aware drop.
    guard !importing && !busy else { return }
    importing = true
    defer { importing = false }
    let old = draft.state
    do {
      let bytes = try await Task.detached(priority: .userInitiated) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let cap = asJSON ? 512 * 1024 : 32 * 1024 * 1024
        let data = try file.read(upToCount: cap + 1) ?? Data()
        guard data.count <= cap else { throw ConversationFailure.tooLarge }
        return data
      }.value
      try Task.checkCancellation()
      if asJSON {
        applyJSON(String(decoding: bytes, as: UTF8.self))
        return
      }
      let ext = url.pathExtension.lowercased()
      let plain = [
        "txt", "md", "markdown", "csv", "tsv", "json", "log", "xml", "html", "htm", "yaml", "yml",
        "toml", "eml",
      ]
      let text: String
      let decoded: String?
      if plain.contains(ext) {
        decoded = await Task.detached(priority: .userInitiated) {
          String(data: bytes, encoding: .utf8)
        }.value
      } else {
        decoded = nil
      }
      if let utf8 = decoded {
        text = utf8
      } else {
        guard let reader = current else {
          throw ConversationFailure.invalid("Start a reading model before extracting this file.")
        }
        let mime = UTType(filenameExtension: ext)?.preferredMIMEType ?? "application/octet-stream"
        let encoded = await Task.detached(priority: .userInitiated) {
          "data:\(mime);base64,\(bytes.base64EncodedString())"
        }.value
        try Task.checkCancellation()
        let reply = try await api(
          "api/runners/\(reader.port)/extract", "POST",
          .object([
            "filename": .string(url.lastPathComponent),
            "data": .string(encoded),
            "file_metadata": .string("off"),
          ]), [:])
        guard let value = reply["text"]?.string else {
          throw ConversationFailure.invalid("The file did not produce readable text.")
        }
        text = value
      }
      guard draft.state == old else {
        throw ConversationFailure.invalid(
          "The text changed while the file was loading. Your edits were kept.")
      }
      guard text.utf8.count <= 4 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
      draft.state = text
      fileName = url.lastPathComponent
      error = nil
    } catch is CancellationError {} catch { self.error = error.localizedDescription }
  }
}

extension NativeReadsModel.Run {
  init(from decoder: any Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(UUID.self, forKey: .id)
    at = try c.decode(Date.self, forKey: .at)
    fingerprint = try c.decode(String.self, forKey: .fingerprint)
    excerpt = try c.decode(String.self, forKey: .excerpt)
    characters = try c.decode(Int.self, forKey: .characters)
    questions = try c.decode([ReadQuestion].self, forKey: .questions)
    raw = try c.decode(ConversationValue.self, forKey: .raw)
    port = try c.decode(UInt16.self, forKey: .port)
    elapsedMilliseconds = try c.decode(Double.self, forKey: .elapsedMilliseconds)
    state = try c.decodeIfPresent(String.self, forKey: .state)
    fileName = try c.decodeIfPresent(String.self, forKey: .fileName) ?? ""
    samples = try c.decodeIfPresent(Int.self, forKey: .samples) ?? 0
    pictures = try c.decodeIfPresent([ReadPicture].self, forKey: .pictures) ?? []
    steps = try c.decodeIfPresent(Int.self, forKey: .steps) ?? 1
    think = try c.decodeIfPresent(Int.self, forKey: .think) ?? 0
    checkpoint = try c.decodeIfPresent(String.self, forKey: .checkpoint)
    response = try JSONDecoder().decode(ReadResponse.self, from: JSONEncoder().encode(raw))
  }
}
