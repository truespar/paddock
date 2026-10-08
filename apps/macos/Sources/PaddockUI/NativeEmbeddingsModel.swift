import Foundation
import Observation
import PaddockClient
import PaddockConversationCore
import UniformTypeIdentifiers

private actor EmbeddingsConnection {
  let client: any ManagerLoading
  var transport: NativeConversationTransport?
  init(client: any ManagerLoading) { self.client = client }
  func call(_ path: String, _ method: String, _ body: ConversationValue?) async throws
    -> ConversationValue
  {
    if transport == nil {
      transport = try NativeConversationTransport(host: await client.nativeConversationHost())
    }
    let data = try await transport!.bytes(
      path, method: method, body: body.map { try JSONEncoder().encode($0) })
    return try JSONDecoder().decode(ConversationValue.self, from: data)
  }
}

/// Same /v1 endpoints and capability discovery as web Embeddings. Media bytes,
/// JSON serialization and pairwise comparisons never run on the main actor.
@MainActor @Observable final class NativeEmbeddingsModel {
  struct Endpoint: Identifiable, Equatable, Sendable {
    let port: UInt16
    let pid: Int
    let model: String
    let title: String
    let vendor: String?
    let reranker: Bool
    let inputs: Set<String>
    let dimensions: [Int]
    let tasks: [String]
    var id: String { "\(port):\(pid)" }
  }
  struct Medium: Identifiable, Sendable {
    let id = UUID()
    let name: String
    let kind: String
    let data: Data
    let mime: String
    let format: String
    var frames: [NativeEmbeddingVideo.Frame] = []
    var byteCount: Int { data.count + frames.reduce(0) { $0 + $1.png.count } }
    var item: ConversationValue {
      if kind == "video" {
        return .object([
          "content": .array([
            .object([
              "type": .string("input_video"),
              "input_video": .object([
                "add_timestamps": .bool(false),
                "frames": .array(
                  frames.map { frame in
                    .object([
                      "timestamp": .number(Decimal(frame.timestamp)),
                      "image_url": .string(
                        "data:image/png;base64,\(frame.png.base64EncodedString())"),
                    ])
                  }),
              ]),
            ])
          ])
        ])
      }
      let part: ConversationValue =
        kind == "image"
        ? .object([
          "type": .string("image_url"),
          "image_url": .object(["url": .string("data:\(mime);base64,\(data.base64EncodedString())")]
          ),
        ])
        : .object([
          "type": .string("input_audio"),
          "input_audio": .object([
            "data": .string(data.base64EncodedString()), "format": .string(format),
          ]),
        ])
      return .object(["content": .array([part])])
    }
  }
  struct Output: Sendable {
    let endpoint: Endpoint
    let labels: [String]
    let vectors: [[Double]]
    let similarity: [[Double]]
    let rankings: [(index: Int, score: Double)]
    let tokens: Int?
    let milliseconds: Double
    let json: Data
  }
  typealias API = @MainActor (String, String, ConversationValue?) async throws -> ConversationValue
  @ObservationIgnored var api: API
  private(set) var endpoints: [Endpoint] = []
  var port: UInt16 = 0 {
    didSet {
      if port != oldValue {
        task = ""
        dimensions = 0
      }
    }
  }
  var text = ""
  var query = ""
  var task = ""
  var dimensions = 0
  private(set) var media: [Medium] = []
  private(set) var result: Output?
  private(set) var error: String?
  private(set) var discoveryError: String?
  private(set) var loading = false
  private(set) var importing = false
  private(set) var busy = false
  @ObservationIgnored private var runTask: Task<Void, Never>?
  @ObservationIgnored private var importTask: Task<Void, Never>?
  private var attempt = UUID()
  var current: Endpoint? { endpoints.first { $0.port == port } }
  var lines: [String] { Self.lines(text) }
  var hasWork: Bool { !text.isEmpty || !query.isEmpty || !media.isEmpty || busy || importing }
  var canRun: Bool {
    guard let current, !busy, !importing else { return false }
    let count = lines.count + (current.reranker ? 0 : media.count)
    return (1...32).contains(count)
      && (!current.reranker || !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
      && (current.reranker || media.allSatisfy { current.inputs.contains($0.kind) })
  }
  init(client: any ManagerLoading) {
    let connection = EmbeddingsConnection(client: client)
    api = { path, method, body in try await connection.call(path, method, body) }
  }
  nonisolated static func lines(_ text: String) -> [String] {
    text.components(separatedBy: .newlines).map {
      $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }.filter { !$0.isEmpty }
  }
  func refresh() async {
    guard !loading, !busy else { return }
    loading = true
    defer { loading = false }
    do {
      let fleet = try await api("api/runners", "GET", nil).array ?? []
      var found: [Endpoint] = []
      var failures: [String] = []
      for runner in fleet {
        guard let name = runner["embedder"]?.string, runner["status"]?.string == "ok",
          let n = runner["port"]?.integer, let port = UInt16(exactly: n), port != 0
        else { continue }
        try Task.checkCancellation()
        do {
          let info = try await api("api/runners/\(port)/server", "GET", nil)
          found.append(
            Endpoint(
              port: port, pid: runner["pid"]?.integer ?? 0, model: name,
              title: runner["display"]?.string ?? name, vendor: runner["vendor"]?.string,
              reranker: info["reranker"]?.bool == true,
              inputs: Set(info["embedder_inputs"]?.array?.compactMap(\.string) ?? ["text"]),
              dimensions: info["embedder_dimensions"]?.array?.compactMap(\.integer) ?? [],
              tasks: info["embedder_tasks"]?.array?.compactMap(\.string) ?? []))
        } catch is CancellationError { throw CancellationError() } catch {
          failures.append("\(name): \(error.localizedDescription)")
        }
      }
      try Task.checkCancellation()
      endpoints = found
      if !found.contains(where: { $0.port == port }) { port = found.first?.port ?? 0 }
      if current?.dimensions.contains(dimensions) != true { dimensions = 0 }
      if current?.tasks.contains(task) != true { task = "" }
      discoveryError = failures.isEmpty ? nil : failures.joined(separator: "\n")
    } catch is CancellationError {} catch { discoveryError = error.localizedDescription }
  }
  func remove(_ id: UUID) { if !busy { media.removeAll { $0.id == id } } }
  func report(_ value: any Error) { error = value.localizedDescription }
  func importFiles(_ urls: [URL]) {
    guard !busy, !importing, let current else { return }
    importing = true
    let existingBytes = media.reduce(0) { $0 + $1.byteCount }
    let available = 32 - media.count - lines.count
    let worker = Task.detached(priority: .userInitiated) { () throws -> [Medium] in
      guard urls.count <= available else {
        throw ConversationFailure.invalid("Use at most 32 items per comparison.")
      }
      var items: [Medium] = []
      var bytes = existingBytes
      for url in urls {
        try Task.checkCancellation()
        guard url.isFileURL else {
          throw ConversationFailure.invalid("Choose a local image, audio or video file.")
        }
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        let info = try url.resourceValues(forKeys: [
          .contentTypeKey, .fileSizeKey, .isRegularFileKey,
        ])
        guard info.isRegularFile == true, let size = info.fileSize, size > 0,
          let type = info.contentType,
          type.conforms(to: .image) || type.conforms(to: .audio) || type.conforms(to: .movie)
        else {
          throw ConversationFailure.invalid(
            "Choose a local image, audio or video file.")
        }
        let kind =
          type.conforms(to: .movie) ? "video" : (type.conforms(to: .image) ? "image" : "audio")
        guard current.inputs.contains(kind) else {
          throw ConversationFailure.invalid(
            "This endpoint does not accept \(kind). Enable its tower in Settings > Instances.")
        }
        if kind == "video" {
          let frames = try await NativeEmbeddingVideo.decode(
            url, byteBudget: 32 * 1024 * 1024 - bytes)
          let medium = Medium(
            name: url.lastPathComponent, kind: kind, data: Data(),
            mime: "video", format: url.pathExtension.lowercased(), frames: frames)
          bytes += medium.byteCount
          items.append(medium)
          continue
        }
        guard size <= 24 * 1024 * 1024, bytes + size <= 32 * 1024 * 1024 else {
          throw ConversationFailure.tooLarge
        }
        // Bounded read also covers a file that grows after resourceValues.
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let data = try file.read(upToCount: 24 * 1024 * 1024 + 1) ?? Data()
        guard !data.isEmpty, data.count <= 24 * 1024 * 1024, bytes + data.count <= 32 * 1024 * 1024
        else { throw ConversationFailure.tooLarge }
        bytes += data.count
        items.append(
          Medium(
            name: url.lastPathComponent, kind: kind, data: data,
            mime: type.preferredMIMEType ?? "application/octet-stream",
            format: url.pathExtension.lowercased()))
      }
      return items
    }
    importTask = Task { [weak self] in
      guard let self else {
        worker.cancel()
        return
      }
      defer {
        importing = false
        importTask = nil
      }
      do {
        let items = try await withTaskCancellationHandler {
          try await worker.value
        } onCancel: {
          worker.cancel()
        }
        try Task.checkCancellation()
        guard self.current?.id == current.id else { throw ConversationFailure.stale }
        media += items
        error = nil
      } catch is CancellationError {} catch { report(error) }
    }
  }
  func cancel() {
    attempt = UUID()
    runTask?.cancel()
    runTask = nil
    importTask?.cancel()
    importTask = nil
    busy = false
  }
  func settle() async {
    await importTask?.value
    await runTask?.value
  }
  func run() {
    guard canRun, let current else { return }
    let text = lines
    let query = query
    let media = media
    let task = task
    let dimensions = dimensions
    let token = UUID()
    attempt = token
    busy = true
    error = nil
    runTask = Task { [weak self] in
      guard let self else { return }
      defer {
        if attempt == token {
          busy = false
          runTask = nil
        }
      }
      do {
        let builder = Task.detached { () throws -> ConversationValue in
          var body: [String: ConversationValue] = ["model": .string(current.model)]
          guard text.reduce(0, { $0 + $1.utf8.count }) + query.utf8.count <= 1024 * 1024 else {
            throw ConversationFailure.tooLarge
          }
          if current.reranker {
            body["query"] = .string(query)
            body["documents"] = .array(text.map(ConversationValue.string))
            body["return_documents"] = .bool(true)
          } else {
            body["input"] = .array(text.map(ConversationValue.string) + media.map(\.item))
            if !task.isEmpty { body["task"] = .string(task) }
            if dimensions != 0 { body["dimensions"] = .number(Decimal(dimensions)) }
          }
          return .object(body)
        }
        let body = try await withTaskCancellationHandler {
          try await builder.value
        } onCancel: {
          builder.cancel()
        }
        try Task.checkCancellation()
        let start = ContinuousClock.now
        let response = try await api(
          "api/runners/\(current.port)/v1/\(current.reranker ? "rerank" : "embeddings")", "POST",
          body)
        let elapsed = start.duration(to: .now).components
        let ms = Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
        let labels = text + (current.reranker ? [] : media.map(\.name))
        let parser = Task.detached {
          try Self.decode(response, endpoint: current, labels: labels, milliseconds: ms)
        }
        let output = try await withTaskCancellationHandler {
          try await parser.value
        } onCancel: {
          parser.cancel()
        }
        try Task.checkCancellation()
        if attempt == token { result = output }
      } catch is CancellationError {} catch { if attempt == token { report(error) } }
    }
  }
  nonisolated static func decode(
    _ response: ConversationValue, endpoint: Endpoint, labels: [String], milliseconds: Double
  ) throws -> Output {
    func number(_ v: ConversationValue?) throws -> Double {
      guard case .number(let n) = v else {
        throw ConversationFailure.invalid("Malformed encoder result.")
      }
      let value = NSDecimalNumber(decimal: n).doubleValue
      guard value.isFinite else { throw ConversationFailure.invalid("Nonfinite encoder result.") }
      return value
    }
    var vectors: [[Double]] = []
    var rankings: [(index: Int, score: Double)] = []
    if endpoint.reranker {
      guard let rows = response["results"]?.array, rows.count == labels.count else {
        throw ConversationFailure.invalid("Incomplete reranking result.")
      }
      var indices = Set<Int>()
      for row in rows {
        guard let i = row["index"]?.integer, labels.indices.contains(i), indices.insert(i).inserted
        else { throw ConversationFailure.invalid("Invalid reranking index.") }
        rankings.append((i, try number(row["relevance_score"])))
      }
    } else {
      guard let rows = response["data"]?.array, rows.count == labels.count else {
        throw ConversationFailure.invalid("Incomplete embedding result.")
      }
      var ordered = [Int: [Double]]()
      for row in rows {
        try Task.checkCancellation()
        guard let i = row["index"]?.integer, labels.indices.contains(i), ordered[i] == nil,
          let v = row["embedding"]?.array, !v.isEmpty, v.count <= 16384
        else { throw ConversationFailure.invalid("Invalid embedding index or dimensions.") }
        ordered[i] = try v.map { try number($0) }
      }
      vectors = try labels.indices.map { i in
        guard let v = ordered[i], v.count == ordered[0]?.count else {
          throw ConversationFailure.invalid("Embedding dimensions differ.")
        }
        return v
      }
    }
    let norms = vectors.map { sqrt($0.reduce(0) { $0 + $1 * $1 }) }
    let similarity = vectors.enumerated().map { i, a in
      vectors.enumerated().map { j, b in
        let den = norms[i] * norms[j]
        return den > 0 ? zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } / den : 0
      }
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return Output(
      endpoint: endpoint, labels: labels, vectors: vectors, similarity: similarity,
      rankings: rankings,
      tokens: response["usage"]?["total_tokens"]?.integer, milliseconds: milliseconds,
      json: try encoder.encode(response))
  }
}
