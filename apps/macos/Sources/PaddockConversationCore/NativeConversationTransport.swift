import Foundation
import PaddockClient

/// Pure native networking to the existing private Rust relay. No browser,
/// JavaScript, runner keys, external destinations, redirects or shared cookies.
public actor NativeConversationTransport: ConversationStorage {
  public enum Endpoint: Sendable {
    case runner(UInt16)
    case cloud(String)
    var path: String? {
      switch self {
      case .runner(let port): port == 0 ? nil : "api/runners/\(port)/v1/responses"
      case .cloud(let id): ConversationDocument.validID(id) ? "api/cloud/\(id)/v1/responses" : nil
      }
    }
  }
  private let host: StudioHost
  private let network: URLSession
  private var closed = false

  public init(host: StudioHost) throws {
    try self.init(host: host, configuration: .ephemeral)
  }
  // Test injection is internal. Product calls cannot select a proxy/session.
  init(host: StudioHost, configuration: URLSessionConfiguration) throws {
    guard host.isValidPrivateHost else {
      throw ConversationFailure.invalid("Invalid private conversation host")
    }
    self.host = host
    configuration.httpCookieStorage = nil
    configuration.httpShouldSetCookies = false
    configuration.urlCredentialStorage = nil
    configuration.urlCache = nil
    configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
    configuration.timeoutIntervalForRequest = 120
    configuration.timeoutIntervalForResource = 3600
    network = URLSession(
      configuration: configuration, delegate: RejectRedirects(), delegateQueue: nil)
  }
  public func close() {
    closed = true
    network.invalidateAndCancel()
  }
  deinit { network.invalidateAndCancel() }

  public var origin: URL { host.origin }
  public func realtime(port: UInt16) throws -> NativeRealtimeConnection {
    guard port > 0 else { throw ConversationFailure.invalid("Invalid speech endpoint") }
    var req = try request("api/runners/\(port)/v1/realtime")
    var url = URLComponents(url: req.url!, resolvingAgainstBaseURL: false)!
    url.scheme = "ws"
    url.queryItems = [.init(name: "intent", value: "transcription")]
    req.url = url.url
    let task = network.webSocketTask(with: req)
    task.maximumMessageSize = 4 * 1024 * 1024
    task.resume()
    return NativeRealtimeConnection(task: task)
  }

  /// All native service calls stay on the authenticated private origin. Query
  /// values are encoded separately; a caller cannot inject a host or traversal.
  public func api(
    _ path: String, method: String = "GET", body: ConversationValue? = nil,
    query: [String: String] = [:]
  ) async throws -> ConversationValue {
    let bytes = try await bytes(
      path, method: method,
      body: body.map { try JSONEncoder().encode($0) }, query: query)
    return bytes.isEmpty ? .null : try JSONDecoder().decode(ConversationValue.self, from: bytes)
  }
  public func bytes(
    _ path: String, method: String = "GET", body: Data? = nil,
    contentType: String = "application/json", query: [String: String] = [:],
    maximum: Int = 16 * 1024 * 1024
  ) async throws -> Data {
    guard path.hasPrefix("api/"), !path.contains(".."), !path.contains("%"),
      !path.contains("?"), !path.contains("#"), !path.contains("\\"),
      ["GET", "POST", "PUT", "DELETE"].contains(method)
    else {
      throw ConversationFailure.invalid("Invalid native service route")
    }
    var req = try request(path, method: method, body: body)
    var url = URLComponents(url: req.url!, resolvingAgainstBaseURL: false)!
    if !query.isEmpty {
      url.queryItems = query.sorted { $0.key < $1.key }.map { .init(name: $0.key, value: $0.value) }
    }
    req.url = url.url
    if body != nil { req.setValue(contentType, forHTTPHeaderField: "Content-Type") }
    return try await data(req, maximum: maximum)
  }

  private func request(_ path: String, method: String = "GET", body: Data? = nil) throws
    -> URLRequest
  {
    guard !closed else { throw ConversationFailure.closed }
    try Task.checkCancellation()
    var request = URLRequest(url: host.origin.appending(path: path))
    request.httpMethod = method
    request.httpBody = body
    request.setValue("\(host.cookieName)=\(host.session)", forHTTPHeaderField: "Cookie")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
    return request
  }
  private func status(_ response: URLResponse) throws -> HTTPURLResponse {
    guard let response = response as? HTTPURLResponse else {
      throw ConversationFailure.invalid("Invalid local service response")
    }
    guard (200..<300).contains(response.statusCode) else {
      throw ConversationFailure.http(response.statusCode)
    }
    return response
  }
  private func data(_ request: URLRequest, maximum: Int) async throws -> Data {
    let (bytes, response) = try await network.bytes(for: request)
    defer { bytes.task.cancel() }
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
      var errorData = Data()
      for try await byte in bytes {
        guard errorData.count < 64 * 1024 else { break }
        errorData.append(byte)
      }
      if let value = try? JSONDecoder().decode(ConversationValue.self, from: errorData),
        let message = value["error"]?["message"]?.string, !message.isEmpty
      {
        throw ConversationFailure.invalid(message)
      }
      throw ConversationFailure.http(http.statusCode)
    }
    let http = try status(response)
    guard http.expectedContentLength <= Int64(maximum) else { throw ConversationFailure.tooLarge }
    var data = Data()
    for try await byte in bytes {
      guard data.count < maximum else { throw ConversationFailure.tooLarge }
      data.append(byte)
    }
    try Task.checkCancellation()
    return data
  }
  public func loadConversation(_ id: String) async throws -> ConversationDocument {
    guard ConversationDocument.validID(id) else {
      throw ConversationFailure.invalid("Invalid conversation identity")
    }
    let data = try await data(
      request("api/conversations/\(id)"), maximum: ConversationDocument.maximumBytes)
    let doc = try ConversationDocument(data: data)
    guard doc.id == id else { throw ConversationFailure.invalid("Conversation identity mismatch") }
    return doc
  }
  public func saveConversation(_ document: ConversationDocument) async throws {
    try await saveConversation(document, consumingDraft: nil)
  }
  func saveConversation(_ document: ConversationDocument, consumingDraft: String?) async throws {
    var req = try request(
      "api/conversations/\(document.id)", method: "PUT", body: document.encoded())
    if let consumingDraft {
      req.setValue(consumingDraft, forHTTPHeaderField: "X-Paddock-Consume-Draft")
    }
    _ = try await data(
      req,
      maximum: 1024 * 1024)
  }
  public func listConversations() async throws -> [[String: ConversationValue]] {
    let data = try await data(request("api/conversations"), maximum: 16 * 1024 * 1024)
    return try JSONDecoder().decode([[String: ConversationValue]].self, from: data)
  }

  public func transcription(
    path: String, body: Data, boundary: String,
    receive: @Sendable ([String: ConversationValue]) async throws -> Void
  ) async throws -> [String: ConversationValue] {
    guard path.hasPrefix("api/runners/") || path.hasPrefix("api/cloud/"),
      path.hasSuffix("/v1/audio/transcriptions"), !path.contains(".."), !path.contains("%"),
      body.count <= 104 * 1024 * 1024, ConversationDocument.validID(boundary)
    else { throw ConversationFailure.invalid("Invalid transcription request") }
    var req = try request(path, method: "POST", body: body)
    req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    req.setValue("text/event-stream, application/json", forHTTPHeaderField: "Accept")
    let (bytes, response) = try await network.bytes(for: req)
    defer { bytes.task.cancel() }
    let http = try status(response)
    if http.mimeType != "text/event-stream" {
      var data = Data()
      for try await byte in bytes {
        guard data.count < 16 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
        data.append(byte)
      }
      return try JSONDecoder().decode([String: ConversationValue].self, from: data)
    }
    var parser = ServerSentEvents()
    for try await byte in bytes {
      if let frame = try parser.push(byte), frame.data != "[DONE]" {
        let event = try JSONDecoder().decode(
          [String: ConversationValue].self, from: Data(frame.data.utf8))
        try await receive(event)
        if event["type"]?.string == "transcript.text.done" { return event }
        if event["type"]?.string == "error" {
          throw ConversationFailure.invalid(
            event["error"]?["message"]?.string ?? "Transcription failed")
        }
      }
    }
    throw ConversationFailure.interrupted
  }

  /// Image SSE carries bounded encoded pictures, not Responses token events.
  /// Read with backpressure and finish on the semantic completed event.
  public func images(
    port: UInt16, body: [String: ConversationValue],
    references: [[String: ConversationValue]] = [],
    receive: @Sendable ([String: ConversationValue]) async throws -> Void
  ) async throws -> [String: ConversationValue] {
    guard port > 0 else { throw ConversationFailure.invalid("Invalid image endpoint") }
    let encoded = try JSONEncoder().encode(body)
    guard encoded.count <= 256 * 1024 else { throw ConversationFailure.tooLarge }
    let form =
      references.isEmpty ? nil : try await imageEditBody(fields: body, references: references)
    defer { form?.discard() }
    var req = try request(
      "api/runners/\(port)/v1/images/\(form == nil ? "generations" : "edits")", method: "POST",
      body: try form?.finish() ?? encoded)
    if let form {
      req.setValue(
        "multipart/form-data; boundary=\(form.boundary)", forHTTPHeaderField: "Content-Type")
    }
    req.timeoutInterval = 3600
    req.setValue("text/event-stream, application/json", forHTTPHeaderField: "Accept")
    let started = ContinuousClock.now
    let (bytes, response) = try await network.bytes(for: req)
    defer { bytes.task.cancel() }
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
      var data = Data()
      for try await byte in bytes {
        guard data.count < 64 * 1024 else { throw ConversationFailure.http(http.statusCode) }
        data.append(byte)
      }
      let problem = try? JSONDecoder().decode(ConversationValue.self, from: data)
      throw ConversationFailure.invalid(
        problem?["error"]?["message"]?.string ?? "Image generation failed (\(http.statusCode))")
    }
    let http = try status(response)
    if http.mimeType != "text/event-stream" {
      var data = Data()
      for try await byte in bytes {
        guard data.count < 256 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
        data.append(byte)
      }
      try Task.checkCancellation()
      return try JSONDecoder().decode([String: ConversationValue].self, from: data)
    }
    var parser = ServerSentEvents(maximumBytes: 64 * 1024 * 1024)
    var count = 0
    var firstPreviewMs: Double?
    for try await byte in bytes {
      if let frame = try parser.push(byte), frame.data != "[DONE]" {
        let event = try JSONDecoder().decode(
          [String: ConversationValue].self, from: Data(frame.data.utf8))
        if frame.name == "error" || event["error"] != nil {
          throw ConversationFailure.invalid(
            event["error"]?["message"]?.string ?? "Image generation failed")
        }
        if event["type"]?.string == "image_generation.partial_image" {
          count += 1
          if firstPreviewMs == nil {
            let elapsed = started.duration(to: .now).components
            firstPreviewMs = Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
          }
          guard count <= 3 else { throw ConversationFailure.invalid("Too many image previews") }
          try await receive(event)
        } else if event["type"]?.string == "image_generation.completed" {
          guard let image = event["b64_json"]?.string, !image.isEmpty else {
            throw ConversationFailure.interrupted
          }
          var reply = event
          if let firstPreviewMs { reply["firstPreviewMs"] = .number(Decimal(firstPreviewMs)) }
          reply["data"] = .array([.object(["b64_json": .string(image)])])
          try Task.checkCancellation()
          return reply
        }
      }
    }
    throw ConversationFailure.interrupted
  }

  /// Pull-based and awaited: no unbounded AsyncStream queue between transport
  /// and state reducer. Only the caller's coalesced presentation reaches UI.
  /// Returns on the semantic terminal event (the socket need not close first).
  public func responses(
    endpoint: Endpoint, body: [String: ConversationValue],
    maximumBytes: Int = 16 * 1024 * 1024,
    receive: @Sendable ([String: ConversationValue]) async throws -> Void
  ) async throws -> ResponseAccumulator {
    guard let path = endpoint.path, body["stream"]?.bool == true,
      let model = body["model"]?.string, !model.isEmpty
    else {
      throw ConversationFailure.invalid("Invalid native Responses request")
    }
    let encoded = try JSONEncoder().encode(body)
    guard encoded.count <= 192 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
    var request = try request(path, method: "POST", body: encoded)
    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
    let (bytes, response) = try await network.bytes(for: request)
    defer { bytes.task.cancel() }
    if let failed = response as? HTTPURLResponse, !(200..<300).contains(failed.statusCode) {
      var data = Data()
      for try await byte in bytes {
        if data.count >= 64 * 1024 {
          data.removeAll()
          break
        }
        data.append(byte)
      }
      var problem = (try? JSONDecoder().decode(ConversationValue.self, from: data))?.object
      problem = problem?["error"]?.object ?? problem
      var error: [String: ConversationValue] =
        problem ?? ["message": .string("The provider could not complete this request.")]
      if error["code"] == nil { error["code"] = .number(Decimal(failed.statusCode)) }
      throw ConversationFailure.invalid(
        String(decoding: try JSONEncoder().encode(error), as: UTF8.self))
    }
    let http = try status(response)
    guard http.mimeType?.lowercased() == "text/event-stream" else {
      throw ConversationFailure.invalid("The model did not return a response event stream")
    }
    var parser = ServerSentEvents(maximumBytes: min(maximumBytes, 4 * 1024 * 1024))
    var reducer = ResponseAccumulator(maximumBytes: maximumBytes)
    for try await byte in bytes {
      if let frame = try parser.push(byte), let event = try reducer.apply(frame.data) {
        try Task.checkCancellation()
        try await receive(event)
        if reducer.status != nil { return reducer }
      }
    }
    try Task.checkCancellation()
    parser.finish()
    try reducer.requireTerminal()
    return reducer
  }
}

private final class RejectRedirects: NSObject, URLSessionTaskDelegate {
  func urlSession(
    _ session: URLSession, task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
    completionHandler: @escaping @Sendable (URLRequest?) -> Void
  ) {
    completionHandler(nil)
  }
}
