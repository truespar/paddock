import Foundation

extension NativeStudioRuntime {
  /// Enrich a completed transcript, never a currently changing live turn. The
  /// same Rust endpoint performs attribution for both Studios. Only metadata
  /// changes; speech text, word clocks and recognition confidence are retained.
  func identifySpeakers(_ p: O) async throws {
    guard p["conversationId"]?.string == document?.id,
      let message = document?.activeMessages.first(where: { $0["id"] == p["messageId"] }),
      message["streaming"]?.bool != true, let meta = message["transcript"]?.object,
      let parent = document?.messages.first(where: { $0["id"] == message["parentId"] }),
      let clip = parent["content"]?.array?.first(where: { $0["type"]?.string == "audio" })?.object,
      let lane = models.first(where: {
        $0["id"] == p["modelId"] && $0["port"] == p["port"] && $0["kind"]?.string == "diarizer"
          && $0["status"]?.string == "ok"
      }), let port = lane["port"]?.integer
    else { throw ConversationFailure.stale }
    let conversationID = document?.id
    let messageID = try Self.id(message["id"])
    let aid = try Self.id(clip["attachmentId"])
    let bytes = try await transport.bytes("api/attachments/\(aid)", maximum: 25 * 1024 * 1024)
    let boundary = UUID().uuidString
    var body = Data()
    func field(_ name: String, _ value: Data) {
      body.append(
        Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
      body.append(value)
      body.append(Data("\r\n".utf8))
    }
    field("preset", Data("offline".utf8))
    field("model", Data((lane["id"]?.string ?? "").utf8))
    let words = meta["words"]?.array ?? []
    field("words", try JSONEncoder().encode(words))
    let mime = clip["mime"]?.string ?? "audio/wav"
    guard !mime.contains("\r"), !mime.contains("\n") else {
      throw ConversationFailure.invalid("Invalid audio type")
    }
    body.append(
      Data(
        "--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording\"\r\nContent-Type: \(mime)\r\n\r\n"
          .utf8))
    body.append(bytes)
    body.append(Data("\r\n--\(boundary)--\r\n".utf8))
    let response = try await transport.bytes(
      "api/runners/\(port)/v1/audio/diarizations",
      method: "POST", body: body, contentType: "multipart/form-data; boundary=\(boundary)")
    let result = try JSONDecoder().decode(O.self, from: response)
    try Task.checkCancellation()
    let enriched = try NativeSpeechDiarization.merge(meta: meta, result: result)
    guard document?.id == conversationID,
      document?.messages.first(where: { $0["id"]?.string == messageID })?["transcript"]
        == .object(meta)
    else { throw ConversationFailure.stale }
    let before = document
    try updateMessage(messageID) { $0["transcript"] = .object(enriched) }
    do { try await persist() } catch {
      document = before
      throw error
    }
  }
}

public enum NativeSpeechDiarization {
  public typealias O = [String: ConversationValue]
  public static func merge(meta: O, result: O) throws -> O {
    let original = meta["words"]?.array ?? []
    guard result["attribution"]?.string == "time_overlap_v1",
      let words = result["words"]?.array, words.count == original.count,
      let duration = result["duration"]?.double, duration.isFinite, duration > 0, duration <= 600,
      let segments = result["segments"]?.array,
      segments.allSatisfy({ s in
        guard let speaker = s["speaker"]?.integer, (0..<8).contains(speaker),
          let a = s["start"]?.double, let b = s["end"]?.double
        else { return false }
        return a.isFinite && b.isFinite && a >= 0 && b > a && b <= duration + 0.001
      }),
      zip(original, words).allSatisfy({ a, b in
        a["word"] == b["word"] && (a["start"] ?? .null) == (b["start"] ?? .null)
          && (a["end"] ?? .null) == (b["end"] ?? .null)
      })
    else { throw ConversationFailure.invalid("Speaker analysis did not match this transcript") }
    var out = meta
    out["words"] = .array(words)
    var timeline = result
    timeline["words"] = nil  // single authoritative word list in persisted metadata
    out["diarization"] = .object(timeline)
    return out
  }
}
