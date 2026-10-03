import Foundation
import PaddockClient

extension NativeStudioRuntime {
  func refreshModels() async throws {
    async let runners = transport.api("api/runners")
    async let cloud = transport.api("api/cloud")
    let rows = (try await runners.array ?? []).compactMap(\.object).map {
      $0.filter { $0.value != .null }
    }
    var next: [O] = []
    for row in rows {
      guard
        let id = row["model"]?.string ?? row["asr"]?.string ?? row["embedder"]?.string
          ?? row["aligner"]?.string ?? row["image"]?.string ?? row["diarization"]?.string
      else { continue }
      let kind =
        row["model"]?.string != nil
        ? "chat"
        : row["asr"]?.string != nil
          ? "transcriber"
          : row["embedder"]?.string != nil
            ? "encoder"
            : row["image"]?.string != nil
              ? "image" : row["diarization"]?.string != nil ? "diarizer" : "aligner"
      var value: O = [
        "id": .string(id), "title": row["display"] ?? .string(id), "provider": .string("Local"),
        "vendor": row["vendor"] ?? .string(""), "port": row["port"] ?? .null,
        "status": row["status"] ?? .string("unknown"), "kind": .string(kind),
        "spec": row["spec"] ?? .string(""),
      ]
      if value["status"]?.string == "unreachable",
        let prior = models.first(where: { $0["id"] == value["id"] && $0["port"] == value["port"] }),
        tasks.values.count > 0
      {
        value["status"] = prior["status"]
      }
      if let old = models.first(where: { $0["id"]?.string == id }),
        old["port"] != value["port"] || old["status"] != value["status"]
      {
        caps[id] = nil
      }
      value["spec"] = .string(Self.recordedSpec(value))
      next.append(value)
    }
    if let endpoints = try? await cloud.array {
      for raw in endpoints
      where raw["hasKey"]?.bool == true || raw["allowUnauthenticated"]?.bool == true {
        let endpoint = (raw.object ?? [:]).filter { $0.value != .null }
        guard let eid = endpoint["id"]?.string, ConversationDocument.validID(eid) else { continue }
        for rawModel in endpoint["models"]?.array ?? [] {
          let model = (rawModel.object ?? [:]).filter { $0.value != .null }
          guard let name = model["id"]?.string else { continue }
          let provider = model["provider"]?.string
          let wire = name + (provider.map { "@\($0)" } ?? "")
          let id = "cloud:\(eid):\(wire)"
          let identity = CloudModelIdentity.resolve(
            id: name, display: model["display"]?.string, kind: endpoint["kind"]?.string ?? "",
            provider: provider)
          next.append([
            "id": .string(id), "title": .string(identity.name),
            "provider": endpoint["name"] ?? .string("Cloud"),
            "vendor": .string(identity.vendor ?? ""),
            "status": .string(
              endpoint["credentialReady"]?.bool == false ? "credential-unavailable" : "ok"),
            "kind": .string(model["asr"]?.bool == true ? "transcriber" : "chat"),
            "endpoint": .string(eid), "wireModel": .string(wire),
            "spec": .string(""),
          ])
          caps[id] = [
            "vision": model["vision"] ?? .bool(true), "max_ctx": model["ctx"] ?? .number(0),
            "default_max_output_tokens": model["maxOut"] ?? .null,
            "reasoning": .string(model["reasoning"]?.bool == true ? "toggle" : "none"),
            "reasoning_off": .bool(true),
            "web_search": .bool(
              endpoint["kind"]?.string == "openai-compat"
                && (endpoint["baseUrl"]?.string ?? "").contains("openrouter.ai")),
          ]
        }
      }
    } else {
      next.append(contentsOf: models.filter { $0["endpoint"] != nil })
    }
    models = next
    for model in next where model["status"]?.string == "ok" && caps[model["id"]!.string!] == nil {
      guard let port = model["port"]?.integer else { continue }
      if let value = try? await transport.api("api/runners/\(port)/server").object {
        caps[model["id"]!.string!] = value.filter { $0.value != .null }
      }
    }
  }
  func capability(_ id: String) -> O { caps[id] ?? [:] }
  func model(_ id: String) throws -> O {
    guard
      let value = models.first(where: { $0["id"]?.string == id && $0["status"]?.string == "ok" })
    else {
      throw ConversationFailure.invalid(
        "The selected model is not reachable. Start it in Manager or choose a running model.")
    }
    return value
  }
  func endpoint(_ model: O) throws -> NativeConversationTransport.Endpoint {
    if let id = model["endpoint"]?.string { return .cloud(id) }
    guard let port = model["port"]?.integer, let value = UInt16(exactly: port), value > 0 else {
      throw ConversationFailure.invalid("Invalid model endpoint")
    }
    return .runner(value)
  }
  func canAudio(_ id: String) -> Bool {
    models.first { $0["id"]?.string == id }?["kind"]?.string == "transcriber"
      || caps[id]?["audio"]?.bool == true
  }
  func canChat(_ id: String) -> Bool {
    models.first { $0["id"]?.string == id }?["kind"]?.string == "chat"
  }
  func canImagine(_ id: String) -> Bool {
    models.first { $0["id"]?.string == id }?["kind"]?.string == "image"
      && caps[id]?["image_generation"]?.object != nil
  }
  var contextLimit: Int {
    selected.compactMap { caps[$0]?["max_ctx"]?.integer }.filter { $0 > 0 }.min() ?? 0
  }
  func preferenceBool(_ name: String, fallback: Bool) -> Bool {
    preferences["pk_\(name)"]?.string.map { ["true", "1", "on"].contains($0) } ?? fallback
  }
  var maxTokens: V {
    preferences["pk_max_tokens"]?.string.flatMap(Int.init).flatMap {
      $0 > 0 ? .number(Decimal($0)) : nil
    } ?? .null
  }
}
