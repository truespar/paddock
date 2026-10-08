import Foundation

/// Saved endpoint identity, with no TOML, runner key or connector credentials.
public struct ConfiguredEndpoint: Decodable, Identifiable, Sendable {
  public let port: UInt16
  public let model: String?
  public let artifact: String?
  public let display: String?
  public let capability: [String]?
  public let running: Bool
  public let revision: String?
  public let localOnly: Bool?
  public let maxCtx: Int?
  public let maxBatch: Int?
  public let settings: EndpointSettings?
  public let configError: String?
  public let runtimeState: EndpointRuntimeState?
  public var id: UInt16 { port }
  public var title: String { display ?? model ?? "Endpoint \(port)" }
}

public struct EndpointRuntimeState: Decodable, Sendable {
  public let pid: UInt32
  public let restartRequired: Bool?
  public let residencyLive: Bool?
  public let changed: [String]
  public let maxCtx: Int
  public let maxBatch: Int
}

public struct ManagementJob: Decodable, Identifiable, Sendable {
  public let id: UInt64
  public let port: UInt16?
  public let action: String
  public let state: String
  public let message: String
  public var isActive: Bool { state == "running" }
}

public struct CreateEndpointRequest: Sendable, Encodable {
  public let model: String
  public let artifact: String
  public let port: UInt16?
  public let maxCtx: Int?
  public let maxBatch: Int?

  public init(
    model: String, artifact: String, port: UInt16? = nil, maxCtx: Int? = nil, maxBatch: Int? = nil,
    changes: [EndpointChange] = [], allowNetwork: Bool = false, tools: EndpointCreationTools? = nil
  ) {
    self.model = model
    self.artifact = artifact
    self.port = port
    self.maxCtx = maxCtx
    self.maxBatch = maxBatch
    self.changes = changes
    self.allowNetwork = allowNetwork
    self.tools = tools
  }
  public let changes: [EndpointChange]
  public let allowNetwork: Bool
  public let tools: EndpointCreationTools?
}

public struct EndpointCreationTools: Sendable, Encodable {
  public struct Connector: Sendable, Encodable {
    public let id: String
    public let revision: UInt64
    public init(id: String, revision: UInt64) {
      self.id = id
      self.revision = revision
    }
  }
  public let provider: String
  public let key: String
  public let connectors: [Connector]
  public init(provider: String, key: String, connectors: [Connector]) {
    self.provider = provider
    self.key = key
    self.connectors = connectors
  }
}

/// Deliberately not a method/URL/body tunnel to privileged management routes.
public enum ModelCommand: Sendable, Encodable {
  case prepare(model: String, artifact: String)
  case poll(id: UInt64)
  case create(CreateEndpointRequest)
  case start(port: UInt16, revision: String, allowNetwork: Bool = false)
  case stop(port: UInt16, pid: UInt32)
  case edit(
    port: UInt16, revision: String, pid: UInt32?, changes: [EndpointChange], apply: EndpointApply,
    allowNetwork: Bool)
  case remove(port: UInt16, revision: String)

  private enum CodingKeys: String, CodingKey {
    case kind, model, artifact, port, maxCtx, maxBatch, revision, pid, changes, apply, allowNetwork,
      id, tools
  }

  public func encode(to encoder: any Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .prepare(let model, let artifact):
      try values.encode("prepare", forKey: .kind)
      try values.encode(model, forKey: .model)
      try values.encode(artifact, forKey: .artifact)
    case .poll(let id):
      try values.encode("poll", forKey: .kind)
      try values.encode(id, forKey: .id)
    case .create(let request):
      try values.encode("create", forKey: .kind)
      try values.encode(request.model, forKey: .model)
      try values.encode(request.artifact, forKey: .artifact)
      try values.encodeIfPresent(request.port, forKey: .port)
      try values.encodeIfPresent(request.maxCtx, forKey: .maxCtx)
      try values.encodeIfPresent(request.maxBatch, forKey: .maxBatch)
      try values.encode(request.changes, forKey: .changes)
      try values.encode(request.allowNetwork, forKey: .allowNetwork)
      try values.encodeIfPresent(request.tools, forKey: .tools)
    case .start(let port, let revision, let allowNetwork):
      try values.encode("start", forKey: .kind)
      try values.encode(port, forKey: .port)
      try values.encode(revision, forKey: .revision)
      try values.encode(allowNetwork, forKey: .allowNetwork)
    case .stop(let port, let pid):
      try values.encode("stop", forKey: .kind)
      try values.encode(port, forKey: .port)
      try values.encode(pid, forKey: .pid)
    case .edit(let port, let revision, let pid, let changes, let apply, let allowNetwork):
      try values.encode("edit", forKey: .kind)
      try values.encode(port, forKey: .port)
      try values.encode(revision, forKey: .revision)
      try values.encodeIfPresent(pid, forKey: .pid)
      try values.encode(changes, forKey: .changes)
      try values.encode(apply, forKey: .apply)
      try values.encode(allowNetwork, forKey: .allowNetwork)
    case .remove(let port, let revision):
      try values.encode("remove", forKey: .kind)
      try values.encode(port, forKey: .port)
      try values.encode(revision, forKey: .revision)
    }
  }
}

public struct EndpointSettings: Decodable, Sendable, Equatable {
  public let host: String
  public let maxCtx: Int?
  public let maxBatch: Int?
  public let spec: String?
  public let noSpec: Bool?
  public let runtimeOptions: [EndpointRuntimeOption]?
  public let kvCacheDtype: String?
  public let hasApiKey: Bool
  public let vision: Bool
  public let audio: Bool?
  public let forensics: Bool
  public let device: String
  public let drafter: String?
  public let vramBudget: Int?
  public let kvOffload: EndpointKVOffload?
  public let kvOffloadSupported: Bool?
  public let residency: EndpointResidency?
  public let residencySupported: Bool?
}

public struct EndpointResidency: Codable, Sendable, Equatable {
  public var load: String
  public var unloadAfterIdleSeconds: Int?
  public var loadTimeoutSeconds: Int
  public init(
    load: String = "at_startup", unloadAfterIdleSeconds: Int? = nil, loadTimeoutSeconds: Int = 120
  ) {
    self.load = load
    self.unloadAfterIdleSeconds = unloadAfterIdleSeconds
    self.loadTimeoutSeconds = loadTimeoutSeconds
  }
}

public struct EndpointKVOffload: Codable, Sendable, Equatable {
  public var enabled: Bool
  public var ramGb: Double
  public var nvmeGb: Double
  public init(enabled: Bool, ramGb: Double, nvmeGb: Double) {
    self.enabled = enabled
    self.ramGb = ramGb
    self.nvmeGb = nvmeGb
  }
}

/// Registry identifiers, never arbitrary model or companion paths.
public struct EndpointComposition: Encodable, Sendable {
  public let model: String
  public let artifact: String
  public let vision: Bool
  public let audio: Bool?
  public let drafter: String?

  public init(model: String, artifact: String, vision: Bool, drafter: String?, audio: Bool? = nil) {
    self.model = model
    self.artifact = artifact
    self.vision = vision
    self.audio = audio
    self.drafter = drafter
  }
}

public enum EndpointApply: String, Encodable, Sendable { case `defer`, restart }

/// Only changed controls cross the boundary. Null restores a model default;
/// omission preserves the saved value, including settings unknown to this UI.
public enum EndpointChange: Encodable, Sendable {
  case maxCtx(Int?)
  case maxBatch(Int?)
  case spec(String?)
  case host(String)
  case apiKey(String)
  case forensics(Bool)
  case kvCacheDtype(String?)
  case vramBudget(Int?)
  case composition(EndpointComposition)
  case runtime([String: EndpointRuntimeValue?])
  case kvOffload(EndpointKVOffload)
  case residency(EndpointResidency)
  private enum CodingKeys: String, CodingKey { case field, value }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .maxCtx(let v):
      try c.encode("max_ctx", forKey: .field)
      try c.encode(v, forKey: .value)
    case .maxBatch(let v):
      try c.encode("max_batch", forKey: .field)
      try c.encode(v, forKey: .value)
    case .spec(let v):
      try c.encode("spec", forKey: .field)
      try c.encode(v, forKey: .value)
    case .host(let v):
      try c.encode("host", forKey: .field)
      try c.encode(v, forKey: .value)
    case .apiKey(let v):
      try c.encode("api_key", forKey: .field)
      try c.encode(v, forKey: .value)
    case .forensics(let v):
      try c.encode("forensics", forKey: .field)
      try c.encode(v, forKey: .value)
    case .kvCacheDtype(let v):
      try c.encode("kv_cache_dtype", forKey: .field)
      try c.encode(v, forKey: .value)
    case .vramBudget(let v):
      try c.encode("vram_budget", forKey: .field)
      try c.encode(v, forKey: .value)
    case .composition(let v):
      try c.encode("composition", forKey: .field)
      try c.encode(v, forKey: .value)
    case .runtime(let v):
      try c.encode("runtime", forKey: .field)
      try c.encode(v, forKey: .value)
    case .kvOffload(let v):
      try c.encode("kv_offload", forKey: .field)
      try c.encode(v, forKey: .value)
    case .residency(let v):
      try c.encode("residency", forKey: .field)
      try c.encode(v, forKey: .value)
    }
  }
}
