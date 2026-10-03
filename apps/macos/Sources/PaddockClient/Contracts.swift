import Foundation

// Read projections of Rust routes.rs, readiness.rs, registry.rs/runtime.rs and
// supervisor.rs. Optional measurements stay optional; absence never means zero.
// Unknown string states are retained for forward-compatible, honest display.
public struct ManagerIdentity: Decodable, Sendable {
  public let role: String
  public let version: String
  public let build: String?
  public let registry: RegistryLocation?

  public struct RegistryLocation: Decodable, Sendable {
    public let enabled: Bool
    public let modelsDir: String
    public let diskFree: UInt64?
    public let diskTotal: UInt64?
  }
}

public struct Readiness: Decodable, Sendable {
  public let backend: String?
  public let state: String
  public let card: String?
  public let generation: String?
  public let os: String

  /// Show actionable failures, not qualification bookkeeping or a generic
  /// success badge. The backend's state remains available for diagnostics.
  public var warning: String? {
    switch state {
    case "driver-too-old": "Driver update needed"
    case "no-card": "Local serving unavailable"
    default: nil
    }
  }
}

public struct ModelCatalog: Decodable, Sendable {
  public let schema: Int
  public let models: [CatalogModel]
}

public struct CatalogModel: Decodable, Identifiable, Sendable {
  public let id: String
  public let display: String
  public let vendor: String?
  public let family: String?
  public let capability: [String]
  public let license: String?
  public let specs: ModelSpecs?
  public let artifacts: [CatalogArtifact]
  public let installed: Bool
  public let totalSize: UInt64
}

public struct ModelSpecs: Decodable, Sendable {
  public let publishedAt: String?
  public let publishedSource: String?
  public let about: String?
  public let params: String?
  public let context: String?
  public let contextMax: String?
  public let homepage: String?
  public let strengths: [String]?
  public let tradeoffs: [String]?
}

public struct CatalogArtifact: Decodable, Identifiable, Sendable {
  public let id: String
  public let kind: String
  public let format: String
  public let label: String
  public let quant: String?
  public let installed: Bool
  public let totalSize: UInt64
  public let runtime: ArtifactRuntime?
  public let source: ArtifactSource?
  public let backendSupported: Bool?
  public let required: Bool?
  public let `default`: Bool?

  /// Missing legacy metadata means CUDA, exactly as Rust's legacy_backends.
  /// This is a display fact, not a Swift admission/loader implementation.
  public func supports(backend: String?) -> Bool {
    guard let backend else { return false }
    if let backendSupported { return backendSupported }
    return (runtime?.backends ?? ["cuda"]).contains(backend)
  }

  /// Only an actual backend mismatch is unavailable. Legacy test-progress
  /// metadata must not hide or disable otherwise compatible model exports.
  public var supportNotice: String? {
    backendSupported == false ? "Unavailable on this backend" : nil
  }

  public var displayLabel: String {
    label.replacingOccurrences(of: " (Metal preview)", with: "")
      .replacingOccurrences(of: " (macOS preview)", with: "")
  }
}

public struct ArtifactRuntime: Decodable, Sendable {
  public let backends: [String]?
  public let checkpointDir: Bool?
  public let capability: [String]?
  public let embeddedVision: Bool?
  public let kvCacheDtype: String?
  public let experimental: Bool?
  public let qualification: String?
  public let note: String?
  public let defaultMaxCtx: Int?
  public let defaultMaxBatch: Int?
  public let companions: [String]?
  public let memory: Memory?

  public struct Memory: Decodable, Sendable {
    public let maxCtx: Int
    public let maxBatch: Int
  }
}

public struct ArtifactSource: Decodable, Sendable {
  public let repo: String
  public let revision: String
  public let license: String?
}

public struct RunnerInfo: Decodable, Identifiable, Sendable {
  public let port: UInt16
  public let pid: UInt32
  public let status: String
  public let model: String?
  public let embedder: String?
  public let asr: String?
  public let aligner: String?
  public let image: String?
  public let reader: String?
  public let tabular: String?
  public let diarization: String?
  public let display: String?
  public let endpoint: String
  public let version: String?
  public let inFlight: UInt64?
  public let origin: String?
  public let uptimeS: UInt64?
  public let pinned: Bool?

  // A replacement runner on the same port is a new identity.
  public var id: String { "\(port):\(pid)" }
  public var title: String {
    display ?? model ?? embedder ?? asr ?? aligner ?? image ?? reader ?? tabular ?? diarization
      ?? "Runner \(port)"
  }
  public var hasStudioSurface: Bool { diarization == nil }
  public var studioActionTitle: String {
    tabular != nil ? "Open Tables" : reader != nil ? "Open Reads" : "Open Studio"
  }
  public var studioActionSymbol: String {
    tabular != nil ? "tablecells" : reader != nil ? "list.bullet.clipboard" : "bubble"
  }
}

public struct ManagerSnapshot: Decodable, Sendable {
  public let identity: ManagerIdentity
  public let readiness: Readiness
  public let catalog: ModelCatalog
  public let runners: [RunnerInfo]
  public let servers: [ConfiguredEndpoint]?
  public let jobs: [ManagementJob]?
  public let gpu: GPUSnapshot?

  public init(
    identity: ManagerIdentity, readiness: Readiness, catalog: ModelCatalog, runners: [RunnerInfo],
    servers: [ConfiguredEndpoint] = [], jobs: [ManagementJob] = [], gpu: GPUSnapshot? = nil
  ) {
    self.identity = identity
    self.readiness = readiness
    self.catalog = catalog
    self.runners = runners
    self.servers = servers
    self.jobs = jobs
    self.gpu = gpu
  }
}

public enum ManagerWire {
  public static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: data)
  }
}
