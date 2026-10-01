import CryptoKit
import Foundation

/// Versioned contract shared with studio/src/lib/table-history.ts. Sources are
/// deduplicated by SHA-256; every run refers to its immutable input snapshot.
public struct TableSession: Codable, Sendable {
  public var version = 1
  public var id: String
  public var title: String
  public var model: String
  public var createdAt: Int
  public var updatedAt: Int
  public var datasets: [String: String]
  public var draft: TableInput
  public var runs: [TableRun]
  public init(input: TableInput, source: String) {
    id = UUID().uuidString
    title = input.fileName.isEmpty ? "Untitled table" : input.fileName
    model = input.model
    createdAt = Int(Date().timeIntervalSince1970 * 1000)
    updatedAt = createdAt
    datasets = [input.dataset: source]
    draft = input
    runs = []
  }
}

public struct TableInput: Codable, Sendable, Equatable {
  public var dataset: String
  public var fileName: String
  public var spec: TabularSpec?
  public var model: String
  public var port: UInt16
  public var estimators: Int
  public var seed: Int
  public init(
    source: String, fileName: String, spec: TabularSpec?, model: String,
    port: UInt16, estimators: Int, seed: Int
  ) {
    dataset = SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined()
    self.fileName = fileName
    self.spec = spec
    self.model = model
    self.port = port
    self.estimators = estimators
    self.seed = seed
  }
}

public struct TableRun: Codable, Sendable, Identifiable {
  public var id: String
  public var at: Int
  public var input: TableInput
  public var task: TabularTask
  public var ms: Double
  public var response: ConversationValue
  public init(input: TableInput, task: TabularTask, ms: Double, response: ConversationValue) {
    id = UUID().uuidString
    at = Int(Date().timeIntervalSince1970 * 1000)
    self.input = input
    self.task = task
    self.ms = ms
    self.response = response
  }
}

public struct TableSummary: Codable, Sendable, Identifiable {
  public let id: String
  public let title: String
  public let model: String
  public let runs: Int
  public let createdAt: Int
  public let updatedAt: Int
  public let revision: String
}
