import Foundation

/// The persisted document is extensible, not a rendering projection. Retain
/// unknown fields (including tool output, attachments and provider metadata).
/// Decimal avoids rounding stored integer fields through a Double conversion.
public enum ConversationValue: Codable, Sendable, Equatable {
  case string(String)
  case number(Decimal)
  case bool(Bool)
  case array([Self])
  case object([String: Self])
  case null

  public init(from decoder: any Decoder) throws {
    let c = try decoder.singleValueContainer()
    if c.decodeNil() {
      self = .null
    } else if let v = try? c.decode(Bool.self) {
      self = .bool(v)
    } else if let v = try? c.decode(String.self) {
      self = .string(v)
    } else if let v = try? c.decode(Decimal.self) {
      self = .number(v)
    } else if let v = try? c.decode([Self].self) {
      self = .array(v)
    } else {
      self = .object(try c.decode([String: Self].self))
    }
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.singleValueContainer()
    switch self {
    case .string(let v): try c.encode(v)
    case .number(let v): try c.encode(v)
    case .bool(let v): try c.encode(v)
    case .array(let v): try c.encode(v)
    case .object(let v): try c.encode(v)
    case .null: try c.encodeNil()
    }
  }
  public var string: String? { if case .string(let v) = self { v } else { nil } }
  public var object: [String: Self]? { if case .object(let v) = self { v } else { nil } }
  public var array: [Self]? { if case .array(let v) = self { v } else { nil } }
  public var bool: Bool? { if case .bool(let v) = self { v } else { nil } }
  public var integer: Int? {
    guard case .number(let n) = self, n >= Decimal(Int.min), n <= Decimal(Int.max) else {
      return nil
    }
    let i = NSDecimalNumber(decimal: n).intValue
    return Decimal(i) == n ? i : nil
  }
  public subscript(_ key: String) -> Self? { object?[key] }
  /// Conservative admission cost, not a claim about the allocator's footprint.
  /// Avoid serializing/base64 copying cached results on the UI thread.
  public var estimatedRetainedBytes: Int {
    switch self {
    case .string(let value): return 32 + value.utf8.count
    case .array(let values): return 32 + values.reduce(0) { $0 + $1.estimatedRetainedBytes }
    case .object(let values):
      return 64 + values.reduce(0) { $0 + 64 + $1.key.utf8.count + $1.value.estimatedRetainedBytes }
    default: return 32
    }
  }
}

public enum ConversationFailure: Error, LocalizedError, Sendable, Equatable {
  case invalid(String)
  case stale, tooLarge, interrupted, closed
  case http(Int)
  public var errorDescription: String? {
    switch self {
    case .invalid(let why): why
    case .stale: "The conversation changed. Reopen the action; your draft has been kept."
    case .tooLarge: "The conversation or response exceeds its native safety limit."
    case .interrupted: "The response ended before a completion event arrived."
    case .closed: "The native conversation session is closed."
    case .http(let status): "The local conversation service returned HTTP \(status)."
    }
  }
}
