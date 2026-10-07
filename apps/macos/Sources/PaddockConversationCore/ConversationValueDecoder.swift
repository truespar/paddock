import Foundation

/// Decode the native presentation directly, without creating a second JSON
/// document and reparsing all of its strings on each streamed delta.
public struct ConversationValueDecoder {
  public init() {}
  public func decode<T: Decodable>(_ type: T.Type, from value: ConversationValue) throws -> T {
    try ValueDecoder(value: value).read(type)
  }
}

/// Confined to its owner's actor. Entries live for one presentation only, and
/// reuse unchanged top-level fields and transcript messages (not arbitrary
/// tool payload subtrees). Reconciliation drops deleted conversations/branches.
public final class ConversationDecodingCache {
  fileprivate struct Entry {
    let raw: ConversationValue
    let decoded: Any
  }
  fileprivate var previous: [String: Entry] = [:]
  fileprivate var next: [String: Entry] = [:]
  public init() {}
  public func decode<T: Decodable>(_ type: T.Type, from value: ConversationValue) throws -> T {
    next.removeAll(keepingCapacity: true)
    defer {
      previous = next
      next = [:]
    }
    return try ValueDecoder(value: value, cache: self).read(type)
  }
  public func reclaim() {
    previous = [:]
    next = [:]
  }
}

private struct ValueDecoder: Decoder, SingleValueDecodingContainer {
  let value: ConversationValue
  var codingPath: [any CodingKey] = []
  var cache: ConversationDecodingCache?
  var userInfo: [CodingUserInfoKey: Any] { [:] }

  func read<T: Decodable>(_ type: T.Type) throws -> T {
    if type == ConversationValue.self { return value as! T }
    if type == Decimal.self, case .number(let number) = value { return number as! T }
    let eligible =
      codingPath.count == 1
      || (codingPath.count == 3 && codingPath[0].stringValue == "nativeTranscript"
        && codingPath[1].stringValue == "messages")
    let key = codingPath.map(\.stringValue).joined(separator: "/")
    if eligible, let entry = cache?.previous[key], entry.raw == value,
      let decoded = entry.decoded as? T
    {
      cache?.next[key] = entry
      // Preserve cached message children when the whole transcript is reused.
      if key == "nativeTranscript", let cache {
        for (child, entry) in cache.previous where child.hasPrefix("nativeTranscript/") {
          cache.next[child] = entry
        }
      }
      return decoded
    }
    let decoded = try T(from: self)
    if eligible { cache?.next[key] = .init(raw: value, decoded: decoded) }
    return decoded
  }
  func child(_ value: ConversationValue, key: any CodingKey) -> Self {
    Self(value: value, codingPath: codingPath + [key], cache: cache)
  }
  func failure(_ type: Any.Type) -> DecodingError {
    .typeMismatch(type, .init(codingPath: codingPath, debugDescription: "Unexpected native value"))
  }
  func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
    guard let object = value.object else { throw failure([String: ConversationValue].self) }
    return KeyedDecodingContainer(Keyed<Key>(decoder: self, values: object))
  }
  func unkeyedContainer() throws -> any UnkeyedDecodingContainer {
    guard let array = value.array else { throw failure([ConversationValue].self) }
    return Unkeyed(decoder: self, values: array)
  }
  func singleValueContainer() throws -> any SingleValueDecodingContainer { self }
  func decodeNil() -> Bool { value == .null }
  func decode(_ type: Bool.Type) throws -> Bool {
    guard let result = value.bool else { throw failure(type) }
    return result
  }
  func decode(_ type: String.Type) throws -> String {
    guard let result = value.string else { throw failure(type) }
    return result
  }
  func decode(_ type: Double.Type) throws -> Double {
    guard case .number(let number) = value else { throw failure(type) }
    let result = NSDecimalNumber(decimal: number).doubleValue
    guard result.isFinite else { throw failure(type) }
    return result
  }
  func decode(_ type: Float.Type) throws -> Float {
    let result = Float(try decode(Double.self))
    guard result.isFinite else { throw failure(type) }
    return result
  }
  private func integer<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
    guard case .number(let number) = value,
      let result = T(NSDecimalNumber(decimal: number).stringValue)
    else { throw failure(type) }
    return result
  }
  func decode(_ type: Int.Type) throws -> Int { try integer(type) }
  func decode(_ type: Int8.Type) throws -> Int8 { try integer(type) }
  func decode(_ type: Int16.Type) throws -> Int16 { try integer(type) }
  func decode(_ type: Int32.Type) throws -> Int32 { try integer(type) }
  func decode(_ type: Int64.Type) throws -> Int64 { try integer(type) }
  func decode(_ type: UInt.Type) throws -> UInt { try integer(type) }
  func decode(_ type: UInt8.Type) throws -> UInt8 { try integer(type) }
  func decode(_ type: UInt16.Type) throws -> UInt16 { try integer(type) }
  func decode(_ type: UInt32.Type) throws -> UInt32 { try integer(type) }
  func decode(_ type: UInt64.Type) throws -> UInt64 { try integer(type) }
  func decode<T: Decodable>(_ type: T.Type) throws -> T { try read(type) }
}

private struct ValueKey: CodingKey {
  let stringValue: String
  let intValue: Int?
  init(_ index: Int) {
    stringValue = String(index)
    intValue = index
  }
  init?(stringValue: String) {
    self.stringValue = stringValue
    intValue = nil
  }
  init?(intValue: Int) { self.init(intValue) }
}

private struct Keyed<Key: CodingKey>: KeyedDecodingContainerProtocol {
  let decoder: ValueDecoder
  let values: [String: ConversationValue]
  var codingPath: [any CodingKey] { decoder.codingPath }
  var allKeys: [Key] { values.keys.compactMap(Key.init(stringValue:)) }
  func contains(_ key: Key) -> Bool { values[key.stringValue] != nil }
  private func child(_ key: Key) throws -> ValueDecoder {
    guard let value = values[key.stringValue] else {
      throw DecodingError.keyNotFound(
        key, .init(codingPath: codingPath, debugDescription: "Missing native field"))
    }
    return decoder.child(value, key: key)
  }
  func decodeNil(forKey key: Key) throws -> Bool { try child(key).decodeNil() }
  func decode(_ type: Bool.Type, forKey key: Key) throws -> Bool { try child(key).decode(type) }
  func decode(_ type: String.Type, forKey key: Key) throws -> String { try child(key).decode(type) }
  func decode(_ type: Double.Type, forKey key: Key) throws -> Double { try child(key).decode(type) }
  func decode(_ type: Float.Type, forKey key: Key) throws -> Float { try child(key).decode(type) }
  func decode(_ type: Int.Type, forKey key: Key) throws -> Int { try child(key).decode(type) }
  func decode(_ type: Int8.Type, forKey key: Key) throws -> Int8 { try child(key).decode(type) }
  func decode(_ type: Int16.Type, forKey key: Key) throws -> Int16 { try child(key).decode(type) }
  func decode(_ type: Int32.Type, forKey key: Key) throws -> Int32 { try child(key).decode(type) }
  func decode(_ type: Int64.Type, forKey key: Key) throws -> Int64 { try child(key).decode(type) }
  func decode(_ type: UInt.Type, forKey key: Key) throws -> UInt { try child(key).decode(type) }
  func decode(_ type: UInt8.Type, forKey key: Key) throws -> UInt8 { try child(key).decode(type) }
  func decode(_ type: UInt16.Type, forKey key: Key) throws -> UInt16 { try child(key).decode(type) }
  func decode(_ type: UInt32.Type, forKey key: Key) throws -> UInt32 { try child(key).decode(type) }
  func decode(_ type: UInt64.Type, forKey key: Key) throws -> UInt64 { try child(key).decode(type) }
  func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T {
    try child(key).read(type)
  }
  func nestedContainer<N: CodingKey>(keyedBy type: N.Type, forKey key: Key) throws
    -> KeyedDecodingContainer<N>
  {
    try child(key).container(keyedBy: type)
  }
  func nestedUnkeyedContainer(forKey key: Key) throws -> any UnkeyedDecodingContainer {
    try child(key).unkeyedContainer()
  }
  func superDecoder() throws -> any Decoder {
    decoder.child(values["super"] ?? .null, key: ValueKey(stringValue: "super")!)
  }
  func superDecoder(forKey key: Key) throws -> any Decoder { try child(key) }
}

private struct Unkeyed: UnkeyedDecodingContainer {
  let decoder: ValueDecoder
  let values: [ConversationValue]
  var currentIndex = 0
  var codingPath: [any CodingKey] { decoder.codingPath }
  var count: Int? { values.count }
  var isAtEnd: Bool { currentIndex >= values.count }
  private mutating func child() throws -> ValueDecoder {
    guard !isAtEnd else {
      throw DecodingError.valueNotFound(
        ConversationValue.self,
        .init(codingPath: codingPath, debugDescription: "End of native array"))
    }
    defer { currentIndex += 1 }
    return decoder.child(values[currentIndex], key: ValueKey(currentIndex))
  }
  mutating func decodeNil() throws -> Bool {
    guard !isAtEnd else {
      _ = try child()
      return false
    }
    if values[currentIndex] == .null {
      currentIndex += 1
      return true
    }
    return false
  }
  mutating func decode(_ type: Bool.Type) throws -> Bool { try child().decode(type) }
  mutating func decode(_ type: String.Type) throws -> String { try child().decode(type) }
  mutating func decode(_ type: Double.Type) throws -> Double { try child().decode(type) }
  mutating func decode(_ type: Float.Type) throws -> Float { try child().decode(type) }
  mutating func decode(_ type: Int.Type) throws -> Int { try child().decode(type) }
  mutating func decode(_ type: Int8.Type) throws -> Int8 { try child().decode(type) }
  mutating func decode(_ type: Int16.Type) throws -> Int16 { try child().decode(type) }
  mutating func decode(_ type: Int32.Type) throws -> Int32 { try child().decode(type) }
  mutating func decode(_ type: Int64.Type) throws -> Int64 { try child().decode(type) }
  mutating func decode(_ type: UInt.Type) throws -> UInt { try child().decode(type) }
  mutating func decode(_ type: UInt8.Type) throws -> UInt8 { try child().decode(type) }
  mutating func decode(_ type: UInt16.Type) throws -> UInt16 { try child().decode(type) }
  mutating func decode(_ type: UInt32.Type) throws -> UInt32 { try child().decode(type) }
  mutating func decode(_ type: UInt64.Type) throws -> UInt64 { try child().decode(type) }
  mutating func decode<T: Decodable>(_ type: T.Type) throws -> T { try child().read(type) }
  mutating func nestedContainer<N: CodingKey>(keyedBy type: N.Type) throws
    -> KeyedDecodingContainer<N>
  {
    try child().container(keyedBy: type)
  }
  mutating func nestedUnkeyedContainer() throws -> any UnkeyedDecodingContainer {
    try child().unkeyedContainer()
  }
  mutating func superDecoder() throws -> any Decoder { try child() }
}
