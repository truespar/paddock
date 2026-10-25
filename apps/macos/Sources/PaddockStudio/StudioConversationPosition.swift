import CryptoKit
import Foundation
import PaddockConversationCore

public struct StudioConversationViewState: Codable, Equatable, Sendable {
  public struct Reading: Codable, Equatable, Sendable {
    public var following: Bool
    public var blockID: String
    public var index: Int
    public var offset: Double
    public var height: Double
    public var width: Double
    public init(
      following: Bool, blockID: String = "", index: Int = 0, offset: Double = 0,
      height: Double = 0, width: Double = 0
    ) {
      self.following = following
      self.blockID = blockID
      self.index = index
      self.offset = offset
      self.height = height
      self.width = width
    }
  }
  public struct Composer: Codable, Equatable, Sendable {
    public var location: Int
    public var length: Int
    public var digest: String
    public init(selection: NSRange, text: String) {
      location = selection.location
      length = selection.length
      digest = Self.digest(text)
    }
    public func selection(in text: String) -> NSRange? {
      guard digest == Self.digest(text), location >= 0, length >= 0 else { return nil }
      let size = (text as NSString).length
      let start = min(location, size)
      let range = NSRange(location: start, length: min(length, size - start))
      // Keep surrogate pairs and combining characters intact after restoration.
      if range.length == 0, start < size {
        return NSRange(
          location: (text as NSString).rangeOfComposedCharacterSequence(at: start).location,
          length: 0)
      }
      return range.length == 0
        ? range : (text as NSString).rangeOfComposedCharacterSequences(for: range)
    }
    private static func digest(_ text: String) -> String {
      SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
  }
  public var version = 1
  public var reading: Reading?
  public var composer: Composer?
  public init() {}
}

/// Mutable presentation memory deliberately does not publish per-wheel/per-caret
/// changes into SwiftUI's transcript tree. Each view captures its own record.
@MainActor public final class StudioConversationPosition {
  public let id: String
  public private(set) var value = StudioConversationViewState()
  var loaded = false
  var revision = 0
  var savedRevision = 0
  var changed: (() -> Void)?
  public init(id: String) { self.id = id }
  public func reading(_ value: StudioConversationViewState.Reading) {
    guard self.value.reading != value else { return }
    self.value.reading = value
    revision &+= 1
    changed?()
  }
  public func composer(_ range: NSRange, text: String) {
    let size = (text as NSString).length
    let location = min(max(0, range.location), size)
    let safeRange = NSRange(location: location, length: min(max(0, range.length), size - location))
    let selection = StudioConversationViewState.Composer(selection: safeRange, text: text)
    guard value.composer != selection else { return }
    value.composer = selection
    revision &+= 1
    changed?()
  }
  func restore(_ value: StudioConversationViewState) {
    if revision == 0 { self.value = value }
    loaded = true
  }
}

@MainActor public final class StudioConversationPositions {
  typealias Load = @Sendable (String) async throws -> StudioConversationViewState?
  typealias Save = @Sendable (String, StudioConversationViewState) async throws -> Void
  private var records: [String: StudioConversationPosition] = [:]
  private var order: [String] = []
  private var load: Load?
  private var save: Save?
  private var transport: NativeConversationTransport?
  private var debounce: Task<Void, Never>?
  private var writer: Task<Void, Error>?
  public var onError: ((Error) -> Void)?
  public var onSaved: (() -> Void)?
  public init() {}
  init(load: @escaping Load, save: @escaping Save) {
    self.load = load
    self.save = save
  }
  deinit { debounce?.cancel() }

  func connect(_ transport: NativeConversationTransport) {
    self.transport = transport
    load = { id in
      let result = try await transport.api("api/conversations/\(id)/view-state/macos")
      guard result["version"]?.integer == 1 else { return nil }
      return try JSONDecoder().decode(
        StudioConversationViewState.self, from: JSONEncoder().encode(result))
    }
    save = { id, value in
      let body = try JSONDecoder().decode(ConversationValue.self, from: JSONEncoder().encode(value))
      _ = try await transport.api(
        "api/conversations/\(id)/view-state/macos", method: "PUT", body: body)
    }
  }
  public func record(_ id: String) -> StudioConversationPosition {
    if let record = records[id] { return record }
    let record = StudioConversationPosition(id: id)
    record.changed = { [weak self] in self?.schedule() }
    records[id] = record
    order.append(id)
    return record
  }
  func prepare(_ id: String) async throws {
    guard ConversationDocument.validID(id) else {
      throw ConversationFailure.invalid("Invalid conversation identity")
    }
    let record = record(id)
    order.removeAll { $0 == id }
    order.append(id)
    if !record.loaded {
      let value = try await load?(id)
      record.restore(value ?? StudioConversationViewState())
    }
    // Evict only durable records. A mounted view retains its own record even
    // while the store releases idle history; no rendered views are cached here.
    while order.count > 128,
      let candidate = order.first(where: {
        $0 != id && records[$0]?.revision == records[$0]?.savedRevision
      })
    {
      records[candidate] = nil
      order.removeAll { $0 == candidate }
    }
  }
  private func schedule() {
    debounce?.cancel()
    debounce = Task { [weak self] in
      do {
        try await Task.sleep(for: .milliseconds(600))
        try await self?.flush()
      } catch {
        // Cancellation is expected; the writer reports save failures.
      }
    }
  }
  func resave(_ id: String) {
    guard let record = records[id], record.value.composer != nil || record.value.reading != nil
    else { return }
    record.revision &+= 1
    schedule()
  }
  public func flush() async throws {
    debounce?.cancel()
    debounce = nil
    guard let save else { return }
    while let writer { try await writer.value }
    // One writer per workspace; delayed acknowledgements cannot mark later
    // caret/reading changes saved or overwrite another conversation's state.
    let task = Task { [self] in
      defer { writer = nil }
      do {
        while let record = records.values.first(where: { $0.revision != $0.savedRevision }) {
          let revision = record.revision
          let snapshot = record.value
          try await save(record.id, snapshot)
          record.savedRevision = revision
        }
        onSaved?()
      } catch {
        onError?(error)
        throw error
      }
    }
    writer = task
    try await task.value
  }
  func close() async {
    debounce?.cancel()
    debounce = nil
    await transport?.close()
    transport = nil
    load = nil
    save = nil
    for record in records.values { record.changed = nil }
    records.removeAll()
    order.removeAll()
  }
}
