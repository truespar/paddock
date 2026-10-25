import Foundation
import Observation
import PaddockClient
import PaddockConversationCore

/// Small, non-secret UI preferences in the existing Rust SQLite settings store.
/// One app-owned writer patches only keys it changed; it never replaces the web
/// Studio's preferences or persists model credentials/catalog projections.
@MainActor @Observable public final class NativeWorkspacePreferences {
  public enum Key: String, CaseIterable, Sendable {
    case favorites, recentModels, studioSidebar, settingsSidebar
    case studioWidth, settingsWidth, settingsPage
    var wire: String { "macos.workspace.\(rawValue)" }
  }
  public private(set) var values: [Key: ConversationValue] = [:]
  public private(set) var loaded = false
  public private(set) var error: String?
  @ObservationIgnored public var onRestore: (() -> Void)?
  @ObservationIgnored private let client: (any ManagerLoading)?
  @ObservationIgnored private var transport: NativeConversationTransport?
  @ObservationIgnored private var read: (@Sendable () async throws -> ConversationValue)?
  @ObservationIgnored private var write: (@Sendable (ConversationValue) async throws -> Void)?
  @ObservationIgnored private var pending: [Key: ConversationValue] = [:]
  @ObservationIgnored private var loading: Task<Void, Error>?
  @ObservationIgnored private var writer: Task<Void, Error>?
  @ObservationIgnored private var debounce: Task<Void, Never>?
  @ObservationIgnored private var closed = false
  @ObservationIgnored private var started = false

  public init(client: any ManagerLoading) { self.client = client }
  init(
    read: @escaping @Sendable () async throws -> ConversationValue,
    write: @escaping @Sendable (ConversationValue) async throws -> Void
  ) {
    client = nil
    self.read = read
    self.write = write
  }
  deinit { debounce?.cancel() }

  public var favorites: Set<String> { Set(ids(.favorites)) }
  public var recentModels: [String] { ids(.recentModels) }
  private func ids(_ key: Key) -> [String] {
    var seen = Set<String>()
    return (values[key]?.array ?? []).compactMap { value in
      guard let id = value.string, !id.isEmpty, id.utf8.count <= 2048,
        seen.insert(id).inserted
      else { return nil }
      return id
    }.prefix(key == .recentModels ? 12 : 256).map { $0 }
  }
  public func toggleFavorite(_ id: String) {
    guard !id.isEmpty, id.utf8.count <= 2048 else { return }
    var next = favorites
    if next.contains(id) {
      next.remove(id)
    } else if next.count < 256 {
      next.insert(id)
    } else {
      error = "Remove a favorite before adding more than 256 models."
      return
    }
    set(.favorites, .array(next.sorted().map(ConversationValue.string)))
  }
  public func usedModels(_ ids: [String]) {
    var seen = Set<String>()
    let next = (ids + recentModels).filter {
      !$0.isEmpty && $0.utf8.count <= 2048 && seen.insert($0).inserted
    }.prefix(12)
    set(.recentModels, .array(next.map(ConversationValue.string)))
  }
  public func set(_ key: Key, _ value: ConversationValue) {
    guard !closed, values[key] != value else { return }
    values[key] = value
    pending[key] = value
    schedule()
  }
  public func start() async {
    started = true
    do { try await load() } catch {
      self.error = "Could not load app preferences: \(error.localizedDescription)"
    }
  }
  private func load() async throws {
    guard !closed, !loaded else { return }
    if let loading { return try await loading.value }
    let task = Task { [self] in
      defer { loading = nil }
      if read == nil, let client {
        let host = try await client.nativeConversationHost()
        guard !closed else { throw CancellationError() }
        let transport = try NativeConversationTransport(host: host)
        self.transport = transport
        read = { try await transport.api("api/settings") }
        write = { _ = try await transport.api("api/settings", method: "PUT", body: $0) }
      }
      guard let read else { throw CancellationError() }
      let saved = try await read()
      guard !closed else { throw CancellationError() }
      for key in Key.allCases where pending[key] == nil {
        values[key] = saved[key.wire]
      }
      // A model can accept a message while preferences are still loading.
      // Keep that use first without losing the previously saved recents.
      if let current = pending[.recentModels]?.array {
        var seen = Set<String>()
        let merged = (current + (saved[Key.recentModels.wire]?.array ?? [])).compactMap {
          $0.string
        }
        .filter { !$0.isEmpty && $0.utf8.count <= 2048 && seen.insert($0).inserted }.prefix(12)
        let value = ConversationValue.array(merged.map(ConversationValue.string))
        values[.recentModels] = value
        pending[.recentModels] = value
      }
      loaded = true
      error = nil
      onRestore?()
      if !pending.isEmpty { schedule() }
    }
    loading = task
    try await task.value
  }
  private func schedule() {
    guard started else { return }
    debounce?.cancel()
    debounce = Task { [weak self] in
      do {
        try await Task.sleep(for: .milliseconds(400))
        try await self?.flush()
      } catch {
        // flush exposes errors; cancellation is expected while editing.
      }
    }
  }
  public func flush() async throws {
    debounce?.cancel()
    debounce = nil
    guard !closed, started else { return }
    // Unused test/preview workspaces need no service just to close.
    guard !pending.isEmpty || writer != nil else { return }
    do {
      try await load()
      while let writer { try await writer.value }
      guard let write else { throw CancellationError() }
      let task = Task { [self] in
        defer { writer = nil }
        while !pending.isEmpty {
          let patch = pending
          try await write(
            .object(Dictionary(uniqueKeysWithValues: patch.map { ($0.key.wire, $0.value) })))
          for (key, value) in patch where pending[key] == value { pending[key] = nil }
        }
      }
      writer = task
      try await task.value
      error = nil
    } catch {
      self.error = "Could not save app preferences: \(error.localizedDescription)"
      throw error
    }
  }
  public func retry() async {
    do {
      try await load()
      try await flush()
    } catch { self.error = "Could not save app preferences: \(error.localizedDescription)" }
  }
  public func close() async {
    closed = true
    debounce?.cancel()
    loading?.cancel()
    await transport?.close()
    _ = await loading?.result
    _ = await writer?.result
    transport = nil
    read = nil
    write = nil
  }
}
