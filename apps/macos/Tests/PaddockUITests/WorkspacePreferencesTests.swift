import Foundation
import PaddockConversationCore
import Testing

@testable import PaddockClient
@testable import PaddockStudio
@testable import PaddockUI

@Suite("SQLite workspace preferences", .serialized) @MainActor
struct WorkspacePreferencesTests {
  @Test(
    .enabled(if: ProcessInfo.processInfo.environment["PADDOCK_WORKSPACE_PREFS_TEST_ROOT"] != nil))
  func realRustSQLiteRestoresFavoritesAndLayoutAcrossCoreReopen() async throws {
    let env = ProcessInfo.processInfo.environment
    let root = try #require(env["PADDOCK_WORKSPACE_PREFS_TEST_ROOT"])
    #expect(root.hasPrefix("/tmp/paddock-workspace-preferences."))
    guard root.hasPrefix("/tmp/paddock-workspace-preferences."), env["PADDOCK_DATA"] == root else {
      return
    }
    let library = URL(fileURLWithPath: try #require(env["PADDOCK_DESKTOP_TEST_LIBRARY"]))
    let firstClient = NativeManager(libraryURL: library)
    let first = NativeWorkspacePreferences(client: firstClient)
    await first.start()
    #expect(first.loaded && first.error == nil)
    first.toggleFavorite("cloud:test:model@provider")
    first.usedModels(["qwen", "cloud:test:model@provider"])
    first.set(.studioWidth, .number(315))
    first.set(.settingsPage, .string("Downloads"))
    try await first.flush()
    await first.close()
    await firstClient.close()
    let nextClient = NativeManager(libraryURL: library)
    let next = NativeWorkspacePreferences(client: nextClient)
    await next.start()
    #expect(next.loaded && next.error == nil)
    #expect(next.favorites == ["cloud:test:model@provider"])
    #expect(next.recentModels == ["qwen", "cloud:test:model@provider"])
    #expect(next.values[.studioWidth] == .number(315))
    #expect(next.values[.settingsPage] == .string("Downloads"))
    await next.close()
    await nextClient.close()
  }

  @Test func lateRestorePreservesInteractionAndPatchesOnlyOwnedKeys() async throws {
    let disk = PreferencesDisk([
      "studio.pk_theme": .string("dark"),
      "macos.workspace.studioWidth": .number(310),
      "macos.workspace.settingsPage": .string("Downloads"),
      "macos.workspace.recentModels": .array([.string("older")]),
      "macos.workspace.favorites": .array([.string("kept")]),
    ])
    let store = preferences(disk)
    let loading = Task { await store.start() }
    while await disk.reads == 0 { await Task.yield() }
    store.set(.studioWidth, .number(290))
    store.usedModels(["current"])
    await loading.value
    #expect(store.values[.studioWidth] == .number(290))
    #expect(store.values[.settingsPage] == .string("Downloads"))
    #expect(store.favorites == ["kept"])
    #expect(store.recentModels == ["current", "older"])
    try await store.flush()
    #expect(await disk.value("studio.pk_theme") == .string("dark"))
    #expect(
      await disk.writtenKeys == ["macos.workspace.studioWidth", "macos.workspace.recentModels"])
    await store.close()
  }

  @Test func inFlightWritesAreSerializedAndFailuresKeepEditsForRetry() async throws {
    let disk = PreferencesDisk([:])
    let store = preferences(disk)
    await store.start()
    store.toggleFavorite("cloud:one:model@provider")
    let first = Task { try await store.flush() }
    while await disk.writes == 0 { await Task.yield() }
    store.toggleFavorite("local")
    async let second: Void = store.flush()
    try await first.value
    try await second
    #expect(await disk.maximumConcurrent == 1)
    await disk.fail(true)
    store.set(.settingsSidebar, .bool(false))
    await #expect(throws: (any Error).self) { try await store.flush() }
    #expect(store.error != nil)
    await disk.fail(false)
    await store.retry()
    #expect(store.error == nil)
    let reopened = preferences(disk)
    await reopened.start()
    #expect(reopened.favorites == ["cloud:one:model@provider", "local"])
    #expect(reopened.values[.settingsSidebar] == .bool(false))
    await store.close()
    await reopened.close()
  }

  @Test func recentsAreBoundedUniqueAndDoNotChangeWhenBrowsing() async throws {
    let disk = PreferencesDisk([:])
    let store = preferences(disk)
    await store.start()
    store.usedModels((0..<20).map { "model-\($0)" })
    #expect(store.recentModels.count == 12)
    store.usedModels(["model-6", "model-2", "model-6"])
    #expect(Array(store.recentModels.prefix(3)) == ["model-6", "model-2", "model-0"])
    store.toggleFavorite("cloud:a:model@p")
    store.toggleFavorite("cloud:b:model@p")
    store.toggleFavorite("cloud:a:model@p")
    #expect(store.favorites == ["cloud:b:model@p"])
    try await store.flush()
    await store.close()
  }

  @Test func layoutRoundTripsWithoutRestoringTransientActionsOrInvalidDimensions() {
    var old = WorkspaceNavigation()
    old.sidebarVisible = true
    old.panelWidth = 333
    old.showManager(.downloads)
    old.sidebarVisible = false
    old.panelWidth = 245
    var next = WorkspaceNavigation()
    next.restore(old.preferences)
    #expect(next.mode == .studio && next.studio == .newChat)
    #expect(next.sidebarVisible && next.panelWidth == 333)
    #expect(next.manager == .downloads)
    next.mode = .manager
    #expect(!next.sidebarVisible && next.panelWidth == 245)
    next.restore([.settingsPage: .string("obsolete-page"), .settingsWidth: .number(99999)])
    #expect(next.manager == .downloads && next.panelWidth == 260)
    next.restore([.settingsWidth: .string("bad"), .settingsSidebar: .string("bad")])
    #expect(next.panelWidth == 260 && !next.sidebarVisible)
  }

  private func preferences(_ disk: PreferencesDisk) -> NativeWorkspacePreferences {
    NativeWorkspacePreferences(read: { await disk.read() }, write: { try await disk.write($0) })
  }
}

private actor PreferencesDisk {
  var data: [String: ConversationValue]
  var reads = 0
  var writes = 0
  var writtenKeys = Set<String>()
  var maximumConcurrent = 0
  private var concurrent = 0
  private var failing = false
  init(_ data: [String: ConversationValue]) { self.data = data }
  func value(_ key: String) -> ConversationValue? { data[key] }
  func fail(_ value: Bool) { failing = value }
  func read() async -> ConversationValue {
    reads += 1
    try? await Task.sleep(for: .milliseconds(40))
    return .object(data)
  }
  func write(_ patch: ConversationValue) async throws {
    writes += 1
    concurrent += 1
    maximumConcurrent = max(maximumConcurrent, concurrent)
    defer { concurrent -= 1 }
    try await Task.sleep(for: .milliseconds(30))
    if failing { throw ManagerError.core("Synthetic disk failure") }
    for (key, value) in patch.object ?? [:] {
      data[key] = value
      writtenKeys.insert(key)
    }
  }
}
