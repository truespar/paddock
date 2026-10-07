import Foundation
import Observation
import PaddockClient
import PaddockNativeMarkdown
import PaddockStudio

@MainActor @Observable
public final class WorkspaceModel {
  public enum State: Equatable {
    case idle, loading, ready, stopped
    case failed(String)
  }

  public private(set) var state: State = .idle
  public private(set) var snapshot: ManagerSnapshot?
  public private(set) var refreshedAt: Date?
  public private(set) var isSubmitting = false
  public private(set) var commandError: String?
  public private(set) var lastReceipt: ManagementJob?
  @ObservationIgnored private var monitor: Task<Void, Never>?
  @ObservationIgnored private var memoryPressure: NativeMemoryPressure?
  @ObservationIgnored private let client: any ManagerLoading
  var settingsClient: any ManagerLoading { client }
  @ObservationIgnored private var pending: Task<ManagerSnapshot, any Error>?
  @ObservationIgnored private var generation: UInt64 = 0
  public let cloud: CloudBrowserModel
  public let downloads: DownloadsModel
  public let connections: ConnectionsModel
  public let integrations: IntegrationsModel
  let insights: InsightsModel
  let benchmarks: BenchmarksModel
  let dataStorage: DataStorageModel
  public var maintenanceInProgress: Bool { dataStorage.exporting || benchmarks.running }
  let speech: StudioSpeechModels
  var draft = StudioDraft()
  var navigation = WorkspaceNavigation()
  var showsRendererSamples = false
  var showsGPUMetrics = false
  var gpuHistory = GPUHistory()
  var gpuMeasurementsStale: Bool {
    if case .failed = state { return true }
    guard let gpu = snapshot?.gpu else { return false }
    return abs(Date().timeIntervalSince1970 - Double(gpu.ts)) > 10
  }
  var historySearchRequest: UUID?
  var selectedEndpointPort: UInt16?
  var systemToolsPort: UInt16?
  var detailEndpointPort: UInt16?
  var editingEndpoint = false
  var endpointEditor: EndpointEditor?
  let endpointLogs: EndpointLogsModel
  let studioLibrary = StudioLibraryModel()
  let reads: NativeReadsModel
  let tables: NativeTablesModel
  let studioPreferences = StudioPreferencesModel()
  public var desktopRequest: DesktopRequest?
  public var desktopError: String?
  public var desktopTransition = false
  public var quitting = false
  @ObservationIgnored public var onSnapshot: ((ManagerSnapshot) -> Void)?
  @ObservationIgnored public var onStudioState: ((StudioState) -> Void)?
  public var studioNeedsQuitConfirmation: Bool {
    reads.hasWork || tables.hasWork || connections.hasDraft || integrations.hasDraft
      || endpointEditor?.dirty == true
      || studioLibrary.hasWork || studioLibrary.instructions.hasWork || studioPreferences.hasWork
      || draft.hasContent || chatStorage?.busy == true
      || chatStorage?.uploading == true
      || chatStorage?.hasAttachments == true
      || chatStorage?.hasMessageEdit == true
      || chatStorage?.state?.unsavedEdits == true
      || chatStorage?.hasArtifactEdits == true
  }
  @ObservationIgnored private var chatStorage: StudioWorkspace?
  public var chat: StudioWorkspace {
    if let chatStorage { return chatStorage }
    let value = StudioWorkspace(client: client)
    value.onPresentation = { [weak self] in self?.onStudioState?($0) }
    chatStorage = value
    return value
  }

  public init(
    client: any ManagerLoading = NativeManager(),
    cloudClient: any OpenRouterLoading = NativeOpenRouter()
  ) {
    self.client = client
    reads = NativeReadsModel(client: client)
    tables = NativeTablesModel(client: client)
    insights = InsightsModel(client: client)
    benchmarks = BenchmarksModel(client: client)
    dataStorage = DataStorageModel(client: client)
    endpointLogs = EndpointLogsModel(client: client)
    cloud = CloudBrowserModel(client: cloudClient)
    downloads = DownloadsModel(client: client)
    connections = ConnectionsModel(client: client)
    integrations = IntegrationsModel(client: client)
    speech = StudioSpeechModels(client: client)
    speech.canAct = { [weak self] in self?.canSubmit == true && self?.chat.busy == false }
    speech.onChange = { [weak self] in
      guard let self else { return }
      await self.refresh()
      if self.chat.ready { await self.chat.perform("refresh") }
    }
    speech.onNetworkReview = { [weak self] endpoint in
      self?.openEndpoint(
        EndpointRow(port: endpoint.port, runner: nil, configured: endpoint, job: nil))
    }
    downloads.onCompletion = { [weak self] in await self?.refresh() }
    memoryPressure = NativeMemoryPressure { [weak self] in
      guard let self else { return }
      self.reads.trimHistory(maxBytes: 4 * 1024 * 1024)
      await self.chatStorage?.reclaimCaches()
      await NativeMarkdown.reclaimCaches()
    }
    connections.onChange = { [weak self] in
      // Refresh the one retained shared model store, never reload WebContent
      // or replace the user's current conversation/composer draft.
      if let chat = self?.chatStorage, chat.ready { await chat.perform("refresh") }
    }
    integrations.onChange = { [weak self] in
      if let chat = self?.chatStorage, chat.ready { await chat.perform("refresh") }
      await self?.refresh()
    }
    integrations.onManage = { [weak self] in
      // Managing the library leaves the Studio destination and draft intact.
      // The composer only selects tools; Manager owns connector configuration.
      self?.navigation.showManager(.connectors)
      self?.navigation.sidebarVisible = true
    }
    integrations.onConfigureSearch = { [weak self] port in
      self?.systemToolsPort = port
      self?.selectedEndpointPort = port
      self?.navigation.showManager(.runners)
    }
    integrations.prepareSignIn = { [weak self] in await self?.chat.start() }
    let studioCommand: StudioCommand = { [weak self] kind, payload in
      guard let self else { throw StudioLibraryError.message("The workspace closed.") }
      await self.chat.start()
      do { return try await self.chat.command(kind, payload) } catch {
        throw StudioLibraryError.message(
          (error as NSError).userInfo["WKJavaScriptExceptionMessage"] as? String
            ?? error.localizedDescription)
      }
    }
    studioLibrary.command = studioCommand
    studioPreferences.command = studioCommand
    studioLibrary.onManage = { [weak self] in
      self?.navigation.mode = .studio
      self?.navigation.studio = .prompts
    }
  }

  // Functional checks reuse their isolated, already authenticated workspace.
  // Production always creates its single workspace lazily through `chat`.
  convenience init(client: any ManagerLoading, preparedStudio: StudioWorkspace) {
    self.init(client: client)
    chatStorage = preparedStudio
    preparedStudio.onPresentation = { [weak self] in self?.onStudioState?($0) }
  }

  public var latestJob: ManagementJob? {
    let reported = snapshot?.jobs?.max(by: { $0.id < $1.id })
    if let lastReceipt, (reported?.id ?? 0) < lastReceipt.id { return lastReceipt }
    return reported
  }

  var managementNotice: ManagementJob? {
    guard let job = latestJob, job.isActive || job.state == "failed" else { return nil }
    // EndpointsView already renders this operation's row and spinner, even
    // before the runner is ready. Do not duplicate it in a workspace banner.
    // An automatic create without a port has no row yet, and failures always
    // need their actionable explanation rather than just a status label.
    let progressInModelRow =
      snapshot != nil && job.port != nil
      && navigation.mode == .manager && navigation.manager == .runners
      && detailEndpointPort == nil && systemToolsPort == nil
    return job.isActive && progressInModelRow ? nil : job
  }

  public var operationInProgress: Bool {
    isSubmitting || latestJob?.isActive == true || connections.saving || integrations.saving
      || speech.busy
      || endpointEditor?.saving == true
      || studioLibrary.saving || studioLibrary.instructions.saving || studioPreferences.saving
  }
  public var canSubmit: Bool { state == .ready && !operationInProgress && !quitting }

  /// A pure area switch for the menu bar. Unlike notification/deep-link
  /// actions, this does not request a particular page or conversation.
  public func selectWorkspace(_ mode: WorkspaceMode) {
    guard !desktopTransition, !quitting else { return }
    navigation.mode = mode
  }

  public func searchConversations() {
    guard navigation.mode == .studio, !desktopTransition, !quitting else { return }
    navigation.showStudioChats()
    historySearchRequest = UUID()
  }

  func openEndpoint(_ row: EndpointRow) {
    if let editor = endpointEditor, editor.endpoint.port != row.port, editor.dirty || editor.saving
    {
      desktopError =
        "Save or discard the settings for port \(editor.endpoint.port) before editing a different endpoint. Your draft is kept."
      return
    }
    if endpointEditor?.endpoint.port != row.port || endpointEditor?.removed == true {
      endpointEditor = row.configured.map {
        EndpointEditor(
          client: client, endpoint: $0, pid: row.runner?.pid,
          catalog: snapshot?.catalog.models ?? [])
      }
      endpointEditor?.onChange = { [weak self] in await self?.refresh() }
    }
    detailEndpointPort = row.port
    editingEndpoint = false
    selectedEndpointPort = row.port
    systemToolsPort = nil
    navigation.showManager(.runners)
  }

  func editEndpoint() {
    guard let editor = endpointEditor, editor.endpoint.port == detailEndpointPort,
      !editor.removed
    else { return }
    editingEndpoint = true
  }

  /// App-owned polling outlives windows. Slow loading stays in Rust; idle
  /// polling is modest, active operations get a one-second completion update.
  public func beginMonitoring() {
    guard monitor == nil, state != .stopped else { return }
    monitor = Task { [weak self] in
      while !Task.isCancelled {
        let active = self?.operationInProgress == true || self?.showsGPUMetrics == true
        do { try await Task.sleep(for: .seconds(active ? 1 : 5)) } catch { return }
        guard let self, self.state != .stopped else { return }
        if self.state != .loading && !self.quitting { await self.refresh() }
      }
    }
  }

  @discardableResult
  public func submit(_ command: ModelCommand, duringQuit: Bool = false) async -> Bool {
    guard state == .ready, !operationInProgress, !quitting || duringQuit else { return false }
    isSubmitting = true
    commandError = nil
    defer { isSubmitting = false }
    do {
      let receipt = try await client.submit(command)
      guard state != .stopped else { return false }
      lastReceipt = receipt
      await refresh()
      return true
    } catch {
      guard state != .stopped else { return false }
      commandError = error.localizedDescription
      return false
    }
  }

  public func dismissCommandError() { commandError = nil }

  public func start() async {
    guard state == .idle else { return }
    await refresh()
    await downloads.refresh()
  }

  /// Latest refresh wins even when an older request ignores cancellation.
  /// Errors retain the previous snapshot with its original timestamp.
  public func refresh() async {
    guard state != .stopped else { return }
    generation &+= 1
    let attempt = generation
    pending?.cancel()
    state = .loading
    let client = self.client
    let task = Task { try await client.snapshot() }
    pending = task
    do {
      let value = try await withTaskCancellationHandler {
        try await task.value
      } onCancel: {
        task.cancel()
      }
      guard generation == attempt else { return }
      try Task.checkCancellation()
      snapshot = value
      gpuHistory.ingest(value.gpu)
      endpointEditor?.observeRuntime(value)
      if let job = latestJob, job.action == "remove", job.state == "succeeded",
        let port = job.port, value.servers?.contains(where: { $0.port == port }) == false,
        !value.runners.contains(where: { $0.port == port })
      {
        if endpointEditor?.endpoint.port == port, endpointEditor?.dirty != true,
          endpointEditor?.saving != true
        {
          endpointEditor = nil
        }
        if detailEndpointPort == port {
          detailEndpointPort = nil
          editingEndpoint = false
        }
        if selectedEndpointPort == port { selectedEndpointPort = nil }
        if systemToolsPort == port { systemToolsPort = nil }
      }
      refreshedAt = Date()
      state = .ready
      onSnapshot?(value)
    } catch {
      guard generation == attempt else { return }
      gpuHistory.disconnect()
      state = .failed(
        error is CancellationError ? "Refresh cancelled." : error.localizedDescription)
    }
    if generation == attempt { pending = nil }
  }

  public func shutdown() async {
    memoryPressure?.stop()
    memoryPressure = nil
    await reads.shutdown()
    tables.cancel()
    await speech.settle()
    await studioLibrary.settle()
    await studioPreferences.settle()
    await endpointEditor?.settle()
    await integrations.stop()
    await connections.stop()
    await chatStorage?.shutdown()
    cloud.stop()
    downloads.stop()
    monitor?.cancel()
    monitor = nil
    generation &+= 1
    pending?.cancel()
    pending = nil
    state = .stopped
    snapshot = nil
    refreshedAt = nil
    lastReceipt = nil
    commandError = nil
    await client.close()
  }

  public func saveTablesBeforeQuit() async -> Bool {
    let saved = await tables.prepareForQuit()
    if !saved {
      desktopError = tables.history.error ?? "The table could not be saved. Paddock stayed open."
    }
    return saved
  }
}
