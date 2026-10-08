import PaddockClient
import SwiftUI

public struct WorkspaceView: View {
  @Bindable private var model: WorkspaceModel
  private var applicationSettings: AnyView?
  @AppStorage("workspaceAppearance") private var appearance: WorkspaceAppearance = .system
  private var navigation: WorkspaceNavigation {
    get { model.navigation }
    nonmutating set { model.navigation = newValue }
  }
  private var draft: StudioDraft {
    get { model.draft }
    nonmutating set { model.draft = newValue }
  }
  @State private var confirmNewChat = false
  @State private var startSelection: StartSelection?
  @State private var downloadSelection: StartSelection?
  private struct StartSelection: Identifiable {
    let id = UUID()
    var model: String?
    var artifact: String?
    var purpose: ModelStartPurpose = .all
  }
  public init(model: WorkspaceModel, applicationSettings: AnyView? = nil) {
    self.model = model
    self.applicationSettings = applicationSettings
  }

  // Internal injection for layout tests; production uses the first-launch defaults.
  init(model: WorkspaceModel, navigation: WorkspaceNavigation) {
    self.model = model
    model.navigation = navigation
  }

  static func minimumWidth(for navigation: WorkspaceNavigation) -> CGFloat {
    WorkspacePanelMetrics.minimumWidth(navigation)
  }

  private var showsConversation: Bool {
    navigation.mode == .studio && navigation.studio.isConversation
  }
  private var showsCatalogColumns: Bool {
    navigation.mode == .manager
      && [.models, .cloudProviders, .customEndpoints].contains(navigation.manager)
  }
  private var hasNotices: Bool {
    if case .failed = model.state { return true }
    return model.commandError != nil || model.managementNotice != nil
  }

  public var body: some View {
    GeometryReader { geometry in
      // Edge-to-edge columns. Native traffic lights/navigation belong to the
      // leading column; other pane headers can use their own top row.
      ZStack {
        HStack(spacing: 0) {
          if navigation.showsSidebar {
            let width = WorkspacePanelMetrics.width(
              preferred: navigation.panelWidth, mode: navigation.mode,
              available: geometry.size.width)
            let maximum = WorkspacePanelMetrics.width(
              preferred: .greatestFiniteMagnitude, mode: navigation.mode,
              available: geometry.size.width)
            VStack(alignment: .leading, spacing: 0) {
              sidebar
            }
            .padding(.top, geometry.safeAreaInsets.top)
            .frame(width: width)
            .background(PaddockStyle.sidebar)
            WorkspacePanelGrip(
              width: Binding(get: { width }, set: { navigation.panelWidth = $0 }),
              range: WorkspacePanelMetrics.limits(navigation.mode).lowerBound...maximum,
              label: navigation.mode == .studio
                ? (navigation.showsReadsHistory
                  ? "Resize reads panel"
                  : navigation.studio == .tables ? "Resize tables panel" : "Resize chats panel")
                : "Resize Settings panel"
            ).id(navigation.mode)
          }
          VStack(spacing: 0) {
            if !showsConversation { notices }
            if navigation.mode == .studio {
              studioContent
            } else if navigation.manager == .general {
              StudioPreferencesView(
                model: model.studioPreferences, busy: model.chat.busy, chat: model.chat)
            } else if navigation.manager == .application {
              applicationSettings
            } else if navigation.manager == .cloudProviders {
              CloudModelsView(model: model.cloud, connections: model.connections)
            } else if navigation.manager == .customEndpoints {
              CloudModelsView(model: model.cloud, connections: model.connections, service: .custom)
            } else if navigation.manager == .connectors {
              IntegrationsView(model: model.integrations, endpoints: model.snapshot?.servers ?? [])
            } else if let snapshot = model.snapshot {
              managerContent(snapshot)
            } else {
              loading
            }
          }.frame(
            minWidth: WorkspacePanelMetrics.contentMinimum, maxWidth: .infinity,
            maxHeight: .infinity
          )
          .padding(.top, showsConversation || showsCatalogColumns ? 0 : geometry.safeAreaInsets.top)
          .environment(
            \.workspaceCatalogContentInset, showsCatalogColumns ? geometry.safeAreaInsets.top : 0
          )
          .environment(
            \.workspaceLeadingPaneInset, navigation.showsSidebar ? 0 : geometry.safeAreaInsets.top
          )
          .background(PaddockStyle.canvas)
        }
      }
      .ignoresSafeArea(.container, edges: .top)
    }
    .tint(PaddockStyle.accent)
    .frame(minWidth: Self.minimumWidth(for: navigation), minHeight: 650)
    .background(PaddockStyle.canvas)
    .containerBackground(PaddockStyle.canvas, for: .window)
    .preferredColorScheme(appearance.colorScheme)
    // The actual native title bar, not a second in-content header. AppKit owns
    // traffic lights, window dragging and full-screen placement.
    .toolbar {
      WorkspaceToolbar(
        navigation: $model.navigation, model: model,
        onNewChat: newChat, onStart: { navigation.showModelLibrary(purpose: .all) })
    }
    // Like oMLX's native settings shell, leave the actual window toolbar's
    // backdrop to macOS. Hiding it defeats legibility when text scrolls under
    // traffic lights and navigation controls; do not replace it with a mask.
    .toolbarBackgroundVisibility(.automatic, for: .windowToolbar)
    .task(id: model.desktopRequest?.id) {
      guard let request = model.desktopRequest else { return }
      await model.handleDesktopRequest(request)
      if model.desktopRequest?.id == request.id { model.desktopRequest = nil }
    }
    .alert(
      "Paddock",
      isPresented: Binding(
        get: { model.desktopError != nil }, set: { if !$0 { model.desktopError = nil } }
      )
    ) {
      Button("OK") { model.desktopError = nil }
    } message: {
      Text(model.desktopError ?? "")
    }
    .disabled(model.desktopTransition || model.quitting)
    .environment(\.studioToolsManager, model.integrations)
    .environment(\.studioLibrary, model.studioLibrary)
    .environment(\.studioSpeechWorkspace, model)
    .sheet(isPresented: $model.showsRendererSamples) { NativeMarkdownSamples() }
    .sheet(
      item: Binding(get: { model.integrations.editor }, set: { model.integrations.editor = $0 })
    ) { editor in
      ConnectorReviewView(
        editor: editor, model: model.integrations, endpoints: model.snapshot?.servers ?? []
      ) { Task { await model.integrations.cancelReview() } }.tint(
        PaddockStyle.accent)
    }
    .sheet(item: Binding(get: { model.connections.editor }, set: { model.connections.editor = $0 }))
    { editor in
      ConnectionReviewView(editor: editor) { Task { await model.connections.cancelReview() } }
        .tint(PaddockStyle.accent)
    }
    .sheet(item: $downloadSelection) { chosen in
      if let modelID = chosen.model, let artifact = chosen.artifact {
        DownloadReviewView(downloads: model.downloads, model: modelID, artifact: artifact) {
          navigation.showManager(.downloads)
        }
      }
    }
    .confirmationDialog(
      "Start a new chat?", isPresented: $confirmNewChat, titleVisibility: .visible
    ) {
      Button("Discard draft", role: .destructive) {
        draft = StudioDraft()
        navigation.studio = .newChat
        Task { await model.chat.newChat() }
      }
      Button("Keep draft", role: .cancel) { navigation.studio = .newChat }
    } message: {
      Text("The current draft has not been saved. Starting a new chat will clear it.")
    }
  }

  @ViewBuilder private var studioContent: some View {
    switch navigation.studio {
    case .newChat, .chats:
      StudioConversationView(
        chat: model.chat, draft: $model.draft, notices: hasNotices ? AnyView(notices) : nil)
    case .prompts: StudioLibraryView(model: model.studioLibrary)
    case .reads:
      NativeReadsView(model: model.reads, showsHistorySidebar: navigation.showsSidebar) {
        navigation.showManager(.runners)
      }
    case .tables:
      NativeTablesView(model: model.tables) {
        navigation.showModelLibrary(purpose: .tables)
      }
    case .embeddings:
      NativeEmbeddingsView(model: model.embeddings) {
        navigation.showModelLibrary(purpose: .embeddings)
      }
    case .settings:
      StudioPreferencesView(model: model.studioPreferences, busy: model.chat.busy, chat: model.chat)
    }
  }

  @ViewBuilder private func managerContent(_ snapshot: ManagerSnapshot) -> some View {
    switch navigation.manager {
    case .general:
      StudioPreferencesView(model: model.studioPreferences, busy: model.chat.busy, chat: model.chat)
    case .application: applicationSettings
    case .models:
      ModelLibraryView(
        snapshot: snapshot, canStart: model.canSubmit,
        purpose: $model.navigation.libraryPurpose,
        onDownload: { modelID, artifactID in
          downloadSelection = StartSelection(model: modelID, artifact: artifactID)
        },
        onStart: { modelID, artifactID in
          configureInstance(modelID, artifactID, purpose: navigation.libraryPurpose)
        })
    case .downloads:
      DownloadsView(
        downloads: model.downloads, canStart: model.canSubmit,
        onStart: { modelID, artifactID in
          configureInstance(modelID, artifactID)
        })
    case .runners:
      if let port = model.systemToolsPort {
        EndpointToolsView(
          model: model.integrations, port: port,
          title: snapshot.servers?.first { $0.port == port }?.title ?? "Port \(port)",
          backLabel: model.editingEndpoint ? "Model settings" : ManagerDestination.runners.rawValue
        ) {
          model.systemToolsPort = nil
        }
      } else if let port = model.detailEndpointPort {
        if model.editingEndpoint {
          EndpointEditView(
            workspace: model, port: port,
            onDownload: { downloadSelection = StartSelection(model: $0, artifact: $1) })
        } else {
          EndpointDetailView(workspace: model, port: port, snapshot: snapshot)
        }
      } else {
        EndpointsView(
          workspace: model, snapshot: snapshot,
          onCreate: { navigation.showModelLibrary(purpose: .all) },
          creation: instanceCreation(snapshot))
      }
    case .overview: OverviewView(snapshot: snapshot)
    case .insights: InsightsView(model: model.insights, endpoints: snapshot.servers ?? [])
    case .clients: ExternalClientsView(client: model.settingsClient, runners: snapshot.runners)
    case .storage: DataStorageView(model: model.dataStorage, snapshot: snapshot)
    case .benchmarks: BenchmarksView(model: model.benchmarks, runners: snapshot.runners)
    case .cloudProviders: CloudModelsView(model: model.cloud, connections: model.connections)
    case .customEndpoints:
      CloudModelsView(model: model.cloud, connections: model.connections, service: .custom)
    case .connectors:
      IntegrationsView(model: model.integrations, endpoints: snapshot.servers ?? [])
    }
  }

  private func configureInstance(
    _ modelID: String, _ artifactID: String, purpose: ModelStartPurpose = .all
  ) {
    model.detailEndpointPort = nil
    model.systemToolsPort = nil
    model.editingEndpoint = false
    startSelection = StartSelection(model: modelID, artifact: artifactID, purpose: purpose)
    navigation.showManager(.runners)
  }

  private func instanceCreation(_ snapshot: ManagerSnapshot) -> AnyView? {
    guard let chosen = startSelection else { return nil }
    return AnyView(
      StartModelView(
        snapshot: snapshot, model: chosen.model, artifact: chosen.artifact,
        purpose: chosen.purpose, client: model.settingsClient,
        submissionError: { model.commandError },
        onBrowse: {
          startSelection = nil
          navigation.showModelLibrary(purpose: chosen.purpose)
        },
        onDownload: { downloadSelection = StartSelection(model: $0, artifact: $1) },
        onClose: { startSelection = nil }
      ) { request in
        await model.submit(.create(request))
      }.id(chosen.id))
  }

  @ViewBuilder private var sidebar: some View {
    if navigation.mode == .studio {
      VStack(spacing: 0) {
        if navigation.showsReadsHistory {
          NativeReadsSidebar(model: model.reads)
        } else if navigation.studio == .tables {
          NativeTablesSidebar(model: model.tables)
        } else {
          StudioConversationSidebar(
            chat: model.chat, hasDraft: draft.hasContent, onNewChat: newChat,
            onFold: { navigation.sidebarVisible = false },
            onOpen: { navigation.studio = .chats }, searchRequest: model.historySearchRequest)
        }
        StudioSidebarFooter(navigation: $model.navigation)
      }
    } else {
      managerSidebar
    }
  }

  private var managerSidebar: some View {
    VStack(alignment: .leading, spacing: 0) {
      PaddockScrollView {
        VStack(spacing: 4) {
          Text("Settings").font(.system(size: 18, weight: .semibold))
            .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 10).padding(
              .bottom, 16)
          ForEach([ManagerDestination.general, .application, .storage]) { item in
            navigationRow(item.rawValue, symbol: item.symbol, selected: navigation.manager == item)
            {
              navigation.manager = item
            }
          }
          sidebarSection("Local models").padding(.top, 24)
          ForEach([ManagerDestination.runners, .models, .downloads, .insights, .benchmarks]) {
            item in
            navigationRow(item.rawValue, symbol: item.symbol, selected: navigation.manager == item)
            {
              navigation.manager = item
            }
          }
          sidebarSection("Connections").padding(.top, 22)
          ForEach([ManagerDestination.cloudProviders, .customEndpoints, .connectors, .clients]) {
            item in
            navigationRow(item.rawValue, symbol: item.symbol, selected: navigation.manager == item)
            {
              navigation.manager = item
            }
          }
        }.padding(.horizontal, 10).padding(.top, 18).padding(.bottom, 20)
      }.clipped()
      VStack(alignment: .leading, spacing: 12) {
        if let snapshot = model.snapshot {
          Label(
            "\(DisplayFormat.bytes(snapshot.identity.registry?.diskFree)) free",
            systemImage: "internaldrive"
          )
          .font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary)
          .padding(.horizontal, 10)
        }
        navigationRow(
          "This Mac", symbol: "laptopcomputer", selected: navigation.manager == .overview
        ) {
          navigation.manager = .overview
        }
      }.padding(.horizontal, 10).padding(.top, 12).padding(.bottom, 18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(PaddockStyle.sidebar).zIndex(1)
    }.frame(maxHeight: .infinity).background(PaddockStyle.sidebar)
      .accessibilityIdentifier("settings-sidebar")
  }

  private func navigationRow(
    _ title: String, symbol: String, selected: Bool, action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      HStack(spacing: 10) {
        Image(systemName: symbol).frame(width: 18)
        Text(title)
        Spacer(minLength: 0)
      }.font(.system(size: 13, weight: selected ? .medium : .regular)).padding(10)
        .background(
          selected ? PaddockStyle.elevated : .clear,
          in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
    }.buttonStyle(QuietButtonStyle()).accessibilityAddTraits(selected ? .isSelected : [])
      .accessibilityLabel(title)
      .accessibilityIdentifier("navigate-\(title)")
  }

  private func sidebarSection(_ title: String) -> some View {
    Text(title).font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
      .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 10).padding(.bottom, 6)
  }

  private func newChat() {
    guard !model.chat.busy, !model.chat.uploading else { return }
    if model.chat.hasMessageEdit {
      navigation.studio = .newChat
      model.desktopError = "Send or cancel the message edit first. Nothing was discarded."
      return
    }
    if draft.hasContent || model.chat.hasAttachments {
      confirmNewChat = true
    } else {
      navigation.studio = .newChat
      Task { await model.chat.newChat() }
    }
  }

  private var loading: some View {
    VStack(spacing: 16) {
      if model.state == .loading || model.state == .idle {
        ProgressView("Reading your model library…").controlSize(.small)
      } else if model.state == .stopped {
        Text("Closing…").foregroundStyle(.secondary)
      } else {
        Button("Try Again") { Task { await model.refresh() } }.modifier(PrimaryAction())
      }
    }.frame(maxWidth: .infinity, maxHeight: .infinity)
  }

  @ViewBuilder private var notices: some View {
    if hasNotices {
      VStack(spacing: 8) {
        if case .failed(let reason) = model.state {
          WorkspaceNotice(message: reason, kind: .warning)
        }
        if let error = model.commandError {
          WorkspaceNotice(message: error, kind: .warning, dismiss: model.dismissCommandError)
        } else if let job = model.managementNotice {
          WorkspaceNotice(message: job.message, kind: job.isActive ? .progress : .warning)
        }
      }.padding(.horizontal, 32).padding(.vertical, 8)
        .frame(maxWidth: 960).frame(maxWidth: .infinity)
    }
  }
}
