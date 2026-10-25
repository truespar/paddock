import Foundation
import PaddockConversationCore
import PaddockStudio

public enum WorkspaceMode: String, CaseIterable, Identifiable, Sendable {
  case studio = "Studio"
  // Internal routing only; there is no workspace switch in the product.
  case manager = "Settings"
  public var id: Self { self }
}

enum StudioDestination: String, CaseIterable, Identifiable {
  case newChat = "New chat"
  case chats = "Chats"
  case prompts = "Prompts"
  case reads = "Reads"
  case tables = "Tables"
  case embeddings = "Embeddings"
  case settings = "Settings"
  // The native shell puts these destinations at the sidebar bottom, unlike
  // the web's separate ActivityBar.
  var isConversation: Bool { self == .newChat || self == .chats }
  var id: Self { self }
  var symbol: String {
    switch self {
    case .newChat: "square.and.pencil"
    case .chats: "bubble.left.and.bubble.right"
    case .prompts: "text.alignleft"
    case .reads: "list.bullet.clipboard"
    case .tables: "tablecells"
    case .embeddings: "point.3.connected.trianglepath.dotted"
    case .settings: "gearshape"
    }
  }
}

enum ManagerDestination: String, CaseIterable, Identifiable {
  case general = "Conversation"
  case application = "Application"
  case models = "Catalog"
  case downloads = "Downloads"
  case runners = "Instances"
  case cloudProviders = "Cloud providers"
  case customEndpoints = "Custom endpoints"
  case connectors = "Connectors"
  case overview = "This Mac"
  case insights = "Usage & activity"
  case clients = "Client setup"
  case storage = "Data & storage"
  case benchmarks = "Benchmarks"
  var id: Self { self }
  var symbol: String {
    switch self {
    case .general: "bubble.left"
    case .application: "macwindow"
    case .models: "square.stack"
    case .downloads: "arrow.down.circle"
    case .runners: "terminal"
    case .cloudProviders: "cloud"
    case .customEndpoints: "network"
    case .connectors: "puzzlepiece.extension"
    case .overview: "laptopcomputer"
    case .insights: "chart.bar.xaxis"
    case .clients: "terminal"
    case .storage: "externaldrive"
    case .benchmarks: "speedometer"
    }
  }
}

/// The first launch is composer-first. Only layout and the last Settings page
/// are restored; opening the app cannot replay a command or start a model.
struct WorkspaceNavigation {
  var mode: WorkspaceMode = .studio
  var studio: StudioDestination = .newChat
  var manager: ManagerDestination = .runners
  var libraryPurpose: ModelStartPurpose = .all
  private var studioSidebar = false
  private var managerSidebar = true
  private var studioWidth: CGFloat = 260
  private var managerWidth: CGFloat = 220

  var preferences: [NativeWorkspacePreferences.Key: ConversationValue] {
    [
      .studioSidebar: .bool(studioSidebar), .settingsSidebar: .bool(managerSidebar),
      .studioWidth: .number(Decimal(Double(studioWidth))),
      .settingsWidth: .number(Decimal(Double(managerWidth))),
      .settingsPage: .string(manager.rawValue),
    ]
  }
  mutating func restore(_ values: [NativeWorkspacePreferences.Key: ConversationValue]) {
    if let value = values[.studioSidebar]?.bool { studioSidebar = value }
    if let value = values[.settingsSidebar]?.bool { managerSidebar = value }
    if let value = values[.settingsPage]?.string.flatMap(ManagerDestination.init(rawValue:)) {
      manager = value
    }
    if case .number(let raw) = values[.studioWidth], !raw.isNaN {
      studioWidth = min(360, max(220, NSDecimalNumber(decimal: raw).doubleValue))
    }
    if case .number(let raw) = values[.settingsWidth], !raw.isNaN {
      managerWidth = min(260, max(200, NSDecimalNumber(decimal: raw).doubleValue))
    }
  }

  var panelWidth: CGFloat {
    get { mode == .studio ? studioWidth : managerWidth }
    set {
      let limits = WorkspacePanelMetrics.limits(mode)
      let value = min(limits.upperBound, max(limits.lowerBound, newValue))
      if mode == .studio { studioWidth = value } else { managerWidth = value }
    }
  }

  var sidebarVisible: Bool {
    get { mode == .studio ? studioSidebar : managerSidebar }
    set {
      if mode == .studio { studioSidebar = newValue } else { managerSidebar = newValue }
    }
  }

  var showsSidebar: Bool {
    sidebarVisible
  }
  var showsReadsHistory: Bool { mode == .studio && studio == .reads }

  mutating func showStudioChats() {
    mode = .studio
    studio = .chats
    studioSidebar = true
  }

  mutating func toggleSidebar() {
    sidebarVisible.toggle()
  }

  mutating func showManager(_ destination: ManagerDestination) {
    mode = .manager
    manager = destination
  }

  mutating func showSettings() {
    mode = .manager
    managerSidebar = true
  }

  mutating func returnToChat() {
    mode = .studio
    // Keep the current conversation, draft, and its sidebar preference.
    if !studio.isConversation { studio = .chats }
  }

  mutating func showModelLibrary(purpose: ModelStartPurpose) {
    libraryPurpose = purpose
    showManager(.models)
  }
}

/// Editor binding only. StudioWorkspace owns the per-conversation durable draft.
struct StudioDraft {
  var message = ""
  var hasContent: Bool { !message.isEmpty }
}
