import SwiftUI

/// Typed destinations shared by the index and the actual controls. This is
/// navigation metadata, never a copy of preferences or credential values.
enum SettingsAnchor: String, CaseIterable {
  case replyLength, toolLimit, summarize, microphone, mapTiles
  case menuBar, login, shortcut, notifications, updates, backup, modelsFolder
  case context, workload, memoryBudget, speculation, modelLoading, idleUnload, loadWait
  case kvOffload, endpointAccess, webSearch, mcpServers, connectorDirectory, connectorLibrary

  var isEndpoint: Bool { destination == .runners }

  var destination: ManagerDestination {
    switch self {
    case .replyLength, .toolLimit, .summarize, .microphone, .mapTiles: .general
    case .menuBar, .login, .shortcut, .notifications, .updates: .application
    case .backup, .modelsFolder: .storage
    case .context, .workload, .memoryBudget, .speculation, .modelLoading, .idleUnload,
      .loadWait, .kvOffload, .endpointAccess, .webSearch, .mcpServers:
      .runners
    case .connectorDirectory, .connectorLibrary: .connectors
    }
  }
  var title: String {
    switch self {
    case .replyLength: "Reply length"
    case .toolLimit: "Tool call limit"
    case .summarize: "Summarize conversations"
    case .microphone: "Microphone & audio"
    case .mapTiles: "Map tile server"
    case .menuBar: "Show in menu bar"
    case .login: "Launch at login"
    case .shortcut: "Quick Question shortcut"
    case .notifications: "Notifications, sounds & previews"
    case .updates: "App updates"
    case .backup: "Conversation backup"
    case .modelsFolder: "Models folder"
    case .context: "Context per conversation"
    case .workload: "Concurrent requests"
    case .memoryBudget: "Memory budget"
    case .speculation: "Speculative decoding"
    case .modelLoading: "Load model on demand"
    case .idleUnload: "Unload after inactivity"
    case .loadWait: "Load wait limit"
    case .kvOffload: "KV cache offloading"
    case .endpointAccess: "Endpoint access & API key"
    case .webSearch: "Web search provider"
    case .mcpServers: "Enabled MCP servers"
    case .connectorDirectory: "Find connectors"
    case .connectorLibrary: "Your connectors"
    }
  }
  var aliases: String {
    switch self {
    case .replyLength: "maximum output tokens response length"
    case .toolLimit: "maximum tools iterations calls"
    case .summarize: "summary compact context automatic"
    case .microphone: "recording dictation transcription input device speech"
    case .mapTiles: "maps photos location privacy"
    case .menuBar: "menubar status tray"
    case .login: "startup autostart launch"
    case .shortcut: "hotkey keyboard quick question"
    case .notifications: "desktop alert sound completion response preview"
    case .updates: "version upgrade automatic check release"
    case .backup: "export conversations attachments database sqlite"
    case .modelsFolder: "disk space storage weights files directory"
    case .context: "context window length tokens prompt"
    case .workload: "concurrency max batch workload requests"
    case .memoryBudget: "VRAM unified memory GPU budget limit"
    case .speculation: "MTP DFlash DFlash2 drafter speculative decoding"
    case .modelLoading: "automatic residency dynamic loading startup on demand"
    case .idleUnload: "residency idle timeout unload release memory"
    case .loadWait: "residency loading timeout seconds wait"
    case .kvOffload: "prefix cache reuse offload SSD disk RAM"
    case .endpointAccess: "port network listen host API key access"
    case .webSearch: "Exa Tavily Brave Firecrawl Perplexity search provider API key"
    case .mcpServers: "MCP enabled tools connectors endpoint"
    case .connectorDirectory: "MCP directory discover tools servers"
    case .connectorLibrary: "MCP installed saved connectors OAuth credentials"
    }
  }
  static func conversationSection(_ id: String) -> Self? {
    switch id {
    case "maxTokens": .replyLength
    case "maxToolCalls": .toolLimit
    case "summarize": .summarize
    case "microphone": .microphone
    case "mapTiles": .mapTiles
    default: nil
    }
  }
}

struct SettingsSearchEntry: Identifiable, Equatable {
  let destination: ManagerDestination
  var anchor: SettingsAnchor?
  var port: UInt16?
  var instanceTitle: String?
  var runtimeOption: String?
  var label: String?
  var keywords = ""
  var targetID: String? { runtimeOption.map { "runtime:\($0)" } ?? anchor?.rawValue }
  var id: String {
    (port.map { "endpoint:\($0):" } ?? "") + (targetID ?? "page:\(destination.rawValue)")
  }
  var title: String { label ?? anchor?.title ?? destination.rawValue }
  var subtitle: String {
    if let port { return "\(instanceTitle ?? "Instance") · \(port)" }
    return destination.rawValue
  }
}

enum SettingsSearchIndex {
  static let entries =
    ManagerDestination.allCases.map { SettingsSearchEntry(destination: $0) }
    + SettingsAnchor.allCases.filter { !$0.isEndpoint }.map {
      SettingsSearchEntry(destination: $0.destination, anchor: $0)
    }

  static func search(_ query: String, instances: [SettingsSearchEntry] = [])
    -> [SettingsSearchEntry]
  {
    let terms = query.split(whereSeparator: \.isWhitespace).map(String.init)
    guard !terms.isEmpty else { return [] }
    return (entries + instances).filter { entry in
      let text = [
        entry.title, entry.destination.rawValue, entry.anchor?.aliases ?? "",
        entry.targetID == nil ? pageAliases(entry.destination) : "",
        entry.instanceTitle ?? "", entry.port.map(String.init) ?? "", entry.keywords,
      ].joined(separator: " ")
      return terms.allSatisfy { text.localizedStandardContains($0) }
    }.sorted { left, right in
      let a = left.title.localizedStandardContains(
        query.trimmingCharacters(in: .whitespacesAndNewlines))
      let b = right.title.localizedStandardContains(
        query.trimmingCharacters(in: .whitespacesAndNewlines))
      if a != b { return a }
      let order = left.title.localizedStandardCompare(right.title)
      return order == .orderedSame ? left.id < right.id : order == .orderedAscending
    }
  }
  private static func pageAliases(_ page: ManagerDestination) -> String {
    switch page {
    case .runners: "models configured start stop port logs settings memory context"
    case .models: "download weights library models variants"
    case .downloads: "progress pause resume queue"
    case .cloudProviders: "OpenRouter API key account credits providers"
    case .customEndpoints: "OpenAI compatible remote URL API key"
    case .connectors: "MCP tools servers OAuth web search"
    case .insights: "usage activity cache tokens requests"
    case .clients: "external client terminal Claude Code OpenCode credentials"
    case .overview: "hardware GPU memory thermal pressure"
    case .benchmarks: "performance latency throughput speed tokens"
    default: ""
    }
  }
}

struct SettingsSearchRequest: Equatable {
  let id = UUID()
  let entry: SettingsSearchEntry
}
extension EnvironmentValues {
  @Entry var settingsSearchRequest: SettingsSearchRequest? = nil
}
private struct SettingsAnchorsKey: PreferenceKey {
  static let defaultValue: Set<String> = []
  static func reduce(value: inout Set<String>, nextValue: () -> Set<String>) {
    value.formUnion(nextValue())
  }
}
private struct SettingsSearchAnchorModifier: ViewModifier {
  let target: String?
  @Environment(\.settingsSearchRequest) private var request
  @AccessibilityFocusState private var accessibilityFocused: Bool
  @ViewBuilder
  func body(content: Content) -> some View {
    if let target {
      content.id(target)
        .preference(key: SettingsAnchorsKey.self, value: [target])
        .overlay {
          if request?.entry.targetID == target {
            RoundedRectangle(cornerRadius: 8).strokeBorder(
              PaddockStyle.secondary.opacity(0.6), lineWidth: 1
            )
            .padding(-6).allowsHitTesting(false).accessibilityHidden(true)
          }
        }
        .accessibilityFocused($accessibilityFocused)
        .task(id: request?.id) {
          accessibilityFocused = request?.entry.targetID == target
        }
    } else {
      content
    }
  }
}
extension View {
  func settingsSearchAnchor(_ anchor: SettingsAnchor?) -> some View {
    settingsSearchTarget(anchor?.rawValue)
  }
  func settingsSearchTarget(_ id: String?) -> some View {
    modifier(SettingsSearchAnchorModifier(target: id))
  }
}

/// Wait for asynchronous preference rows to exist before scrolling. Later
/// model/telemetry publications must not pull the user back to the result.
struct SettingsSearchScroll: ViewModifier {
  let proxy: ScrollViewProxy
  @Environment(\.settingsSearchRequest) private var request
  @State private var anchors: Set<String> = []
  @State private var handled: UUID?
  func body(content: Content) -> some View {
    content.onPreferenceChange(SettingsAnchorsKey.self) { anchors = $0 }
      .onChange(of: anchors) { _, _ in reveal() }
      .onChange(of: request) { _, _ in reveal() }
      .onAppear { reveal() }
  }
  private func reveal() {
    guard let request, handled != request.id, let target = request.entry.targetID,
      anchors.contains(target)
    else { return }
    proxy.scrollTo(target, anchor: .center)
    handled = request.id
  }
}

struct SettingsSearchResults: View {
  let entries: [SettingsSearchEntry]
  let highlighted: String?
  let select: (SettingsSearchEntry) -> Void
  var body: some View {
    ScrollViewReader { proxy in
      PaddockScrollView {
        LazyVStack(alignment: .leading, spacing: 4) {
          if entries.isEmpty {
            Text("No matching settings").font(.system(size: 12)).foregroundStyle(.secondary)
              .padding(10)
          }
          ForEach(entries) { entry in
            Button {
              select(entry)
            } label: {
              HStack(alignment: .center, spacing: 8) {
                Image(systemName: entry.destination.symbol).frame(width: 18)
                VStack(alignment: .leading, spacing: 3) {
                  Text(entry.title).font(.system(size: 12, weight: .medium))
                  if entry.targetID != nil {
                    Text(entry.subtitle).font(.system(size: 10)).foregroundStyle(
                      .secondary)
                  }
                }
                Spacer(minLength: 0)
              }.frame(maxWidth: .infinity, alignment: .leading).padding(10)
                .contentShape(Rectangle())
                .background(
                  highlighted == entry.id ? PaddockStyle.elevated : .clear,
                  in: RoundedRectangle(cornerRadius: 8))
            }.buttonStyle(.plain).id(entry.id)
              .accessibilityIdentifier("settings-result-\(entry.id)")
          }
        }.padding(.horizontal, 10).padding(.bottom, 20)
      }.onChange(of: highlighted) { _, id in
        if let id { proxy.scrollTo(id) }
      }
    }
  }
}
