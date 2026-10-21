import SwiftUI

/// Typed destinations shared by the index and the actual controls. This is
/// navigation metadata, never a copy of preferences or credential values.
enum SettingsAnchor: String, CaseIterable {
  case replyLength, toolLimit, summarize, microphone, mapTiles
  case menuBar, login, shortcut, notifications, updates, backup, modelsFolder

  var destination: ManagerDestination {
    switch self {
    case .replyLength, .toolLimit, .summarize, .microphone, .mapTiles: .general
    case .menuBar, .login, .shortcut, .notifications, .updates: .application
    case .backup, .modelsFolder: .storage
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
  var id: String { anchor?.rawValue ?? "page:\(destination.rawValue)" }
  var title: String { anchor?.title ?? destination.rawValue }
}

enum SettingsSearchIndex {
  static let entries =
    ManagerDestination.allCases.map { SettingsSearchEntry(destination: $0) }
    + SettingsAnchor.allCases.map { SettingsSearchEntry(destination: $0.destination, anchor: $0) }

  static func search(_ query: String) -> [SettingsSearchEntry] {
    let terms = query.split(whereSeparator: \.isWhitespace).map(String.init)
    guard !terms.isEmpty else { return [] }
    return entries.filter { entry in
      let text = [
        entry.title, entry.destination.rawValue, entry.anchor?.aliases ?? "",
        pageAliases(entry.destination),
      ].joined(separator: " ")
      return terms.allSatisfy { text.localizedStandardContains($0) }
    }.sorted { left, right in
      let a = left.title.localizedStandardContains(
        query.trimmingCharacters(in: .whitespacesAndNewlines))
      let b = right.title.localizedStandardContains(
        query.trimmingCharacters(in: .whitespacesAndNewlines))
      if a != b { return a }
      return left.title.localizedStandardCompare(right.title) == .orderedAscending
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
  let anchor: SettingsAnchor?
  @Environment(\.settingsSearchRequest) private var request
  @AccessibilityFocusState private var accessibilityFocused: Bool
  @ViewBuilder
  func body(content: Content) -> some View {
    if let anchor {
      content.id(anchor.rawValue)
        .preference(key: SettingsAnchorsKey.self, value: [anchor.rawValue])
        .overlay {
          if request?.entry.anchor == anchor {
            RoundedRectangle(cornerRadius: 8).strokeBorder(
              PaddockStyle.secondary.opacity(0.6), lineWidth: 1
            )
            .padding(-6).allowsHitTesting(false).accessibilityHidden(true)
          }
        }
        .accessibilityFocused($accessibilityFocused)
        .task(id: request?.id) {
          accessibilityFocused = request?.entry.anchor == anchor
        }
    } else {
      content
    }
  }
}
extension View {
  func settingsSearchAnchor(_ anchor: SettingsAnchor?) -> some View {
    modifier(SettingsSearchAnchorModifier(anchor: anchor))
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
    guard let request, handled != request.id, let anchor = request.entry.anchor,
      anchors.contains(anchor.rawValue)
    else { return }
    proxy.scrollTo(anchor.rawValue, anchor: .center)
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
                  if entry.anchor != nil {
                    Text(entry.destination.rawValue).font(.system(size: 10)).foregroundStyle(
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
