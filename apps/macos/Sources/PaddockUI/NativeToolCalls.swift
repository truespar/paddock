import AppKit
import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI

struct NativeToolCallView: View {
  let call: StudioState.NativeTranscript.Message.ToolCall
  let workspace: StudioWorkspace?
  let target: StudioMessageTarget?
  @Environment(\.transcriptDisclosure) private var readerDisclosure
  @Environment(\.conversationFind) private var find
  @State private var expanded = false
  private var findTarget: Bool {
    find?.current?.itemID == call.id + "/arguments" || find?.current?.itemID == call.id + "/output"
  }
  @State private var deciding = false
  @State private var decisionSent = false
  @State private var error: String?
  private var pending: Bool { call.status == "pending" }
  private var running: Bool { call.status == "in_progress" }
  private var statusColor: Color {
    switch call.status {
    case "completed": Color(nsColor: PaddockStyle.nsColor("success"))
    case "failed", "denied": Color(nsColor: PaddockStyle.nsColor("error"))
    case "pending": PaddockStyle.caution
    default: PaddockStyle.secondary
    }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack(spacing: 7) {
        Button {
          readerDisclosure?.reveal()
          if findTarget {
            find?.close()
            expanded = false
          } else {
            expanded.toggle()
          }
        } label: {
          HStack(spacing: 7) {
            if running {
              ProgressView().controlSize(.mini).frame(width: 14, height: 14)
            } else {
              Image(
                systemName: pending
                  ? "checkmark.shield"
                  : call.status == "failed"
                    ? "exclamationmark.triangle"
                    : call.status == "denied"
                      ? "xmark" : call.status == "completed" ? "checkmark" : "wrench"
              )
              .frame(width: 14, height: 14).foregroundStyle(statusColor)
            }
            if pending || running {
              Text(pending ? "Wants to run" : "Running").foregroundStyle(.secondary)
            }
            Text(ToolCallPresentation.title(call.name, server: call.server))
              .fontWeight(.semibold).lineLimit(1).truncationMode(.tail)
            if !call.server.isEmpty {
              Text(call.server).font(.system(size: 11)).foregroundStyle(.secondary)
                .padding(.horizontal, 6).frame(height: 16)
                .background(
                  PaddockStyle.sidebar,
                  in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small))
            }
            Spacer(minLength: 0)
            if call.status == "failed" || call.status == "denied" {
              Text(call.status == "failed" ? "ERROR" : "DENIED")
                .font(.system(size: 10, weight: .semibold)).foregroundStyle(statusColor)
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(
                  Color(nsColor: PaddockStyle.nsColor("errorSubtle")),
                  in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small))
            }
            Image(systemName: expanded || findTarget ? "chevron.up" : "chevron.down")
              .frame(width: 14, height: 14).foregroundStyle(.secondary)
          }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
        }.buttonStyle(.plain)
          .accessibilityValue(expanded || findTarget ? "Expanded" : "Collapsed")
          .accessibilityIdentifier("native-tool-toggle-\(call.id)")
        if let id = call.artifactId, let workspace {
          Button("Show") { Task { await workspace.showArtifact(id) } }
            .buttonStyle(StudioInlineOutlineStyle())
            .accessibilityIdentifier("native-tool-show-\(call.id)")
        }
      }.font(.system(size: 12)).padding(.vertical, 7).padding(.horizontal, 10)
      if pending || expanded || findTarget {
        VStack(alignment: .leading, spacing: 10) {
          section("Tool") {
            NativeSelectableText(call.name, size: 12, monospaced: true)
          }
          section("Arguments") {
            code(ToolCallPresentation.arguments(call.arguments), id: call.id + "/arguments")
          }
          if !pending, !call.output.isEmpty || !call.error.isEmpty {
            section(call.error.isEmpty ? "Result" : "Error") {
              code(
                ToolCallPresentation.output(call.error.isEmpty ? call.output : call.error),
                id: call.id + "/output")
            }
          }
          if pending {
            HStack {
              Text(decisionSent ? "Decision sent" : "Run this tool?")
              Spacer()
              Button("Deny") { decide(false) }.buttonStyle(FlatButtonStyle())
              Button("Approve") { decide(true) }.buttonStyle(FlatButtonStyle(primary: true))
            }.disabled(
              deciding || decisionSent || workspace == nil || target == nil
                || call.approvalId == nil)
          }
          if let error { Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled) }
        }.font(.system(size: 12)).padding(.horizontal, 10).padding(.bottom, 10)
      }
    }.frame(maxWidth: .infinity, alignment: .leading)
      .background(
        PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
      )
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
          .strokeBorder(pending ? PaddockStyle.caution : PaddockStyle.border).allowsHitTesting(
            false)
      )
      .accessibilityIdentifier("native-tool-\(call.id)")
      .onChange(of: call.approvalId) { _, _ in decisionSent = false }
  }
  private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content)
    -> some View
  {
    VStack(alignment: .leading, spacing: 4) {
      Text(title.uppercased()).font(.system(size: 11, weight: .semibold)).foregroundStyle(
        .secondary)
      content()
    }
  }
  private func code(_ text: String, id: String) -> some View {
    NativeToolCode(text: text.isEmpty ? "-" : text, textID: id)
      .background(
        PaddockStyle.sidebar, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small))
  }
  private func decide(_ approve: Bool) {
    guard !deciding, let workspace, let target, let approval = call.approvalId else { return }
    deciding = true
    error = nil
    Task {
      defer { deciding = false }
      var payload = target.payload(action: "approve")
      payload["callId"] = .string(call.id)
      payload["approvalId"] = .string(approval)
      payload["approve"] = .bool(approve)
      do {
        try await workspace.command("toolApproval", payload)
        decisionSent = true
      } catch { self.error = error.localizedDescription }
    }
  }
}

enum ToolCallPresentation {
  static func pretty(_ value: Any) -> String {
    NativeToolText.pretty(value)
  }
  static func arguments(_ raw: String) -> String {
    NativeToolText.arguments(raw)
  }
  static func output(_ raw: String) -> String {
    NativeToolText.output(raw)
  }
  /// Same identifier humanization as Web Studio's ToolCall.vue.
  static func title(_ raw: String, server: String) -> String {
    var text = raw
    if !server.isEmpty {
      text = text.replacingOccurrences(
        of: "^[A-Za-z0-9][\\w-]*?__", with: "", options: .regularExpression)
    }
    text = text.replacingOccurrences(of: "^mcp_", with: "", options: .regularExpression)
      .replacingOccurrences(of: "([a-z0-9])([A-Z])", with: "$1 $2", options: .regularExpression)
      .replacingOccurrences(of: "[_-]+", with: " ", options: .regularExpression)
      .trimmingCharacters(in: .whitespacesAndNewlines)
    return text.isEmpty ? raw : text.prefix(1).uppercased() + text.dropFirst()
  }
}

struct NativeSearchCallView: View {
  let search: StudioState.NativeTranscript.Message.Search
  @Environment(\.transcriptDisclosure) private var readerDisclosure
  @State private var expanded = false
  private var running: Bool { search.status == "searching" || search.status == "in_progress" }
  private var failed: Bool { search.status == "failed" }
  private var provider: SearchProviderArtwork? { SearchProviderArtwork(rawValue: search.provider) }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      Button {
        readerDisclosure?.reveal()
        expanded.toggle()
      } label: {
        HStack(spacing: 7) {
          indicator
          Text(running ? "Searching the web for" : "Searched the web for")
            .foregroundStyle(.secondary).lineLimit(1)
          Text(verbatim: search.query.isEmpty ? "..." : search.query)
            .fontWeight(.semibold).lineLimit(1).truncationMode(.tail)
          Spacer(minLength: 0)
          if failed {
            Text("ERROR").font(.system(size: 10, weight: .semibold))
              .foregroundStyle(Color(nsColor: PaddockStyle.nsColor("error")))
              .padding(.horizontal, 6).padding(.vertical, 2)
              .background(
                Color(nsColor: PaddockStyle.nsColor("errorSubtle")),
                in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small))
          } else if !search.sources.isEmpty {
            Text("\(search.sources.count) \(search.sources.count == 1 ? "source" : "sources")")
              .foregroundStyle(.secondary).fixedSize()
          }
          Image(systemName: expanded ? "chevron.up" : "chevron.down")
            .frame(width: 14, height: 14).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
          .padding(.vertical, 7).padding(.horizontal, 10)
      }.buttonStyle(.plain)
        .accessibilityIdentifier("native-search-toggle-\(search.id)")
        .accessibilityValue(expanded ? "Expanded" : "Collapsed")
      if expanded {
        VStack(alignment: .leading, spacing: 10) {
          if failed {
            Text(verbatim: search.error.isEmpty ? "The search failed." : search.error)
              .foregroundStyle(Color(nsColor: PaddockStyle.nsColor("error"))).textSelection(
                .enabled)
          } else if search.sources.isEmpty {
            Text("No results.").foregroundStyle(.secondary)
          } else {
            VStack(alignment: .leading, spacing: 6) {
              ForEach(Array(search.sources.enumerated()), id: \.offset) { index, source in
                HStack(alignment: .firstTextBaseline, spacing: 9) {
                  Text("\(index + 1)").monospaced().foregroundStyle(.secondary)
                    .frame(minWidth: 12, alignment: .trailing)
                  if let url = Self.safeURL(source.url) {
                    Link(destination: url) {
                      HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(verbatim: source.title.isEmpty ? Self.host(source.url) : source.title)
                          .font(.system(size: 13)).lineLimit(1).truncationMode(.tail)
                        Text(verbatim: Self.host(source.url)).foregroundStyle(.secondary)
                          .lineLimit(1).layoutPriority(1)
                      }.help(source.title.isEmpty ? source.url : source.title)
                    }.buttonStyle(.plain).foregroundStyle(.primary)
                  } else {
                    Text(verbatim: source.title.isEmpty ? source.url : source.title).textSelection(
                      .enabled)
                  }
                }
              }
            }
          }
          if let provider {
            HStack(spacing: 6) {
              SearchProviderLogo(provider: provider, size: 12)
              Text("Searched with \(provider.label)").foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
              .overlay(alignment: .top) { PaddockStyle.border.frame(height: 1) }
          }
        }.padding(.horizontal, 10).padding(.bottom, 10)
      }
    }.font(.system(size: 12)).frame(maxWidth: .infinity, alignment: .leading)
      .background(
        PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
      )
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
          .strokeBorder(PaddockStyle.border).allowsHitTesting(false)
      )
      .accessibilityIdentifier("native-search-\(search.id)")
  }

  @ViewBuilder private var indicator: some View {
    switch SearchCallIndicator(status: search.status, provider: search.provider) {
    case .progress: ProgressView().controlSize(.mini).frame(width: 14, height: 14)
    case .error:
      Image(systemName: "exclamationmark.triangle").frame(width: 14, height: 14)
        .foregroundStyle(Color(nsColor: PaddockStyle.nsColor("error")))
    case .provider(let mark): SearchProviderLogo(provider: mark, size: 14)
    case .globe: Image(systemName: "globe").frame(width: 14, height: 14).foregroundStyle(.secondary)
    }
  }
  nonisolated static func host(_ value: String) -> String {
    let host = URL(string: value)?.host() ?? value
    return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
  }
  nonisolated static func safeURL(_ value: String) -> URL? {
    guard let url = URL(string: value), ["https", "http"].contains(url.scheme?.lowercased() ?? ""),
      url.host != nil, url.user == nil, url.password == nil
    else { return nil }
    return url
  }
}

/// Compact outline controls used by the web tool strip and artifact header.
struct StudioInlineOutlineStyle: ButtonStyle {
  var selected = false
  func makeBody(configuration: Configuration) -> some View {
    configuration.label.font(.system(size: 11))
      .foregroundStyle(selected ? PaddockStyle.primary : PaddockStyle.secondary)
      .padding(.horizontal, 8).padding(.vertical, 1)
      .background(
        configuration.isPressed ? PaddockStyle.elevated : .clear,
        in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small)
      )
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.small).strokeBorder(PaddockStyle.border))
  }
}
