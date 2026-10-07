import AppKit
import PaddockClient
import SwiftUI

enum ExternalClient: String, CaseIterable, Identifiable {
  case responses = "Responses API"
  case openCode = "OpenCode"
  case openCode2 = "OpenCode 2"
  var id: Self { self }
  func configuration(_ setup: LocalClientSetup) -> String {
    let encoder: (Any) -> String = { value in
      guard
        let bytes = try? JSONSerialization.data(
          withJSONObject: value, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
      else { return "" }
      return String(decoding: bytes, as: UTF8.self)
    }
    let models = [setup.model: ["name": setup.model]]
    switch self {
    case .responses:
      let body = encoder(["model": setup.model, "input": "Say hello.", "stream": true])
      return
        "curl -N \(Self.quote(setup.baseUrl + "/responses")) \\\n  -H 'Content-Type: application/json' \\\n\(setup.hasKey ? "  -H \"Authorization: Bearer $PADDOCK_API_KEY\" \\\n" : "")  --data \(Self.quote(body))"
    case .openCode:
      var options = ["baseURL": setup.baseUrl]
      if setup.hasKey { options["apiKey"] = "{env:PADDOCK_API_KEY}" }
      return encoder([
        "$schema": "https://opencode.ai/config.json", "model": "paddock/\(setup.model)",
        "provider": [
          "paddock": [
            "npm": "@ai-sdk/openai-compatible", "name": "Paddock", "options": options,
            "models": models,
          ]
        ],
      ])
    case .openCode2:
      var provider: [String: Any] = [
        "name": "Paddock", "package": "@opencode/ai/providers/openai-compatible",
        "settings": ["baseURL": setup.baseUrl], "models": models,
      ]
      if setup.hasKey { provider["env"] = ["PADDOCK_API_KEY"] }
      return encoder([
        "$schema": "https://opencode.ai/config.json", "model": "paddock/\(setup.model)",
        "providers": ["paddock": provider],
      ])
    }
  }
  static func quote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}

struct ExternalClientsView: View {
  let client: any ManagerLoading
  let runners: [RunnerInfo]
  @State private var selected: String?
  @State private var format: ExternalClient = .openCode
  @State private var setup: LocalClientSetup?
  @State private var error: String?
  @State private var exporting = false
  @State private var confirmExport = false
  @State private var exportedPath: String?
  private var choices: [RunnerInfo] {
    runners.filter { $0.model != nil && $0.status != "unreachable" }
  }
  private var runner: RunnerInfo? { choices.first { $0.id == selected } ?? choices.first }
  var body: some View {
    SettingsPage(title: "Client setup") { stacked in
      if let runner {
        SettingsGroup(title: "Local endpoint") {
          SettingsRow(title: "Instance", stacked: stacked) {
            Dropdown(title: "Instance", value: runner.title, fillsWidth: true) {
              ForEach(choices) { r in Button("\(r.title) · \(r.port)") { selected = r.id } }
            }.accessibilityIdentifier("client-instance")
          }
          if let setup {
            SettingsRow(title: "Base URL", stacked: stacked) {
              HStack(alignment: .center, spacing: 12) {
                Text(setup.baseUrl).textSelection(.enabled)
                  .font(.system(size: 12, design: .monospaced))
                  .fixedSize(horizontal: false, vertical: true)
                  .frame(maxWidth: .infinity, alignment: .leading)
                Button {
                  copy(setup.baseUrl)
                } label: {
                  Image(systemName: "doc.on.doc")
                }
                .accessibilityLabel("Copy URL").help("Copy URL")
              }.frame(minHeight: 30)
            }
            SettingsRow(title: "Model", stacked: stacked) {
              Text(setup.model).foregroundStyle(.secondary).textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(minHeight: 30, alignment: .leading)
            }
          } else if error == nil {
            ProgressView().controlSize(.small).accessibilityLabel("Loading endpoint")
          }
        }
        if let setup {
          ClientConfigurationSection(
            setup: setup, format: $format, stacked: stacked,
            exporting: exporting, exportedPath: exportedPath,
            copy: copy, export: { confirmExport = true })
        }
      } else {
        ContentUnavailableView(
          "Start a chat model first", systemImage: "network",
          description: Text("Running local models provide endpoints for other apps."))
      }
      if let error {
        Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
    .task(id: runner?.id) {
      setup = nil
      error = nil
      exportedPath = nil
      guard let runner else { return }
      do {
        let value = try await client.inspect(
          .clientInfo(port: runner.port, pid: runner.pid), as: LocalClientSetup.self)
        try Task.checkCancellation()
        setup = value
      } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
    }
    .confirmationDialog(
      "Export this instance's API key?", isPresented: $confirmExport, titleVisibility: .visible
    ) {
      Button("Choose export location…") { exportKey() }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text(
        "The file contains a plaintext credential and is readable only by your macOS account. Keep it private and do not add it to source control."
      )
    }
  }
  private func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
  }
  private func exportKey() {
    guard let runner else { return }
    let panel = NSSavePanel()
    panel.nameFieldStringValue = "paddock-\(runner.port)-credentials.sh"
    panel.canCreateDirectories = true
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      exporting = true
      error = nil
      Task {
        defer { exporting = false }
        do {
          _ = try await client.inspect(
            .exportCredentials(port: runner.port, pid: runner.pid, path: url.path),
            as: NativeExportReceipt.self)
          exportedPath = url.path
        } catch { self.error = error.localizedDescription }
      }
    }
  }
}

/// Configuration stays selectable native text. Soft wrapping affects display
/// only; copying always returns the original JSON/shell source unchanged.
struct ClientConfigurationSection: View {
  let setup: LocalClientSetup
  @Binding var format: ExternalClient
  let stacked: Bool
  let exporting: Bool
  let exportedPath: String?
  let copy: (String) -> Void
  let export: () -> Void

  var body: some View {
    SettingsGroup(title: "Configuration") {
      SettingsRow(title: "Client", stacked: stacked) {
        Dropdown(title: "Client", value: format.rawValue, fillsWidth: true) {
          ForEach(ExternalClient.allCases) { option in
            Button(option.rawValue) { format = option }
          }
        }.accessibilityIdentifier("client-format")
      }
      Text(format.configuration(setup)).font(.system(size: 12, design: .monospaced))
        .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
        .accessibilityIdentifier("client-configuration-source")
      ViewThatFits(in: .horizontal) {
        HStack(spacing: 12) { actions }
        VStack(alignment: .leading, spacing: 12) { actions }
      }
      if let exportedPath {
        VStack(alignment: .leading, spacing: 12) {
          Text("source \(ExternalClient.quote(exportedPath))")
            .font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
          Button("Copy command", systemImage: "doc.on.doc") {
            copy("source \(ExternalClient.quote(exportedPath))")
          }
        }
      }
    }
  }

  @ViewBuilder private var actions: some View {
    Button("Copy configuration", systemImage: "doc.on.doc") {
      copy(format.configuration(setup))
    }.fixedSize().accessibilityIdentifier("client-copy-configuration")
    if setup.hasKey {
      Button(exporting ? "Exporting…" : "Export credentials…", systemImage: "key", action: export)
        .fixedSize().disabled(exporting).accessibilityIdentifier("client-export-credentials")
    }
  }
}
