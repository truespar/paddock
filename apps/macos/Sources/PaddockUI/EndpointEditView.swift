import SwiftUI

/// ServerDetail is information + actions; ServerForm is a separate edit page.
/// The app owns the draft, so Back or an area switch never silently saves or
/// discards it. Existing reviewed-write/restart confirmations remain unchanged.
struct EndpointEditView: View {
  @Bindable var workspace: WorkspaceModel
  let port: UInt16
  var onDownload: ((String, String) -> Void)? = nil
  var body: some View {
    ScrollViewReader { proxy in
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 24) {
          Button("Model details", systemImage: "chevron.left") { workspace.editingEndpoint = false }
            .buttonStyle(QuietButtonStyle()).accessibilityIdentifier("endpoint-edit-back")
          if let editor = workspace.endpointEditor, editor.endpoint.port == port, !editor.removed {
            PageHeading(title: "Edit \(editor.title)") {
              EmptyView()
            }
            EndpointSettingsView(
              editor: editor, canMutate: workspace.canSubmit,
              onTools: { workspace.systemToolsPort = port },
              onDownload: onDownload.map { download in { download(editor.modelID, $0) } })
          } else {
            ContentUnavailableView(
              "Configuration unavailable", systemImage: "slider.horizontal.3",
              description: Text("No editable endpoint is owned by this app on this port."))
          }
        }.font(.system(size: 12)).padding(28).frame(maxWidth: 1080, alignment: .leading)
          .frame(maxWidth: .infinity, alignment: .top)
      }.background(PaddockStyle.canvas).accessibilityIdentifier("endpoint-edit-page")
        .modifier(SettingsSearchScroll(proxy: proxy))
    }
  }
}

struct EndpointSummaryView: View {
  let row: EndpointRow
  var body: some View {
    HStack(alignment: .top, spacing: 24) {
      if let runner = row.runner {
        VStack(alignment: .leading, spacing: 14) {
          Text("Live").font(.system(size: 16, weight: .semibold))
          FactRow(
            title: "Uptime",
            value: runner.uptimeS.map {
              Duration.seconds($0).formatted(.time(pattern: .hourMinuteSecond))
            } ?? "Not reported")
          FactRow(
            title: "Active requests", value: runner.inFlight.map(String.init) ?? "Not reported")
          if let live = row.configured?.runtimeState, live.pid == runner.pid {
            FactRow(title: "Context", value: "\(live.maxCtx.formatted()) tokens")
            FactRow(title: "Workload", value: "\(live.maxBatch) at once")
            if live.restartRequired == true {
              Text("Saved changes awaiting restart: " + live.changed.joined(separator: ", "))
                .font(.caption).foregroundStyle(PaddockStyle.caution)
            }
          }
          FactRow(title: "PID", value: String(runner.pid))
          FactRow(title: "Runner", value: runner.version ?? "Not reported")
        }.frame(maxWidth: .infinity, alignment: .topLeading)
          .accessibilityIdentifier("endpoint-live")
      }
      VStack(alignment: .leading, spacing: 14) {
        Text("Configuration").font(.system(size: 16, weight: .semibold))
        if let saved = row.configured {
          FactRow(title: "Model", value: saved.model ?? "Not reported")
          FactRow(
            title: "Context",
            value: saved.maxCtx.map { "\($0.formatted()) tokens" } ?? "Runner default")
          FactRow(title: "Concurrency", value: saved.maxBatch.map(String.init) ?? "Runner default")
          FactRow(title: "KV cache", value: saved.settings?.kvCacheDtype ?? "Auto")
          FactRow(title: "GPU", value: saved.settings?.device ?? "Not reported")
        } else {
          Text(
            "Adopted process - its launch configuration is not owned by this app. You can chat with it or explicitly stop it; its settings cannot be edited here."
          ).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
      }.frame(maxWidth: .infinity, alignment: .topLeading)
        .accessibilityIdentifier("endpoint-configuration")
    }.textSelection(.enabled)
  }
}
