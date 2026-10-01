import AppKit
import SwiftUI

/// A real system menu: keyboard navigation, VoiceOver, hit testing, dismissal
/// and light/dark materials remain AppKit-owned. No embedded dashboard or GPU UI.
public struct DesktopMenu: View {
  @Bindable var workspace: WorkspaceModel
  let open: (DesktopAction) -> Void
  let question: () -> Void
  let settings: () -> Void
  public init(
    workspace: WorkspaceModel, open: @escaping (DesktopAction) -> Void,
    question: @escaping () -> Void, settings: @escaping () -> Void
  ) {
    self.workspace = workspace
    self.open = open
    self.question = question
    self.settings = settings
  }
  public var body: some View {
    Button("New Question…", systemImage: "square.and.pencil", action: question)
    Button("Open Paddock") { open(.studio) }
    Divider()
    if workspace.snapshot == nil {
      Text("Connecting to local models…")
    } else {
      if case .failed = workspace.state {
        Text("Model status unavailable - showing last known state")
      }
      let active = workspace.desktopRows.filter {
        $0.runner != nil || $0.configured?.running == true || $0.job?.isActive == true
      }
      if active.isEmpty { Text("No running models") }
      ForEach(active) { row in
        Menu(row.desktopMenuTitle(in: active)) {
          Text("Port \(String(row.port))")
          if let runner = row.runner {
            Text(
              "\(runner.inFlight.map { "\($0) active requests" } ?? "Active request count unavailable")"
            )
            Button(
              runner.tabular != nil
                ? "Open Tables"
                : runner.reader != nil
                  ? "Open Reads" : runner.image == nil ? "Chat with Model" : "Create Image"
            ) {
              open(.chat(port: row.port))
            }
            .disabled(
              runner.status != "ok"
                || (runner.model == nil && runner.image == nil && runner.reader == nil
                  && runner.tabular == nil)
                || !workspace.canSubmit)
            Button("Copy API Address") {
              NSPasteboard.general.clearContents()
              NSPasteboard.general.setString(row.baseURL, forType: .string)
            }
            Button("View Details") { open(.endpoint(port: row.port)) }
            Divider()
            Button("Stop Model…") {
              // A captured identity, never a lookup of a possibly replaced PID
              // after the person has approved the stop.
              let alert = DesktopAlert.make()
              alert.messageText = "Stop \(runner.title)?"
              alert.informativeText =
                "This affects Studio and external clients using port \(runner.port). Requests may drain for up to 30 seconds; longer requests can be interrupted. Weights and endpoint settings remain saved."
              alert.addButton(withTitle: "Keep Running")
              alert.addButton(withTitle: "Stop Model")
              NSApp.activate(ignoringOtherApps: true)
              if alert.runModal() == .alertSecondButtonReturn {
                Task {
                  if !((await workspace.submit(.stop(port: runner.port, pid: runner.pid)))) {
                    workspace.desktopError =
                      workspace.commandError ?? "The model operation was not accepted."
                    open(.manager)
                  }
                }
              }
            }.disabled(!workspace.canSubmit)
          } else {
            Button("View Details") { open(.endpoint(port: row.port)) }
          }
        }
      }
    }
    Menu("Start Model") {
      let stopped = workspace.desktopRows.filter(\.canQuickStart)
      ForEach(stopped) { row in
        Button(row.desktopMenuTitle(in: stopped)) {
          guard let revision = row.configured?.revision else { return }
          Task {
            if !((await workspace.submit(.start(port: row.port, revision: revision)))) {
              workspace.desktopError =
                workspace.commandError ?? "The model operation was not accepted."
              open(.manager)
            }
          }
        }.disabled(!workspace.canSubmit)
      }
      Divider()
      Button("Choose Model…") { open(.startModel) }.disabled(!workspace.canSubmit)
    }
    if let job = workspace.latestJob, job.isActive { Text(job.message) }
    if workspace.latestJob?.state == "failed" {
      Button("Model operation failed - View Details") { open(.manager) }
    }
    Divider()
    Button("Settings…", action: settings)
    Button("Quit Paddock…") { NSApp.terminate(nil) }.keyboardShortcut("q")
  }
}

extension EndpointRow {
  var desktopStatus: String { status == "Running" ? "Ready" : status }
  func desktopMenuTitle(in rows: [EndpointRow]) -> String {
    let duplicate = rows.contains {
      $0.port != port && $0.title.localizedCaseInsensitiveCompare(title) == .orderedSame
    }
    let name = duplicate ? "\(title) (Port \(String(port)))" : title
    // Keep failures/transitions visible, without appending "Ready" to every model.
    return ["Ready", "Stopped"].contains(desktopStatus) ? name : "\(name) — \(desktopStatus)"
  }
  var canQuickStart: Bool {
    runner == nil && configured?.running == false && configured?.localOnly == true
      && configured?.revision != nil && job?.isActive != true
  }
}
