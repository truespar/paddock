import AppKit
import PaddockClient
import SwiftUI

/// A real Manager page, matching the endpoint lifetime in the web Manager.
/// Flat native controls; no raw configuration or saved credential crosses in.
struct EndpointDetailView: View {
  @Bindable var workspace: WorkspaceModel
  let port: UInt16
  let snapshot: ManagerSnapshot
  @State private var stopReview: RunnerInfo?
  @State private var startReview: ConfiguredEndpoint?
  @State private var copied = false
  @State private var removing: EndpointRemovalReview?
  private var row: EndpointRow? {
    EndpointRow.rows(snapshot: snapshot, latestJob: workspace.latestJob).first { $0.port == port }
  }
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 24) {
        Button(ManagerDestination.runners.rawValue, systemImage: "chevron.left") {
          workspace.detailEndpointPort = nil
        }
        .buttonStyle(QuietButtonStyle()).accessibilityIdentifier("endpoint-back")
        if let row {
          PageHeading(title: row.title, subtitle: "Port \(String(port)) · \(row.status)") {
            if let runner = row.runner {
              Button(
                runner.studioActionTitle,
                systemImage: runner.studioActionSymbol
              ) { workspace.request(.chat(port: port)) }
              .buttonStyle(FlatButtonStyle()).disabled(
                runner.status != "ok" || workspace.desktopNavigationBlocked)
              editButton
              Button("Stop", systemImage: "stop.fill") { stopReview = runner }
                .buttonStyle(FlatButtonStyle()).disabled(!workspace.canSubmit)
            } else if let configured = row.configured {
              editButton
              if let review = EndpointRemovalReview(row) {
                Button("Remove", systemImage: "trash", role: .destructive) { removing = review }
                  .buttonStyle(FlatButtonStyle()).disabled(!workspace.canRemoveEndpoint(review))
                  .accessibilityIdentifier("endpoint-remove-\(port)")
              }
              Button("Start", systemImage: "play.fill") {
                if configured.localOnly != true {
                  startReview = configured
                  return
                }
                guard let revision = configured.revision else { return }
                Task { await workspace.submit(.start(port: port, revision: revision)) }
              }.buttonStyle(FlatButtonStyle(primary: true))
                .disabled(
                  !workspace.canSubmit || configured.revision == nil
                    || workspace.endpointEditor?.dirty == true)
            }
          }
          HStack(spacing: 12) {
            Text("Endpoint").foregroundStyle(.secondary)
            Text(row.baseURL).font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
            Spacer()
            Button(copied ? "Copied" : "Copy URL", systemImage: "doc.on.doc") {
              NSPasteboard.general.clearContents()
              copied = NSPasteboard.general.setString(row.baseURL, forType: .string)
            }.buttonStyle(FlatButtonStyle())
          }
          WorkspaceRule()
          EndpointSummaryView(row: row)
          WorkspaceRule()
          EndpointLogsView(model: workspace.endpointLogs, port: port)
        } else {
          ContentUnavailableView(
            "Endpoint removed", systemImage: "terminal",
            description: Text("Model weights and conversations are unchanged."))
        }
      }.font(.system(size: 12)).padding(28).frame(maxWidth: 960, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .top)
    }.background(PaddockStyle.canvas)
      .endpointRemovalConfirmation(workspace: workspace, review: $removing)
      .confirmationDialog(
        "Stop this model?",
        isPresented: Binding(get: { stopReview != nil }, set: { if !$0 { stopReview = nil } }),
        titleVisibility: .visible
      ) {
        Button("Stop model", role: .destructive) {
          if let runner = stopReview {
            Task { await workspace.submit(.stop(port: port, pid: runner.pid)) }
          }
        }
        Button("Keep running", role: .cancel) {}
      } message: {
        Text(
          "Active requests will drain. Saved settings, model weights and conversations are kept.")
      }
      .confirmationDialog(
        "Start with network access?",
        isPresented: Binding(get: { startReview != nil }, set: { if !$0 { startReview = nil } }),
        titleVisibility: .visible
      ) {
        Button("Start network endpoint") {
          if let review = startReview, let revision = review.revision {
            Task {
              await workspace.submit(
                .start(port: review.port, revision: revision, allowNetwork: true))
            }
          }
        }
        Button("Cancel", role: .cancel) {}
      } message: {
        Text(
          "The model will listen on \(startReview?.settings?.host ?? "its saved network address") at port \(String(port)). Other devices may reach it and must supply the saved API key. Use only trusted networks."
        )
      }
  }
  @ViewBuilder private var editButton: some View {
    if workspace.endpointEditor?.endpoint.port == port {
      Button("Edit", systemImage: "pencil") { workspace.editEndpoint() }
        .buttonStyle(FlatButtonStyle()).accessibilityIdentifier("endpoint-edit")
    }
  }
}
