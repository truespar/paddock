import AppKit
import PaddockClient
import SwiftUI

/// One row per port, stable across starting -> running -> stopped. This is the
/// Studio fleet's unit of navigation, not a separate row for every projection.
public struct EndpointRow: Identifiable {
  let port: UInt16
  let runner: RunnerInfo?
  let configured: ConfiguredEndpoint?
  let job: ManagementJob?
  public var id: UInt16 { port }
  var title: String { runner?.title ?? configured?.title ?? "Endpoint \(port)" }
  var baseURL: String { "http://127.0.0.1:\(port)/v1" }
  var isActive: Bool { runner != nil || configured?.running == true || job?.isActive == true }

  func matches(_ search: String) -> Bool {
    let fields = [title, String(port), configured?.model ?? "", configured?.artifact ?? "", status]
    return search.split(whereSeparator: \.isWhitespace).allSatisfy { term in
      fields.contains { $0.localizedStandardContains(String(term)) }
    }
  }
  var status: String {
    if let job, job.isActive {
      switch job.action {
      case "stop": return "Stopping"
      case "save": return "Saving"
      case "restart": return "Restarting"
      case "remove": return "Removing"
      default: return "Starting"
      }
    }
    if let runner {
      switch runner.status {
      case "ok": return "Running"
      case "draining": return "Stopping"
      default: return runner.status.capitalized
      }
    }
    if configured?.running == true { return "Checking runner" }
    return "Stopped"
  }

  static func rows(snapshot: ManagerSnapshot, latestJob: ManagementJob?) -> [Self] {
    let ports = Set(
      snapshot.runners.map(\.port) + (snapshot.servers ?? []).map(\.port)
        + (latestJob?.isActive == true ? [latestJob!.port].compactMap { $0 } : []))
    return ports.sorted().map { port in
      Self(
        port: port, runner: snapshot.runners.first { $0.port == port },
        configured: snapshot.servers?.first { $0.port == port },
        job: latestJob?.port == port ? latestJob : nil)
    }
  }
}

struct EndpointsView: View {
  @Bindable var workspace: WorkspaceModel
  let snapshot: ManagerSnapshot
  let onCreate: () -> Void
  var creation: AnyView? = nil
  private var selection: UInt16? {
    get { workspace.selectedEndpointPort }
    nonmutating set { workspace.selectedEndpointPort = newValue }
  }
  @State private var stopping: RunnerInfo?
  @State private var removing: EndpointRemovalReview?
  private var rows: [EndpointRow] {
    EndpointRow.rows(snapshot: snapshot, latestJob: workspace.latestJob)
  }
  @State private var copiedPort: UInt16?
  @State private var search = ""
  @State private var activeOnly = false
  private var visibleRows: [EndpointRow] {
    rows.filter { (!activeOnly || $0.isActive) && $0.matches(search) }
  }

  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 20) {
        PageHeading(title: ManagerDestination.runners.rawValue) {
          Button("Add instance", systemImage: "plus", action: onCreate)
            .modifier(PrimaryAction()).disabled(!workspace.canSubmit || creation != nil)
            .accessibilityIdentifier("instances-add")
        }
        if let creation {
          creation.accessibilityIdentifier("instance-creation")
        }
        if !rows.isEmpty {
          HStack(spacing: 16) {
            NativeSearchField(text: $search, prompt: "Search instances")
              .accessibilityIdentifier("instances-search")
            Picker("Instances", selection: $activeOnly) {
              Text("All").tag(false)
              Text("Active").tag(true)
            }.pickerStyle(.segmented).labelsHidden().frame(width: 140)
              .accessibilityIdentifier("instances-filter")
          }
        }
        if rows.isEmpty && creation == nil {
          VStack(spacing: 14) {
            Image(systemName: "terminal").font(.system(size: 30, weight: .light))
              .foregroundStyle(.secondary)
            Text("No instances yet").font(.system(size: 18, weight: .medium))
          }.frame(maxWidth: .infinity, minHeight: 320)
        } else {
          LazyVStack(spacing: 6) {
            ForEach(visibleRows) { row in
              endpoint(row)
            }
            if visibleRows.isEmpty && !rows.isEmpty {
              Text("No matching instances").foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, minHeight: 100)
            }
          }
        }
      }.padding(.horizontal, 32).padding(.top, 20).padding(.bottom, 32)
        .frame(maxWidth: 960, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .top)
    }.background(PaddockStyle.canvas)
      .endpointRemovalConfirmation(workspace: workspace, review: $removing)
      .confirmationDialog(
        "Stop \(stopping?.title ?? "this model")?",
        isPresented: Binding(get: { stopping != nil }, set: { if !$0 { stopping = nil } }),
        titleVisibility: .visible
      ) {
        if let runner = stopping {
          Button("Stop Model", role: .destructive) {
            stopping = nil
            Task { await workspace.submit(.stop(port: runner.port, pid: runner.pid)) }
          }
        }
        Button("Keep Running", role: .cancel) { stopping = nil }
      } message: {
        Text(
          "Active requests will drain before the model stops. Your weights and endpoint settings will stay saved."
        )
      }
  }

  private func endpoint(_ row: EndpointRow) -> some View {
    // A single opaque row owns identity, status and actions. Selection opens
    // the detail page; it is not a persistent highlight on half of this row.
    ViewThatFits(in: .horizontal) {
      HStack(spacing: 20) {
        identity(row).frame(minWidth: 220)
        actions(row)
      }
      VStack(alignment: .leading, spacing: 16) {
        identity(row)
        HStack {
          Spacer(minLength: 0)
          actions(row)
        }
      }
    }
    .padding(.horizontal, 14).padding(.vertical, 12)
    .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
  }

  private func identity(_ row: EndpointRow) -> some View {
    let model = snapshot.catalog.models.first { $0.id == row.configured?.model }
    let artifact = model?.artifacts.first { $0.id == row.configured?.artifact }
    return Button {
      selection = row.port
      workspace.openEndpoint(row)
    } label: {
      HStack(alignment: .center, spacing: 12) {
        ModelAvatar(vendor: model?.vendor, size: 30)
        VStack(alignment: .leading, spacing: 5) {
          HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(row.title).font(.system(size: 13, weight: .semibold)).lineLimit(1)
              .truncationMode(.middle)
            Text(":\(String(row.port))").font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.secondary).fixedSize()
          }
          HStack(spacing: 8) {
            EndpointStatusView(row: row)
            Text("·").accessibilityHidden(true)
            Text(artifact?.shortFormat ?? row.configured?.artifact ?? "Native runner")
              .lineLimit(1).truncationMode(.middle)
          }.font(.system(size: 11)).foregroundStyle(.secondary)
        }
        Spacer(minLength: 0)
      }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
    }.buttonStyle(.plain)
      .help("\(row.title)\n\(row.baseURL)\n\(artifact?.quant ?? row.configured?.artifact ?? "")")
      .accessibilityIdentifier("select-endpoint-\(row.port)")
  }

  private func actions(_ row: EndpointRow) -> some View {
    HStack(spacing: 8) {
      if row.runner?.status == "ok", row.runner?.hasStudioSurface == true {
        Button(
          row.runner?.studioActionTitle ?? "Open Studio",
          systemImage: row.runner?.studioActionSymbol ?? "bubble"
        ) { workspace.request(.chat(port: row.port)) }
        .buttonStyle(FlatButtonStyle(primary: true))
        .disabled(workspace.desktopNavigationBlocked || row.job?.isActive == true)
        .help(
          row.runner?.studioActionTitle ?? "Open Studio"
        )
        .accessibilityIdentifier("endpoint-studio-\(row.port)")
      } else if row.runner == nil, row.job?.isActive != true {
        Button("Start", systemImage: "play.fill") {
          if row.configured?.localOnly != true {
            workspace.openEndpoint(row)
            return
          }
          guard let revision = row.configured?.revision else { return }
          Task { await workspace.submit(.start(port: row.port, revision: revision)) }
        }.modifier(PrimaryAction())
          .disabled(
            !workspace.canSubmit || row.configured?.revision == nil
              || row.configured?.running == true
          )
          .accessibilityLabel("Start \(row.title)").accessibilityIdentifier("endpoint-start")
      }
      Menu {
        Button("View details and logs", systemImage: "info.circle") { workspace.openEndpoint(row) }
        if row.configured != nil {
          Button("Edit settings…", systemImage: "slider.horizontal.3") {
            workspace.openEndpoint(row)
            // A dirty editor for another port may have rejected navigation.
            if workspace.detailEndpointPort == row.port { workspace.editEndpoint() }
          }.disabled(row.job?.isActive == true)
        }
        Button(copiedPort == row.port ? "Copied URL" : "Copy URL", systemImage: "doc.on.doc") {
          NSPasteboard.general.clearContents()
          copiedPort =
            NSPasteboard.general.setString(row.baseURL, forType: .string) ? row.port : nil
        }
        if let runner = row.runner {
          Divider()
          Button("Stop model…", systemImage: "stop", role: .destructive) { stopping = runner }
            .disabled(!workspace.canSubmit || row.job?.isActive == true)
        } else if let review = EndpointRemovalReview(row) {
          Divider()
          Button("Remove configuration…", systemImage: "trash", role: .destructive) {
            removing = review
          }.disabled(!workspace.canRemoveEndpoint(review))
            .accessibilityIdentifier("endpoint-remove-menu-\(row.port)")
        }
      } label: {
        Image(systemName: "ellipsis").font(.system(size: 15, weight: .medium))
          .frame(width: 32, height: 32).contentShape(Rectangle())
      }.menuStyle(.button).menuIndicator(.hidden).buttonStyle(QuietButtonStyle())
        .accessibilityLabel("Actions for \(row.title) on port \(String(row.port))")
        .accessibilityIdentifier("endpoint-actions-\(row.port)")
        .help("Model actions")
    }.fixedSize()
  }
}

/// Status is information, not a capsule competing with the row's buttons.
/// Text and symbols remain understandable without relying on colour.
struct EndpointStatusView: View {
  let row: EndpointRow
  var body: some View {
    HStack(spacing: 5) {
      if row.job?.isActive == true {
        ProgressView().controlSize(.mini).frame(width: 10, height: 10)
          .accessibilityHidden(true)
      } else {
        Image(systemName: row.runner?.status == "ok" ? "circle.fill" : "circle")
          .font(.system(size: 6, weight: .semibold)).accessibilityHidden(true)
      }
      Text(row.status).fontWeight(.medium)
    }.fixedSize().accessibilityElement(children: .ignore).accessibilityLabel(row.status)
      .help(row.job.flatMap { $0.isActive ? $0.message : nil } ?? row.status)
      .foregroundStyle(
        row.runner != nil && row.runner?.status != "ok" && row.runner?.status != "draining"
          ? PaddockStyle.caution : PaddockStyle.secondary)
  }
}
