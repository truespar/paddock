import PaddockClient
import SwiftUI

/// The web connector catalog translated into native controls, owned by Manager.
struct IntegrationsView: View {
  @Bindable var model: IntegrationsModel
  let endpoints: [ConfiguredEndpoint]
  @State private var discovering = true
  @State private var catalog = ConnectorCatalog()
  @State private var expanded = Set<String>()
  @State private var removing: NativeConnector?
  @Environment(\.settingsSearchRequest) private var settingsRequest
  private var hits: [ConnectorHit] { catalog.rows(model.hits) }

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text("Connectors").font(.system(size: 24, weight: .semibold))
      HStack(spacing: 18) {
        tab("Find connectors", active: discovering) { discovering = true }
          .settingsSearchAnchor(.connectorDirectory)
        tab("Your connectors (\(model.rows.count))", active: !discovering) { discovering = false }
          .settingsSearchAnchor(.connectorLibrary)
        Spacer()
        Button("Add by URL", systemImage: "plus") { model.review() }
          .buttonStyle(FlatButtonStyle()).disabled(model.saving)
      }
      WorkspaceRule()
      if discovering {
        HStack(spacing: 8) {
          Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
          TextField("Search the MCP directory - stripe, github, weather…", text: $model.query)
            .textFieldStyle(.plain).accessibilityIdentifier("connector-catalog-search")
          if model.searching { ProgressView().controlSize(.small) }
          if !model.query.isEmpty {
            Button("Clear search", systemImage: "xmark") { model.query = "" }
              .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
          }
        }.padding(11).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 7))
          .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder(PaddockStyle.border))
        HStack(spacing: 8) {
          chip("Reachable", active: catalog.reachableOnly) { catalog.reachableOnly.toggle() }
          Divider().frame(height: 16)
          ForEach(["First-party", "Trusted", "Community"], id: \.self) { tier in
            chip(tier, active: catalog.tiers.contains(tier)) {
              if catalog.tiers.contains(tier) {
                catalog.tiers.remove(tier)
              } else {
                catalog.tiers.insert(tier)
              }
            }
          }
          Spacer()
          if catalog.sort != .rank {
            Button("Registry ranking") { catalog.sort = .rank }.buttonStyle(QuietButtonStyle())
          }
          Text("\(hits.count) servers").foregroundStyle(.secondary).monospacedDigit()
        }.font(.system(size: 11))
        if let error = model.catalogError {
          HStack {
            Text(error).foregroundStyle(PaddockStyle.caution)
            Button("Retry") { Task { await model.search() } }.buttonStyle(QuietButtonStyle())
          }
        }
        if hits.isEmpty {
          Text(
            model.searching
              ? "Asking the directory…" : "Nothing matches this search and these filters."
          )
          .foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
          catalogTable
        }
      } else {
        installed
      }
      if let message = model.message { Text(message).foregroundStyle(.secondary) }
      if let error = model.error {
        Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
      }
    }.font(.system(size: 12)).padding(.horizontal, 28).padding(.top, 22).padding(.bottom, 18)
      .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
      .background(PaddockStyle.canvas)
      .task { await model.refresh() }
      .task(id: settingsRequest?.id) {
        switch settingsRequest?.entry.anchor {
        case .connectorDirectory: discovering = true
        case .connectorLibrary: discovering = false
        default: break
        }
      }
      .task(id: discovering ? model.query : "\u{0}") { if discovering { await model.search() } }
      .confirmationDialog(
        "Remove this connector?",
        isPresented: Binding(get: { removing != nil }, set: { if !$0 { removing = nil } }),
        titleVisibility: .visible
      ) {
        if let row = removing {
          Button("Remove \(row.label)", role: .destructive) {
            Task { await model.write(.remove(id: row.id, revision: row.revision)) }
            removing = nil
          }
        }
        Button("Cancel", role: .cancel) { removing = nil }
      } message: {
        Text(
          "Remove it from your library and model endpoints. Saved conversations are kept; models keep running."
        )
      }
  }
  private func tab(_ title: String, active: Bool, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      Text(title).font(.system(size: 13, weight: active ? .semibold : .regular))
        .foregroundStyle(active ? .primary : .secondary).padding(.vertical, 8)
        .overlay(alignment: .bottom) { if active { Rectangle().frame(height: 2) } }
    }.buttonStyle(QuietButtonStyle()).accessibilityLabel(title).accessibilityAddTraits(
      active ? .isSelected : [])
  }
  private func chip(_ title: String, active: Bool, action: @escaping () -> Void) -> some View {
    Button(title, action: action).buttonStyle(FlatButtonStyle(primary: active))
      .accessibilityValue(active ? "Selected" : "Not selected")
  }
  private func heading(_ name: String, _ sort: ConnectorCatalog.Sort, width: CGFloat? = nil)
    -> some View
  {
    Button {
      catalog.order(by: sort)
    } label: {
      HStack(spacing: 4) {
        Text(name)
        if catalog.sort == sort {
          Image(systemName: catalog.ascending ? "chevron.up" : "chevron.down").font(
            .system(size: 8))
        }
      }.frame(width: width, alignment: .leading)
    }.buttonStyle(QuietButtonStyle()).accessibilityLabel("Sort by \(name)")
      .accessibilityValue(
        catalog.sort == sort ? (catalog.ascending ? "Ascending" : "Descending") : "Unsorted")
  }
  private var catalogTable: some View {
    GeometryReader { geometry in
      PaddockScrollView(.horizontal) {
        VStack(spacing: 0) {
          HStack(spacing: 12) {
            heading("Server", .name).frame(maxWidth: .infinity, alignment: .leading)
            Text("Domain").frame(width: 130, alignment: .leading)
            heading("Provenance", .tier, width: 86)
              .help("Directory attribution, not a Paddock security review.")
            heading("Tools", .tools, width: 42)
            heading("Stars", .stars, width: 46)
            heading("Status", .status, width: 40)
            Color.clear.frame(width: 66, height: 1)
          }.font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
            .padding(.horizontal, 12).padding(.vertical, 10)
          WorkspaceRule()
          PaddockScrollView {
            LazyVStack(spacing: 0) {
              ForEach(hits) { hit in
                catalogRow(hit)
                if expanded.contains(hit.key) { catalogDetail(hit) }
                WorkspaceRule()
              }
            }
          }
        }.frame(width: max(730, geometry.size.width), height: geometry.size.height)
      }
    }
  }
  private func catalogRow(_ hit: ConnectorHit) -> some View {
    let status = ConnectorCatalog.status(hit)
    let added = ConnectorCatalog.added(model.details[hit.key] ?? hit, rows: model.rows)
    return HStack(spacing: 12) {
      Button {
        if expanded.contains(hit.key) {
          expanded.remove(hit.key)
        } else {
          expanded.insert(hit.key)
          Task { await model.loadDetail(hit) }
        }
      } label: {
        HStack(spacing: 8) {
          Image(systemName: expanded.contains(hit.key) ? "chevron.down" : "chevron.right").font(
            .system(size: 9))
          Text(hit.name.isEmpty ? hit.domain : hit.name).fontWeight(.medium).lineLimit(1)
            .frame(maxWidth: .infinity, alignment: .leading)
          Text(hit.domain).foregroundStyle(.secondary).frame(width: 130, alignment: .leading)
            .lineLimit(1)
          Text(ConnectorCatalog.tier(hit)).foregroundStyle(.secondary).frame(
            width: 86, alignment: .leading)
          Text(hit.toolCount.map { String($0) } ?? "-").frame(width: 42, alignment: .trailing)
          Text(hit.githubStars.map { String($0) } ?? "-").frame(width: 46, alignment: .trailing)
          Image(systemName: status.symbol).frame(width: 40).help(status.label)
        }.frame(maxWidth: .infinity).contentShape(Rectangle())
      }.buttonStyle(QuietButtonStyle()).accessibilityLabel(
        "\(hit.name), \(status.label), show details")
      Group {
        if added {
          Label("Added", systemImage: "checkmark").font(.system(size: 10)).foregroundStyle(
            .secondary)
        } else if hit.liveness != "dead" {
          Button(model.addingKey == hit.key ? "Adding…" : "Add") { Task { await model.add(hit) } }
            .buttonStyle(FlatButtonStyle()).disabled(model.addingKey != nil || model.saving)
            .accessibilityLabel("Add \(hit.name)")
        } else {
          Text("-").foregroundStyle(.secondary)
        }
      }.frame(width: 66)
    }.font(.system(size: 11)).padding(.horizontal, 12).padding(.vertical, 9)
      .background(expanded.contains(hit.key) ? PaddockStyle.surface : PaddockStyle.canvas)
  }
  private func catalogDetail(_ hit: ConnectorHit) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      if model.detailLoading.contains(hit.key) {
        ProgressView("Asking the directory…").controlSize(.small)
      }
      if let error = model.detailErrors[hit.key] {
        HStack {
          Text(error).foregroundStyle(PaddockStyle.caution)
          Button("Retry") { Task { await model.loadDetail(hit) } }.buttonStyle(QuietButtonStyle())
        }
      }
      let detail = model.details[hit.key] ?? hit
      Text(detail.description).textSelection(.enabled)
      if let endpoint = ConnectorCatalog.endpoint(detail) {
        detailFact("Endpoint", endpoint)
      }
      if let categories = detail.categories, !categories.isEmpty {
        detailFact("Categories", categories.joined(separator: ", "))
      }
      if let license = detail.spdxLicense { detailFact("License", license) }
      if let note = detail.connection?.note, !note.isEmpty {
        Text(note).foregroundStyle(.secondary)
      }
      if let tools = detail.tools, !tools.isEmpty {
        Text(
          tools.prefix(24).joined(separator: " · ")
            + (tools.count > 24 ? " · +\(tools.count - 24) more" : "")
        )
        .font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
      }
      HStack(spacing: 18) {
        if let raw = detail.repoUrl, let url = URL(string: raw) {
          Link("Repository ↗", destination: url)
        }
        if let raw = detail.homepage, let url = URL(string: raw) {
          Link("Homepage ↗", destination: url)
        }
      }.foregroundStyle(.primary)
    }.font(.system(size: 12)).padding(20).frame(maxWidth: .infinity, alignment: .leading)
      .background(PaddockStyle.surface)
  }
  private func detailFact(_ label: String, _ value: String) -> some View {
    HStack(alignment: .top) {
      Text(label).foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
      Text(value).textSelection(.enabled)
    }
  }
  private var installed: some View {
    PaddockScrollView {
      LazyVStack(spacing: 0) {
        if model.rows.isEmpty {
          VStack(spacing: 14) {
            Image(systemName: "puzzlepiece.extension").font(.system(size: 26)).foregroundStyle(
              .secondary)
            Text(model.loading ? "Loading connectors…" : "Nothing added yet.").foregroundStyle(
              .secondary)
            Button("Add by URL", systemImage: "plus") { model.review() }.buttonStyle(
              FlatButtonStyle(primary: true))
          }.frame(maxWidth: .infinity).padding(.vertical, 60)
        }
        ForEach(model.rows) { row in
          HStack(spacing: 12) {
            Button {
              model.review(row)
            } label: {
              VStack(alignment: .leading, spacing: 5) {
                Text(row.label).fontWeight(.medium)
                Text(row.url).foregroundStyle(.secondary).lineLimit(1)
              }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
            }.buttonStyle(QuietButtonStyle()).accessibilityLabel("Edit \(row.label)")
            if row.hasHeaders { Text("Headers").foregroundStyle(.secondary) }
            if row.credentialReady == false {
              Button("Unlock…", systemImage: "lock") {
                Task { await model.write(.unlock(id: row.id, revision: row.revision)) }
              }.buttonStyle(FlatButtonStyle()).help(
                "Authorize saved credentials for this app session")
            } else if row.connected {
              Image(systemName: "checkmark.circle").help("Signed in")
            }
            if row.system || !row.ports.isEmpty {
              Text(row.system ? "Every model" : "\(row.ports.count) models").foregroundStyle(
                .secondary
              )
              .help("These endpoints and their API clients can use this connector.")
            }
            Button("Edit", systemImage: "pencil") { model.review(row) }.labelStyle(.iconOnly)
              .buttonStyle(QuietButtonStyle())
            Button("Remove", systemImage: "trash") { removing = row }.labelStyle(.iconOnly)
              .buttonStyle(QuietButtonStyle())
          }.padding(.vertical, 16).disabled(model.saving)
          WorkspaceRule()
        }
      }
    }.frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}
