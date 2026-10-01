import PaddockClient
import SwiftUI

/// One model row on the left, persistent exact-export detail on the right.
/// Search/filter changes reconcile selection; ordinary snapshot refreshes do not.
struct ModelLibraryView: View {
  let snapshot: ManagerSnapshot
  var canStart = false
  var onStart: ((String, String) -> Void)?
  var onDownload: ((String, String) -> Void)?
  @Binding private var purpose: ModelStartPurpose
  @State private var query = ""
  @State private var downloadedOnly = false
  @State private var format: LibraryFormat
  @State private var selection: LibrarySelection?
  @AppStorage("modelLibrarySortOrder") private var order: LibraryOrder = .newest
  @FocusState private var searchFocused: Bool
  @FocusState private var listFocused: Bool

  init(
    snapshot: ManagerSnapshot, canStart: Bool = false,
    purpose: Binding<ModelStartPurpose> = .constant(.all),
    format: LibraryFormat = .all, onDownload: ((String, String) -> Void)? = nil,
    onStart: ((String, String) -> Void)? = nil
  ) {
    self.snapshot = snapshot
    self.canStart = canStart
    self.onStart = onStart
    self.onDownload = onDownload
    _purpose = purpose
    _format = State(initialValue: format)
  }

  private var visible: [LibraryEntry] {
    LibraryCatalog.entries(
      catalog: snapshot.catalog, backend: snapshot.readiness.backend,
      format: format, query: query, downloadedOnly: downloadedOnly, order: order, purpose: purpose)
  }
  private var active: LibrarySelection? {
    LibraryCatalog.selection(
      selection, in: visible, backend: snapshot.readiness.backend, purpose: purpose)
  }
  private var matchingExports: [String] {
    // Reordering is not filtering: keep an explicitly chosen export on sort.
    visible.map { $0.id + ":" + $0.artifacts.map(\.id).joined(separator: ",") }.sorted()
  }

  var body: some View {
    let visible = self.visible
    let active = LibraryCatalog.selection(
      selection, in: visible, backend: snapshot.readiness.backend, purpose: purpose)
    CatalogColumns {
      modelList(visible, active: active)
    } detail: {
      if let active, let entry = visible.first(where: { $0.id == active.model }) {
        ModelDetailView(
          model: entry.model, backend: snapshot.readiness.backend, canStart: canStart,
          onStart: onStart,
          onDownload: onDownload,
          purpose: purpose,
          artifactID: Binding(
            get: { self.active?.artifact },
            set: { if let id = $0 { selection = LibrarySelection(model: entry.id, artifact: id) } })
        ).id(entry.id).frame(minWidth: 350, maxWidth: .infinity, maxHeight: .infinity)
      } else {
        VStack(spacing: 12) {
          Image(systemName: "square.stack").font(.system(size: 26, weight: .light))
          Text(purpose == .speech ? "No matching speech models" : "No matching models")
            .font(.system(size: 17, weight: .medium))
          Text("Try another search or change the filters.")
            .font(.system(size: 12)).foregroundStyle(.secondary)
          Button("Clear filters") {
            query = ""
            format = .all
            downloadedOnly = false
          }.buttonStyle(FlatButtonStyle())
        }.foregroundStyle(.secondary)
          .frame(minWidth: 350, maxWidth: .infinity, maxHeight: .infinity)
      }
    }.frame(maxWidth: .infinity, maxHeight: .infinity).background(PaddockStyle.canvas)
      .accessibilityIdentifier("model-browser")
      .onChange(of: matchingExports, initial: true) { _, _ in reconcile() }
      .onChange(of: format) { _, _ in reconcile() }
      .onChange(of: query) { _, _ in reconcile() }
      .onChange(of: downloadedOnly) { _, _ in reconcile() }
      .onChange(of: purpose) { _, _ in reconcile() }
  }

  private func reconcile() {
    selection = LibraryCatalog.selection(
      selection, in: visible, backend: snapshot.readiness.backend, requireMatchingExport: true,
      purpose: purpose)
  }

  private func modelList(_ visible: [LibraryEntry], active: LibrarySelection?) -> some View {
    VStack(alignment: .leading, spacing: 0) {
      VStack(spacing: 14) {
        search
        if purpose != .all {
          HStack {
            Label(purpose.rawValue, systemImage: purpose == .speech ? "microphone" : "tablecells")
            Spacer(minLength: 4)
            Button("Show all models") { purpose = .all }
              .buttonStyle(.plain).foregroundStyle(.secondary)
          }.font(.system(size: 11))
            .accessibilityIdentifier(
              purpose == .speech ? "model-speech-filter" : "model-tables-filter")
        }
        HStack(spacing: 8) {
          Dropdown(title: "Collection", value: downloadedOnly ? "On this Mac" : "All models") {
            Picker("Collection", selection: $downloadedOnly) {
              Text("All models").tag(false)
              Text("On this Mac").tag(true)
            }.pickerStyle(.inline)
          }
          .accessibilityIdentifier("model-collection-filter")
          Spacer(minLength: 4)
          Dropdown(title: "Format", value: format.rawValue) {
            Picker("Format", selection: $format) {
              ForEach(LibraryFormat.allCases) { option in
                Text(option.rawValue).tag(option)
              }
            }.pickerStyle(.inline)
          }
          .accessibilityIdentifier("model-format-filter")
        }.font(.system(size: 11)).foregroundStyle(.secondary)
        HStack {
          Text("Order by").foregroundStyle(.secondary)
          Spacer(minLength: 4)
          Dropdown(title: "Order by", value: order.shortTitle) {
            Picker("Order by", selection: $order) {
              ForEach(LibraryOrder.allCases) { option in
                Text(option.title).tag(option)
              }
            }.pickerStyle(.inline)
          }
          .accessibilityValue(order.title)
          .accessibilityIdentifier("model-sort-order")
          .help(
            "Upstream publication dates; undated models sort last. Download size compares the smallest matching weights option, without companions."
          )
        }.font(.system(size: 11))
      }.padding(16)
      WorkspaceRule()
      HStack {
        Text("\(visible.count) \(visible.count == 1 ? "model" : "models")")
        Spacer()
        Text("Vision · Thinking · Tools")
      }.font(.system(size: 10)).foregroundStyle(.secondary)
        .padding(.horizontal, 16).padding(.vertical, 12)
      ScrollViewReader { proxy in
        PaddockScrollView {
          LazyVStack(spacing: 2) {
            ForEach(visible) { entry in
              modelRow(entry, selected: active?.model == entry.id).id(entry.id)
            }
          }.padding(.horizontal, 8).padding(.bottom, 16)
        }.focusable().focusEffectDisabled().focused($listFocused)
          .onMoveCommand { direction in
            guard direction == .up || direction == .down, !visible.isEmpty else { return }
            let index = visible.firstIndex { $0.id == active?.model } ?? 0
            let next = max(0, min(visible.count - 1, index + (direction == .down ? 1 : -1)))
            select(visible[next])
            proxy.scrollTo(visible[next].id)
          }
          .onChange(of: order) { _, _ in
            if let first = visible.first { proxy.scrollTo(first.id, anchor: .top) }
          }
      }.accessibilityIdentifier("model-list")
    }.background(PaddockStyle.canvas)
  }

  private var search: some View {
    HStack(spacing: 8) {
      Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
      TextField("Search models", text: $query).textFieldStyle(.plain)
        .focused($searchFocused).accessibilityLabel("Search models")
        .accessibilityIdentifier("model-search")
      if !query.isEmpty {
        Button("Clear search", systemImage: "xmark.circle.fill") { query = "" }
          .labelStyle(.iconOnly).buttonStyle(.plain).foregroundStyle(.secondary)
      }
      Button("⌘F") { searchFocused = true }.buttonStyle(.plain)
        .font(.system(size: 10)).foregroundStyle(.secondary)
        .keyboardShortcut("f", modifiers: .command).accessibilityLabel("Focus model search")
    }.font(.system(size: 12)).padding(.horizontal, 10).frame(height: 36)
      .background(
        PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
      )
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
          .stroke(searchFocused ? PaddockStyle.accent : PaddockStyle.border, lineWidth: 1))
  }

  private func select(_ entry: LibraryEntry) {
    selection = LibraryCatalog.selection(
      selection?.model == entry.id ? selection : nil, in: [entry],
      backend: snapshot.readiness.backend, requireMatchingExport: true, purpose: purpose)
  }

  private func modelRow(_ entry: LibraryEntry, selected: Bool) -> some View {
    ModelListRow(height: 76, selected: selected) {
      select(entry)
      listFocused = true
    } content: {
      HStack(spacing: 10) {
        ModelAvatar(vendor: entry.model.vendor, size: 34)
        VStack(alignment: .leading, spacing: 6) {
          HStack(spacing: 6) {
            Text(entry.model.display).font(.system(size: 12, weight: .medium))
              .lineLimit(1).help(entry.model.display)
            Spacer(minLength: 0)
            ModelFeatureIcons(capabilities: entry.capabilities)
          }
          Text(entry.model.specs?.about ?? entry.model.vendor ?? entry.model.id)
            .font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
          HStack(spacing: 6) {
            Text(
              Array(Set(entry.artifacts.map { $0.isMLX ? "MLX" : $0.format.uppercased() })).sorted()
                .joined(separator: " · "))
            if entry.installed {
              Image(systemName: "checkmark.circle").help("Matching weights downloaded")
            }
            Spacer(minLength: 0)
            if order == .smallest || order == .largest, let size = entry.smallestDownload {
              Text("From \(DisplayFormat.bytes(size))")
                .help("Smallest matching weights option, without companions")
            } else if let date = entry.publishedAt {
              Text(date).help("Upstream publication date")
            }
          }.font(.system(size: 10)).foregroundStyle(.secondary)
        }
      }
    }
    .accessibilityLabel(
      "\(entry.model.display). \(ModelFeature.allCases.map { $0.description(in: entry.capabilities) }.joined(separator: ". ")). Capabilities depend on the weights option."
    ).accessibilityIdentifier("model-row-\(entry.id)")
  }
}

struct ModelFeatureIcons: View {
  let capabilities: Set<String>
  var body: some View {
    HStack(spacing: 6) {
      ForEach(ModelFeature.allCases) { feature in
        Image(systemName: feature.symbol).font(.system(size: 11))
          .foregroundStyle(
            capabilities.contains(feature.rawValue) ? Color.primary : Color.secondary.opacity(0.35)
          )
          .help(
            "\(feature.description(in: capabilities)) for matching weights. Review the selected option's capabilities."
          )
      }
    }.accessibilityElement(children: .ignore)
      .accessibilityLabel(
        ModelFeature.allCases.map { $0.description(in: capabilities) }.joined(separator: ". "))
  }
}
