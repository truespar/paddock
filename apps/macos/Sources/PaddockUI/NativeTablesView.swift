import AppKit
import PaddockConversationCore
import SwiftUI
import UniformTypeIdentifiers

struct NativeTablesView: View {
  @Bindable var model: NativeTablesModel
  var onStart: () -> Void
  @State private var importing = false
  @State private var exporting = false
  @State private var exportDocument = TabularCSVDocument(text: "")
  @State private var showSource = false
  @State private var showAPI = false
  @State private var confirmExample = false

  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 20) {
        ViewThatFits(in: .horizontal) {
          HStack {
            title
            Spacer()
            modelPicker.frame(width: 340)
          }
          VStack(alignment: .leading, spacing: 12) {
            title
            modelPicker
          }
        }
        if let error = model.history.error ?? model.error ?? model.discoveryError {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
            .accessibilityIdentifier("tables-error")
          if model.history.error != nil {
            HStack {
              Button("Retry save") { Task { await model.saveSession() } }
              Button("Save a copy") { Task { await model.saveCopy() } }
            }.buttonStyle(FlatButtonStyle()).disabled(model.historyNavigationBlocked)
          }
        }
        if model.predictors.isEmpty {
          HStack(spacing: 12) {
            if model.loading { ProgressView().controlSize(.small) }
            Text("No table model is running").foregroundStyle(.secondary)
            Spacer()
            Button("Start a model", action: onStart).buttonStyle(FlatButtonStyle(primary: true))
          }
        }
        input
        if let table = model.table, let spec = model.spec {
          configuration(table, spec)
        }
        if let result = model.result { results(result) }
        if !model.curl.isEmpty {
          DisclosureGroup("API call", isExpanded: $showAPI) {
            HStack {
              Text("/v1/tabular/predictions").font(.system(.caption, design: .monospaced))
              Spacer()
              Button("Copy complete request") { copy(model.curl) }.buttonStyle(FlatButtonStyle())
            }.padding(.top, 8)
            // The full request remains available to copy without mounting a
            // multi-megabyte selectable text view on the UI thread.
            Text(model.curlPreview)
              .font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading).padding(12)
          }
        }
      }.padding(24).frame(maxWidth: 1400).frame(maxWidth: .infinity)
    }.background(PaddockStyle.canvas).font(.system(size: 13))
      .accessibilityIdentifier("native-tables")
      .task {
        await model.history.refresh(api: model.api)
        repeat {
          await model.refresh()
          do { try await Task.sleep(for: .seconds(5)) } catch { return }
        } while !Task.isCancelled
      }
      .onDisappear { Task { await model.saveSession() } }
      .fileImporter(
        isPresented: $importing,
        allowedContentTypes: [.commaSeparatedText, .tabSeparatedText, .plainText]
      ) { result in
        switch result {
        case .success(let url): Task { await model.importFile(url) }
        case .failure(let error): model.report(error)
        }
      }
      .fileExporter(
        isPresented: $exporting, document: exportDocument, contentType: .commaSeparatedText,
        defaultFilename: "predictions.csv"
      ) { result in
        if case .failure(let error) = result { model.report(error) }
      }
      .confirmationDialog(
        "Replace this table with the example?", isPresented: $confirmExample,
        titleVisibility: .visible
      ) {
        Button("Load example") { model.loadExample() }
        Button("Cancel", role: .cancel) {}
      }
  }
  private var title: some View {
    Text(model.sessionTitle).font(.system(size: 23, weight: .semibold)).tracking(-0.5)
      .lineLimit(2).help(model.sessionTitle).accessibilityAddTraits(.isHeader)
  }
  private var modelPicker: some View {
    Dropdown(
      title: "Table model", value: model.current?.title ?? "Choose model", fillsWidth: true,
      vendor: model.current?.vendor
    ) {
      ForEach(model.predictors) { predictor in
        Button {
          model.port = predictor.port
        } label: {
          ModelProviderMenuLabel(
            title: predictor.title + " · \(predictor.port)", vendor: predictor.vendor)
        }
      }
    }.disabled(model.busy || model.predictors.isEmpty).accessibilityIdentifier(
      "tables-model-picker")
  }
  private var input: some View {
    EndpointFormCard("Your table") {
      ViewThatFits(in: .horizontal) {
        HStack {
          importButtons
          Spacer()
          sourceToggle
        }
        VStack(alignment: .leading, spacing: 10) {
          importButtons
          sourceToggle
        }
      }
      if showSource || model.table == nil {
        TextEditor(text: Binding(get: { model.source }, set: { model.setSource($0) }))
          .font(.system(size: 12, design: .monospaced))
          .frame(height: 135).scrollContentBackground(.hidden)
          .padding(8).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
          .accessibilityLabel("CSV or spreadsheet table").disabled(model.busy)
        if model.source.isEmpty {
          Text("Paste a table with a header. Leave the target empty on the rows to predict.")
            .font(.caption).foregroundStyle(.secondary)
        }
      }
      if let table = model.table, !model.importing {
        NativeTabularGrid(version: model.tableVersion, header: table.header, rows: table.rows)
          .frame(height: min(280, CGFloat(table.rows.count + 1) * 26 + 16))
          .clipShape(RoundedRectangle(cornerRadius: 8)).accessibilityLabel("Input table")
        Text(
          "\(table.rows.count) rows · \(table.header.count) columns"
            + (model.fileName.isEmpty ? "" : " · \(model.fileName)")
        )
        .font(.caption).foregroundStyle(.secondary)
      }
      if model.importing { ProgressView().controlSize(.small) }
      if model.table == nil, let validation = model.validation {
        Text(validation).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
      }
    }.dropDestination(for: URL.self) { urls, _ in
      guard urls.count == 1, let url = urls.first, url.isFileURL, !model.busy else { return false }
      Task { await model.importFile(url) }
      return true
    }
  }
  private var sourceToggle: some View {
    HStack {
      if model.table != nil {
        Button(showSource ? "Hide source" : "Edit source") { showSource.toggle() }
        Button("Clear") { model.setSource("") }.disabled(model.busy)
      }
    }.buttonStyle(FlatButtonStyle())
  }
  private var importButtons: some View {
    HStack(spacing: 8) {
      Button("Open CSV", systemImage: "doc") { importing = true }
      Button("Paste", systemImage: "clipboard") {
        if let text = NSPasteboard.general.string(forType: .string) { model.setSource(text) }
      }
      Button("Load an example") {
        if model.hasWork { confirmExample = true } else { model.loadExample() }
      }
    }.buttonStyle(FlatButtonStyle()).disabled(model.busy)
  }
  private func configuration(_ table: TabularTable, _ spec: TabularSpec) -> some View {
    EndpointFormCard("Predict") {
      Dropdown(title: "Target column", value: table.header[spec.target], fillsWidth: true) {
        ForEach(table.header.indices, id: \.self) { i in
          Button(table.header[i]) { model.spec?.target = i }
        }
      }.disabled(model.busy).accessibilityIdentifier("tables-target")
      Table(model.columns) {
        TableColumn("Use") { column in
          if column.id == spec.target {
            Text("Target").font(.caption).foregroundStyle(.secondary)
          } else {
            Toggle(
              "Use \(column.name)",
              isOn: Binding(
                get: { model.spec?.use[column.id] ?? false },
                set: { model.spec?.use[column.id] = $0 })
            ).labelsHidden().toggleStyle(.checkbox)
          }
        }.width(48)
        TableColumn("Column", value: \.name).width(min: 90, ideal: 170)
        TableColumn("Read as") { column in
          if column.id == spec.target {
            Text(model.current?.limits.task == .regression ? "Number" : "Class").foregroundStyle(
              .secondary)
          } else {
            Dropdown(title: "Type of \(column.name)", value: spec.types[column.id].rawValue) {
              ForEach(TabularColumnType.allCases, id: \.self) { type in
                Button(type.rawValue) { model.spec?.types[column.id] = type }
              }
            }
          }
        }.width(min: 120, ideal: 140)
        TableColumn("Values", value: \.sample).width(min: 100, ideal: 220)
        TableColumn("Missing") { column in
          Text(column.missing.formatted()).monospacedDigit().foregroundStyle(.secondary)
        }.width(58)
      }.frame(height: min(260, CGFloat(table.header.count + 1) * 32 + 12))
        .disabled(model.busy).accessibilityIdentifier("tables-columns")
      ViewThatFits(in: .horizontal) {
        HStack(spacing: 16) {
          options
          Spacer()
          runButton
        }
        VStack(alignment: .leading, spacing: 12) {
          options
          runButton
        }
      }
      if let plan = model.plan {
        Text(
          "\(plan.contextRows.count) labelled rows · \(plan.queryRows.count) to predict · \(plan.features.count) input columns"
        )
        .font(.caption).foregroundStyle(.secondary)
      }
      if let validation = model.validation {
        Text(validation).foregroundStyle(.secondary).textSelection(.enabled)
      }
    }
  }
  private var options: some View {
    HStack(spacing: 14) {
      HStack {
        Text("Ensemble")
        TextField("8", value: $model.estimators, format: .number.grouping(.never))
          .frame(width: 50).textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel(
            "Ensemble size")
      }
      HStack {
        Text("Seed")
        TextField("0", value: $model.seed, format: .number.grouping(.never))
          .frame(width: 85).textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel(
            "Random seed")
      }
    }.disabled(model.busy)
  }
  private var runButton: some View {
    HStack(spacing: 8) {
      if model.busy {
        ProgressView().controlSize(.small)
        Button("Cancel") { model.cancel() }.buttonStyle(FlatButtonStyle())
      } else {
        Button("Predict", systemImage: "play.fill") { model.run() }
          .keyboardShortcut(.return, modifiers: .command)
          .buttonStyle(FlatButtonStyle(primary: true)).disabled(!model.canRun)
          .accessibilityIdentifier("tables-predict")
      }
    }
  }
  private func results(_ result: NativeTablesModel.Result) -> some View {
    EndpointFormCard("Predictions") {
      if let runs = model.history.document?.runs, !runs.isEmpty {
        HStack {
          Dropdown(
            title: "Prediction run",
            value:
              "Run \((runs.firstIndex { $0.id == model.selectedRun } ?? 0) + 1) of \(runs.count)"
          ) {
            ForEach(Array(runs.enumerated()), id: \.element.id) { i, run in
              Button(
                "Run \(i + 1) · \(Date(timeIntervalSince1970: Double(run.at) / 1000).formatted(date: .abbreviated, time: .shortened))"
              ) {
                Task { await model.selectRun(run) }
              }
            }
          }.disabled(model.busy || model.restoring)
          Spacer()
          if let run = runs.first(where: { $0.id == model.selectedRun }) {
            Button("Use these inputs") { Task { await model.restoreRun(run) } }
              .buttonStyle(FlatButtonStyle()).disabled(model.historyNavigationBlocked)
          }
        }
      }
      if model.stale {
        Text(
          "These results belong to the previous table or settings. Export keeps that original table."
        )
        .font(.caption).foregroundStyle(.secondary)
      }
      NativeTabularGrid(
        version: result.id, header: result.header, rows: result.rows,
        columnWidths: result.header.indices.map { $0 == 0 ? 50 : 115 }
      )
      .frame(height: min(360, CGFloat(result.rows.count + 1) * 26 + 16))
      .clipShape(RoundedRectangle(cornerRadius: 8)).accessibilityLabel("Predictions")
      ViewThatFits(in: .horizontal) {
        HStack {
          usage(result)
          Spacer()
          exportButtons(result)
        }
        VStack(alignment: .leading, spacing: 12) {
          usage(result)
          exportButtons(result)
        }
      }
      if result.task == .regression {
        Text("p10–p90 is the model’s predicted 80% interval, not a measured accuracy guarantee.")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
  }
  private func usage(_ result: NativeTablesModel.Result) -> some View {
    Text(
      "\(result.milliseconds.formatted(.number.precision(.fractionLength(0)))) ms"
        + (result.response.gpuMilliseconds.map {
          " · GPU \($0.formatted(.number.precision(.fractionLength(0)))) ms"
        } ?? "")
        + (result.response.estimators.map { " · \($0) estimators" } ?? "")
    )
    .font(.caption).monospacedDigit().foregroundStyle(.secondary)
  }
  private func exportButtons(_ result: NativeTablesModel.Result) -> some View {
    HStack {
      Button("Copy table") { copy(result.csv) }
      Button("Export CSV…") {
        exportDocument = TabularCSVDocument(text: result.csv)
        exporting = true
      }
    }.buttonStyle(FlatButtonStyle())
  }
  private func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
  }
}

private struct TabularCSVDocument: FileDocument {
  static let readableContentTypes: [UTType] = [.commaSeparatedText]
  let text: String
  init(text: String) { self.text = text }
  init(configuration: ReadConfiguration) throws {
    text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
  }
  func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
    FileWrapper(regularFileWithContents: Data(text.utf8))
  }
}

/// Reuses visible AppKit cells; no truncation of the inference dataset. A
/// result/version update alone reloads data, so polling does not reset scroll
/// position, selected text or user-resized columns.
struct NativeTabularGrid: NSViewRepresentable {
  let version: UUID
  let header: [String]
  let rows: [[String]]
  var columnWidths: [CGFloat] = []
  func makeCoordinator() -> Coordinator { Coordinator() }
  func makeNSView(context: Context) -> NSScrollView {
    let scroll = NSScrollView()
    let table = NSTableView()
    table.delegate = context.coordinator
    table.dataSource = context.coordinator
    table.rowHeight = 25
    table.usesAlternatingRowBackgroundColors = true
    table.columnAutoresizingStyle = .noColumnAutoresizing
    table.allowsMultipleSelection = true
    scroll.documentView = table
    scroll.hasVerticalScroller = true
    scroll.hasHorizontalScroller = true
    scroll.autohidesScrollers = true
    PaddockScrollbars.install(on: scroll)
    return scroll
  }
  func updateNSView(_ scroll: NSScrollView, context: Context) {
    let coordinator = context.coordinator
    guard coordinator.version != version, let table = scroll.documentView as? NSTableView else {
      return
    }
    coordinator.version = version
    coordinator.rows = rows
    if coordinator.header != header {
      coordinator.header = header
      table.tableColumns.forEach(table.removeTableColumn)
      for (i, title) in header.enumerated() {
        let column = NSTableColumn(identifier: .init(String(i)))
        column.title = String(title.prefix(512))
        column.width = columnWidths.indices.contains(i) ? columnWidths[i] : 160
        column.minWidth = 50
        column.maxWidth = 800
        table.addTableColumn(column)
      }
    }
    table.reloadData()
  }
  final class Coordinator: NSObject, NSTableViewDataSource, NSTableViewDelegate {
    var version: UUID?
    var header: [String] = []
    var rows: [[String]] = []
    func numberOfRows(in tableView: NSTableView) -> Int { rows.count }
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int)
      -> NSView?
    {
      guard let id = tableColumn?.identifier, let column = Int(id.rawValue),
        rows.indices.contains(row), rows[row].indices.contains(column)
      else { return nil }
      let field =
        tableView.makeView(withIdentifier: id, owner: nil) as? NSTextField
        ?? NSTextField(labelWithString: "")
      field.identifier = id
      field.isSelectable = true
      field.lineBreakMode = .byTruncatingTail
      field.font = .systemFont(ofSize: 12)
      field.stringValue = String(rows[row][column].prefix(4096))
      field.toolTip = field.stringValue
      return field
    }
  }
}
