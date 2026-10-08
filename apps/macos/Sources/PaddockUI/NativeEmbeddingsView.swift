import AppKit
import PaddockConversationCore
import SwiftUI
import UniformTypeIdentifiers

struct NativeEmbeddingsView: View {
  @Bindable var model: NativeEmbeddingsModel
  var onStart: () -> Void
  @State private var importing = false
  @State private var exporting = false
  @State private var exportDocument = EmbeddingsJSONDocument(data: Data())
  private var title: String { model.current?.reranker == true ? "Rerank" : "Embeddings" }
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 20) {
        ViewThatFits(in: .horizontal) {
          HStack {
            heading
            Spacer()
            picker.frame(width: 340)
          }
          VStack(alignment: .leading, spacing: 12) {
            heading
            picker
          }
        }
        if let error = model.error ?? model.discoveryError {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
            .accessibilityIdentifier("embeddings-error")
        }
        if model.endpoints.isEmpty {
          ContentUnavailableView {
            Label(
              "Start an embedding or reranking model",
              systemImage: "point.3.connected.trianglepath.dotted")
          } actions: {
            Button("Choose model", action: onStart).buttonStyle(FlatButtonStyle())
          }
        } else {
          input
          HStack {
            Button(model.current?.reranker == true ? "Rerank" : "Embed") { model.run() }
              .buttonStyle(FlatButtonStyle()).disabled(!model.canRun)
              .keyboardShortcut(.return, modifiers: .command).accessibilityIdentifier(
                "embeddings-run")
            if model.busy || model.importing {
              ProgressView().controlSize(.small)
              Button("Cancel") { model.cancel() }.buttonStyle(QuietButtonStyle())
            }
            Spacer()
            Text(model.current?.reranker == true ? "/v1/rerank" : "/v1/embeddings")
              .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
          }
        }
        if let output = model.result { results(output) }
      }.padding(24).frame(maxWidth: 1200).frame(maxWidth: .infinity)
    }.background(PaddockStyle.canvas).font(.system(size: 13))
      .accessibilityIdentifier("native-embeddings")
      .task {
        repeat {
          await model.refresh()
          do { try await Task.sleep(for: .seconds(5)) } catch { return }
        } while !Task.isCancelled
      }
      .fileImporter(
        isPresented: $importing, allowedContentTypes: allowedTypes, allowsMultipleSelection: true
      ) { result in
        switch result {
        case .success(let urls): model.importFiles(urls)
        case .failure(let error): model.report(error)
        }
      }
      .fileExporter(
        isPresented: $exporting, document: exportDocument, contentType: .json,
        defaultFilename: "embeddings.json"
      ) { result in
        if case .failure(let error) = result { model.report(error) }
      }
  }
  private var allowedTypes: [UTType] {
    (model.current?.inputs.contains("image") == true ? [.image] : [])
      + (model.current?.inputs.contains("audio") == true ? [.audio] : [])
      + (model.current?.inputs.contains("video") == true ? [.movie] : [])
  }
  private var heading: some View {
    Text(title).font(.system(size: 23, weight: .semibold)).tracking(-0.5).accessibilityAddTraits(
      .isHeader)
  }
  private var picker: some View {
    Dropdown(
      title: "Embedding model", value: model.current?.title ?? "Choose model", fillsWidth: true,
      vendor: model.current?.vendor
    ) {
      ForEach(model.endpoints) { endpoint in
        Button {
          model.port = endpoint.port
        } label: {
          ModelProviderMenuLabel(
            title: endpoint.title + " · \(endpoint.port)", vendor: endpoint.vendor)
        }
      }
    }.disabled(model.busy || model.importing || model.endpoints.isEmpty)
      .accessibilityIdentifier("embeddings-model-picker")
  }
  private var input: some View {
    EndpointFormCard(model.current?.reranker == true ? "Query and documents" : "Compare meaning") {
      if model.current?.reranker == true {
        TextField("Query", text: $model.query).textFieldStyle(.roundedBorder).accessibilityLabel(
          "Reranking query")
      }
      Text(model.current?.reranker == true ? "One document per line" : "One text per line")
        .font(.caption).foregroundStyle(.secondary)
      TextEditor(text: $model.text).font(.system(size: 13))
        .frame(minHeight: 140, maxHeight: 240).scrollContentBackground(.hidden)
        .padding(8).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
        .accessibilityLabel("Embedding texts").accessibilityIdentifier("embeddings-input")
      if model.current?.reranker != true {
        ViewThatFits(in: .horizontal) {
          HStack(spacing: 12) { controls }
          VStack(alignment: .leading, spacing: 12) { controls }
        }
        ForEach(model.media) { medium in
          HStack(spacing: 10) {
            Image(
              systemName: medium.kind == "video"
                ? "film" : (medium.kind == "image" ? "photo" : "waveform")
            )
            .foregroundStyle(.secondary).frame(width: 20)
            if medium.kind == "video" {
              Text("\(medium.frames.count) frames · visual only").foregroundStyle(.secondary)
            }
            Text(medium.name).lineLimit(1).truncationMode(.middle).help(medium.name)
            Spacer()
            Text(
              ByteCountFormatter.string(fromByteCount: Int64(medium.byteCount), countStyle: .file)
            )
            .font(.caption).foregroundStyle(.secondary)
            Button {
              model.remove(medium.id)
            } label: {
              Image(systemName: "xmark")
            }
            .buttonStyle(QuietButtonStyle()).accessibilityLabel("Remove \(medium.name)")
          }.padding(8).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
        }
      }
      if model.lines.count + model.media.count > 32 {
        Text("Use at most 32 items per comparison.").foregroundStyle(PaddockStyle.caution)
      }
    }.disabled(model.busy || model.importing)
      .dropDestination(for: URL.self) { urls, _ in
        guard !model.busy, !model.importing, model.current?.reranker != true,
          !urls.isEmpty, urls.allSatisfy(\.isFileURL), !allowedTypes.isEmpty
        else { return false }
        model.importFiles(urls)
        return true
      }
  }
  @ViewBuilder private var controls: some View {
    if !allowedTypes.isEmpty {
      Button {
        importing = true
      } label: {
        Label("Attach", systemImage: "paperclip")
      }.buttonStyle(FlatButtonStyle())
    }
    if let current = model.current, !current.tasks.isEmpty {
      Dropdown(
        title: "Task",
        value: model.task.isEmpty
          ? "No task prefix" : model.task.replacingOccurrences(of: "_", with: " ")
      ) {
        Button("No task prefix") { model.task = "" }
        ForEach(current.tasks, id: \.self) { task in
          Button(task.replacingOccurrences(of: "_", with: " ")) { model.task = task }
        }
      }
    }
    if let current = model.current, !current.dimensions.isEmpty {
      Dropdown(
        title: "Dimensions",
        value: model.dimensions == 0 ? "768 dimensions" : "\(model.dimensions) dimensions"
      ) {
        ForEach(current.dimensions, id: \.self) { dim in
          Button("\(dim) dimensions") { model.dimensions = dim }
        }
      }
    }
  }
  private func results(_ output: NativeEmbeddingsModel.Output) -> some View {
    EndpointFormCard(output.endpoint.reranker ? "Ranking" : "Similarity") {
      HStack {
        Text(output.endpoint.title).fontWeight(.medium)
        Spacer()
        Button("Export JSON") {
          exportDocument = EmbeddingsJSONDocument(data: output.json)
          exporting = true
        }.buttonStyle(FlatButtonStyle())
      }
      if output.endpoint.reranker {
        ForEach(output.rankings.indices, id: \.self) { row in
          HStack(alignment: .top) {
            Text("\(row+1)").foregroundStyle(.secondary).frame(width: 24)
            Text(output.labels[output.rankings[row].index]).textSelection(.enabled).frame(
              maxWidth: .infinity, alignment: .leading)
            Text(output.rankings[row].score.formatted(.number.precision(.fractionLength(4))))
              .monospacedDigit()
          }.padding(.vertical, 6)
        }
      } else {
        if output.labels.count > 1 {
          ScrollView(.horizontal) {
            Grid(horizontalSpacing: 8, verticalSpacing: 8) {
              GridRow {
                Text("")
                ForEach(output.labels.indices, id: \.self) {
                  Text("\($0+1)").foregroundStyle(.secondary)
                }
              }
              ForEach(output.labels.indices, id: \.self) { i in
                GridRow {
                  Text("\(i+1)").foregroundStyle(.secondary)
                  ForEach(output.labels.indices, id: \.self) { j in
                    Text(output.similarity[i][j].formatted(.number.precision(.fractionLength(3))))
                      .monospacedDigit().frame(width: 62, height: 28)
                      .background(PaddockStyle.elevated, in: RoundedRectangle(cornerRadius: 6))
                      .help("\(output.labels[i]) / \(output.labels[j])")
                  }
                }
              }
            }
          }
        }
        ForEach(output.labels.indices, id: \.self) { i in
          HStack(alignment: .top) {
            Text("\(i+1)").foregroundStyle(.secondary).frame(width: 24)
            Text(output.labels[i]).textSelection(.enabled).frame(
              maxWidth: .infinity, alignment: .leading)
            Button("Copy vector") {
              let values = output.vectors[i].map { String($0) }.joined(separator: ",")
              NSPasteboard.general.clearContents()
              NSPasteboard.general.setString("[\(values)]", forType: .string)
            }.buttonStyle(QuietButtonStyle())
          }.padding(.vertical, 4)
        }
      }
      Text(
        "\(output.labels.count) items · \(Int(output.milliseconds)) ms"
          + (output.vectors.first.map { " · \($0.count) dimensions" } ?? "")
          + (output.tokens.map { " · \($0) tokens" } ?? "")
      )
      .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
    }.accessibilityIdentifier("embeddings-result")
  }
}

private struct EmbeddingsJSONDocument: FileDocument {
  static var readableContentTypes: [UTType] { [.json] }
  var data: Data
  init(data: Data) { self.data = data }
  init(configuration: ReadConfiguration) throws {
    data = configuration.file.regularFileContents ?? Data()
  }
  func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
    FileWrapper(regularFileWithContents: data)
  }
}
