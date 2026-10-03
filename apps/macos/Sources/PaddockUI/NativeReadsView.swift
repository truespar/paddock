import AppKit
import PaddockConversationCore
import SwiftUI
import UniformTypeIdentifiers

struct NativeReadsView: View {
  @Bindable var model: NativeReadsModel
  var showsHistorySidebar = false
  var onStart: () -> Void
  @State private var jsonMode = false
  @State private var showFilePicker = false
  @State private var importKind = NativeReadImport.state
  @State private var confirmNew = false
  @State private var confirmDelete = false
  @State private var confirmExample = false
  @State private var pendingSet: NativeReadsModel.SavedSet?
  @State private var confirmDeleteRead = false
  @State private var cameraMode = false
  @State private var camera = NativeReadCamera()
  var body: some View {
    GeometryReader { geometry in
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 18) {
          header(stacked: geometry.size.width < (model.current?.images == true ? 800 : 600))
          if let error = model.error {
            Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
              .accessibilityIdentifier("reads-error")
          }
          if model.readers.isEmpty {
            VStack(spacing: 16) {
              if model.loading {
                ProgressView().controlSize(.small)
              } else {
                Image(systemName: "list.bullet.clipboard").font(.system(size: 28)).foregroundStyle(
                  .secondary)
                Text("No model that can read is running").font(.headline)
                Button("Start a model", action: onStart).buttonStyle(FlatButtonStyle(primary: true))
              }
            }.frame(maxWidth: .infinity).padding(.vertical, 60)
          }
          if cameraMode {
            NativeReadCameraPanel(model: model, camera: camera) { cameraMode = false }
          } else if geometry.size.width >= 1100 {
            HStack(alignment: .top, spacing: 20) {
              editor.frame(maxWidth: .infinity)
              answers.frame(maxWidth: .infinity)
            }
          } else {
            editor
            answers
          }
        }.padding(24).frame(maxWidth: 1400).frame(maxWidth: .infinity)
      }
    }.background(PaddockStyle.canvas).font(.system(size: 13))
      .task {
        repeat {
          await model.refresh()
          do { try await Task.sleep(for: .seconds(5)) } catch { return }
        } while !Task.isCancelled
      }
      // A single presentation owner: stacked fileImporter modifiers on this
      // view left only the last (images) picker reachable on macOS.
      .fileImporter(
        isPresented: $showFilePicker, allowedContentTypes: importKind.contentTypes,
        allowsMultipleSelection: importKind.allowsMultipleSelection
      ) { result in
        let kind = importKind
        Task { await model.importSelection(result, kind: kind) }
      }
      .confirmationDialog("Start a new read?", isPresented: $confirmNew, titleVisibility: .visible)
    {
      Button("Discard current draft", role: .destructive) { model.reset() }
      Button("Cancel", role: .cancel) {}
    }
      .confirmationDialog(
        "Delete this question set?", isPresented: $confirmDelete, titleVisibility: .visible
      ) {
        Button("Delete set", role: .destructive) { Task { await model.remove() } }
        Button("Cancel", role: .cancel) {}
      }
      .confirmationDialog(
        "Replace unsaved questions?",
        isPresented: Binding(
          get: { pendingSet != nil }, set: { if !$0 { pendingSet = nil } }),
        titleVisibility: .visible
      ) {
        if let set = pendingSet {
          Button("Open \(set.name)", role: .destructive) {
            model.open(set)
            pendingSet = nil
          }
        }
        Button("Cancel", role: .cancel) { pendingSet = nil }
      }
      .accessibilityIdentifier("native-reads")
      .onChange(of: cameraMode) { _, active in if active { camera.open() } else { camera.close() } }
      .onChange(of: model.port) { _, _ in
        camera.pause(clear: true)
        if model.current?.images != true { cameraMode = false }
      }
      .onChange(of: model.current?.images) { _, supported in
        if supported != true { cameraMode = false }
      }
      .onChange(of: model.activeSession?.id) { old, _ in if old != nil { camera.pause(clear: true) }
      }
      .onDisappear { camera.close() }
      .onAppear { if cameraMode { camera.open() } }
      .confirmationDialog(
        "Delete this read and its runs?", isPresented: $confirmDeleteRead, titleVisibility: .visible
      ) {
        Button("Delete read", role: .destructive) { Task { await model.clearHistory() } }
        Button("Cancel", role: .cancel) {}
      }
      .confirmationDialog(
        "Replace the draft with the example?", isPresented: $confirmExample,
        titleVisibility: .visible
      ) {
        Button("Run example") { model.runExample() }
        Button("Cancel", role: .cancel) {}
      }
  }

  private func header(stacked: Bool) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack(spacing: 16) {
        Text("Reads").font(.system(size: 25, weight: .semibold)).tracking(-0.5)
          .fixedSize().accessibilityAddTraits(.isHeader)
        if !stacked, model.current?.images == true { modePicker }
        Spacer(minLength: 8)
        if !showsHistorySidebar {
          Button("New read", systemImage: "plus") {
            if model.unsavedRead { confirmNew = true } else { model.reset() }
          }
          .buttonStyle(FlatButtonStyle()).fixedSize()
          .disabled(model.historyNavigationBlocked)
        }
        if !stacked, !model.readers.isEmpty { modelPicker.frame(width: 270) }
      }
      if stacked, model.current?.images == true { modePicker }
      if stacked, !model.readers.isEmpty { modelPicker }
      if model.openingSession || model.historyUnsaved {
        HStack {
          if model.openingSession { ProgressView().controlSize(.small) }
          Spacer()
          if model.historyUnsaved {
            Button("Retry saving") { Task { await model.saveHistory() } }
              .buttonStyle(FlatButtonStyle()).disabled(model.busy || model.saving)
          }
        }
      }
    }.accessibilityIdentifier("reads-header")
  }

  private var modePicker: some View {
    Picker("What to read", selection: $cameraMode) {
      Text("Text").tag(false)
      Text("Camera").tag(true)
    }.pickerStyle(.segmented).frame(width: 160).accessibilityIdentifier("reads-mode")
  }

  private var modelPicker: some View {
    Dropdown(
      title: "Reading model", value: model.current?.title ?? "Choose model",
      fillsWidth: true, vendor: model.current?.vendor
    ) {
      ForEach(model.readers) { reader in
        Button {
          model.port = reader.port
        } label: {
          ModelProviderMenuLabel(title: reader.title, vendor: reader.vendor)
        }
      }
    }.accessibilityIdentifier("reads-model-picker")
  }

  private var editor: some View {
    VStack(alignment: .leading, spacing: 16) {
      card("State") {
        NativeReadStateEditor(text: $model.draft.state, onPasteImages: model.pastePictures).frame(
          height: 160
        )
        .onChange(of: model.draft.state) { _, _ in model.stateError = nil }
        .clipShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
        .accessibilityLabel("Text to read")
        .dropDestination(for: URL.self) { urls, _ in
          guard let url = urls.first, url.isFileURL, !model.importing, !model.busy else {
            return false
          }
          if urls.allSatisfy({
            UTType(filenameExtension: $0.pathExtension)?.conforms(to: .image) == true
          }) {
            Task { await model.addPictures(urls) }
          } else {
            Task { await model.loadFile(url) }
          }
          return true
        }
        HStack(spacing: 8) {
          Button("Load a file", systemImage: "paperclip") { chooseFile(.state) }
            .buttonStyle(FlatButtonStyle()).fixedSize().disabled(model.historyNavigationBlocked)
            .accessibilityIdentifier("reads-load-file")
          Button("Try an example", systemImage: "play") {
            if model.hasWork { confirmExample = true } else { model.runExample() }
          }.buttonStyle(FlatButtonStyle()).fixedSize()
            .accessibilityIdentifier("reads-example")
            .disabled(model.current == nil || model.historyNavigationBlocked)
          Spacer()
          if !model.draft.state.isEmpty {
            Button("Clear") {
              model.draft.state = ""
              model.fileName = ""
            }.buttonStyle(QuietButtonStyle())
          }
        }.accessibilityIdentifier("reads-input-actions")
        if model.current?.images == true {
          Button("Add images", systemImage: "photo.badge.plus") { chooseFile(.images) }
            .buttonStyle(FlatButtonStyle()).disabled(
              model.historyNavigationBlocked || model.draft.images.count >= 16
            )
            .accessibilityIdentifier("reads-add-images")
        }
        if !model.draft.images.isEmpty {
          LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 180), alignment: .leading)], alignment: .leading
          ) {
            ForEach(Array(model.draft.images.enumerated()), id: \.offset) { index, picture in
              NativeReadPictureChip(picture: picture) { model.draft.images.remove(at: index) }
                .disabled(model.historyNavigationBlocked)
            }
          }.accessibilityIdentifier("reads-images")
        }
        if model.importing || !model.fileName.isEmpty {
          HStack(spacing: 8) {
            if model.importing { ProgressView().controlSize(.small) }
            Text(model.fileName).lineLimit(1).truncationMode(.middle)
              .foregroundStyle(.secondary).help(model.fileName)
          }
        }
        if let error = model.stateError {
          Text(error).font(.caption).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
        }
      }
      card(
        "Questions", detail: "\(model.draft.questions.count) / \(model.current?.maxQuestions ?? 64)"
      ) {
        HStack(spacing: 8) {
          TextField("Set name", text: $model.setName).textFieldStyle(StudioPopoverFieldStyle())
            .accessibilityLabel("Question set name")
          Button("Save") { Task { await model.save() } }.buttonStyle(FlatButtonStyle()).fixedSize()
            .disabled(
              model.saving || model.busy || model.validation != nil || model.setName.isEmpty)
          Menu {
            ForEach(model.sets) { set in
              Button(set.name) { if model.dirty { pendingSet = set } else { model.open(set) } }
            }
            Divider()
            Button("Save as new set") { Task { await model.save(asNew: true) } }
              .disabled(model.validation != nil || model.setName.isEmpty)
            Button("Import JSON…") { chooseFile(.questions) }
              .accessibilityIdentifier("reads-import-json")
            Button("Export JSON…") {
              exportText((try? model.draft.orderedJSON()) ?? "", name: "read-questions.json")
            }
            if model.selectedSet != nil {
              Button("Delete set…", role: .destructive) { confirmDelete = true }
            }
          } label: {
            Image(systemName: "ellipsis").frame(width: 28, height: 28)
          }
          .menuStyle(.button).menuIndicator(.hidden).buttonStyle(QuietButtonStyle()).fixedSize()
          .accessibilityLabel("Question set actions").disabled(model.historyNavigationBlocked)
        }
        editorControls
        if jsonMode {
          PaddockTextEditor(text: $model.jsonText).frame(height: 280).accessibilityLabel(
            "Questions JSON")
          Button("Apply JSON") { model.applyJSON(model.jsonText) }.buttonStyle(FlatButtonStyle())
        } else {
          ForEach($model.draft.questions) { $question in
            NativeReadQuestionRow(
              question: $question, onDuplicate: { model.duplicate(question.id) },
              onMove: { model.move(question.id, by: $0) },
              onRemove: { model.draft.questions.removeAll { $0.id == question.id } },
              position: model.draft.questions.firstIndex { $0.id == question.id } ?? 0,
              count: model.draft.questions.count,
              supportedTypes: model.current?.types ?? ["noul", "choice", "score"],
              serverError: model.rowErrors[question.questionID],
              onID: { model.editID(question.id, text: $0) },
              onInstructions: { model.editInstructions(question.id, text: $0) })
          }
          HStack(spacing: 8) {
            ForEach(
              ReadQuestion.Kind.allCases.filter {
                model.current?.types.contains($0.rawValue) != false
              }, id: \.self
            ) { kind in
              Button(kind.title, systemImage: "plus") { model.add(kind) }.buttonStyle(
                FlatButtonStyle()
              )
              .fixedSize()
            }
          }.disabled(model.draft.questions.count >= model.current?.maxQuestions ?? 64)
        }
        if let error = model.validation {
          Text(error).font(.caption).foregroundStyle(PaddockStyle.caution)
        }
        if let error = model.questionsError {
          Text(error).font(.caption).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
        }
        HStack {
          Button("Run", systemImage: "play.fill") { model.run(applyJSON: jsonMode) }.buttonStyle(
            FlatButtonStyle(primary: true)
          )
          .disabled(
            jsonMode
              ? model.current == nil || model.busy || model.importing
                || (model.draft.state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                  && model.draft.images.isEmpty)
              : !model.canRun
          ).keyboardShortcut(.return, modifiers: .command)
          if model.busy {
            ProgressView().controlSize(.small)
            Button("Cancel") { model.cancel() }.buttonStyle(QuietButtonStyle())
          }
          Spacer()
        }
      }
      DisclosureGroup("API request") {
        let request =
          (try? model.draft.orderedJSON(model: model.requestModel, includeImageData: false))
          ?? ""
        Text("POST /v1/systemone\n\n" + request).font(.system(size: 11, design: .monospaced))
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
        Button("Copy request") {
          let draft = model.draft
          let name = model.requestModel
          Task {
            if let full = await Task.detached(
              priority: .userInitiated, operation: { try? draft.orderedJSON(model: name) }
            ).value {
              copy(full)
            }
          }
        }.buttonStyle(QuietButtonStyle())
      }
    }
  }

  private var editorControls: some View {
    VStack(alignment: .leading, spacing: 12) {
      ViewThatFits(in: .horizontal) {
        HStack(spacing: 16) {
          editorTabs
          Spacer(minLength: 0)
          NativeReadSamples(model: model)
        }
        VStack(alignment: .leading, spacing: 12) {
          editorTabs
          NativeReadSamples(model: model)
        }
      }
      NativeReadAdvancedOptions(model: model)
    }
    .onChange(of: jsonMode) { _, json in
      if json {
        model.beginJSON()
      } else if model.hasUnappliedJSON {
        if !model.applyJSON(model.jsonText) { jsonMode = true }
      }
    }
    .accessibilityIdentifier("reads-editor-controls")
  }

  private var editorTabs: some View {
    Picker("Question editor", selection: $jsonMode) {
      Text("Form").tag(false)
      Text("JSON").tag(true)
    }.pickerStyle(.segmented).labelsHidden().frame(width: 140)
      .accessibilityIdentifier("reads-editor-tabs")
  }

  private var answers: some View {
    card("Answers") {
      if let error = model.historyError {
        Text(error).font(.caption).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
      }
      if let result = model.result {
        HStack {
          Text(
            "\(result.response.diagnostics.timing.totalMilliseconds, specifier: "%.0f") ms · \(result.response.diagnostics.reads) reads"
          )
          .monospacedDigit().foregroundStyle(.secondary)
          Spacer()
          Dropdown(title: "Read history", value: "History") {
            ForEach(model.runs) { run in
              Button(run.at.formatted(date: .abbreviated, time: .standard) + " · " + run.excerpt) {
                model.selectedRun = run.id
              }
            }
            Divider()
            Button("Delete read", role: .destructive) { confirmDeleteRead = true }
              .disabled(model.busy)
          }.fixedSize()
        }
        if result.state == nil {
          Text("The original input was not retained with this older result.")
            .font(.caption).foregroundStyle(.secondary)
        } else if model.stale {
          Text("Edited since this read").font(.caption).foregroundStyle(PaddockStyle.caution)
        } else if model.previousRead {
          Text("Previous read · " + result.excerpt).font(.caption).foregroundStyle(.secondary)
        }
        if !result.pictures.isEmpty {
          LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 180), alignment: .leading)], alignment: .leading
          ) {
            ForEach(Array(result.pictures.enumerated()), id: \.offset) { _, picture in
              NativeReadPictureChip(picture: picture)
            }
          }
        }
        ForEach(result.questions) { question in
          if let answer = result.response.answers[question.questionID] {
            NativeReadAnswerView(
              question: question, answer: answer,
              diagnostic: result.response.diagnostics.questions.first {
                $0.id == question.questionID
              }, readCount: result.response.diagnostics.reads)
          }
        }
        NativeReadDiagnostics(
          response: result.response, elapsedMilliseconds: result.elapsedMilliseconds)
        DisclosureGroup("Response JSON") {
          Text((try? ReadDraft.json(result.raw)) ?? "").font(.system(size: 11, design: .monospaced))
            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
        Button("Export results…") { export(result.raw, name: "read-results.json") }.buttonStyle(
          FlatButtonStyle())
      } else {
        Text(model.busy ? "Reading…" : "Run a read to see answers.").foregroundStyle(.secondary)
          .padding(.vertical, 24)
      }
    }
  }
  private func card<Content: View>(
    _ title: String, detail: String? = nil, @ViewBuilder content: () -> Content
  )
    -> some View
  {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text(title).font(.system(size: 13, weight: .semibold)).accessibilityAddTraits(.isHeader)
        Spacer()
        if let detail { Text(detail).font(.caption).foregroundStyle(.secondary).monospacedDigit() }
      }
      content()
    }
    .padding(18).frame(maxWidth: .infinity, alignment: .leading)
    .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
  }
  private func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
  }
  private func chooseFile(_ kind: NativeReadImport) {
    guard !showFilePicker, !model.historyNavigationBlocked else { return }
    importKind = kind
    showFilePicker = true
  }
  private func export(_ value: ConversationValue, name: String) {
    do { exportText(try ReadDraft.json(value), name: name) } catch {
      model.error = error.localizedDescription
    }
  }
  private func exportText(_ text: String, name: String) {
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.json]
    panel.nameFieldStringValue = name
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      do { try text.write(to: url, atomically: true, encoding: .utf8) } catch {
        model.error = error.localizedDescription
      }
    }
  }
}
