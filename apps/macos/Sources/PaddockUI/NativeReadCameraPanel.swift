import PaddockConversationCore
import SwiftUI

struct NativeReadCameraPanel: View {
  @Bindable var model: NativeReadsModel
  @Bindable var camera: NativeReadCamera
  var close: () -> Void
  @State private var contextOpen = false
  @State private var pendingSet: NativeReadsModel.SavedSet?

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      ZStack(alignment: .bottom) {
        Color.black
        if let capture = camera.capture {
          NativeReadCameraPreview(capture: capture, camera: camera)
        }
        if camera.opening { ProgressView().tint(.white).frame(maxHeight: .infinity) }
        ScrollView(.horizontal) {
          HStack(alignment: .bottom, spacing: 8) {
            ForEach(camera.latest?.questions ?? model.draft.questions) { question in
              let answer = camera.latest?.response.answers[question.questionID]
              VStack(alignment: .leading, spacing: 5) {
                Text(question.instructions).font(.caption).lineLimit(2)
                HStack {
                  Text(answer?.label ?? "-").font(.title3.weight(.semibold))
                  Spacer(minLength: 8)
                  if let answer {
                    Text(answer.confidence, format: .percent.precision(.fractionLength(0)))
                      .monospacedDigit()
                  }
                }
                ProgressView(value: answer?.confidence ?? 0).tint(.white)
              }
              .foregroundStyle(.white).padding(12).frame(width: 220)
              .background(answerColor(answer), in: RoundedRectangle(cornerRadius: 10))
              .animation(.easeInOut(duration: 0.2), value: answer?.label)
            }
          }.padding(12)
        }.scrollIndicators(.hidden).fixedSize(horizontal: false, vertical: true)
      }
      .aspectRatio(16 / 9, contentMode: .fit)
      .clipShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
      .accessibilityIdentifier("reads-camera-preview")

      ViewThatFits(in: .horizontal) {
        HStack(spacing: 10) {
          actions
          Spacer()
          measurements
          frameOptions
        }
        VStack(alignment: .leading, spacing: 10) {
          HStack {
            actions
            Spacer()
            frameOptions
          }
          HStack { measurements }
        }
      }
      if model.runs.contains(where: \.isCameraFrame) {
        ScrollView(.horizontal) {
          HStack(spacing: 8) {
            ForEach(model.runs.reversed().filter(\.isCameraFrame)) { run in
              if let picture = run.pictures.first {
                Button {
                  model.selectedRun = run.id
                  close()
                } label: {
                  NativeReadPictureChip(picture: picture).frame(maxWidth: 180)
                }
                .buttonStyle(.plain).accessibilityLabel("Open saved camera frame")
              }
            }
          }
        }.scrollIndicators(.hidden)
      }
      configuration
      if let result = camera.latest {
        NativeReadResult(result: result)
          .accessibilityIdentifier("reads-camera-answers")
      }
    }.accessibilityIdentifier("reads-camera")
      .confirmationDialog(
        "Replace unsaved questions?",
        isPresented: Binding(
          get: { pendingSet != nil }, set: { if !$0 { pendingSet = nil } }),
        titleVisibility: .visible
      ) {
        if let set = pendingSet {
          Button("Open \(set.name)") {
            camera.pause(clear: true)
            model.open(set)
            pendingSet = nil
          }
        }
        Button("Cancel", role: .cancel) { pendingSet = nil }
      }
  }

  private var actions: some View {
    HStack(spacing: 10) {
      Button(
        camera.live ? "Stop" : "Start", systemImage: camera.live ? "pause.fill" : "play.fill"
      ) {
        if camera.live { camera.pause() } else { camera.start(model: model) }
      }.buttonStyle(FlatButtonStyle(primary: !camera.live))
        .disabled(
          camera.opening || camera.capture == nil || (!camera.live && !model.canReadCamera)
        )
        .accessibilityIdentifier("reads-camera-start")
      Button("Snapshot", systemImage: "camera") {
        if let frame = camera.latest { Task { await model.keepCameraFrame(frame) } }
      }.buttonStyle(FlatButtonStyle()).disabled(
        camera.latest == nil || model.historyNavigationBlocked
      )
      .accessibilityIdentifier("reads-camera-snapshot")
    }.fixedSize()
  }
  private var measurements: some View {
    HStack(spacing: 10) {
      if let result = camera.latest {
        Text("\(Int(result.elapsedMilliseconds)) ms").monospacedDigit().foregroundStyle(
          .secondary)
        if camera.rate > 0 {
          Text("\(camera.rate, format: .number.precision(.fractionLength(1)))/s")
        }
      }
    }.fixedSize()
  }
  private var frameOptions: some View {
    HStack(spacing: 10) {
      Dropdown(title: "Frame", value: "\(camera.side) px") {
        ForEach([384, 512, 768, 1024], id: \.self) { side in
          Button("\(side) px") { camera.side = side }
        }
      }.fixedSize()
      Button(action: close) { Image(systemName: "xmark") }
        .buttonStyle(QuietButtonStyle()).accessibilityLabel("Close the camera")
    }
  }
  private var configuration: some View {
    VStack(alignment: .leading, spacing: 14) {
      if camera.devices.count > 1 {
        Dropdown(
          title: "Camera",
          value: camera.devices.first { $0.id == camera.deviceID }?.name ?? "Default camera"
        ) {
          ForEach(camera.devices, id: \.id) { device in
            Button(device.name) {
              camera.deviceID = device.id
              camera.open(resume: model)
            }
          }
        }
      }
      if let error = camera.error {
        HStack {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
          Button("Try again") { camera.open() }.buttonStyle(FlatButtonStyle())
        }
      }
      if let error = model.validation { Text(error).foregroundStyle(PaddockStyle.caution) }
      HStack {
        Text("Questions").font(.headline)
        Text("\(model.draft.questions.count) of \(model.current?.maxQuestions ?? 64)")
          .foregroundStyle(.secondary).monospacedDigit()
        Spacer()
        Dropdown(title: "Question set", value: model.selectedSet?.name ?? "Unsaved") {
          ForEach(model.sets) { set in
            Button(set.name) {
              if model.dirty {
                pendingSet = set
              } else {
                camera.pause(clear: true)
                model.open(set)
              }
            }
          }
        }.disabled(model.historyNavigationBlocked || model.sets.isEmpty)
          .accessibilityIdentifier("reads-camera-question-set")
      }
      NativeReadSamples(model: model)
      NativeReadAdvancedOptions(model: model)
      ForEach($model.draft.questions) { $question in
        VStack(alignment: .leading, spacing: 6) {
          HStack {
            Dropdown(title: "Type", value: question.kind.title) {
              ForEach(
                ReadQuestion.Kind.allCases.filter {
                  model.current?.types.contains($0.rawValue) != false
                }, id: \.self
              ) { kind in
                Button(kind.title) { question.kind = kind }
              }
            }.frame(width: 110)
            TextField(
              "Ask something about what the camera sees",
              text: Binding(
                get: { question.instructions },
                set: { model.editInstructions(question.id, text: $0) }
              )
            )
            .textFieldStyle(StudioPopoverFieldStyle())
            Button {
              model.draft.removeQuestion(question.id)
            } label: {
              Image(systemName: "xmark")
            }
            .buttonStyle(QuietButtonStyle()).disabled(model.draft.questions.count == 1)
            .accessibilityLabel("Remove question")
          }
          if question.kind != .noul {
            TextField(
              "Answers, separated by commas",
              text: Binding(
                get: {
                  (question.kind == .choice ? question.options : question.levels).map(\.name)
                    .joined(separator: ", ")
                },
                set: { text in
                  let names = text.components(separatedBy: ",").map {
                    $0.trimmingCharacters(in: .whitespaces)
                  }
                  let old = question.kind == .choice ? question.options : question.levels
                  let next = names.map { name in
                    old.first { $0.name == name } ?? ReadQuestion.Option(name: name)
                  }
                  if question.kind == .choice {
                    question.options = next
                  } else {
                    question.levels = next
                  }
                }
              )
            ).textFieldStyle(StudioPopoverFieldStyle())
          }
        }
      }
      HStack {
        ForEach(
          ReadQuestion.Kind.allCases.filter { model.current?.types.contains($0.rawValue) != false },
          id: \.self
        ) { kind in
          Button(kind.title, systemImage: "plus") { model.add(kind) }.buttonStyle(FlatButtonStyle())
        }
      }.disabled(model.draft.questions.count >= (model.current?.maxQuestions ?? 64))
      DisclosureGroup("Context text", isExpanded: $contextOpen) {
        PaddockTextEditor(text: $model.draft.state).frame(height: 90)
      }
    }
  }
  private func answerColor(_ answer: ReadResponse.Answer?) -> Color {
    guard let value = answer?.noul else { return Color.black.opacity(0.75) }
    return (value >= 0.5 ? Color.green : Color.red).opacity(0.75)
  }
}
