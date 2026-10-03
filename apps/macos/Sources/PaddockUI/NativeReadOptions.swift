import SwiftUI

/// Shared by the text and camera editors: there is only one set of request options.
struct NativeReadAdvancedOptions: View {
  @Bindable var model: NativeReadsModel
  var body: some View {
    HStack(spacing: 16) {
      if model.current?.backend == "laya" || model.draft.checkpoint != nil {
        Dropdown(title: "Checkpoint", value: model.draft.checkpoint ?? "Automatic language") {
          Button("Automatic language") { model.draft.checkpoint = nil }
          ForEach(model.current?.checkpoints ?? [], id: \.self) { checkpoint in
            Button(checkpoint) { model.draft.checkpoint = checkpoint }
          }
        }.accessibilityIdentifier("reads-checkpoint")
      }
      if let max = model.current?.maxSteps, max > 1 || model.draft.steps > max {
        Text("Steps").font(.system(size: 12)).foregroundStyle(.secondary)
        Dropdown(title: "Steps", value: "\(model.draft.steps)") {
          ForEach(1...max, id: \.self) { n in Button("\(n)") { model.draft.steps = n } }
        }.accessibilityIdentifier("reads-steps")
      }
      if model.current?.think == true || model.draft.think > 0 {
        Text("Thought").font(.system(size: 12)).foregroundStyle(.secondary)
        Dropdown(
          title: "Thought", value: model.draft.think == 0 ? "None" : "\(model.draft.think) tokens"
        ) {
          ForEach(
            model.current?.think == true ? [0, 128, 256, 512, 1024, 2048, 4096] : [0], id: \.self
          ) { n in
            Button(n == 0 ? "None" : "\(n) tokens") { model.draft.think = n }
          }
        }.accessibilityIdentifier("reads-thought")
      }
    }
  }
}

struct NativeReadSamples: View {
  @Bindable var model: NativeReadsModel
  @ViewBuilder var body: some View {
    if model.current?.maxSamples != 1 || model.draft.samples > 1 {
      HStack(spacing: 8) {
        Text("Reads per question").font(.system(size: 12)).foregroundStyle(.secondary).fixedSize()
        Dropdown(
          title: "Reads per question",
          value: model.draft.samples == 0 ? "Auto" : "\(model.draft.samples)"
        ) {
          Button("Auto") { model.draft.samples = 0 }
          ForEach(
            [1, 2, 4, 8, 16, 32].filter { $0 <= model.current?.maxSamples ?? 32 }, id: \.self
          ) {
            count in
            Button("\(count)") { model.draft.samples = count }
          }
        }.fixedSize()
      }.fixedSize(horizontal: true, vertical: false)
    }
  }
}
