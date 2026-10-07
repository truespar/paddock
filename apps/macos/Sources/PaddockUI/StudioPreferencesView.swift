import PaddockStudio
import SwiftUI

struct StudioPreferencesView: View {
  @Bindable var model: StudioPreferencesModel
  let busy: Bool
  var chat: StudioWorkspace?
  @State private var microphoneOptions = false
  @State private var confirmReload = false
  var body: some View {
    SettingsPage(title: "Conversation") { stacked in
      if let error = model.error {
        Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
      }
      if let notice = model.notice { Text(notice).foregroundStyle(.secondary) }
      if model.loading { ProgressView().controlSize(.small) }
      if model.loaded, let layout = model.layout {
        VStack(alignment: .leading, spacing: 22) {
          ForEach(layout.sections) { section in
            settingRow(section, layout: layout, stacked: stacked)
              .accessibilityElement(children: .contain)
              .accessibilityIdentifier("studio-setting-\(section.id)")
            if section.id != layout.sections.last?.id { WorkspaceRule() }
          }
        }.disabled(model.saving || model.loading)
        if let validation = model.validation, validation != model.reply.validation {
          Text(validation).foregroundStyle(PaddockStyle.caution)
        }
      }
      HStack {
        Button("Reload saved settings") {
          if model.dirty { confirmReload = true } else { Task { await model.load() } }
        }.disabled(model.saving || model.loading)
        Spacer()
        Button(model.saving ? "Saving…" : "Apply") { model.save() }
          .buttonStyle(FlatButtonStyle(primary: true)).disabled(
            !model.dirty || model.validation != nil || model.saving || model.loading || busy)
      }.buttonStyle(FlatButtonStyle())
      if busy {
        Text("Finish the response before applying preferences.").font(.caption).foregroundStyle(
          .secondary)
      }
    }.task { await model.load() }
      .confirmationDialog(
        "Discard your settings draft and reload?", isPresented: $confirmReload,
        titleVisibility: .visible
      ) {
        Button("Discard and reload", role: .destructive) {
          Task { await model.load(discard: true) }
        }
        Button("Keep editing", role: .cancel) {}
      }
  }
  private func settingRow(
    _ section: StudioSettingsLayout.Section, layout: StudioSettingsLayout, stacked: Bool
  ) -> some View {
    SettingsRow(title: section.title, stacked: stacked) {
      setting(section, layout: layout)
    }
  }
  @ViewBuilder private func setting(
    _ section: StudioSettingsLayout.Section, layout: StudioSettingsLayout
  ) -> some View {
    switch section.id {
    case "maxTokens":
      StudioReplyLimitControl(draft: $model.reply)
    case "maxToolCalls":
      Dropdown(
        title: section.title,
        value: layout.toolStops.first { $0.value == (Int(model.toolLimit) ?? 0) }?.label
          ?? "\(model.toolLimit) tool calls", fillsWidth: true
      ) {
        ForEach(layout.toolStops) { stop in
          Button(stop.label) { model.toolLimit = stop.value == 0 ? "" : String(stop.value) }
        }
      }.help("Maximum tool calls per reply.")
    case "summarize":
      Toggle(section.title, isOn: $model.summarize)
        .labelsHidden().toggleStyle(.switch).controlSize(.small)
        .frame(height: 30)
        .help(
          "Summarize older messages when context fills. When off, the oldest messages are dropped."
        )
    case "microphone":
      VStack(alignment: .leading, spacing: 10) {
        if let chat {
          StudioAudioInputSettings(chat: chat, inset: 0)
          Button("Audio settings…") { microphoneOptions = true }
            .buttonStyle(FlatButtonStyle())
            .popover(isPresented: $microphoneOptions) {
              StudioMicrophoneSettings(chat: chat).studioPopoverSurface()
            }
        }
      }
    case "mapTiles":
      VStack(alignment: .leading, spacing: 10) {
        TextField("Follow the theme", text: $model.mapTiles)
          .textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel(section.title)
        explanation("Interactive maps share the photo's location with this tile server.")
      }
    default:
      // New shared sections must not silently disappear before a native adapter lands.
      Text("\(section.title) is not yet available in the native app.").foregroundStyle(.secondary)
    }
  }
  private func explanation(_ text: String) -> some View {
    Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(
      horizontal: false, vertical: true)
  }
}
