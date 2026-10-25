import PaddockClient
import SwiftUI

struct EndpointAdvancedSettings: View {
  @Bindable var editor: EndpointEditor
  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      ForEach(editor.runtimeGroups, id: \.self) { group in
        EndpointFormCard(group) {
          ForEach(editor.visibleRuntimeOptions.filter { $0.group == group }) { field in
            SettingsRow(title: field.label) {
              VStack(alignment: .leading, spacing: 5) {
                if field.kind == "boolean" {
                  Dropdown(title: field.label, value: boolLabel(field), fillsWidth: true) {
                    Button("Default · \(field.placeholder)") { editor.runtimeDraft[field.id] = "" }
                    Button("On") { editor.runtimeDraft[field.id] = "true" }
                    Button("Off") { editor.runtimeDraft[field.id] = "false" }
                  }
                } else {
                  TextField(
                    field.placeholder,
                    text: Binding(
                      get: { editor.runtimeDraft[field.id] ?? "" },
                      set: { editor.runtimeDraft[field.id] = $0 })
                  )
                  .textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel(field.label)
                  if field.kind == "number" || field.kind == "integer" {
                    Text("\(field.minimum.formatted())-\(field.maximum.formatted())")
                      .font(.system(size: 10)).foregroundStyle(.secondary)
                  }
                }
                if !(editor.runtimeDraft[field.id] ?? "").isEmpty {
                  Button("Use default") { editor.runtimeDraft[field.id] = "" }
                    .buttonStyle(QuietButtonStyle()).font(.system(size: 10))
                }
              }
              .accessibilityIdentifier("endpoint-option-\(field.id)")
              .help(field.help)
            }.settingsSearchTarget("runtime:\(field.id)")
          }
        }
      }
    }.accessibilityIdentifier("endpoint-advanced-settings")
  }
  private func boolLabel(_ field: EndpointRuntimeOption) -> String {
    switch editor.runtimeDraft[field.id] {
    case "true": "On"
    case "false": "Off"
    default: "Default · \(field.placeholder)"
    }
  }
}
