import SwiftUI

struct EndpointResidencySettings: View {
  @Bindable var editor: EndpointEditor
  var body: some View {
    if editor.residencySupported {
      EndpointFormCard("Model loading") {
        SettingsRow(title: "Load model") {
          Dropdown(
            title: "Load model",
            value: editor.loadOnDemand ? "On first request" : "At runner startup", fillsWidth: true
          ) {
            Button("At runner startup") { editor.loadOnDemand = false }
            Button("On first request") { editor.loadOnDemand = true }
          }
        }
        .settingsSearchAnchor(.modelLoading)
        SettingsRow(title: "Unload after inactivity") {
          HStack(spacing: 8) {
            TextField("Never", text: $editor.unloadIdleSeconds)
              .textFieldStyle(StudioPopoverFieldStyle())
              .accessibilityIdentifier("residency-idle-seconds")
            Text("seconds").foregroundStyle(.secondary)
          }.help(
            "Leave empty to keep the model loaded. Zero unloads after all requests and realtime sessions finish. The API stays available."
          )
        }
        .settingsSearchAnchor(.idleUnload)
        if editor.advanced {
          SettingsRow(title: "Load wait limit") {
            HStack(spacing: 8) {
              TextField("120", text: $editor.loadWaitSeconds).textFieldStyle(
                StudioPopoverFieldStyle()
              )
              .accessibilityIdentifier("residency-load-wait")
              Text("seconds").foregroundStyle(.secondary)
            }
          }.settingsSearchAnchor(.loadWait)
        }
      }.accessibilityIdentifier("endpoint-residency-settings")
    }
  }
}
