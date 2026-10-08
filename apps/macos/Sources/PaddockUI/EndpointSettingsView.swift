import PaddockClient
import SwiftUI

/// One draft and one acknowledged save, presented as Simple/Advanced lenses.
/// ServerForm.vue supplies the information order; Metal supplies the controls.
struct EndpointSettingsView: View {
  @Bindable var editor: EndpointEditor
  let canMutate: Bool
  let onTools: () -> Void
  var onCreate: ((Bool) -> Void)? = nil
  var onChangeModel: (() -> Void)? = nil
  var onDownload: ((String) -> Void)? = nil
  @State private var confirmation: Confirmation?
  private enum Confirmation { case restart, networkSave }
  private var disabled: Bool {
    !canMutate || editor.saving || editor.refreshing || editor.needsReload
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 22) {
      if editor.endpoint.settings != nil {
        EndpointProfilesView(editor: editor)
        if editor.pid != nil {
          if editor.restartRequired {
            VStack(alignment: .leading, spacing: 6) {
              Text("Saved changes awaiting restart").fontWeight(.semibold)
              Text((editor.runtimeState?.changed ?? []).joined(separator: " · "))
              if let live = editor.runtimeState {
                Text(
                  "Running: \(live.maxCtx.formatted()) context tokens · \(live.maxBatch) request(s) at once"
                )
              }
              Button("Restart to apply saved settings") { confirmation = .restart }
                .buttonStyle(FlatButtonStyle()).disabled(disabled || editor.dirty)
                .accessibilityIdentifier("endpoint-apply-saved")
            }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
              .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
          } else if editor.runtimeState?.restartRequired != false {
            issue("Running settings could not be verified.")
          }
        }
        HStack {
          Picker("Settings view", selection: $editor.advanced) {
            Text("Simple").tag(false)
            Text("Advanced").tag(true)
          }.pickerStyle(.segmented).labelsHidden().frame(width: 210).disabled(editor.saving)
            .accessibilityIdentifier("endpoint-settings-mode")
          Spacer()
          Text("Metal · \(editor.checkpointLabel)")
            .font(.system(size: 11)).foregroundStyle(.secondary)
        }
        if editor.advanced {
          EndpointAdvancedSettings(editor: editor).disabled(disabled)
          EndpointMemorySettings(editor: editor).disabled(disabled)
          EndpointResidencySettings(editor: editor).disabled(disabled)
          EndpointKVOffloadSettings(editor: editor).disabled(disabled)
          access(advanced: true).disabled(disabled)
        } else {
          simple
        }
        status
        HStack(spacing: 12) {
          Button(editor.isCreating ? "Start Model" : (editor.saving ? "Saving…" : "Save")) {
            if editor.isCreating {
              if editor.localOnly { onCreate?(false) } else { confirmation = .networkSave }
            } else if editor.pid != nil && editor.onlyResidencyChanges
              && editor.runtimeState?.residencyLive == true && !editor.restartRequired
            {
              editor.save(.restart, networkConfirmed: !editor.localOnly)
            } else if editor.pid != nil {
              confirmation = .restart
            } else if editor.localOnly {
              editor.save(.defer)
            } else {
              confirmation = .networkSave
            }
          }.buttonStyle(FlatButtonStyle(primary: true))
            .disabled(disabled || (!editor.isCreating && !editor.dirty) || editor.validation != nil)
            .accessibilityIdentifier(editor.isCreating ? "start-model-confirm" : "endpoint-save")
          if !editor.isCreating {
            Button("Discard changes") { editor.reset() }.buttonStyle(QuietButtonStyle())
              .disabled(!editor.dirty || editor.saving)
            if editor.dirty {
              Text("Unsaved changes").font(.system(size: 11)).foregroundStyle(.secondary)
            }
          }
          Spacer()
        }.padding(.vertical, 6)
      } else {
        issue(editor.endpoint.configError ?? "Saved configuration unavailable.")
        Button("Retry loading settings") {
          Task {
            if editor.isCreating {
              await editor.prepareCreationSelection()
            } else {
              await editor.reload()
            }
          }
        }.buttonStyle(FlatButtonStyle())
      }
    }
    .confirmationDialog(
      confirmation == .restart ? "Restart \(editor.title) to apply?" : "Allow network access?",
      isPresented: Binding(get: { confirmation != nil }, set: { if !$0 { confirmation = nil } }),
      titleVisibility: .visible
    ) {
      if confirmation == .restart {
        if editor.dirty {
          Button("Save for later") { editor.save(.defer, networkConfirmed: !editor.localOnly) }
        }
        Button(editor.dirty ? "Save & restart" : "Restart to apply") {
          editor.save(.restart, networkConfirmed: !editor.localOnly)
        }
      } else {
        Button(editor.isCreating ? "Start with network access" : "Save network configuration") {
          if editor.isCreating {
            onCreate?(true)
          } else {
            editor.save(.defer, networkConfirmed: true)
          }
        }
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text(
        confirmation == .restart
          ? "Save for later keeps the current model running. Restarting drains requests and reloads on the same port."
            + (editor.localOnly ? "" : " Other devices will be able to connect using its API key.")
          : (editor.isCreating
            ? "Other devices will be able to connect using this model's API key."
            : "Other devices may reach this model on its next start. They must supply its API key. No model starts now.")
      )
    }
  }
  private var simple: some View {
    VStack(alignment: .leading, spacing: 20) {
      EndpointFormCard("Model & workload") {
        EndpointModelWorkload(editor: editor, onChangeModel: onChangeModel, onDownload: onDownload)
      }.disabled(disabled)
      EndpointMemorySettings(editor: editor).disabled(disabled)
      EndpointResidencySettings(editor: editor).disabled(disabled)
      EndpointKVOffloadSettings(editor: editor).disabled(disabled)
      if editor.capabilities.contains("chat") || editor.forensicsPossible {
        EndpointFormCard("Document & image intelligence") {
          if editor.forensicsPossible {
            Toggle(isOn: $editor.forensics) {
              VStack(alignment: .leading, spacing: 5) {
                Text("Forensics").fontWeight(.medium)
                if !editor.visionServed { hint("Enable image input to use forensics.") }
              }
            }.toggleStyle(.switch).controlSize(.small).disabled(!editor.visionServed)
              .help("Checks images for signs of tampering.")
          }
          HStack {
            Text("File metadata").fontWeight(.medium)
            Spacer()
            Text("Always on").foregroundStyle(.secondary)
          }
        }.disabled(disabled)
      }
      if editor.canTools {
        EndpointFormCard("System tools") {
          Button(
            "Configure web search & MCP servers…", systemImage: "puzzlepiece.extension",
            action: onTools
          )
          .buttonStyle(FlatButtonStyle()).disabled(editor.saving)
        }
      }
      access(advanced: false).disabled(disabled)
    }
  }
  private func access(advanced: Bool) -> some View {
    EndpointFormCard("Access & policy") {
      if advanced {
        EndpointFormField("Port") {
          if editor.isCreating {
            Toggle("Choose port automatically", isOn: $editor.automaticPort)
              .toggleStyle(.checkbox).accessibilityIdentifier("start-port-automatic")
            if !editor.automaticPort {
              TextField("11540", text: $editor.newPort).textFieldStyle(StudioPopoverFieldStyle())
                .frame(width: 140).accessibilityIdentifier("endpoint-port")
            }
          } else {
            Text(String(editor.endpoint.port)).monospacedDigit().textSelection(.enabled)
          }
        }
        EndpointFormField("Listen on") {
          Dropdown(
            title: "Listen on", value: editor.localOnly ? "Only this Mac" : editor.host,
            fillsWidth: true
          ) {
            Button("Only this Mac") { editor.host = "127.0.0.1" }
            Button("Local network · all IPv4 interfaces") { editor.host = "0.0.0.0" }
          }.accessibilityIdentifier("endpoint-listen")
        }
      } else {
        HStack {
          Text("Available to")
          Spacer()
          Text(editor.localOnly ? "Only this Mac" : "Network clients").foregroundStyle(.secondary)
        }
      }
      EndpointFormField("API key") {
        HStack(spacing: 8) {
          SecureField(
            editor.endpoint.settings?.hasApiKey == true ? "******" : "Optional for local access",
            text: $editor.replacementKey
          )
          .textFieldStyle(StudioPopoverFieldStyle()).accessibilityIdentifier("endpoint-api-key")
          .accessibilityLabel("API key")
          .accessibilityHint(
            editor.endpoint.settings?.hasApiKey == true
              ? "A key is saved. Enter a new key to replace it." : "Optional for local access.")
          Button("Generate") { editor.generateKey() }.buttonStyle(FlatButtonStyle())
        }
      }
    }
  }
  @ViewBuilder private var status: some View {
    if let validation = editor.validation, editor.isCreating || editor.dirty { issue(validation) }
    if let error = editor.error { issue(error) }
    if let message = editor.message { hint(message) }
    if editor.needsReload {
      Button("Retry loading settings") {
        Task {
          if editor.isCreating {
            await editor.prepareCreationSelection()
          } else {
            await editor.reload()
          }
        }
      }.buttonStyle(FlatButtonStyle())
    }
    if let pending = editor.pending {
      HStack {
        if editor.checking { ProgressView().controlSize(.small) }
        Text(pending.message)
        if !editor.checking {
          Button("Retry status") { editor.retryStatus() }.buttonStyle(FlatButtonStyle())
        }
      }
    }
  }
  private func hint(_ text: String) -> some View { EndpointHint(text: text) }
  private func issue(_ text: String) -> some View {
    Text(text).foregroundStyle(PaddockStyle.caution).fixedSize(horizontal: false, vertical: true)
      .textSelection(.enabled)
  }
}

struct EndpointFormCard<Content: View>: View {
  let title: String
  @ViewBuilder let content: Content
  init(_ title: String, @ViewBuilder content: () -> Content) {
    self.title = title
    self.content = content()
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      Text(title).font(.system(size: 13, weight: .semibold))
      content
    }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
      .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
  }
}
struct EndpointFormField<Content: View>: View {
  let title: String
  @ViewBuilder let content: Content
  init(_ title: String, @ViewBuilder content: () -> Content) {
    self.title = title
    self.content = content()
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(title).font(.system(size: 12, weight: .medium))
      content
    }.frame(maxWidth: .infinity, alignment: .leading)
  }
}
struct EndpointHint: View {
  let text: String
  var body: some View {
    Text(text).font(.system(size: 11)).foregroundStyle(.secondary).fixedSize(
      horizontal: false, vertical: true)
  }
}
