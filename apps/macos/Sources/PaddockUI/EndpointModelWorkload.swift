import PaddockClient
import SwiftUI

struct EndpointModelWorkload: View {
  @Bindable var editor: EndpointEditor
  var onChangeModel: (() -> Void)? = nil
  var onDownload: ((String) -> Void)? = nil
  var body: some View {
    EndpointFormField("Model") {
      if editor.isCreating, let onChangeModel {
        HStack(spacing: 10) {
          ModelProviderLogo(vendor: editor.selectedModel?.vendor)
          Text(editor.title).fontWeight(.medium)
          Spacer()
          Button("Change", action: onChangeModel).buttonStyle(QuietButtonStyle())
        }
      } else {
        Dropdown(
          title: "Model", value: editor.title, fillsWidth: true,
          vendor: editor.selectedModel?.vendor
        ) {
          ForEach(editor.modelChoices) { model in
            Button {
              editor.selectModel(model)
            } label: {
              ModelProviderMenuLabel(title: model.display, vendor: model.vendor)
            }
          }
        }.disabled(editor.modelChoices.isEmpty).accessibilityIdentifier("endpoint-model")
      }
    }
    if !editor.weights.isEmpty {
      EndpointFormField("Quality") {
        LazyVGrid(
          columns: Array(
            repeating: GridItem(.flexible(minimum: 0), spacing: 8),
            count: min(3, editor.weights.count)), spacing: 8
        ) {
          ForEach(editor.weights) { artifact in
            Button {
              if artifact.installed {
                editor.selectArtifact(artifact)
              } else {
                onDownload?(artifact.id)
              }
            } label: {
              VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline) {
                  Text(artifact.displayLabel).font(.system(size: 12, weight: .medium))
                  Spacer(minLength: 4)
                  Image(
                    systemName: editor.artifactID == artifact.id
                      ? "checkmark.circle.fill" : "circle"
                  )
                  .font(.system(size: 12)).foregroundStyle(.secondary)
                }
                Text(DisplayFormat.bytes(artifact.totalSize)).font(
                  .system(size: 16, weight: .medium))
                Text(artifact.quant ?? artifact.format).font(.system(size: 10)).foregroundStyle(
                  .secondary)
                Text(
                  artifact.installed
                    ? "Downloaded" : onDownload == nil ? "Download in Catalog" : "Download…"
                )
                .font(.system(size: 11)).foregroundStyle(.secondary)
              }.padding(12).frame(maxWidth: .infinity, minHeight: 88, alignment: .topLeading)
                .background(
                  editor.artifactID == artifact.id ? PaddockStyle.elevated : PaddockStyle.canvas,
                  in: RoundedRectangle(cornerRadius: 8)
                )
                .overlay(
                  RoundedRectangle(cornerRadius: 8).strokeBorder(
                    editor.artifactID == artifact.id ? Color.primary.opacity(0.45) : .clear)
                )
                .contentShape(RoundedRectangle(cornerRadius: 8))
            }.buttonStyle(.plain).disabled(!artifact.installed && onDownload == nil)
              .accessibilityAddTraits(editor.artifactID == artifact.id ? .isSelected : [])
              .accessibilityIdentifier("endpoint-quality-\(artifact.id)")
          }
        }
      }
    }
    if editor.embeddedVision {
      Label("Vision included", systemImage: "photo")
        .font(.system(size: 12)).foregroundStyle(.secondary)
    } else if let vision = editor.visionArtifact {
      Toggle("Vision · image input", isOn: $editor.vision).toggleStyle(.switch).controlSize(.small)
        .disabled(vision.required == true || !vision.installed)
      if !vision.installed {
        if let onDownload {
          Button("Download vision", systemImage: "arrow.down") { onDownload(editor.artifactID) }
            .buttonStyle(QuietButtonStyle()).accessibilityIdentifier("endpoint-download-vision")
        } else {
          hint("Download the vision companion in Catalog to enable image input.")
        }
      }
    }
    if !editor.isImageGeneration && !editor.capabilities.contains("decision")
      && !editor.capabilities.contains("tabular")
      && !editor.capabilities.contains("diarization")
    {
      EndpointFormField("Workload") {
        LazyVGrid(
          columns: Array(repeating: GridItem(.flexible(minimum: 0), spacing: 6), count: 4),
          spacing: 6
        ) {
          ForEach(EndpointEditor.workloads, id: \.batch) { workload in
            choice(
              workload.label, subtitle: "\(workload.batch) at once",
              selected: !editor.customWorkload && editor.effectiveConcurrency == workload.batch
            ) {
              editor.customWorkload = false
              editor.concurrency = String(workload.batch)
            }.disabled(workload.batch > editor.concurrencyCap)
          }
          choice("Custom", selected: editor.customWorkload) { editor.customWorkload = true }
        }
        if editor.customWorkload {
          HStack {
            TextField("1", text: $editor.concurrency).textFieldStyle(StudioPopoverFieldStyle())
              .frame(width: 110).accessibilityLabel("Concurrent requests")
            Stepper(
              "Concurrent requests",
              value: Binding(
                get: { editor.effectiveConcurrency }, set: { editor.concurrency = String($0) }),
              in: 1...max(1, editor.concurrencyCap)
            ).labelsHidden()
          }
        }
      }
      EndpointFormField(
        editor.capabilities.contains("asr") ? "Decoder context" : "Context per conversation"
      ) {
        Dropdown(
          title: "Context per conversation",
          value: editor.context.isEmpty
            ? "Runner default · \(contextLabel(editor.effectiveContext))"
            : editor.customContext ? "Custom" : contextLabel(editor.effectiveContext),
          fillsWidth: true
        ) {
          Button("Model default · \(contextLabel(editor.recommendedContext))") {
            editor.context = String(editor.recommendedContext)
            editor.customContext = false
          }
          ForEach(editor.contextOptions, id: \.self) { tokens in
            Button(contextLabel(tokens)) {
              editor.context = String(tokens)
              editor.customContext = false
            }
          }
          Button("Custom…") { editor.customContext = true }
        }.accessibilityIdentifier("endpoint-context")
        if editor.customContext {
          TextField("Model default", text: $editor.context).textFieldStyle(
            StudioPopoverFieldStyle()
          )
          .frame(width: 160).accessibilityLabel("Context tokens")
        }
      }
      EndpointFormField("Conversation memory") {
        Text(editor.kvLabel).font(.system(size: 12))
        if !editor.kvChoices.contains(editor.kvDtype) {
          Button("Use supported memory format") {
            editor.kvDtype = editor.selectedArtifact?.runtime?.kvCacheDtype ?? "auto"
          }
          .buttonStyle(FlatButtonStyle())
          hint("Saved format \(editor.kvDtype) cannot run with these weights.")
        }
      }
    }
    if editor.canSpeculate || (editor.selectedModel == nil && editor.endpoint.settings?.spec != nil)
    {
      EndpointFormField("Speculative") {
        Dropdown(
          title: "Speculative",
          value: EndpointEditor.speculationChoices.first { $0.0 == editor.speculation }?.1
            ?? "Fixed draft depth: \(editor.speculation)", fillsWidth: true
        ) {
          ForEach(EndpointEditor.speculationChoices, id: \.0) { value, label in
            Button(label) { editor.speculation = value }
          }
        }.accessibilityIdentifier("endpoint-speculative")
          .help(editor.speculationSummary)
      }
      if editor.speculation != "off", editor.drafters.count > 1 {
        EndpointFormField("Drafter") {
          Dropdown(
            title: "Drafter",
            value: editor.drafters.first { $0.id == editor.drafter }?.displayLabel ?? "Automatic",
            fillsWidth: true
          ) {
            Button("Automatic") { editor.drafter = "" }
            ForEach(editor.drafters) { drafter in
              Button(drafter.displayLabel + (drafter.installed ? "" : " · not downloaded")) {
                editor.drafter = drafter.id
              }
              .disabled(!drafter.installed)
            }
          }
        }
      }
    }
  }
  private func contextLabel(_ tokens: Int) -> String {
    tokens % 1024 == 0 ? "\(tokens / 1024)K tokens" : "\(tokens.formatted()) tokens"
  }
  private func choice(
    _ title: String, subtitle: String? = nil, selected: Bool, action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      VStack(alignment: .leading, spacing: 5) {
        Text(title).font(.system(size: 11, weight: .medium))
        if let subtitle { Text(subtitle).font(.system(size: 10)).foregroundStyle(.secondary) }
      }.padding(11).frame(maxWidth: .infinity, minHeight: 57, alignment: .leading)
        .background(
          selected ? PaddockStyle.elevated : PaddockStyle.canvas,
          in: RoundedRectangle(cornerRadius: 7)
        )
        .overlay(
          RoundedRectangle(cornerRadius: 7).strokeBorder(
            selected ? Color.primary.opacity(0.45) : .clear)
        )
        .contentShape(RoundedRectangle(cornerRadius: 7))
    }.buttonStyle(.plain).accessibilityAddTraits(selected ? .isSelected : [])
  }
  private func hint(_ text: String) -> some View {
    Text(text).font(.system(size: 11)).foregroundStyle(.secondary).fixedSize(
      horizontal: false, vertical: true)
  }
}
