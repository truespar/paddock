import PaddockClient
import SwiftUI

/// System tools are properties of this endpoint, never global Studio settings.
struct EndpointToolsView: View {
  @Bindable var model: IntegrationsModel
  let port: UInt16
  let title: String
  var backLabel = "Model settings"
  let onBack: () -> Void
  @State private var confirmLeave = false
  private var dirty: Bool { model.searchEditor?.dirty == true || !model.endpointChanges.isEmpty }
  var body: some View {
    ScrollViewReader { proxy in
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 22) {
          Button(backLabel, systemImage: "chevron.left") {
            if dirty { confirmLeave = true } else { onBack() }
          }.buttonStyle(QuietButtonStyle())
          VStack(alignment: .leading, spacing: 6) {
            Text("System tools").font(.system(size: 24, weight: .semibold))
            Text(title).foregroundStyle(.secondary)
          }
          if let editor = model.searchEditor, editor.settings.port == port {
            EndpointFormCard("Web search") { WebSearchForm(editor: editor) }.settingsSearchAnchor(
              .webSearch)
            EndpointMCPSection(model: model, port: port)
            HStack {
              Button(model.endpointSaving ? "Saving…" : "Save changes") {
                Task { await model.saveEndpoint(port) }
              }
              .buttonStyle(FlatButtonStyle(primary: true))
              .disabled(!dirty || model.saving || editor.validation != nil)
              .accessibilityIdentifier("endpoint-tools-save")
              Button("Discard changes") { model.discardEndpoint() }.buttonStyle(QuietButtonStyle())
                .disabled(!dirty || model.saving)
              Spacer()
              if dirty {
                Text("Unsaved changes").font(.system(size: 11)).foregroundStyle(.secondary)
              }
            }
          } else if let editor = model.searchEditor, dirty {
            Text(
              "Finish the unsaved changes for port \(editor.settings.port) before editing another endpoint."
            )
            Button("Return to port \(editor.settings.port)") {
              model.onConfigureSearch?(editor.settings.port)
            }
            .buttonStyle(FlatButtonStyle())
          } else if model.error == nil {
            ProgressView("Loading system tools…").controlSize(.small)
          }
          if let error = model.error {
            Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
            Button("Retry") {
              Task {
                await model.refresh()
                await model.loadSearch(port)
              }
            }.buttonStyle(QuietButtonStyle())
          }
          if let message = model.message { Text(message).foregroundStyle(.secondary) }
        }.font(.system(size: 13)).padding(28).frame(maxWidth: 850, alignment: .leading)
          .frame(maxWidth: .infinity, alignment: .top)
      }.background(PaddockStyle.canvas).accessibilityIdentifier("endpoint-tools-page")
        .modifier(SettingsSearchScroll(proxy: proxy))
        .task(id: port) {
          await model.refresh()
          await model.loadSearch(port)
        }
        .confirmationDialog(
          "Keep these changes?", isPresented: $confirmLeave, titleVisibility: .visible
        ) {
          Button("Keep editing", role: .cancel) {}
          Button("Discard changes", role: .destructive) {
            model.discardEndpoint()
            onBack()
          }
        } message: {
          Text("The system-tool changes for this endpoint have not been saved.")
        }
    }
  }
}

struct WebSearchForm: View {
  @Bindable var editor: WebSearchEditor
  // Same wire IDs and orientation text as studio/src/lib/websearch.ts.
  static let providers = [
    ("", "Off"), ("exa", "Exa"), ("tavily", "Tavily"), ("firecrawl", "Firecrawl"),
    ("brave", "Brave"), ("perplexity", "Perplexity"),
  ]
  static let details: [String: (String, String)] = [
    "exa": (
      "Semantic search - matches on meaning rather than keywords.", "https://dashboard.exa.ai"
    ),
    "tavily": (
      "Built for models: returns the relevant chunks of each page.", "https://app.tavily.com"
    ),
    "firecrawl": (
      "Fetches every result and returns the full page as markdown.",
      "https://firecrawl.dev/app/api-keys"
    ),
    "brave": (
      "An independent index - a genuinely different set of results.",
      "https://api-dashboard.search.brave.com"
    ),
    "perplexity": (
      "The index behind the answer engine, ranked for questions.",
      "https://www.perplexity.ai/account/api/keys"
    ),
  ]
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      providerChoices
      if let (_, keyURL) = Self.details[editor.provider] {
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            Text("\(editor.providerLabel) API key").fontWeight(.medium)
            Spacer()
            if let url = URL(string: keyURL) {
              Link("Get an API key ↗", destination: url).foregroundStyle(.secondary)
            }
          }
          SecureField(editor.keepsSavedKey ? "******" : "Enter API key", text: $editor.key)
            .textFieldStyle(StudioPopoverFieldStyle())
            .accessibilityIdentifier("web-search-api-key")
            .accessibilityLabel("\(editor.providerLabel) API key")
            .accessibilityHint(
              editor.keepsSavedKey
                ? "A key is saved. Enter a new key to replace it." : "Enter an API key.")
          if let validation = editor.validation {
            Text(validation).font(.system(size: 11)).foregroundStyle(.secondary)
          }
        }
      } else if !editor.provider.isEmpty {
        Text(
          "Unknown saved provider: \(editor.provider). Choose a supported provider to replace it."
        )
        .foregroundStyle(PaddockStyle.caution)
      }
      if let error = editor.error {
        Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
      }
    }.font(.system(size: 12)).disabled(editor.saving)
  }
  private var providerChoices: some View {
    SettingsChoiceLayout {
      ForEach(Self.providers, id: \.0) { provider in
        Button {
          editor.provider = provider.0
        } label: {
          HStack(spacing: 6) {
            if let mark = SearchProviderArtwork(rawValue: provider.0) {
              SearchProviderLogo(provider: mark, size: 15)
            }
            Text(provider.1)
          }
        }.buttonStyle(SearchProviderChoiceStyle(selected: editor.provider == provider.0))
          .help(Self.details[provider.0]?.0 ?? "Disable web search")
          .fixedSize()
          .accessibilityIdentifier("web-search-provider-\(provider.0.isEmpty ? "off" : provider.0)")
          .accessibilityLabel(provider.1).accessibilityValue(
            editor.provider == provider.0 ? "Selected" : "Not selected")
      }
    }.frame(maxWidth: .infinity, alignment: .leading)
  }
}

/// ServerForm.vue's sf__pill: selection changes the outline and subtle fill,
/// not the foreground/background polarity beneath the provider's brand mark.
struct SearchProviderChoiceStyle: ButtonStyle {
  var selected: Bool
  @Environment(\.isEnabled) private var enabled
  @State private var hovered = false

  func makeBody(configuration: Configuration) -> some View {
    configuration.label.font(.system(size: 12, weight: selected ? .semibold : .regular))
      .foregroundStyle(.primary).padding(.horizontal, 11).padding(.vertical, 5)
      .background(
        selected ? Color(nsColor: PaddockStyle.nsColor("selection")) : PaddockStyle.surface,
        in: Capsule()
      )
      .overlay(
        Capsule().strokeBorder(selected || hovered ? PaddockStyle.accent : PaddockStyle.border)
      )
      .contentShape(Capsule())
      .opacity(enabled ? (configuration.isPressed ? 0.7 : 1) : 0.4)
      .onHover { hovered = $0 }
  }
}
