import AppKit
import PaddockStudio
import SwiftUI
import UniformTypeIdentifiers

/// Web Studio's controls and policy, grouped for the native composer like
/// Bionic: attachments/settings on the left; model, microphone, send on the right.
struct StudioComposerView: View {
  static let pictureSettingsSymbol = "photo"
  @Bindable var chat: StudioWorkspace
  @Environment(\.studioToolsManager) private var toolsManager
  @Binding var draft: StudioDraft
  var maximumWidth: CGFloat = 760
  @State private var editorHeight: CGFloat = 72
  @State private var composerWidth: CGFloat = 760
  @State private var panel: Panel?
  enum Panel: CaseIterable {
    case reasoning, tools, instructions, sampling, context, document, image
    /// Direct controls own their popovers even when the toolbar is compact.
    /// Two presenters observing the same selection cancel each other on macOS.
    var usesOverflow: Bool {
      switch self {
      case .image: false
      default: true
      }
    }
  }
  struct ToolsPolicy {
    let imageMode: Bool
    let imageEditing: Bool
    let audioMode: Bool
    let speechAvailable: Bool
    var attachments: Bool { !imageMode || imageEditing }
    var speechLanguage: Bool { !imageMode && speechAvailable }
    func overflow(compact: Bool) -> Bool {
      compact && !imageMode && (!audioMode || speechLanguage)
    }
  }
  private var presentation: StudioState.Composer? { chat.state?.composer }
  private var settings: [String: StudioValue] { chat.state?.settings ?? [:] }
  private var params: [String: StudioValue] { settings["params"]?.object ?? [:] }
  private var models: [StudioState.Model] { chat.state?.models ?? [] }
  private var selected: [String] { chat.state?.selectedModels ?? [] }
  private var enabled: Bool { chat.ready && !chat.busy && !chat.hasMessageEdit }
  private var responding: Bool { chat.busy && !chat.microphoneBusy }
  private var toolsPolicy: ToolsPolicy {
    ToolsPolicy(
      imageMode: presentation?.imageMode == true, imageEditing: presentation?.imageEditing == true,
      audioMode: presentation?.audioMode == true,
      speechAvailable: chat.state?.audio?.audioOk == true
        || chat.state?.audio?.jobs.isEmpty == false)
  }
  private var provisionalDictation: String { chat.state?.audio?.composerProvisional ?? "" }
  private var hasReadableDocument: Bool {
    chat.state?.capabilities.hasDocument == true
      || chat.attachments.contains { $0.mime.hasPrefix("image/") || ($0.isPDF && !$0.textOnly) }
  }
  private var canSend: Bool {
    enabled && !chat.uploading
      && chat.attachmentBudgetIssue(for: draft.message) == nil
      && chat.attachments.allSatisfy { $0.ready && $0.selectionError == nil }
      && !selected.isEmpty && (presentation?.inputIssue.isEmpty ?? true)
      && (!draft.message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        || chat.hasAttachments || presentation?.docParser == true)
      && (presentation?.docParser != true || hasReadableDocument)
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if chat.hasAttachments {
        PaddockScrollView(.horizontal) {
          HStack(spacing: 8) {
            ForEach(chat.attachments) { StudioAttachmentChip(chat: chat, attachment: $0) }
          }
        }.scrollIndicators(.hidden).frame(height: 64)
      }
      if hasReadableDocument, let cap = chat.state?.capabilities, !cap.ocrModes.isEmpty {
        StudioDocumentReadingControls(
          modes: cap.ocrModes, grounding: cap.ocrGrounding == true,
          mode: settings["ocrMode"]?.text ?? "", regions: settings["ocrRegions"]?.boolean == true,
          onMode: { setting("ocrMode", .string($0)) },
          onRegions: { setting("ocrRegions", .bool($0)) }
        )
        .disabled(!enabled)
      }
      if presentation?.audioMode == true {
        Button(action: chooseFiles) {
          VStack(spacing: 6) {
            Image(systemName: "waveform").font(.system(size: 24)).foregroundStyle(.secondary)
            Text(
              chat.hasAttachments
                ? "Audio ready to transcribe" : "Drop an audio clip, or choose a file"
            )
            .font(.system(size: 14, weight: .medium))
          }.frame(maxWidth: .infinity, minHeight: 72).contentShape(Rectangle())
        }.buttonStyle(.plain).disabled(!enabled).accessibilityIdentifier("transcription-input")
      } else if presentation?.docParser == true {
        if !hasReadableDocument {
          Button("Drop an image or PDF to read", systemImage: "doc.text", action: chooseFiles)
            .buttonStyle(.plain).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, minHeight: 48)
            .disabled(!enabled)
        }
      } else {
        ZStack(alignment: .topLeading) {
          if draft.message.isEmpty && provisionalDictation.isEmpty {
            Text(presentation?.imageMode == true ? "Describe an image…" : "Ask anything…")
              .foregroundStyle(.secondary)
              .padding(.leading, StudioDraftEditor.horizontalTextPadding)
              .padding(.top, StudioDraftEditor.verticalTextPadding)
              .allowsHitTesting(false)
          }
          StudioDraftEditor(
            text: $draft.message, onSend: send, onFiles: chat.addFiles,
            onImage: chat.addPastedImage,
            dictationSession: chat.state?.audio?.session ?? "", dictation: chat.pendingDictation,
            provisionalDictation: provisionalDictation,
            onDictated: chat.acknowledgeDictation,
            onWindow: { chat.presentationWindow = $0 },
            onHeight: { editorHeight = $0 },
            position: chat.state?.composerDraft.map { chat.conversationPositions.record($0.id) }
          )
          // TextKit undo and marked-text state belong to this conversation,
          // never to whichever chat happened to occupy the editor previously.
          .id(chat.state?.composerDraft?.id)
          .frame(height: editorHeight)
          .accessibilityLabel("Message draft").accessibilityIdentifier("studio-message")
        }.font(.system(size: StudioDraftEditor.textFontSize)).padding(.horizontal, 8)
      }
      if let audio = chat.state?.audio {
        StudioMicrophoneStatus(audio: audio, meter: chat.microphoneMeter).padding(.horizontal, 8)
      }
      toolbar(compact: composerWidth < 640)
        .padding(
          .horizontal,
          StudioComposerLayout.horizontalControlInset - StudioComposerLayout.contentInset)
      if !chat.busy, let warning = warning {
        Text(warning).font(.system(size: 11)).foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true).padding(.horizontal, 8)
          .accessibilityIdentifier("composer-warning")
      }
    }
    .padding(.top, 12).padding(.horizontal, StudioComposerLayout.contentInset)
    .padding(.bottom, StudioComposerLayout.bottomInset).frame(maxWidth: maximumWidth)
    .background(
      PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.composer)
    )
    .overlay(
      RoundedRectangle(cornerRadius: PaddockStyle.Radius.composer).strokeBorder(PaddockStyle.border)
    )
    .onGeometryChange(for: CGFloat.self) {
      $0.size.width
    } action: {
      composerWidth = $0
    }
    .onChange(of: chat.busy) { _, busy in if busy { panel = nil } }
  }
  private var warning: String? {
    if chat.hasMessageEdit {
      return "Send or cancel the message edit above. Your composer draft is kept."
    }
    if let attachment = chat.attachments.first(where: { $0.selectionError != nil }) {
      return "\(attachment.name): \(attachment.selectionError ?? "")"
    }
    if let issue = chat.attachmentBudgetIssue(for: draft.message) { return issue }
    if chat.hasAttachments, let issue = presentation?.inputIssue, !issue.isEmpty { return issue }
    return presentation?.warnings.first
  }
  private func toolbar(compact: Bool) -> some View {
    HStack(spacing: 4) {
      composerTools(compact: compact)
      Spacer(minLength: 8)
      composerActions(compact: compact)
    }
  }
  private func composerTools(compact: Bool) -> some View {
    HStack(spacing: 4) {
      if toolsPolicy.attachments {
        Button(
          presentation?.imageMode == true ? "Attach a picture to edit" : "Attach files",
          systemImage: "paperclip", action: chooseFiles
        )
        .labelStyle(.iconOnly).buttonStyle(ComposerButtonStyle())
        .help(presentation?.imageMode == true ? "Attach a picture to edit" : "Attach files")
        .disabled(!enabled).accessibilityIdentifier("composer-attach")
      }
      if presentation?.imageMode == true {
        control("Picture settings", icon: Self.pictureSettingsSymbol, panel: .image)
          .accessibilityIdentifier("composer-image")
      }
      if !compact, let p = presentation, p.reasoning.count > 1, !p.audioMode, !p.docParser,
        p.imageMode != true
      {
        control(
          "Thinking", icon: "brain", panel: .reasoning, active: p.reasoningChoice != "off",
          caption: p.reasoning.first(where: { $0.value == p.reasoningChoice })?.label)
      }
      if !compact, toolsPolicy.speechLanguage, let audio = chat.state?.audio {
        Menu {
          languageChoices(audio)
        } label: {
          Text(audio.languages.first { $0.value == audio.language }?.label ?? "Language")
            .font(.system(size: 11)).lineLimit(1)
        }.menuStyle(.button).menuIndicator(.hidden).buttonStyle(ComposerButtonStyle())
          .disabled(!enabled).accessibilityLabel("Speech language")
      }
      if !compact {
        if presentation?.audioMode != true, presentation?.docParser != true,
          presentation?.imageMode != true
        {
          Button("Web search", systemImage: "globe") {
            toggleWebSearch()
          }.labelStyle(.iconOnly)
            .buttonStyle(
              ComposerButtonStyle(
                active: presentation?.webSearch == true
                  && (settings["webSearchEnabled"]?.boolean ?? true))
            )
            .disabled(!enabled || (presentation?.webSearch != true && selectedPort == nil))
            .help(
              presentation?.webSearch == true
                ? "Web search" : "Configure web search in Settings > Instances"
            )
            .accessibilityIdentifier("composer-web")
        }
        if presentation?.audioMode != true, presentation?.docParser != true,
          presentation?.imageMode != true
        {
          control(
            "Tools and connectors", icon: "puzzlepiece.extension", panel: .tools,
            active: (presentation?.toolCount ?? 0) > 0,
            caption: (presentation?.toolCount ?? 0) > 0 ? String(presentation!.toolCount) : nil)
          control(
            "Instructions", icon: "slider.horizontal.3", panel: .instructions,
            active: !(settings["systemPrompt"]?.text ?? "").isEmpty)
          control(
            "Sampling", icon: "thermometer.medium", panel: .sampling,
            active: presentation?.samplerSet == true)
        }
      } else if toolsPolicy.overflow(compact: compact) {
        Menu {
          if toolsPolicy.speechLanguage, let audio = chat.state?.audio {
            Menu("Speech language") { languageChoices(audio) }
          }
          if let p = presentation, p.reasoning.count > 1, !p.audioMode, !p.docParser,
            p.imageMode != true
          {
            Button("Thinking…") { panel = .reasoning }
          }
          if presentation?.audioMode != true, presentation?.docParser != true,
            presentation?.imageMode != true
          {
            Button(
              presentation?.webSearch != true
                ? "Configure web search…"
                : (settings["webSearchEnabled"]?.boolean ?? true)
                  ? "Turn off web search" : "Turn on web search"
            ) {
              toggleWebSearch()
            }.disabled(presentation?.webSearch != true && selectedPort == nil)
          }
          if presentation?.audioMode != true, presentation?.docParser != true,
            presentation?.imageMode != true
          {
            Button("Tools and connectors…") { panel = .tools }
            Button("Instructions…") { panel = .instructions }
            Button("Sampling…") { panel = .sampling }
          }
          if presentation?.audioMode != true, presentation?.imageMode != true {
            Button("Context and reading…") { panel = .context }
          }
        } label: {
          Image(systemName: "ellipsis")
        }
        .menuStyle(.button).buttonStyle(ComposerButtonStyle()).menuIndicator(.hidden).fixedSize()
        .disabled(
          !enabled
        )
        .accessibilityLabel("More composer options").accessibilityIdentifier("composer-more")
        .popover(isPresented: overflowPresented, arrowEdge: .top) { panelContent }
      }
      if !compact, let p = presentation, !p.audioMode, p.imageMode != true {
        Button {
          panel = .context
        } label: {
          HStack(spacing: 5) {
            ZStack {
              Circle().strokeBorder(.primary.opacity(0.12), lineWidth: 2)
              Circle().trim(
                from: 0,
                to: min(
                  1, Double(p.contextUsed) / Double(max(1, chat.state?.capabilities.context ?? 1)))
              )
              .stroke(.primary.opacity(0.65), style: StrokeStyle(lineWidth: 2, lineCap: .round))
              .rotationEffect(.degrees(-90))
            }.frame(width: 13, height: 13)
            if p.cost > 0 {
              Text(p.cost, format: .currency(code: "USD").precision(.fractionLength(2...4)))
            }
          }.font(.system(size: 10)).foregroundStyle(.secondary)
        }.buttonStyle(ComposerButtonStyle()).help(
          "Estimated context: \(p.contextUsed) tokens. Cost is reported spend, not a prediction."
        )
        .popover(isPresented: showing(.context), arrowEdge: .top) { panelContent }
        .accessibilityLabel("Context and conversation cost")
        .accessibilityIdentifier("composer-context")
      }
    }.fixedSize(horizontal: true, vertical: false)
      .accessibilityElement(children: .contain).accessibilityLabel("Attachments and settings")
      .accessibilityIdentifier("composer-tools")
  }
  private func composerActions(compact: Bool) -> some View {
    HStack(spacing: 4) {
      StudioComposerModelPicker(chat: chat, compact: compact)
        .layoutPriority(-1)
      if presentation?.docParser != true, presentation?.imageMode != true {
        StudioMicrophoneButton(chat: chat, draft: $draft, compact: compact)
          .fixedSize()
      }
      Button(
        responding
          ? "Stop response"
          : presentation?.audioMode == true
            ? "Transcribe audio"
            : presentation?.docParser == true ? "Read document" : "Send message",
        systemImage: responding ? "stop.fill" : "arrow.up"
      ) {
        if responding { Task { await chat.cancel() } } else { send() }
      }.labelStyle(.iconOnly).font(.system(size: 14, weight: .semibold))
        .buttonStyle(ComposerSendStyle()).disabled(chat.microphoneBusy || (!responding && !canSend))
        .help(
          chat.busy
            ? "Stop response"
            : presentation?.inputIssue.isEmpty == false ? presentation!.inputIssue : "Send · Return"
        )
        .accessibilityIdentifier(responding ? "studio-stop" : "studio-send")
    }.accessibilityElement(children: .contain).accessibilityLabel("Model and sending")
      .accessibilityIdentifier("composer-actions")
  }
  private func control(
    _ title: String, icon: String, panel target: Panel, active: Bool = false, caption: String? = nil
  ) -> some View {
    Button {
      panel = target
    } label: {
      HStack(spacing: 5) {
        Image(systemName: icon)
        if let caption { Text(caption).font(.system(size: 11)).lineLimit(1).fixedSize() }
      }
    }.buttonStyle(ComposerButtonStyle(active: active)).disabled(!enabled)
      .help(title).accessibilityLabel(title)
      .popover(isPresented: showing(target), arrowEdge: .top) { panelContent }
  }
  @ViewBuilder private func languageChoices(_ audio: StudioState.Audio) -> some View {
    ForEach(audio.languages) { language in
      Button {
        Task { await chat.perform("microphoneSettings", ["language": .string(language.value)]) }
      } label: {
        if audio.language == language.value {
          Label(language.label, systemImage: "checkmark")
        } else {
          Text(language.label)
        }
      }
    }
  }
  private func showing(_ target: Panel) -> Binding<Bool> {
    Binding(get: { panel == target }, set: { if !$0, panel == target { panel = nil } })
  }
  private var overflowPresented: Binding<Bool> {
    Binding(
      get: { panel?.usesOverflow == true },
      set: { if !$0, panel?.usesOverflow == true { panel = nil } })
  }
  @ViewBuilder private var panelContent: some View {
    Group {
      switch panel {
      case .reasoning: StudioReasoningControls(chat: chat)
      case .tools: StudioToolsView(chat: chat)
      case .instructions: StudioInstructionControls(chat: chat)
      case .sampling: StudioSamplingControls(chat: chat)
      case .image: StudioImageControls(chat: chat)
      case .context, .document: StudioContextControls(chat: chat)
      case nil: EmptyView()
      }
    }.studioPopoverSurface()
  }
  private var selectedPort: UInt16? { models.first { $0.id == selected.first }?.port }
  private func toggleWebSearch() {
    if presentation?.webSearch == true {
      setting("webSearchEnabled", .bool(!(settings["webSearchEnabled"]?.boolean ?? true)))
    } else if let port = selectedPort {
      toolsManager?.onConfigureSearch?(port)
    }
  }
  private func setting(_ key: String, _ value: StudioValue) {
    Task { await chat.perform("settings", [key: value]) }
  }
  private func send() {
    guard canSend else { return }
    let text = draft.message
    let transcription = presentation?.audioMode == true
    let organizedDocument = presentation?.docParser == true
    Task {
      if await chat.send(transcription || organizedDocument ? "" : text), !transcription,
        !organizedDocument, draft.message == text
      {
        draft.message = ""
      }
    }
  }
  private func chooseFiles() {
    guard let window = chat.presentationWindow ?? NSApp.keyWindow ?? NSApp.mainWindow else {
      return
    }
    let picker = NSOpenPanel()
    picker.canChooseDirectories = false
    picker.allowsMultipleSelection = true
    if presentation?.imageMode == true { picker.allowedContentTypes = [.image] }
    picker.beginSheetModal(for: window) { if $0 == .OK { chat.addFiles(picker.urls) } }
  }
}

enum StudioComposerLayout {
  static let contentInset: CGFloat = 8
  // Clear the rounded corners horizontally without lifting the button row.
  // Vertical spacing stays at the original eight points, independent of radius.
  static let horizontalControlInset: CGFloat = 16
  static let bottomInset: CGFloat = 8
}

struct ComposerButtonStyle: ButtonStyle {
  var active = false
  @Environment(\.isEnabled) private var enabled
  @State private var hovering = false
  func makeBody(configuration: Configuration) -> some View {
    configuration.label.font(.system(size: 14)).frame(minWidth: 18, minHeight: 18).padding(7)
      .foregroundStyle(active ? Color.primary : Color.secondary)
      .background(
        Color.primary.opacity(
          configuration.isPressed ? 0.12 : hovering ? 0.075 : active ? 0.055 : 0),
        in: Capsule()
      )
      .overlay(Capsule().strokeBorder(PaddockStyle.border, lineWidth: 1))
      .contentShape(Capsule()).onHover { hovering = $0 }
      .opacity(enabled ? 1 : 0.4)
  }
}
private struct ComposerSendStyle: ButtonStyle {
  @Environment(\.isEnabled) private var enabled
  func makeBody(configuration: Configuration) -> some View {
    configuration.label.frame(width: 32, height: 32)
      .foregroundStyle(PaddockStyle.surface).background(Color.primary, in: Circle())
      .opacity(enabled ? configuration.isPressed ? 0.65 : 1 : 0.25)
      .contentShape(Circle())
  }
}
