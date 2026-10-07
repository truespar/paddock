import AppKit
import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI

extension EnvironmentValues {
  @Entry var transcriptDisclosure: TranscriptScrollIntent? = nil
}

/// Stable identity across streamed deltas; no per-token environment closure.
@Observable @MainActor final class TranscriptScrollIntent {
  var pinned = true
  var readingDisclosure = false
  var awayFromBottom = false
  var layoutRevision = 0
  @ObservationIgnored private var measuredRows: [String: CGSize] = [:]

  func didLayout(_ id: String, size: CGSize) {
    guard size.width.isFinite, size.height.isFinite, size.width > 0, size.height >= 0,
      measuredRows[id] != size
    else { return }
    measuredRows[id] = size
    // A remounted row of the same size is not new content. Keep this cache
    // outside lazy row state, which SwiftUI may discard when scrolling away.
    if pinned { layoutRevision &+= 1 }
  }
  func reveal() {
    readingDisclosure = true
    pinned = false
  }
  func retainRows(_ ids: Set<String>) {
    measuredRows = measuredRows.filter { ids.contains($0.key) || $0.key == "native-bottom" }
  }
}

/// Native presentation of the shared tree, not a second store or stream consumer.
struct NativeStudioTranscript: View {
  let transcript: StudioState.NativeTranscript
  let columnWidth: CGFloat
  let composerHeight: CGFloat
  var composer: AnyView?
  var workspace: StudioWorkspace?
  var onOpenDocument: ((String, String) -> Void)?
  @State private var scrollIntent = TranscriptScrollIntent()
  var body: some View {
    ConversationSelectionSurface(items: selectionItems) {
      ScrollViewReader { proxy in
        ZStack(alignment: .bottom) {
          PaddockScrollView(centersContent: true) {
            // Rich-text rows report their height across an AppKit boundary.
            // LazyVStack's offscreen estimates can oscillate while a tall row
            // straddles this viewport, trapping macOS in repeated layout passes.
            // Use real row sizes; never feed those estimates back into AppKit.
            VStack(alignment: .leading, spacing: 28) {
              ForEach(transcript.blocks) { block in
                VStack(alignment: .leading, spacing: 28) {
                  if let context = transcript.context,
                    block.messages.contains(where: { $0.id == context.before })
                  {
                    NativeContextBoundary(context: context)
                  }
                  if block.comparison {
                    NativeCompareBlock(
                      block: block, transcript: transcript, width: columnWidth, workspace: workspace
                    )
                  } else if let message = block.messages.first {
                    NativeStudioMessage(
                      message: message, workspace: workspace,
                      target: StudioMessageTarget(transcript: transcript, messageId: message.id),
                      onOpenDocument: onOpenDocument
                    ).equatable()
                  }
                }
                .onGeometryChange(for: CGSize.self) {
                  $0.size
                } action: { size in
                  scrollIntent.didLayout(block.id, size: size)
                }
              }
              VStack(alignment: .leading, spacing: 28) {
                if let context = transcript.context, context.before == nil {
                  NativeContextBoundary(context: context)
                }
                // The native bar reserves composer clearance. This is only a
                // stable follow anchor, never a second copy of the bar's height.
                Color.clear.frame(height: 1)
              }
              .onGeometryChange(for: CGSize.self) {
                $0.size
              } action: { size in
                scrollIntent.didLayout("native-bottom", size: size)
              }
              .id("native-bottom")
            }
            .frame(width: columnWidth, alignment: .leading)
            .padding(.top, StudioConversationSpacing.edgeInset)
            .frame(maxWidth: .infinity)
          }
          // The viewport starts below the header; the bottom still reaches
          // the window edge behind the floating composer.
          .contentMargins(
            .vertical, StudioConversationSpacing.scrollIndicatorInset, for: .scrollIndicators
          )
          .modifier(NativeTranscriptScrolling(intent: scrollIntent, proxy: proxy))
          .onChange(of: transcript.leafId) { _, _ in
            // A branch swap is deliberate navigation, not a request to jump to
            // the bottom. Keep the branch point being operated on in view.
            if let id = workspace?.messageNavigationAnchor {
              scrollIntent.pinned = false
              proxy.scrollTo(id, anchor: .top)
            }
          }
          .onChange(of: workspace?.messageEdit?.target.messageId) { _, id in
            if let id {
              scrollIntent.pinned = false
              proxy.scrollTo(id, anchor: .top)
            }
          }
          .modifier(
            StudioConversationBars {
              if workspace?.state?.nativeGraph?.available == true {
                graphControl
              }
            } footer: {
              // An erased EmptyView accepts the bar's full-height proposal.
              // A genuinely absent footer must reserve no content clearance.
              if let composer { composer }
            }
          )
          NativeTranscriptLatestControl(
            intent: scrollIntent, columnWidth: columnWidth, composerHeight: composerHeight,
            proxy: proxy)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .environment(\.transcriptDisclosure, scrollIntent)
        .environment(\.nativeMarkdownViewportReclamation, true)
        .onChange(of: transcript.blocks.map(\.id), initial: true) { _, ids in
          scrollIntent.retainRows(Set(ids))
        }
      }.background(PaddockStyle.canvas).accessibilityIdentifier("native-studio-transcript")
    }
  }
  private var graphControl: some View {
    HStack {
      Spacer()
      Button("Graph", systemImage: "point.3.connected.trianglepath.dotted") {
        guard let workspace else { return }
        workspace.selectedArtifactId = nil
        Task {
          await workspace.perform(
            "graphPanel", ["open": .bool(workspace.state?.nativeGraph?.visible != true)])
        }
      }.buttonStyle(.plain).font(.system(size: 12))
    }.padding(.horizontal, 28).padding(.vertical, 8)
  }
  private var selectionItems: [ConversationSelectionItem] {
    transcript.messages.flatMap { message in
      var items: [ConversationSelectionItem] = []
      if !message.reasoning.isEmpty {
        items.append(
          .init(
            id: message.id + "/reasoning", text: message.reasoning,
            includeWhenUnmounted: false, markdown: true))
      }
      for tool in message.toolCalls ?? [] {
        items.append(
          .init(id: tool.id + "/arguments", text: tool.arguments, includeWhenUnmounted: false))
        items.append(.init(id: tool.id + "/output", text: tool.output, includeWhenUnmounted: false))
      }
      if let result = message.documentResult {
        for page in result.pages {
          items.append(.init(id: message.id + "/page-\(page.id)", text: page.text, markdown: true))
        }
      } else {
        items.append(
          .init(
            id: message.id + "/body",
            text: message.speech?.words.map(\.word).joined(separator: " ") ?? message.text,
            markdown: message.role != "user" && message.speech == nil))
      }
      if !message.error.isEmpty {
        items.append(
          .init(id: message.id + "/error", text: ResponseErrorPresentation(message.error).message))
      }
      return items
    }
  }
}

/// A viewport sibling of the scroll view, not part of its scrolling overlay.
/// The composer measurement includes previews, warnings and bottom padding.
struct NativeTranscriptLatestButton: View {
  let columnWidth: CGFloat
  let composerHeight: CGFloat
  var action: () -> Void

  var body: some View {
    Button("Latest", systemImage: "arrow.down", action: action)
      .buttonStyle(FlatButtonStyle())
      .fixedSize()
      .accessibilityIdentifier("native-transcript-latest")
      .help("Scroll to the latest response")
      .padding(.trailing, 12)
      .frame(width: columnWidth, alignment: .trailing)
      .padding(.bottom, max(0, composerHeight) + 12)
  }
}

struct NativeStudioMessage: View, Equatable {
  let message: StudioState.NativeTranscript.Message
  var workspace: StudioWorkspace?
  var target: StudioMessageTarget?
  var onOpenDocument: ((String, String) -> Void)?
  var inLane = false
  nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.message == rhs.message && lhs.target == rhs.target && lhs.workspace === rhs.workspace
      && lhs.inLane == rhs.inLane
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      content
      NativeMessageFooter(
        message: message, workspace: workspace, target: target, showsBranches: !inLane)
    }
    // Streaming prose needs clearance before the footer exists. Once it
    // arrives, the lane's own bottom inset is sufficient below its controls.
    .padding(.bottom, inLane && message.streaming ? 28 : 0)
  }
  private var content: some View {
    VStack(alignment: .leading, spacing: 12) {
      if message.automatic == true {
        Label("Automatic import report", systemImage: "arrow.trianglehead.2.clockwise.rotate.90")
          .font(.caption).foregroundStyle(.secondary)
      }
      if message.role == "user" {
        if let workspace, let edit = workspace.messageEdit, edit.target.messageId == message.id {
          NativeMessageEditor(chat: workspace, edit: edit)
        } else if !message.text.isEmpty {
          NativeSelectableText(message.text)
            .environment(\.conversationTextID, message.id + "/body")
            .padding(16).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 18))
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        ForEach(message.attachments ?? []) { document in
          NativeDocumentBadge(document: document) { onOpenDocument?(message.id, document.id) }
            .disabled(onOpenDocument == nil).frame(maxWidth: .infinity, alignment: .trailing)
        }
        if let workspace {
          ForEach(message.audioClips ?? []) { clip in
            NativeAudioPlayerView(clip: clip, workspace: workspace)
          }
        }
        if message.audioPending == true {
          Label("Audio capture", systemImage: "waveform").font(.system(size: 12)).foregroundStyle(
            .secondary)
        }
      } else {
        if !inLane { NativeMessageHeader(message: message) }
        if let workspace {
          ForEach(message.pictures ?? []) { picture in
            NativeGeneratedImage(picture: picture, workspace: workspace) {
              onOpenDocument?(message.id, picture.id)
            }
          }
        }
        if !message.reasoning.isEmpty {
          NativeThinkingBlock(message: message)
            .environment(\.conversationTextID, message.id + "/reasoning")
        }
        if !(message.searches ?? []).isEmpty || !(message.toolCalls ?? []).isEmpty {
          VStack(alignment: .leading, spacing: 8) {
            ForEach(message.searches ?? []) { search in NativeSearchCallView(search: search) }
            ForEach(message.toolCalls ?? []) { call in
              NativeToolCallView(call: call, workspace: workspace, target: target)
            }
          }
        }
        if let result = message.documentResult {
          NativeDocumentResult(
            result: result, selectionPrefix: message.id, onOpenDocument: onOpenDocument,
            workspace: workspace)
        } else if let speech = message.speech {
          NativeSpeechView(speech: speech, workspace: workspace)
            .environment(\.conversationTextID, message.id + "/body")
        } else if !message.text.isEmpty {
          NativeMarkdown(message.text, streaming: message.streaming).equatable()
            .environment(\.conversationTextID, message.id + "/body")
            .font(.system(size: 15)).frame(maxWidth: .infinity, alignment: .leading)
        }
        if message.streaming && message.text.isEmpty && message.reasoning.isEmpty
          && message.documentResult == nil
          && (message.pictures ?? []).isEmpty
        {
          NativeResponseWaiting()
        }
        if !message.error.isEmpty {
          NativeResponseErrorView(
            error: message.error,
            onRetry:
              message.actions?.retry == true && workspace != nil && target != nil
              ? {
                guard let workspace, let target else { return }
                Task { await workspace.messageAction("retry", target: target) }
              } : nil,
            retryDisabled: workspace?.ready != true || workspace?.busy == true
              || workspace?.hasMessageEdit == true
          )
          .environment(\.conversationTextID, message.id + "/error")
          .accessibilityIdentifier("native-message-error-\(message.id)")
        }
        if message.stopped && message.text.isEmpty {
          Text("Stopped").font(.caption).foregroundStyle(.secondary)
        }
        if message.incomplete && !message.streaming {
          HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(message.chrome?.cutNote ?? "Output limit reached").font(.caption).foregroundStyle(
              .secondary)
            if message.actions?.continueReply == true, let workspace, let target {
              Button("Continue") {
                Task { await workspace.messageAction("continue", target: target) }
              }
              .buttonStyle(.plain).font(.caption)
              .disabled(!workspace.ready || workspace.busy || workspace.hasMessageEdit)
            }
          }
        }
      }
      ForEach(message.files ?? []) { file in
        HStack {
          Label(
            file.name,
            systemImage: file.kind == "graph" ? "point.3.connected.trianglepath.dotted" : "doc")
          if file.kind == "graph", let workspace {
            Button("Open graph") {
              Task { await workspace.perform("graphPanel", ["open": .bool(true)]) }
            }
          }
          if file.stored, let workspace {
            Button("Save original") {
              Task { await workspace.saveOriginal(file.id, name: file.name) }
            }
          } else {
            Text("Original attachment unavailable").foregroundStyle(.secondary)
          }
        }.font(.system(size: 12)).buttonStyle(.plain)
      }
    }
  }
}

/// MessageBubble.vue's delayed three-dot pill. No empty Markdown surface or
/// flexible-height spinner can insert a blank region before reasoning arrives.
struct NativeResponseWaiting: View {
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @State private var visible = false
  @State private var started = Date()
  var body: some View {
    ZStack(alignment: .leading) {
      if visible {
        TimelineView(.animation(minimumInterval: 1.0 / 15, paused: reduceMotion)) { context in
          HStack(spacing: 4) {
            ForEach(0..<3) { index in
              let phase = (context.date.timeIntervalSince(started) - Double(index) * 0.15)
                .truncatingRemainder(dividingBy: 1.4)
              Circle().fill(
                PaddockStyle.secondary.opacity(
                  reduceMotion ? 0.5 : (phase >= 0 && phase < 0.6 ? 0.9 : 0.25))
              )
              .frame(width: 6, height: 6)
            }
          }.padding(.vertical, 11).padding(.horizontal, 15)
            .background(PaddockStyle.elevated, in: RoundedRectangle(cornerRadius: 16))
        }.fixedSize().accessibilityElement(children: .ignore).accessibilityLabel(
          "Waiting for response"
        )
        .accessibilityIdentifier("native-response-waiting")
      }
    }.task {
      do {
        try await Task.sleep(for: .milliseconds(450))
        try Task.checkCancellation()
      } catch { return }
      started = Date()
      visible = true
    }
  }
}
