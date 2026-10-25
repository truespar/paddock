import AppKit
import PaddockClient
import PaddockConversationCore
import PaddockStudio
import SwiftUI
import WebKit

/// Native conversation and composer. Embedded rendering is allowlisted to
/// Lector/Scriptor (left), Traverse and isolated HTML/SVG artifacts (right),
/// never a transcript fallback.
struct StudioConversationView: View {
  @Bindable var chat: StudioWorkspace
  @Binding var draft: StudioDraft
  var notices: AnyView?
  @Environment(\.colorScheme) private var colorScheme
  @Environment(\.workspaceLeadingPaneInset) private var leadingPaneInset
  @State private var measuredComposerHeight: CGFloat = 172
  @State private var dropTargeted = false
  var welcome: Bool {
    // Saving an unsent draft must not move the composer while someone types.
    (chat.conversation?.messageCount ?? 0) == 0
  }
  private var graphOpen: Bool { chat.state?.nativeGraph?.visible == true }
  private var artifact: StudioState.Artifact? {
    chat.state?.nativeArtifacts?.first { $0.id == chat.selectedArtifactId }
  }
  var body: some View {
    StudioDocumentSplit(open: chat.state?.nativeDocument != nil) {
      StudioDocumentSplit(open: graphOpen || artifact != nil, right: true) {
        conversationContent
      } document: {
        Group {
          if graphOpen {
            VStack(spacing: 0) {
              HStack {
                Text("Traverse").font(.system(size: 12))
                Spacer()
                Button("Close graph", systemImage: "xmark") {
                  Task { await chat.perform("graphPanel", ["open": .bool(false)]) }
                }.labelStyle(.iconOnly).buttonStyle(.plain)
              }.padding(12)
              WorkspaceContentView(session: chat, role: .graph)
            }.background(PaddockStyle.canvas)
          } else if artifact != nil {
            NativeArtifactPanel(workspace: chat)
          }
        }.padding(.top, leadingPaneInset)
      }.ignoresSafeArea(.container, edges: .top)
    } document: {
      Group {
        if let document = chat.state?.nativeDocument, document.kind == "image" {
          NativeImagePreview(document: document, workspace: chat)
        } else {
          VStack(spacing: 0) {
            StudioDocumentHeader(
              document: chat.state?.nativeDocument,
              onClose: { Task { await chat.perform("closePreview") } },
              onAction: { action in
                Task { await chat.perform("documentAction", ["action": .string(action)]) }
              })
            if let document = chat.state?.nativeDocument, ["pdf", "docx"].contains(document.kind) {
              WorkspaceContentView(session: chat, role: .document)
            }
          }.background(PaddockStyle.canvas)
        }
      }.padding(.top, leadingPaneInset)
    }
    .ignoresSafeArea(.container, edges: .top)
    .task { await chat.start() }
    .task(id: colorScheme) { await chat.setDark(colorScheme == .dark) }
    .task(id: welcome ? 0 : measuredComposerHeight) {
      await chat.setComposerInset(welcome ? 0 : measuredComposerHeight)
    }
    .onDrop(of: [.fileURL, .png, .tiff], isTargeted: $dropTargeted, perform: chat.addDroppedItems)
    .overlay {
      if dropTargeted {
        RoundedRectangle(cornerRadius: 12)
          .fill(PaddockStyle.canvas.opacity(0.96))
          .overlay(
            RoundedRectangle(cornerRadius: 12).strokeBorder(
              PaddockStyle.border, style: StrokeStyle(lineWidth: 2, dash: [6]))
          )
          .overlay {
            Label(
              chat.state?.composer?.audioMode == true
                ? "Drop an audio clip" : "Drop images, audio, PDFs, or documents",
              systemImage: "paperclip"
            )
            .font(.system(size: 16, weight: .medium)).foregroundStyle(.primary)
          }.padding(16).allowsHitTesting(false)
      }
    }
  }
  private var conversationContent: some View {
    GeometryReader { geometry in
      let column = StudioColumnLayout.resolve(
        available: geometry.size.width, viewport: nil,
        comparison: chat.state?.nativeTranscript?.hasComparisons == true)
      StudioConversationChrome {
        VStack(spacing: 0) {
          if let notices { notices }
          Group {
            // NativeStudioRuntime owns streaming; no hidden WebView is required.
            // Attach web content only in the allowlisted viewer panels above.
            if !welcome, let transcript = chat.state?.nativeTranscript {
              NativeStudioTranscript(
                transcript: transcript, columnWidth: column.width,
                composerHeight: measuredComposerHeight,
                composer: AnyView(composer(columnWidth: column.width)),
                workspace: chat,
                onOpenDocument: { messageId, attachmentId in
                  Task {
                    await chat.perform(
                      "openDocument",
                      ["messageId": .string(messageId), "attachmentId": .string(attachmentId)])
                  }
                }
              )
              .id(chat.conversation?.id)
            } else {
              composer(columnWidth: column.width)
            }
          }
          .frame(width: geometry.size.width)
          .frame(maxHeight: .infinity)
          .background(PaddockStyle.canvas)
        }
      }
    }
  }

  private func composer(columnWidth: CGFloat) -> some View {
    VStack(spacing: 0) {
      if welcome {
        Spacer(minLength: 30)
        Text("What would you like to work on?")
          .font(.system(size: 28, weight: .medium)).tracking(-0.6).padding(.bottom, 26)
      }
      if let error = chat.error {
        HStack(alignment: .top, spacing: 8) {
          Text(error).textSelection(.enabled).font(.system(size: 12)).foregroundStyle(
            .secondary)
          if !chat.ready {
            Button("Reload content") { chat.reload() }.buttonStyle(FlatButtonStyle())
          } else if chat.state?.saveFailed == true || chat.positionSaveError != nil {
            Button("Retry save") { Task { await chat.perform("retrySave") } }
              .buttonStyle(FlatButtonStyle()).disabled(chat.switchingConversation)
          }
        }.padding(.bottom, 10).accessibilityIdentifier("chat-error")
      }
      if let clip = chat.state?.nativeAudioPreview {
        HStack(alignment: .top, spacing: 8) {
          NativeAudioPlayerView(clip: clip, workspace: chat)
          Button("Close recording preview", systemImage: "xmark") {
            Task { await chat.perform("closePreview") }
          }.labelStyle(.iconOnly).buttonStyle(.plain).padding(.top, 14)
        }.padding(.bottom, 12)
      }
      StudioComposerView(chat: chat, draft: $draft, maximumWidth: columnWidth)
      if welcome {
        Spacer(minLength: 30)
      }
    }
    .padding(.bottom, welcome ? 0 : StudioConversationSpacing.edgeInset)
    .frame(width: columnWidth)
    .onGeometryChange(for: CGFloat.self) {
      $0.size.height
    } action: { height in
      if !welcome, height > 0 { measuredComposerHeight = height }
    }
  }
}

enum StudioConversationSpacing {
  static let edgeInset: CGFloat = 20
  static let composerGap: CGFloat = 20
  static let scrollIndicatorInset: CGFloat = 12
}

enum StudioColumnLayout {
  /// The content workspace is authoritative once measured. In particular,
  /// don't add another gutter, force a minimum or cap wide compare columns.
  static func resolve(available: CGFloat, viewport: StudioState.Viewport?, comparison: Bool = false)
    -> CGRect
  {
    let available = max(0, available)
    if let viewport, viewport.width > 0, viewport.width.isFinite, viewport.left.isFinite {
      let left = min(available, max(0, CGFloat(viewport.left)))
      return CGRect(x: left, y: 0, width: min(CGFloat(viewport.width), available - left), height: 0)
    }
    // Start page / first frame before WebKit has measured its content column.
    let width = min(comparison ? 1240 : 760, max(0, available - 56))
    return CGRect(x: (available - width) / 2, y: 0, width: width, height: 0)
  }
}

struct WorkspaceContentView: NSViewRepresentable {
  let session: StudioWorkspace
  let role: NativeViewerRole
  func makeNSView(context: Context) -> WorkspaceWebContainer { WorkspaceWebContainer() }
  func updateNSView(_ view: WorkspaceWebContainer, context: Context) {
    view.embed(session.webView(for: role))
  }
}

/// SwiftUI owns each slot, not its role's WKWebView. Returning that same web
/// view from two representable identities lets the retiring slot's teardown
/// remove it from its new parent after a chat/document/Settings transition.
final class WorkspaceWebContainer: NSView {
  private weak var content: WKWebView?
  func embed(_ webView: WKWebView) {
    content = webView
    attach()
  }
  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    attach()
  }
  override func layout() {
    super.layout()
    attach()
  }
  private func attach() {
    // SwiftUI also builds temporary sizing hosts. Those must never take the
    // live audio/document view away from the real, window-attached slot.
    guard window != nil, !isHiddenOrHasHiddenAncestor, let webView = content else { return }
    guard webView.superview !== self else { return }
    webView.removeFromSuperview()
    webView.frame = bounds
    webView.autoresizingMask = [.width, .height]
    addSubview(webView)
  }
}
