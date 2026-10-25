import AppKit
import PaddockStudio
import SwiftUI

/// Two retained hosting roots. Opening a document only collapses/uncollapses
/// the AppKit split item: it never replaces the chat, editor or WKWebView.
/// The existing workspace renders either a shared chat or a document, never
/// two copies. There is no second WebKit session or conversation store.
struct StudioDocumentSplit<Chat: View, Document: View>: NSViewControllerRepresentable {
  let open: Bool
  var right = false
  @ViewBuilder let chat: () -> Chat
  @ViewBuilder let document: () -> Document

  func makeNSViewController(context: Context) -> StudioSplitController {
    let controller = StudioSplitController()
    controller.right = right
    updateNSViewController(controller, context: context)
    return controller
  }
  func updateNSViewController(_ controller: StudioSplitController, context: Context) {
    controller.update(
      chat: AnyView(chat().environment(\.self, context.environment)),
      document: AnyView(document().environment(\.self, context.environment)), open: open)
  }
  func sizeThatFits(
    _ proposal: ProposedViewSize, nsViewController: StudioSplitController, context: Context
  ) -> CGSize? {
    // This is a viewport, not content-sized UI. Do not feed the split's last
    // fitted child sizes back into the enclosing SwiftUI/sidebar split.
    guard let width = proposal.width, let height = proposal.height,
      width.isFinite, height.isFinite
    else { return nil }
    return CGSize(width: max(0, width), height: max(0, height))
  }
}

@MainActor final class StudioSplitController: NSSplitViewController {
  var right = false
  let chatHost = NSHostingController(rootView: AnyView(EmptyView()))
  let documentHost = NSHostingController(rootView: AnyView(EmptyView()))
  private(set) var documentItem: NSSplitViewItem!
  private weak var returnFocus: NSResponder?
  private var documentWidth: CGFloat = 460
  private var opened = false
  private var readingAnchor: StudioReadingAnchor?
  private var lastReadingAnchor: StudioReadingAnchor?
  private var lastReadingOffset: CGFloat?
  private var restoringAnchor = false
  private var restoreQueued = false
  private var paneLayoutTask: Task<Void, Never>?

  @objc private func willResize(_ notification: Notification) {
    captureReadingPosition()
  }
  private func captureReadingPosition() {
    if !restoringAnchor, readingAnchor == nil {
      // Keep the exact character across successive widths until the reader
      // scrolls. Recapturing the start of each newly wrapped line accumulates
      // a line of drift on every open/close cycle.
      if let previous = lastReadingAnchor, let scroll = previous.scroll,
        let offset = lastReadingOffset, abs(scroll.documentVisibleRect.minY - offset) < 1
      {
        readingAnchor = previous
      } else {
        readingAnchor = StudioReadingAnchor.capture(in: chatHost.view)
      }
    }
  }
  @objc private func didResize(_ notification: Notification) {
    guard !restoreQueued, !restoringAnchor, readingAnchor != nil else { return }
    restoreQueued = true
    // Let GeometryReader publish the new column width and TextKit finish its
    // reflow. Coalesce split notifications; never queue work per text token.
    Task { @MainActor [weak self] in
      await Task.yield()
      guard let self else { return }
      view.layoutSubtreeIfNeeded()
      await Task.yield()
      view.layoutSubtreeIfNeeded()
      restoringAnchor = true
      readingAnchor?.restore()
      lastReadingAnchor = readingAnchor
      lastReadingOffset = readingAnchor?.scroll?.documentVisibleRect.minY
      readingAnchor = nil
      restoringAnchor = false
      restoreQueued = false
    }
  }

  override func viewDidLoad() {
    super.viewDidLoad()
    splitView.isVertical = true
    splitView.dividerStyle = .thin
    splitView.setAccessibilityIdentifier("studio-document-split")
    // Hosting's content ideal size must not impose an unbounded minimum (a
    // long code block, file name or welcome heading cannot widen the window).
    chatHost.sizingOptions = []
    documentHost.sizingOptions = []
    // The shell distributes native chrome clearance to the owning column.
    // Re-inheriting the window safe area in each nested host would insert an
    // empty title-bar strip above every document/artifact header a second time.
    chatHost.safeAreaRegions = []
    documentHost.safeAreaRegions = []
    let chatItem = NSSplitViewItem(viewController: chatHost)
    chatItem.minimumThickness = 320
    documentItem = NSSplitViewItem(viewController: documentHost)
    documentItem.minimumThickness = 320
    // NSSplitViewItem turns this into an Auto Layout constant. Infinity-sized
    // constants overflow AppKit's solver and can leave a reopened pane at zero
    // width. This ceiling exceeds supported desktop widths without overflow.
    documentItem.maximumThickness = right ? 10000 : 1000
    documentItem.holdingPriority = .init(260)
    documentItem.collapseBehavior = .preferResizingSiblingsWithFixedSplitView
    // Close explicitly, not accidentally by dragging through the minimum.
    documentItem.canCollapse = false
    documentItem.isCollapsed = true
    // Match web Studio: history | documents | chat | graphs/artifacts.
    // These are document previews, not the right-hand artifact inspector.
    if right {
      addSplitViewItem(chatItem)
      addSplitViewItem(documentItem)
    } else {
      addSplitViewItem(documentItem)
      addSplitViewItem(chatItem)
    }
    _ = documentHost.view  // Retain the workspace even while the item is folded.
    NotificationCenter.default.addObserver(
      self, selector: #selector(willResize), name: NSSplitView.willResizeSubviewsNotification,
      object: splitView)
    NotificationCenter.default.addObserver(
      self, selector: #selector(didResize), name: NSSplitView.didResizeSubviewsNotification,
      object: splitView)
  }

  func update(chat: AnyView, document: AnyView, open: Bool) {
    loadViewIfNeeded()
    if open != opened { captureReadingPosition() }
    let window = view.window
    let closingFromDocument = !open && opened && belongsToDocument(window?.firstResponder)
    if open && !opened { returnFocus = window?.firstResponder }
    if !open && opened { documentWidth = max(320, documentHost.view.frame.width) }
    chatHost.rootView = chat
    documentHost.rootView = document
    guard opened != open else { return }
    opened = open
    paneLayoutTask?.cancel()
    // Representable updates run inside SwiftUI's layout transaction. Forcing
    // the nested split to lay out here reenters the parent graph while it is
    // resolving the sidebar/header size. Apply only the latest pane change
    // after that transaction, preserving the same hosts and reading anchor.
    paneLayoutTask = Task { @MainActor [weak self] in
      await Task.yield()
      guard let self, !Task.isCancelled, opened == open else { return }
      documentItem.isCollapsed = !open
      view.layoutSubtreeIfNeeded()
      if open {
        let available = splitView.bounds.width - splitView.dividerThickness
        if available >= 640 {
          let width = min(documentWidth, available - 320)
          splitView.setPosition(right ? available - width : width, ofDividerAt: 0)
        }
      } else if closingFromDocument {
        if let target = returnFocus as? NSView, target.window === window {
          window?.makeFirstResponder(target)
        } else if let editor = editor(in: chatHost.view) {
          window?.makeFirstResponder(editor)
        }
      }
    }
  }

  private func belongsToDocument(_ responder: NSResponder?) -> Bool {
    guard let view = responder as? NSView else { return false }
    return view === documentHost.view || view.isDescendant(of: documentHost.view)
  }
  private func editor(in view: NSView) -> DraftTextView? {
    if let editor = view as? DraftTextView { return editor }
    return view.subviews.lazy.compactMap { self.editor(in: $0) }.first
  }
}

struct StudioDocumentHeader: View {
  let document: StudioState.Document?
  let onClose: () -> Void
  var onAction: (String) -> Void = { _ in }
  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: document?.kind == "image" ? "photo" : "doc.richtext")
        .foregroundStyle(.secondary)
      Text(verbatim: document?.name ?? "Document").font(.system(size: 12, weight: .medium))
        .lineLimit(1).truncationMode(.middle).help(document?.name ?? "Document")
      Spacer(minLength: 8)
      Button("Document details", systemImage: "info.circle") { onAction("info") }
        .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).help("Document details")
      Button("Save original", systemImage: "square.and.arrow.down") { onAction("download") }
        .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).help("Save original")
      Button("Close document", systemImage: "xmark", action: onClose)
        .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
        .help("Close document").accessibilityIdentifier("close-native-document")
    }.padding(.horizontal, 12).frame(height: 38)
      .background(PaddockStyle.canvas).overlay(alignment: .bottom) { WorkspaceRule() }
      .accessibilityIdentifier("native-document-header")
  }
}

struct NativeDocumentBadge: View {
  let document: StudioState.Document
  let onOpen: () -> Void
  var body: some View {
    Button(action: onOpen) {
      HStack(spacing: 10) {
        Image(systemName: document.kind == "image" ? "photo" : "doc.richtext")
          .font(.system(size: 20)).foregroundStyle(.secondary)
        VStack(alignment: .leading, spacing: 4) {
          Text(verbatim: document.name).font(.system(size: 12, weight: .medium))
            .lineLimit(1).truncationMode(.middle)
          Text(summary).font(.system(size: 10)).foregroundStyle(.secondary)
        }
      }.padding(10).frame(maxWidth: 260, alignment: .leading)
        .background(
          PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
        )
        .overlay(
          RoundedRectangle(cornerRadius: PaddockStyle.Radius.card).strokeBorder(PaddockStyle.border)
        )
    }.buttonStyle(QuietButtonStyle()).help("Open \(document.name)")
      .accessibilityLabel("Open \(document.name), \(summary)")
      .accessibilityIdentifier("native-document-\(document.id)")
  }
  private var summary: String {
    let pages =
      document.pageRange.map { "Pages \($0)" }
      ?? document.pages.map { "\($0) \($0 == 1 ? "page" : "pages")" }
      ?? document.kind.uppercased()
    return pages + (document.textOnly ? " · Text only" : "")
  }
}
