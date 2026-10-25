import AppKit
import PaddockStudio
import SwiftUI

/// A real native editor: IME composition is never mistaken for Send. Return
/// sends, Shift-Return inserts a newline; paste/drop files goes to Swift too.
struct StudioDraftEditor: NSViewRepresentable {
  // The SwiftUI placeholders share these metrics with TextKit, rather than
  // guessing its implicit line-fragment inset or using a different font.
  static let textFontSize: CGFloat = 15
  static let horizontalTextPadding: CGFloat = 5
  static let verticalTextPadding: CGFloat = 1
  @Environment(\.isEnabled) private var isEnabled
  @Binding var text: String
  let onSend: () -> Void
  var onFiles: (([URL]) -> Void)?
  var onImage: ((Data, Bool) -> Void)?
  var onCancel: (() -> Void)?
  var dictationSession = ""
  var dictation: [StudioState.Audio.Item] = []
  var provisionalDictation = ""
  var onDictated: ((String, Int) -> Void)?
  var focusOnMount = false
  var editorID = "studio-message"
  var editorLabel = "Message draft"
  var onWindow: (NSWindow?) -> Void = { _ in }
  var onHeight: (CGFloat) -> Void = { _ in }
  var position: StudioConversationPosition?
  func makeCoordinator() -> Coordinator { Coordinator(self) }
  func makeNSView(context: Context) -> NSScrollView {
    // Match the initial document width before installing its width autoresizing
    // mask. Starting the scroll view at zero added the first layout's width to
    // the document's 640 points, making wrapping/caret geometry overflow.
    let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 640, height: 72))
    scroll.drawsBackground = false
    scroll.hasVerticalScroller = true
    scroll.autohidesScrollers = true
    PaddockScrollbars.install(on: scroll)
    // Explicit TextKit 2 graph. Never access the legacy layoutManager property:
    // doing so would switch this editor back to TextKit 1.
    let storage = NSTextContentStorage()
    let layout = NSTextLayoutManager()
    storage.addTextLayoutManager(layout)
    let container = NSTextContainer(
      size: NSSize(width: 640, height: CGFloat.greatestFiniteMagnitude))
    container.lineFragmentPadding = Self.horizontalTextPadding
    layout.textContainer = container
    let editor = DraftTextView(
      frame: NSRect(x: 0, y: 0, width: 640, height: 72), textContainer: container)
    editor.contentStorage = storage
    editor.isRichText = false
    editor.isEditable = isEnabled
    editor.allowsUndo = true
    editor.drawsBackground = false
    editor.font = .systemFont(ofSize: Self.textFontSize)
    editor.textColor = .labelColor
    editor.textContainerInset = NSSize(width: 0, height: Self.verticalTextPadding)
    editor.isVerticallyResizable = true
    editor.isHorizontallyResizable = false
    editor.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
    editor.autoresizingMask = [.width]
    editor.textContainer?.widthTracksTextView = true
    context.coordinator.editor = editor
    editor.string = text
    if let selection = position?.value.composer?.selection(in: text) {
      editor.setSelectedRange(selection)
    } else {
      editor.setSelectedRange(NSRange(location: (text as NSString).length, length: 0))
    }
    editor.delegate = context.coordinator
    editor.onSend = isEnabled ? onSend : nil
    editor.onFiles = isEnabled ? onFiles : nil
    editor.onImage = isEnabled ? onImage : nil
    editor.onCancel = isEnabled ? onCancel : nil
    editor.focusOnMount = focusOnMount
    editor.onHeight = onHeight
    editor.onWindow = onWindow
    editor.setAccessibilityIdentifier(editorID)
    editor.setAccessibilityLabel(editorLabel)
    scroll.documentView = editor
    editor.provisionalDictation = provisionalDictation
    context.coordinator.scheduleDictation()
    return scroll
  }
  static func dismantleNSView(_ scroll: NSScrollView, coordinator: Coordinator) {
    coordinator.captureSelection()
    coordinator.editor?.delegate = nil
  }
  func updateNSView(_ scroll: NSScrollView, context: Context) {
    context.coordinator.parent = self
    guard let editor = scroll.documentView as? DraftTextView else { return }
    editor.isEditable = isEnabled
    editor.onSend = isEnabled ? onSend : nil
    editor.onFiles = isEnabled ? onFiles : nil
    editor.onImage = isEnabled ? onImage : nil
    editor.onCancel = isEnabled ? onCancel : nil
    editor.onHeight = onHeight
    editor.onWindow = onWindow
    if editor.string != text && !editor.hasMarkedText() {
      let selected = editor.selectedRange()
      editor.string = text
      editor.setSelectedRange(
        Self.clampedSelection(selected, length: (text as NSString).length))
      editor.measureHeight()
    }
    editor.provisionalDictation = provisionalDictation
    context.coordinator.scheduleDictation()
  }
  static func clampedSelection(_ selection: NSRange, length: Int) -> NSRange {
    let location = min(selection.location, length)
    return NSRange(location: location, length: min(selection.length, length - location))
  }
  static func composerHeight(_ measured: CGFloat) -> CGFloat { min(264, max(72, ceil(measured))) }
  @MainActor final class Coordinator: NSObject, NSTextViewDelegate {
    var parent: StudioDraftEditor
    private let edits = UndoManager()
    private var dictatedSession = ""
    private var dictatedIndex = -1
    private var dictationScheduled = false
    weak var editor: DraftTextView?
    init(_ parent: StudioDraftEditor) {
      self.parent = parent
      super.init()
      for name in [Notification.Name.NSUndoManagerDidUndoChange, .NSUndoManagerDidRedoChange] {
        NotificationCenter.default.addObserver(
          self, selector: #selector(historyChanged), name: name, object: edits)
      }
    }
    deinit { NotificationCenter.default.removeObserver(self) }
    // Inline edits and the composer share a window, not an undo history.
    func undoManager(for view: NSTextView) -> UndoManager? { edits }
    func captureSelection() {
      guard let editor, !editor.hasMarkedText() else { return }
      parent.position?.composer(editor.selectedRange(), text: editor.string)
    }
    func textViewDidChangeSelection(_ notification: Notification) { captureSelection() }
    @objc private func historyChanged(_ notification: Notification) {
      // TextKit 2 restores the text storage on undo without consistently
      // calling textDidChange. Keep the Swift draft in sync, or Send would
      // submit the pre-undo text even though the editor shows the correction.
      guard let editor else { return }
      parent.text = editor.string
      captureSelection()
      editor.measureHeight()
    }
    func textDidChange(_ notification: Notification) {
      if let editor = notification.object as? DraftTextView {
        parent.text = editor.string
        captureSelection()
        editor.measureHeight()
      }
      scheduleDictation()
    }
    func scheduleDictation() {
      guard !dictationScheduled, !parent.dictation.isEmpty else { return }
      dictationScheduled = true
      Task { @MainActor [weak self] in
        guard let self else { return }
        dictationScheduled = false
        guard let editor, !editor.hasMarkedText() else { return }
        if dictatedSession != parent.dictationSession {
          dictatedSession = parent.dictationSession
          dictatedIndex = -1
        }
        for item in parent.dictation where item.index > dictatedIndex {
          guard editor.appendDictated(item.text) else { break }
          dictatedIndex = item.index
          parent.onDictated?(dictatedSession, item.index)
        }
      }
    }
  }
}
@MainActor final class DraftTextView: NSTextView {
  var provisionalDictation = "" {
    didSet {
      if provisionalDictation != oldValue { refreshDictation() }
    }
  }
  private(set) var dictationOverlay: DraftDictationOverlay?
  /// A decoration can increase the scrollable document, but cannot move the
  /// caret or force somebody editing an earlier sentence to the bottom.
  private func refreshDictation() {
    let suffix = DraftDictationOverlay.suffix(after: string, provisional: provisionalDictation)
    if suffix.isEmpty {
      guard dictationOverlay != nil else { return }
      dictationOverlay?.removeFromSuperview()
      dictationOverlay = nil
      setAccessibilityHelp(nil)
      minSize.height = 0
      sizeToFit()
      // TextKit 2's sizeToFit may rewrite minSize to the viewport height.
      // Releasing a decoration must not retain that artificial minimum.
      minSize.height = 0
    } else {
      let overlay = dictationOverlay ?? DraftDictationOverlay(frame: .zero)
      if dictationOverlay == nil {
        dictationOverlay = overlay
        addSubview(overlay)
      }
      overlay.update(editor: self, provisional: provisionalDictation)
      if minSize.height != overlay.contentHeight || frame.height < overlay.contentHeight {
        sizeToFit()
        // The provisional words are outside text storage, so TextKit cannot
        // include them in sizeToFit. Keep them scrollable without committing
        // them to the draft or moving its selection/caret.
        minSize.height = overlay.contentHeight
        setFrameSize(NSSize(width: frame.width, height: max(frame.height, overlay.contentHeight)))
      }
      // AX value remains the confirmed draft, just like Copy and Send.
      setAccessibilityHelp("Provisional dictation: \(provisionalDictation)")
    }
    measureHeight()
  }
  /// Match web appendDictated: finalized words append, edits/caret stay put.
  /// Provisional words never enter storage, undo, or the submitted message.
  @discardableResult func appendDictated(_ text: String) -> Bool {
    guard isEditable, !hasMarkedText() else { return false }
    let said = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !said.isEmpty else { return true }
    let selection = selectedRange()
    let scrollOrigin = enclosingScrollView?.contentView.bounds.origin
    let end = (string as NSString).length
    let lead = string.last.map { $0.isWhitespace ? "" : " " } ?? ""
    breakUndoCoalescing()
    insertText(lead + said, replacementRange: NSRange(location: end, length: 0))
    breakUndoCoalescing()
    setSelectedRange(selection)
    refreshDictation()
    if let scrollOrigin, let scroll = enclosingScrollView {
      scroll.contentView.scroll(to: scrollOrigin)
      scroll.reflectScrolledClipView(scroll.contentView)
    }
    measureHeight()
    return true
  }
  var contentStorage: NSTextContentStorage?
  var onSend: (() -> Void)?
  var onFiles: (([URL]) -> Void)?
  var onImage: ((Data, Bool) -> Void)?
  var onCancel: (() -> Void)?
  var focusOnMount = false
  private var didFocus = false
  var onHeight: ((CGFloat) -> Void)?
  var onWindow: ((NSWindow?) -> Void)?
  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    onWindow?(window)
    if focusOnMount, !didFocus, window != nil {
      didFocus = true
      Task { @MainActor [weak self] in
        guard let self, let window, isEditable else { return }
        window.makeFirstResponder(self)
        setSelectedRange(NSRange(location: (string as NSString).length, length: 0))
        measureHeight()
      }
    }
  }
  private var lastHeight: CGFloat = 0
  private var measurementQueued = false
  override func setFrameSize(_ newSize: NSSize) {
    let changed = abs(newSize.width - frame.width) > 0.5
    // TextKit can resize again after refreshDictation returns. Its document
    // excludes the provisional decoration, so protect that extent on every
    // resize, not only the immediate sizeToFit call.
    super.setFrameSize(
      NSSize(
        width: newSize.width, height: max(newSize.height, dictationOverlay?.contentHeight ?? 0)))
    if changed {
      refreshDictation()
      measureHeight()
    }
  }
  override func didChangeText() {
    super.didChangeText()
    refreshDictation()
  }
  func measureHeight() {
    guard !measurementQueued else { return }
    measurementQueued = true
    // Coalesce layout and text changes, and never mutate SwiftUI state inside
    // updateNSView. Only lay out the bounded composer, not the transcript.
    Task { @MainActor [weak self] in
      guard let self else { return }
      refreshDictation()
      measurementQueued = false
      guard let layout = textLayoutManager else { return }
      layout.ensureLayout(for: CGRect(x: 0, y: 0, width: bounds.width, height: 280))
      let value = StudioDraftEditor.composerHeight(
        max(layout.usageBoundsForTextContainer.height, dictationOverlay?.contentHeight ?? 0) + 8)
      if value != lastHeight {
        lastHeight = value
        onHeight?(value)
      }
    }
  }
  override func keyDown(with event: NSEvent) {
    if event.keyCode == 53, !hasMarkedText(), let onCancel {
      onCancel()
      return
    }
    if event.keyCode == 36, onSend != nil, !event.modifierFlags.contains(.shift),
      !event.modifierFlags.contains(.option), !hasMarkedText()
    {
      onSend?()
      return
    }
    super.keyDown(with: event)
  }
  override func paste(_ sender: Any?) {
    if !acceptAttachment(NSPasteboard.general) { super.paste(sender) }
  }
  /// Shared paste/drop intake, testable with a private pasteboard. File URLs
  /// take precedence over their Finder preview so a PDF remains its original.
  @discardableResult func acceptAttachment(_ pasteboard: NSPasteboard) -> Bool {
    let urls =
      pasteboard.readObjects(
        forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
    if !urls.isEmpty, let onFiles {
      onFiles(urls)
    } else if let onImage, let png = pasteboard.data(forType: .png) {
      onImage(png, true)
    } else if let onImage, let tiff = pasteboard.data(forType: .tiff) {
      onImage(tiff, false)
    } else {
      return false
    }
    return true
  }
  override func performDragOperation(_ sender: any NSDraggingInfo) -> Bool {
    acceptAttachment(sender.draggingPasteboard) || super.performDragOperation(sender)
  }
}
