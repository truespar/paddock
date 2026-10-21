import AppKit
import SwiftUI

/// A field-local command delegate, not a window/global key monitor. Input
/// methods retain arrows and Return while composing marked text.
struct NativeSearchField: NSViewRepresentable {
  @Binding var text: String
  var prompt: String
  var editorOnly = false
  var focusOnAppear = false
  var onMove: ((Int) -> Void)?
  var onSubmit: (() -> Void)?
  var onCancel: (() -> Void)?

  func makeCoordinator() -> Coordinator { Coordinator(self) }
  func makeNSView(context: Context) -> Field {
    let field = Field()
    field.editorOnly = editorOnly
    field.editor.delegate = context.coordinator
    field.clearButton.target = context.coordinator
    field.clearButton.action = #selector(Coordinator.clearSearch(_:))
    field.focusOnAppear = focusOnAppear
    field.setContentHuggingPriority(.defaultLow, for: .horizontal)
    return field
  }
  func updateNSView(_ field: Field, context: Context) {
    context.coordinator.parent = self
    field.editorOnly = editorOnly
    field.editor.placeholderString = prompt
    field.editor.setAccessibilityLabel(prompt)
    field.editor.isEnabled = context.environment.isEnabled
    field.clearButton.isEnabled = context.environment.isEnabled
    field.layer?.backgroundColor =
      editorOnly
      ? nil
      : PaddockStyle.nsColor(
        "elevated", dark: context.environment.colorScheme == .dark,
        increasedContrast: context.environment.colorSchemeContrast == .increased
      ).cgColor
    // Never replace an IME's in-progress editor string during a publication.
    if (field.editor.currentEditor() as? NSTextView)?.hasMarkedText() != true,
      field.editor.stringValue != text
    {
      field.editor.stringValue = text
    }
    field.refreshState()
  }
  func sizeThatFits(_ proposal: ProposedViewSize, nsView: Field, context: Context) -> CGSize? {
    CGSize(width: proposal.width ?? 200, height: nsView.intrinsicContentSize.height)
  }
  static func dismantleNSView(_ view: Field, coordinator: Coordinator) {
    view.editor.delegate = nil
    view.clearButton.target = nil
  }

  /// Keep the normal AppKit field editor at its intrinsic height. NSSearchField
  /// uses different display/editing insets when its system bezel is removed;
  /// overriding its display rect alone does not fix the active field editor.
  final class Field: NSView {
    let editor = Input()
    let clearButton = NSButton()
    private let searchIcon = NSImageView()
    var focusOnAppear = false
    var isEditing = false
    var editorOnly = false {
      didSet {
        guard editorOnly != oldValue else { return }
        textLeading.constant = editorOnly ? 0 : 26
        textTrailing.constant = editorOnly ? 0 : -28
        searchIcon.isHidden = editorOnly
        layer?.cornerRadius = editorOnly ? 0 : 14
        invalidateIntrinsicContentSize()
        refreshState()
      }
    }
    private lazy var textLeading = editor.leadingAnchor.constraint(
      equalTo: leadingAnchor, constant: 26)
    private lazy var textTrailing = editor.trailingAnchor.constraint(
      equalTo: trailingAnchor, constant: -28)
    override init(frame frameRect: NSRect) {
      super.init(frame: frameRect)
      wantsLayer = true
      layer?.cornerRadius = 14
      editor.font = .systemFont(ofSize: 13)
      editor.isBezeled = false
      editor.isBordered = false
      editor.drawsBackground = false
      editor.focusRingType = .none
      editor.usesSingleLineMode = true
      editor.cell?.isScrollable = true
      editor.setAccessibilitySubrole(.searchField)
      editor.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
      searchIcon.image = NSImage(systemSymbolName: "magnifyingglass", accessibilityDescription: nil)
      searchIcon.contentTintColor = .secondaryLabelColor
      searchIcon.setAccessibilityElement(false)
      clearButton.image = NSImage(
        systemSymbolName: "xmark.circle.fill", accessibilityDescription: nil)
      clearButton.imagePosition = .imageOnly
      clearButton.isBordered = false
      clearButton.contentTintColor = .secondaryLabelColor
      clearButton.setAccessibilityLabel("Clear search")
      clearButton.toolTip = "Clear search"
      clearButton.isHidden = true
      for child in [searchIcon, editor, clearButton] {
        child.translatesAutoresizingMaskIntoConstraints = false
        addSubview(child)
      }
      NSLayoutConstraint.activate([
        searchIcon.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
        searchIcon.centerYAnchor.constraint(equalTo: centerYAnchor),
        searchIcon.widthAnchor.constraint(equalToConstant: 14),
        searchIcon.heightAnchor.constraint(equalToConstant: 14),
        textLeading,
        textTrailing,
        editor.centerYAnchor.constraint(equalTo: centerYAnchor),
        editor.heightAnchor.constraint(equalToConstant: editor.intrinsicContentSize.height),
        clearButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -6),
        clearButton.centerYAnchor.constraint(equalTo: centerYAnchor),
        clearButton.widthAnchor.constraint(equalToConstant: 18),
        clearButton.heightAnchor.constraint(equalToConstant: 18),
      ])
    }
    required init?(coder: NSCoder) { nil }
    override var intrinsicContentSize: NSSize {
      NSSize(width: 200, height: editorOnly ? editor.intrinsicContentSize.height : 28)
    }
    override func mouseDown(with event: NSEvent) {
      guard editor.isEnabled else { return }
      window?.makeFirstResponder(editor)
    }
    func refreshState() {
      clearButton.isHidden = editorOnly || editor.stringValue.isEmpty
      layer?.borderWidth = isEditing && !editorOnly ? 1 : 0
      effectiveAppearance.performAsCurrentDrawingAppearance {
        layer?.borderColor = NSColor.secondaryLabelColor.cgColor
      }
    }
    override func viewDidMoveToWindow() {
      super.viewDidMoveToWindow()
      guard focusOnAppear, window != nil else { return }
      focusOnAppear = false
      Task { @MainActor [weak self] in
        guard let self, let window = self.window else { return }
        window.makeFirstResponder(self.editor)
      }
    }
  }
  final class Input: NSTextField {
    override func becomeFirstResponder() -> Bool {
      let accepted = super.becomeFirstResponder()
      if accepted, let field = superview as? Field {
        field.isEditing = true
        field.refreshState()
      }
      return accepted
    }
  }
  @MainActor final class Coordinator: NSObject, NSTextFieldDelegate {
    var parent: NativeSearchField
    init(_ parent: NativeSearchField) { self.parent = parent }
    func controlTextDidChange(_ notification: Notification) {
      guard let field = notification.object as? NSTextField else { return }
      (field.superview as? Field)?.refreshState()
      parent.text = field.stringValue
    }
    func controlTextDidBeginEditing(_ notification: Notification) {
      guard let field = (notification.object as? NSTextField)?.superview as? Field else { return }
      field.isEditing = true
      field.refreshState()
    }
    func controlTextDidEndEditing(_ notification: Notification) {
      guard let field = (notification.object as? NSTextField)?.superview as? Field else { return }
      field.isEditing = false
      field.refreshState()
    }
    @objc func clearSearch(_ sender: NSButton) {
      guard let field = sender.superview as? Field else { return }
      field.editor.stringValue = ""
      parent.text = ""
      field.window?.makeFirstResponder(field.editor)
      field.refreshState()
    }
    func control(_ control: NSControl, textView: NSTextView, doCommandBy command: Selector) -> Bool
    {
      guard !textView.hasMarkedText() else { return false }
      switch command {
      case #selector(NSResponder.moveDown(_:)):
        guard let move = parent.onMove else { return false }
        move(1)
      case #selector(NSResponder.moveUp(_:)):
        guard let move = parent.onMove else { return false }
        move(-1)
      case #selector(NSResponder.insertNewline(_:)):
        guard let submit = parent.onSubmit else { return false }
        submit()
      case #selector(NSResponder.cancelOperation(_:)):
        if !parent.text.isEmpty {
          parent.text = ""
        } else {
          guard let cancel = parent.onCancel else { return false }
          cancel()
        }
      default: return false
      }
      return true
    }
  }
}

enum SearchNavigation {
  static func moved<ID: Equatable>(_ current: ID?, in ids: [ID], by delta: Int) -> ID? {
    guard !ids.isEmpty else { return nil }
    guard let current, let index = ids.firstIndex(of: current) else {
      return delta < 0 ? ids.last : ids.first
    }
    return ids[min(ids.count - 1, max(0, index + delta))]
  }
}
