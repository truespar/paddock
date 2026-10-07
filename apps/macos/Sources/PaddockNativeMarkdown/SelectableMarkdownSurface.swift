import AppKit
import PaddockDesign
import SwiftUI

/// Own the hosting boundary so a library text-storage replacement cannot move
/// an existing selection to the end of a streaming response. No global view
/// introspection, delegate replacement or modifications to dependency sources.
struct SelectableMarkdownSurface<Content: View>: NSViewRepresentable {
  let content: Content
  var unsureWords: [String]

  init(unsureWords: [String] = [], @ViewBuilder content: () -> Content) {
    self.content = content()
    self.unsureWords = unsureWords
  }

  func makeNSView(context: Context) -> SelectionHostingView {
    let view = SelectionHostingView(
      rootView: AnyView(content.environment(\.self, context.environment)))
    context.environment.nativeMarkdownWork?.surface = view
    return view
  }

  func updateNSView(_ view: SelectionHostingView, context: Context) {
    context.environment.nativeMarkdownWork?.surface = view
    #if DEBUG
      view.contentUpdates += 1
    #endif
    view.identifier = .init(context.environment.conversationTextID)
    view.confidence.words = unsureWords
    view.preserveSelectionForUpdate()
    view.rootView = AnyView(content.environment(\.self, context.environment))
    view.needsLayout = true
  }
}

final class SelectionHostingView: PaddockScrollHostingView {
  #if DEBUG
    var contentUpdates = 0
  #endif
  let confidence = NativeConfidenceDecoration()
  private struct Selection {
    weak var view: NSTextView?
    let text: String
    let ranges: [NSValue]
  }
  private var pending: Selection?

  /// Call outside layout(), immediately before viewport eviction. Inline
  /// SwiftUI hosts invalidate TextKit asynchronously; their last published
  /// intrinsic size can still omit a newly installed diagram under load.
  /// Complete the active TextKit engine's layout instead of caching that stale
  /// size. Never force a TextKit 2 view into TextKit 1 compatibility mode.
  func completedTextHeight() -> CGFloat? {
    guard let text = textSurface(in: self), text.bounds.width > 0 else { return nil }
    let maximumY: CGFloat
    if let manager = text.textLayoutManager {
      manager.invalidateLayout(for: manager.documentRange)
      manager.ensureLayout(for: manager.documentRange)
      var bottom: CGFloat = 0
      manager.enumerateTextSegments(
        in: manager.documentRange, type: .standard, options: [.rangeNotRequired]
      ) { _, frame, _, _ in
        bottom = max(bottom, frame.maxY)
        return true
      }
      maximumY = bottom
    } else if let manager = text.layoutManager, let container = text.textContainer {
      // MarkdownView currently selects TextKit 1 on macOS. This is an already
      // legacy view, not a fallback triggered by querying layoutManager first.
      manager.invalidateLayout(
        forCharacterRange: NSRange(location: 0, length: text.string.utf16.count),
        actualCharacterRange: nil)
      manager.ensureLayout(for: container)
      maximumY = manager.usedRect(for: container).maxY
    } else {
      return nil
    }
    let height = ceil(maximumY + text.textContainerOrigin.y)
    return height.isFinite && height >= 0 ? height : nil
  }

  func preserveSelectionForUpdate() {
    guard pending == nil, let text = textSurface(in: self),
      text.selectedRanges.contains(where: { $0.rangeValue.length > 0 })
    else { return }
    pending = Selection(view: text, text: text.string, ranges: text.selectedRanges)
  }

  override func layout() {
    super.layout()
    confidence.reconcile(self)
    guard let selection = pending else { return }
    pending = nil
    guard let view = selection.view else { return }
    // Restore only unchanged selected text. If an edit invalidates the selected
    // region, leave AppKit's selection alone instead of selecting different text.
    let next = view.string as NSString
    let previous = selection.text as NSString
    guard
      selection.ranges.allSatisfy({ value in
        let range = value.rangeValue
        return NSMaxRange(range) <= next.length
          && next.substring(with: range) == previous.substring(with: range)
      })
    else { return }
    view.selectedRanges = selection.ranges
  }

  private func textSurface(in view: NSView) -> NSTextView? {
    if let text = view as? NSTextView { return text }
    return view.subviews.lazy.compactMap { self.textSurface(in: $0) }.first
  }
}
