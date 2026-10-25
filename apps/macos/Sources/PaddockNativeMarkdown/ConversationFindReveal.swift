import AppKit

/// Only explicit Find navigation measures text. Never invoked by layout,
/// hit-testing or scrolling, and never changes a user's selectedRanges.
@MainActor final class ConversationFindReveal {
  private weak var host: ConversationSelectionHost?
  private weak var indicator: NSTextView?
  private weak var previousResponder: NSResponder?
  private var task: Task<Void, Never>?
  private var revision: Int?
  private var presented = false
  init(host: ConversationSelectionHost) { self.host = host }
  deinit { task?.cancel() }

  func cancel() {
    task?.cancel()
    task = nil
    indicator?.showFindIndicator(for: NSRange(location: 0, length: 0))
    indicator = nil
    revision = nil
  }
  func update(_ find: ConversationFind?) {
    let visible = find?.presented == true
    if visible != presented {
      if visible {
        previousResponder = host?.window?.firstResponder
      } else if let view = previousResponder as? NSView, let window = host?.window,
        view.window === window
      {
        window.makeFirstResponder(view)
      }
      presented = visible
    }
    guard let find, visible else {
      cancel()
      return
    }
    guard revision != find.navigationRevision else { return }
    cancel()
    revision = find.navigationRevision
    guard let match = find.current else { return }
    let revision = find.navigationRevision
    let query = find.query
    task = Task { [weak self, weak find] in
      // ScrollViewReader first reveals the row; its heavy Markdown mount may
      // then need several frames. Wait only for this target, not all rows.
      do {
        for _ in 0..<90 {
          try await Task.sleep(for: .milliseconds(33))
          guard let self, let host, host.window != nil else { return }
          if let target = host.findTarget(match, query: query),
            let rect = Self.textRect(target.view, range: target.range)
          {
            target.view.scrollRangeToVisible(target.range)
            // Nested code panes and Compare lanes have their own scroll views.
            // Reveal the actual text rect through every containing viewport.
            var parent = target.view.superview
            while let view = parent {
              if let scroll = view as? NSScrollView, let document = scroll.documentView {
                let converted = document.convert(rect, from: target.view)
                document.scrollToVisible(converted.insetBy(dx: -8, dy: -12))
              }
              if view === host { break }
              parent = view.superview
            }
            try await Task.sleep(for: .milliseconds(33))
            try Task.checkCancellation()
            guard target.view.visibleRect.intersects(rect) else { continue }
            target.view.showFindIndicator(for: target.range)
            indicator = target.view
            return
          }
        }
        find?.reportRevealFailure(revision: revision)
      } catch {}
    }
  }

  static func textRect(_ view: NSTextView, range: NSRange) -> NSRect? {
    guard range.length > 0, NSMaxRange(range) <= (view.string as NSString).length,
      view.bounds.width > 0, view.bounds.height > 0
    else { return nil }
    var rect: NSRect?
    if let layout = view.textLayoutManager, let content = layout.textContentManager {
      guard let start = content.location(content.documentRange.location, offsetBy: range.location),
        let end = content.location(start, offsetBy: range.length),
        let textRange = NSTextRange(location: start, end: end)
      else { return nil }
      layout.ensureLayout(for: textRange)
      layout.enumerateTextSegments(in: textRange, type: .standard, options: [.rangeNotRequired]) {
        _, frame, _, _ in
        rect = frame
        return false
      }
    } else if let layout = view.layoutManager, let container = view.textContainer {
      layout.ensureLayout(forCharacterRange: range)
      let glyphs = layout.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
      // The first line is sufficient, even when the match spans a huge block.
      layout.enumerateEnclosingRects(
        forGlyphRange: glyphs, withinSelectedGlyphRange: glyphs, in: container
      ) { frame, stop in
        rect = frame
        stop.pointee = true
      }
    }
    guard let rect, !rect.isEmpty else { return nil }
    return rect.offsetBy(dx: view.textContainerOrigin.x, dy: view.textContainerOrigin.y)
  }
}

extension ConversationSelectionHost {
  func findTarget(_ match: ConversationFind.Match, query: String) -> (
    view: NSTextView, range: NSRange
  )? {
    var occurrence = match.occurrence
    for map in maps(matching: match.itemID) {
      let ranges = (try? ConversationFindIndex.ranges(in: map.text as String, query: query)) ?? []
      guard occurrence < ranges.count else {
        occurrence -= ranges.count
        continue
      }
      let range = ranges[occurrence]
      // Embedded code/table text belongs to the attachment. Prefer its real
      // text view over highlighting the parent's U+FFFC placeholder.
      var cursor = 0
      for child in textDescendants(map.view) where child !== map.view && !child.string.isEmpty {
        // Restrict child lookup to attachments. Identical prose before a
        // fenced block must never be mistaken for that block's text view.
        let location = map.segments.lazy.filter(\.attachment).compactMap { segment -> NSRange? in
          let start = max(cursor, segment.plain.location)
          guard start < NSMaxRange(segment.plain) else { return nil }
          let range = map.text.range(
            of: child.string,
            range: NSRange(location: start, length: NSMaxRange(segment.plain) - start))
          return range.location == NSNotFound ? nil : range
        }.first
        guard let location else { continue }
        cursor = NSMaxRange(location)
        if NSIntersectionRange(range, location) == range {
          return (
            child, NSRange(location: range.location - location.location, length: range.length)
          )
        }
      }
      return (map.view, map.originalRange(range))
    }
    return nil
  }
}
