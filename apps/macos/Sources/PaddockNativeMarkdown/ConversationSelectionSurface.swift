import AppKit
import SwiftUI

/// Stable logical order also covers messages outside a LazyVStack's mounted
/// viewport. Hidden reasoning is not copied unless its disclosure is mounted.
public struct ConversationSelectionItem: Equatable, Sendable {
  public enum Presentation: Sendable { case text, toolArguments, toolOutput }
  public let id: String
  public let text: String
  public let includeWhenUnmounted: Bool
  public let markdown: Bool
  public let presentation: Presentation
  public init(
    id: String, text: String, includeWhenUnmounted: Bool = true, markdown: Bool = false,
    presentation: Presentation = .text
  ) {
    self.id = id
    self.text = text
    self.includeWhenUnmounted = includeWhenUnmounted
    self.markdown = markdown
    self.presentation = presentation
  }
}

/// One AppKit selection responder for registered transcript text only. Native
/// safe-area bars may share its host, but composer/preview text is excluded.
public struct ConversationSelectionSurface<Content: View>: NSViewRepresentable {
  let items: [ConversationSelectionItem]
  let content: Content
  public init(items: [ConversationSelectionItem], @ViewBuilder content: () -> Content) {
    self.items = items
    self.content = content()
  }
  public func makeNSView(context: Context) -> ConversationSelectionHost {
    let view = ConversationSelectionHost(
      rootView: AnyView(content.environment(\.self, context.environment)))
    // The owning column places this viewport below its window header. Do not
    // inherit that safe area again inside this nested hosting boundary.
    view.safeAreaRegions = []
    // This host owns a viewport, not a document-sized window. Its parent
    // supplies both dimensions through sizeThatFits; do not add competing
    // minimum/maximum/intrinsic constraints for the scrolling contents.
    view.sizingOptions = []
    return view
  }
  public func updateNSView(_ view: ConversationSelectionHost, context: Context) {
    #if DEBUG
      view.contentUpdates += 1
    #endif
    view.items = items
    view.rootView = AnyView(content.environment(\.self, context.environment))
    view.findReveal.update(context.environment.conversationFind)
  }
  public func sizeThatFits(
    _ proposal: ProposedViewSize, nsView: ConversationSelectionHost, context: Context
  ) -> CGSize? {
    guard let width = proposal.width, let height = proposal.height else { return nil }
    return CGSize(width: width, height: height)
  }
}

@MainActor public final class ConversationSelectionHost: NSHostingView<AnyView> {
  #if DEBUG
    var contentUpdates = 0
  #endif
  var items: [ConversationSelectionItem] = []
  lazy var findReveal = ConversationFindReveal(host: self)
  private var anchor: Point?
  private var extent: Point?
  private var selectedText = ""
  private var copyPreparation: Task<String, Never>?
  private var selectionEpoch = 0
  private var eventMonitor: Any?
  private var dragTimer: Timer?
  private var dragEvent: NSEvent?
  private(set) var selecting = false
  private struct Point {
    let view: NSTextView
    let offset: Int
    let id: String
  }
  @MainActor struct Map {
    struct Segment {
      let original: NSRange
      let plain: NSRange
      let attachment: Bool
    }
    let view: NSTextView
    let id: String
    let text: NSString
    let segments: [Segment]
    init(_ view: NSTextView, idOverride: String? = nil) {
      self.view = view
      var parent: NSView? = view
      var id = ""
      while id.isEmpty, let next = parent?.superview {
        parent = next
        if next is SelectionHostingView { id = next.identifier?.rawValue ?? "" }
      }
      self.id = idOverride ?? (id.isEmpty ? view.identifier?.rawValue ?? "" : id)
      let storage = view.textStorage ?? NSTextStorage(string: view.string)
      let output = NSMutableString()
      let full = NSRange(location: 0, length: storage.length)
      var segments: [Segment] = []
      storage.enumerateAttribute(.attachment, in: full) { attachment, range, _ in
        let value =
          attachment == nil
          ? (storage.string as NSString).substring(with: range)
          : view.attributedSubstring(forProposedRange: range, actualRange: nil)?.string ?? ""
        segments.append(
          Segment(
            original: range,
            plain: NSRange(location: output.length, length: (value as NSString).length),
            attachment: attachment != nil))
        output.append(value)
      }
      text = output.copy() as! NSString
      self.segments = segments
    }
    func offset(_ original: Int) -> Int {
      for s in segments where original < NSMaxRange(s.original) {
        if s.original.length == s.plain.length {
          return s.plain.location + max(0, original - s.original.location)
        }
        return s.plain.location
      }
      return text.length
    }
    func originalRange(_ plain: NSRange) -> NSRange {
      var start = Int.max
      var end = 0
      for s in segments {
        let part = NSIntersectionRange(plain, s.plain)
        guard part.length > 0 else { continue }
        if s.original.length == s.plain.length {
          start = min(start, s.original.location + part.location - s.plain.location)
          end = max(end, s.original.location + NSMaxRange(part) - s.plain.location)
        } else {
          start = min(start, s.original.location)
          end = max(end, NSMaxRange(s.original))
        }
      }
      return start == Int.max
        ? NSRange(location: 0, length: 0) : NSRange(location: start, length: end - start)
    }
  }
  public override var acceptsFirstResponder: Bool { true }
  public override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    if let eventMonitor {
      NSEvent.removeMonitor(eventMonitor)
      self.eventMonitor = nil
    }
    dragTimer?.invalidate()
    dragTimer = nil
    guard window != nil else {
      findReveal.cancel()
      copyPreparation?.cancel()
      copyPreparation = nil
      selectionEpoch += 1
      anchor = nil
      extent = nil
      selectedText = ""
      dragEvent = nil
      selecting = false
      return
    }
    eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
      let handled = MainActor.assumeIsolated {
        guard let self, event.window === self.window,
          event.modifierFlags.intersection(.deviceIndependentFlagsMask) == .command,
          let responder = self.window?.firstResponder as? NSView,
          responder === self || responder.isDescendant(of: self),
          (responder as? NSTextView)?.isEditable != true
        else { return false }
        if let text = responder as? NSTextView, !self.ownsText(text) { return false }
        if event.charactersIgnoringModifiers == "a" {
          self.selectAll(nil)
          return true
        }
        if event.charactersIgnoringModifiers == "c", responder === self, !self.selectedText.isEmpty
        {
          self.copy(nil)
          return true
        }
        return false
      }
      return handled ? nil : event
    }
  }
  public override func hitTest(_ point: NSPoint) -> NSView? {
    hitTest(point, event: NSApp.currentEvent)
  }
  func hitTest(_ point: NSPoint, event: NSEvent?) -> NSView? {
    let hit = super.hitTest(point)
    // AppKit also hit-tests during hover/layout, and currentEvent may still
    // be a mouse press then. Never ask TextKit for glyphs or insertion points
    // here: table attachments can re-enter SwiftUI's constraint pass.
    guard let event,
      event.type == .leftMouseDown || event.type == .leftMouseDragged
        || event.type == .rightMouseDown
    else { return hit }
    // Leave buttons, scrollbars, links, editable controls and context menus to
    // their actual owners. Only text drags belong to this responder.
    guard let text = hit as? NSTextView, text.isSelectable, !text.isEditable, ownsText(text) else {
      return hit
    }
    // Speech uses TextKit 2 and timed-word links. Never force its layout into
    // TextKit 1 or replace its word activation/selection behavior.
    if text.textLayoutManager != nil { return hit }
    if event.type == .rightMouseDown {
      return selectedText.isEmpty ? hit : self
    }
    return self
  }
  public override func mouseDown(with event: NSEvent) {
    // SwiftUI controls can deliver their event to this hosting responder even
    // when hitTest correctly returned a descendant. Only take text presses;
    // otherwise NSHostingView must receive the event to run its button action.
    let local = superview?.convert(event.locationInWindow, from: nil) ?? event.locationInWindow
    let hit = super.hitTest(local)
    guard let text = hit as? NSTextView, text.isSelectable, !text.isEditable,
      text.textLayoutManager == nil, ownsText(text)
    else {
      selecting = false
      super.mouseDown(with: event)
      return
    }
    let index = Self.insertionIndex(text, at: text.convert(event.locationInWindow, from: nil))
    if index < (text.string as NSString).length,
      text.textStorage?.attribute(.link, at: index, effectiveRange: nil) != nil
    {
      selecting = false
      // Resolve links only while handling the actual press, never in hitTest.
      // Forward to the real text owner so its delegate/openURL policy handles
      // activation exactly as it would without conversation-wide selection.
      text.mouseDown(with: event)
      return
    }
    guard let point = point(at: event.locationInWindow) else { return }
    selecting = true
    window?.makeFirstResponder(self)
    copyPreparation?.cancel()
    copyPreparation = nil
    selectionEpoch += 1
    if !event.modifierFlags.contains(.shift) || anchor == nil { anchor = point }
    extent = point
    if event.clickCount >= 2 {
      let map = Map(point.view)
      if map.text.length > 0 {
        let position = min(point.offset, map.text.length - 1)
        let range =
          event.clickCount >= 3
          ? map.text.paragraphRange(for: NSRange(location: position, length: 0))
          : Self.wordRange(map.text, at: position)
        anchor = Point(view: point.view, offset: range.location, id: map.id)
        extent = Point(view: point.view, offset: NSMaxRange(range), id: map.id)
      }
    }
    updateSelection()
  }
  public override func mouseDragged(with event: NSEvent) {
    guard selecting else {
      super.mouseDragged(with: event)
      return
    }
    dragEvent = event
    extendSelection(event)
    if dragTimer == nil {
      let timer = Timer(
        timeInterval: 0.05, target: self, selector: #selector(dragTick), userInfo: nil,
        repeats: true)
      dragTimer = timer
      RunLoop.main.add(timer, forMode: .common)
    }
  }
  private func extendSelection(_ event: NSEvent) {
    _ = anchor?.view.enclosingScrollView?.autoscroll(with: event)
    if let point = point(at: event.locationInWindow) {
      extent = point
      updateSelection()
    }
  }
  @objc private func dragTick() {
    if let event = dragEvent { extendSelection(event) }
  }
  public override func mouseUp(with event: NSEvent) {
    guard selecting else {
      super.mouseUp(with: event)
      return
    }
    selecting = false
    dragTimer?.invalidate()
    dragTimer = nil
    dragEvent = nil
    if event.clickCount == 1, let point = point(at: event.locationInWindow) {
      extent = point
      updateSelection()
    }
  }
  public override func selectAll(_ sender: Any?) {
    window?.makeFirstResponder(self)
    anchor = nil
    extent = nil
    let maps = maps()
    var byID: [String: String] = [:]
    for map in maps {
      highlight(map, range: NSRange(location: 0, length: map.text.length))
      if !map.id.isEmpty { byID[map.id] = map.text as String }
    }
    let ordered = items.compactMap { item -> (String, Bool)? in
      if let text = byID[item.id] { return (text, false) }
      return item.includeWhenUnmounted ? (item.text, item.markdown) : nil
    }
    prepareCopy(ordered.isEmpty ? maps.map { ($0.text as String, false) } : ordered)
  }
  @objc public func copy(_ sender: Any?) {
    guard !selectedText.isEmpty else { return }
    if let preparation = copyPreparation {
      let epoch = selectionEpoch
      let clipboard = NSPasteboard.general.changeCount
      Task { @MainActor [weak self] in
        let text = await preparation.value
        guard let self, self.selectionEpoch == epoch, NSPasteboard.general.changeCount == clipboard
        else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
      }
      return
    }
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(selectedText, forType: .string)
  }
  public override func keyDown(with event: NSEvent) {
    if event.keyCode == 53 {
      clearSelection()
      return
    }
    super.keyDown(with: event)
  }
  public override func menu(for event: NSEvent) -> NSMenu? {
    guard !selectedText.isEmpty else { return super.menu(for: event) }
    let menu = NSMenu()
    let copy = menu.addItem(withTitle: "Copy", action: #selector(copy(_:)), keyEquivalent: "")
    copy.target = self
    let all = menu.addItem(
      withTitle: "Select All", action: #selector(selectAll(_:)), keyEquivalent: "")
    all.target = self
    return menu
  }
  func clearSelection() {
    copyPreparation?.cancel()
    copyPreparation = nil
    selectionEpoch += 1
    anchor = nil
    extent = nil
    selectedText = ""
    for map in maps() { highlight(map, range: NSRange(location: 0, length: 0)) }
  }
  private func ownsText(_ view: NSView) -> Bool {
    // Read identifiers only: hit testing must never trigger TextKit layout.
    // Unregistered labels/audio previews in native bars keep native selection.
    var ancestor: NSView? = view
    while let current = ancestor, current !== self {
      if let id = current.identifier?.rawValue, !id.isEmpty,
        items.contains(where: { $0.id == id })
      {
        return true
      }
      ancestor = current.superview
    }
    return false
  }

  func maps(matching id: String? = nil) -> [Map] {
    func roots(_ view: NSView) -> [NSTextView] {
      guard !view.isHidden else { return [] }
      if let text = view as? NSTextView, text.isSelectable, !text.isEditable {
        return ownsText(text) ? [text] : []
      }
      return view.subviews.flatMap(roots)
    }
    let order = Dictionary(
      items.enumerated().map { ($0.element.id, $0.offset) }, uniquingKeysWith: { a, _ in a })
    return roots(self).filter { text in
      guard let id else { return true }
      var ancestor: NSView? = text
      while let view = ancestor, view !== self {
        if view.identifier?.rawValue == id { return true }
        ancestor = view.superview
      }
      return false
    }.map { Map($0) }.sorted {
      (order[$0.id] ?? Int.max) < (order[$1.id] ?? Int.max)
    }
  }
  private func point(at windowPoint: NSPoint) -> Point? {
    let maps = maps()
    guard !maps.isEmpty else { return nil }
    let point = convert(windowPoint, from: nil)
    func distance(_ map: Map) -> CGFloat {
      let r = map.view.convert(map.view.bounds, to: self)
      let dx = max(r.minX - point.x, 0, point.x - r.maxX)
      let dy = max(r.minY - point.y, 0, point.y - r.maxY)
      return dx * dx + dy * dy
    }
    guard let map = maps.min(by: { distance($0) < distance($1) }) else { return nil }
    let local = map.view.convert(windowPoint, from: nil)
    let index = Self.insertionIndex(map.view, at: local)
    // Code/table attachments contain their own text surfaces. Resolve a click
    // inside one against its expanded copy text, not the U+FFFC placeholder.
    for child in textDescendants(map.view) where child !== map.view {
      let p = child.convert(windowPoint, from: nil)
      guard child.bounds.contains(p), !child.string.isEmpty else { continue }
      let candidate = map.text.range(of: child.string)
      if candidate.location != NSNotFound {
        return Point(
          view: map.view,
          offset: candidate.location + min(Self.insertionIndex(child, at: p), candidate.length),
          id: map.id)
      }
    }
    return Point(view: map.view, offset: map.offset(index), id: map.id)
  }
  private func updateSelection() {
    guard let anchor, let extent else { return }
    var maps = maps()
    // Retain the drag's anchor even if lazy scrolling unmounts that message.
    for endpoint in [anchor, extent] where !maps.contains(where: { $0.view === endpoint.view }) {
      // SwiftUI can discard the hosting ancestor that carried this ID while
      // the drag still retains its text view. Keep the logical identity too.
      maps.append(Map(endpoint.view, idOverride: endpoint.id))
    }
    let order = Dictionary(
      items.enumerated().map { ($0.element.id, $0.offset) }, uniquingKeysWith: { a, _ in a })
    maps.sort { (order[$0.id] ?? Int.max) < (order[$1.id] ?? Int.max) }
    guard let a = maps.firstIndex(where: { $0.view === anchor.view }),
      let b = maps.firstIndex(where: { $0.view === extent.view })
    else { return }
    let forward = a < b || a == b && anchor.offset <= extent.offset
    let start = forward ? anchor : extent
    let end = forward ? extent : anchor
    let first = min(a, b)
    let last = max(a, b)
    var text: [String] = []
    var selected: [String: String] = [:]
    for (i, map) in maps.enumerated() {
      let lo = i == first ? min(start.offset, map.text.length) : 0
      let hi = i == last ? min(end.offset, map.text.length) : map.text.length
      let range =
        (first...last).contains(i)
        ? NSRange(location: lo, length: max(0, hi - lo)) : NSRange(location: 0, length: 0)
      highlight(map, range: range)
      if range.length > 0 {
        let value = map.text.substring(with: range)
        text.append(value)
        selected[map.id] = value
      }
    }
    if let lo = order[maps[first].id], let hi = order[maps[last].id] {
      prepareCopy(
        items[lo...hi].compactMap { item in
          if let value = selected[item.id] { return (value, false) }
          // Only complete interior messages can be supplied from virtualized
          // history. Never add an unselected endpoint or a folded disclosure.
          guard item.id != maps[first].id, item.id != maps[last].id, item.includeWhenUnmounted
          else { return nil }
          return (item.text, item.markdown)
        })
    } else {
      prepareCopy(text.map { ($0, false) })
    }
  }
  private func highlight(_ map: Map, range: NSRange) {
    map.view.setSelectedRange(map.originalRange(range))
    for child in textDescendants(map.view) where child !== map.view && !child.string.isEmpty {
      let location = map.text.range(of: child.string)
      guard location.location != NSNotFound else { continue }
      let part = NSIntersectionRange(range, location)
      child.setSelectedRange(
        NSRange(
          location: part.length > 0 ? part.location - location.location : 0, length: part.length))
    }
  }
  func textDescendants(_ root: NSView) -> [NSTextView] {
    ((root as? NSTextView).map { [$0] } ?? []) + root.subviews.flatMap(textDescendants)
  }
  private static func wordRange(_ text: NSString, at position: Int) -> NSRange {
    let delimiters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_")).inverted
    let before = text.rangeOfCharacter(
      from: delimiters, options: .backwards, range: NSRange(location: 0, length: position))
    let after = text.rangeOfCharacter(
      from: delimiters, range: NSRange(location: position, length: text.length - position))
    let start = before.location == NSNotFound ? 0 : NSMaxRange(before)
    let end = after.location == NSNotFound ? text.length : after.location
    return end > start
      ? NSRange(location: start, length: end - start)
      : text.rangeOfComposedCharacterSequence(at: position)
  }
  private static func insertionIndex(_ view: NSTextView, at local: NSPoint) -> Int {
    if view.textLayoutManager != nil { return view.characterIndexForInsertion(at: local) }
    guard let manager = view.layoutManager, let container = view.textContainer else { return 0 }
    let text = view.string as NSString
    guard text.length > 0 else { return 0 }
    let p = NSPoint(
      x: local.x - view.textContainerOrigin.x, y: local.y - view.textContainerOrigin.y)
    manager.ensureLayout(for: container)
    if p.y < 0 { return 0 }
    if p.y > manager.usedRect(for: container).maxY { return text.length }
    var fraction: CGFloat = 0
    let index = min(
      text.length,
      manager.characterIndex(
        for: p, in: container, fractionOfDistanceBetweenInsertionPoints: &fraction))
    if fraction > 0.5, index < text.length {
      return NSMaxRange(text.rangeOfComposedCharacterSequence(at: index))
    }
    return index
  }
  private func prepareCopy(_ parts: [(String, Bool)]) {
    copyPreparation?.cancel()
    copyPreparation = nil
    selectionEpoch += 1
    selectedText = parts.map(\.0).filter { !$0.isEmpty }.joined(separator: "\n\n")
    guard parts.contains(where: { $0.1 }) else { return }
    // Only unmounted Markdown requires parsing. Do that on demand off-main,
    // not on every streamed token, and snapshot the user's selected content.
    copyPreparation = Task.detached(priority: .userInitiated) {
      var text: [String] = []
      for (source, markdown) in parts {
        if Task.isCancelled { return "" }
        let value =
          markdown
          ? (try? AttributedString(markdown: NativeMarkdownPolicy.source(source))).map {
            String($0.characters)
          } ?? source : source
        if !value.isEmpty { text.append(value) }
      }
      return text.joined(separator: "\n\n")
    }
  }
}
