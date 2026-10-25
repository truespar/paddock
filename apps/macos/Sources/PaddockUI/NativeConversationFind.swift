import AppKit
import PaddockNativeMarkdown
import SwiftUI

private struct ConversationFindKey: FocusedValueKey {
  typealias Value = ConversationFind
}
extension FocusedValues {
  var conversationFind: ConversationFind? {
    get { self[ConversationFindKey.self] }
    set { self[ConversationFindKey.self] = newValue }
  }
}

struct NativeConversationFindBar: View {
  let find: ConversationFind
  var body: some View {
    HStack(spacing: 8) {
      NativeSearchField(
        text: Binding(get: { find.query }, set: { find.search($0) }),
        prompt: "Find in conversation", focusOnAppear: true, cancelClearsText: false,
        onMove: { find.move($0) },
        onSubmit: {
          find.move(NSApp.currentEvent?.modifierFlags.contains(.shift) == true ? -1 : 1)
        },
        onCancel: { find.close() }
      ).id(find.focusRevision).frame(minWidth: 80, maxWidth: 320)
        .accessibilityIdentifier("conversation-find-field")
      Text(status).font(.system(size: 12)).monospacedDigit().foregroundStyle(.secondary)
        .lineLimit(1).accessibilityIdentifier("conversation-find-count")
      Spacer(minLength: 0)
      Button("Previous match", systemImage: "chevron.up") { find.move(-1) }
        .help("Previous match (⇧⌘G)").disabled(find.matches.isEmpty)
      Button("Next match", systemImage: "chevron.down") { find.move(1) }
        .help("Next match (⌘G)").disabled(find.matches.isEmpty)
      Button("Close Find", systemImage: "xmark") { find.close() }.help("Close Find (Esc)")
    }
    .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
    .padding(.horizontal, 20).padding(.vertical, 6)
    .background(PaddockStyle.canvas)
    .accessibilityElement(children: .contain)
    .accessibilityIdentifier("conversation-find-bar")
    .help(find.revealFailed ? "The matching text could not be displayed. Try the next match." : "")
  }
  private var status: String {
    if find.query.isEmpty { return "" }
    if find.searching && find.matches.isEmpty { return "Searching…" }
    if find.matches.isEmpty { return "No matches" }
    if find.revealFailed { return "Match unavailable" }
    return "\(find.position + 1) of \(find.matches.count)\(find.truncated ? "+" : "")"
  }
}

@MainActor enum ConversationFindRouting {
  static func perform(_ action: NSTextFinder.Action, find: ConversationFind?) {
    // Menu activation must respect the same ownership as the editor's key
    // equivalents; searching source is independent of transcript Find.
    if let editor = NSApp.keyWindow?.firstResponder as? ArtifactCodeTextView {
      let item = NSMenuItem()
      item.tag = action.rawValue
      editor.performFindPanelAction(item)
      return
    }
    switch action {
    case .showFindInterface: find?.show()
    case .nextMatch: find?.move(1)
    case .previousMatch: find?.move(-1)
    default: break
    }
  }
}
