import AppKit
import PaddockNativeMarkdown
import SwiftUI

/// AppKit owns the shortcut even with the sidebar folded or WebKit focused.
/// The shared ChatView intentionally delegates application shortcuts to us.
public struct StudioHistoryCommands: Commands {
  @Bindable private var model: WorkspaceModel
  @FocusedValue(\.conversationFind) private var find
  public init(model: WorkspaceModel) { self.model = model }
  public var body: some Commands {
    CommandGroup(after: .textEditing) {
      Button("Find in Conversation…") {
        ConversationFindRouting.perform(.showFindInterface, find: find)
      }.keyboardShortcut("f", modifiers: .command)
        .disabled(find == nil || model.desktopTransition || model.quitting)
      Button("Find Next") { ConversationFindRouting.perform(.nextMatch, find: find) }
        .keyboardShortcut("g", modifiers: .command)
        .disabled(find == nil || model.desktopTransition || model.quitting)
      Button("Find Previous") { ConversationFindRouting.perform(.previousMatch, find: find) }
        .keyboardShortcut("g", modifiers: [.command, .shift])
        .disabled(find == nil || model.desktopTransition || model.quitting)
      Divider()
      Button("Search Chats") { model.searchConversations() }
        .keyboardShortcut("k", modifiers: .command)
        .disabled(model.navigation.mode != .studio || model.desktopTransition || model.quitting)
    }
  }
}
