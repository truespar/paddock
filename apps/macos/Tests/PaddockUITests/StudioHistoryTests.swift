import AppKit
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockConversationCore
@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native conversation management", .serialized) @MainActor
struct StudioHistoryTests {
  @Test func selectingAConversationLeavesReadsBeforeAnyPresentationArrives() async {
    let workspace = WorkspaceModel(client: HistoryNoCore())
    workspace.navigation.studio = .reads
    workspace.reads.draft.state = "Keep my read"
    let sidebar = StudioConversationSidebar(
      chat: workspace.chat,
      onNewChat: {}, onFold: {}, onOpen: { workspace.navigation.returnToChat() })
    sidebar.openConversation("saved-chat")
    #expect(workspace.navigation.studio.isConversation)
    #expect(workspace.reads.draft.state == "Keep my read")
    // No runtime was started: navigation cannot depend on a later chat snapshot.
    #expect(workspace.chat.conversation == nil)
    await workspace.chat.shutdown()
  }
  @Test func currentConversationRemainsReachableWhileBusyOrHoldingADraft() async throws {
    let workspace = WorkspaceModel(client: HistoryNoCore())
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: Data(
        #"{"origin":"http://127.0.0.1:43219","cookieName":"paddock_desktop_session","session":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#
          .utf8))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in
    }
    var fields = await runtime.presentation()
    fields["revision"] = .number(1)
    fields["busy"] = .bool(true)
    fields["conversation"] = .object([
      "id": .string("active"), "title": .string("Fixture"), "model": .string("fixture"),
      "messageCount": .number(1),
    ])
    workspace.chat.apply(
      try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields)))
    workspace.navigation.studio = .reads
    workspace.draft.message = "Keep draft"
    workspace.chat.microphoneStarting = true
    let sidebar = StudioConversationSidebar(
      chat: workspace.chat, hasDraft: true,
      onNewChat: {}, onFold: {}, onOpen: { workspace.navigation.returnToChat() })
    #expect(sidebar.canOpenConversation("active") && !sidebar.canOpenConversation("different"))
    sidebar.openConversation("active")
    #expect(workspace.navigation.studio.isConversation && workspace.chat.busy)
    #expect(workspace.draft.message == "Keep draft")
    await workspace.chat.shutdown()
  }
  @Test func actionMenuHasAButtonSizedLabelInBothThemes() {
    _ = NSApplication.shared
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: StudioHistoryActionsMenu(title: "A conversation", id: "fixture") {
          Button("Rename…") { Issue.record("Mounting a menu must not run an action") }
        }.environment(\.colorScheme, dark ? .dark : .light))
      let size = host.sizeThatFits(in: CGSize(width: 28, height: 28))
      #expect(size.width == 28 && size.height == 28)
    }
  }

  @Test func fullTitlePresentationIsBoundedAndDoesNotNeedAWorkspace() {
    _ = NSApplication.shared
    for dark in [false, true] {
      for title in ["Short title", String(repeating: "A long title 👋 ", count: 32)] {
        let host = NSHostingController(
          rootView: StudioFullChatTitle(title: title)
            .environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: CGSize(width: 340, height: 500))
        #expect(size.width == 340 && size.height <= 350)
      }
    }
  }

  @Test func searchShortcutUnfoldsHistoryWithoutOpeningAChatOrDiscardingDraft() {
    let model = WorkspaceModel(client: HistoryNoCore())
    model.draft.message = "Keep this draft"
    model.navigation.studio = .prompts
    model.searchConversations()
    #expect(model.navigation.studio == .chats && model.navigation.showsSidebar)
    let first = model.historySearchRequest
    #expect(first != nil && model.draft.message == "Keep this draft")
    model.searchConversations()
    #expect(model.historySearchRequest != first)
    let second = model.historySearchRequest
    model.desktopTransition = true
    model.searchConversations()
    #expect(model.historySearchRequest == second)
    model.desktopTransition = false
    model.navigation.mode = .manager
    model.searchConversations()
    #expect(model.navigation.mode == .manager && model.historySearchRequest == second)
  }

  private func row(_ id: String, title: String, date: Int, pinned: Bool = false) throws
    -> StudioState.History
  {
    let data = try JSONSerialization.data(withJSONObject: [
      "id": id, "title": title, "model": "fixture", "updatedAt": date, "pinned": pinned,
    ])
    return try JSONDecoder().decode(StudioState.History.self, from: data)
  }
  @Test func recentOrderPinsFirstAndHasStableTies() throws {
    let a = try row("a", title: "First", date: 1, pinned: true)
    let b = try row("b", title: "Newest", date: 100)
    let c = try row("c", title: "Older", date: 50)
    #expect(StudioHistorySort.newest.rows([c, b, a]).map(\.id) == ["a", "b", "c"])
    #expect(StudioHistorySort.oldest.rows([c, b, a]).map(\.id) == ["a", "c", "b"])
    #expect(StudioHistorySort.title.rows([c, b, a], search: " NEW ").map(\.id) == ["b"])
  }
  @Test func rowsAndRenameFitNarrowAndWideInBothAppearances() async throws {
    _ = NSApplication.shared
    let row = try row(
      "row", title: String(repeating: "Long conversation title ", count: 8), date: 1, pinned: true)
    let chat = StudioWorkspace(client: HistoryNoCore())
    for dark in [false, true] {
      for width: CGFloat in [200, 640] {
        let controller = NSHostingController(
          rootView: StudioHistoryRow(chat: chat, row: row, compact: width == 200) {}
            .preferredColorScheme(dark ? .dark : .light))
        let size = controller.sizeThatFits(in: CGSize(width: width, height: 500))
        #expect(size.width <= width + 1)
        #expect(size.height < 90)
      }
      let rename = NSHostingController(
        rootView: StudioRenameChat(chat: chat, row: row) {}.preferredColorScheme(
          dark ? .dark : .light))
      #expect(rename.sizeThatFits(in: CGSize(width: 340, height: 500)).width == 340)
    }
    await chat.shutdown()
  }
  @Test func pasteAndDropShareFileAndPixelRoutingWithoutUsingTheSystemClipboard() {
    let editor = DraftTextView(frame: .zero)
    let board = NSPasteboard.withUniqueName()
    defer { board.releaseGlobally() }
    var files: [URL] = []
    var pixels: Data?
    var isPNG = false
    editor.onFiles = { files = $0 }
    editor.onImage = {
      pixels = $0
      isPNG = $1
    }
    let original = URL(fileURLWithPath: "/tmp/synthetic-drop.pdf")
    board.writeObjects([original as NSURL])
    board.setData(Data([1, 2]), forType: .png)
    #expect(editor.acceptAttachment(board))
    #expect(files == [original])
    #expect(pixels == nil)  // Finder's preview must not replace the PDF.
    board.clearContents()
    board.setData(Data([3, 4]), forType: .png)
    #expect(editor.acceptAttachment(board))
    #expect(pixels == Data([3, 4]) && isPNG)
    board.clearContents()
    board.setData(Data([5, 6]), forType: .tiff)
    #expect(editor.acceptAttachment(board))
    #expect(pixels == Data([5, 6]) && !isPNG)
    board.clearContents()
    board.setString("Normal text", forType: .string)
    #expect(!editor.acceptAttachment(board))
  }

  @Test func chatPanelAndFixedFooterFitInBothThemes() async throws {
    _ = NSApplication.shared
    let chat = StudioWorkspace(client: HistoryNoCore())
    for dark in [false, true] {
      for width: CGFloat in [220, 260, 360] {
        let panel = StudioConversationSidebar(
          chat: chat, onNewChat: {}, onFold: {}, onOpen: {})
        let controller = NSHostingController(
          rootView: panel.environment(\.colorScheme, dark ? .dark : .light))
        let size = controller.sizeThatFits(in: CGSize(width: width, height: 650))
        #expect(size.width <= width + 1)
        #expect(size.height <= 650)
        // Reads, Tables, Prompt library and Settings: four 34-point rows, three
        // 3-point gaps and 22 points of vertical padding. The footer stays
        // outside the scrolling history and must not compress any row.
        let footerHeight: CGFloat = 4 * 34 + 3 * 3 + 22
        let footer = NSHostingController(
          rootView: StudioSidebarFooter(navigation: .constant(WorkspaceNavigation()))
            .environment(\.colorScheme, dark ? .dark : .light))
        let fit = footer.sizeThatFits(in: CGSize(width: width, height: footerHeight))
        #expect(fit.width <= width + 1)
        #expect(abs(fit.height - footerHeight) <= 1)
      }
    }
    await chat.shutdown()
  }
}

private struct HistoryNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    throw ManagerError.core("No core in layout tests")
  }
  func close() async {}
}
