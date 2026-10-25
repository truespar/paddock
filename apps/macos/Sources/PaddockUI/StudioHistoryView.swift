import PaddockStudio
import SwiftUI

enum StudioHistorySort: String, CaseIterable {
  case newest = "Newest first"
  case oldest = "Oldest first"
  case title = "Title"
  var wire: String { self == .newest ? "newest" : self == .oldest ? "oldest" : "title" }
  func rows(_ rows: [StudioState.History], search: String = "") -> [StudioState.History] {
    let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
    return rows.filter { query.isEmpty || $0.title.localizedCaseInsensitiveContains(query) }
      .sorted { a, b in
        if (a.pinned == true) != (b.pinned == true) { return a.pinned == true }
        if self == .title {
          let comparison = a.title.localizedStandardCompare(b.title)
          if comparison != .orderedSame { return comparison == .orderedAscending }
        } else if a.updatedAt != b.updatedAt {
          return self == .newest ? a.updatedAt > b.updatedAt : a.updatedAt < b.updatedAt
        }
        return a.id < b.id
      }
  }
}

/// ConversationSidebar's row/actions. Native
/// controls send typed commands; no second SQLite model or title generator.
struct StudioHistoryRow: View {
  @Bindable var chat: StudioWorkspace
  let row: StudioState.History
  var compact = false
  var canOpen = true
  var canDelete = true
  var selected = false
  var selecting = false
  let onOpen: () -> Void
  @State private var renaming = false
  @State private var showingTitle = false
  @State private var deleting = false
  private var changing: Bool { chat.historyMutations.contains(row.id) }
  private var symbol: String {
    row.kind == "document" ? "doc.text" : row.kind == "transcription" ? "waveform" : "bubble.left"
  }
  var body: some View {
    HStack(alignment: .top, spacing: 4) {
      Button(action: onOpen) {
        HStack(alignment: .top, spacing: 8) {
          Image(systemName: selecting ? (selected ? "checkmark.circle.fill" : "circle") : symbol)
            .frame(width: 18, height: 18).foregroundStyle(.secondary).accessibilityHidden(true)
          VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .top, spacing: 5) {
              Text(verbatim: row.title).lineLimit(2).truncationMode(.tail)
                .multilineTextAlignment(.leading).frame(maxWidth: .infinity, alignment: .leading)
              if row.pinned == true { Image(systemName: "pin.fill").font(.system(size: 9)) }
            }
            HStack(spacing: 8) {
              if row.composerDraft == true { Text("Draft").fontWeight(.medium) }
              if !compact {
                Text(verbatim: row.model).lineLimit(1).truncationMode(.middle)
              }
              Text(Date(timeIntervalSince1970: row.updatedAt / 1000), style: .date)
                .help(Date(timeIntervalSince1970: row.updatedAt / 1000).formatted())
            }.font(.system(size: 10)).foregroundStyle(.secondary)
          }
          Spacer(minLength: 0)
        }.font(.system(size: 13)).padding(.vertical, compact ? 8 : 12)
          .padding(.leading, 10).contentShape(Rectangle())
      }.buttonStyle(QuietButtonStyle()).disabled(!canOpen || changing)
        .help(row.title).accessibilityLabel(row.title)
        .accessibilityIdentifier("open-chat-\(row.id)")
      StudioHistoryRowControls(
        title: row.title, id: row.id,
        busy: row.busy == true || row.titleState == "generating" || changing,
        status: row.titleState, progressLabel: row.busy == true ? "Answering" : "Saving",
        showsActions: !selecting, actionsEnabled: !changing
      ) { actions }.padding(.top, compact ? 3 : 7)
    }.padding(.trailing, 6)
      .background(
        selected ? PaddockStyle.elevated : .clear,
        in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
      )
      .contextMenu { if !selecting { actions } }
      .popover(isPresented: $renaming, arrowEdge: .trailing) {
        StudioRenameChat(chat: chat, row: row) { renaming = false }
      }
      .popover(isPresented: $showingTitle, arrowEdge: .trailing) {
        StudioFullChatTitle(title: row.title)
      }
      .confirmationDialog("Delete conversation?", isPresented: $deleting, titleVisibility: .visible)
    {
      Button("Delete conversation", role: .destructive) {
        Task { await chat.changeHistory("deleteChats", ids: [row.id]) }
      }
    } message: {
      Text("\"\(row.title)\" and all its message branches will be deleted. This cannot be undone.")
    }
  }
  @ViewBuilder private var actions: some View {
    Button("Rename…", systemImage: "pencil") { renaming = true }.disabled(changing)
    Button(row.pinned == true ? "Unpin" : "Pin", systemImage: "pin") {
      Task { await chat.changeHistory("pinChat", ids: [row.id]) }
    }.disabled(changing)
    Button("Generate title", systemImage: "text.badge.star") {
      Task { await chat.changeHistory("generateTitle", ids: [row.id]) }
    }.disabled(chat.busy || row.busy == true || row.titleState == "generating" || changing)
    Button("Show full title…", systemImage: "text.alignleft") { showingTitle = true }
    Divider()
    Button("Delete…", systemImage: "trash", role: .destructive) { deleting = true }
      .disabled(!canDelete || row.busy == true || changing)
  }
}

struct StudioRenameChat: View {
  @Bindable var chat: StudioWorkspace
  let row: StudioState.History
  let onDone: () -> Void
  @State private var title = ""
  @FocusState private var focused: Bool
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      StudioPopoverHeading(title: "Rename conversation")
      TextField("Conversation name", text: $title).textFieldStyle(.plain)
        .padding(10).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 6))
        .focused($focused).onSubmit(save).accessibilityIdentifier("rename-chat-title")
      if let error = chat.error {
        Text(error).font(.system(size: 11)).foregroundStyle(.secondary).textSelection(.enabled)
      }
      HStack {
        Button("Cancel", action: onDone).buttonStyle(FlatButtonStyle()).keyboardShortcut(
          .cancelAction)
        Spacer()
        Button("Save", action: save).buttonStyle(FlatButtonStyle(primary: true))
          .disabled(!valid || chat.historyMutations.contains(row.id))
          .accessibilityIdentifier("rename-chat-save")
      }
    }.font(.system(size: 12)).padding(16).frame(width: 340).studioPopoverSurface()
      .onAppear {
        title = row.title
        focused = true
      }
  }
  private var valid: Bool {
    !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && title.utf16.count <= 512
      && !title.unicodeScalars.contains { CharacterSet.controlCharacters.contains($0) }
  }
  private func save() {
    guard valid, !chat.historyMutations.contains(row.id) else { return }
    Task { if await chat.changeHistory("renameChat", ids: [row.id], title: title) { onDone() } }
  }
}

/// Chat history uses the shared store. Workspace owns its fixed Settings footer.
struct StudioConversationSidebar: View {
  @Bindable var chat: StudioWorkspace
  var hasDraft = false
  var onNewChat: () -> Void
  var onFold: () -> Void
  var onOpen: () -> Void
  var searchRequest: UUID?
  @AppStorage("studioChatSort") private var sort = StudioHistorySort.newest
  @State private var search = ""
  @State private var selecting = false
  @State private var selection = Set<String>()
  @State private var deleting = false
  @FocusState private var searchFocused: Bool
  private var rows: [StudioState.History] {
    chat.state?.library?.rows ?? sort.rows(chat.history, search: search)
  }
  private var canNavigate: Bool {
    chat.canNavigateConversation
  }
  func canOpenConversation(_ id: String) -> Bool {
    canNavigate || chat.conversation?.id == id
  }
  func openConversation(_ id: String) {
    guard canOpenConversation(id) else { return }
    // Navigation must not wait for the runtime's separate presentation stream.
    onOpen()
    guard chat.conversation?.id != id else { return }
    Task { await chat.open(id) }
  }
  private var canDeleteSelection: Bool {
    !selection.isEmpty
      && selection.allSatisfy { id in
        rows.first(where: { $0.id == id })?.busy != true
          && !chat.historyMutations.contains(id)
          && (id != chat.conversation?.id || canNavigate)
      }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack(spacing: 8) {
        Button(action: onNewChat) {
          Label("New chat", systemImage: "square.and.pencil")
            .font(.system(size: 13, weight: .medium))
            .padding(.horizontal, 8).frame(height: 32)
            .frame(maxWidth: .infinity, alignment: .leading)
        }.buttonStyle(QuietButtonStyle())
          .disabled(!chat.canNavigateConversation)
          .accessibilityIdentifier("sidebar-new-chat")
      }
      SidebarSearchField(text: $search) {
        TextField("Search chats", text: $search).textFieldStyle(.plain)
          .focused($searchFocused)
          .accessibilityIdentifier("history-search")
          .help("Search chats · ⌘K")
      }
      HStack(spacing: 8) {
        Menu {
          ForEach([StudioHistorySort.newest, .oldest], id: \.self) { order in
            Toggle(
              order.rawValue, isOn: Binding(get: { sort == order }, set: { if $0 { sort = order } })
            )
          }
          Divider()
          Toggle("Automatically name new chats", isOn: autoTitle)
            .help("Uses a short request to the conversation's model. Provider charges may apply.")
        } label: {
          Label(sort.rawValue, systemImage: "arrow.up.arrow.down")
            .font(.system(size: 11))
        }.menuStyle(.button).buttonStyle(.plain).fixedSize()
          .accessibilityLabel("Sort chats").accessibilityIdentifier("sidebar-sort-chats")
        Spacer(minLength: 0)
        Button(selecting ? "Cancel" : "Select") {
          selecting.toggle()
          selection.removeAll()
        }.buttonStyle(.plain).font(.system(size: 11))
          .disabled(chat.history.isEmpty && rows.isEmpty)
          .accessibilityIdentifier("sidebar-select-chats")
      }
      if selecting {
        HStack {
          Button(selection.isSuperset(of: rows.map(\.id)) ? "Clear all" : "Select all") {
            if selection.isSuperset(of: rows.map(\.id)) {
              selection.subtract(rows.map(\.id))
            } else {
              selection.formUnion(rows.map(\.id))
            }
          }.buttonStyle(.plain).disabled(rows.isEmpty)
          Text("\(selection.count) selected").font(.system(size: 12)).foregroundStyle(.secondary)
          Spacer()
        }.font(.system(size: 11))
      }
      PaddockScrollView {
        LazyVStack(spacing: 4) {
          if rows.isEmpty {
            Text(
              search.isEmpty
                ? "No chats yet" : "No matches"
            )
            .foregroundStyle(.secondary).frame(maxWidth: .infinity).padding(.top, 40)
          }
          ForEach(rows) { row in
            StudioHistoryRow(
              chat: chat, row: row, compact: true,
              canOpen: selecting || canOpenConversation(row.id),
              canDelete: row.id != chat.conversation?.id || canNavigate,
              selected: selecting ? selection.contains(row.id) : chat.conversation?.id == row.id,
              selecting: selecting
            ) {
              if selecting {
                if selection.contains(row.id) {
                  selection.remove(row.id)
                } else {
                  selection.insert(row.id)
                }
              } else {
                openConversation(row.id)
              }
            }
          }
        }
      }.frame(maxHeight: .infinity)
      if let library = chat.state?.library, library.matched > library.pageSize {
        HStack {
          Text(
            "\(library.page * library.pageSize + 1)-\(min((library.page + 1) * library.pageSize, library.matched)) of \(library.matched)"
          )
          .font(.system(size: 12)).foregroundStyle(.secondary)
          Spacer()
          Button("Previous", systemImage: "chevron.left") { load(page: library.page - 1) }
            .labelStyle(.iconOnly).buttonStyle(FlatButtonStyle())
            .disabled(library.page == 0)
          Button("Next", systemImage: "chevron.right") { load(page: library.page + 1) }
            .labelStyle(.iconOnly).buttonStyle(FlatButtonStyle())
            .disabled((library.page + 1) * library.pageSize >= library.matched)
        }
      }
      if selecting {
        Button("Delete \(selection.count) selected…", role: .destructive) { deleting = true }
          .buttonStyle(FlatButtonStyle()).disabled(!canDeleteSelection)
      }
      if let error = chat.error {
        Text(error).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled)
      }
    }.padding(.horizontal, 10).padding(.vertical, 18)
      // Workspace owns the full-height column backdrop and fixed footer.
      .accessibilityElement(children: .contain).accessibilityLabel("Chats")
      .accessibilityIdentifier("studio-conversation-sidebar")
      .task {
        if sort == .title { sort = .newest }
        await chat.refreshHistory()
      }
      .task(id: searchRequest) {
        guard searchRequest != nil else { return }
        // The shortcut can mount a folded sidebar. Focus after its field has
        // joined the window, without opening a chat or replacing the draft.
        await Task.yield()
        guard !Task.isCancelled else { return }
        searchFocused = true
      }
      .task(id: search + "\u{0}" + sort.wire + String(chat.ready)) {
        do { try await Task.sleep(for: .milliseconds(150)) } catch { return }
        guard chat.ready, !Task.isCancelled else { return }
        selection.removeAll()
        await chat.perform(
          "historyFilter",
          ["search": .string(search), "sort": .string(sort.wire), "page": .number(0)])
      }
      .onChange(of: rows.map(\.id)) { _, ids in selection.formIntersection(ids) }
      .confirmationDialog(
        "Delete selected conversations?", isPresented: $deleting, titleVisibility: .visible
      ) {
        Button("Delete \(selection.count) conversations", role: .destructive) {
          let ids = Array(selection)
          Task {
            if await chat.changeHistory("deleteChats", ids: ids) {
              selection.removeAll()
              selecting = false
            }
          }
        }
      } message: {
        Text("All message branches in these conversations will be deleted. This cannot be undone.")
      }
  }
  private var autoTitle: Binding<Bool> {
    Binding(
      get: { chat.state?.autoTitle ?? true },
      set: { enabled in Task { await chat.perform("autoTitle", ["enabled": .bool(enabled)]) } })
  }
  private func load(page: Int) {
    selection.removeAll()
    Task {
      await chat.perform(
        "historyFilter",
        ["search": .string(search), "sort": .string(sort.wire), "page": .number(Double(page))])
    }
  }
}
