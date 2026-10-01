import PaddockConversationCore
import SwiftUI

struct NativeTablesSidebar: View {
  @Bindable var model: NativeTablesModel
  @State private var search = ""
  @State private var renaming: String?
  @State private var title = ""
  @State private var deleting: TableSummary?
  @FocusState private var renameFocused: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Button {
        Task { await model.newSession() }
      } label: {
        Label("New table", systemImage: "plus")
          .font(.system(size: 13, weight: .medium)).padding(.horizontal, 8)
          .frame(height: 32).frame(maxWidth: .infinity, alignment: .leading)
      }.buttonStyle(QuietButtonStyle()).disabled(model.historyNavigationBlocked)
        .accessibilityIdentifier("sidebar-new-table")
      HStack(spacing: 8) {
        Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
        TextField("Search tables", text: $search).textFieldStyle(.plain)
        if !search.isEmpty {
          Button("Clear search", systemImage: "xmark") { search = "" }
            .labelStyle(.iconOnly).buttonStyle(.plain)
        }
      }.padding(10).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 8))
      PaddockScrollView {
        LazyVStack(spacing: 4) {
          let rows = model.history.sessions.filter {
            search.isEmpty || $0.title.localizedCaseInsensitiveContains(search)
          }
          if !model.history.loaded {
            ProgressView().controlSize(.small).frame(maxWidth: .infinity).padding(.top, 40)
          } else if rows.isEmpty {
            Text(search.isEmpty ? "No tables yet" : "No matches")
              .foregroundStyle(.secondary).frame(maxWidth: .infinity).padding(.top, 40)
          }
          ForEach(rows) { session in
            HStack(spacing: 4) {
              if renaming == session.id {
                TextField("Table title", text: $title).textFieldStyle(.plain)
                  .focused($renameFocused).onSubmit { commitRename() }
                  .onExitCommand {
                    renaming = nil
                    renameFocused = false
                  }
                  .padding(.horizontal, 8).frame(height: 42)
              } else {
                Button {
                  Task { await model.openSession(session.id) }
                } label: {
                  HStack(spacing: 8) {
                    Image(systemName: "tablecells").foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 3) {
                      Text(session.title).lineLimit(1)
                      Text("\(session.runs) \(session.runs == 1 ? "run" : "runs")")
                        .font(.system(size: 10)).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 0)
                  }.font(.system(size: 12)).padding(.leading, 8).frame(height: 42)
                    .contentShape(Rectangle())
                }.buttonStyle(QuietButtonStyle()).disabled(model.historyNavigationBlocked)
                  .help(session.title).accessibilityIdentifier("tables-open-\(session.id)")
                  .accessibilityAddTraits(
                    model.history.document?.id == session.id ? .isSelected : [])
                Menu {
                  Button("Rename", systemImage: "pencil") {
                    title = session.title
                    renaming = session.id
                    renameFocused = true
                  }
                  Divider()
                  Button("Delete", systemImage: "trash", role: .destructive) { deleting = session }
                } label: {
                  Image(systemName: "ellipsis").frame(width: 26, height: 28)
                }.menuStyle(.button).buttonStyle(QuietButtonStyle()).menuIndicator(.hidden)
                  .fixedSize().foregroundStyle(.secondary).disabled(model.historyNavigationBlocked)
                  .accessibilityLabel("Actions for \(session.title)")
              }
            }.background(
              model.history.document?.id == session.id ? PaddockStyle.elevated : .clear,
              in: RoundedRectangle(cornerRadius: 7))
          }
        }
      }.frame(maxHeight: .infinity)
    }.padding(.horizontal, 10).padding(.vertical, 18)
      .accessibilityElement(children: .contain).accessibilityLabel("Tables")
      .accessibilityIdentifier("studio-tables-sidebar")
      .task { await model.history.refresh(api: model.api) }
      .onChange(of: renameFocused) { _, focused in if !focused { commitRename() } }
      .confirmationDialog(
        "Delete table and all its runs?",
        isPresented: Binding(
          get: { deleting != nil }, set: { if !$0 { deleting = nil } }), titleVisibility: .visible
      ) {
        Button("Delete table", role: .destructive) {
          if let row = deleting { Task { await model.removeSession(row) } }
          deleting = nil
        }
        Button("Cancel", role: .cancel) { deleting = nil }
      }
  }
  private func commitRename() {
    guard let id = renaming else { return }
    renaming = nil
    renameFocused = false
    Task { await model.renameSession(id, title: title) }
  }
}
