import SwiftUI

/// The established chat-history search appearance, shared by sidebar searches.
/// Each caller retains its own native editor, focus and command routing.
struct SidebarSearchField<Editor: View>: View {
  @Binding var text: String
  @ViewBuilder var editor: Editor

  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
        .accessibilityHidden(true)
      editor
      if !text.isEmpty {
        Button("Clear search", systemImage: "xmark") { text = "" }
          .labelStyle(.iconOnly).buttonStyle(.plain)
      }
    }.padding(10).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 8))
  }
}
