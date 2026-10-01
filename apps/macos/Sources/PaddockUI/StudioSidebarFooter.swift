import SwiftUI

/// Anchored outside the scrolling history. No second navigation rail.
struct StudioSidebarFooter: View {
  @Binding var navigation: WorkspaceNavigation
  @AppStorage("workspaceAppearance") private var appearance: WorkspaceAppearance = .system
  var body: some View {
    VStack(spacing: 3) {
      if navigation.showsReadsHistory || navigation.studio == .tables {
        destination(.chats, title: "Chats")
      }
      if !navigation.showsReadsHistory {
        destination(.reads, title: "Reads")
      }
      if navigation.studio != .tables { destination(.tables, title: "Tables") }
      destination(.prompts, title: "Prompt library")
      HStack(spacing: 2) {
        destination(.settings, title: "Settings")
        Menu {
          Picker("Appearance", selection: $appearance) {
            ForEach(WorkspaceAppearance.allCases) { option in Text(option.rawValue).tag(option) }
          }.pickerStyle(.inline).labelsHidden()
        } label: {
          Image(systemName: "circle.lefthalf.filled").frame(width: 30, height: 32)
        }.menuStyle(.button).buttonStyle(QuietButtonStyle()).menuIndicator(.hidden)
          .fixedSize().foregroundStyle(.secondary).accessibilityLabel("Appearance")
      }
    }.padding(.horizontal, 12).padding(.bottom, 14).padding(.top, 8)
      .accessibilityIdentifier("studio-sidebar-footer")
  }
  private func destination(_ destination: StudioDestination, title: String) -> some View {
    Button {
      if destination == .settings {
        navigation.showSettings()
      } else {
        navigation.studio = destination
      }
    } label: {
      HStack(spacing: 9) {
        Image(systemName: destination.symbol).frame(width: 18)
        Text(title)
        Spacer(minLength: 0)
      }.font(.system(size: 12)).padding(.horizontal, 8).frame(height: 34)
        .foregroundStyle(navigation.studio == destination ? .primary : .secondary)
        .background(
          navigation.studio == destination ? PaddockStyle.elevated : .clear,
          in: RoundedRectangle(cornerRadius: 7)
        ).contentShape(Rectangle())
    }.buttonStyle(QuietButtonStyle()).accessibilityIdentifier("sidebar-\(destination.id)")
      .accessibilityAddTraits(navigation.studio == destination ? .isSelected : [])
  }
}
