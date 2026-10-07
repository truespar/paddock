import AppKit
import SwiftUI

/// Single-column settings share a readable measure, heading and scroll owner.
/// Catalogs and other split-pane browsers deliberately don't use this surface.
struct SettingsPage<Content: View>: View {
  let title: String
  @ViewBuilder var content: (_ stacked: Bool) -> Content

  var body: some View {
    GeometryReader { geometry in
      let stacked = geometry.size.width < 600
      // Reserve both sides of the native scrollbar lane even on short pages.
      // Changing sections must not shift every heading/control sideways when
      // a longer page acquires a scroller. The scroll wrapper centers the lane.
      let gutter = NSScroller.scrollerWidth(for: .regular, scrollerStyle: .legacy)
      let measure = min(820, max(0, geometry.size.width - 2 * gutter))
      PaddockScrollView(centersContent: true) {
        VStack(alignment: .leading, spacing: 28) {
          PageHeading(title: title) { EmptyView() }
            .accessibilityIdentifier("settings-page-heading")
          content(stacked)
        }
        .padding(stacked ? 20 : 32)
        .frame(width: measure, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .top)
      }
    }
    .font(.system(size: 13)).tint(PaddockStyle.accent)
    .buttonStyle(FlatButtonStyle()).background(PaddockStyle.canvas)
  }
}
