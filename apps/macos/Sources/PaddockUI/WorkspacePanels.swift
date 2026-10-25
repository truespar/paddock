import AppKit
import SwiftUI

extension EnvironmentValues {
  /// The shell measures the native toolbar once. Pane controls reserve this
  /// row while their backgrounds and dividers fill the entire window.
  @Entry var workspaceLeadingPaneInset: CGFloat = 0
  /// Catalog controls clear native chrome inside each column; their shared
  /// resize divider remains full-height instead of inheriting this padding.
  @Entry var workspaceCatalogContentInset: CGFloat = 0
}

/// One sidebar edge, no outer activity rail or inset content cards.
enum WorkspacePanelMetrics {
  static let gap: CGFloat = 4
  static let radius: CGFloat = 8
  static let contentMinimum: CGFloat = 640
  static func railWidth(_ mode: WorkspaceMode) -> CGFloat { 0 }
  static func limits(_ mode: WorkspaceMode) -> ClosedRange<CGFloat> {
    mode == .studio ? 220...360 : 200...260
  }
  static func minimumWidth(_ navigation: WorkspaceNavigation) -> CGFloat {
    let panels = navigation.showsSidebar ? limits(navigation.mode).lowerBound + gap : 0
    return max(900, panels + contentMinimum)
  }
  static func width(preferred: CGFloat, mode: WorkspaceMode, available: CGFloat) -> CGFloat {
    let range = limits(mode)
    let maximum = max(
      range.lowerBound,
      min(range.upperBound, available - gap - contentMinimum))
    return min(max(preferred.isFinite ? preferred : range.lowerBound, range.lowerBound), maximum)
  }
}

extension View {
  /// Split children get fresh SwiftUI hosting roots. Keep the column surface
  /// edge-to-edge instead of allowing those roots to inherit window chrome.
  func fullHeightWorkspaceColumn() -> some View {
    frame(maxWidth: .infinity, maxHeight: .infinity)
      .background(PaddockStyle.canvas)
      .ignoresSafeArea(.container, edges: .top)
  }

  func workspacePanel(_ color: Color) -> some View {
    // Separate surfaces with fill and spacing, not nested outlines. The
    // surrounding window and the composer's input already supply boundaries.
    background(color, ignoresSafeAreaEdges: [])
      .clipShape(RoundedRectangle(cornerRadius: WorkspacePanelMetrics.radius))
  }
}

/// The gutter itself is the handle, like Vaka's PanelSplitter. Global gesture
/// coordinates matter: local coordinates move with the handle and halve a drag.
struct WorkspacePanelGrip: View {
  @Binding var width: CGFloat
  var range: ClosedRange<CGFloat>
  var label: String
  @State private var dragStart: CGFloat?
  @State private var hovering = false
  @FocusState private var focused: Bool

  var body: some View {
    Color.clear
      .frame(width: WorkspacePanelMetrics.gap).frame(maxHeight: .infinity)
      .overlay(alignment: .leading) {
        Rectangle().fill(PaddockStyle.border).frame(width: 1)
      }
      .contentShape(Rectangle())
      .onHover { inside in
        guard inside != hovering else { return }
        hovering = inside
        if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
      }
      .gesture(
        DragGesture(minimumDistance: 0, coordinateSpace: .global)
          .onChanged { value in
            let start = dragStart ?? width
            if dragStart == nil { dragStart = start }
            resize(to: start + value.translation.width)
          }
          .onEnded { _ in dragStart = nil }
      )
      .focusable().focused($focused).focusEffectDisabled()
      .onKeyPress(.leftArrow) {
        resize(to: width - 10)
        return .handled
      }
      .onKeyPress(.rightArrow) {
        resize(to: width + 10)
        return .handled
      }
      .accessibilityElement(children: .ignore)
      .accessibilityLabel(label).accessibilityValue("\(Int(width)) points")
      .accessibilityIdentifier("workspace-panel-grip")
      .accessibilityAdjustableAction { direction in
        switch direction {
        case .increment: resize(to: width + 10)
        case .decrement: resize(to: width - 10)
        @unknown default: break
        }
      }
      .help("Drag to resize. When focused, use the left and right arrow keys.")
      .onDisappear {
        if hovering {
          NSCursor.pop()
          hovering = false
        }
        dragStart = nil
      }
  }
  private func resize(to value: CGFloat) {
    width = min(range.upperBound, max(range.lowerBound, value))
  }
}
