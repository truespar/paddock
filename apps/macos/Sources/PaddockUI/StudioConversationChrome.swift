import AppKit
import SwiftUI

/// Top controls are outside the scrolling viewport. The bottom composer keeps
/// its native floating bar and automatic scroll-edge treatment.
struct StudioConversationBars<Header: View, Footer: View>: ViewModifier {
  @ViewBuilder var header: () -> Header
  @ViewBuilder var footer: () -> Footer

  @ViewBuilder func body(content: Content) -> some View {
    VStack(spacing: 0) {
      header()
      bottomBar(content: content).clipped()
    }
  }

  @ViewBuilder private func bottomBar(content: Content) -> some View {
    if #available(macOS 26.0, *) {
      content
        .safeAreaBar(edge: .bottom, spacing: 0, content: footer)
        .scrollEdgeEffectHidden(true, for: .top)
        .scrollEdgeEffectStyle(.automatic, for: .bottom)
    } else {
      // The package's diagnostic tools still build on macOS 15. Shipping app
      // builds target 26+, where the system provides scroll-edge rendering.
      content
        .safeAreaInset(edge: .bottom, spacing: 0, content: footer)
    }
  }
}

struct StudioConversationChrome<Content: View>: View {
  @ViewBuilder var content: () -> Content
  @Environment(\.workspaceWindowChrome) private var windowChrome
  @State private var standaloneInset: CGFloat = 0
  private var titlebarInset: CGFloat { windowChrome?.topInset ?? standaloneInset }

  var body: some View {
    VStack(spacing: 0) {
      if titlebarInset > 0 {
        PaddockStyle.canvas.frame(height: titlebarInset)
          .contentShape(Rectangle()).gesture(WindowDragGesture())
          .overlay(alignment: .bottom) {
            if windowChrome == nil { Divider() }
          }
      }
      content().frame(maxWidth: .infinity, maxHeight: .infinity).clipped()
    }
    .background(PaddockStyle.canvas)
    .background(alignment: .top) {
      // Standalone rendering fixtures have no workspace shell. The app uses
      // the shell's single measurement for sidebar, catalog and every pane.
      if windowChrome == nil {
        StudioWindowChromeProbe { standaloneInset = $0 }
          .frame(height: 0).allowsHitTesting(false).accessibilityHidden(true)
      }
    }
  }
}

private struct StudioWindowChromeProbe: NSViewRepresentable {
  var onInsetChange: (CGFloat) -> Void
  func makeNSView(context: Context) -> StudioWindowChromeProbeView { StudioWindowChromeProbeView() }
  func updateNSView(_ view: StudioWindowChromeProbeView, context: Context) {
    view.onInsetChange = onInsetChange
    view.refreshClearance()
  }
}

/// The nested split hosts deliberately do not re-inherit window safe areas.
/// Measure the actual titlebar overlap for this pane's non-scrolling header.
/// This zero-height probe draws nothing and never observes scroll offsets.
final class StudioWindowChromeProbeView: NSView {
  var onInsetChange: ((CGFloat) -> Void)?
  private var windowObservation: NSKeyValueObservation?
  private var publishedInset: CGFloat = -1
  private var measuredInset: CGFloat = 0
  private var updateQueued = false

  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    windowObservation = window?.observe(\.contentLayoutRect, options: [.new]) { [weak self] _, _ in
      Task { @MainActor [weak self] in self?.refreshClearance() }
    }
    refreshClearance()
  }

  override func layout() {
    super.layout()
    refreshClearance()
  }

  func refreshClearance() {
    measuredInset =
      window.map { max(0, convert(bounds, to: nil).maxY - $0.contentLayoutRect.maxY) } ?? 0
    guard measuredInset != publishedInset, !updateQueued else { return }
    updateQueued = true
    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      updateQueued = false
      guard measuredInset != publishedInset else { return }
      publishedInset = measuredInset
      onInsetChange?(measuredInset)
    }
  }
}
