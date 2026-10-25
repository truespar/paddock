import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Bounded catalog columns", .serialized) @MainActor
struct CatalogColumnsTests {
  private typealias Columns = CatalogColumns<Color, Color>

  @Test func resizingAlwaysKeepsBothColumnsInsideTheViewport() {
    for limits: ClosedRange<CGFloat> in [240...400, 260...420] {
      for available: CGFloat in [640, 688, 920, 1200] {
        for preferred: CGFloat in [0, 260, 310, 400, 1000] {
          let width = Columns.listWidth(preferred: preferred, available: available, limits: limits)
          #expect(limits.contains(width))
          #expect(available - width - WorkspacePanelMetrics.gap >= Columns.detailMinimum)
        }
      }
    }
  }

  @Test func contentAcceptsNarrowAndWideProposalsInBothThemes() {
    _ = NSApplication.shared
    for dark in [false, true] {
      let controller = NSHostingController(
        rootView: Columns {
          Color.clear
        } detail: {
          Color.clear
        }
        .environment(\.colorScheme, dark ? .dark : .light))
      for width: CGFloat in [640, 1200, 640] {
        let proposed = CGSize(width: width, height: 600)
        let fitted = controller.sizeThatFits(in: proposed)
        #expect(fitted.width == width && fitted.height == 600)
      }
    }
  }

  @Test func allCatalogDividersReachWindowTopWithoutLiftingControlsIntoChrome() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: CatalogChromeClient(), cloudClient: CloudFixture())
    await model.refresh()
    await model.cloud.refresh()
    model.navigation.showManager(.models)
    let host = NSHostingController(rootView: WorkspaceView(model: model))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1100, height: 760),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbarStyle = .unifiedCompact
    window.toolbar = NSToolbar(identifier: "catalog-chrome-regression")
    window.contentViewController = host
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for dark in [false, true] {
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      for sidebar in [true, false] {
        model.navigation.sidebarVisible = sidebar
        for width: CGFloat in [900, 1100] {
          window.setContentSize(NSSize(width: width, height: 760))
          for destination in [ManagerDestination.models, .cloudProviders, .customEndpoints] {
            model.navigation.manager = destination
            try await Task.sleep(for: .milliseconds(100))
            host.view.layoutSubtreeIfNeeded()
            let bounds = host.view.bounds
            let chromeHeight = bounds.height - window.contentLayoutRect.height
            #expect(chromeHeight > 10, "Exercise a real title-bar inset, not a borderless fixture")
            let shells = Self.views(host.view).compactMap {
              ($0 as? NSSplitView)?.delegate as? WorkspaceSplitController
            }
            let shell = try #require(shells.first)
            let sidebarWidth = shell.contentHost.view.convert(
              shell.contentHost.view.bounds, to: host.view
            ).minX
            let dividerX =
              sidebarWidth
              + Columns.listWidth(
                preferred: 310, available: bounds.width - sidebarWidth, limits: 240...400)
            let bitmap = try #require(host.view.bitmapImageRepForCachingDisplay(in: bounds))
            host.view.cacheDisplay(in: bounds, to: bitmap)
            let sx = CGFloat(bitmap.pixelsWide) / bounds.width
            let sy = CGFloat(bitmap.pixelsHigh) / bounds.height
            func color(_ x: CGFloat, _ y: CGFloat) throws -> NSColor {
              try #require(
                bitmap.colorAt(x: Int(x * sx), y: Int(y * sy))?.usingColorSpace(.deviceRGB))
            }
            let middle = try color(dividerX + 0.25, bounds.height / 2)
            let beside = try color(dividerX + 2, bounds.height / 2)
            #expect(
              Self.distance(middle, beside) > 0.03,
              "Measure the actual divider, not empty background")
            for y in [CGFloat(8), chromeHeight / 2, bounds.height - 8] {
              let edge = try color(dividerX + 0.25, y)
              #expect(
                Self.distance(edge, middle) < 0.025,
                "\(destination.rawValue) divider is inset at y=\(y), sidebar=\(sidebar), width=\(width)"
              )
            }
            for scroll in Self.scrolls(host.view) {
              let frame = scroll.convert(scroll.bounds, to: host.view)
              if frame.minX > sidebarWidth {
                // SwiftUI may extend the native viewport through the title
                // bar while preserving the controls' offset as contentInsets.
                #expect(
                  frame.minY + scroll.contentInsets.top >= chromeHeight - 1,
                  "Column controls must still clear native chrome: \(frame)")
              }
            }
            for field in Self.views(host.view).compactMap({ $0 as? NSTextField })
            where field.isEditable {
              let frame = field.convert(field.bounds, to: host.view)
              #expect(
                frame.minY >= chromeHeight - 1,
                "Search must stay below the window controls: \(frame)")
            }
          }
        }
      }
    }
    await model.shutdown()
  }

  private static func distance(_ a: NSColor, _ b: NSColor) -> CGFloat {
    abs(a.redComponent - b.redComponent) + abs(a.greenComponent - b.greenComponent)
      + abs(a.blueComponent - b.blueComponent)
  }
  private static func scrolls(_ view: NSView) -> [NSScrollView] {
    if let scroll = view as? NSScrollView { return [scroll] }
    return view.subviews.flatMap(scrolls)
  }
  private static func views(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(views) }
}

private struct CatalogChromeClient: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { try endpointFixture() }
}
