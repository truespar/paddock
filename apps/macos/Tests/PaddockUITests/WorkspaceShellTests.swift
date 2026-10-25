import AppKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native workspace resizing", .serialized) @MainActor
struct WorkspaceShellTests {
  @Test func fullscreenBackdropTracksPaletteAndResizingWithoutCapturingInput() async throws {
    _ = NSApplication.shared
    let backdrop = WorkspaceToolbarBackdrop(frame: NSRect(x: 0, y: 0, width: 1200, height: 52))
    let surface = NSView(frame: backdrop.frame)
    surface.addSubview(backdrop)
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1200, height: 52),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = surface
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for dark in [true, false, true] {
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      for width: CGFloat in [900, 1728] {
        window.setContentSize(NSSize(width: width, height: 52))
        for sidebar: CGFloat in [0, 260, 350] {
          backdrop.sidebarWidth = sidebar
          try await settle(surface)
          #expect(backdrop.frame == surface.bounds)
          #expect(backdrop.hitTest(NSPoint(x: 20, y: 20)) == nil)
          #expect(!backdrop.acceptsFirstResponder)
          // Use a known color space, independent of the current monitor's profile.
          let context = try #require(
            CGContext(
              data: nil, width: Int(width), height: 52, bitsPerComponent: 8,
              bytesPerRow: Int(width) * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
          backdrop.displayIgnoringOpacity(
            backdrop.bounds, in: NSGraphicsContext(cgContext: context, flipped: false))
          let pixels = try #require(context.data?.assumingMemoryBound(to: UInt8.self))
          for (x, role) in [(20, sidebar > 0 ? "sidebar" : "canvas"), (500, "canvas")] {
            let expected = PaddockStyle.nsColor(role, dark: dark)
            let offset = 20 * context.bytesPerRow + x * 4
            for (channel, value) in [
              expected.redComponent, expected.greenComponent, expected.blueComponent,
            ].enumerated() {
              #expect(abs(CGFloat(pixels[offset + channel]) / 255 - value) < 0.005)
            }
            #expect(pixels[offset + 3] == 255)
          }
        }
      }
    }
  }

  @Test func resizingAndFoldingRetainViewsAndKeepToolbarAligned() async throws {
    _ = NSApplication.shared
    let shell = WorkspaceSplitController()
    var navigation = WorkspaceNavigation()
    navigation.sidebarVisible = true
    navigation.panelWidth = 350
    func update() {
      shell.update(
        navigation: navigation, title: "A conversation title that remains visible when enlarged",
        enabled: true, sidebar: AnyView(Color.clear), content: AnyView(Color.clear),
        metrics: AnyView(Color.clear.frame(width: 32, height: 32)))
    }
    update()
    let sidebarView = shell.sidebarHost.view
    let contentView = shell.contentHost.view
    let window = makeWindow(shell)
    defer { window.close() }
    var persistedWidths: [CGFloat] = []
    shell.onWidth = { persistedWidths.append($0) }
    for mode in WorkspaceMode.allCases {
      navigation.mode = mode
      for visible in [true, false, true] {
        navigation.sidebarVisible = visible
        update()
        for width: CGFloat in [1800, 900, 1200, 900, 1800] {
          window.setContentSize(NSSize(width: width, height: 760))
          try await settle(shell.view)
          #expect(shell.sidebarHost.view === sidebarView && shell.contentHost.view === contentView)
          #expect(contentView.bounds.width >= WorkspacePanelMetrics.contentMinimum - 1)
          #expect(shell.sidebarItem.isCollapsed == !visible)
          if visible {
            let expected = min(
              navigation.panelWidth, width - shell.splitView.dividerThickness - 640)
            #expect(abs(sidebarView.frame.width - expected) < 1)
            let separator = try #require(
              window.toolbar?.items.compactMap { $0 as? NSTrackingSeparatorToolbarItem }.first)
            #expect(separator.splitView === shell.splitView && separator.dividerIndex == 0)
          } else {
            #expect(abs(contentView.frame.width - width) < 1)
            #expect(
              window.toolbar?.items.contains { $0 is NSTrackingSeparatorToolbarItem } == false)
          }
          let title = try toolbarTitle(window)
          #expect(!title.stringValue.isEmpty && !title.isHidden)
        }
      }
    }
    #expect(persistedWidths.isEmpty, "Window fitting must not save a temporary sidebar width")
  }

  @Test func toolbarChangesDoNotDuplicateControlsAndActionsRemainReachable() async throws {
    _ = NSApplication.shared
    let shell = WorkspaceSplitController()
    var navigation = WorkspaceNavigation()
    var actions: [String] = []
    shell.onSidebar = { actions.append("sidebar") }
    shell.onNewChat = { actions.append("chat") }
    shell.onSettings = { actions.append("settings") }
    shell.onBack = { actions.append("back") }
    func update(_ enabled: Bool = true) {
      shell.update(
        navigation: navigation, title: "Conversation", enabled: enabled,
        sidebar: AnyView(Color.clear), content: AnyView(Color.clear),
        metrics: AnyView(Color.clear.frame(width: 32, height: 32)))
    }
    update()
    let window = makeWindow(shell)
    defer { window.close() }
    for visible in [false, true, false, true, false] {
      navigation.sidebarVisible = visible
      update()
      try await settle(shell.view)
      let items = try #require(window.toolbar?.items)
      #expect(Set(items.map(\.itemIdentifier)).count == items.count)
      let buttons = items.compactMap { $0.view as? NSButton }
      #expect(buttons.count == (visible ? 2 : 3))
      for button in buttons { button.performClick(nil) }
    }
    #expect(actions.filter { $0 == "sidebar" }.count == 5)
    #expect(actions.filter { $0 == "chat" }.count == 5)
    #expect(actions.filter { $0 == "settings" }.count == 3)
    navigation.mode = .manager
    update()
    try await settle(shell.view)
    let back = try #require(
      window.toolbar?.items.compactMap { $0.view as? NSButton }.first {
        $0.toolTip == "Back to chat"
      })
    back.performClick(nil)
    #expect(actions.last == "back")
    update(false)
    try await settle(shell.view)
    #expect(
      window.toolbar?.items.compactMap { $0.view as? NSButton }.allSatisfy { !$0.isEnabled } == true
    )
    // Full-screen moves the toolbar outside the content frame. Model that
    // layout change here without switching the user's desktop to another Space.
    window.styleMask.remove(.fullSizeContentView)
    try await settle(shell.view)
    #expect(shell.chrome.topInset == 0)
    #expect(try toolbarTitle(window).stringValue == "Conversation")
  }

  private func toolbarTitle(_ window: NSWindow) throws -> NSTextField {
    func fields(_ view: NSView) -> [NSTextField] {
      (view as? NSTextField).map { [$0] } ?? view.subviews.flatMap(fields)
    }
    let titles = (window.toolbar?.items.compactMap(\.view) ?? []).flatMap(fields)
    return try #require(titles.first { $0.accessibilityIdentifier() == "workspace-header-title" })
  }

  private func makeWindow(_ shell: WorkspaceSplitController) -> NSWindow {
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1200, height: 760),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView], backing: .buffered,
      defer: false)
    window.isReleasedWhenClosed = false
    window.toolbarStyle = .unified
    window.contentViewController = shell
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    return window
  }

  private func settle(_ view: NSView) async throws {
    for _ in 0..<3 {
      view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(25))
    }
  }
}
