import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native compare model picker", .serialized) @MainActor
struct StudioCompareTests {
  @Test func favoriteAndRecentGroupsKeepProviderPinsAndNeverDuplicateChoices() throws {
    let models = try fixture()
    let pinned = "cloud:account:meta/muse@meta"
    let groups = StudioCompareList.groups(
      models, search: "", includeUnavailable: true,
      favorites: [pinned, "missing"], recent: [pinned, "qwen", "qwen", "missing"])
    #expect(groups.map(\.title) == ["Favorites", "Recent", "OpenRouter"])
    #expect(groups.flatMap(\.models).map(\.id) == [pinned, "qwen", "stopped"])
    let filtered = StudioCompareList.groups(
      models, search: "OpenRouter @meta", favorites: [pinned], recent: ["qwen"])
    #expect(filtered.flatMap(\.models).map(\.id) == [pinned])
    #expect(!StudioCompareList.canApply([pinned, "stopped"], models: models))
  }

  @Test func imageDimensionsSnapToEndpointGridWithoutSilentStaleValues() {
    #expect(StudioImageControls.snappedSide("999", grid: 32, maxSide: 2752, fallback: 1024) == 992)
    #expect(
      StudioImageControls.snappedSide("1008", grid: 32, maxSide: 2752, fallback: 1024) == 1024)
    #expect(
      StudioImageControls.snappedSide("999999", grid: 32, maxSide: 2752, fallback: 1024) == 2752)
    #expect(StudioImageControls.snappedSide("-1", grid: 32, maxSide: 2752, fallback: 1024) == 32)
    #expect(StudioImageControls.snappedSide("", grid: 32, maxSide: 2752, fallback: 768) == 768)
    #expect(
      StudioImageControls.snappedSide(
        "99999999999999999999", grid: 32, maxSide: 2752, fallback: 768) == 768)
  }

  @Test func savedCloudPicksAndLocalModelsShareThePickerWithoutLosingPins() throws {
    let models = try fixture()
    let groups = StudioCompareList.groups(models, search: "")
    #expect(groups.map(\.title) == ["Local", "OpenRouter"])
    #expect(groups.flatMap(\.models).map(\.id) == ["qwen", "cloud:account:meta/muse@meta"])
    #expect(
      StudioCompareList.groups(models, search: "  openrouter muse  ").flatMap(\.models).count == 1)
    #expect(StudioCompareList.groups(models, search: "@meta").flatMap(\.models).count == 1)
    #expect(StudioCompareList.groups(models, search: "not present").isEmpty)
    #expect(StudioCompareList.canApply(["qwen", "cloud:account:meta/muse@meta"], models: models))
    #expect(!StudioCompareList.canApply([], models: models))
    #expect(!StudioCompareList.canApply(["qwen", "stopped"], models: models))
    #expect(!StudioCompareList.canApply(["qwen", "missing"], models: models))
  }

  @Test func compareKeepsTheWebInputAndFourLaneGuards() throws {
    let models =
      try (0..<5).map { try model(id: "text-\($0)") }
      + [model(id: "speech", chat: false, audio: true), model(id: "both", chat: true, audio: true)]
    #expect(StudioCompareList.canApply(["text-0", "text-1", "text-2", "text-3"], models: models))
    #expect(!StudioCompareList.canApply(Set((0..<5).map { "text-\($0)" }), models: models))
    #expect(!StudioCompareList.canApply(["text-0", "speech"], models: models))
    #expect(StudioCompareList.canApply(["both", "speech"], models: models))
    #expect(StudioCompareList.canApply(["both", "text-0"], models: models))
  }

  @Test func compareImagesShareALaneTypeButNeverMixWithChat() throws {
    let models = try [
      model(id: "image-1", chat: false, image: true),
      model(id: "image-2", chat: false, image: true), model(id: "chat"),
    ]
    #expect(StudioCompareList.canApply(["image-1", "image-2"], models: models))
    #expect(!StudioCompareList.canApply(["image-1", "chat"], models: models))
  }

  @Test func clearSelectionEditsOnlyTheDraftIncludingHiddenAndUnavailableModels() throws {
    _ = NSApplication.shared
    let models = try fixture()
    let saved = Set(["qwen", "cloud:account:meta/muse@meta", "stopped", "missing"])
    for dark in [false, true] {
      var draft = saved
      let binding = Binding(get: { draft }, set: { draft = $0 })
      let bar = StudioCompareSelectionBar(selected: binding)
      let host = NSHostingController(
        rootView: bar.frame(width: 348).environment(\.colorScheme, dark ? .dark : .light))
      let fit = host.sizeThatFits(in: CGSize(width: 348, height: 100))
      #expect(fit.width <= 348 && fit.height <= 36)
      #expect(StudioCompareList.groups(models, search: "muse").flatMap(\.models).count == 1)
      bar.clearSelection()
      #expect(draft.isEmpty, "Clear must include filtered, unavailable and missing model IDs")
      #expect(saved.count == 4, "The applied conversation selection must stay unchanged")
      #expect(!StudioCompareList.canApply(draft, models: models))
      bar.clearSelection()
      #expect(draft.isEmpty, "Clearing twice is harmless")
      draft = saved
      StudioCompareSelectionBar(selected: binding, busy: true).clearSelection()
      #expect(draft == saved, "An in-flight action must not clear the draft")
    }
  }

  @Test func localAndCloudRowsHaveARealViewportAndLargerLibrariesScroll() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      for count in [2, 30] {
        let models =
          count == 2 ? try fixture() : try (0..<count).map { try model(id: "cloud-\($0)") }
        let host = NSHostingController(
          rootView:
            StudioCompareList(
              models: models, selected: .constant(["qwen"]),
              favorites: [models[0].id], recent: [models[1].id], onFavorite: { _ in }
            )
            .environment(\.colorScheme, dark ? .dark : .light))
        let fit = host.sizeThatFits(in: CGSize(width: 348, height: 600))
        #expect(fit.width <= 348)
        #expect(abs(fit.height - (count == 2 ? 172 : 300)) < 1)
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: 348, height: fit.height),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        host.sizingOptions = []
        window.contentViewController = host
        window.setContentSize(CGSize(width: 348, height: fit.height))
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(80))
        host.view.layoutSubtreeIfNeeded()
        let scroll = try #require(scrollViews(host.view).first)
        let visible = scroll.contentView.bounds.height
        let content = try #require(scroll.documentView?.frame.height)
        #expect(
          visible >= (count == 2 ? 171 : 299), "The old popover clipped this list to 16 pixels")
        if count == 2 {
          #expect(content <= visible + 1, "Both local and cloud rows must be visible")
        } else {
          #expect(content > visible)
          let bottom = content - visible
          scroll.contentView.scroll(to: NSPoint(x: 0, y: bottom))
          scroll.reflectScrolledClipView(scroll.contentView)
          #expect(abs(scroll.contentView.bounds.origin.y - bottom) < 1)
        }
      }
    }
  }

  @Test func nativeCheckboxIsCenteredInBothThemesAndSelectionStates() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      for selected in [false, true] {
        let host = NSHostingController(
          rootView:
            StudioCompareModelRow(
              model: try model(id: "qwen", title: "Qwen 3.8 27B", port: 11541),
              isSelected: .constant(selected)
            )
            .frame(width: 348).environment(\.colorScheme, dark ? .dark : .light))
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: 348, height: 52),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        host.sizingOptions = []
        window.contentViewController = host
        window.setContentSize(CGSize(width: 348, height: 52))
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(80))
        host.view.layoutSubtreeIfNeeded()
        func buttons(_ view: NSView) -> [NSButton] {
          (view as? NSButton).map { [$0] } ?? view.subviews.flatMap(buttons)
        }
        let checkboxes = buttons(host.view)
        let checkbox = try #require(checkboxes.first)
        #expect(checkboxes.count == 1)
        let control = checkbox.convert(checkbox.bounds, to: host.view)
        #expect(
          abs(control.midY - host.view.bounds.midY) < 1,
          "Center the control beside the icon, not the title baseline")
      }
    }
  }

  private func fixture() throws -> [StudioState.Model] {
    try [
      model(id: "qwen", title: "Qwen 3.8 27B", provider: "Local", port: 12481),
      model(id: "cloud:account:meta/muse@meta", title: "Muse Spark 1.3 (meta)"),
      model(id: "stopped", status: "unreachable"),
    ]
  }
  private func model(
    id: String, title: String = "Cloud model", provider: String = "OpenRouter",
    port: Int? = nil, status: String = "ok", chat: Bool = true, audio: Bool = false,
    image: Bool = false
  ) throws -> StudioState.Model {
    var object: [String: Any] = [
      "id": id, "title": title, "provider": provider, "vendor": "Meta",
      "status": status, "vision": false, "chat": chat, "audio": audio, "image": image,
    ]
    if let port { object["port"] = port }
    return try JSONDecoder().decode(
      StudioState.Model.self, from: JSONSerialization.data(withJSONObject: object))
  }
  private func scrollViews(_ view: NSView) -> [NSScrollView] {
    (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap(scrollViews)
  }
}
