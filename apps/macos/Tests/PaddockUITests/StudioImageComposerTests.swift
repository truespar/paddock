import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Image composer controls", .serialized) @MainActor
struct StudioImageComposerTests {
  @Test func pictureSettingsFitBothThemesIncludingCustomDimensionsAndPinnedSeed() {
    // Offscreen layout only: no test window or synthetic input in the user's app.
    let caps: [String: StudioValue] = [
      "max_n": .number(1), "output_formats": .array([.string("png")]),
      "stream": .bool(true), "max_partial_images": .number(3),
      "default_size": .string("1024x1024"), "size_multiple": .number(32),
    ]
    for dark in [false, true] {
      for custom in [false, true] {
        let params: [String: StudioValue] =
          custom
          ? ["size": .string("768x1280"), "seed": .number(2_147_483_647), "steps": .number(40)]
          : [:]
        var writes = 0
        let form = StudioImageSettingsForm(
          params: params, caps: caps, update: { _, _ in writes += 1 }, reset: { writes += 1 })
        let host = NSHostingController(
          rootView: form.studioPopoverSurface().environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: CGSize(width: 360, height: 800))
        #expect(size.width == 360)
        #expect(size.height > (custom ? 420 : 340))
        #expect(size.height < (custom ? 500 : 420))
        host.view.frame = NSRect(origin: .zero, size: size)
        host.view.layoutSubtreeIfNeeded()
        #expect(writes == 0, "Rendering saved settings must never write them back")
      }
    }
  }
  @Test func pictureSettingsUsesAnAvailableSymbolAndExactlyOnePresenter() {
    #expect(
      NSImage(
        systemSymbolName: StudioComposerView.pictureSettingsSymbol,
        accessibilityDescription: nil) != nil)
    for panel in StudioComposerView.Panel.allCases {
      #expect(panel.usesOverflow == (panel != .image))
    }
  }
  @Test func imageModeOnlyOffersControlsThatTheEndpointCanUse() {
    for editing in [false, true] {
      for speech in [false, true] {
        let policy = StudioComposerView.ToolsPolicy(
          imageMode: true, imageEditing: editing, audioMode: false, speechAvailable: speech)
        #expect(policy.attachments == editing)
        #expect(
          !policy.speechLanguage, "Previous speech work cannot leak language controls into images")
        #expect(!policy.overflow(compact: true), "Image mode has no overflow items")
        #expect(!policy.overflow(compact: false))
      }
    }
  }
  @Test func chatAndSpeechKeepTheirUsefulCompactControls() {
    for audio in [false, true] {
      for speech in [false, true] {
        let policy = StudioComposerView.ToolsPolicy(
          imageMode: false, imageEditing: false, audioMode: audio, speechAvailable: speech)
        #expect(policy.attachments)
        #expect(policy.speechLanguage == speech)
        #expect(policy.overflow(compact: true) == (!audio || speech))
        #expect(!policy.overflow(compact: false))
      }
    }
  }
}
