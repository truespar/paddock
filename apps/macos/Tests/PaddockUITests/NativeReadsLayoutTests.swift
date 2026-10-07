import AppKit
import PaddockClient
import PaddockConversationCore
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native Reads layout", .serialized) @MainActor
struct NativeReadsLayoutTests {
  @Test func inputModePickerFitsOneRowInBothThemes() {
    for dark in [false, true] {
      for cameraMode in [false, true] {
        let host = NSHostingController(
          rootView: NativeReadModePicker(cameraMode: .constant(cameraMode))
            .font(.system(size: 13))
            .environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: NSSize(width: 160, height: 1000))
        #expect(
          size.width <= 160 && size.height <= 28,
          "The input selector must be a single row, not a wrapped label: \(size)")
      }
    }
  }

  @Test func liveCameraAnswersFitBothThemesWithoutCameraAccess() async throws {
    let model = NativeReadsModel(client: NativeManager())
    model.api = { path, _, _, _ in
      let json: String
      if path == "api/runners" {
        json = #"[{"port":1234,"reader":"clef-flash","vendor":"Cloudflare"}]"#
      } else if path.hasSuffix("/server") {
        json =
          #"{"structured_read":{"backend":"clef","images":true,"max_questions":256,"max_samples":1,"max_steps":1,"think":false}}"#
      } else {
        json = "[]"
      }
      return try JSONDecoder().decode(ConversationValue.self, from: Data(json.utf8))
    }
    model.readAPI = { _, _ in
      try JSONDecoder().decode(
        ConversationValue.self,
        from: Data(
          #"{"model":"clef-flash","answers":{"q1":{"type":"noul","noul":0.98,"confidence":0.96,"answer_confidence":0.98}},"usage":{"input_tokens":260,"output_tokens":0},"diagnostics":{"backend":"clef","reads":1,"images":1,"pictures":[{"width":512,"height":288,"resized":[512,288],"tokens":144}],"questions":[],"timing":{"total_ms":12,"gpu_ms":10,"pass_requests":1}}}"#
            .utf8))
    }
    await model.refresh()
    model.draft.questions[0].instructions = "Is a person visible at the door in the picture?"
    let camera = NativeReadCamera()
    camera.startLoop(model: model, frame: { _ in Data([1]) }, visible: { true })
    for _ in 0..<1000 {
      if camera.latest != nil { break }
      await Task.yield()
    }
    camera.pause()
    let response = try #require(camera.latest?.response)
    #expect(response.diagnostics.images == 1)
    #expect(response.diagnostics.pictures?.first?.tokens == 144)
    #expect(response.diagnostics.pictures?.first?.resized == [512, 288])
    for dark in [false, true] {
      for width: CGFloat in [344, 392, 700] {
        let host = NSHostingController(
          rootView:
            NativeReadCameraPanel(model: model, camera: camera, close: {})
            .environment(\.colorScheme, dark ? .dark : .light))
        #expect(host.sizeThatFits(in: NSSize(width: width, height: 4000)).width <= width + 1)
      }
    }
    #expect(camera.capture == nil && !camera.opening)
    camera.close()
  }

  @Test func cameraControlsFitWithoutOpeningCameraInBothThemes() {
    let model = NativeReadsModel(client: NativeManager())
    let camera = NativeReadCamera()
    for dark in [false, true] {
      for width: CGFloat in [344, 392, 700] {
        let host = NSHostingController(
          rootView:
            NativeReadCameraPanel(model: model, camera: camera, close: {})
            .environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: NSSize(width: width, height: 2000))
        #expect(size.width <= width + 1, "Camera controls overflow \(width): \(size)")
      }
    }
    #expect(camera.capture == nil && !camera.opening)
  }
  @Test func readsHistoryFitsTheSameSidebarWidthsInBothThemes() async throws {
    let model = NativeReadsModel(client: NativeManager())
    model.api = { _, _, _, _ in
      try JSONDecoder().decode(
        ConversationValue.self,
        from: Data(
          #"[{"id":"one","title":"A long saved read title that must truncate inside the sidebar instead of expanding its width","model":"diffusion","runs":3,"updatedAt":1000}]"#
            .utf8))
    }
    await model.refreshHistory()
    for dark in [false, true] {
      for width: CGFloat in [220, 260, 340] {
        let host = NSHostingController(
          rootView: NativeReadsSidebar(model: model).environment(
            \.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: NSSize(width: width, height: 650))
        #expect(size.width <= width + 1 && size.height <= 650)
        var navigation = WorkspaceNavigation()
        navigation.studio = .reads
        let footer = NSHostingController(
          rootView: StudioSidebarFooter(navigation: .constant(navigation)))
        // Tables adds one destination; reserve its row instead of squeezing
        // the existing, accessible 32-point footer targets.
        #expect(footer.sizeThatFits(in: NSSize(width: width, height: 168)).height <= 168)
      }
    }
  }
  @Test func nativeMenuCannotExpandProviderArtworkToItsOriginalSVGSize() {
    for vendor in ProviderArtwork.names.keys.sorted() {
      let host = NSHostingController(
        rootView: Menu {
          Button("Select") {}
        } label: {
          HStack(spacing: 8) {
            ModelProviderLogo(vendor: vendor)
            Text("DiffusionGemma 26B A4B")
          }
        }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize())
      let size = host.sizeThatFits(in: NSSize(width: 600, height: 600))
      #expect(size.height <= 36, "Oversized native menu trigger for \(vendor): \(size)")
      #expect(size.width <= 340)
    }
  }

  @Test func readsFormFitsNarrowAndWideWindowsInBothThemes() async throws {
    _ = NSApplication.shared
    let model = NativeReadsModel(client: NativeManager())
    model.api = { path, _, _, _ in
      let raw: String
      if path == "api/runners" {
        raw =
          #"[{"port":1234,"model":"diffusiongemma","display":"DiffusionGemma 26B A4B","vendor":"Google"}]"#
      } else if path.hasSuffix("/server") {
        raw =
          #"{"structured_read":{"canvas_width":256,"max_questions":64,"max_samples":32,"types":["noul","choice","score"],"images":true,"max_steps":8,"think":true}}"#
      } else {
        raw = "[]"
      }
      return try JSONDecoder().decode(ConversationValue.self, from: Data(raw.utf8))
    }
    await model.refresh()
    model.draft.images = [
      ReadPicture(name: "A long image filename with spaces.png", url: "data:image/png;base64,YQ==")
    ]
    for dark in [false, true] {
      for width: CGFloat in [440, 680, 1200] {
        let host = NSHostingController(
          rootView: NativeReadsView(model: model, onStart: {})
            .environment(\.colorScheme, dark ? .dark : .light))
        host.sizingOptions = []
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: width, height: 1000),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentViewController = host
        window.setContentSize(NSSize(width: width, height: 1000))
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(100))
        host.view.layoutSubtreeIfNeeded()
        #expect(abs(host.view.frame.width - width) < 1)
        let scroll = try #require(allViews(host.view).compactMap { $0 as? NSScrollView }.first)
        #expect((scroll.documentView?.frame.width ?? 0) <= width + 1)
        for control in allViews(host.view).compactMap({ $0 as? NSPopUpButton }) {
          #expect(control.frame.height <= 36)
          if let image = control.image { #expect(image.size.height <= 18) }
        }
        if let folder = ProcessInfo.processInfo.environment["PADDOCK_READS_SNAPSHOTS"],
          let bitmap = host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds)
        {
          host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
          try bitmap.representation(using: .png, properties: [:])?.write(
            to: URL(fileURLWithPath: folder).appending(
              path: "reads-\(Int(width))-\(dark ? "dark" : "light").png"))
        }
      }
    }
  }

  @Test func everyQuestionTypeFitsTheNarrowColumn() {
    for kind in ReadQuestion.Kind.allCases {
      let question = ReadQuestion(questionID: "question_id", kind: kind)
      let host = NSHostingController(
        rootView: NativeReadQuestionRow(
          question: .constant(question), onDuplicate: {}, onMove: { _ in }, onRemove: {}))
      let size = host.sizeThatFits(in: NSSize(width: 356, height: 1000))
      #expect(size.width <= 356, "Question controls must fit the column: \(kind), \(size)")
      #expect(size.height < 380)
    }
  }

  private func allViews(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(allViews)
  }
}
