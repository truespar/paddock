import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockNativeMarkdown
@testable import PaddockUI

/// All scrolling is sent directly to this offscreen NSClipView. Never posts
/// HID events, activates a window, or touches the user's running application.
@Suite("Transcript viewport residency", .serialized, .timeLimit(.minutes(2))) @MainActor
struct NativeTranscriptResidencyTests {
  @Test(arguments: [false, true])
  func offscreenTextIsReleasedWithoutChangingMeasuredScrollExtent(richContent: Bool) async throws {
    let messages: [[String: Any]] = (0..<40).map { index in
      let rich = """

        ```mermaid
        graph LR
          A[Input \(index)] --> B[Result]
        ```

        | Stage | State |
        | --- | --- |
        | Parse | Complete |

        ```swift
        let message = \(index)
        print(message)
        ```

        """
      return
        [
          "id": "m\(index)", "role": "assistant", "model": "fixture",
          "text": (0..<6).map {
            "Message \(index), paragraph \($0). "
              + String(repeating: "Native text keeps its geometry. ", count: 8)
          }.joined(separator: "\n\n") + (richContent ? rich : ""),
          "reasoning": "", "streaming": false, "stopped": false, "error": "", "incomplete": false,
        ]
    }
    let transcript = try JSONDecoder().decode(
      StudioState.NativeTranscript.self,
      from: JSONSerialization.data(withJSONObject: [
        "available": true, "notice": "", "messages": messages,
      ]))
    let host = NSHostingController(
      rootView: NativeStudioTranscript(
        transcript: transcript, columnWidth: 720, composerHeight: 0
      ).frame(width: 780, height: 650))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 780, height: 650),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await settle(host.view, milliseconds: richContent ? 8000 : 2500)
    let scroll = try #require(descendants(host.view).compactMap { $0 as? NSScrollView }.first)
    let document = try #require(scroll.documentView)
    let height = document.bounds.height
    #expect(height > 15000)
    let mounted = textCount(document)
    print(
      "40-message viewport (rich=\(richContent)): \(mounted) mounted text views, \(height) pt scroll extent"
    )
    let maximum = richContent ? 16 : 12
    #expect(
      mounted > 0 && mounted < maximum, "Only viewport text hosts should be resident: \(mounted)")
    var diagrams = descendants(document).filter { $0 is NativeDiagramView }.count
    let offsets =
      Array(stride(from: CGFloat(0), to: height - 650, by: 500)) + [max(0, height - 650), 0]
    for offset in offsets {
      scroll.contentView.scroll(to: NSPoint(x: 0, y: offset))
      scroll.reflectScrolledClipView(scroll.contentView)
      try await settle(host.view, milliseconds: 150)
      #expect(
        abs(document.bounds.height - height) < 2, "Unmounting must not change measured row heights")
      #expect(textCount(document) > 0 && textCount(document) < maximum)
      diagrams += descendants(document).filter { $0 is NativeDiagramView }.count
    }
    if richContent { #expect(diagrams > 0, "Viewport remount must restore real native diagrams") }
  }
  private func textCount(_ view: NSView) -> Int {
    descendants(view).compactMap { $0 as? NSTextView }.filter { !$0.string.isEmpty }.count
  }
  private func descendants(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(descendants)
  }
  private func settle(_ view: NSView, milliseconds: Int) async throws {
    for _ in 0..<(milliseconds / 25) {
      view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(25))
    }
  }
}
