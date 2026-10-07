import AppKit
import MarkdownView
import PaddockDesign
import SwiftUI

extension EnvironmentValues {
  @Entry public var nativeMarkdownViewportReclamation = false
}

/// One retained incremental parser per message. A completed source cannot be
/// reused for a continuation: replacing it then starts a fresh parsing cycle.
public struct NativeMarkdown: View, Equatable {
  public static func reclaimCaches() async { await NativeMarkdownPreparation.shared.reclaim() }
  public let text: String
  public let streaming: Bool
  public let textSize: CGFloat
  public let unsureWords: [String]
  @State private var source = StreamingMarkdownSource()
  @State private var finished = false
  @State private var renderedHeight: CGFloat?
  @State private var candidateHeight: CGFloat?
  @State private var visible = false
  @State private var admitted = false
  @State private var measuredWidth: CGFloat = 0
  @State private var work = NativeMarkdownWork()
  @State private var reclaimable = false
  @Environment(\.nativeMarkdownViewportReclamation) private var viewportReclamation

  public init(
    _ text: String, streaming: Bool = false, textSize: CGFloat = 15, unsureWords: [String] = []
  ) {
    self.text = text
    self.streaming = streaming
    self.textSize = textSize
    self.unsureWords = unsureWords
  }
  public nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.text == rhs.text && lhs.streaming == rhs.streaming && lhs.textSize == rhs.textSize
      && lhs.unsureWords == rhs.unsureWords
  }
  public var body: some View {
    Group {
      if !viewportReclamation || admitted {
        richText
      } else {
        // A real measured height, never LazyVStack's estimates. Parent message
        // controls/disclosures keep their state while TextKit/ASTs are released.
        Color.clear.frame(height: renderedHeight ?? 22)
      }
    }
    .onScrollVisibilityChange(threshold: 0.001) { visible = $0 }
    .onGeometryChange(for: CGFloat.self) {
      $0.size.width
    } action: { width in
      if measuredWidth > 0, abs(width - measuredWidth) > 1 {
        renderedHeight = nil
        candidateHeight = nil
      }
      measuredWidth = width
    }
    .task(id: MountInput(visible: visible, streaming: streaming, measured: renderedHeight != nil)) {
      guard viewportReclamation, !admitted, visible || streaming || renderedHeight == nil else {
        return
      }
      do {
        try await NativeMarkdownMounts.shared.admit(visible: visible || streaming)
        try Task.checkCancellation()
        work.beginMount()
        admitted = true
      } catch {}
    }
    .onChange(of: text) { _, _ in
      reclaimable = false
      if !visible && !streaming { renderedHeight = nil }
    }
    .onChange(of: visible) { _, _ in releaseMeasuredHost() }
    .onChange(of: work.pinned) { _, _ in releaseMeasuredHost() }
    .onChange(of: textSize) { _, _ in
      renderedHeight = nil
      candidateHeight = nil
      reclaimable = false
    }
    .task(
      id: Completion(
        height: candidateHeight, pending: work.remaining, text: text, width: measuredWidth)
    ) {
      reclaimable = false
      guard let candidateHeight, work.remaining == 0 else { return }
      // Allow child attachment tasks to enter, and their final geometry to
      // cross the nested AppKit host, before releasing this subtree.
      do {
        try await Task.sleep(for: .milliseconds(50))
        try Task.checkCancellation()
        guard work.remaining == 0, self.candidateHeight == candidateHeight else { return }
        renderedHeight = candidateHeight
        reclaimable = true
        releaseMeasuredHost()
      } catch {}
    }
  }
  private func releaseMeasuredHost() {
    guard viewportReclamation, admitted, !visible, !streaming, reclaimable,
      !work.pinned, work.remaining == 0, renderedHeight != nil
    else { return }
    guard let height = work.surface?.completedTextHeight() else { return }
    renderedHeight = height
    candidateHeight = height
    // Only a new admission may restore the subtree. Late child teardown/work
    // callbacks must not remount it by changing measurement bookkeeping.
    admitted = false
  }
  private struct Completion: Equatable {
    let height: CGFloat?
    let pending: Int
    let text: String
    let width: CGFloat
  }
  private struct MountInput: Equatable {
    let visible: Bool
    let streaming: Bool
    let measured: Bool
  }
  private var richText: some View {
    StableMarkdownRender(
      key: RenderIdentity(
        text: text, streaming: streaming, finished: finished, size: textSize,
        unsureWords: unsureWords, source: ObjectIdentifier(source))
    ) {
      formattedText
    }
    .equatable()
    .frame(height: renderedHeight)
    .task(id: RenderInput(text: text, streaming: streaming)) {
      do {
        let value = try await NativeMarkdownPreparation.shared.prepared(text, streaming: streaming)
        try Task.checkCancellation()
        // Register expected work before TextKit creates any inline hosts.
        // An offscreen diagram's task may start later than its parent layout.
        work.expect(value.diagramSources)
        update(value.text)
      } catch is CancellationError {
        // A newer source or disappearance owns the next render.
      } catch {
        assertionFailure("Markdown preparation failed: \(error)")
      }
    }
  }
  private struct RenderIdentity: Equatable, Sendable {
    let text: String
    let streaming: Bool
    let finished: Bool
    let size: CGFloat
    let unsureWords: [String]
    let source: ObjectIdentifier
  }
  private var formattedText: some View {
    StreamingMarkdownReader(source) { result in
      // A single AppKit text surface, rather than independently selectable
      // SwiftUI blocks. Embedded code, math and diagrams keep their renderers.
      SelectableMarkdownSurface(unsureWords: streaming ? [] : unsureWords) {
        MarkdownText(result)
          .markdownCodeBlockStyle(NativeCodeBlockStyle(streaming: streaming, work: work))
          .textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
          .onGeometryChange(for: CGFloat.self) {
            $0.size.height
          } action: { height in
            // TextKit tears down inline attachments during unmount and can
            // report a final, shrunken geometry. It no longer owns this row's
            // measurement once the placeholder has taken over.
            guard !viewportReclamation || admitted else { return }
            // Cross the nested AppKit hosting boundary explicitly. Otherwise
            // a lazy parent can retain the empty parse's height until an
            // unrelated composer/selection update, clipping the first answer.
            // The reader first lays out an empty parse. That is not the
            // message's measured height and must never authorize eviction.
            guard result.sourceSnapshot.text == source.text, finished || streaming else { return }
            if height.isFinite && (height > 0 || text.isEmpty) {
              candidateHeight = height
              // Reopening an unchanged measured row must not expose the
              // attachment tree's intermediate heights to the scroll view.
              if renderedHeight == nil || streaming { renderedHeight = height }
            }
          }
      }
    }
    .markdownFontGroup(ChatMarkdownFonts(size: textSize))
    .environment(\.nativeMarkdownWork, work)
    .font(.system(size: textSize))
    .foregroundStyle(PaddockAppearance.primary)
    .tint(PaddockAppearance.accent)
    .lineSpacing(4)
    .markdownComponentSpacing(12)
    .tint(.primary, for: .inlineCodeBlock)
    .markdownMathRenderingEnabled()
    .markdownStreamingRenderThrottle(.milliseconds(33))
    // Loading arbitrary assistant-authored image URLs is not part of this
    // renderer test. Original attachments keep their authenticated web viewer.
    .markdownElementRenderer(.image(DeferredImage(), urlScheme: "https"))
    .markdownElementRenderer(.image(DeferredImage(), urlScheme: "http"))
    .markdownElementRenderer(.image(DeferredImage(), urlScheme: "file"))
    .environment(
      \.openURL,
      OpenURLAction { url in
        guard ["https", "http", "mailto"].contains(url.scheme?.lowercased() ?? "") else {
          return .discarded
        }
        return .systemAction
      }
    )
  }
  private struct RenderInput: Equatable {
    let text: String
    let streaming: Bool
  }
  private func update(_ value: String) {
    if finished && (source.text != value || streaming) {
      source = StreamingMarkdownSource(value)
      finished = false
    } else {
      source.text = value
    }
    if !streaming {
      source.finishStreaming()
      finished = true
    }
  }
}

/// Measurement/work bookkeeping must not rebuild the attachment configuration:
/// doing so restarts child layout, which writes more bookkeeping in a loop.
/// Actual source, appearance environment and width updates still propagate.
private struct StableMarkdownRender<Key: Equatable & Sendable, Content: View>: View, Equatable {
  let key: Key
  let content: Content
  init(key: Key, @ViewBuilder content: () -> Content) {
    self.key = key
    self.content = content()
  }
  nonisolated static func == (lhs: Self, rhs: Self) -> Bool { lhs.key == rhs.key }
  var body: some View { content }
}

/// Do not inherit the library's smaller macOS body text or serif block quotes.
/// Platform fonts also keep the same metrics on our macOS 15 deployment target.
private struct ChatMarkdownFonts: MarkdownFontGroup {
  let size: CGFloat
  var body: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size) }
  var h1: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size + 9, weight: .semibold) }
  var h2: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size + 5, weight: .semibold) }
  var h3: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size + 2, weight: .semibold) }
  var h4: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size, weight: .semibold) }
  var h5: any CustomCTFontConvertible { h4 }
  var h6: any CustomCTFontConvertible { h4 }
  var blockQuote: any CustomCTFontConvertible { body }
  var codeBlock: any CustomCTFontConvertible {
    NSFont.monospacedSystemFont(ofSize: size - 2, weight: .regular)
  }
  var tableBody: any CustomCTFontConvertible { body }
  var tableHeader: any CustomCTFontConvertible { h4 }
  var inlineMath: any CustomCTFontConvertible { body }
  var displayMath: any CustomCTFontConvertible { body }
}

private struct NativeCodeBlockStyle: MarkdownCodeBlockStyle {
  let streaming: Bool
  let work: NativeMarkdownWork
  @ViewBuilder func makeBody(configuration: Configuration) -> some View {
    if configuration.language?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      == "mermaid"
    {
      NativeMermaid(source: configuration.code, streaming: streaming)
        // Inline attachments create their own hosting controllers. Pass our
        // work tracker explicitly; arbitrary environment keys do not cross
        // the library's attachment boundary with its rendering configuration.
        .environment(\.nativeMarkdownWork, work)
    } else {
      NativeCodeBlock(
        code: configuration.code, language: configuration.language, streaming: streaming)
    }
  }
}

private struct DeferredImage: MarkdownImageRenderer {
  func makeBody(configuration: Configuration) -> some View {
    Label(configuration.alternativeText ?? "Image", systemImage: "photo")
      .font(.callout).foregroundStyle(.secondary)
      .help(
        "Remote images are not fetched automatically. Open an attached original in the native image viewer."
      )
  }
}
