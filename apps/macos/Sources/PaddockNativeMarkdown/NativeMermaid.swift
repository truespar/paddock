import AppKit
import BeautifulMermaid
import PaddockDesign
import SwiftUI

/// Parsing/layout is serialized off-main across all chat messages. No global
/// image cache: the visible block owns its positioned graph and releases it on
/// unmount. Resizing and appearance changes only redraw, never relayout.
actor MermaidLayoutWorker {
  static let shared = MermaidLayoutWorker()
  struct Limit: LocalizedError {
    var errorDescription: String? {
      "This diagram exceeds the layout budget. Its source is preserved below."
    }
  }
  func layout(_ source: String) throws -> PositionedGraph {
    try Task.checkCancellation()
    guard source.utf8.count <= 32 * 1024,
      source.split(whereSeparator: { $0 == "\n" || $0 == ";" }).count <= 256
    else { throw Limit() }
    let parsed = try MermaidRenderer.parse(source)
    if case .flowchart(let graph) = parsed.typedPayload {
      guard graph.nodesInOrder.count <= 256, graph.edges.count <= 512 else { throw Limit() }
    }
    if case .stateDiagram(let graph) = parsed.typedPayload {
      guard graph.nodesInOrder.count <= 256, graph.edges.count <= 512 else { throw Limit() }
    }
    try Task.checkCancellation()
    let graph = try GraphLayout().layout(parsed)
    try Task.checkCancellation()
    guard graph.width.isFinite, graph.height.isFinite, graph.width > 0, graph.height > 0 else {
      throw Limit()
    }
    return graph
  }
}

struct NativeMermaid: View {
  let source: String
  let streaming: Bool
  @Environment(\.colorScheme) private var colorScheme
  @Environment(\.nativeMarkdownWork) private var markdownWork
  @State private var graph: PositionedGraph?
  @State private var error: String?
  @State private var sourceVisible = false
  @State private var pinID = UUID()
  @State private var width: CGFloat = 700
  @State private var renderedSource = ""
  @State private var geometrySource: String?
  private struct Measurement: Equatable {
    let size: CGSize
    let source: String?
  }
  private var dark: Bool { colorScheme == .dark }
  private var pending: Bool { renderedSource != source }
  var body: some View {
    let measuredSource = geometrySource
    VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 10) {
        Text("Mermaid").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
        if pending && error == nil { Text("Updating…").font(.caption).foregroundStyle(.secondary) }
        Spacer()
        Button(sourceVisible ? "Diagram" : "Source") {
          // Explicit interaction pins this message's local controls. Do not
          // destroy a source disclosure just because the user scrolls away.
          sourceVisible.toggle()
          if sourceVisible {
            markdownWork?.pins.insert(pinID)
          } else {
            markdownWork?.pins.remove(pinID)
          }
        }
        Button("Copy") {
          NSPasteboard.general.clearContents()
          NSPasteboard.general.setString(source, forType: .string)
        }
      }.buttonStyle(.plain).font(.caption)
      if !sourceVisible, let graph {
        DiagramCanvas(graph: graph, dark: dark, source: renderedSource)
          .frame(height: min(520, max(100, width * graph.height / graph.width)))
          .opacity(pending ? 0.5 : 1)
      }
      if let error {
        Text(streaming ? "Waiting for a complete diagram…" : "Diagram could not render: \(error)")
          .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
      }
      if sourceVisible || error != nil {
        PaddockScrollView(.horizontal) {
          Text(verbatim: source).font(.system(size: 12, design: .monospaced)).textSelection(
            .enabled)
        }.frame(maxHeight: 320)
      }
    }
    .padding(14)
    .background(
      PaddockAppearance.surface, in: RoundedRectangle(cornerRadius: PaddockAppearance.Radius.card)
    )
    .overlay(
      RoundedRectangle(cornerRadius: PaddockAppearance.Radius.card).strokeBorder(
        PaddockAppearance.border)
    )
    .onGeometryChange(for: CGFloat.self) {
      $0.size.width - 28
    } action: {
      width = max(1, $0)
    }
    .onGeometryChange(for: Measurement.self) {
      Measurement(size: $0.size, source: measuredSource)
    } action: { measurement in
      guard measurement.source == source,
        measurement.size.width > 0, measurement.size.height > 0
      else { return }
      // Returning positioned nodes is not the same as installing their view
      // geometry. Only the latter may authorize the parent's height cache.
      markdownWork?.laidOut(source)
    }
    .task(id: source) {
      markdownWork?.begin(source)
      defer { markdownWork?.end(source, completed: false) }
      do {
        error = nil
        geometrySource = nil
        if streaming { try await Task.sleep(for: .milliseconds(180)) }
        let next = try await MermaidLayoutWorker.shared.layout(source)
        try Task.checkCancellation()
        graph = next
        renderedSource = source
        geometrySource = source
      } catch is CancellationError {
        // Superseded partial syntax never replaces the latest graph.
      } catch {
        guard !Task.isCancelled else { return }
        self.error = error.localizedDescription
        geometrySource = source
      }
    }
    .onDisappear { markdownWork?.pins.remove(pinID) }
  }
}

private struct DiagramCanvas: NSViewRepresentable {
  let graph: PositionedGraph
  let dark: Bool
  let source: String
  func makeNSView(context: Context) -> NativeDiagramView { NativeDiagramView() }
  func updateNSView(_ view: NativeDiagramView, context: Context) {
    guard view.source != source || view.dark != dark else { return }
    view.source = source
    view.dark = dark
    view.graph = graph
    view.setAccessibilityLabel("Mermaid diagram. Source: \(source)")
    view.needsDisplay = true
  }
}

final class NativeDiagramView: NSView {
  var graph: PositionedGraph?
  var source = ""
  var dark = false
  override init(frame: NSRect) {
    super.init(frame: frame)
    setAccessibilityElement(true)
    setAccessibilityRole(.image)
  }
  required init?(coder: NSCoder) { nil }
  override func draw(_ dirtyRect: NSRect) {
    guard let graph, let context = NSGraphicsContext.current?.cgContext else { return }
    // BeautifulMermaid flips its AppKit labels locally. Keep NSGraphicsContext
    // unflipped, then flip only the CGContext to the layout's top-left origin;
    // an isFlipped NSView double-flips the labels and draws them upside down.
    context.saveGState()
    defer { context.restoreGState() }
    context.translateBy(x: 0, y: bounds.height)
    context.scaleBy(x: 1, y: -1)
    // Theme changes redraw the retained geometry; they never trigger layout.
    let theme = DiagramTheme(
      background: PaddockAppearance.nsColor("surface", dark: dark),
      foreground: PaddockAppearance.nsColor("primary", dark: dark),
      line: PaddockAppearance.nsColor("secondary", dark: dark),
      accent: PaddockAppearance.nsColor("accent", dark: dark),
      muted: PaddockAppearance.nsColor("secondary", dark: dark),
      surface: PaddockAppearance.nsColor("elevated", dark: dark),
      border: PaddockAppearance.nsColor("borderStrong", dark: dark),
      cornerRadius: PaddockAppearance.Radius.control)
    DiagramRenderer(theme: theme).render(graph, in: context, bounds: bounds)
  }
}
