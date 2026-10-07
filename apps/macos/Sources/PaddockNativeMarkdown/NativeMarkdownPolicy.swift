import Foundation
import Markdown

/// MarkdownView has HTML/SVG web renderers. Convert those AST nodes before
/// they can reach it. This is not a regex sanitizer: fenced code, indentation,
/// nested blocks and Unicode use CommonMark's own source ranges.
enum NativeMarkdownPolicy {
  static func source(_ text: String) -> String {
    // Synchronous clipboard use has no task cancellation contract.
    (try? prepare(text, checkCancellation: {})) ?? text
  }
  static func prepare(
    _ text: String, checkCancellation: () throws -> Void = { try Task.checkCancellation() }
  ) throws -> String {
    try checkCancellation()
    guard text.contains("<") || text.contains("![") else { return text }
    let bytes = Array(text.utf8)
    var starts = [0]
    for (i, byte) in bytes.enumerated() {
      if i.isMultiple(of: 16384) { try checkCancellation() }
      if byte == 10 { starts.append(i + 1) }
    }
    func offset(_ location: SourceLocation) -> Int {
      guard location.line > 0, location.line <= starts.count else { return bytes.count }
      return min(bytes.count, starts[location.line - 1] + max(0, location.column - 1))
    }
    var walker = EmbeddedNodes()
    walker.visit(Document(parsing: text))
    try checkCancellation()
    // Build once in source order. Repeated inserts/replacements in a multi-MB
    // array turn image-heavy documents into quadratic copying.
    var result: [UInt8] = []
    result.reserveCapacity(bytes.count)
    var cursor = 0
    for node in walker.nodes.sorted(by: { $0.range.lowerBound < $1.range.lowerBound }) {
      try checkCancellation()
      let lower = offset(node.range.lowerBound)
      let upper = offset(node.range.upperBound)
      guard lower >= cursor, lower < upper else { continue }
      result.append(contentsOf: bytes[cursor..<lower])
      let raw = String(decoding: bytes[lower..<upper], as: UTF8.self)
      if node.html {
        // A fence longer than every source run cannot be closed by its body.
        let length = max(3, (raw.split(whereSeparator: { $0 != "`" }).map(\.count).max() ?? 0) + 1)
        let fence = String(repeating: "`", count: length)
        result.append(contentsOf: "\n\(fence)html\n\(raw)\n\(fence)\n".utf8)
      } else {
        // The link remains visible/clickable, but no image renderer can fetch
        // arbitrary URLs or instantiate a WebKit SVG surface automatically.
        result.append(92)
        result.append(contentsOf: bytes[lower..<upper])
      }
      cursor = upper
    }
    result.append(contentsOf: bytes[cursor...])
    return String(decoding: result, as: UTF8.self)
  }
  private struct EmbeddedNodes: MarkupWalker {
    struct Node {
      let range: SourceRange
      let html: Bool
    }
    var nodes: [Node] = []
    mutating func visitHTMLBlock(_ node: HTMLBlock) {
      if let range = node.range { nodes.append(Node(range: range, html: true)) }
    }
    mutating func visitImage(_ node: Markdown.Image) {
      if let range = node.range { nodes.append(Node(range: range, html: false)) }
    }
  }
}
