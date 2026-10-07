import Foundation
import Markdown

/// A single off-main admission lane: CommonMark's C parser cannot be interrupted
/// halfway through a parse. Serial admission bounds that uninterruptible work;
/// cancelled queued updates are rejected before allocating a syntax tree.
actor NativeMarkdownPreparation {
  static let shared = NativeMarkdownPreparation()
  struct Prepared: Sendable {
    let text: String
    let diagramSources: Set<String>
  }
  private struct Entry {
    let input: String
    let output: Prepared
    var cost: Int {
      input.utf8.count + output.text.utf8.count
        + output.diagramSources.reduce(0) { $0 + $1.utf8.count }
    }
  }
  private var cache: [Entry] = []
  private let budget = 2 * 1024 * 1024

  func source(_ text: String, streaming: Bool) throws -> String {
    try prepared(text, streaming: streaming).text
  }
  func prepared(_ text: String, streaming: Bool) throws -> Prepared {
    try Task.checkCancellation()
    if let index = cache.firstIndex(where: { $0.input == text }) {
      let entry = cache.remove(at: index)
      cache.append(entry)
      return entry.output
    }
    let source = try NativeMarkdownPolicy.prepare(text)
    var diagrams = DiagramSources()
    if source.localizedCaseInsensitiveContains("mermaid") {
      diagrams.visit(Document(parsing: source))
    }
    let result = Prepared(text: source, diagramSources: diagrams.sources)
    try Task.checkCancellation()
    if !streaming {
      let entry = Entry(input: text, output: result)
      if entry.cost <= budget {
        while !cache.isEmpty && (cache.count >= 8 || retainedBytes + entry.cost > budget) {
          cache.removeFirst()
        }
        cache.append(entry)
      }
    }
    return result
  }
  var retainedBytes: Int { cache.reduce(0) { $0 + $1.cost } }
  func reclaim() { cache.removeAll(keepingCapacity: false) }

  private struct DiagramSources: MarkupWalker {
    var sources = Set<String>()
    mutating func visitCodeBlock(_ block: CodeBlock) {
      if block.language?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "mermaid" {
        // MarkdownView passes code to styles with surrounding newlines removed.
        sources.insert(block.code.trimmingCharacters(in: .newlines))
      }
    }
  }
}
