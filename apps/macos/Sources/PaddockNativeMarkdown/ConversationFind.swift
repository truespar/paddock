import Foundation
import Markdown
import Observation
import SwiftUI

extension EnvironmentValues {
  @Entry public var conversationFind: ConversationFind? = nil
}

/// Source-only indexing: never retains an AST, text view or offscreen layout.
/// IDs and UTF-16 ranges keep the current result stable while a reply streams.
@Observable @MainActor public final class ConversationFind {
  public struct Match: Equatable, Sendable {
    public let itemID: String
    public let range: NSRange
    public let occurrence: Int
  }
  public private(set) var presented = false
  public private(set) var query = ""
  public private(set) var matches: [Match] = []
  public private(set) var position = 0
  public private(set) var searching = false
  public private(set) var truncated = false
  public private(set) var focusRevision = 0
  public private(set) var navigationRevision = 0
  public private(set) var revealFailed = false
  public var current: Match? {
    presented && matches.indices.contains(position) ? matches[position] : nil
  }
  @ObservationIgnored private var items: [ConversationSelectionItem] = []
  @ObservationIgnored private var worker: Task<Void, Never>?
  @ObservationIgnored private var revision = 0
  @ObservationIgnored private var resumeMatch: Match?
  @ObservationIgnored private var pendingStep = 0
  @ObservationIgnored private let index = ConversationFindIndex()

  public init() {}
  deinit { worker?.cancel() }

  public func show() {
    presented = true
    focusRevision &+= 1
    navigationRevision &+= 1
    schedule()
  }
  public func close() {
    if let current { resumeMatch = current }
    pendingStep = 0
    presented = false
    worker?.cancel()
    worker = nil
    searching = false
    revision &+= 1
    navigationRevision &+= 1
    revealFailed = false
    matches = []
    position = 0
    truncated = false
    Task { [index] in await index.clear() }
  }
  public func search(_ value: String) {
    guard query != value else { return }
    query = value
    resumeMatch = nil
    pendingStep = 0
    worker?.cancel()
    worker = nil
    matches = []
    position = 0
    truncated = false
    revealFailed = false
    navigationRevision &+= 1
    revision &+= 1
    schedule()
  }
  public func update(_ items: [ConversationSelectionItem]) {
    guard self.items != items else { return }
    if !Self.isAppendOnly(from: self.items, to: items) {
      resumeMatch = nil
      matches = []
      position = 0
      truncated = false
      revealFailed = false
      navigationRevision &+= 1
    }
    self.items = items
    revision &+= 1
    schedule()
  }
  public func move(_ delta: Int) {
    if !presented, !query.isEmpty {
      pendingStep = delta
      show()
      return
    }
    guard !matches.isEmpty else { return }
    position = (position + delta % matches.count + matches.count) % matches.count
    revealFailed = false
    navigationRevision &+= 1
  }
  public func reportRevealFailure(revision: Int) {
    if revision == navigationRevision, current != nil { revealFailed = true }
  }

  private func schedule() {
    guard presented, !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      searching = false
      return
    }
    guard worker == nil else { return }
    searching = true
    worker = Task { [weak self, index] in
      do {
        // Coalesce typing and streamed deltas, without resetting the timer on
        // every token and starving an actively streaming conversation's search.
        try await Task.sleep(for: .milliseconds(100))
        while let snapshot = self?.snapshot() {
          let result = try await index.search(snapshot.items, query: snapshot.query)
          try Task.checkCancellation()
          guard let self else { return }
          if query == snapshot.query,
            revision == snapshot.revision || Self.isAppendOnly(from: snapshot.items, to: items)
          {
            let previous = current
            matches = result.matches
            truncated = result.truncated
            position =
              (previous ?? resumeMatch).flatMap { old in
                matches.firstIndex { $0.itemID == old.itemID && $0.occurrence == old.occurrence }
              } ?? 0
            if !matches.isEmpty {
              position = (position + pendingStep % matches.count + matches.count) % matches.count
              pendingStep = 0
              resumeMatch = nil
            }
            if previous != current {
              revealFailed = false
              navigationRevision &+= 1
            }
          }
          if revision == snapshot.revision {
            searching = false
            worker = nil
            return
          }
          try await Task.sleep(for: .milliseconds(150))
        }
      } catch {
        // A closed bar or a newer query owns the next publication.
      }
    }
  }
  private func snapshot() -> (items: [ConversationSelectionItem], query: String, revision: Int)? {
    presented ? (items, query, revision) : nil
  }
  private static func isAppendOnly(
    from previous: [ConversationSelectionItem], to next: [ConversationSelectionItem]
  ) -> Bool {
    guard previous.count <= next.count else { return false }
    return zip(previous, next).allSatisfy { old, new in
      old.id == new.id && old.markdown == new.markdown && old.presentation == new.presentation
        && (old.text == new.text || new.text.hasPrefix(old.text))
    }
  }
}

actor ConversationFindIndex {
  struct Result: Sendable {
    var matches: [ConversationFind.Match] = []
    var truncated = false
  }
  private struct Entry {
    let source: String
    let markdown: Bool
    let presentation: ConversationSelectionItem.Presentation
    let text: String
  }
  private var cache: [String: Entry] = [:]
  static let limit = 10_000
  func clear() { cache.removeAll(keepingCapacity: false) }

  func search(_ items: [ConversationSelectionItem], query: String) throws -> Result {
    let ids = Set(items.map(\.id))
    cache = cache.filter { ids.contains($0.key) }
    var result = Result()
    guard !query.isEmpty else { return result }
    for item in items {
      try Task.checkCancellation()
      let text: String
      if let cached = cache[item.id], cached.source == item.text, cached.markdown == item.markdown,
        cached.presentation == item.presentation
      {
        text = cached.text
      } else {
        switch item.presentation {
        case .text: text = item.markdown ? try Self.plainText(item.text) : item.text
        case .toolArguments: text = NativeToolText.arguments(item.text)
        case .toolOutput: text = NativeToolText.output(item.text)
        }
        cache[item.id] = Entry(
          source: item.text, markdown: item.markdown, presentation: item.presentation, text: text)
      }
      let ranges = try Self.ranges(
        in: text, query: query, limit: Self.limit - result.matches.count + 1)
      for (occurrence, range) in ranges.enumerated() {
        guard result.matches.count < Self.limit else {
          result.truncated = true
          return result
        }
        result.matches.append(.init(itemID: item.id, range: range, occurrence: occurrence))
      }
    }
    return result
  }

  nonisolated static func ranges(in text: String, query: String, limit: Int = limit) throws
    -> [NSRange]
  {
    guard !query.isEmpty, limit > 0 else { return [] }
    let source = text as NSString
    var cursor = 0
    var result: [NSRange] = []
    while cursor < source.length, result.count < limit {
      try Task.checkCancellation()
      let range = source.range(
        of: query, options: [.caseInsensitive, .diacriticInsensitive],
        range: NSRange(location: cursor, length: source.length - cursor))
      guard range.location != NSNotFound, range.length > 0 else { break }
      // NSString's diacritic-insensitive search can omit a trailing combining
      // mark. Highlight complete composed characters, including NFD accents.
      let composed = source.rangeOfComposedCharacterSequences(for: range)
      result.append(composed)
      cursor = NSMaxRange(composed)
    }
    return result
  }

  nonisolated static func plainText(_ source: String) throws -> String {
    let document = Document(parsing: try NativeMarkdownPolicy.prepare(source))
    var output = ""
    func append(_ node: any Markup) throws {
      try Task.checkCancellation()
      switch node {
      case let node as Markdown.Text: output += node.string
      case let node as InlineCode: output += node.code
      case let node as InlineHTML: output += node.rawHTML
      case let node as CodeBlock: output += node.code
      case is SoftBreak: output += " "
      case is LineBreak: output += "\n"
      default:
        for child in node.children { try append(child) }
      }
      if node is BlockMarkup || node is Markdown.Table.Cell { output += "\n" }
    }
    try append(document)
    return output
  }
}
