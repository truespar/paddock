import Foundation

/// Web Studio's persisted OCR schema, accepting pre-migration native snake_case
/// rows as well. Only meaningful facts are presented, not raw boolean flags.
enum NativeOCRMetadata {
  typealias V = ConversationValue
  typealias O = [String: V]
  static func stored(_ value: V?) -> V? {
    guard let raw = value?.object else { return nil }
    var result: O = [:]
    for key in ["mode", "crop"] { if let text = raw[key]?.string { result[key] = .string(text) } }
    for (saved, wire) in [
      ("grounding", "grounding"), ("passThrough", "pass_through"), ("droppedText", "dropped_text"),
    ] {
      result[saved] = .bool((raw[saved] ?? raw[wire])?.bool == true)
    }
    for (saved, wire) in [
      ("pages", "pages"), ("views", "views"), ("tiles", "tiles"), ("imageTokens", "image_tokens"),
    ] {
      if let count = (raw[saved] ?? raw[wire])?.integer, (0...1_000_000_000).contains(count) {
        result[saved] = .number(Decimal(count))
      }
    }
    result["regions"] = .array(NativeDocumentRunState.regions(raw["regions"]?.array ?? []))
    return .object(result)
  }
  static func facts(_ value: V?) -> [V] {
    guard let meta = stored(value)?.object else { return [] }
    var result: [V] = []
    func add(_ label: String, _ value: String) {
      result.append(.object(["label": .string(label), "value": .string(value)]))
    }
    if meta["passThrough"]?.bool == true {
      add("Read as", "Your prompt")
    } else if let mode = meta["mode"]?.string, !mode.isEmpty {
      let labels = [
        "document": "Document", "multipage": "Pages of one document", "free": "Plain text",
        "layout": "Layout map", "figure": "Figure", "ocr": "Text", "table": "Table",
        "formula": "Formula", "chart": "Chart", "spotting": "Text spotting", "seal": "Seal",
        "plain": "Markdown", "grounding": "Blocks with boxes",
      ]
      add("Read as", labels[mode] ?? mode.prefix(1).uppercased() + mode.dropFirst())
    }
    if let pages = meta["pages"]?.integer, pages > 1 { add("Pages", String(pages)) }
    if meta["crop"]?.string == "gundam", let tiles = meta["tiles"]?.integer, tiles > 0 {
      add("Detail", "full page + \(tiles) tiles")
    } else if meta["crop"]?.string == "base" {
      add("Detail", "whole page")
    }
    if let tokens = meta["imageTokens"]?.integer, tokens > 0 { add("Image tokens", String(tokens)) }
    let count = (meta["regions"]?.array ?? []).reduce(0) { $0 + ($1["boxes"]?.array?.count ?? 0) }
    if count > 0 { add("Regions", String(count)) }
    return result
  }
}
