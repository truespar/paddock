import Foundation
import Testing

@testable import PaddockConversationCore

struct OCRPresentationTests {
  @Test func lightOnBlocksPreserveWordsAndContinuationWithoutBrokenImages() throws {
    let raw =
      "![title](102,61,898,95) Quarterly Report\n![text](102,120,480,410)\nRevenue grew.\n![text+](520,120,898,300) The north led."
    #expect(NativeOCRText.display(raw) == "Quarterly Report\n\nRevenue grew.\nThe north led.")
    #expect(NativeOCRText.display("Quarterly Report\n![text](102,12") == "Quarterly Report\n")
    #expect(NativeOCRText.display("![logo](logo.png) hi") == "![logo](logo.png) hi")
    let regions = NativeDocumentRunState.regions([
      .object([
        "label": .string("text"), "continues": .bool(true),
        "boxes": .array([.array([.number(1), .number(2), .number(3), .number(4)])]),
      ])
    ])
    #expect(regions.first?["continues"]?.bool == true)
  }

  struct Fixture: Decodable {
    let name: String
    let raw: String
    let display: String
  }
  @Test func sharesTheWebTableFixtures() throws {
    let url = try #require(
      Bundle.module.url(forResource: "ocr-tables", withExtension: "json", subdirectory: "Fixtures"))
    for row in try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url)) {
      #expect(NativeOCRText.display(row.raw) == row.display, "\(row.name)")
    }
  }
  @Test func handlesHTMLWithoutRunningItOrInventingMarkdown() {
    let raw =
      #"<TABLE><tr><td>&lt;script&gt; **not bold** [not a link](file:///secret)</td><td><img src="https://invalid.example">plain</td></tr></TABLE>"#
    let converted = NativeOCRText.display(raw)
    #expect(converted.contains(#"\<script\> \*\*not bold\*\* \[not a link\](file:///secret)"#))
    #expect(converted.contains("| plain |"))
    #expect(!converted.contains("<img"))
    let entity = #"<table><tr><td>&xxe;</td></tr></table>"#
    #expect(!NativeOCRText.display(entity).contains("root:"))
    #expect(NativeOCRText.display("<|det|>table [0,0,999,999]<|/det|>" + raw) == converted)
  }
  @Test func limitsAndNestedTablesPreserveRawInsteadOfTruncating() {
    let nested = "<table><tr><td><table><tr><td>inner</td></tr></table></td></tr></table>"
    #expect(NativeOCRText.display(nested) == nested)
    let huge = "<table><tr>" + String(repeating: "<td>x</td>", count: 257) + "</tr></table>"
    #expect(NativeOCRText.display(huge) == huge)
    let deep =
      "<table><tr><td>" + String(repeating: "<span>", count: 80) + "text"
      + String(repeating: "</span>", count: 80) + "</td></tr></table>"
    #expect(NativeOCRText.display(deep) == deep)
    let large = "<table><tr><td>" + String(repeating: "x", count: 1_048_577) + "</td></tr></table>"
    #expect(NativeOCRText.display(large) == large)
  }
  @Test func storedOCRMatchesWebAndFactsAreMeaningful() throws {
    typealias V = ConversationValue
    let raw: V = .object([
      "mode": .string("table"), "image_tokens": .number(1024),
      "pages": .number(1), "crop": .string("base"), "grounding": .bool(false),
      "pass_through": .bool(false), "dropped_text": .bool(false),
    ])
    let saved = try #require(NativeOCRMetadata.stored(raw))
    #expect(saved["imageTokens"]?.integer == 1024)
    #expect(saved["image_tokens"] == nil)
    #expect(saved["passThrough"]?.bool == false)
    let facts = NativeOCRMetadata.facts(saved)
    #expect(facts.map { $0["label"]?.string } == ["Read as", "Detail", "Image tokens"])
    #expect(facts.map { $0["value"]?.string } == ["Table", "whole page", "1024"])
    #expect(NativeOCRMetadata.facts(raw) == facts)
    #expect(
      NativeOCRMetadata.facts(.object(["grounding": .bool(false), "pages": .number(-2)])).isEmpty)
  }

  @Test func tablePresentationReuseIsBoundedAndNeverReturnsAnOldRevision() {
    var cache = NativeOCRDisplayCache()
    let raw = "<table><tr><td>Original</td></tr></table>"
    let expected = NativeOCRText.display(raw)
    #expect(cache.display(id: "page", raw: raw) == expected)
    let bytes = cache.bytes
    for _ in 0..<100 { #expect(cache.display(id: "page", raw: raw) == expected) }
    #expect(cache.bytes == bytes)
    #expect(cache.display(id: "page", raw: "New plain text") == "New plain text")
    #expect(cache.bytes == 0)
    for index in 0..<100 {
      _ = cache.display(id: "\(index)", raw: raw)
      #expect(cache.bytes <= 64 * bytes)
    }
    let large =
      "<table><tr><td>" + String(repeating: "words ", count: 40_000) + "</td></tr></table>"
    for index in 0..<24 {
      _ = cache.display(id: "large-\(index)", raw: large)
      #expect(cache.bytes <= 8 * 1024 * 1024)
    }
  }
}
