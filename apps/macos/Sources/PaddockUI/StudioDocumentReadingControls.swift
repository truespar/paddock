import SwiftUI

/// Same advertised modes and placement as web Composer's OCR bar. No chat
/// instruction box for fixed-vocabulary decoders, and no invented capabilities.
struct StudioDocumentReadingControls: View {
  let modes: [String]
  let grounding: Bool
  let mode: String
  let regions: Bool
  let onMode: (String) -> Void
  let onRegions: (Bool) -> Void
  var body: some View {
    PaddockScrollView(.horizontal) {
      HStack(spacing: 6) {
        Button("Automatic") { onMode("") }
          .buttonStyle(ComposerButtonStyle(active: mode.isEmpty || !modes.contains(mode)))
        ForEach(modes, id: \.self) { value in
          Button(Self.label(value)) { onMode(value) }
            .buttonStyle(ComposerButtonStyle(active: mode == value))
        }
        if grounding && !modes.contains("grounding") {
          Button("Show where", systemImage: "viewfinder") { onRegions(!regions) }
            .buttonStyle(ComposerButtonStyle(active: regions))
        }
      }.padding(.horizontal, 8)
    }.scrollIndicators(.hidden).fixedSize(horizontal: false, vertical: true)
      .accessibilityElement(children: .contain).accessibilityLabel("Document reading")
      .accessibilityIdentifier("composer-document-reading")
  }
  static func label(_ mode: String) -> String {
    [
      "document": "Document", "multipage": "Pages of one document", "free": "Plain text",
      "layout": "Layout map", "figure": "Figure", "ocr": "Text", "table": "Table",
      "formula": "Formula", "chart": "Chart", "spotting": "Text spotting", "seal": "Seal",
      "plain": "Markdown", "grounding": "Blocks with boxes",
    ][mode]
      ?? mode.prefix(1).uppercased() + mode.dropFirst()
  }
}
