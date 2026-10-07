import AppKit
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Studio provider artwork") @MainActor
struct ProviderArtworkTests {
  @Test func menuArtworkIsBoundedWithoutResizingCatalogOriginals() throws {
    let originals = ProviderArtwork.names.keys.compactMap { vendor in
      ProviderArtwork.image(for: vendor).map { (vendor, $0.size, $0.isTemplate) }
    }
    for (vendor, size, template) in originals {
      let original = try #require(ProviderArtwork.image(for: vendor))
      let menu = try #require(ProviderArtwork.menuImage(for: vendor))
      #expect(menu !== original)
      #expect(menu === ProviderArtwork.menuImage(for: vendor))
      #expect(menu.isValid)
      #expect(menu.size.width > 0 && menu.size.height > 0)
      #expect(max(menu.size.width, menu.size.height) <= ProviderArtwork.menuIconSize)
      #expect(abs(menu.size.width / menu.size.height - size.width / size.height) < 0.001)
      #expect(menu.isTemplate == ProviderArtwork.usesTemplate(for: vendor))
      #expect(original.size == size)
      #expect(original.isTemplate == template)
    }
    #expect(ProviderArtwork.menuImage(for: nil) == nil)
    #expect(ProviderArtwork.menuImage(for: "Unknown maker") == nil)
  }

  @Test func everyWebProviderHasNonemptyNativeArtwork() throws {
    #expect(ProviderArtwork.names["Cloudflare"] == "Cloudflare")
    for vendor in ProviderArtwork.names.keys.sorted() {
      let image = try #require(ProviderArtwork.image(for: vendor), "Missing \(vendor)")
      #expect(image.isValid)
      #expect(image.size.width > 0 && image.size.height > 0)
      let data = try #require(image.tiffRepresentation, "Cannot rasterize \(vendor)")
      let bitmap = try #require(NSBitmapImageRep(data: data))
      var ink = 0
      var transparent = 0
      for y in 0..<bitmap.pixelsHigh {
        for x in 0..<bitmap.pixelsWide {
          if let color = bitmap.colorAt(x: x, y: y) {
            if color.alphaComponent > 0.1 { ink += 1 }
            if color.alphaComponent < 0.1 { transparent += 1 }
          }
        }
      }
      #expect(ink > 0, "Blank SVG: \(vendor)")
      #expect(transparent > 0, "Solid SVG: \(vendor)")
    }
    #expect(ProviderArtwork.image(for: "Unknown maker") == nil)
    #expect(ProviderArtwork.image(for: nil) == nil)
    let ibm = try #require(ProviderArtwork.image(for: "IBM"))
    #expect(abs(ibm.size.width / ibm.size.height - 58.0 / 23.0) < 0.01)
    #expect(!ProviderArtwork.usesTemplate(for: "KBLab"))
    #expect(ProviderArtwork.usesTemplate(for: "Google"))
    #expect(ProviderArtwork.names["Prism ML"] == "PrismML")
    #expect(ProviderArtwork.usesTemplate(for: "Prism ML"))
    #expect(ProviderArtwork.names["Aleph Alpha"] == "AlephAlpha")
    #expect(ProviderArtwork.usesTemplate(for: "Aleph Alpha"))
  }

  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Run apps/macos/scripts/check.sh for the real registry projection."))
  func allCatalogMakersUseStudioMarksOrItsLetterFallback() throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"])
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(contentsOf: URL(fileURLWithPath: path).appending(path: "catalog.json")))
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let web = try String(
      contentsOf: root.appending(path: "studio/src/components/manage/VendorLogo.vue"),
      encoding: .utf8)
    for vendor in Set(catalog.models.compactMap(\.vendor)) {
      let hasWebMark =
        web.contains("vendor === '\(vendor)'")
        || web.contains("\(vendor): si") || web.contains("'\(vendor)': si")
      if hasWebMark {
        #expect(ProviderArtwork.image(for: vendor) != nil, "Missing Studio mark: \(vendor)")
      } else {
        // Private catalogs can add makers without a bundled brand asset, just
        // as cloud catalogs do. Match the web's first-letter badge in that case.
        #expect(ModelAvatar(vendor: vendor).monogram == String(vendor.prefix(1)))
      }
    }
  }
}
