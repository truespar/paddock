import AppKit
import Foundation

/// The web Studio's VendorLogo.vue is the source of truth. SVGs retain its
/// geometry; native rendering supplies neutral ink without a network fetch.
@MainActor enum ProviderArtwork {
  static let names: [String: String] = [
    "Alibaba": "Qwen", "OpenAI": "OpenAI", "Google": "Google", "Poolside": "Poolside",
    "IBM": "IBM", "KBLab": "KBLab", "NB AI-Lab": "NBAILab", "CoRal Project": "CoRal",
    "Anthropic": "Anthropic", "DeepSeek": "DeepSeek", "Meta": "Meta", "Mistral": "Mistral",
    "NVIDIA": "NVIDIA", "Xiaomi": "Xiaomi", "Moonshot": "Moonshot", "Perplexity": "Perplexity",
    "Baidu": "Baidu", "ByteDance": "ByteDance", "PaddlePaddle": "PaddlePaddle",
    "Cloudflare": "Cloudflare",
    "MiniMax": "MiniMax", "Hugging Face": "HuggingFace", "OpenRouter": "OpenRouter",
    "Prism ML": "PrismML", "Aleph Alpha": "AlephAlpha",
    "Exa": "Exa", "Tavily": "Tavily", "Firecrawl": "Firecrawl", "Brave": "Brave",
  ]

  /// PaddockUI's resource bundle; the status item's menu bar mark lives here too.
  static let resources: Bundle? = {
    // A packaged app must not silently fall back to the developer's build tree.
    if Bundle.main.bundleURL.pathExtension == "app" {
      return Bundle.main.url(forResource: "PaddockMac_PaddockUI", withExtension: "bundle")
        .flatMap(Bundle.init(url:))
    }
    return Bundle.module
  }()

  private static let images: [String: NSImage] = names.values.reduce(into: [:]) { images, name in
    if let url = resources?.url(forResource: name, withExtension: "svg"),
      let image = NSImage(contentsOf: url)
    {
      images[name] = image
    }
  }

  static func image(for vendor: String?) -> NSImage? {
    guard let vendor, let name = names[vendor] else { return nil }
    return images[name]
  }

  // NSMenu extracts the underlying image from a SwiftUI Label, ignoring view
  // frames and resizable(). Give it an actual menu-sized image, without changing
  // the shared SVGs used by larger catalog avatars. Copies retain vector reps.
  static let menuIconSize: CGFloat = 16
  private static let menuImages: [String: NSImage] = names.reduce(into: [:]) { result, entry in
    let (vendor, name) = entry
    guard let source = images[name], source.size.width > 0, source.size.height > 0,
      let image = source.copy() as? NSImage
    else { return }
    let scale = menuIconSize / max(source.size.width, source.size.height)
    image.size = NSSize(width: source.size.width * scale, height: source.size.height * scale)
    image.isTemplate = usesTemplate(for: vendor)
    result[vendor] = image
  }

  static func menuImage(for vendor: String?) -> NSImage? {
    vendor.flatMap { menuImages[$0] }
  }

  // KBLab's opaque background carries meaningful luminance differences: an
  // alpha-only template would erase the artwork into a solid circle. Desaturate
  // just this original image, never the workspace or future user image content.
  static func usesTemplate(for vendor: String?) -> Bool { vendor != "KBLab" }

  // IBM has no square symbol. Preserve the 58:23 wordmark, with optical insets
  // that keep its eight bars legible in the 34-point list avatar on Retina.
  static func insetFraction(for vendor: String?) -> CGFloat { vendor == "IBM" ? 0.06 : 0.2 }
}
