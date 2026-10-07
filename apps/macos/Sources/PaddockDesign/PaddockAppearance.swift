import AppKit
import SwiftUI

/// One bundled palette for Swift and the embedded Studio (which imports the
/// same JSON at build time). Document ink, syntax and categorical charts are
/// content, not chrome: never apply a global tint/desaturation to those.
public enum PaddockAppearance {
  private struct Palette: Decodable {
    struct Pair: Decodable {
      let light: String
      let dark: String
    }
    let colors: [String: Pair]
    let radii: [String: CGFloat]
  }
  private static let palette: Palette = {
    // A missing/malformed build resource is a packaging defect, not a reason
    // to silently show an unrelated fallback palette.
    let url = Bundle.module.url(forResource: "appearance", withExtension: "json")!
    return try! JSONDecoder().decode(Palette.self, from: Data(contentsOf: url))
  }()

  public static func nsColor(_ role: String, dark: Bool, increasedContrast: Bool = false) -> NSColor
  {
    if increasedContrast {
      switch role {
      case "border": return NSColor(white: dark ? 0.62 : 0.38, alpha: 1)
      case "secondary": return NSColor(white: dark ? 0.9 : 0.15, alpha: 1)
      case "primary": return NSColor(white: dark ? 1 : 0, alpha: 1)
      default: break  // Preserve semantic warnings and content colors.
      }
    }
    let pair = palette.colors[role]!
    let hex = String((dark ? pair.dark : pair.light).dropFirst())
    let value = UInt32(hex, radix: 16)!
    let rgb = hex.count == 8 ? value >> 8 : value
    return NSColor(
      srgbRed: CGFloat((rgb >> 16) & 255) / 255,
      green: CGFloat((rgb >> 8) & 255) / 255, blue: CGFloat(rgb & 255) / 255,
      alpha: hex.count == 8 ? CGFloat(value & 255) / 255 : 1)
  }
  /// AppKit windows must use the same opaque palette as their SwiftUI content.
  /// The system window background is wallpaper-tinted, even when isOpaque is true.
  public static func nsColor(_ role: String) -> NSColor {
    NSColor(name: nil) { appearance in
      let match = appearance.bestMatch(from: [
        .aqua, .darkAqua, .accessibilityHighContrastAqua, .accessibilityHighContrastDarkAqua,
      ])
      return nsColor(
        role,
        dark: match == .darkAqua || match == .accessibilityHighContrastDarkAqua,
        increasedContrast: match == .accessibilityHighContrastAqua
          || match == .accessibilityHighContrastDarkAqua)
    }
  }
  private static func color(_ role: String) -> Color { Color(nsColor: nsColor(role)) }
  public static let canvas = color("canvas")
  public static let sidebar = color("sidebar")
  public static let surface = color("surface")
  public static let popup = color("popup")
  public static let elevated = color("elevated")
  public static let accent = color("accent")
  public static let actionForeground = color("actionForeground")
  public static let border = color("border")
  public static let caution = color("caution")
  public static let primary = color("primary")
  public static let secondary = color("secondary")

  public enum Radius {
    public static let small = palette.radii["sm"]!
    public static let control = palette.radii["md"]!
    public static let card = palette.radii["lg"]!
    public static let panel = palette.radii["xl"]!
    public static let composer = palette.radii["composer"]!
  }
}
