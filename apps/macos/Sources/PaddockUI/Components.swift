import AppKit
import PaddockClient
import PaddockDesign
import SwiftUI

enum WorkspaceAppearance: String, CaseIterable, Identifiable {
  case system = "System"
  case light = "Light"
  case dark = "Dark"
  var id: Self { self }
  var colorScheme: ColorScheme? {
    switch self {
    case .system: nil
    case .light: .light
    case .dark: .dark
    }
  }
}

/// Neutral, opaque chrome. Colour is semantic (warnings/errors or rich content),
/// not a tint across the entire workspace. Keep the future renderer on the same
/// surface roles; do not apply global desaturation to images or syntax colours.
typealias PaddockStyle = PaddockAppearance

struct ModelAvatar: View {
  let vendor: String?
  var size: CGFloat = 40
  var monogram: String {
    guard let vendor, !vendor.isEmpty else { return "M" }
    return String(vendor.prefix(1))
  }
  var body: some View {
    ZStack {
      RoundedRectangle(cornerRadius: size * 0.26).fill(PaddockStyle.elevated)
      if let image = ProviderArtwork.image(for: vendor) {
        Image(nsImage: image).resizable()
          .renderingMode(ProviderArtwork.usesTemplate(for: vendor) ? .template : .original)
          .scaledToFit().saturation(0)
          .padding(size * ProviderArtwork.insetFraction(for: vendor)).foregroundStyle(.secondary)
      } else {
        Text(monogram)
          .font(.system(size: size * 0.42, weight: .semibold, design: .rounded))
          .foregroundStyle(.secondary)
      }
    }.frame(width: size, height: size).accessibilityHidden(true)
  }
}

struct PrimaryAction: ViewModifier {
  func body(content: Content) -> some View {
    content.buttonStyle(FlatButtonStyle(primary: true))
  }
}

/// Native Button semantics, keyboard activation and disabled state, without
/// Tahoe's floating glass treatment. Primary actions use inverse neutral ink.
struct FlatButtonStyle: ButtonStyle {
  var primary = false
  @Environment(\.isEnabled) private var enabled
  func makeBody(configuration: Configuration) -> some View {
    configuration.label.font(.system(size: 12, weight: .medium))
      .padding(.horizontal, 12).padding(.vertical, 8)
      .foregroundStyle(primary ? PaddockStyle.actionForeground : .primary)
      .background(
        primary ? PaddockStyle.accent : PaddockStyle.elevated,
        in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
      )
      .opacity(enabled ? (configuration.isPressed ? 0.7 : 1) : 0.4)
      .contentShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
  }
}

struct QuietButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    QuietButtonBody(configuration: configuration)
  }
  private struct QuietButtonBody: View {
    let configuration: ButtonStyleConfiguration
    @Environment(\.isEnabled) private var enabled
    @State private var hovered = false
    var body: some View {
      configuration.label
        .background(
          hovered || configuration.isPressed ? PaddockStyle.elevated : .clear,
          in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
        )
        .contentShape(Rectangle()).opacity(enabled ? 1 : 0.4)
        .onHover { hovered = $0 }
    }
  }
}

/// Stable-height catalog rows share the same pointer and keyboard semantics.
/// Selection is the only background state; scrolling does not animate hover.
struct ModelListRow<Content: View>: View {
  let height: CGFloat
  let selected: Bool
  var action: () -> Void
  @ViewBuilder var content: Content

  var body: some View {
    Button(action: action) {
      content.padding(.horizontal, 10).frame(height: height)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
          selected ? PaddockStyle.elevated : .clear,
          in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
        )
        // Plain buttons otherwise hit-test only their rendered content. Keep
        // padding, spacers and metadata-free areas clickable, selected or not.
        .contentShape(Rectangle())
    }.buttonStyle(.plain).accessibilityAddTraits(selected ? .isSelected : [])
  }
}

struct WorkspaceRule: View {
  var body: some View { PaddockStyle.border.frame(height: 1).accessibilityHidden(true) }
}

struct PageHeading<Trailing: View>: View {
  let title: String
  var subtitle: String? = nil
  @ViewBuilder var trailing: Trailing
  var body: some View {
    HStack(alignment: .center, spacing: 20) {
      VStack(alignment: .leading, spacing: 7) {
        Text(title).font(.system(size: 25, weight: .semibold)).tracking(-0.5)
        if let subtitle, !subtitle.isEmpty {
          Text(subtitle).font(.system(size: 13)).foregroundStyle(.secondary)
        }
      }
      Spacer(minLength: 8)
      trailing
    }
  }
}

struct Eyebrow: View {
  let text: String
  var body: some View {
    Text(text).font(.system(size: 11, weight: .semibold)).foregroundStyle(.secondary)
  }
}

struct CapabilityLine: View {
  let capabilities: [String]
  private let labels: [(String, String, String)] = [
    ("chat", "Text", "text.bubble"), ("vision", "Vision", "photo"),
    ("documents", "Documents", "doc.text"), ("tools", "Tools", "wrench.and.screwdriver"),
    ("reasoning", "Reasoning", "brain"), ("embeddings", "Embeddings", "square.stack.3d.down.right"),
    ("transcription", "Speech", "waveform"), ("rerank", "Rerank", "arrow.up.arrow.down"),
  ]
  var body: some View {
    ViewThatFits(in: .horizontal) {
      HStack(spacing: 13) {
        ForEach(labels.filter { capabilities.contains($0.0) }, id: \.0) { _, label, symbol in
          Label(label, systemImage: symbol)
        }
      }
      Text(labels.filter { capabilities.contains($0.0) }.map(\.1).joined(separator: " · "))
    }.font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
  }
}

extension CatalogModel {
  func weights(on backend: String?) -> [CatalogArtifact] {
    artifacts.filter { $0.kind == "weights" && $0.supports(backend: backend) }
  }

  func hasInstalledWeights(on backend: String?) -> Bool {
    weights(on: backend).contains(where: \.installed)
  }

  func preferredWeights(on backend: String?) -> CatalogArtifact? {
    let choices = weights(on: backend)
    return choices.first { $0.installed && $0.default == true }
      ?? choices.first(where: \.installed)
      ?? choices.first { $0.default == true } ?? choices.first
  }
}

extension CatalogArtifact {
  var shortFormat: String {
    if quant?.uppercased() == "MLX-TERNARY-2-G128" { return "MLX · Ternary 2-bit" }
    if format.lowercased() == "safetensors", quant?.lowercased() == "affine-4/8" {
      return "MLX · Mixed 4/8-bit"
    }
    if let quant, quant.uppercased().hasPrefix("MLX-AFFINE-"),
      let bits = quant.split(separator: "-").dropFirst(2).first, Int(bits) != nil
    {
      return "MLX · \(bits)-bit"
    }
    return quant.map { "\(format.uppercased()) · \($0)" } ?? format.uppercased()
  }
}

struct FactRow: View {
  let title: String
  let value: String

  var body: some View {
    HStack(alignment: .firstTextBaseline, spacing: 16) {
      Text(title)
      Spacer(minLength: 12)
      Text(value).foregroundStyle(.secondary).textSelection(.enabled)
        .multilineTextAlignment(.trailing)
    }
    .accessibilityElement(children: .combine)
  }
}

struct StatusBadge: View {
  enum Tone { case neutral, success, caution }
  let title: String
  var tone: Tone = .neutral

  private var color: Color {
    switch tone {
    case .neutral: .secondary
    case .success: .primary
    case .caution: PaddockStyle.caution
    }
  }

  private var symbol: String {
    switch tone {
    case .neutral: "circle"
    case .success: "checkmark.circle"
    case .caution: "exclamationmark.triangle"
    }
  }

  var body: some View {
    HStack(spacing: 5) {
      Image(systemName: symbol).font(.system(size: 10))
      Text(title)
    }.font(.system(size: 11, weight: .medium)).foregroundStyle(color)
      .padding(.horizontal, 8).padding(.vertical, 4)
      .background(color.opacity(0.10), in: Capsule())
      .accessibilityElement(children: .ignore).accessibilityLabel(title)
  }
}

enum DisplayFormat {
  static func defaultBundle(_ value: UInt64) -> String {
    // The registry reports zero when this backend has no elected default.
    // It does not mean a model is free to download or has no weight bytes.
    value == 0 ? "No default" : bytes(value)
  }

  static func bytes(_ value: UInt64?) -> String {
    guard let value, value <= UInt64(Int64.max) else { return "Not reported" }
    return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .binary)
  }
}
