import AppKit
import ImageIO
import PaddockStudio
import SwiftUI

/// Saved pixels reuse the bounded, off-main-thread native image cache. A
/// progressive preview replaces one bitmap, never a history of full images.
struct NativeGeneratedImage: View {
  let picture: StudioState.NativeTranscript.Message.Picture
  let workspace: StudioWorkspace
  let open: () -> Void
  @State private var pixels: CGImage?
  @State private var failure: String?
  @State private var visible = false
  @State private var aspect: CGFloat?
  private struct Request: Equatable {
    let id: String
    let dataURL: String?
    let visible: Bool
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if let pixels {
        Button(action: open) {
          Image(decorative: pixels, scale: 1).resizable().scaledToFit()
            .frame(maxWidth: 640, maxHeight: 640)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }.buttonStyle(.plain).disabled(picture.preview).accessibilityLabel(picture.name)
      } else if !visible, let aspect {
        Color.clear.aspectRatio(aspect, contentMode: .fit).frame(maxWidth: 640, maxHeight: 640)
      } else if let failure {
        Text(failure).font(.caption).foregroundStyle(.secondary)
      } else {
        ProgressView().frame(width: 160, height: 120)
      }
      if picture.preview {
        Label("Rendering…", systemImage: "photo").font(.caption).foregroundStyle(.secondary)
      } else {
        Button("Save image", systemImage: "square.and.arrow.down") {
          Task { await workspace.saveOriginal(picture.id, name: picture.name) }
        }.font(.caption).buttonStyle(.plain)
      }
    }.accessibilityIdentifier("generated-image-\(picture.id)")
      .onScrollVisibilityChange(threshold: 0.001) { visible = $0 }
      .task(id: Request(id: picture.id, dataURL: picture.dataURL, visible: visible)) {
        pixels = nil
        guard visible else { return }
        failure = nil
        do {
          if let value = picture.dataURL {
            let decoded = try await GeneratedImageDecoder.shared.decode(value)
            try Task.checkCancellation()
            pixels = decoded.image
            aspect = CGFloat(decoded.image.width) / CGFloat(decoded.image.height)
          } else {
            let image = try await workspace.documentMedia.image(
              .init(attachmentID: picture.id, pdfPage: nil)
            ) {
              try await workspace.downloadOriginal(picture.id)
            }
            try Task.checkCancellation()
            pixels = image
            aspect = CGFloat(image.width) / CGFloat(image.height)
          }
        } catch is CancellationError {} catch { failure = error.localizedDescription }
      }
      .onDisappear {
        visible = false
        pixels = nil
      }
  }
}

private actor GeneratedImageDecoder {
  static let shared = GeneratedImageDecoder()
  func decode(_ value: String) throws -> Pixels {
    try Task.checkCancellation()
    return try autoreleasepool {
      guard value.hasPrefix("data:image/"), let comma = value.firstIndex(of: ","),
        value.utf8.count <= 64 * 1024 * 1024,
        let bytes = Data(base64Encoded: String(value[value.index(after: comma)...])),
        let source = CGImageSourceCreateWithData(bytes as CFData, nil),
        let image = CGImageSourceCreateThumbnailAtIndex(
          source, 0,
          [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 1536,
            kCGImageSourceShouldCacheImmediately: true,
          ] as CFDictionary)
      else { throw CocoaError(.fileReadCorruptFile) }
      try Task.checkCancellation()
      return Pixels(image: image)
    }
  }
  struct Pixels: @unchecked Sendable { let image: CGImage }
}
