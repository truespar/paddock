import PaddockConversationCore
import SwiftUI

/// The web Studio confidence bins, including their explicit numeric legend.
/// Colour never substitutes for an answer, probability or accessibility label.
struct ReadConfidenceSwatch: View {
  let value: Double
  @Environment(\.colorScheme) private var scheme
  private var color: Color {
    let values: [UInt32] =
      scheme == .dark
      ? [0x4E5B7A, 0x7E8386, 0xB7AE6E, 0xF2DF3F]
      : [0x23305A, 0x64686D, 0x9A8C3E, 0xD2B800]
    let rgb = values[ReadResponse.Answer.confidenceBin(value)]
    return Color(
      red: Double((rgb >> 16) & 255) / 255,
      green: Double((rgb >> 8) & 255) / 255, blue: Double(rgb & 255) / 255)
  }
  var body: some View {
    RoundedRectangle(cornerRadius: 3).fill(color).frame(width: 12, height: 12)
      .accessibilityHidden(true)
  }
}

struct ReadScale: View {
  let value: Double
  let lower: String
  let upper: String
  var ticks: [Double] = [0.3, 0.7]
  var body: some View {
    VStack(spacing: 4) {
      GeometryReader { geometry in
        ZStack(alignment: .leading) {
          Capsule().fill(.secondary.opacity(0.18)).frame(height: 4)
          ForEach(ticks, id: \.self) { tick in
            Rectangle().fill(.secondary.opacity(0.45)).frame(width: 1, height: 10)
              .offset(x: (geometry.size.width - 8) * tick + 4)
          }
          Circle().fill(.primary).frame(width: 8, height: 8)
            .offset(x: (geometry.size.width - 8) * min(1, max(0, value)))
        }.frame(height: 14)
      }.frame(height: 14).accessibilityHidden(true)
      HStack {
        Text(lower)
        Spacer()
        Text(upper)
      }.font(.caption).foregroundStyle(.secondary)
    }
  }
}

struct NativeReadDiagnostics: View {
  let response: ReadResponse
  let elapsedMilliseconds: Double
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      ViewThatFits(in: .horizontal) {
        HStack(spacing: 10) { legend }
        VStack(alignment: .leading, spacing: 6) { legend }
      }
      DisclosureGroup("Diagnostics") {
        VStack(alignment: .leading, spacing: 12) {
          Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            if let canvas = response.diagnostics.canvas { metric("Canvas", "\(canvas) positions") }
            if let checkpoint = response.diagnostics.checkpoint { metric("Checkpoint", checkpoint) }
            if let tokens = response.diagnostics.stateTokens { metric("State tokens", "\(tokens)") }
            if let read = response.diagnostics.stateRead, read != response.diagnostics.stateTokens {
              metric("State tokens read", "\(read)")
            }
            if let images = response.diagnostics.images, images > 0 {
              metric("Images", "\(images)")
            }
            ForEach(Array((response.diagnostics.pictures ?? []).enumerated()), id: \.offset) {
              index, picture in
              if picture.resized.count == 2 {
                metric(
                  "Image \(index + 1)",
                  "\(picture.width) x \(picture.height) to \(picture.resized[0]) x \(picture.resized[1]) · \(picture.tokens) tokens"
                )
              }
            }
            if let routing = response.routing { metric("Routing", routing.reason) }
            metric("Reads", "\(response.diagnostics.reads)")
            if let input = response.usage?.inputTokens { metric("Prompt tokens", "\(input)") }
            if let output = response.usage?.outputTokens { metric("Output tokens", "\(output)") }
            metric(
              "Model time", String(format: "%.0f ms", response.diagnostics.timing.totalMilliseconds)
            )
            if let gpu = response.diagnostics.timing.gpuMilliseconds {
              metric("Decision pass", String(format: "%.0f ms", gpu))
            }
            if let requests = response.diagnostics.timing.passRequests {
              metric("Batch", "\(requests) \(requests == 1 ? "request" : "requests")")
            }
            metric("Total time", String(format: "%.0f ms", elapsedMilliseconds))
          }.font(.caption).textSelection(.enabled)
          ForEach(response.diagnostics.questions, id: \.id) { row in
            VStack(alignment: .leading, spacing: 6) {
              Text(row.id).font(.caption.weight(.medium)).textSelection(.enabled)
              HStack {
                if let position = row.position { Text("Position \(position)") }
                if let mass = row.labelMass { Text("Label mass \(mass, specifier: "%.3f")") }
                if response.diagnostics.backend == "clef" {
                  Text(row.label)
                  if let options = row.options { Text("\(options) options") }
                }
                Text("Entropy \(row.entropy, specifier: "%.3f")")
              }.font(.caption).foregroundStyle(.secondary).monospacedDigit()
              if let temperature = row.temperature {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                  metric("Temperature", String(format: "%.2f", temperature))
                  if let count = row.options { metric("Options", "\(count)") }
                  if let tokens = row.tokens {
                    metric("Tokens", tokens.map(String.init).joined(separator: ", "))
                  }
                  if let confidence = row.entropyConfidence {
                    metric("Entropy confidence", String(format: "%.3f", confidence))
                  }
                  if let window = row.window {
                    metric(
                      "Window",
                      "\(window.index + 1) of \(window.count) · tokens \(window.tokenStart)-\(window.tokenEnd)"
                    )
                  }
                }.font(.caption).textSelection(.enabled)
              }
              if let reads = row.reads, reads.count > 1 {
                // Native, wrapping heatmap: every cell remains keyboard/VoiceOver
                // readable; no clipped labels or horizontal page overflow.
                LazyVGrid(
                  columns: [GridItem(.adaptive(minimum: 115), alignment: .leading)],
                  alignment: .leading, spacing: 6
                ) {
                  ForEach(Array(reads.enumerated()), id: \.offset) { index, read in
                    HStack(alignment: .top, spacing: 6) {
                      ReadConfidenceSwatch(value: read.confidence)
                      VStack(alignment: .leading, spacing: 3) {
                        Text("Read \(index + 1)").foregroundStyle(.secondary)
                        Text(read.pick).lineLimit(2)
                        Text(read.confidence, format: .number.precision(.fractionLength(2)))
                          .monospacedDigit()
                      }
                    }.font(.caption).padding(8).frame(maxWidth: .infinity, alignment: .leading)
                      .background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 6))
                      .help(
                        "Read \(index + 1): \(read.pick); confidence \(read.confidence); entropy \(read.entropy)"
                      )
                      .accessibilityElement(children: .combine)
                  }
                }
              }
            }
          }
        }.padding(.top, 10)
      }
    }
  }
  private var legend: some View {
    ForEach(Array(["under 0.5", "0.5-0.7", "0.7-0.9", "0.9 and over"].enumerated()), id: \.offset) {
      index, label in
      HStack(spacing: 4) {
        ReadConfidenceSwatch(value: [0.0, 0.5, 0.7, 0.9][index])
        Text(label).font(.system(size: 10)).foregroundStyle(.secondary).fixedSize()
      }
    }
  }
  private func metric(_ label: String, _ value: String) -> some View {
    GridRow {
      Text(label).foregroundStyle(.secondary)
      Text(value).monospacedDigit()
    }
  }
}
