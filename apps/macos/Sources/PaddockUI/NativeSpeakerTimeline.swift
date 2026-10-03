import PaddockStudio
import SwiftUI

/// One canvas per speaker, not thousands of SwiftUI segment views. Separate
/// lanes preserve overlap. Previous/next controls expose every interval to
/// keyboard/VoiceOver without creating a control for every short segment.
struct NativeSpeakerTimeline: View {
  let timeline: StudioState.Speech.Diarization
  var onSeek: (Double) -> Void
  private var speakers: [Int] { Set(timeline.segments.map(\.speaker)).sorted() }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("Speakers").font(.system(size: 12, weight: .semibold))
        Spacer()
        Text(speechClock(timeline.duration)).monospacedDigit().foregroundStyle(.secondary)
      }
      if speakers.isEmpty { Text("No speech detected").foregroundStyle(.secondary) }
      ForEach(speakers, id: \.self) { speaker in
        let spans = timeline.segments.filter { $0.speaker == speaker }
        NativeSpeakerLane(
          speaker: speaker, spans: spans, duration: timeline.duration, onSeek: onSeek)
      }
    }.font(.system(size: 11)).padding(12)
      .background(.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 10))
      .accessibilityIdentifier("speaker-timeline")
  }
}

private struct NativeSpeakerLane: View {
  let speaker: Int
  let spans: [StudioState.Speech.Diarization.Interval]
  let duration: Double
  var onSeek: (Double) -> Void
  @State private var selected = 0
  private var current: Int { min(selected, max(0, spans.count - 1)) }
  private func step(_ delta: Int) {
    guard !spans.isEmpty else { return }
    selected = min(spans.count - 1, max(0, current + delta))
    onSeek(spans[selected].start)
  }
  var body: some View {
    HStack(spacing: 8) {
      Button("Speaker \(speaker + 1)") { step(0) }.frame(width: 65, alignment: .leading)
      Button {
        step(-1)
      } label: {
        Image(systemName: "chevron.left")
      }
      .disabled(current == 0).accessibilityLabel("Previous interval for Speaker \(speaker + 1)")
      Button {
        step(1)
      } label: {
        Image(systemName: "chevron.right")
      }
      .disabled(current + 1 >= spans.count).accessibilityLabel(
        "Next interval for Speaker \(speaker + 1)")
      GeometryReader { geometry in
        Canvas { context, size in
          var path = Path()
          for s in spans {
            let x = s.start / duration * size.width
            let width = max(1, (s.end - s.start) / duration * size.width)
            path.addRoundedRect(
              in: CGRect(x: x, y: 3, width: width, height: 14),
              cornerSize: CGSize(width: 3, height: 3))
          }
          context.fill(path, with: .color(.primary.opacity(0.55)))
        }
        .background(.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .gesture(
          SpatialTapGesture().onEnded { tap in
            onSeek(
              min(
                duration,
                max(0, tap.location.x / max(1, geometry.size.width) * duration)))
          }
        )
        .accessibilityHidden(true)
      }.frame(height: 20)
    }.buttonStyle(.plain)
  }
}
