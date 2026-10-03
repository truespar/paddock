import AppKit
import PaddockStudio
import SwiftUI

struct NativeSpeechView: View {
  let speech: StudioState.Speech
  var workspace: StudioWorkspace?
  @State private var nextDifference = 0
  @State private var focusedWord: Int?
  @State private var identifying = false
  @State private var identification: Task<Void, Never>?
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if let timeline = speech.diarization {
        NativeSpeakerTimeline(timeline: timeline) { seconds in
          guard let workspace, let clip = speech.clip else { return }
          Task { await workspace.playAudio(clip, at: seconds) }
        }
      }
      if let workspace, let models = speech.diarizers, !models.isEmpty,
        let messageID = speech.messageId, let conversationID = speech.conversationId
      {
        Menu {
          ForEach(models, id: \.port) { model in
            Button(model.title) {
              identifying = true
              identification = Task {
                defer { identifying = false }
                await workspace.perform(
                  "identifySpeakers",
                  [
                    "messageId": .string(messageID), "conversationId": .string(conversationID),
                    "modelId": .string(model.id), "port": .number(Double(model.port)),
                  ])
              }
            }
          }
        } label: {
          Label(
            identifying ? "Identifying speakers…" : "Identify speakers",
            systemImage: "person.wave.2")
        }.menuStyle(.borderlessButton).fixedSize().disabled(identifying)
          .accessibilityIdentifier("identify-speakers")
        if identifying {
          Button("Cancel") { identification?.cancel() }.buttonStyle(.plain)
        }
      }
      ForEach(speech.guards.indices, id: \.self) { index in
        let guardrail = speech.guards[index]
        Label(
          "\(speechClock(guardrail.start))-\(speechClock(guardrail.end)) · \(guardrail.note)",
          systemImage: "exclamationmark.triangle"
        )
        .font(.system(size: 12)).foregroundStyle(.orange).textSelection(.enabled)
      }
      if !speech.differs.isEmpty || (SpeechMarkPolicy.anyUnsure(speech) && workspace != nil) {
        HStack(spacing: 12) {
          if !speech.differs.isEmpty {
            Button("Heard differently · \(speech.differs.count)", systemImage: "arrow.right") {
              let index = speech.differs[nextDifference % speech.differs.count]
              nextDifference += 1
              focusedWord = index
              if speech.words.indices.contains(index), let start = speech.words[index].start,
                let clip = speech.clip, let workspace
              {
                Task { await workspace.playAudio(clip, at: start) }
              }
            }.buttonStyle(.plain)
          }
          if SpeechMarkPolicy.anyUnsure(speech), let workspace {
            let enabled = workspace.state?.markUnsure ?? true
            Button {
              Task { await workspace.perform("transcriptMarks", ["enabled": .bool(!enabled)]) }
            } label: {
              HStack(spacing: 8) {
                Text("Unsure words").underline(enabled, color: .red)
                Text(enabled ? "marked" : "not marked").foregroundStyle(.secondary)
              }
            }.buttonStyle(.plain)
              .accessibilityIdentifier("transcript-unsure-words")
              .accessibilityLabel("Unsure words")
              .accessibilityValue(enabled ? "marked" : "not marked")
              .help(
                "Marks the words this model scored lowest, not measured errors. A marked word may be right and an unmarked one may be wrong. Scores are not comparable between models."
              )
          }
        }.font(.system(size: 11)).foregroundStyle(.secondary)
      }
      NativeSpeechPlaybackText(
        speech: speech, player: workspace?.audioPlayback,
        markUnsure: workspace?.state?.markUnsure ?? true, focusedWord: focusedWord,
        onSeek: { seconds in
          guard let workspace, let clip = speech.clip else { return }
          Task { await workspace.playAudio(clip, at: seconds) }
        }
      )
      .frame(maxWidth: .infinity, alignment: .leading)
      .accessibilityIdentifier("native-speech-text")
      ViewThatFits(in: .horizontal) {
        HStack(spacing: 14) { facts }
        VStack(alignment: .leading, spacing: 4) { facts }
      }.font(.system(size: 11)).foregroundStyle(.secondary)
    }.onDisappear { identification?.cancel() }
  }
  private var facts: some View {
    ForEach(speech.facts.indices, id: \.self) { index in
      Text("\(speech.facts[index].label) · \(speech.facts[index].value)").textSelection(.enabled)
    }
  }
}

enum SpeechMarkPolicy {
  static let unsureBelow = 0.45
  static func unsure(_ word: StudioState.Speech.Word) -> Bool {
    word.confidence.map { $0.isFinite && $0 < unsureBelow } ?? false
  }
  static func anyUnsure(_ speech: StudioState.Speech) -> Bool {
    speech.words.contains(where: unsure)
  }
}

/// The media clock ticks only while playing. Paused seeks observe the published
/// position outside TimelineView: a paused animation closure does not subscribe
/// its parent to those changes. Neither path rebuilds the surrounding chat.
struct NativeSpeechPlaybackText: View {
  let speech: StudioState.Speech
  var player: StudioAudioPlayback?
  let markUnsure: Bool
  let focusedWord: Int?
  var onSeek: (Double) -> Void
  var body: some View {
    let active = speech.clip != nil && player?.clipId == speech.clip?.id
    let playing = active && player?.playing == true
    let pausedPosition = active && !playing ? player?.position : nil
    TimelineView(.animation(minimumInterval: 1.0 / 60, paused: !playing)) { _ in
      NativeSpeechText(
        speech: speech, markUnsure: markUnsure,
        position: playing ? player?.renderingPosition : pausedPosition, focusedWord: focusedWord,
        onSeek: onSeek)
    }
  }
}

/// One TextKit 2 surface: selection spans rows/segments, and transport ticks
/// change only highlighting attributes, never the text storage or selection.
struct NativeSpeechText: NSViewRepresentable {
  let speech: StudioState.Speech
  let markUnsure: Bool
  let position: Double?
  let focusedWord: Int?
  var onSeek: (Double) -> Void
  func makeNSView(context: Context) -> SpeechTextView {
    let view = SpeechTextView(usingTextLayoutManager: true)
    view.isEditable = false
    view.isSelectable = true
    view.drawsBackground = false
    view.textContainerInset = .zero
    view.textContainer?.lineFragmentPadding = 0
    view.isHorizontallyResizable = false
    view.isVerticallyResizable = true
    view.textContainer?.widthTracksTextView = true
    view.linkTextAttributes = [.foregroundColor: NSColor.labelColor]
    view.isAutomaticLinkDetectionEnabled = false
    view.isContinuousSpellCheckingEnabled = false
    view.delegate = view
    view.setAccessibilityLabel("Transcript. Select text to copy; activate a timed word to seek.")
    return view
  }
  func updateNSView(_ view: SpeechTextView, context: Context) {
    view.identifier = .init(context.environment.conversationTextID)
    view.onSeek = onSeek
    view.update(speech: speech, marks: markUnsure, position: position, focus: focusedWord)
  }
  func sizeThatFits(_ proposal: ProposedViewSize, nsView: SpeechTextView, context: Context)
    -> CGSize?
  {
    let width = max(1, proposal.width ?? 500)
    nsView.frame.size.width = width
    nsView.textContainer?.size = NSSize(width: width, height: CGFloat.greatestFiniteMagnitude)
    if let layout = nsView.textLayoutManager, let content = layout.textContentManager {
      layout.ensureLayout(for: content.documentRange)
      return CGSize(width: width, height: max(24, ceil(layout.usageBoundsForTextContainer.height)))
    }
    return CGSize(width: width, height: 24)
  }
}

@MainActor final class SpeechTextView: NSTextView, NSTextViewDelegate {
  var onSeek: ((Double) -> Void)?
  private var words: [StudioState.Speech.Word] = []
  private var differences: [Int] = []
  private var differenceSet = Set<Int>()
  private var ranges: [NSRange] = []
  private var segments: [StudioState.Speech.Segment] = []
  private var segmentWords: [Int: [Int]] = [:]
  private var timed: [(at: Int, start: Double, end: Double)] = []
  private var clipId: String?
  private var marks = false
  private var highlight: Int?
  private var activeSegment: Int?
  private var focused: Int?

  override func viewDidChangeEffectiveAppearance() {
    super.viewDidChangeEffectiveAppearance()
    guard let storage = textStorage else { return }
    storage.beginEditing()
    for index in words.indices { background(index) }
    storage.endEditing()
  }

  func update(speech: StudioState.Speech, marks: Bool, position: Double?, focus: Int?) {
    guard let storage = textStorage else { return }
    if words != speech.words || differences != speech.differs || self.marks != marks
      || segments != (speech.segments ?? []) || clipId != speech.clip?.id
    {
      let selection = selectedRanges
      words = speech.words
      differences = speech.differs
      self.marks = marks
      segments = speech.segments ?? []
      clipId = speech.clip?.id
      differenceSet = Set(differences)
      ranges = []
      timed = []
      segmentWords = [:]
      let body = NSMutableAttributedString()
      let paragraph = NSMutableParagraphStyle()
      paragraph.lineSpacing = 5
      for (index, word) in words.enumerated() {
        if word.speakers != (index > 0 ? words[index - 1].speakers : nil),
          let speakers = word.speakers
        {
          let label =
            speakers.isEmpty
            ? "Unassigned" : speakers.map { "Speaker \($0 + 1)" }.joined(separator: " / ")
          body.append(
            NSAttributedString(
              string: "\(index > 0 ? "\n" : "")\(label)\n",
              attributes: [
                .font: NSFont.systemFont(ofSize: 11, weight: .semibold),
                .foregroundColor: NSColor.secondaryLabelColor, .paragraphStyle: paragraph,
              ]))
        } else if index > 0 {
          body.append(NSAttributedString(string: " "))
        }
        let range = NSRange(location: body.length, length: (word.word as NSString).length)
        ranges.append(range)
        segmentWords[word.segment, default: []].append(index)
        if let start = word.start, let end = word.end,
          start.isFinite, end.isFinite, start >= 0, end > start
        {
          timed.append((index, start, end))
        }
        var attributes: [NSAttributedString.Key: Any] = [
          .font: NSFont.systemFont(ofSize: 15), .foregroundColor: NSColor.labelColor,
          .paragraphStyle: paragraph,
        ]
        if let start = word.start, start.isFinite, start >= 0, speech.clip != nil {
          attributes[.link] = URL(string: "paddock-word:\(index)")!
        }
        var hints: [String] = []
        if let speakers = word.speakers, speakers.count > 1 {
          hints.append("Multiple speakers cover this word's time range; its speaker is ambiguous.")
        }
        if differenceSet.contains(index) {
          attributes[.underlineStyle] = NSUnderlineStyle.double.rawValue
          attributes[.underlineColor] = NSColor.systemOrange
          hints.append("The other model heard this differently.")
        } else if marks, SpeechMarkPolicy.unsure(word) {
          attributes[.underlineStyle] =
            NSUnderlineStyle.single.rawValue | NSUnderlineStyle.patternDot.rawValue
          attributes[.underlineColor] = NSColor.systemRed
        }
        if marks, SpeechMarkPolicy.unsure(word), let confidence = word.confidence {
          let hint =
            if let alt = word.alt, let margin = word.margin, margin < 0.2 {
              "This model nearly said '\(alt)'."
            } else {
              "This model scored it \(Int((confidence * 100).rounded()))%. This is not an accuracy estimate."
            }
          hints.append(hint)
        }
        if let start = word.start, start.isFinite, start >= 0, speech.clip != nil {
          hints.append("Click to play from \(speechClock(start)).")
        }
        if !hints.isEmpty { attributes[.toolTip] = hints.joined(separator: " ") }
        body.append(NSAttributedString(string: word.word, attributes: attributes))
      }
      // Sorting protects the bisection against out-of-order external metadata;
      // equal starts retain the web's last-started rendering order.
      timed.sort { $0.start == $1.start ? $0.at < $1.at : $0.start < $1.start }
      storage.setAttributedString(body)
      selectedRanges = selection.filter { NSMaxRange($0.rangeValue) <= body.length }
      highlight = nil
      activeSegment = nil
      for index in words.indices { background(index) }
      invalidateIntrinsicContentSize()
    }
    // A segment start alone is not a word-level clock. Do not invent karaoke
    // timing for models that supplied only sentence-level timestamps.
    let clock = position.flatMap { $0.isFinite && $0 >= 0 ? $0 + 0.001 : nil }
    let segment = clock.flatMap { seconds in
      segments.lastIndex { $0.start.isFinite && $0.start <= seconds }
    }
    let current = clock.flatMap { seconds -> Int? in
      var lo = 0
      var hi = timed.count
      while lo < hi {
        let mid = (lo + hi) / 2
        if timed[mid].start <= seconds { lo = mid + 1 } else { hi = mid }
      }
      guard lo > 0, seconds < timed[lo - 1].end else { return nil }
      return timed[lo - 1].at
    }
    if segment != activeSegment || current != highlight {
      var changed = Set([highlight, current].compactMap { $0 })
      if segment != activeSegment {
        for id in [activeSegment, segment].compactMap({ $0 }) {
          changed.formUnion(segmentWords[id] ?? [])
        }
      }
      activeSegment = segment
      highlight = current
      storage.beginEditing()
      for index in changed { background(index) }
      storage.endEditing()
    }
    if focus != focused {
      focused = focus
      if let focus, ranges.indices.contains(focus) { scrollRangeToVisible(ranges[focus]) }
    }
  }
  private func background(_ index: Int) {
    guard words.indices.contains(index), ranges.indices.contains(index), let storage = textStorage
    else { return }
    var color: NSColor?
    // Adding alpha resolves these semantic colors immediately. SwiftUI/clock
    // updates can arrive outside this view's drawing appearance (e.g. light
    // app over a dark OS), otherwise a white highlight disappears on white.
    effectiveAppearance.performAsCurrentDrawingAppearance {
      if index == highlight {
        color = NSColor.labelColor.withAlphaComponent(0.22)
      } else if differenceSet.contains(index) {
        color = NSColor.systemOrange.withAlphaComponent(0.14)
      } else if marks, SpeechMarkPolicy.unsure(words[index]) {
        color = NSColor.systemRed.withAlphaComponent(0.09)
      } else if activeSegment == words[index].segment {
        color = NSColor.labelColor.withAlphaComponent(0.06)
      }
    }
    if let color {
      storage.addAttribute(.backgroundColor, value: color, range: ranges[index])
    } else {
      storage.removeAttribute(.backgroundColor, range: ranges[index])
    }
  }
  func textView(_ textView: NSTextView, clickedOnLink link: Any, at charIndex: Int) -> Bool {
    guard let url = link as? URL, url.scheme == "paddock-word",
      let index = Int(url.absoluteString.dropFirst("paddock-word:".count)),
      words.indices.contains(index), let start = words[index].start
    else { return true }
    onSeek?(start)
    return true
  }
}
