import AVFoundation
import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native speech rendering and playback", .serialized) @MainActor
struct NativeSpeechTests {
  @Test func speakerLabelsKeepOneSelectableTextSurfaceAndWordClock() throws {
    let value = try JSONDecoder().decode(
      StudioState.Speech.self,
      from: Data(
        #"""
        {"clip":{"id":"clip","name":"recording.wav","mime":"audio/wav"},
         "words":[{"word":"Hello","start":0,"end":1,"segment":0,"speaker":0,"speakers":[0]},
                  {"word":"both","start":1,"end":2,"segment":0,"speakers":[0,1]},
                  {"word":"unknown","start":2,"end":3,"segment":0,"speakers":[]}],
         "differs":[],"facts":[],"guards":[],"subtitleExport":true,
         "diarization":{"model":"Nemotron","duration":3,"segments":[
           {"speaker":0,"start":0,"end":2},{"speaker":1,"start":1,"end":2}]}}
        """#.utf8))
    let view = SpeechTextView(usingTextLayoutManager: true)
    view.isSelectable = true
    view.update(speech: value, marks: false, position: nil, focus: nil)
    #expect(view.string == "Speaker 1\nHello\nSpeaker 1 / Speaker 2\nboth\nUnassigned\nunknown")
    let selection = NSRange(location: 0, length: (view.string as NSString).length)
    view.setSelectedRange(selection)
    let storage = try #require(view.textStorage)
    let at = (view.string as NSString).range(of: "both").location
    view.update(speech: value, marks: false, position: 1.5, focus: nil)
    #expect(view.selectedRange() == selection)
    #expect(view.textStorage === storage)
    #expect(storage.attribute(.backgroundColor, at: at, effectiveRange: nil) != nil)
  }
  @Test func highlightFollowsViewAppearanceInsteadOfTheClockCallbackAppearance() throws {
    let speech = try speech()
    let view = SpeechTextView(usingTextLayoutManager: true)
    let light = try #require(NSAppearance(named: .aqua))
    let dark = try #require(NSAppearance(named: .darkAqua))
    for (appearance, wrongContext, white) in [(light, dark, false), (dark, light, true)] {
      view.appearance = appearance
      wrongContext.performAsCurrentDrawingAppearance {
        view.update(speech: speech, marks: false, position: 0.5, focus: nil)
      }
      let color = try #require(
        (view.textStorage?.attribute(.backgroundColor, at: 0, effectiveRange: nil)
          as? NSColor)?.usingColorSpace(.deviceRGB))
      #expect(abs(color.alphaComponent - 0.22) < 0.001)
      #expect(white ? color.redComponent > 0.9 : color.redComponent < 0.1)
    }
  }
  @Test func confidenceControlOnlyAppearsWhenItCanChangeTheTranscript() throws {
    let clean = try JSONDecoder().decode(
      StudioState.Speech.self,
      from: Data(
        #"""
        {"words":[{"word":"Testar","confidence":0.98,"segment":-1},
                  {"word":"funktionen","confidence":0.45,"segment":-1}],
         "differs":[],"facts":[],"guards":[],"subtitleExport":false}
        """#.utf8))
    #expect(!SpeechMarkPolicy.anyUnsure(clean))
    #expect(SpeechMarkPolicy.anyUnsure(try speech()))
    #expect(!SpeechMarkPolicy.unsure(try speech().words[1]))
  }
  @Test func realWordClockDistinguishesShortWordsGapsAndSentenceWash() throws {
    let speech = try JSONDecoder().decode(
      StudioState.Speech.self,
      from: Data(
        #"""
        {"clip":{"id":"clip","name":"recording.wav","mime":"audio/wav"},
         "segments":[{"start":0}],
         "words":[{"word":"Hej","start":0.1,"end":0.14,"segment":0},
                  {"word":"världen","start":0.2,"end":0.6,"segment":0}],
         "differs":[],"facts":[],"guards":[],"subtitleExport":true}
        """#.utf8))
    let view = SpeechTextView(usingTextLayoutManager: true)
    view.isEditable = false
    view.isSelectable = true
    view.update(speech: speech, marks: true, position: nil, focus: nil)
    let storage = try #require(view.textStorage)
    let selection = NSRange(location: 1, length: 7)
    view.setSelectedRange(selection)
    func background(_ at: Int) -> NSColor? {
      storage.attribute(.backgroundColor, at: at, effectiveRange: nil) as? NSColor
    }
    view.update(speech: speech, marks: true, position: 0.11, focus: nil)
    #expect(background(0) == NSColor.labelColor.withAlphaComponent(0.22))
    #expect(background(4) == NSColor.labelColor.withAlphaComponent(0.06))
    view.update(speech: speech, marks: true, position: 0.16, focus: nil)
    #expect(background(0) == background(4), "Silence has no current word, only the sentence wash")
    view.update(speech: speech, marks: true, position: 0.25, focus: nil)
    #expect(background(4) == NSColor.labelColor.withAlphaComponent(0.22))
    #expect(view.selectedRange() == selection)
    #expect(view.textStorage === storage)
    #expect(view.string == "Hej världen")
    view.update(speech: speech, marks: true, position: nil, focus: nil)
    #expect(background(0) == nil && background(4) == nil)
  }
  @Test func pausedSeeksUpdateBothTranscriptLanesWithoutAutoplayOrReplacingSelection() async throws
  {
    _ = NSApplication.shared
    let speech = try speech()
    let clip = try #require(speech.clip)
    let player = StudioAudioPlayback()
    defer { player.reset() }
    let host = NSHostingView(
      rootView: VStack {
        ForEach(0..<2, id: \.self) { _ in
          NativeSpeechPlaybackText(
            speech: speech, player: player, markUnsure: false,
            focusedWord: nil, onSeek: { _ in })
        }
      }.frame(width: 500))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 500, height: 150),
      styleMask: .borderless, backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    defer { window.close() }
    host.layoutSubtreeIfNeeded()
    func texts(_ view: NSView) -> [SpeechTextView] {
      if let text = view as? SpeechTextView { return [text] }
      return view.subviews.flatMap(texts)
    }
    let lanes = texts(host)
    #expect(lanes.count == 2)
    let selection = NSRange(location: 3, length: 11)
    let stores = lanes.compactMap(\.textStorage)
    for lane in lanes { lane.setSelectedRange(selection) }
    #expect(
      await player.prepare(clip) {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
          "Paddock-paused-seek-\(UUID()).wav")
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 48000))
        buffer.frameLength = 48000
        buffer.floatChannelData![0].initialize(repeating: 0, count: 48000)
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
      })
    for (position, character) in [(0.5, 0), (1.5, 6), (0.25, 0), (1.25, 6)] {
      player.seek(position)
      try await Task.sleep(for: .milliseconds(45))
      host.layoutSubtreeIfNeeded()
      #expect(!player.playing)
      for (index, lane) in lanes.enumerated() {
        #expect(
          lane.textStorage?.attribute(.backgroundColor, at: character, effectiveRange: nil)
            as? NSColor
            == NSColor.labelColor.withAlphaComponent(0.22),
          "Paused seek at \(position) must highlight in lane \(index)")
        #expect(lane.selectedRange() == selection)
        #expect(lane.textStorage === stores[index])
      }
    }
    player.seek(2.5)
    try await Task.sleep(for: .milliseconds(45))
    host.layoutSubtreeIfNeeded()
    for lane in lanes {
      for character in [0, 6, 13] {
        #expect(
          lane.textStorage?.attribute(.backgroundColor, at: character, effectiveRange: nil)
            as? NSColor
            != NSColor.labelColor.withAlphaComponent(0.22),
          "Untimed words must not acquire fabricated timing when scrubbing")
      }
    }
    player.reset()
    try await Task.sleep(for: .milliseconds(45))
    host.layoutSubtreeIfNeeded()
    for lane in lanes {
      #expect(lane.textStorage?.attribute(.backgroundColor, at: 0, effectiveRange: nil) == nil)
      #expect(
        lane.textStorage?.attribute(.backgroundColor, at: 6, effectiveRange: nil) as? NSColor
          != NSColor.labelColor.withAlphaComponent(0.22))
    }
  }
  @Test func finishingTheRecordingEnablesSeekingWithoutChangingItsWords() throws {
    let withClip = try speech()
    let withoutClip = try JSONDecoder().decode(
      StudioState.Speech.self,
      from: Data(
        #"""
        {"words":[{"word":"Hello","start":0,"end":1,"segment":0,"confidence":0.2},
                  {"word":"across","start":1,"end":2,"segment":0},
                  {"word":"rows","start":2,"segment":0}],
         "differs":[1],"facts":[],"guards":[],"subtitleExport":true}
        """#.utf8))
    let view = SpeechTextView(usingTextLayoutManager: true)
    view.update(speech: withoutClip, marks: false, position: nil, focus: nil)
    #expect(view.textStorage?.attribute(.link, at: 0, effectiveRange: nil) == nil)
    view.update(speech: withClip, marks: false, position: nil, focus: nil)
    #expect(view.textStorage?.attribute(.link, at: 0, effectiveRange: nil) != nil)
    #expect(
      view.textStorage?.attribute(.toolTip, at: 0, effectiveRange: nil) as? String
        == "Click to play from 0:00.")
  }
  func speech() throws -> StudioState.Speech {
    try JSONDecoder().decode(
      StudioState.Speech.self,
      from: Data(
        #"""
        {"clip":{"id":"clip","name":"recording.m4a","mime":"audio/mp4"},
         "words":[{"word":"Hello","start":0,"end":1,"segment":0,"confidence":0.2},
                  {"word":"across","start":1,"end":2,"segment":0},
                  {"word":"rows","start":2,"segment":0}],
         "differs":[1],"facts":[],"guards":[],"subtitleExport":true}
        """#.utf8))
  }
  @Test func selectableTextAndSelectionSurvivePlayheadAndConfidenceUpdates() throws {
    _ = NSApplication.shared
    let speech = try speech()
    var sought: Double?
    let view = SpeechTextView(usingTextLayoutManager: true)
    view.isEditable = false
    view.isSelectable = true
    view.onSeek = { sought = $0 }
    view.update(speech: speech, marks: true, position: nil, focus: nil)
    #expect(view.textLayoutManager != nil)
    #expect(view.string == "Hello across rows")
    let selection = NSRange(location: 3, length: 11)
    view.setSelectedRange(selection)
    view.update(speech: speech, marks: true, position: 0.5, focus: nil)
    #expect(view.selectedRange() == selection)
    #expect(view.textStorage?.attribute(.backgroundColor, at: 0, effectiveRange: nil) != nil)
    view.update(speech: speech, marks: false, position: 1.5, focus: nil)
    #expect(view.selectedRange() == selection)
    #expect(view.textStorage?.attribute(.underlineStyle, at: 0, effectiveRange: nil) == nil)
    #expect(view.textStorage?.attribute(.underlineStyle, at: 6, effectiveRange: nil) != nil)
    #expect(view.textStorage?.attribute(.backgroundColor, at: 0, effectiveRange: nil) == nil)
    #expect(view.textStorage?.attribute(.backgroundColor, at: 6, effectiveRange: nil) != nil)
    view.update(speech: speech, marks: false, position: 2.5, focus: nil)
    #expect(
      view.textStorage?.attribute(.backgroundColor, at: 13, effectiveRange: nil) == nil,
      "Sentence-only timing must not pretend to be word timing")
    #expect(view.textView(view, clickedOnLink: URL(string: "paddock-word:1")!, at: 6))
    #expect(sought == 1)
    #expect(view.textView(view, clickedOnLink: URL(string: "https://untrusted.test")!, at: 0))
    #expect(sought == 1, "Non-transcript links must never escape to another app")
    #expect(view.textLayoutManager != nil, "Rendering must not fall back to TextKit 1")
  }
  @Test func speechMeasuresWrappedRowsWithoutAWebViewInBothAppearances() throws {
    let speech = try speech()
    for appearance in [NSAppearance.Name.aqua, .darkAqua] {
      let host = NSHostingView(
        rootView: NativeSpeechText(
          speech: speech, markUnsure: true,
          position: nil, focusedWord: nil, onSeek: { _ in }
        ).frame(width: 70))
      host.appearance = NSAppearance(named: appearance)
      let size = host.fittingSize
      host.frame = NSRect(origin: .zero, size: size)
      host.layoutSubtreeIfNeeded()
      #expect(size.height > 40 && size.height < 200)
      func find(_ view: NSView) -> SpeechTextView? {
        (view as? SpeechTextView) ?? view.subviews.lazy.compactMap(find).first
      }
      let text = try #require(find(host))
      #expect(text.isSelectable && !text.isEditable && !text.drawsBackground)
      text.setSelectedRange(NSRange(location: 0, length: (text.string as NSString).length))
      #expect(text.selectedRange().length == 17)
    }
  }
  @Test func nativeDecoderOwnsOnlyItsTemporaryCopyAndResetCancelsPendingLoad() async throws {
    let clip = try #require(try speech().clip)
    let player = StudioAudioPlayback()
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Paddock-native-test-\(UUID()).wav")
    let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1))
    let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16000))
    buffer.frameLength = 16000
    for i in 0..<16000 { buffer.floatChannelData![0][i] = Float(sin(Double(i) * 0.1) * 0.1) }
    do {
      let audio = try AVAudioFile(forWriting: url, settings: format.settings)
      try audio.write(from: buffer)
    }
    let before = try Data(contentsOf: url)
    var copy: URL?
    #expect(
      await player.prepare(clip) {
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
          "Paddock-playback-test-\(UUID()).wav")
        try FileManager.default.copyItem(at: url, to: destination)
        copy = destination
        return destination
      })
    #expect(player.duration == 1 && !player.playing)
    player.seek(0.5)
    #expect(player.position == 0.5)
    player.reset()
    #expect(!FileManager.default.fileExists(atPath: try #require(copy).path))
    #expect(try Data(contentsOf: url) == before)
    let pending = Task {
      await player.prepare(clip) {
        try await Task.sleep(for: .seconds(1))
        throw CancellationError()
      }
    }
    await Task.yield()
    player.reset()
    #expect(await pending.value == false)
    #expect(!player.loading && player.clipId == nil && player.error == nil)
    try FileManager.default.removeItem(at: url)
  }
}
