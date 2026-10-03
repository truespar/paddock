import Foundation
import Testing

@testable import PaddockConversationCore

struct SpeechDiarizationTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  let meta: O = [
    "words": .array([
      .object([
        "word": .string("Hello!"), "start": .number(0), "end": .number(1),
        "confidence": .number(0.7),
      ])
    ]), "language": .string("en"),
  ]
  func response() -> O {
    var word = meta["words"]!.array![0].object!
    word["speaker"] = .number(0)
    word["speakers"] = .array([.number(0)])
    return [
      "words": .array([.object(word)]), "duration": .number(3),
      "attribution": .string("time_overlap_v1"), "model": .string("Nemotron"),
      "segments": .array([.object(["speaker": .number(0), "start": .number(0), "end": .number(2)])]
      ),
    ]
  }
  @Test func persistsOriginalWordsAndTimeline() throws {
    let out = try NativeSpeechDiarization.merge(meta: meta, result: response())
    #expect(out["words"]?.array?[0]["word"] == .string("Hello!"))
    #expect(out["words"]?.array?[0]["confidence"] == .number(0.7))
    #expect(out["diarization"]?["words"] == nil)
    #expect(out["language"] == .string("en"))
    let restored = try JSONDecoder().decode(O.self, from: JSONEncoder().encode(out))
    #expect(restored == out)
    #expect(
      NativeSpeechMetadata.renderWords(out, text: "Hello!", streaming: false)[0]["speakers"]
        == .array([.number(0)]))
  }
  @Test func rejectsChangedWordsTimesAndIntervals() {
    for key in ["word", "start", "end"] {
      var value = response()
      var word = value["words"]!.array![0].object!
      word[key] = key == "word" ? .string("Wrong") : .number(2)
      value["words"] = .array([.object(word)])
      #expect(throws: (any Error).self) {
        try NativeSpeechDiarization.merge(meta: meta, result: value)
      }
    }
    var bad = response()
    bad["segments"] = .array([
      .object(["speaker": .number(8), "start": .number(0), "end": .number(2)])
    ])
    #expect(throws: (any Error).self) { try NativeSpeechDiarization.merge(meta: meta, result: bad) }
  }
}
