import Testing

@testable import PaddockConversationCore

struct DocumentRequestOptionsTests {
  @Test func lightOnGroundingIsAModeNotABoolean() {
    let capability: O = [
      "ocr": .object([
        "modes": .array([.string("plain"), .string("grounding")]), "grounding": .bool(true),
      ])
    ]
    for mode in ["grounding", "plain"] {
      let settings: O = ["ocrMode": .string(mode), "ocrRegions": .bool(true)]
      var body: O = [:]
      #expect(
        NativeDocumentRequestOptions.apply(
          fields: settings, capability: capability, input: input, body: &body))
      #expect(body["ocr"] == .object(["mode": .string(mode)]))
    }
    var body: O = [:]
    #expect(
      !NativeDocumentRequestOptions.apply(
        fields: ["ocrRegions": .bool(true)], capability: capability, input: input, body: &body))
    #expect(body["ocr"] == nil)
  }

  typealias V = ConversationValue
  typealias O = [String: V]
  let input: [V] = [
    .object([
      "role": .string("user"), "content": .array([.object(["type": .string("input_image")])]),
    ])
  ]
  let cap: O = [
    "ocr": .object(["modes": .array([.string("markdown")]), "grounding": .bool(true)]),
    "forensics": .object(["vision": .bool(true)]),
  ]
  @Test func gatesPreferencesAgainstEachLaneAndActualInput() {
    let settings: O = [
      "ocrMode": .string("markdown"), "ocrRegions": .bool(true), "forensicsEnabled": .bool(true),
    ]
    var body: O = [:]
    #expect(
      NativeDocumentRequestOptions.apply(
        fields: settings, capability: cap, input: input, body: &body))
    #expect(body["ocr"] == .object(["mode": .string("markdown"), "grounding": .bool(true)]))
    #expect(body["forensics"] == .string("on"))
    #expect(
      !NativeDocumentRequestOptions.apply(
        fields: settings, capability: [:], input: input, body: &body))
    #expect(body["ocr"] == nil && body["forensics"] == nil)
    #expect(
      !NativeDocumentRequestOptions.apply(fields: settings, capability: cap, input: [], body: &body)
    )
    #expect(body["ocr"] == nil)
    #expect(
      !NativeDocumentRequestOptions.apply(
        fields: ["ocrMode": .string("stale")], capability: cap, input: input, body: &body))
    #expect(body["ocr"] == nil && body["forensics"] == .string("off"))
    #expect(
      !NativeDocumentRequestOptions.apply(fields: [:], capability: [:], input: input, body: &body))
    #expect(body["file_metadata"] == nil)
    _ = NativeDocumentRequestOptions.apply(
      fields: ["fileMetadataEnabled": .bool(false)], capability: [:], input: input, body: &body)
    #expect(body["file_metadata"] == .string("off"))
  }
  @Test func toolsAreSuppressedForExtractionNotOrdinaryFollowups() {
    for tag in [
      "<chart2csv>", "<chart2code>", "<chart2summary>", "<tables_json>", "<tables_html>",
      "<tables_otsl>", "<custom_task>",
    ] {
      let request: [V] = [.object(["role": .string("user"), "content": .string(" \(tag)\n")])]
      var body: O = [:]
      let capability: O = ["task_tags": .array([.object(["tag": .string("<custom_task>")])])]
      #expect(
        NativeDocumentRequestOptions.apply(
          fields: [:], capability: capability, input: request, body: &body))
      #expect(
        !NativeDocumentRequestOptions.apply(
          fields: [:], capability: capability,
          input: request + [
            .object(["role": .string("user"), "content": .string("Explain the results")])
          ], body: &body))
    }
  }
}
