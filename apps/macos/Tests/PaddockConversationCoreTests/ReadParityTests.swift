import Foundation
import PaddockConversationCore
import Testing

@Suite("Reads web feature parity")
struct ReadParityTests {
  static let request =
    #"{"state":"A ticket","questions":{"kind":{"type":"choice","instructions":"Classify it","criteria":{"bug":"A defect","question":"A question"}},"urgent":{"type":"noul","ask_if":{"kind":["bug"]},"depends_on":["kind"],"alone":true},"escalate":{"type":"noul","ask_if":{"urgent":["yes"]}}}}"#
  static let response =
    #"{"model":"diffusion","answers":{"kind":{"type":"choice","choice":"question","confidence":0.9,"agreement":1,"outside":0,"stderr":0.02,"probabilities":{"bug":0.1,"question":0.9}},"urgent":null,"escalate":null},"diagnostics":{"canvas":256,"reads":2,"steps":4,"format":"indexed","stages":[["kind"]],"chunks":[["kind"]],"conditioning":"prefill","skipped":{"urgent":{"because":"kind","was":"question","wanted":["bug"]},"escalate":{"because":"urgent","was":null,"wanted":["yes"]}},"thought":{"text":"Checking the ticket.","tokens":5,"closed":false,"ms":12},"questions":[{"id":"kind","label":"question","entropy":0.2,"slot_entropy":0.8,"reads":[{"pick":"question","confidence":0.9,"entropy":0.2,"argmax":"B"}]}],"timing":{"total_ms":20}}}"#
  func draft() throws -> ReadDraft { try ReadDraft.parse(Data(Self.request.utf8)) }
  func response(_ raw: String = Self.response) throws -> ReadResponse {
    try JSONDecoder().decode(ReadResponse.self, from: Data(raw.utf8))
  }

  @Test func conditionalWireRoundTripsThroughQuestionSetsAndSharedHistory() throws {
    let input = try draft()
    let copy = try ReadDraft.parse(Data(input.orderedJSON().utf8))
    #expect(copy.setBody == input.setBody)
    #expect(copy.questions[1].alone)
    #expect(copy.questions[1].after == [copy.questions[0].id])
    let wire = input.setBody["questions"]!
    let run: ConversationValue = .object(["questions": wire, "state": .string(input.state)])
    let restored = try ReadHistoryDocument.draft(run)
    #expect(restored.setBody == input.setBody)
    try response().validate(for: restored.questions)
  }

  @Test func renameAndRemovalPreserveStableReferences() throws {
    var input = try draft()
    let id = input.questions[0].id
    input.questions[0].questionID = "category"
    #expect(
      input.setBody["questions"]?["urgent"]?["ask_if"]?["category"] == .array([.string("bug")]))
    input.questions[0].options[0].name = "defect"
    input.followAnswerRename(id, before: ["bug", "question"], after: ["defect", "question"])
    #expect(input.validation() == nil)
    #expect(input.questions[1].askIf[0].answers == ["defect"])
    input.removeQuestion(id)
    #expect(input.questions[0].askIf.isEmpty && input.questions[0].after.isEmpty)
    #expect(input.validation() == nil)
  }

  @Test func dependencyErrorsCannotBeSilentlyDiscarded() throws {
    for raw in [
      #"{"a":{"type":"noul","depends_on":["missing"]}}"#,
      #"{"a":{"type":"noul","depends_on":["a"]}}"#,
      #"{"a":{"type":"noul","depends_on":["b"]},"b":{"type":"noul","depends_on":["a"]}}"#,
      #"{"a":{"type":"noul"},"b":{"type":"noul","ask_if":{"a":[]}}}"#,
      #"{"a":{"type":"noul"},"b":{"type":"noul","ask_if":{"a":["maybe"]}}}"#,
      #"{"a":{"type":"noul","alone":"true"}}"#,
    ] { #expect(throws: (any Error).self) { try ReadDraft.parse(Data(raw.utf8)) } }
  }

  @Test func webImportShorthandDoesNotCreateDuplicateDependencyControls() throws {
    let input = try ReadDraft.parse(
      Data(
        #"{"a":{"type":"noul"},"b":{"type":"noul","ask_if":{"a":"yes"},"depends_on":["a","a"]}}"#
          .utf8))
    #expect(input.questions[1].after == [input.questions[0].id])
    #expect(input.questions[1].askIf[0].answers == ["yes"])
  }

  @Test func nullAnswersRequireValidConditionalSkipReasons() throws {
    let input = try draft()
    let result = try response()
    try result.validate(for: input.questions)
    #expect(result.answers.count == 1)
    #expect(result.diagnostics.skipped?["escalate"]?.explanation.contains("was not asked") == true)
    for raw in [
      Self.response.replacingOccurrences(of: #""was":"question""#, with: #""was":"bug""#),
      Self.response.replacingOccurrences(of: #""wanted":["bug"]"#, with: #""wanted":["question"]"#),
      Self.response.replacingOccurrences(of: #""escalate":null"#, with: #""unexpected":null"#),
    ] { #expect(throws: (any Error).self) { try response(raw).validate(for: input.questions) } }
  }

  @Test func thoughtsAndStatisticalDiagnosticsAreNotDropped() throws {
    let result = try response()
    #expect(result.answers["kind"]?.stderr == 0.02)
    #expect(result.diagnostics.thought?.values.first?.tokens == 5)
    #expect(result.diagnostics.thought?.values.first?.closed == false)
    #expect(result.diagnostics.questions.first?.slotEntropy == 0.8)
    #expect(result.diagnostics.questions.first?.reads?.first?.argmax == "B")
    #expect(result.diagnostics.steps == 4 && result.diagnostics.stages == [["kind"]])
    let multi = Self.response.replacingOccurrences(
      of: #""thought":{"text":"Checking the ticket.","tokens":5,"closed":false,"ms":12}"#,
      with:
        #""thought":[{"text":"One","tokens":1,"closed":true,"ms":2},{"text":"Two","tokens":1,"closed":false,"ms":3}]"#
    )
    #expect(try response(multi).diagnostics.thought?.values.count == 2)
    let answer = try JSONDecoder().decode(
      ReadResponse.Answer.self,
      from: Data(
        #"{"type":"score","confidence":0.8,"score":1,"score_stderr":0.12,"stderr":0.03}"#.utf8))
    #expect(answer.scoreStderr == 0.12)
  }

  @Test func legacyLocalQuestionsStillDecodeWithoutConditionalFields() throws {
    let old = #"{"id":"00000000-0000-0000-0000-000000000001","questionID":"a","kind":"noul"}"#
    let q = try JSONDecoder().decode(ReadQuestion.self, from: Data(old.utf8))
    #expect(!q.conditional)
    #expect(try JSONDecoder().decode(ReadQuestion.self, from: JSONEncoder().encode(q)) == q)
  }

  @Test func apiExamplesKeepAuthoredOrderAndUseSafeMultipartFiles() throws {
    var input = try draft()
    let text = try input.curl(port: 1234, model: "diffusion")
    #expect(text.contains("http://localhost:1234/v1/systemone"))
    #expect(text.contains("Authorization: Bearer <api key>"))
    let body = try #require(text.components(separatedBy: "<<'JSON'\n").last)
    #expect(try ReadDraft.parse(Data(body.dropLast(5).utf8)).ordering == input.ordering)
    input.images = [.init(name: "a';\"$(touch nope)\n.png", url: "data:image/png;base64,SECRET")]
    let multipart = try input.curl(port: 1234, model: "diffusion")
    #expect(!multipart.contains("SECRET") && !multipart.contains("base64"))
    #expect(multipart.contains("request=<request.json;type=application/json"))
    #expect(multipart.contains("'\\''"))
    #expect(!multipart.contains("\n.png"))
  }
}
