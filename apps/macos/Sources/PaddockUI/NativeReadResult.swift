import PaddockConversationCore
import SwiftUI

/// One result presentation shared by normal reads and the live camera. Every
/// value belongs to the frozen request, not the questions being edited now.
struct NativeReadResult: View {
  let result: NativeReadsModel.Run
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      let thoughts = result.response.diagnostics.thought?.values ?? []
      ForEach(Array(thoughts.enumerated()), id: \.offset) { index, thought in
        DisclosureGroup(thoughts.count > 1 ? "Thought \(index + 1)" : "Thought") {
          VStack(alignment: .leading, spacing: 8) {
            Text(
              "\(thought.tokens) tokens · \(thought.closed ? "finished" : "cut at the budget") · \(thought.ms, specifier: "%.0f") ms"
            )
            .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            Text(
              thought.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "(empty)" : thought.text
            )
            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
          }.padding(.top, 8)
        }.accessibilityIdentifier("reads-thought-result")
      }
      ForEach(result.questions) { question in
        if let skip = result.response.diagnostics.skipped?[question.questionID] {
          VStack(alignment: .leading, spacing: 8) {
            Text(question.questionID).font(.caption).foregroundStyle(.secondary)
            Text(question.instructions)
            Text("Not asked: \(skip.explanation).")
              .foregroundStyle(.secondary)
          }.textSelection(.enabled).padding(14).frame(maxWidth: .infinity, alignment: .leading)
            .background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
            .accessibilityIdentifier("reads-skipped-\(question.questionID)")
        } else if let answer = result.response.answers[question.questionID] {
          NativeReadAnswerView(
            question: question, answer: answer,
            diagnostic: result.response.diagnostics.questions.first {
              $0.id == question.questionID
            },
            readCount: result.response.diagnostics.reads)
        }
      }
      NativeReadDiagnostics(
        response: result.response, elapsedMilliseconds: result.elapsedMilliseconds)
    }
  }
}
