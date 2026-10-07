import PaddockConversationCore
import SwiftUI

struct NativeReadConditionEditor: View {
  @Binding var question: ReadQuestion
  let others: [ReadQuestion]
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      ForEach(Array(question.askIf.enumerated()), id: \.element.question) { index, condition in
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            Text("Ask only if").foregroundStyle(.secondary)
            Dropdown(title: "Question", value: name(condition.question)) {
              ForEach(
                others.filter { row in
                  row.id == condition.question || !question.askIf.contains { $0.question == row.id }
                }
              ) { row in
                Button(row.questionID) {
                  question.askIf[index] = .init(question: row.id)
                }
              }
            }
            Spacer(minLength: 0)
            Button("Remove condition", systemImage: "xmark") { question.askIf.remove(at: index) }
              .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
          }
          LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 120), alignment: .leading)], alignment: .leading
          ) {
            ForEach(others.first { $0.id == condition.question }?.answerNames ?? [], id: \.self) {
              answer in
              Toggle(
                answer.isEmpty ? "(unnamed)" : answer,
                isOn: Binding(
                  get: {
                    question.askIf.indices.contains(index)
                      && question.askIf[index].answers.contains(answer)
                  },
                  set: { on in
                    guard question.askIf.indices.contains(index) else { return }
                    question.askIf[index].answers.removeAll { $0 == answer }
                    if on { question.askIf[index].answers.append(answer) }
                  })
              )
              .toggleStyle(.checkbox)
            }
          }
        }
      }
      ForEach(Array(question.after.enumerated()), id: \.element) { index, id in
        HStack {
          Text("Knows the answer to").foregroundStyle(.secondary)
          Dropdown(title: "Prior question", value: name(id)) {
            ForEach(others.filter { $0.id == id || !question.after.contains($0.id) }) { row in
              Button(row.questionID) { question.after[index] = row.id }
            }
          }
          Spacer(minLength: 0)
          Button("Remove dependency", systemImage: "xmark") { question.after.remove(at: index) }
            .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
        }
      }
      if question.alone {
        HStack {
          Text("Read on its own canvas").foregroundStyle(.secondary)
          Spacer()
          Button("Read with the others", systemImage: "xmark") { question.alone = false }
            .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
        }
      }
    }.font(.caption).accessibilityIdentifier("reads-conditions")
  }
  private func name(_ id: UUID) -> String {
    others.first { $0.id == id }?.questionID ?? "Missing question"
  }
}
