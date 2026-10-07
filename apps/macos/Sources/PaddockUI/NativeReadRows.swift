import PaddockConversationCore
import SwiftUI

struct NativeReadQuestionRow: View {
  @Binding var question: ReadQuestion
  let onDuplicate: () -> Void
  let onMove: (Int) -> Void
  let onRemove: () -> Void
  var position = 0
  var count = 1
  var supportedTypes = ReadQuestion.Kind.allCases.map(\.rawValue)
  var serverError: String?
  var onID: ((String) -> Void)?
  var onInstructions: ((String) -> Void)?
  var conditional = false
  var others: [ReadQuestion] = []
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 8) {
        Text("\(position + 1)").font(.caption).monospacedDigit().foregroundStyle(.secondary)
          .frame(minWidth: 12)
        Dropdown(title: "Question type", value: question.kind.title) {
          ForEach(
            ReadQuestion.Kind.allCases.filter { supportedTypes.contains($0.rawValue) }, id: \.self
          ) { kind in
            Button(kind.title) { question.kind = kind }
          }
        }.fixedSize()
        TextField(
          "Question ID",
          text: Binding(
            get: { question.questionID },
            set: {
              if let onID { onID($0) } else { question.questionID = $0 }
            })
        )
        .textFieldStyle(StudioPopoverFieldStyle())
        .font(.system(size: 12, design: .monospaced))
        .frame(minWidth: 60, maxWidth: 220).accessibilityLabel("Question ID")
        Spacer(minLength: 0)
        Menu {
          Button("Move up", systemImage: "arrow.up") { onMove(-1) }.disabled(position == 0)
          Button("Move down", systemImage: "arrow.down") { onMove(1) }.disabled(
            position == count - 1)
          Button("Duplicate", systemImage: "plus.square.on.square", action: onDuplicate)
          if conditional {
            Divider()
            Button("Ask only if…") {
              if let other = others.first(where: { row in
                !question.askIf.contains { $0.question == row.id }
              }) {
                question.askIf.append(.init(question: other.id))
              }
            }.disabled(
              others.allSatisfy { row in question.askIf.contains { $0.question == row.id } })
            Button("Knows the answer to…") {
              if let other = others.first(where: { !question.after.contains($0.id) }) {
                question.after.append(other.id)
              }
            }.disabled(others.allSatisfy { question.after.contains($0.id) })
            Button(question.alone ? "Read with the others" : "Read on its own canvas") {
              question.alone.toggle()
            }
          }
          Button("Remove", systemImage: "trash", role: .destructive, action: onRemove)
        } label: {
          Image(systemName: "ellipsis").frame(width: 24, height: 28)
        }
        .menuStyle(.button).menuIndicator(.hidden).buttonStyle(QuietButtonStyle()).fixedSize()
        .accessibilityLabel(
          "Question actions")
      }
      TextField(
        "What should the model decide?",
        text: Binding(
          get: { question.instructions },
          set: {
            if let onInstructions { onInstructions($0) } else { question.instructions = $0 }
          }), axis: .vertical
      )
      .textFieldStyle(StudioPopoverFieldStyle()).lineLimit(1...5)
      switch question.kind {
      case .noul:
        ViewThatFits(in: .horizontal) {
          HStack(spacing: 8) {
            yesField.frame(minWidth: 180)
            noField.frame(minWidth: 180)
          }
          VStack(spacing: 8) {
            yesField
            noField
          }
        }
      case .choice:
        ForEach($question.options) { $option in
          HStack {
            TextField("Option", text: $option.name).textFieldStyle(StudioPopoverFieldStyle()).frame(
              maxWidth: 140)
            TextField("Description", text: $option.description).textFieldStyle(
              StudioPopoverFieldStyle())
            Button {
              question.options.removeAll { $0.id == option.id }
            } label: {
              Image(systemName: "minus.circle")
            }
            .buttonStyle(QuietButtonStyle()).disabled(question.options.count <= 2)
            .accessibilityLabel("Remove option")
          }
        }
        Button("Add option", systemImage: "plus") { question.options.append(.init()) }
          .buttonStyle(QuietButtonStyle()).disabled(question.options.count >= 26)
      case .score:
        ForEach($question.levels) { $level in
          HStack {
            let index = question.levels.firstIndex { $0.id == level.id } ?? 0
            Text("\(index + 1)").font(.caption).monospacedDigit().foregroundStyle(.secondary)
              .frame(width: 16)
            TextField("Level", text: $level.name).textFieldStyle(StudioPopoverFieldStyle())
            Button {
              question.levels.swapAt(index, index - 1)
            } label: {
              Image(systemName: "arrow.up")
            }
            .buttonStyle(QuietButtonStyle()).disabled(index == 0).accessibilityLabel(
              "Move level up")
            Button {
              question.levels.swapAt(index, index + 1)
            } label: {
              Image(systemName: "arrow.down")
            }
            .buttonStyle(QuietButtonStyle()).disabled(index == question.levels.count - 1)
            .accessibilityLabel("Move level down")
            Button {
              question.levels.removeAll { $0.id == level.id }
            } label: {
              Image(systemName: "minus.circle")
            }
            .buttonStyle(QuietButtonStyle()).disabled(question.levels.count <= 2)
            .accessibilityLabel("Remove level")
          }
        }
        Button("Add level", systemImage: "plus") { question.levels.append(.init()) }
          .buttonStyle(QuietButtonStyle()).disabled(question.levels.count >= 26)
      }
      if question.conditional { NativeReadConditionEditor(question: $question, others: others) }
      if let validation = question.validation {
        Text(validation).font(.caption).foregroundStyle(PaddockStyle.caution)
      }
      if let serverError {
        Text(serverError).font(.caption).foregroundStyle(PaddockStyle.caution).textSelection(
          .enabled)
      }
    }.padding(12).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
  }

  private var yesField: some View {
    TextField("Yes means (optional)", text: $question.yesMeans).textFieldStyle(
      StudioPopoverFieldStyle())
  }
  private var noField: some View {
    TextField("No means (optional)", text: $question.noMeans).textFieldStyle(
      StudioPopoverFieldStyle())
  }
}

struct NativeReadAnswerView: View {
  let question: ReadQuestion
  let answer: ReadResponse.Answer
  let diagnostic: ReadResponse.Diagnostics.Question?
  var readCount = 1
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text(question.questionID).font(.caption).foregroundStyle(.secondary)
      Text(question.instructions).textSelection(.enabled)
      HStack(alignment: .firstTextBaseline) {
        Text(answer.label).font(.title3.weight(.semibold))
        if let score = answer.score {
          Text(score, format: .number.precision(.fractionLength(2))).monospacedDigit()
        }
        Spacer()
        HStack(spacing: 5) {
          ReadConfidenceSwatch(value: answer.confidence)
          Text("Confidence \(answer.confidence, specifier: "%.2f")").foregroundStyle(.secondary)
        }.font(.caption).fixedSize()
      }
      if answer.nearTie {
        Text("Near tie").font(.caption.weight(.medium)).foregroundStyle(PaddockStyle.caution)
      }
      if let noul = answer.noul { ReadScale(value: noul, lower: "No", upper: "Yes") }
      if let score = answer.score, let legend = answer.legend, legend.count > 1 {
        ReadScale(
          value: answer.scorePosition, lower: answer.bars.first?.name ?? "0",
          upper: answer.bars.last?.name ?? "\(legend.count - 1)",
          ticks: (0..<legend.count).map { Double($0) / Double(legend.count - 1) })
        Text("Score \(score, specifier: "%.2f") on 0 to \(legend.count - 1)")
          .font(.caption).foregroundStyle(.secondary)
      }
      ForEach(Array(answer.bars.enumerated()), id: \.offset) { _, bar in
        probability(bar.name, bar.probability)
      }
      // Conditional label probabilities and outside mass are different
      // quantities; never combine them into a normalized stacked chart.
      if let outside = answer.outside { probability("Outside the options", outside) }
      if (answer.outside ?? 0) > 0.5 {
        Text("Most probability is outside these options.").font(.caption).foregroundStyle(
          PaddockStyle.caution)
      }
      if let diagnostic {
        HStack {
          Text(
            "\(diagnostic.slotEntropy != nil || diagnostic.position == nil ? "Answer" : "Slot") entropy \(diagnostic.entropy, specifier: "%.3f") · \(diagnostic.entropy < 0.1 ? "settled" : diagnostic.entropy < log(2) ? "unsettled" : "split")"
          )
          Spacer()
          if readCount > 1, let agreement = answer.agreement {
            Text(
              "\(Int((agreement * Double(readCount)).rounded())) of \(readCount) reads agree"
            )
          }
        }.font(.caption).foregroundStyle(.secondary).monospacedDigit()
        if let reads = diagnostic.reads, reads.count > 1 {
          DisclosureGroup("Individual reads") {
            ForEach(Array(reads.enumerated()), id: \.offset) { i, read in
              HStack {
                Text("\(i + 1)").frame(width: 20, alignment: .leading)
                Text(read.pick)
                Spacer()
                Text(
                  "\(read.confidence, specifier: "%.2f") confidence · \(read.entropy, specifier: "%.3f") entropy"
                )
              }.font(.caption).monospacedDigit()
            }
          }
        }
      }
      if let confidence = answer.answerConfidence {
        Text("Answer probability \(confidence, format: .percent.precision(.fractionLength(1)))")
          .font(.caption).foregroundStyle(.secondary).monospacedDigit()
      }
      if let stderr = answer.stderr {
        Text("Standard error \(stderr, format: .percent.precision(.fractionLength(1)))")
          .font(.caption).foregroundStyle(.secondary).monospacedDigit()
      }
      if let stderr = answer.scoreStderr {
        Text("Score standard error \(stderr, specifier: "%.2f")")
          .font(.caption).foregroundStyle(.secondary).monospacedDigit()
      }
    }.padding(14).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
  }
  private func probability(_ name: String, _ value: Double) -> some View {
    HStack(spacing: 12) {
      Text(name).frame(width: 130, alignment: .leading).lineLimit(2)
      ProgressView(value: min(1, max(0, value))).tint(.primary).accessibilityLabel(name)
      Text(value, format: .percent.precision(.fractionLength(1))).monospacedDigit().frame(
        width: 58, alignment: .trailing)
    }.font(.caption)
  }
}
