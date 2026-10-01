import Foundation

public struct TabularResultRow: Sendable, Equatable {
  public let row: Int
  public let value: String
  public let confidence: Double?
  public let probabilities: [Double]
  public let low: Double?
  public let high: Double?
}

public struct TabularResults: Sendable {
  public let rows: [TabularResultRow]
  public let classes: [String]
  public let gpuMilliseconds: Double?
  public let elapsedMilliseconds: Double?
  public let estimators: Int?

  /// Never render a partial/mis-shaped reply as successful predictions.
  public init(response: ConversationValue, plan: TabularPlan) throws {
    func invalid() -> ConversationFailure {
      .invalid("The table model returned an invalid prediction response.")
    }
    guard response["task"]?.string == plan.task.rawValue,
      let predictions = response["predictions"]?.array, predictions.count == plan.queryRows.count
    else { throw invalid() }
    func scalar(_ cell: ConversationValue) -> String? {
      switch cell {
      case .string(let s): s
      case .number(let n): NSDecimalNumber(decimal: n).stringValue
      case .bool(let b): b ? "true" : "false"
      default: nil
      }
    }
    let labels = response["classes"]?.array ?? []
    classes = labels.compactMap(scalar)
    if plan.task == .classification
      && (classes.count != labels.count || !(2...10).contains(classes.count))
    {
      throw invalid()
    }
    if let levels = response["quantile_levels"]?.array {
      guard levels.count == 999,
        levels.enumerated().allSatisfy({
          abs(($0.element.double ?? -1) - Double($0.offset + 1) / 1000) < 1e-6
        })
      else { throw invalid() }
    }
    rows = try predictions.enumerated().map { index, p in
      if plan.task == .classification {
        guard let winner = p["class"]?.integer, labels.indices.contains(winner),
          let raw = p["probabilities"]?.array, raw.count == labels.count
        else { throw invalid() }
        let probabilities = raw.compactMap(\.double)
        guard probabilities.count == raw.count,
          probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
          abs(probabilities.reduce(0, +) - 1) < 0.001,
          p["label"] == nil || p["label"] == labels[winner]
        else { throw invalid() }
        return TabularResultRow(
          row: plan.queryRows[index], value: scalar(labels[winner])!,
          confidence: probabilities[winner], probabilities: probabilities, low: nil, high: nil)
      }
      guard let raw = p["quantiles"]?.array, raw.count == 999 else { throw invalid() }
      let qs = raw.compactMap(\.double)
      guard qs.count == 999, qs.allSatisfy(\.isFinite),
        zip(qs, qs.dropFirst()).allSatisfy({ $0 <= $1 }),
        let median = p["median"]?.double, median.isFinite
      else { throw invalid() }
      return TabularResultRow(
        row: plan.queryRows[index], value: Self.plain(median),
        confidence: nil, probabilities: [], low: qs[99], high: qs[899])
    }
    gpuMilliseconds = response["usage"]?["gpu_ms"]?.double.flatMap {
      $0 >= 0 && $0.isFinite ? $0 : nil
    }
    elapsedMilliseconds = response["usage"]?["elapsed_ms"]?.double.flatMap {
      $0 >= 0 && $0.isFinite ? $0 : nil
    }
    estimators = response["num_estimators"]?.integer
  }

  public static func plain(_ value: Double) -> String {
    NSDecimalNumber(value: value).stringValue
  }
  public func csv(table: TabularTable, spec: TabularSpec) -> String {
    let predictions = Dictionary(uniqueKeysWithValues: rows.map { ($0.row, $0) })
    let regression = rows.first?.low != nil
    let extra = regression ? ["p10", "p90"] : ["confidence"]
    func quote(_ s: String) -> String {
      s.contains(where: { ",\"\r\n".contains($0) })
        ? "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\"" : s
    }
    var lines = [(table.header + extra).map(quote).joined(separator: ",")]
    for (i, source) in table.rows.enumerated() {
      var row = source
      if let result = predictions[i] {
        row[spec.target] = result.value
        if let low = result.low, let high = result.high {
          row += [Self.plain(low), Self.plain(high)]
        } else {
          row += [
            result.confidence.map {
              String(format: "%.4f", locale: Locale(identifier: "en_US_POSIX"), $0)
            } ?? ""
          ]
        }
      } else {
        row += extra.map { _ in "" }
      }
      lines.append(row.map(quote).joined(separator: ","))
    }
    return lines.joined(separator: "\n") + "\n"
  }
}
