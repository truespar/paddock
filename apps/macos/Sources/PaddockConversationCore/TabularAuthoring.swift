import Foundation

/// Native counterpart of studio/src/lib/tables.ts. Inference and fitted
/// preprocessing stay in Rust; this is import, column mapping and presentation.
public enum TabularTask: String, Sendable, Codable {
  case classification, regression
  /// Exact synthetic examples from web Studio's fixed generator. Tests pin
  /// these resources to that generator; no real customer dataset is bundled.
  public var example: String {
    guard let url = Bundle.module.url(forResource: "tabular-\(rawValue)", withExtension: "csv"),
      let text = try? String(contentsOf: url, encoding: .utf8)
    else { return "" }
    return text
  }
}

public struct TabularLimits: Sendable, Equatable {
  public let task: TabularTask
  public var maxContextRows = 4096
  public var maxQueryRows = 1024
  public var maxColumns = 500
  public var maxCells = 131072
  public var maxClasses = 10
  public var defaultEstimators = 8
  public var maxEstimators = 16

  public init(task: TabularTask) { self.task = task }
  public init(server: ConversationValue) throws {
    guard let caps = server["tabular"]?["capabilities"],
      let task = caps["task"]?.string.flatMap(TabularTask.init(rawValue:))
    else { throw ConversationFailure.invalid("This endpoint is not a table predictor.") }
    self.init(task: task)
    func limit(_ key: String, _ fallback: Int) throws -> Int {
      guard let value = caps[key] else { return fallback }
      guard let n = value.integer, n > 0, n <= fallback else {
        throw ConversationFailure.invalid("Unsupported table limit: \(key).")
      }
      return n
    }
    maxContextRows = try limit("max_context_rows", maxContextRows)
    maxQueryRows = try limit("max_query_rows", maxQueryRows)
    maxColumns = try limit("max_columns", maxColumns)
    maxCells = try limit("max_cells", maxCells)
    maxClasses = try limit("max_classes", maxClasses)
    maxEstimators = try limit("max_estimators", maxEstimators)
    defaultEstimators = try limit("default_estimators", maxEstimators)
    if caps["default_estimators"] == nil { defaultEstimators = min(8, maxEstimators) }
  }
}

public enum TabularColumnType: String, Sendable, CaseIterable, Codable {
  case numerical = "Numbers"
  case categorical = "Categories"
  public init(from decoder: any Decoder) throws {
    let c = try decoder.singleValueContainer()
    switch try c.decode(String.self) {
    case "numerical": self = .numerical
    case "categorical": self = .categorical
    default: throw DecodingError.dataCorruptedError(in: c, debugDescription: "Unknown column type")
    }
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.singleValueContainer()
    try c.encode(self == .numerical ? "numerical" : "categorical")
  }
}

public struct TabularSpec: Sendable, Equatable, Codable {
  public var target: Int
  public var use: [Bool]
  public var types: [TabularColumnType]
  public init(table: TabularTable) {
    target = max(0, table.header.count - 1)
    use = table.header.map { _ in true }
    types = table.header.indices.map { j in
      let present = table.rows.map { $0[j] }.filter { !TabularTable.isMissing($0) }
      return !present.isEmpty && present.allSatisfy { TabularTable.number($0) != nil }
        ? .numerical : .categorical
    }
  }
}

public struct TabularTable: Sendable, Equatable {
  public static let maximumBytes = 8 * 1024 * 1024
  public let header: [String]
  public let rows: [[String]]

  public static func isMissing(_ value: String) -> Bool {
    ["", "na", "n/a", "nan", "null", "none", "?", "-"].contains(
      value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
  }
  public static func number(_ value: String) -> Double? {
    let text = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard
      text.range(
        of: #"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"#,
        options: .regularExpression) != nil,
      let n = Double(text), n.isFinite, abs(n) <= 1e30
    else { return nil }
    return n
  }

  /// RFC 4180 plus spreadsheet TSV and semicolon CSV, as in web Studio.
  /// Malformed/truncated files are rejected, never silently reinterpreted.
  public static func parse(_ source: String) throws -> Self {
    guard source.utf8.count <= maximumBytes else {
      throw ConversationFailure.invalid("Open a table smaller than 8 MiB.")
    }
    var bytes = Array(source.utf8)
    if bytes.starts(with: [0xef, 0xbb, 0xbf]) { bytes.removeFirst(3) }
    let first = bytes.prefix { $0 != 10 && $0 != 13 }
    let delimiter: UInt8 =
      first.contains(9) && !first.contains(44)
      ? 9
      : first.contains(59) && !first.contains(44) ? 59 : 44
    var records: [[String]] = []
    var row: [String] = []
    var cell: [UInt8] = []
    var quoted = false
    var wasQuoted = false
    var closedQuote = false
    var cells = 0
    func endCell() throws {
      let raw = String(decoding: cell, as: UTF8.self)
      row.append(wasQuoted ? raw : raw.trimmingCharacters(in: .whitespacesAndNewlines))
      cell.removeAll(keepingCapacity: true)
      wasQuoted = false
      closedQuote = false
      guard row.count <= 501 else {
        throw ConversationFailure.invalid(
          "A table can contain at most 501 columns, including the target.")
      }
    }
    func endRow() throws {
      try endCell()
      if row.count > 1 || row[0] != "" {
        records.append(row)
        cells += row.count
      }
      row.removeAll(keepingCapacity: true)
      guard records.count <= 5121, cells <= 137693 else {
        throw ConversationFailure.invalid(
          "The table exceeds the row or cell limit. Split it explicitly before importing.")
      }
    }
    var i = 0
    while i < bytes.count {
      if i % 4096 == 0 { try Task.checkCancellation() }
      let c = bytes[i]
      if quoted {
        if c == 34 {
          if i + 1 < bytes.count && bytes[i + 1] == 34 {
            cell.append(34)
            i += 1
          } else {
            quoted = false
            closedQuote = true
          }
        } else {
          cell.append(c)
        }
      } else if c == delimiter {
        try endCell()
      } else if c == 10 || c == 13 {
        if c == 13 && i + 1 < bytes.count && bytes[i + 1] == 10 { i += 1 }
        try endRow()
      } else if c == 34 && cell.allSatisfy({ $0 == 32 || $0 == 9 }) && !closedQuote {
        quoted = true
        wasQuoted = true
        cell.removeAll(keepingCapacity: true)
      } else if closedQuote {
        guard c == 32 || c == 9 else {
          throw ConversationFailure.invalid("Unexpected text after a quoted cell.")
        }
      } else {
        cell.append(c)
      }
      i += 1
    }
    guard !quoted else { throw ConversationFailure.invalid("The table ends inside a quoted cell.") }
    if !cell.isEmpty || !row.isEmpty || wasQuoted { try endRow() }
    guard let firstRow = records.first, records.count > 1 else {
      throw ConversationFailure.invalid("The table needs a header and at least one data row.")
    }
    let header = firstRow.enumerated().map {
      $0.element.isEmpty ? "column \($0.offset + 1)" : $0.element
    }
    let rows = try records.dropFirst().enumerated().map { index, row in
      guard row.count <= header.count else {
        throw ConversationFailure.invalid("Row \(index + 1) has more cells than the header.")
      }
      return row + Array(repeating: "", count: header.count - row.count)
    }
    guard rows.count * header.count <= 137192 else {
      throw ConversationFailure.invalid("The table exceeds the cell limit.")
    }
    return Self(header: header, rows: rows)
  }

  public func plan(spec: TabularSpec, limits: TabularLimits, estimators: Int, seed: Int) throws
    -> TabularPlan
  {
    func require(_ valid: Bool, _ message: String) throws {
      if !valid { throw ConversationFailure.invalid(message) }
    }
    try require(
      header.indices.contains(spec.target) && spec.use.count == header.count
        && spec.types.count == header.count, "Choose a target column.")
    let features = header.indices.filter { $0 != spec.target && spec.use[$0] }
    try require(!features.isEmpty, "Choose at least one input column besides the target.")
    try require(
      features.count <= limits.maxColumns,
      "This model reads at most \(limits.maxColumns) input columns.")
    let context = rows.indices.filter { !Self.isMissing(rows[$0][spec.target]) }
    let query = rows.indices.filter { Self.isMissing(rows[$0][spec.target]) }
    try require(!context.isEmpty, "The model needs labelled rows. Fill in some target values.")
    try require(
      !query.isEmpty, "Every row has a target value. Leave the target empty on the rows to predict."
    )
    try require(
      context.count <= limits.maxContextRows,
      "At most \(limits.maxContextRows) labelled rows are supported.")
    try require(
      query.count <= limits.maxQueryRows,
      "At most \(limits.maxQueryRows) rows can be predicted at once.")
    try require(
      rows.count * features.count <= limits.maxCells,
      "This model reads at most \(limits.maxCells) input cells.")
    try require(
      (1...limits.maxEstimators).contains(estimators), "Ensemble must be 1–\(limits.maxEstimators)."
    )
    try require(
      seed >= 0 && seed <= 9_007_199_254_740_991,
      "Use a whole-number seed from 0 to 9007199254740991.")
    func value(_ text: String, column: Int, row: Int) throws -> ConversationValue {
      if Self.isMissing(text) { return .null }
      if spec.types[column] == .categorical {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        try require(
          trimmed.utf8.count <= 4096,
          "Row \(row + 1), \(header[column]): categories must be at most 4,096 bytes.")
        return .string(trimmed)
      }
      guard let n = Self.number(text) else {
        throw ConversationFailure.invalid(
          "Row \(row + 1), \(header[column]) is not a number. Change its type to Categories or fix the cell."
        )
      }
      return .number(Decimal(n))
    }
    let values = try rows.enumerated().map { i, row in
      ConversationValue.array(try features.map { try value(row[$0], column: $0, row: i) })
    }
    let labels = context.map {
      rows[$0][spec.target].trimmingCharacters(in: .whitespacesAndNewlines)
    }
    let targets: [ConversationValue]
    if limits.task == .classification {
      let classes = Set(labels)
      try require(classes.count >= 2, "The labelled rows need at least two classes.")
      try require(
        classes.count <= limits.maxClasses,
        "This model tells apart at most \(limits.maxClasses) classes.")
      try require(
        labels.allSatisfy { $0.utf8.count <= 4096 }, "Class labels must be at most 4,096 bytes.")
      targets = labels.map(ConversationValue.string)
    } else {
      targets = try labels.enumerated().map { i, label in
        guard let n = Self.number(label) else {
          throw ConversationFailure.invalid(
            "This model predicts numbers. The target of row \(context[i] + 1) is not numeric.")
        }
        return .number(Decimal(n))
      }
    }
    let body: ConversationValue = .object([
      "preprocessing": .string("sdm_v1"), "context": .array(context.map { values[$0] }),
      "targets": .array(targets), "query": .array(query.map { values[$0] }),
      "categorical": .array(features.map { .bool(spec.types[$0] == .categorical) }),
      "num_estimators": .number(Decimal(estimators)), "seed": .number(Decimal(seed)),
    ])
    return TabularPlan(
      task: limits.task, body: body, contextRows: context, queryRows: query, features: features)
  }
}

public struct TabularPlan: Sendable, Equatable {
  public let task: TabularTask
  public let body: ConversationValue
  public let contextRows: [Int]
  public let queryRows: [Int]
  public let features: [Int]
  public func request(model: String) throws -> ConversationValue {
    var fields = body.object ?? [:]
    fields["model"] = .string(model)
    let value = ConversationValue.object(fields)
    guard try JSONEncoder().encode(value).count <= TabularTable.maximumBytes else {
      throw ConversationFailure.invalid(
        "The prediction request exceeds 8 MiB. Use a smaller table.")
    }
    return value
  }
  public func curl(port: UInt16, model: String) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let json = String(decoding: try encoder.encode(request(model: model)), as: UTF8.self)
      .replacingOccurrences(of: "'", with: "'\\''")
    return
      "curl http://localhost:\(port)/v1/tabular/predictions \\\n  -H 'Content-Type: application/json' \\\n  -H 'Authorization: Bearer <api key>' \\\n  -d '\(json)'"
  }
}
