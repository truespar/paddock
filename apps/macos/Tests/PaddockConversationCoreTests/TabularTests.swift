import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Tables web/native wire parity")
struct TabularTests {
  @Test func webFixtureRequestsMatchExactly() throws {
    let url = Bundle.module.url(
      forResource: "tables-web-parity", withExtension: "json", subdirectory: "Fixtures")!
    let fixtures = try JSONDecoder().decode([ConversationValue].self, from: Data(contentsOf: url))
    for f in fixtures {
      let table = try TabularTable.parse(f["text"]!.string!)
      let plan = try table.plan(
        spec: TabularSpec(table: table),
        limits: TabularLimits(task: TabularTask(rawValue: f["task"]!.string!)!), estimators: 8,
        seed: 7)
      #expect(table.header == f["table"]?["header"]?.array?.compactMap(\.string))
      #expect(plan.body == f["body"])
      #expect(plan.contextRows == f["contextRows"]?.array?.compactMap(\.integer))
      #expect(plan.queryRows == f["queryRows"]?.array?.compactMap(\.integer))
      #expect(plan.features == f["features"]?.array?.compactMap(\.integer))
    }
  }
  @Test func quotedCSVAndBOMAndMissingValues() throws {
    let table = try TabularTable.parse(
      "\u{feff}a,b,\r\n1,\"x, y\",\n2,\"say \"\"hi\"\"\",3\n\n4\n5,\"two\nlines\",\n")
    #expect(table.header == ["a", "b", "column 3"])
    #expect(
      table.rows == [
        ["1", "x, y", ""], ["2", "say \"hi\"", "3"], ["4", "", ""], ["5", "two\nlines", ""],
      ])
    #expect(["", "NA", "n/a", "NaN", " null ", "?", "-"].allSatisfy(TabularTable.isMissing))
    #expect(!TabularTable.isMissing("0"))
    #expect(TabularTable.number(" 3.5e2 ") == 350)
    for bad in ["1,5", "1e40", "0x10", "inf", "NaN"] { #expect(TabularTable.number(bad) == nil) }
    for bad in ["a,b\n1,\"unterminated", "a,b\n1,2,3", "a,b\n1,\"x\"broken", "a,b\n"] {
      #expect(throws: (any Error).self) { try TabularTable.parse(bad) }
    }
  }
  @Test func examplesAndLimits() throws {
    for task in [TabularTask.classification, .regression] {
      let table = try TabularTable.parse(task.example)
      let spec = TabularSpec(table: table)
      let plan = try table.plan(
        spec: spec, limits: TabularLimits(task: task), estimators: 8, seed: 0)
      #expect(
        plan.contextRows.count == 120 && plan.queryRows.count == 6 && plan.features.count == 6)
      for estimators in [0, 17] {
        #expect(throws: (any Error).self) {
          try table.plan(
            spec: spec, limits: TabularLimits(task: task), estimators: estimators, seed: 0)
        }
      }
      var small = TabularLimits(task: task)
      small.maxCells = 5
      #expect(throws: (any Error).self) {
        try table.plan(spec: spec, limits: small, estimators: 8, seed: 0)
      }
      #expect(throws: (any Error).self) {
        try table.plan(spec: spec, limits: TabularLimits(task: task), estimators: 8, seed: -1)
      }
    }
    let many = try TabularTable.parse(
      "x,y\n" + (0..<11).map { "\($0),class\($0)" }.joined(separator: "\n") + "\n99,\n")
    #expect(throws: (any Error).self) {
      try many.plan(
        spec: TabularSpec(table: many), limits: TabularLimits(task: .classification), estimators: 8,
        seed: 0)
    }
    #expect(throws: (any Error).self) { try TabularLimits(server: .object([:])) }
  }
  @Test func exportClassificationAndRejectMalformedReplies() throws {
    let table = try TabularTable.parse("x,y\n1,\"a, b\"\n2,c\n3,\n")
    let spec = TabularSpec(table: table)
    let plan = try table.plan(
      spec: spec, limits: TabularLimits(task: .classification), estimators: 8, seed: 0)
    let source =
      #"{"task":"classification","classes":["a, b","c"],"predictions":[{"class":0,"label":"a, b","probabilities":[0.75,0.25]}]}"#
    func result(_ text: String) throws -> TabularResults {
      try TabularResults(
        response: JSONDecoder().decode(ConversationValue.self, from: Data(text.utf8)), plan: plan)
    }
    #expect(
      try result(source).csv(table: table, spec: spec)
        == "x,y,confidence\n1,\"a, b\",\n2,c,\n3,\"a, b\",0.7500\n")
    for bad in [
      source.replacingOccurrences(of: "0.25", with: "0.5"),
      source.replacingOccurrences(of: "\"class\":0", with: "\"class\":2"),
      source.replacingOccurrences(of: "\"label\":\"a, b\"", with: "\"label\":\"c\""),
      source.replacingOccurrences(of: "classification", with: "regression"),
    ] {
      #expect(throws: (any Error).self) { try result(bad) }
    }
    #expect(try plan.curl(port: 1234, model: "kumo").contains("Authorization: Bearer <api key>"))
  }
  @Test func regressionUsesQuantilesInTargetUnits() throws {
    let table = try TabularTable.parse("x,y\n1,2\n2,4\n3,\n")
    let spec = TabularSpec(table: table)
    let plan = try table.plan(
      spec: spec, limits: TabularLimits(task: .regression), estimators: 8, seed: 0)
    let raw: ConversationValue = .object([
      "task": .string("regression"),
      "predictions": .array([
        .object([
          "median": .number(1499), "quantiles": .array((1000..<1999).map { .number(Decimal($0)) }),
        ])
      ]),
    ])
    let result = try TabularResults(response: raw, plan: plan)
    #expect(result.rows[0].low == 1099 && result.rows[0].high == 1899)
    #expect(result.csv(table: table, spec: spec) == "x,y,p10,p90\n1,2,,\n2,4,,\n3,1499,1099,1899\n")
  }
}
