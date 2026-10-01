import Darwin
import Foundation
import PaddockConversationCore
import Testing

@testable import PaddockClient
@testable import PaddockUI

/// Explicit opt-in, isolated profile and sockets. Starts only tiny test
/// checkpoints and stops only the children this test launched.
@Suite("Live native Tables relay", .serialized, .timeLimit(.minutes(3))) @MainActor
struct NativeTablesLiveTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_TABLES_LIVE"] == "1"))
  func classificationAndRegressionThroughTheActualNativeHost() async throws {
    let env = ProcessInfo.processInfo.environment
    let root = try #require(env["PADDOCK_DATA"])
    guard root.hasPrefix("/tmp/paddock-tables-live."), env["XDG_RUNTIME_DIR"] == root + "/runtime"
    else {
      Issue.record("Live Tables checks require an isolated profile and admin sockets.")
      return
    }
    let runner = try #require(env["PADDOCK_TABLES_RUNNER"])
    let fixtures = try #require(env["PADDOCK_TABLES_FIXTURES"])
    let library = URL(fileURLWithPath: try #require(env["PADDOCK_TABLES_LIBRARY"]))
    for (index, task) in [TabularTask.classification, .regression].enumerated() {
      let child = Process()
      child.executableURL = URL(fileURLWithPath: runner)
      child.arguments = [
        "--model", fixtures + "/small-\(task.rawValue)", "--device", "metal",
        "--host", "127.0.0.1", "--port", String(59341 + index), "--no-spec",
        "--vram-budget", "4096", "--served-model-name", "kumo-tabular-small-\(task.rawValue)",
      ]
      let logURL = URL(fileURLWithPath: root).appending(path: "\(task.rawValue).log")
      FileManager.default.createFile(atPath: logURL.path, contents: nil)
      let log = try FileHandle(forWritingTo: logURL)
      defer { try? log.close() }
      child.standardOutput = log
      child.standardError = log
      try child.run()
      let client = NativeManager(libraryURL: library)
      do {
        let model = NativeTablesModel(client: client)
        let deadline = ContinuousClock.now + .seconds(45)
        repeat {
          await model.refresh()
          if model.current?.limits.task == task { break }
          guard child.isRunning else {
            throw ConversationFailure.invalid("Test runner exited; inspect its isolated log.")
          }
          try await Task.sleep(for: .milliseconds(200))
        } while ContinuousClock.now < deadline
        #expect(model.error == nil, "\(model.error ?? "")")
        _ = try #require(model.current)
        let snapshot = try await client.snapshot()
        #expect(snapshot.runners.contains { $0.tabular == "kumo-tabular-small-\(task.rawValue)" })
        model.loadExample()
        await model.settle()
        #expect(model.canRun && model.plan?.contextRows.count == 120)
        model.run()
        await model.settle()
        #expect(model.error == nil, "\(model.error ?? "")")
        let result = try #require(model.result)
        #expect(result.response.rows.count == 6 && result.response.estimators == 8)
        #expect(result.response.gpuMilliseconds.map { $0 > 0 } == true)
        let exported = try TabularTable.parse(result.csv)
        #expect(exported.rows.count == 126)
        #expect(exported.rows.suffix(6).allSatisfy { !TabularTable.isMissing($0[6]) })
        let first = result.response.rows
        model.run()
        await model.settle()
        #expect(model.result?.response.rows == first)
        #expect(model.history.error == nil, "\(model.history.error ?? "")")
        let savedID = try #require(model.history.document?.id)
        #expect(model.history.document?.runs.count == 2)
        #expect(model.history.document?.datasets.count == 1)
        let originalCSV = try #require(model.result?.csv)
        #expect(await model.prepareForQuit())
        await client.close()
        await stop(child)
        // A fresh native core reads SQLite with no model running. Restoring
        // predictions must not require inference or substitute a chat model.
        let reopenedClient = NativeManager(libraryURL: library)
        let reopened = NativeTablesModel(client: reopenedClient)
        await reopened.history.refresh(api: reopened.api)
        #expect(reopened.history.sessions.contains { $0.id == savedID && $0.runs == 2 })
        await reopened.openSession(savedID)
        #expect(reopened.history.error == nil, "\(reopened.history.error ?? "")")
        #expect(reopened.result?.csv == originalCSV)
        #expect(reopened.history.document?.runs.count == 2 && !reopened.canRun)
        if let run = reopened.history.document?.runs.first {
          await reopened.selectRun(run)
          #expect(reopened.result?.response.rows == first)
        }
        await reopened.renameSession(savedID, title: "Reopened \(task.rawValue)")
        #expect(reopened.history.document?.title == "Reopened \(task.rawValue)")
        if let row = reopened.history.sessions.first(where: { $0.id == savedID }) {
          await reopened.removeSession(row)
          #expect(!reopened.history.sessions.contains { $0.id == savedID })
        }
        await reopenedClient.close()
      } catch {
        await client.close()
        await stop(child)
        throw error
      }
    }
  }
  private func stop(_ child: Process) async {
    guard child.isRunning else { return }
    child.terminate()
    for _ in 0..<100 {
      if !child.isRunning { return }
      try? await Task.sleep(for: .milliseconds(50))
    }
    if child.isRunning { kill(child.processIdentifier, SIGKILL) }
  }
}
