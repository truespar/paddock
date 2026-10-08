// Real native management ABI + bundled runner, without windows or user data.
// Compile alongside Sources/PaddockClient/*.swift into an isolated .app.
import Foundation

@main
struct EmbeddingSmoke {
  enum Failure: Error { case message(String) }
  static func main() async {
    do { try await run() } catch {
      FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
      exit(1)
    }
  }
  static func run() async throws {
    guard CommandLine.arguments.count == 3,
      let root = ProcessInfo.processInfo.environment["PADDOCK_DATA"],
      URL(fileURLWithPath: root).lastPathComponent.hasPrefix("paddock-macos-embedding."),
      let port = UInt16(CommandLine.arguments[2]), port >= 1024
    else {
      throw Failure.message(
        "Use an isolated paddock-macos-embedding.* root, library path and port.")
    }
    let client = NativeManager(libraryURL: URL(fileURLWithPath: CommandLine.arguments[1]))
    var owned: UInt32?
    do {
      let initial = try await client.snapshot()
      guard initial.runners.isEmpty, initial.servers?.isEmpty != false,
        let model = initial.catalog.models.first(where: { $0.id == "embeddinggemma-2" }),
        model.vendor == "Google",
        let artifact = model.artifacts.first(where: { $0.id == "mlx8" }), artifact.installed,
        artifact.runtime?.optionalTowers?.vision?.default == true,
        artifact.runtime?.optionalTowers?.audio?.default == false
      else { throw Failure.message("Catalog entry or isolated installed checkpoint is missing.") }
      let plan = try await client.downloads(.plan(model: model.id, artifact: artifact.id))
      guard plan.plan?.selection == ["mlx8"], plan.plan?.remaining == 0,
        plan.plan?.fileCount == 17
      else { throw Failure.message("MLX download plan included foreign or missing files.") }
      print("PASS: native catalog, Google provider, 17-file MLX-only download plan")
      var snapshot = try await settle(
        client,
        .create(
          CreateEndpointRequest(
            model: model.id, artifact: artifact.id, port: port, maxCtx: 8192, maxBatch: 1)))
      guard let first = snapshot.runners.first(where: { $0.port == port }) else {
        throw Failure.message("Runner did not start.")
      }
      owned = first.pid
      let expected = try await check(port, image: true, audio: false)
      for (image, audio) in [(false, false), (false, true), (true, true), (true, false)] {
        guard let revision = snapshot.servers?.first(where: { $0.port == port })?.revision else {
          throw Failure.message("Missing configuration revision.")
        }
        snapshot = try await settle(
          client,
          .edit(
            port: port, revision: revision, pid: owned,
            changes: [
              .composition(
                EndpointComposition(
                  model: model.id, artifact: artifact.id,
                  vision: image, drafter: nil, audio: audio))
            ], apply: .restart, allowNetwork: false))
        guard let runner = snapshot.runners.first(where: { $0.port == port }) else {
          throw Failure.message("Restarted runner disappeared.")
        }
        owned = runner.pid
        let got = try await check(port, image: image, audio: audio)
        guard got == expected else {
          throw Failure.message("Tower composition changed text embedding.")
        }
        let settings = snapshot.servers?.first(where: { $0.port == port })?.settings
        guard settings?.vision == image, settings?.audio == audio else {
          throw Failure.message("Saved settings do not describe the served towers.")
        }
        guard settings?.vramBudget == nil else {
          throw Failure.message("Automatic memory reservation became a manual ceiling.")
        }
        print("PASS: native edit/restart, image=\(image) audio=\(audio), exact text replay")
      }
      if let pid = owned { _ = try await settle(client, .stop(port: port, pid: pid)) }
      owned = nil
      await client.close()
      print("PASS: final stop; user data and models unchanged")
    } catch {
      if let pid = owned { _ = try? await settle(client, .stop(port: port, pid: pid)) }
      await client.close()
      throw error
    }
  }
  static func settle(_ client: NativeManager, _ command: ModelCommand) async throws
    -> ManagerSnapshot
  {
    let job = try await client.submit(command)
    let deadline = ContinuousClock.now.advanced(by: .seconds(120))
    while ContinuousClock.now < deadline {
      let snapshot = try await client.snapshot()
      if let state = snapshot.jobs?.first(where: { $0.id == job.id }), !state.isActive {
        guard state.state == "succeeded" else { throw Failure.message(state.message) }
        return snapshot
      }
      try await Task.sleep(for: .milliseconds(250))
    }
    throw Failure.message("Native management operation timed out.")
  }
  static func json(_ port: UInt16, _ path: String, body: [String: Any]? = nil) async throws
    -> [String: Any]
  {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/\(path)")!)
    request.timeoutInterval = 60
    if let body {
      request.httpMethod = "POST"
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.httpBody = try JSONSerialization.data(withJSONObject: body)
    }
    let (data, response) = try await URLSession.shared.data(for: request)
    guard (response as? HTTPURLResponse)?.statusCode == 200,
      let value = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { throw Failure.message("Embedding endpoint rejected \(path).") }
    return value
  }
  static func check(_ port: UInt16, image: Bool, audio: Bool) async throws -> [Double] {
    let server = try await json(port, "api/server")
    let inputs = Set(server["embedder_inputs"] as? [String] ?? [])
    let expected = Set(["text"] + (image ? ["image", "video"] : []) + (audio ? ["audio"] : []))
    guard inputs == expected else { throw Failure.message("Incorrect media discovery: \(inputs)") }
    let reply = try await json(
      port, "v1/embeddings",
      body: [
        "model": "embeddinggemma-2", "input": ["task: search result | query: Which planet is red?"],
        "dimensions": 768,
      ])
    guard let data = reply["data"] as? [[String: Any]], data.count == 1,
      let vector = data[0]["embedding"] as? [Double], vector.count == 768,
      vector.allSatisfy(\.isFinite), abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 0.0001
    else { throw Failure.message("Invalid embedding output.") }
    return vector
  }
}
