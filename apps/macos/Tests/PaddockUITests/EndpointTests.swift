import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Native endpoint workflow", .timeLimit(.minutes(1))) @MainActor
struct EndpointTests {
  @Test func automaticPendingStartDoesNotInventAnEndpointRow() throws {
    let snapshot = try endpointFixture()
    let pending = try ManagerWire.decode(
      ManagementJob.self,
      from: Data(
        #"{"id":2,"port":null,"action":"create","state":"running","message":"Starting"}"#.utf8))
    let rows = EndpointRow.rows(snapshot: snapshot, latestJob: pending)
    #expect(rows.map(\.port) == [12345])
    #expect(rows[0].status == "Running")
  }

  @Test func samePortAppearsOnceAcrossAllProjections() throws {
    let snapshot = try endpointFixture()
    let job = try commandJob()
    let rows = EndpointRow.rows(snapshot: snapshot, latestJob: job)
    #expect(rows.count == 1)
    #expect(rows[0].port == 12345)
    #expect(rows[0].status == "Starting")
    #expect(rows[0].baseURL == "http://127.0.0.1:12345/v1")
    #expect(rows[0].runner?.pid == 42)
    #expect(rows[0].configured?.revision == "revision")
    #expect(EndpointRow.rows(snapshot: snapshot, latestJob: nil)[0].status == "Running")
  }

  @Test func repeatedSubmitIsRejectedAndReceiptIsRetained() async throws {
    let client = CommandLoader()
    let model = WorkspaceModel(client: client)
    await model.start()
    let first = Task { await model.submit(.stop(port: 12345, pid: 42)) }
    await client.waitForSubmit()
    #expect(model.isSubmitting)
    #expect(!model.canSubmit)
    #expect(await model.submit(.stop(port: 12345, pid: 42)) == false)
    await client.complete(.success(try commandJob()))
    #expect(await first.value)
    #expect(model.latestJob?.id == 1)
    #expect(model.operationInProgress)
    #expect(await client.submissions == 1)
    await model.shutdown()
  }

  @Test func shutdownDiscardsLateReceipt() async throws {
    let client = CommandLoader()
    let model = WorkspaceModel(client: client)
    await model.start()
    let operation = Task { await model.submit(.stop(port: 12345, pid: 42)) }
    await client.waitForSubmit()
    await model.shutdown()
    await client.complete(.success(try commandJob()))
    #expect(await operation.value == false)
    #expect(model.state == .stopped)
    #expect(model.latestJob == nil)
  }

  @Test func submissionErrorIsVisibleAndKeepsInventory() async throws {
    let client = CommandLoader()
    let model = WorkspaceModel(client: client)
    await model.start()
    let operation = Task { await model.submit(.stop(port: 12345, pid: 42)) }
    await client.waitForSubmit()
    await client.complete(.failure(ManagerError.core("Port changed")))
    #expect(await operation.value == false)
    #expect(model.commandError == "Port changed")
    #expect(model.snapshot != nil)
    #expect(model.canSubmit)
    await model.shutdown()
  }

  @Test func pickerHonorsArtifactRatherThanFamilyCapability() throws {
    let snapshot = try endpointFixture()
    let model = try #require(snapshot.catalog.models.first)
    let artifacts = StartModelView.artifacts(model, snapshot: snapshot)
    #expect(artifacts.map(\.id) == ["mlx-4bit"])
    #expect(artifacts[0].runtime?.capability == ["chat"])
    #expect(model.capability == ["chat", "vision"])
    #expect(model.hasInstalledWeights(on: snapshot.readiness.backend))
    #expect(model.preferredWeights(on: snapshot.readiness.backend)?.id == "mlx-4bit")

    let cudaOnly = try ManagerWire.decode(
      CatalogModel.self,
      from: Data(
        #"{"id":"cached","display":"Cached model","capability":["chat"],"installed":true,"total_size":1000,"artifacts":[{"id":"cuda","kind":"weights","format":"safetensors","label":"CUDA","installed":true,"total_size":1000,"backend_supported":false},{"id":"metal","kind":"weights","format":"gguf","label":"Metal","installed":false,"total_size":1000,"backend_supported":true}]}"#
          .utf8))
    #expect(!cudaOnly.hasInstalledWeights(on: "metal"))
    #expect(cudaOnly.preferredWeights(on: "metal")?.id == "metal")
  }

  @Test func quantizationLabelsPreserveUnknownFormats() throws {
    func artifact(_ quant: String) throws -> CatalogArtifact {
      try ManagerWire.decode(
        CatalogArtifact.self,
        from: Data(
          """
          {"id":"weights","kind":"weights","format":"safetensors","label":"Weights",\
          "installed":true,"total_size":1000,"quant":"\(quant)"}
          """.utf8))
    }
    #expect(try artifact("MLX-affine-4-g64").shortFormat == "MLX · 4-bit")
    #expect(try artifact("MLX-AFFINE-4-G64").shortFormat == "MLX · 4-bit")
    #expect(try artifact("MLX-TERNARY-2-G128").shortFormat == "MLX · Ternary 2-bit")
    #expect(try artifact("affine-4/8").shortFormat == "MLX · Mixed 4/8-bit")
    #expect(try artifact("MLX-future").shortFormat == "SAFETENSORS · MLX-future")
    #expect(try artifact("MLX-affine-mixed").shortFormat == "SAFETENSORS · MLX-affine-mixed")
  }
}

private actor CommandLoader: ManagerLoading {
  private var continuation: CheckedContinuation<ManagementJob, any Error>?
  private(set) var submissions = 0
  func snapshot() async throws -> ManagerSnapshot { try endpointFixture() }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    submissions += 1
    return try await withCheckedThrowingContinuation { continuation = $0 }
  }
  func waitForSubmit() async {
    while continuation == nil { await Task.yield() }
  }
  func complete(_ result: Result<ManagementJob, any Error>) {
    continuation?.resume(with: result)
    continuation = nil
  }
}

func commandJob() throws -> ManagementJob {
  try ManagerWire.decode(
    ManagementJob.self,
    from: Data(
      #"{"id":1,"port":12345,"action":"create","state":"running","message":"Loading model"}"#.utf8))
}

func endpointFixture() throws -> ManagerSnapshot {
  let base = try fixture()
  return try ManagerSnapshot(
    identity: base.identity, readiness: base.readiness,
    catalog: ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"qwen3.8-27b","display":"Qwen 3.8 27B","vendor":"Alibaba","capability":["chat","vision"],"installed":true,"total_size":15000000000,"artifacts":[{"id":"mlx-4bit","kind":"weights","format":"safetensors","label":"MLX-community 4-bit","installed":true,"total_size":15000000000,"backend_supported":true,"runtime":{"experimental":true,"capability":["chat"]}},{"id":"cuda","kind":"weights","format":"safetensors","label":"CUDA","installed":true,"total_size":15000000000,"backend_supported":false}]}]}"#
          .utf8)),
    runners: ManagerWire.decode(
      [RunnerInfo].self,
      from: Data(
        #"[{"port":12345,"pid":42,"status":"ok","model":"qwen3.8-27b","display":"Qwen 3.8 27B","endpoint":"http://127.0.0.1:12345"}]"#
          .utf8)),
    servers: ManagerWire.decode(
      [ConfiguredEndpoint].self,
      from: Data(
        #"[{"port":12345,"model":"qwen3.8-27b","artifact":"mlx-4bit","display":"Qwen 3.8 27B","running":true,"revision":"revision","local_only":true,"max_ctx":4096,"max_batch":4}]"#
          .utf8)))
}
