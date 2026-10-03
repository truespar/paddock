import Foundation
import Testing

@testable import PaddockClient

@Suite("Rust manager wire contracts")
struct ContractTests {
  @Test func diarizationKeepsItsNameWithoutOfferingAChatSurface() throws {
    let runner = try ManagerWire.decode(
      RunnerInfo.self,
      from: Data(
        #"{"port":11963,"pid":42,"status":"ok","diarization":"Nemotron-3-Diarization-MLX","endpoint":"http://127.0.0.1:11963"}"#
          .utf8))
    #expect(runner.title == "Nemotron-3-Diarization-MLX")
    #expect(runner.diarization != nil)
    #expect(!runner.hasStudioSurface)
  }
  @Test func automaticPortsAreOmittedButFixedAddressesRemainExplicit() throws {
    for port: UInt16? in [nil, 12345] {
      let request = CreateEndpointRequest(model: "kb-whisper-large", artifact: "f16", port: port)
      let data = try JSONEncoder().encode(ModelCommand.create(request))
      let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
      #expect((object["port"] as? NSNumber)?.uint16Value == port)
      if port == nil { #expect(!object.keys.contains("port")) }
    }
    let pending = try ManagerWire.decode(
      ManagementJob.self,
      from: Data(
        #"{"id":1,"port":null,"action":"create","state":"running","message":"Starting"}"#.utf8))
    #expect(pending.port == nil && pending.isActive)
  }

  @Test func modelSpecAndDefaultMetadataSurvivesDecoding() throws {
    let model = try ManagerWire.decode(
      CatalogModel.self,
      from: Data(
        #"""
        {"id":"model","display":"Model","capability":["chat"],"installed":false,"total_size":42,
         "specs":{"about":"A model","params":"27B","context":"256K","context_max":"1M",
           "published_at":"2026-08-25","published_source":"https://example.com/announcement",
           "homepage":"https://example.com/model","strengths":["Text"],"tradeoffs":["Size"]},
         "artifacts":[{"id":"q4","kind":"weights","format":"gguf","label":"4-bit",
           "installed":false,"total_size":42,"default":true,"runtime":{"companions":["vision"]}}]}
        """#.utf8))
    #expect(model.specs?.about == "A model")
    #expect(model.specs?.publishedAt == "2026-08-25")
    #expect(model.specs?.publishedSource == "https://example.com/announcement")
    #expect(model.specs?.contextMax == "1M")
    #expect(model.specs?.strengths == ["Text"])
    #expect(model.specs?.tradeoffs == ["Size"])
    #expect(model.artifacts.first?.default == true)
    #expect(model.artifacts.first?.runtime?.companions == ["vision"])
  }

  @Test func preservesNullsUnknownFieldsAndModelRoles() throws {
    let readiness = try ManagerWire.decode(
      Readiness.self,
      from: Data(
        #"{"backend":"metal","state":"untested","os":"macos","card":null,"new_field":true}"#.utf8))
    #expect(readiness.card == nil)
    #expect(readiness.warning == nil)
    let runner = try ManagerWire.decode(
      RunnerInfo.self,
      from: Data(
        #"{"port":11540,"pid":42,"status":"starting","model":null,"embedder":null,"aligner":"align-model","endpoint":"http://127.0.0.1:11540","in_flight":null}"#
          .utf8))
    #expect(runner.title == "align-model")
    #expect(runner.inFlight == nil)
    #expect(runner.id == "11540:42")
  }

  @Test func hardwareOverviewOnlyReportsActionableFailures() throws {
    for (state, warning): (String, String?) in [
      ("ready", nil), ("untested", nil), ("future-state", nil),
      ("driver-too-old", "Driver update needed"), ("no-card", "Local serving unavailable"),
    ] {
      let readiness = try ManagerWire.decode(
        Readiness.self,
        from: Data(
          "{\"backend\":\"metal\",\"state\":\"\(state)\",\"os\":\"macos\"}".utf8))
      #expect(readiness.warning == warning)
    }
  }

  @Test func artifactAvailabilityUsesBackendNotLegacyQualification() throws {
    let old = try artifact(runtime: "null")
    #expect(!old.supports(backend: "metal"))
    #expect(!old.supports(backend: nil))
    #expect(old.supports(backend: "cuda"))
    #expect(old.supportNotice == nil)
    let current = try artifact(
      runtime:
        #"{"backends":["metal"],"qualification":"unqualified","embedded_vision":true,"capability":[]}"#
    )
    #expect(current.supports(backend: "metal"))
    #expect(!current.supports(backend: "cuda"))
    #expect(current.supportNotice == nil)
    #expect(current.runtime?.qualification == "unqualified")
    #expect(current.runtime?.embeddedVision == true)
    #expect(current.runtime?.capability == [])
    let empty = try artifact(runtime: #"{"backends":[]}"#)
    #expect(!empty.supports(backend: "cuda"))
  }

  @Test func serverProjectionWinsOverLegacyFallback() throws {
    let value = Data(
      #"{"id":"w","kind":"weights","format":"gguf","label":"Weights","total_size":4,"installed":true,"backend_supported":false,"runtime":{"backends":["metal"]}}"#
        .utf8)
    let artifact = try ManagerWire.decode(CatalogArtifact.self, from: value)
    #expect(!artifact.supports(backend: "metal"))
  }

  @Test func unknownQualificationAndLargeByteCountsAreNotLost() throws {
    let value = try artifact(runtime: #"{"qualification":"future-review"}"#)
    #expect(value.supportNotice == nil)
    #expect(value.runtime?.qualification == "future-review")
    #expect(value.totalSize == 111_546_655_083)
  }

  @Test func productCopyDoesNotExposeInternalReleaseLabels() throws {
    let current = try artifact(runtime: #"{"qualification":"experimental","backends":["metal"]}"#)
    #expect(current.supportNotice == nil)
    #expect(current.runtime?.qualification == "experimental")
    #expect(current.supports(backend: "metal"))
    #expect(try artifact(runtime: #"{"experimental":true}"#).supportNotice == nil)
    for (label, expected) in [
      ("MLX 4-bit (Metal preview)", "MLX 4-bit"),
      ("Vision (macOS preview)", "Vision"),
      ("Upstream Preview Model", "Upstream Preview Model"),
    ] {
      let value = try ManagerWire.decode(
        CatalogArtifact.self,
        from: Data(
          """
          {"id":"w","kind":"weights","format":"gguf","label":"\(label)","installed":true,"total_size":4}
          """.utf8))
      #expect(value.label == label)
      #expect(value.displayLabel == expected)
    }
  }

  private func artifact(runtime: String) throws -> CatalogArtifact {
    let json = """
      {"id":"mlx-4bit","kind":"weights","format":"safetensors","label":"4-bit",
      "quant":"MLX-AFFINE-4-G64","installed":false,"total_size":111546655083,"runtime":\(runtime)}
      """
    return try ManagerWire.decode(CatalogArtifact.self, from: Data(json.utf8))
  }
}
