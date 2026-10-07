import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Native library artifact discovery")
struct LibraryCatalogTests {
  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Run apps/macos/scripts/check.sh for the actual Rust registry projection."))
  @MainActor func publishedKolibriReachesNativeCatalogWithMakerAndMetalDefaults() throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"])
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(contentsOf: URL(fileURLWithPath: path).appending(path: "catalog.json")))
    for query in ["Kolibri", "Aleph Alpha", "eins78/Kolibri-1-mlx-mixed-4-8-bit"] {
      let entries = LibraryCatalog.entries(
        catalog: catalog, backend: "metal", format: .mlx, query: query)
      let entry = try #require(entries.first { $0.id == "kolibri-1" })
      #expect(entry.model.vendor == "Aleph Alpha")
      #expect(entry.publishedAt == "2026-10-03")
      #expect(entry.artifacts.map(\.id) == ["mlx-mixed-4-8bit"])
      #expect(entry.initialArtifact == "mlx-mixed-4-8bit")
      #expect(entry.capabilities == ["chat", "reasoning", "tools"])
      let artifact = try #require(entry.preferredArtifact)
      #expect(artifact.runtime?.backends == ["metal"])
      #expect(artifact.runtime?.defaultMaxCtx == 32768)
      #expect(artifact.runtime?.defaultMaxBatch == 1)
      #expect(artifact.runtime?.kvCacheDtype == "auto")
      #expect(artifact.runtime?.embeddedVision == false)
      #expect(ProviderArtwork.names[entry.model.vendor ?? ""] == "AlephAlpha")
    }
    // This real API fixture is already projected for Metal. Its authoritative
    // backendSupported field must not be reused to simulate a CUDA server.
    #expect(
      LibraryCatalog.entries(catalog: catalog, backend: "metal", format: .gguf, query: "Kolibri")
        .isEmpty)
    let entries = LibraryCatalog.entries(catalog: catalog, backend: "metal")
    let kolibri = try #require(entries.firstIndex { $0.id == "kolibri-1" })
    let bonsai = try #require(entries.firstIndex { $0.id == "bonsai-2-27b" })
    #expect(kolibri < bonsai)
  }

  @Test func publicationOrderIsNewestFirstWithStableUndatedFallback() throws {
    let catalog = try orderedCatalog()
    func ids(_ order: LibraryOrder = .newest) -> [String] {
      LibraryCatalog.entries(catalog: catalog, backend: "metal", order: order).map(\.id)
    }
    #expect(ids() == ["new-a", "new-b", "old", "invalid", "unknown"])
    #expect(ids(.oldest) == ["old", "new-a", "new-b", "invalid", "unknown"])
    #expect(ids(.nameAscending) == ["invalid", "new-a", "new-b", "unknown", "old"])
    #expect(ids(.nameDescending) == ["old", "unknown", "new-b", "new-a", "invalid"])
    #expect(ids(.smallest) == ["old", "invalid", "new-a", "unknown", "new-b"])
    #expect(ids(.largest) == ["new-b", "unknown", "new-a", "invalid", "old"])
    let ascending = LibraryCatalog.entries(catalog: catalog, backend: "metal", order: .oldest)
    let current = LibrarySelection(model: "new-b", artifact: "weights")
    #expect(LibraryCatalog.selection(current, in: ascending, backend: "metal") == current)
    #expect(
      LibraryCatalog.entries(catalog: catalog, backend: "metal", query: "Same").map(\.id) == [
        "new-a", "new-b",
      ])
  }

  @Test func datesNeverUseRevisionsOrNormalizeMalformedCalendarDays() throws {
    for (value, valid) in [
      ("2024-02-29", true), ("2026-02-29", false), ("2026-04-31", false),
      ("2026-00-01", false), ("2026-13-01", false), ("2026-01-00", false),
      ("2026-1-01", false), ("0000-01-01", false), ("2026-08-25T12:00:00Z", false),
    ] {
      let model = try ManagerWire.decode(
        CatalogModel.self,
        from: Data(
          """
          {"id":"date","display":"Date","capability":[],"installed":false,"total_size":0,
           "artifacts":[],"specs":{"published_at":"\(value)"},"revision":"2099-12-31"}
          """.utf8))
      #expect((LibraryCatalog.publicationDate(model) != nil) == valid, "\(value)")
    }
    let undated = try #require(orderedCatalog().models.first { $0.id == "unknown" })
    #expect(LibraryCatalog.publicationDate(undated) == nil)
  }

  private func orderedCatalog() throws -> ModelCatalog {
    let rows: [(String, String, String?, Int)] = [
      ("unknown", "Unknown", nil, 40), ("new-b", "Same 10B", "2026-08-25", 50),
      ("old", "Zed", "2025-06-05", 10), ("invalid", "Invalid", "2026-02-30", 20),
      ("new-a", "Same", "2026-08-25", 30),
    ]
    let models: [[String: Any]] = rows.map { id, display, date, size in
      [
        "id": id, "display": display, "capability": ["chat"], "installed": false,
        "total_size": size, "revision": "2099-12-31",
        "specs": ["published_at": date as Any? ?? NSNull()],
        "artifacts": [
          [
            "id": "weights", "label": "Weights", "kind": "weights", "format": "gguf",
            "installed": false, "total_size": size, "runtime": ["backends": ["metal"]],
          ]
        ],
      ]
    }
    return try ManagerWire.decode(
      ModelCatalog.self,
      from: JSONSerialization.data(withJSONObject: ["schema": 3, "models": models]))
  }

  @Test func alternativesAreNotHiddenByDefaultOrQualification() throws {
    let catalog = try mixedCatalog()
    let all = LibraryCatalog.entries(catalog: catalog, backend: "metal")
    #expect(all.count == 1)
    #expect(all[0].artifacts.map(\.id) == ["q8", "mlx-4bit"])
    #expect(all[0].initialArtifact == nil)
    #expect(all[0].installed)

    let mlx = LibraryCatalog.entries(catalog: catalog, backend: "metal", format: .mlx)
    #expect(mlx.count == 1)
    #expect(mlx[0].initialArtifact == "mlx-4bit")
    #expect(mlx[0].artifacts[0].supportNotice == nil)
    #expect(!mlx[0].installed, "A downloaded GGUF must not mark the MLX export downloaded")
    #expect(
      LibraryCatalog.entries(
        catalog: catalog, backend: "metal", format: .mlx, downloadedOnly: true
      ).isEmpty)
    #expect(LibraryCatalog.entries(catalog: catalog, backend: "cuda", format: .mlx).isEmpty)
  }

  @Test func searchFindsExportMetadataAndPreservesExactChoice() throws {
    let catalog = try mixedCatalog()
    for query in ["MLX", "mlx-community/example", "MLX-AFFINE-4-G64"] {
      let found = LibraryCatalog.entries(catalog: catalog, backend: "metal", query: query)
      #expect(found.count == 1)
      #expect(found.first?.initialArtifact == "mlx-4bit")
    }
    let gguf = LibraryCatalog.entries(catalog: catalog, backend: "metal", format: .gguf)
    #expect(gguf.first?.initialArtifact == "q8")
    #expect(gguf.first?.installed == true)
  }

  @Test func capabilityIconsFollowFilteredExportOverrides() throws {
    let catalog = try mixedCatalog()
    let all = try #require(LibraryCatalog.entries(catalog: catalog, backend: "metal").first)
    let mlx = try #require(
      LibraryCatalog.entries(catalog: catalog, backend: "metal", format: .mlx).first)
    #expect(all.capabilities.contains("vision"))
    #expect(!mlx.capabilities.contains("vision"))
    #expect(mlx.capabilities == ["chat"])
    #expect(ModelFeature.vision.description(in: mlx.capabilities) == "Vision: not listed")
    #expect(ModelFeature.allCases.map(\.rawValue) == ["vision", "reasoning", "tools"])
  }

  @Test func browserSelectionTracksFiltersButRetainsExplicitOptions() throws {
    let catalog = try mixedCatalog()
    let all = LibraryCatalog.entries(catalog: catalog, backend: "metal")
    let mlx = LibraryCatalog.entries(catalog: catalog, backend: "metal", format: .mlx)
    let first = LibraryCatalog.selection(nil, in: all, backend: "metal")
    #expect(first == LibrarySelection(model: "example", artifact: "q8"))
    let filtered = LibraryCatalog.selection(
      first, in: mlx, backend: "metal", requireMatchingExport: true)
    #expect(filtered == LibrarySelection(model: "example", artifact: "mlx-4bit"))
    #expect(LibraryCatalog.selection(filtered, in: all, backend: "metal") == filtered)
    // Explicitly choosing GGUF in the detail pane is allowed even with the list
    // filtered to models offering MLX. A subsequent filter change reconciles it.
    #expect(LibraryCatalog.selection(first, in: mlx, backend: "metal") == first)
    #expect(LibraryCatalog.selection(first, in: [], backend: "metal") == nil)
    let stale = LibrarySelection(model: "removed", artifact: "missing")
    #expect(LibraryCatalog.selection(stale, in: all, backend: "metal") == first)
    let goneExport = LibrarySelection(model: "example", artifact: "fp8")
    #expect(LibraryCatalog.selection(goneExport, in: all, backend: "metal") == first)
  }

  @Test func recommendationsNeverUpgradeUnqualifiedOrUnknownDefaults() throws {
    for (qualification, label) in [
      ("qualified", "Recommended"), ("experimental", "Catalog default"),
      ("unqualified", "Catalog default"), ("future-status", "Catalog default"),
    ] {
      let artifact = try ManagerWire.decode(
        CatalogArtifact.self,
        from: Data(
          """
          {"id":"w","kind":"weights","format":"gguf","label":"Weights","default":true,
           "installed":false,"total_size":10,"runtime":{"backends":["metal"],"qualification":"\(qualification)"}}
          """.utf8))
      #expect(LibraryRecommendation.label(for: artifact) == label)
    }
    let model = try #require(mixedCatalog().models.first)
    #expect(LibraryRecommendation.label(for: model.artifacts[1]) == nil)
    #expect(LibraryRecommendation.label(for: model.artifacts[0]) == "Catalog default")
    #expect(
      LibraryRecommendation.explanation(model: model, backend: "metal").contains(
        "Compare the download sizes and capabilities"))
  }

  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Run apps/macos/scripts/check.sh for the actual Rust registry projection."))
  func publishedBonsaiIsDiscoverableWithNativeVision() throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"])
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(contentsOf: URL(fileURLWithPath: path).appending(path: "catalog.json")))
    for query in ["Bonsai", "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", "MLX-TERNARY-2-G128"] {
      let entries = LibraryCatalog.entries(
        catalog: catalog, backend: "metal", format: .mlx, query: query)
      let entry = try #require(entries.first { $0.id == "bonsai-2-27b" })
      #expect(entry.model.vendor == "Prism ML")
      #expect(entry.publishedAt == "2026-09-17")
      #expect(entry.model.specs?.publishedSource == "https://prismml.com/news/bonsai-2-27b")
      #expect(entry.artifacts.map(\.id) == ["mlx-2bit"])
      #expect(entry.initialArtifact == "mlx-2bit")
      #expect(entry.capabilities == ["chat", "vision", "reasoning", "tools"])
      #expect(!entry.installed)
      let artifact = try #require(entry.preferredArtifact)
      #expect(artifact.supports(backend: "metal"))
      #expect(artifact.runtime?.backends == ["metal"])
      #expect(artifact.runtime?.embeddedVision == true)
      #expect(artifact.runtime?.kvCacheDtype == "f32")
      #expect(artifact.runtime?.defaultMaxCtx == 32768)
      #expect(
        LibraryCatalog.selection(nil, in: entries, backend: "metal")
          == LibrarySelection(model: "bonsai-2-27b", artifact: "mlx-2bit"))
    }
    for format in [LibraryFormat.all, .mlx] {
      let entries = LibraryCatalog.entries(catalog: catalog, backend: "metal", format: format)
      let bonsaiIndex = try #require(entries.firstIndex { $0.id == "bonsai-2-27b" })
      let qwenIndex = try #require(entries.firstIndex { $0.id == "qwen3.8-flash-next" })
      #expect(bonsaiIndex < qwenIndex, "Newest order uses publication, not the catalog revision")
    }
    let gguf = LibraryCatalog.entries(
      catalog: catalog, backend: "metal", format: .gguf, query: "Bonsai")
    let entry = try #require(gguf.first { $0.id == "bonsai-2-27b" })
    #expect(entry.artifacts.map(\.id) == ["ptq1"])
    #expect(entry.capabilities == ["chat", "vision", "reasoning", "tools"])
    let artifact = try #require(entry.preferredArtifact)
    #expect(artifact.runtime?.kvCacheDtype == "f16")
    #expect(artifact.runtime?.defaultMaxCtx == 32768)
    #expect(artifact.runtime?.companions == ["vision"])
    let vision = try #require(entry.model.artifacts.first { $0.id == "vision" })
    #expect(vision.supports(backend: "metal") && vision.required != true)
    let all = LibraryCatalog.entries(
      catalog: catalog, backend: "metal", format: .all, query: "Bonsai")
    #expect(all.first?.preferredArtifact?.id == "mlx-2bit")
    #expect(Set(all.first?.artifacts.map(\.id) ?? []) == ["ptq1", "mlx-2bit"])
  }

  @Test func modelLinksOnlyOpenOrdinaryHTTPSPages() {
    #expect(LibraryCatalog.webURL("https://huggingface.co/Qwen/Qwen3.8-27B") != nil)
    for invalid in [
      nil, "file:///tmp/model", "javascript:alert(1)", "https://", "https://user:pass@example.com",
    ] {
      #expect(LibraryCatalog.webURL(invalid) == nil)
    }
  }

  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Run apps/macos/scripts/check.sh for the actual Rust registry projection."))
  func publishedMLXModelsSurviveRealRegistryAndNativeFilters() throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"])
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(contentsOf: URL(fileURLWithPath: path).appending(path: "catalog.json")))
    let expected: Set<String> = [
      "qwen3.8-27b", "qwen3.8-flash-next", "gemma-4-31b", "muse-glimmer-30b",
    ]
    let entries = LibraryCatalog.entries(catalog: catalog, backend: "metal", format: .mlx)
    #expect(expected.isSubset(of: Set(entries.map(\.id))))
    for entry in entries where expected.contains(entry.id) {
      #expect(entry.artifacts.map(\.id) == ["mlx-4bit"])
      #expect(entry.artifacts[0].source?.repo.hasPrefix("mlx-community/") == true)
      #expect(!entry.installed, "Catalog publication is independent of a local download")
      #expect(entry.artifacts[0].supports(backend: "metal"))
      #expect(entry.artifacts[0].supportNotice == nil)
      #expect(entry.model.specs?.about?.isEmpty == false)
      #expect(entry.model.specs?.params?.isEmpty == false)
      #expect(LibraryCatalog.webURL(entry.model.specs?.homepage) != nil)
    }
    let qwen = try #require(entries.first { $0.id == "qwen3.8-27b" })
    #expect(qwen.preferredArtifact?.runtime?.defaultMaxCtx == 32768)
    #expect(qwen.capabilities.contains("tools"))
    #expect(qwen.capabilities.contains("reasoning"))
    #expect(!qwen.capabilities.contains("vision"), "Qwen MLX must not inherit GGUF vision")
    #expect(
      entries.filter { expected.contains($0.id) }.map(\.id) == [
        "qwen3.8-flash-next", "qwen3.8-27b", "muse-glimmer-30b", "gemma-4-31b",
      ])
    #expect(entries.first { expected.contains($0.id) }?.publishedAt == "2026-08-26")
  }

  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Requires the actual Rust registry projection."))
  func publishedSplashIsSearchableAsItsOwnMetalExport() throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"])
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(contentsOf: URL(fileURLWithPath: path).appending(path: "catalog.json")))
    for query in ["Splash", "incoai/Qwen3.8-27B-Splash", "SPLASH-Q4-G64"] {
      let entries = LibraryCatalog.entries(catalog: catalog, backend: "metal", query: query)
      let entry = try #require(entries.first)
      #expect(entries.count == 1 && entry.id == "qwen3.8-27b")
      #expect(entry.initialArtifact == "splash-4bit")
      #expect(
        entry.capabilities.isSuperset(of: ["chat", "vision", "tools", "reasoning", "speculative"]))
      let artifact = try #require(entry.artifacts.first)
      #expect(artifact.supportNotice == nil && !artifact.isMLX)
      #expect(artifact.runtime?.embeddedVision == true)
      #expect(artifact.runtime?.defaultMaxBatch == 1)
      #expect(artifact.runtime?.backends == ["metal"])
      #expect(
        LibraryCatalog.selection(nil, in: entries, backend: "metal")
          == LibrarySelection(model: "qwen3.8-27b", artifact: "splash-4bit"))
    }
    #expect(
      LibraryCatalog.entries(catalog: catalog, backend: "metal", format: .mlx, query: "Splash")
        .isEmpty)
  }

  private func mixedCatalog() throws -> ModelCatalog {
    try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"""
        {"schema":3,"models":[{
          "id":"example","display":"Example","capability":["chat","vision"],
          "installed":true,"total_size":100,"artifacts":[
            {"id":"q8","kind":"weights","format":"gguf","label":"Full quality",
             "quant":"Q8_0","default":true,"installed":true,"total_size":100,
             "runtime":{"backends":["metal","cuda"]}},
            {"id":"mlx-4bit","kind":"weights","format":"safetensors","label":"MLX 4-bit",
             "quant":"MLX-AFFINE-4-G64","installed":false,"total_size":50,
             "runtime":{"backends":["metal"],"qualification":"unqualified","capability":["chat"]},
             "source":{"repo":"mlx-community/example","revision":"pinned"}},
            {"id":"fp8","kind":"weights","format":"safetensors","label":"Native FP8",
             "quant":"FP8","installed":true,"total_size":90,"runtime":{"backends":["cuda"]}}
          ]
        }]}
        """#.utf8))
  }
}
