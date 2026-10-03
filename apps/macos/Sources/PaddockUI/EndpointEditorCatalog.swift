import Foundation
import PaddockClient

extension EndpointEditor {
  static let contextSteps = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1_048_576,
  ]
  static let workloads: [(label: String, batch: Int)] = [
    ("Just me", 1), ("Coding agents", 4), ("A team / an app", 16),
  ]
  static let speculationChoices = [("on", "On"), ("off", "Off"), ("adaptive", "Adaptive")]

  static func specPolicy(_ value: String?) -> String {
    switch value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case nil, "", "on", "true", "yes", "ladder", "legacy": "on"
    case "auto", "adaptive": "adaptive"
    case "off", "false", "no", "none", "0": "off"
    default: value ?? "on"
    }
  }
  static func budgetText(_ value: Int?) -> String {
    guard let value else { return "" }
    let gb = Double(value) / 1024
    return gb == gb.rounded() ? String(Int(gb)) : String(gb)
  }
  var memoryBudgetMiB: Int? {
    let input = memoryLimit.replacingOccurrences(
      of: Locale.current.decimalSeparator ?? ".", with: ".")
    guard let gb = Double(input), gb.isFinite, (0.25...1024).contains(gb) else { return nil }
    return Int((gb * 1024).rounded())
  }
  func setCustomMemoryBudget(_ enabled: Bool) {
    guard enabled != customMemoryBudget else { return }
    customMemoryBudget = enabled
    memoryLimit = enabled ? Self.budgetText(endpoint.settings?.vramBudget) : ""
  }
  var memoryValidation: String? {
    guard customMemoryBudget || !memoryLimit.isEmpty else { return nil }
    guard let mib = memoryBudgetMiB else {
      return "Enter a numeric limit between 0.25 and 1024 GiB."
    }
    // Keep an older saved setting inspectable; never silently clamp a draft.
    if memoryLimit != Self.budgetText(endpoint.settings?.vramBudget),
      let cap = memoryHardware.recommendedBytes, UInt64(mib) << 20 > cap
    {
      return
        "Apple's recommended GPU maximum is \(MetalMemoryHardware.gib(cap)). Choose a lower limit or Automatic."
    }
    return nil
  }
  func generateKey() {
    let alphabet = Array("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789")
    var generator = SystemRandomNumberGenerator()
    replacementKey =
      "pd-" + String((0..<32).map { _ in alphabet.randomElement(using: &generator)! })
  }
  var selectedModel: CatalogModel? { catalog.first { $0.id == modelID } }
  var savedSpeculation: String {
    endpoint.settings?.noSpec == true ? "off" : Self.specPolicy(endpoint.settings?.spec)
  }
  var dirtyCompositionOrSpec: Bool { compositionChanged || speculation != savedSpeculation }
  var runtimeOptions: [EndpointRuntimeOption] { endpoint.settings?.runtimeOptions ?? [] }
  var visibleRuntimeOptions: [EndpointRuntimeOption] {
    runtimeOptions.filter { field in
      field.capability.isEmpty || capabilities.contains(field.capability)
        || field.capability == "vision" && visionServed
        || field.value != nil || !(runtimeDraft[field.id] ?? "").isEmpty
    }
  }
  var runtimeGroups: [String] {
    visibleRuntimeOptions.reduce(into: []) { groups, field in
      if !groups.contains(field.group) { groups.append(field.group) }
    }
  }
  var kvChoices: [String] {
    if let required = selectedArtifact?.runtime?.kvCacheDtype {
      return required == "f16" ? ["", "auto", "f16"] : ["", required]
    }
    return isMLX ? ["", "auto"] : ["", "auto", "f16"]
  }
  var isMLX: Bool {
    selectedArtifact?.format == "safetensors" && selectedArtifact?.runtime?.checkpointDir != false
      && !capabilities.contains("decision")
      && !capabilities.contains("tabular")
  }
  var isSplash: Bool { selectedArtifact?.format == "splash-packed-q4" }
  var checkpointLabel: String {
    // A decision/table route does not imply a precision. Quantized Clef uses
    // the same artifact label as Catalog, without enabling chat KV controls.
    if capabilities.contains("decision") || capabilities.contains("tabular"),
      let artifact = selectedArtifact, artifact.quant != nil
    {
      return artifact.shortFormat
    }
    if capabilities.contains("tabular") { return "F32 checkpoint" }
    if capabilities.contains("decision") { return "F16 checkpoint" }
    return isSplash
      ? "Splash packed Q4"
      : isMLX ? "MLX checkpoint" : selectedArtifact?.format.uppercased() ?? "Native checkpoint"
  }
  var kvLabel: String {
    if selectedArtifact?.runtime?.kvCacheDtype == "f32" { return "F32 · checkpoint native" }
    return isMLX || isSplash ? "BF16 · checkpoint native" : "F16 · 16-bit"
  }
  var selectedArtifact: CatalogArtifact? { selectedModel?.artifacts.first { $0.id == artifactID } }
  var title: String { selectedModel?.display ?? endpoint.title }
  var modelChoices: [CatalogModel] {
    catalog.filter { model in
      model.id == modelID || purpose.weights(model, backend: "metal").contains { $0.installed }
    }.sorted { $0.display.localizedStandardCompare($1.display) == .orderedAscending }
  }
  var weights: [CatalogArtifact] {
    selectedModel.map { purpose.weights($0, backend: "metal") } ?? []
  }
  var capabilities: [String] {
    selectedArtifact?.runtime?.capability ?? selectedModel?.capability ?? []
  }
  var canSpeculate: Bool { capabilities.contains("speculative") }
  var isImageGeneration: Bool { capabilities.contains("image-generation") }
  var canTools: Bool { selectedModel == nil || capabilities.contains("tools") }
  var embeddedVision: Bool { selectedArtifact?.runtime?.embeddedVision == true }
  var visionArtifact: CatalogArtifact? {
    selectedModel?.artifacts.first { $0.kind == "vision" && companionAllowed($0) }
  }
  var hasAudioCompanion: Bool {
    selectedModel?.artifacts.contains { $0.kind == "audio" && companionAllowed($0) } == true
  }
  var visionServed: Bool { embeddedVision || (vision && !hasAudioCompanion) }
  var forensicsPossible: Bool {
    embeddedVision || visionArtifact != nil
      || (endpoint.settings?.vision == true && !hasAudioCompanion)
  }
  var drafters: [CatalogArtifact] {
    selectedModel?.artifacts.filter { $0.kind == "drafter" && companionAllowed($0) } ?? []
  }
  var electedDrafter: CatalogArtifact? {
    if !drafter.isEmpty { return drafters.first { $0.id == drafter } }
    return drafters.first { $0.installed && $0.default == true } ?? drafters.first { $0.installed }
      ?? drafters.first { $0.default == true } ?? drafters.first
  }
  var speculationSummary: String {
    guard canSpeculate else { return "No speculative decoder for this checkpoint" }
    guard speculation != "off" else { return "Disabled · drafter memory is released at load" }
    if isSplash { return "Bundled DFlash2" }
    if let drafter = electedDrafter {
      return drafter.displayLabel + (drafter.installed ? "" : " · download required")
    }
    return "Built-in MTP · no companion download"
  }
  private func companionAllowed(_ artifact: CatalogArtifact) -> Bool {
    artifact.supports(backend: "metal")
      && (selectedArtifact?.runtime?.companions?.contains(artifact.id) ?? true)
  }
  var contextCap: Int { selectedArtifact?.runtime?.memory?.maxCtx ?? 1_048_576 }
  var concurrencyCap: Int { min(256, selectedArtifact?.runtime?.memory?.maxBatch ?? 256) }
  var contextOptions: [Int] {
    Array(Set(Self.contextSteps.filter { $0 <= contextCap } + [contextCap])).sorted()
  }
  var compositionChanged: Bool {
    modelID != endpoint.model ?? "" || artifactID != endpoint.artifact ?? ""
      || vision != endpoint.settings?.vision ?? false || drafter != endpoint.settings?.drafter ?? ""
  }
  // A saved file is started verbatim. Missing keys use Config::default, not
  // launch recommendations (which only apply when choosing a composition).
  var effectiveContext: Int { Int(context) ?? 4096 }
  var effectiveConcurrency: Int { Int(concurrency) ?? 32 }
  var recommendedContext: Int { StartModelView.defaults(selectedArtifact?.runtime).context }
  var recommendedConcurrency: Int { 1 }

  func selectModel(_ model: CatalogModel) {
    guard model.id != modelID else { return }
    let available = purpose.weights(model, backend: "metal").filter { $0.installed }
    guard let choice = available.first(where: { $0.default == true }) ?? available.first else {
      return
    }
    modelID = model.id
    selectArtifact(choice, newModel: true)
  }
  func selectArtifact(_ artifact: CatalogArtifact, newModel: Bool = false) {
    guard artifact.installed, artifact.supports(backend: "metal"), artifact.kind == "weights" else {
      return
    }
    artifactID = artifact.id
    kvDtype = artifact.runtime?.kvCacheDtype ?? "f16"
    drafter = ""
    vision =
      hasAudioCompanion || visionArtifact?.installed == true || visionArtifact?.required == true
    if !visionServed { forensics = false }
    if !canSpeculate { speculation = "off" }
    if newModel || Int(context).map({ $0 > contextCap }) == true {
      context = String(recommendedContext)
    }
    // A workload is the user's choice, not a weight-format recommendation.
    // Preserve explicit values across model/quality changes; validation explains
    // incompatible values instead of silently resetting 1 to a server default.
    if newModel && concurrency.isEmpty { concurrency = String(recommendedConcurrency) }
    customContext = !Self.contextSteps.contains(Int(context) ?? 0)
    customWorkload = !Self.workloads.contains { $0.batch == Int(concurrency) }
  }
}
