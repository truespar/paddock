import Foundation
import Observation
import PaddockClient

/// App-owned edit draft and receipt. Views may disappear; an acknowledged
/// operation keeps its identity and is never resubmitted after a poll failure.
@MainActor @Observable
final class EndpointEditor {
  private(set) var endpoint: ConfiguredEndpoint
  private(set) var pid: UInt32?
  private(set) var runtimeState: EndpointRuntimeState?
  var context = ""
  var concurrency = ""
  var speculation = ""
  var host = ""
  var replacementKey = ""
  var forensics = false
  var modelID = ""
  var artifactID = ""
  var vision = false
  var audio = false
  var drafter = ""
  var kvDtype = ""
  var memoryLimit = ""
  var customMemoryBudget = false
  var kvOffloadEnabled = false
  var kvOffloadRAM = ""
  var kvOffloadDisk = ""
  var loadOnDemand = false
  var unloadIdleSeconds = ""
  var loadWaitSeconds = "120"
  let memoryHardware: MetalMemoryHardware
  var customWorkload = false
  var customContext = false
  var advanced = false
  var runtimeDraft: [String: String] = [:]
  var catalog: [CatalogModel] = []
  let isCreating: Bool
  let purpose: ModelStartPurpose
  var automaticPort = true
  var newPort = "11540"
  private(set) var error: String?
  private(set) var message: String?
  private(set) var pending: ManagementJob?
  private(set) var submitting = false
  private(set) var checking = false
  private(set) var refreshing = false
  private(set) var removed = false
  private(set) var needsReload = false
  @ObservationIgnored private let client: any ManagerLoading
  var settingsClient: any ManagerLoading { client }

  /// Profiles fill the reviewed draft only. Saving/restarting still uses the
  /// existing revision/PID-checked backend contract and its validation.
  func useProfile(_ profile: ModelProfile) {
    guard !saving, !refreshing, !dirty, profile.model == modelID, profile.artifact == artifactID
    else { return }
    let values = profile.settings
    let residency = values.residency ?? EndpointResidency()
    loadOnDemand = residency.load == "on_demand"
    unloadIdleSeconds = residency.unloadAfterIdleSeconds.map(String.init) ?? ""
    loadWaitSeconds = String(residency.loadTimeoutSeconds)
    context = values.maxCtx.map(String.init) ?? ""
    concurrency = values.maxBatch.map(String.init) ?? ""
    speculation = values.noSpec == true ? "off" : values.spec ?? ""
    kvDtype = values.kvCacheDtype ?? ""
    memoryLimit = Self.budgetText(values.vramBudget)
    customMemoryBudget = values.vramBudget != nil
    kvOffloadEnabled = values.kvOffload?.enabled ?? false
    kvOffloadRAM = Self.offloadText(values.kvOffload?.ramGb)
    kvOffloadDisk = Self.offloadText(values.kvOffload?.nvmeGb)
    for option in values.runtimeOptions ?? []
    where runtimeOptions.contains(where: { $0.id == option.id }) {
      runtimeDraft[option.id] = option.value?.text ?? ""
    }
    customWorkload = !Self.workloads.contains { $0.batch == effectiveConcurrency }
    customContext = !Self.contextSteps.contains(effectiveContext)
  }
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored var onChange: (() async -> Void)?

  init(
    client: any ManagerLoading, endpoint: ConfiguredEndpoint, pid: UInt32?,
    catalog: [CatalogModel] = [], memoryHardware: MetalMemoryHardware = .current,
    isCreating: Bool = false, purpose: ModelStartPurpose = .all
  ) {
    self.client = client
    self.endpoint = endpoint
    self.pid = pid
    self.runtimeState = endpoint.runtimeState
    self.catalog = catalog
    self.memoryHardware = memoryHardware
    self.isCreating = isCreating
    self.purpose = purpose
    reset()
  }

  var restartRequired: Bool { runtimeState?.pid == pid && runtimeState?.restartRequired == true }

  var saving: Bool { submitting || pending != nil }
  var dirty: Bool {
    !changes.isEmpty || customMemoryBudget != (endpoint.settings?.vramBudget != nil)
      || kvOffloadDirty
  }
  var onlyResidencyChanges: Bool {
    !changes.isEmpty && changes.allSatisfy { if case .residency = $0 { true } else { false } }
  }
  var localOnly: Bool { host == "127.0.0.1" || host == "::1" }
  var validation: String? {
    if residencySupported && residencyValue == nil {
      return "Use an idle timeout of 0–604800 seconds and a load wait limit of 1–3600 seconds."
    }
    guard endpoint.settings != nil, isCreating || endpoint.revision != nil else {
      return "Reload this endpoint's saved settings before editing."
    }
    if isCreating
      && (endpoint.model != modelID || endpoint.artifact != artifactID || refreshing || needsReload)
    {
      return "Loading settings for the selected weights…"
    }
    if isCreating && !automaticPort && !(UInt16(newPort).map { $0 >= 1024 } ?? false) {
      return "Use a port between 1024 and 65535, or choose Automatic."
    }
    if !context.isEmpty && !(Int(context).map { (256...1_048_576).contains($0) } ?? false) {
      return "Context must be 256-1048576 tokens, or empty for the runner default."
    }
    // A cleared explicit workload must not silently save null and select the
    // runner's legacy 32-slot server default. Existing absent keys stay intact.
    if concurrency.isEmpty && endpoint.settings?.maxBatch != nil {
      return "Enter a workload of at least 1, or choose Just me."
    }
    if !concurrency.isEmpty && !(Int(concurrency).map { (1...256).contains($0) } ?? false) {
      return "Concurrency must be 1-256."
    }
    if effectiveContext > contextCap {
      return "This export supports up to \(contextCap.formatted()) context tokens."
    }
    if effectiveConcurrency > concurrencyCap {
      return "This export supports up to \(concurrencyCap) concurrent requests."
    }
    if (isCreating || compositionChanged)
      && (selectedArtifact == nil || selectedArtifact?.installed != true)
    {
      return "Download the selected weights before saving."
    }
    if let option = runtimeOptions.first(where: {
      let text = runtimeDraft[$0.id] ?? ""
      return text != ($0.value?.text ?? "") && !text.isEmpty && $0.parse(text) == nil
    }) {
      return
        "Invalid \(option.label.lowercased()). Check its range in Advanced or restore the default."
    }
    if isCreating || dirtyCompositionOrSpec, speculation != "off", !drafters.isEmpty,
      electedDrafter?.installed != true
    {
      return "Download a compatible drafter before enabling speculative decoding."
    }
    if isCreating || compositionChanged, vision, visionArtifact?.installed != true, !embeddedVision,
      !bundledVision,
      !hasAudioCompanion || splitMediaTowers
    {
      return "Download the vision companion before enabling image input."
    }
    if isCreating || compositionChanged, splitMediaTowers, audio, audioArtifact?.installed != true,
      !bundledAudio
    {
      return "Download the audio companion before enabling audio input."
    }
    if forensics && !visionServed && forensics != endpoint.settings?.forensics {
      return "Forensics needs image input. Enable Vision first."
    }
    if kvDtype != endpoint.settings?.kvCacheDtype ?? "", !kvChoices.contains(kvDtype) {
      return "Choose conversation memory supported by this Metal checkpoint."
    }
    if let memoryValidation { return memoryValidation }
    if let kvOffloadValidation { return kvOffloadValidation }
    if !replacementKey.isEmpty
      && (!(16...4096).contains(replacementKey.utf8.count)
        || !replacementKey.utf8.allSatisfy({ (33...126).contains($0) }))
    {
      return "Use at least 16 printable ASCII characters without spaces for the replacement key."
    }
    if !localOnly && replacementKey.isEmpty && endpoint.settings?.hasApiKey != true {
      return "Network access requires an API key."
    }
    return nil
  }

  var changes: [EndpointChange] {
    guard let old = endpoint.settings else { return [] }
    var result: [EndpointChange] = []
    if context != old.maxCtx.map(String.init) ?? "" { result.append(.maxCtx(Int(context))) }
    if concurrency != old.maxBatch.map(String.init) ?? "" {
      result.append(.maxBatch(Int(concurrency)))
    }
    if speculation != savedSpeculation {
      result.append(.spec(speculation.isEmpty ? nil : speculation))
    }
    if host != old.host { result.append(.host(host)) }
    if !replacementKey.isEmpty { result.append(.apiKey(replacementKey)) }
    if forensics != old.forensics { result.append(.forensics(forensics)) }
    if kvDtype != old.kvCacheDtype ?? "" {
      result.append(.kvCacheDtype(kvDtype.isEmpty ? nil : kvDtype))
    }
    if memoryLimit != Self.budgetText(old.vramBudget) {
      result.append(.vramBudget(memoryBudgetMiB))
    }
    if compositionChanged {
      result.append(
        .composition(
          .init(
            model: modelID, artifact: artifactID, vision: vision,
            drafter: drafter.isEmpty ? nil : drafter, audio: splitMediaTowers ? audio : nil)))
    }
    var options: [String: EndpointRuntimeValue?] = [:]
    for field in runtimeOptions {
      let text = runtimeDraft[field.id] ?? ""
      if text != field.value?.text ?? "" {
        options[field.id] = .some(text.isEmpty ? nil : field.parse(text))
      }
    }
    if !options.isEmpty { result.append(.runtime(options)) }
    if kvOffloadDirty, let offload = kvOffloadValue { result.append(.kvOffload(offload)) }
    if let value = residencyValue, value != old.residency ?? EndpointResidency() {
      result.append(.residency(value))
    }
    return result
  }

  func reset() {
    let residency = endpoint.settings?.residency ?? EndpointResidency()
    loadOnDemand = residency.load == "on_demand"
    unloadIdleSeconds = residency.unloadAfterIdleSeconds.map(String.init) ?? ""
    loadWaitSeconds = String(residency.loadTimeoutSeconds)
    guard !saving else { return }
    context = endpoint.settings?.maxCtx.map(String.init) ?? ""
    concurrency = endpoint.settings?.maxBatch.map(String.init) ?? ""
    speculation = savedSpeculation
    host = endpoint.settings?.host ?? "127.0.0.1"
    forensics = endpoint.settings?.forensics ?? false
    modelID = endpoint.model ?? ""
    artifactID = endpoint.artifact ?? ""
    vision = endpoint.settings?.vision ?? false
    audio = endpoint.settings?.audio ?? false
    drafter = endpoint.settings?.drafter ?? ""
    kvDtype = endpoint.settings?.kvCacheDtype ?? ""
    memoryLimit = Self.budgetText(endpoint.settings?.vramBudget)
    customMemoryBudget = endpoint.settings?.vramBudget != nil
    kvOffloadEnabled = endpoint.settings?.kvOffload?.enabled ?? false
    kvOffloadRAM = Self.offloadText(endpoint.settings?.kvOffload?.ramGb)
    kvOffloadDisk = Self.offloadText(endpoint.settings?.kvOffload?.nvmeGb)
    runtimeDraft = Dictionary(
      uniqueKeysWithValues: runtimeOptions.map { ($0.id, $0.value?.text ?? "") })
    customWorkload = !Self.workloads.contains { $0.batch == effectiveConcurrency }
    customContext = !Self.contextSteps.contains(effectiveContext)
    replacementKey = ""
    error = nil
  }

  func save(_ apply: EndpointApply, networkConfirmed: Bool = false) {
    guard !isCreating, !saving, !refreshing, !needsReload,
      dirty || (apply == .restart && restartRequired), validation == nil,
      let revision = endpoint.revision
    else { return }
    if !localOnly && !networkConfirmed {
      error = "Confirm network access before saving."
      return
    }
    if apply == .restart && pid == nil {
      error = "This endpoint is stopped. Save for its next start instead."
      return
    }
    begin(
      .edit(
        port: endpoint.port, revision: revision, pid: pid, changes: changes, apply: apply,
        allowNetwork: networkConfirmed))
  }

  func remove() {
    guard !isCreating, !saving, !refreshing, !dirty, !needsReload, pid == nil,
      let revision = endpoint.revision
    else { return }
    begin(.remove(port: endpoint.port, revision: revision))
  }

  /// Creation sends the complete reviewed draft, not deltas against defaults
  /// which could change between opening the form and pressing Start.
  func creationRequest(networkConfirmed: Bool = false, tools: EndpointCreationTools? = nil)
    -> CreateEndpointRequest?
  {
    guard isCreating, !refreshing, !needsReload, validation == nil,
      localOnly || networkConfirmed
    else { return nil }
    var fields: [EndpointChange] = [
      .maxCtx(effectiveContext), .maxBatch(effectiveConcurrency),
      .spec(speculation), .host(host), .forensics(forensics),
      .kvCacheDtype(kvDtype.isEmpty ? nil : kvDtype), .vramBudget(memoryBudgetMiB),
      .composition(
        .init(
          model: modelID, artifact: artifactID, vision: vision,
          drafter: drafter.isEmpty ? nil : drafter, audio: splitMediaTowers ? audio : nil)),
      .runtime(
        Dictionary(
          uniqueKeysWithValues: runtimeOptions.map {
            ($0.id, (runtimeDraft[$0.id] ?? "").isEmpty ? nil : $0.parse(runtimeDraft[$0.id] ?? ""))
          })),
    ]
    if !replacementKey.isEmpty { fields.append(.apiKey(replacementKey)) }
    if let value = kvOffloadValue { fields.append(.kvOffload(value)) }
    if residencySupported, let value = residencyValue { fields.append(.residency(value)) }
    return CreateEndpointRequest(
      model: modelID, artifact: artifactID,
      port: automaticPort ? nil : UInt16(newPort), changes: fields,
      allowNetwork: networkConfirmed, tools: tools)
  }

  /// Re-probe model-specific controls after a weights change. Keep the user's
  /// workload, access and sampling choices; stale asynchronous reads cannot
  /// replace a newer selection or leave Start enabled with the old schema.
  func prepareCreationSelection() async {
    guard isCreating else { return }
    let model = modelID
    let artifact = artifactID
    refreshing = true
    needsReload = true
    error = nil
    do {
      let prepared = try await client.prepareEndpoint(model: model, artifact: artifact)
      guard !Task.isCancelled, modelID == model, artifactID == artifact else { return }
      endpoint = prepared
      needsReload = false
      refreshing = false
    } catch {
      guard !Task.isCancelled, modelID == model, artifactID == artifact else { return }
      self.error = error.localizedDescription
      refreshing = false
    }
  }

  private func begin(_ command: ModelCommand) {
    submitting = true
    error = nil
    message = nil
    task = Task {
      do {
        pending = try await client.submit(command)
        submitting = false
        await poll()
      } catch {
        submitting = false
        self.error = error.localizedDescription
      }
    }
  }

  func retryStatus() {
    guard pending != nil, !checking else { return }
    task = Task { await poll() }
  }

  private func poll() async {
    guard let receipt = pending else { return }
    checking = true
    defer { checking = false }
    do {
      var job = receipt
      while job.isActive {
        try await Task.sleep(for: .milliseconds(350))
        job = try await client.submit(.poll(id: receipt.id))
        guard job.id == receipt.id, job.port == endpoint.port else {
          throw ManagerError.core(
            "The operation returned a different receipt. Refresh status before continuing.")
        }
        pending = job
      }
      pending = nil
      if job.state == "succeeded" {
        message = job.message
        replacementKey = ""
        if job.action == "remove" {
          removed = true
        } else {
          needsReload = true
          await reload()
        }
      } else {
        error = job.message
      }
      await onChange?()
    } catch {
      self.error =
        "The operation was accepted, but its status could not be read. Retry status; do not submit the operation again."
    }
  }

  /// Caller confirms discarding a dirty draft. A failed refresh keeps it intact.
  func reload() async {
    if isCreating {
      await prepareCreationSelection()
      return
    }
    guard !saving, !refreshing else { return }
    refreshing = true
    defer { refreshing = false }
    do {
      let snapshot = try await client.snapshot()
      guard let row = snapshot.servers?.first(where: { $0.port == endpoint.port }) else {
        throw ManagerError.core("This endpoint was removed. Your draft has not been discarded.")
      }
      guard row.settings != nil else {
        throw ManagerError.core(row.configError ?? "Saved settings are unavailable.")
      }
      endpoint = row
      runtimeState = row.runtimeState
      catalog = snapshot.catalog.models
      pid = snapshot.runners.first(where: { $0.port == row.port })?.pid
      needsReload = false
      reset()
    } catch { self.error = error.localizedDescription }
  }

  func observeRuntime(_ snapshot: ManagerSnapshot) {
    guard !saving else { return }
    observeCatalog(snapshot.catalog.models)
    pid = snapshot.runners.first { $0.port == endpoint.port }?.pid
    runtimeState = snapshot.servers?.first { $0.port == endpoint.port }?.runtimeState
  }

  /// A completed download changes availability, not the user's settings draft.
  func observeCatalog(_ models: [CatalogModel]) {
    guard !saving else { return }
    catalog = models
  }

  func settle() async { await task?.value }
}

extension EndpointEditor {
  var residencySupported: Bool { endpoint.settings?.residencySupported == true }
  var residencyValue: EndpointResidency? {
    guard let wait = Int(loadWaitSeconds), (1...3600).contains(wait) else { return nil }
    let idle = unloadIdleSeconds.isEmpty ? nil : Int(unloadIdleSeconds)
    guard unloadIdleSeconds.isEmpty || idle.map({ (0...604800).contains($0) }) == true else {
      return nil
    }
    return EndpointResidency(
      load: loadOnDemand ? "on_demand" : "at_startup", unloadAfterIdleSeconds: idle,
      loadTimeoutSeconds: wait)
  }
}
