import PaddockClient

extension EndpointEditor {
  var showsContextWorkload: Bool {
    !isImageGeneration && !capabilities.contains("decision")
      && !capabilities.contains("tabular") && !capabilities.contains("diarization")
  }
  var searchableSettings: [SettingsSearchEntry] {
    var anchors: [SettingsAnchor] = [.memoryBudget, .endpointAccess]
    if showsContextWorkload { anchors += [.context, .workload] }
    if canSpeculate || (selectedModel == nil && endpoint.settings?.spec != nil) {
      anchors.append(.speculation)
    }
    if residencySupported { anchors += [.modelLoading, .idleUnload, .loadWait] }
    if endpoint.settings?.kvOffloadSupported == true
      || endpoint.settings?.kvOffload?.enabled == true
    {
      anchors.append(.kvOffload)
    }
    if canTools { anchors += [.webSearch, .mcpServers] }
    return anchors.map {
      SettingsSearchEntry(
        destination: .runners, anchor: $0, port: endpoint.port, instanceTitle: title)
    }
      + visibleRuntimeOptions.map {
        SettingsSearchEntry(
          destination: .runners, port: endpoint.port, instanceTitle: title,
          runtimeOption: $0.id, label: $0.label, keywords: "\($0.group) \($0.help)")
      }
  }
}

extension WorkspaceModel {
  /// Search opens the existing reviewed form; it never mutates model settings,
  /// starts a runner, or chooses an arbitrary instance for a generic result.
  @discardableResult func openSetting(_ entry: SettingsSearchEntry) -> Bool {
    if let port = entry.port {
      guard let snapshot,
        let row = EndpointRow.rows(snapshot: snapshot, latestJob: latestJob).first(where: {
          $0.port == port
        }),
        row.configured?.settings != nil
      else {
        desktopError = "This instance is no longer available. Choose it again in Instances."
        return false
      }
      if integrations.hasDraft, integrations.searchEditor?.settings.port != port {
        desktopError = "Save or discard the connector changes before opening another instance."
        return false
      }
      openEndpoint(row)
      guard endpointEditor?.endpoint.port == port, detailEndpointPort == port else { return false }
      guard endpointEditor?.searchableSettings.contains(where: { $0.id == entry.id }) == true else {
        desktopError = "This setting is not available for the current model configuration."
        return false
      }
      editEndpoint()
      endpointEditor?.advanced =
        entry.runtimeOption != nil || entry.anchor == .loadWait || entry.anchor == .endpointAccess
      if entry.anchor == .webSearch || entry.anchor == .mcpServers { systemToolsPort = port }
    } else {
      navigation.showManager(entry.destination)
    }
    // An explicit choice of the default page still wins over a late restore.
    preferences.set(.settingsPage, .string(entry.destination.rawValue))
    return true
  }
}
