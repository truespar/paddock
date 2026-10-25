import PaddockClient
import SwiftUI

extension EndpointEditor {
  static func offloadText(_ number: Double?) -> String {
    guard let number, number > 0 else { return "" }
    return number.formatted(.number.grouping(.never).precision(.fractionLength(0...3)))
  }
  var kvOffloadDirty: Bool {
    kvOffloadEnabled != (endpoint.settings?.kvOffload?.enabled ?? false)
      || kvOffloadRAM != Self.offloadText(endpoint.settings?.kvOffload?.ramGb)
      || kvOffloadDisk != Self.offloadText(endpoint.settings?.kvOffload?.nvmeGb)
  }
  var kvOffloadValue: EndpointKVOffload? {
    func number(_ text: String) -> Double? {
      if text.isEmpty { return 0 }
      return Double(
        text.replacingOccurrences(of: Locale.current.decimalSeparator ?? ".", with: "."))
    }
    guard let ram = number(kvOffloadRAM), let disk = number(kvOffloadDisk), ram.isFinite,
      disk.isFinite
    else { return nil }
    return .init(enabled: kvOffloadEnabled, ramGb: ram, nvmeGb: disk)
  }
  var kvOffloadValidation: String? {
    guard isCreating || kvOffloadDirty else { return nil }
    guard let value = kvOffloadValue, (0...1024).contains(value.ramGb),
      (0...8192).contains(value.nvmeGb)
    else {
      return "Enter valid KV cache budgets in GiB (RAM up to 1024, disk up to 8192)."
    }
    if value.enabled && endpoint.settings?.kvOffloadSupported != true {
      return "This model's Metal graph does not yet support KV offloading."
    }
    if value.enabled && value.ramGb < 0.5 {
      return
        "Enter a RAM budget of at least 0.5 GiB; model-specific transfer capacity is checked on start."
    }
    if value.ramGb * Double(1 << 30) > Double(memoryHardware.physicalBytes) {
      return "The cache budget exceeds this Mac's physical RAM."
    }
    return nil
  }
}

struct EndpointKVOffloadSettings: View {
  @Bindable var editor: EndpointEditor
  var body: some View {
    if editor.endpoint.settings?.kvOffloadSupported == true
      || editor.endpoint.settings?.kvOffload?.enabled == true
    {
      EndpointFormCard("KV offloading") {
        Toggle("Keep reusable conversation prefixes", isOn: $editor.kvOffloadEnabled)
          .toggleStyle(.switch).controlSize(.small).accessibilityIdentifier("endpoint-kv-offload")
          .help(
            "Caches reusable conversation state in RAM and optionally on SSD. Active state and model weights stay in unified memory."
          )
        if editor.kvOffloadEnabled {
          HStack(alignment: .top, spacing: 24) {
            EndpointFormField("RAM & transfer budget · GiB") {
              TextField("Enter budget", text: $editor.kvOffloadRAM).textFieldStyle(
                StudioPopoverFieldStyle()
              )
              .accessibilityIdentifier("endpoint-kv-ram")
              .help("Includes queued transfers and temporary copies.")
            }
            EndpointFormField("SSD cache · GiB") {
              TextField("0 · RAM only", text: $editor.kvOffloadDisk).textFieldStyle(
                StudioPopoverFieldStyle()
              )
              .accessibilityIdentifier("endpoint-kv-disk")
            }
          }
          if (editor.kvOffloadValue?.nvmeGb ?? 0) > 0 {
            EndpointHint(
              text: "SSD cache retains conversation data after restart, unencrypted on disk.")
          }
        }
      }.accessibilityIdentifier("endpoint-kv-offload-settings").settingsSearchAnchor(.kvOffload)
    }
  }
}
