import Foundation
import Metal
import SwiftUI

/// OS facts only. This is not a second inference-memory planner: in particular,
/// recommendedMaxWorkingSetSize is not free memory or a fit measurement.
struct MetalMemoryHardware: Sendable {
  let physicalBytes: UInt64
  let recommendedBytes: UInt64?
  static var current: Self {
    let device = MTLCreateSystemDefaultDevice()
    return Self(
      physicalBytes: ProcessInfo.processInfo.physicalMemory,
      recommendedBytes: device?.recommendedMaxWorkingSetSize)
  }
  static func gib(_ bytes: UInt64) -> String {
    "\((Double(bytes) / Double(1 << 30)).formatted(.number.precision(.fractionLength(0...1)))) GiB"
  }
}

struct EndpointMemorySettings: View {
  @Bindable var editor: EndpointEditor
  var body: some View {
    EndpointFormCard("Memory budget") {
      HStack(alignment: .firstTextBaseline) {
        Text("Unified memory").font(.system(size: 12))
        Spacer()
        Text(MetalMemoryHardware.gib(editor.memoryHardware.physicalBytes))
          .font(.system(size: 14, weight: .semibold)).monospacedDigit()
      }
      VStack(alignment: .leading, spacing: 12) {
        HStack(spacing: 8) {
          Button("Automatic") { editor.setCustomMemoryBudget(false) }
            .buttonStyle(FlatButtonStyle(primary: !editor.customMemoryBudget))
            .accessibilityValue(editor.customMemoryBudget ? "Not selected" : "Selected")
            .help("Choose a budget automatically when loading the model.")
          Button("Limit to…") { editor.setCustomMemoryBudget(true) }
            .buttonStyle(FlatButtonStyle(primary: editor.customMemoryBudget))
            .accessibilityValue(editor.customMemoryBudget ? "Selected" : "Not selected")
        }.accessibilityIdentifier("endpoint-memory-policy")
        if editor.customMemoryBudget {
          HStack(spacing: 10) {
            TextField("Enter limit", text: $editor.memoryLimit)
              .textFieldStyle(StudioPopoverFieldStyle()).frame(width: 110)
              .accessibilityLabel("Memory limit in GiB")
              .accessibilityIdentifier("endpoint-memory-limit")
              .help(
                "Limits this model's Metal buffers, not the whole process. Too low a limit can prevent loading."
              )
            Text("GiB maximum").font(.system(size: 12)).foregroundStyle(.secondary)
          }
          if let error = editor.memoryValidation {
            Text(error).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
              .fixedSize(horizontal: false, vertical: true)
          }
        }
      }
      if let recommended = editor.memoryHardware.recommendedBytes {
        HStack {
          Text("Recommended GPU limit")
          Spacer()
          Text(MetalMemoryHardware.gib(recommended)).monospacedDigit()
        }.font(.system(size: 12)).foregroundStyle(.secondary)
          .help("Apple's recommended working set, not currently free memory.")
      }
    }.accessibilityIdentifier("endpoint-memory-settings").settingsSearchAnchor(.memoryBudget)
  }
}
