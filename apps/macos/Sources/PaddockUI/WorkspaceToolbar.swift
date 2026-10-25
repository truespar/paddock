import SwiftUI

struct WorkspaceMetricsButton: View {
  @Bindable var model: WorkspaceModel

  var body: some View {
    Button {
      model.showsGPUMetrics.toggle()
    } label: {
      Image(systemName: "chart.xyaxis.line").frame(width: 28, height: 28)
    }.buttonStyle(QuietButtonStyle()).foregroundStyle(.secondary)
      .accessibilityLabel("GPU metrics").accessibilityIdentifier("gpu-metrics")
      .help("GPU metrics")
      .popover(isPresented: $model.showsGPUMetrics, arrowEdge: .bottom) {
        GPUMetricsView(
          snapshot: model.snapshot?.gpu, runners: model.snapshot?.runners ?? [],
          history: model.gpuHistory, stale: model.gpuMeasurementsStale
        )
        .task { await model.refresh() }
      }
  }
}
