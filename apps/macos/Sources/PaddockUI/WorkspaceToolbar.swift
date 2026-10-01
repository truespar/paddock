import SwiftUI

struct WorkspaceToolbar: ToolbarContent {
  @Binding var navigation: WorkspaceNavigation
  @Bindable var model: WorkspaceModel
  var onNewChat: () -> Void
  var onStart: () -> Void

  var body: some ToolbarContent {
    ToolbarItem(placement: .primaryAction) {
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
    }.flatChrome()
    ToolbarItem(placement: .navigation) {
      Button {
        navigation.toggleSidebar()
      } label: {
        Image(systemName: "sidebar.left").frame(width: 28, height: 28)
      }.buttonStyle(QuietButtonStyle()).foregroundStyle(.secondary)
        .accessibilityLabel("Toggle sidebar").accessibilityIdentifier("toggle-sidebar")
        .help(navigation.showsSidebar ? "Hide sidebar" : "Show sidebar")
        .keyboardShortcut("s", modifiers: [.command, .control])
    }.flatChrome()
    ToolbarItem(placement: .navigation) {
      Button {
        if navigation.mode == .studio && navigation.studio == .tables {
          Task { await model.tables.newSession() }
        } else if navigation.mode == .studio {
          onNewChat()
        } else {
          navigation.returnToChat()
        }
      } label: {
        Image(systemName: navigation.mode == .studio ? "square.and.pencil" : "arrow.left")
          .frame(width: 28, height: 28)
      }.buttonStyle(QuietButtonStyle()).foregroundStyle(.secondary)
        .accessibilityLabel(
          navigation.mode == .studio
            ? (navigation.studio == .tables ? "New table" : "New chat") : "Back to chat")
    }.flatChrome()
    if !navigation.showsSidebar {
      ToolbarItem(placement: .navigation) {
        Button {
          navigation.showSettings()
        } label: {
          Image(systemName: "gearshape").frame(width: 28, height: 28)
        }.buttonStyle(QuietButtonStyle()).foregroundStyle(.secondary)
          .accessibilityLabel("Settings").accessibilityIdentifier("open-settings")
          .help("Settings · ⌘,")
      }.flatChrome()
    }
  }
}

extension ToolbarContent {
  /// Tahoe groups toolbar items in shared glass by default. Keep our flat
  /// controls while retaining native title-bar placement and the macOS 15 floor.
  @ToolbarContentBuilder fileprivate func flatChrome() -> some ToolbarContent {
    if #available(macOS 26.0, *) { sharedBackgroundVisibility(.hidden) } else { self }
  }
}
