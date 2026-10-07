import AppKit
import Observation
import PaddockClient
import SwiftUI

@MainActor @Observable
final class BenchmarksModel {
  var selected: String?
  var concurrency = 1
  var long = false
  private(set) var running = false
  private(set) var reports: [BenchmarkReport] = []
  private(set) var error: String?
  private(set) var cancelled = false
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored let client: any ManagerLoading
  init(client: any ManagerLoading) { self.client = client }
  func load() async {
    do {
      reports = try await client.inspect(.benchmarkHistory, as: BenchmarkHistory.self).reports
    } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
  }
  func start(_ runner: RunnerInfo) {
    guard !running else { return }
    selected = runner.id
    running = true
    error = nil
    cancelled = false
    let concurrency = concurrency
    let long = long
    task = Task {
      defer {
        running = false
        task = nil
      }
      do {
        _ = try await client.inspect(
          .benchmark(port: runner.port, pid: runner.pid, concurrency: concurrency, long: long),
          as: BenchmarkReport.self, timeout: .seconds(900))
        await load()
      } catch {
        if Task.isCancelled { cancelled = true } else { self.error = error.localizedDescription }
      }
    }
  }
  func cancel() {
    let pending = task
    pending?.cancel()
    // Completion can race cancellation after Rust persisted a full report.
    // Reconcile history on a fresh task rather than claiming it was discarded.
    Task {
      await pending?.value
      await load()
    }
  }
  func export(_ report: BenchmarkReport) {
    let panel = NSSavePanel()
    panel.nameFieldStringValue = "Paddock-benchmark-\(report.id).json"
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      Task {
        do {
          _ = try await self.client.inspect(
            .exportBenchmark(id: report.id, path: url.path), as: NativeExportReceipt.self)
          NSWorkspace.shared.activateFileViewerSelecting([url])
        } catch { self.error = error.localizedDescription }
      }
    }
  }
}

struct BenchmarksView: View {
  @Bindable var model: BenchmarksModel
  let runners: [RunnerInfo]
  @State private var confirm = false
  @State private var reviewed: RunnerInfo?
  private var choices: [RunnerInfo] { runners.filter { $0.model != nil && $0.status == "ok" } }
  private var selected: RunnerInfo? {
    model.selected == nil ? choices.first : choices.first { $0.id == model.selected }
  }
  var body: some View {
    SettingsPage(title: "Benchmarks") { stacked in
      SettingsGroup(title: "Text generation") {
        configuration(stacked: stacked)
        actions
        if model.cancelled {
          Text("Benchmark cancelled.").foregroundStyle(.secondary)
        }
        if let error = model.error {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("benchmark-error")
        }
      }
      if !model.reports.isEmpty {
        VStack(alignment: .leading, spacing: 16) {
          Text("Results").fontWeight(.semibold).accessibilityAddTraits(.isHeader)
          LazyVStack(spacing: 16) {
            ForEach(model.reports) { report in
              BenchmarkResultCard(report: report, stacked: stacked) { model.export(report) }
            }
          }
        }
      }
    }
    .task { await model.load() }
    .confirmationDialog(
      "Run a local benchmark?", isPresented: $confirm, titleVisibility: .visible
    ) {
      Button("Run") {
        // Confirm the same process the user reviewed, never another runner
        // that happened to take its place while the dialog was open.
        if let reviewed, let current = choices.first(where: { $0.id == reviewed.id }),
          current.inFlight == 0
        {
          model.start(current)
        }
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text(
        "One warmup and three measured trials use synthetic text, with up to 128 output tokens per request. Other requests can affect results. Model settings and conversation caches remain unchanged."
      )
    }
  }

  private func configuration(stacked: Bool) -> some View {
    VStack(alignment: .leading, spacing: 18) {
      SettingsRow(title: "Instance", stacked: stacked) {
        Dropdown(
          title: "Instance", value: selected?.title ?? "Choose an instance", fillsWidth: true
        ) {
          ForEach(choices) { choice in
            Button("\(choice.title) · \(choice.port)") { model.selected = choice.id }
          }
        }.disabled(choices.isEmpty).accessibilityIdentifier("benchmark-instance")
      }
      SettingsRow(title: "Prompt", stacked: stacked) {
        BenchmarkSegments(
          title: "Prompt", options: [(false, "Short"), (true, "Long")], selection: $model.long
        ).frame(height: 30)
          .accessibilityIdentifier("benchmark-prompt")
      }
      SettingsRow(title: "Concurrency", stacked: stacked) {
        BenchmarkSegments(
          title: "Concurrency", options: [(1, "1 request"), (4, "4 requests")],
          selection: $model.concurrency
        ).frame(height: 30)
          .accessibilityIdentifier("benchmark-concurrency")
      }
    }.disabled(model.running)
  }

  private var actions: some View {
    VStack(alignment: .leading, spacing: 12) {
      if !model.running {
        if choices.isEmpty {
          Text("Start a local chat model to run a benchmark.").foregroundStyle(.secondary)
        } else if selected == nil {
          Text("The selected instance stopped. Choose another instance.").foregroundStyle(
            .secondary)
        } else if selected?.inFlight != 0 {
          Text("Waiting for this instance to finish its requests.").foregroundStyle(.secondary)
        }
      }
      HStack(spacing: 8) {
        if model.running {
          ProgressView().controlSize(.small)
          Text("Measuring…").foregroundStyle(.secondary)
          Spacer(minLength: 12)
          Button("Cancel") { model.cancel() }.accessibilityIdentifier("benchmark-cancel")
        } else {
          Spacer(minLength: 0)
          Button("Run benchmark", systemImage: "speedometer") {
            reviewed = selected
            confirm = true
          }.buttonStyle(FlatButtonStyle(primary: true))
            .disabled(selected == nil || selected?.inFlight != 0)
            .accessibilityIdentifier("benchmark-run")
        }
      }.frame(maxWidth: .infinity)
    }
  }
}
