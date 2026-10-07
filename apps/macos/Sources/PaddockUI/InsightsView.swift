import Charts
import PaddockClient
import SwiftUI

struct InsightsView: View {
  @Bindable var model: InsightsModel
  let endpoints: [ConfiguredEndpoint]
  var body: some View {
    SettingsPage(title: "Usage & activity") { stacked in
      VStack(alignment: .leading, spacing: 18) {
        let layout =
          stacked
          ? AnyLayout(VStackLayout(alignment: .leading, spacing: 16))
          : AnyLayout(HStackLayout(spacing: 16))
        layout {
          Picker("View", selection: $model.page) {
            ForEach(InsightPage.allCases) { Text($0.rawValue).tag($0) }
          }.pickerStyle(.segmented).labelsHidden().frame(maxWidth: stacked ? .infinity : 340)
            .accessibilityIdentifier("insights-page")
          if !stacked { Spacer(minLength: 0) }
          HStack {
            Toggle("Live", isOn: $model.live).toggleStyle(.switch).controlSize(.small)
              .accessibilityIdentifier("insights-live")
            if model.loading && model.sampledAt == nil { ProgressView().controlSize(.small) }
          }
        }
      }
      SettingsGroup(title: "Filters") {
        SettingsRow(title: "Instance", stacked: stacked) {
          Dropdown(
            title: "Instance",
            value: endpoints.first { $0.port == model.port }?.title ?? "All instances",
            fillsWidth: true
          ) {
            Button("All instances") { model.port = nil }
            ForEach(endpoints, id: \.port) { endpoint in
              Button("\(endpoint.title) · \(endpoint.port)") { model.port = endpoint.port }
            }
          }.accessibilityIdentifier("insights-instance")
        }
        if model.page == .usage {
          SettingsRow(title: "Time range", stacked: stacked) {
            Dropdown(
              title: "Time range",
              value: model.days == 1 ? "Last 24 hours" : "Last \(model.days) days",
              fillsWidth: true
            ) {
              ForEach([1, 7, 30, 90, 365], id: \.self) { days in
                Button(days == 1 ? "Last 24 hours" : "Last \(days) days") { model.days = days }
              }
            }.accessibilityIdentifier("insights-range")
          }
        }
      }
      if let error = model.error {
        Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
      }
      switch model.page {
      case .usage: if let usage = model.usage { UsageInsights(usage: usage, stacked: stacked) }
      case .activity: if let activity = model.activity { ActivityInsights(snapshot: activity) }
      case .cache:
        if let cache = model.cache { CacheInsights(snapshot: cache, port: model.port) }
      }
    }
    .task(id: model.queryID) { await model.observe() }
    .accessibilityIdentifier("management-insights")
  }
}

private struct UsageInsights: View {
  let usage: UsageHistorySnapshot
  let stacked: Bool
  var body: some View {
    if usage.buckets.isEmpty {
      ContentUnavailableView(
        "No recorded usage", systemImage: "chart.bar",
        description: Text("Requests from running local instances appear here."))
    } else {
      SettingsGroup(title: "Requests & tokens") {
        metric("Requests", usage.buckets.reduce(0) { $0 + $1.requests })
        metric("Input tokens", usage.buckets.reduce(0) { $0 + $1.inputTokens })
        metric("Output tokens", usage.buckets.reduce(0) { $0 + $1.outputTokens })
        metric("Cached input tokens", usage.buckets.reduce(0) { $0 + $1.cachedTokens })
        metric("Errors", usage.buckets.reduce(0) { $0 + $1.errors4xx + $1.errors5xx })
        metric("Disconnected requests", usage.buckets.reduce(0) { $0 + $1.disconnects })
        if usage.buckets.contains(where: { $0.outputTokens > 0 }) {
          Chart(usage.buckets) { bucket in
            BarMark(x: .value("Time", bucket.date), y: .value("Output tokens", bucket.outputTokens))
              .foregroundStyle(PaddockStyle.accent.opacity(0.65))
          }.frame(height: 180).accessibilityLabel("Recorded output tokens over time")
        }
      }
    }
    if !usage.gaps.isEmpty {
      SettingsGroup(title: "Observation gaps") {
        ForEach(usage.gaps) { gap in
          let layout =
            stacked
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 6))
            : AnyLayout(HStackLayout(spacing: 16))
          layout {
            Text("Port \(gap.port) · \(gap.cause)")
            if !stacked { Spacer(minLength: 0) }
            Text(
              "\(Date(timeIntervalSince1970: Double(gap.fromTsMs) / 1000).formatted(date: .abbreviated, time: .shortened)) - \(Date(timeIntervalSince1970: Double(gap.toTsMs) / 1000).formatted(date: .abbreviated, time: .shortened))"
            ).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
          }
        }
      }
    }
    if !usage.web.isEmpty {
      SettingsGroup(title: "Web search") {
        ForEach(Array(Set(usage.web.map(\.provider))).sorted(), id: \.self) { provider in
          let rows = usage.web.filter { $0.provider == provider }
          metric("\(provider) requests", rows.reduce(0) { $0 + $1.requests })
          metric("\(provider) credits", rows.reduce(0) { $0 + $1.credits })
          let dollars = Double(rows.reduce(0) { $0 + $1.microdollars }) / 1_000_000
          HStack {
            Text("\(provider) cost")
            Spacer()
            Text(dollars, format: .currency(code: "USD"))
          }
        }
      }
    }
  }
  private func metric(_ label: String, _ value: Int64) -> some View {
    HStack {
      Text(label).foregroundStyle(.secondary)
      Spacer()
      Text(value.formatted()).monospacedDigit()
    }
  }
}

private struct ActivityInsights: View {
  let snapshot: ActivitySnapshot
  var body: some View {
    if snapshot.events.isEmpty {
      ContentUnavailableView("No request activity", systemImage: "waveform.path")
    } else {
      Text("Latest \(snapshot.events.count) requests").foregroundStyle(.secondary)
      LazyVStack(alignment: .leading, spacing: 12) {
        ForEach(snapshot.events) { event in
          DisclosureGroup {
            VStack(spacing: 10) {
              ForEach(event.fields.keys.sorted(), id: \.self) { key in
                HStack(alignment: .top) {
                  Text(activityLabel(key)).foregroundStyle(.secondary)
                  Spacer(minLength: 20)
                  Text(event.fields[key]?.display ?? "-").textSelection(.enabled)
                    .multilineTextAlignment(.trailing)
                }
              }
            }.padding(.top, 12).font(.caption)
          } label: {
            VStack(alignment: .leading, spacing: 5) {
              Text(event.model).lineLimit(1)
              let time = Date(timeIntervalSince1970: (event.number("ts_ms") ?? 0) / 1000)
              Text(
                "\(time.formatted(date: .abbreviated, time: .standard)) · \(Int(event.number("port") ?? 0)) · HTTP \(Int(event.number("status") ?? 0))"
              )
              .font(.caption).foregroundStyle(.secondary)
            }
          }.padding(16).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
        }
      }
    }
  }
  private func activityLabel(_ key: String) -> String {
    [
      "paddock.ttft_ms": "First token · ms", "paddock.decode_tok_s": "Decode · tok/s",
      "paddock.prefill_ms": "Prefill · ms", "paddock.decode_ms": "Decode · ms",
      "paddock.queue_ms": "Queue · ms", "gen_ai.usage.input_tokens": "Input tokens",
      "gen_ai.usage.output_tokens": "Output tokens", "status": "HTTP status", "port": "Port",
    ][key] ?? key
  }
}

private struct CacheInsights: View {
  let snapshot: CacheSnapshot
  let port: UInt16?
  var body: some View {
    let servers = snapshot.servers.filter { port == nil || $0.port == port }
    if servers.isEmpty {
      ContentUnavailableView(
        "No active KV offloading", systemImage: "internaldrive",
        description: Text("Enable KV offloading in an instance's settings to inspect its cache."))
    }
    ForEach(servers) { server in
      SettingsGroup(title: "\(server.model ?? "Instance") · \(server.port)") {
        if server.tier["tripped"] == .bool(true) {
          Label("Cache circuit breaker is open", systemImage: "exclamationmark.triangle")
            .foregroundStyle(PaddockStyle.caution)
        }
        HStack {
          Text("Cache hit rate")
          Spacer()
          Text(server.hitRate.map { $0.formatted(.percent.precision(.fractionLength(1))) } ?? "-")
        }
        ForEach(
          [("RAM", "ram_ready", "ram_capacity"), ("SSD", "disk_ready", "disk_capacity")], id: \.0
        ) { name, used, capacity in
          if let u = server.number(used), let c = server.number(capacity), c > 0 {
            VStack(alignment: .leading, spacing: 8) {
              HStack {
                Text(name)
                Spacer()
                Text("\(bytes(u)) / \(bytes(c))").monospacedDigit()
              }
              ProgressView(value: min(u, c), total: c).tint(.primary)
            }
          }
        }
        DisclosureGroup("Cache decisions & I/O") {
          VStack(spacing: 10) {
            ForEach(server.tier.keys.sorted(), id: \.self) { field in
              HStack {
                Text(field.replacingOccurrences(of: "_", with: " ").capitalized).foregroundStyle(
                  .secondary)
                Spacer()
                Text(server.tier[field]?.display ?? "-").monospacedDigit()
              }
            }
          }.font(.caption).padding(.top, 12)
        }
      }
    }
  }
  private func bytes(_ value: Double) -> String {
    ByteCountFormatter.string(
      fromByteCount: Int64(max(0, min(value, Double(Int64.max - 1024)))), countStyle: .binary)
  }
}
