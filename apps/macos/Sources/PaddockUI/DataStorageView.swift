import AppKit
import Observation
import PaddockClient
import SwiftUI

/// Own the export receipt beyond the view's lifetime. Navigation must not
/// abandon a large backup or enable a second concurrent export.
@MainActor @Observable final class DataStorageModel {
  @ObservationIgnored private var task: Task<Void, Never>?
  let client: any ManagerLoading
  private(set) var exporting = false
  private(set) var error: String?
  private(set) var backupURL: URL?
  private(set) var receipt: NativeExportReceipt?
  init(client: any ManagerLoading) { self.client = client }

  func chooseExport() {
    guard !exporting else { return }
    let panel = NSSavePanel()
    panel.nameFieldStringValue =
      "Paddock-\(Date().formatted(.iso8601.year().month().day().dateSeparator(.dash)))-backup.sqlite"
    panel.canCreateDirectories = true
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      self.export(to: url)
    }
  }
  func export(to url: URL) {
    guard !exporting else { return }
    exporting = true
    error = nil
    receipt = nil
    backupURL = nil
    task = Task {
      defer {
        exporting = false
        task = nil
      }
      do {
        receipt = try await client.inspect(
          .backup(path: url.path), as: NativeExportReceipt.self, timeout: .seconds(300))
        backupURL = url
      } catch { self.error = error.localizedDescription }
    }
  }
  func settle() async { await task?.value }
}

struct DataStorageView: View {
  @Bindable var model: DataStorageModel
  let snapshot: ManagerSnapshot
  @State private var inventory: StorageInventory?
  @State private var inventoryError: String?
  var body: some View {
    SettingsPage(title: "Data & storage") { _ in
      SettingsGroup(title: "Conversation backup") {
        Text(
          "Conversations, attachments, artifacts and preferences. API keys and connector credentials are excluded."
        )
        .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        HStack {
          Button(
            model.exporting ? "Exporting…" : "Export backup…", systemImage: "square.and.arrow.up"
          ) { model.chooseExport() }.disabled(model.exporting)
          if model.exporting { ProgressView().controlSize(.small) }
          if let backupURL = model.backupURL, let receipt = model.receipt {
            Button("Show in Finder") {
              NSWorkspace.shared.activateFileViewerSelecting([backupURL])
            }
            Text(DisplayFormat.bytes(receipt.bytes)).foregroundStyle(.secondary)
          }
        }
        if let error = model.error {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
        }
      }
      if let registry = snapshot.identity.registry {
        SettingsGroup(title: "Models") {
          HStack {
            Text("Available storage").foregroundStyle(.secondary)
            Spacer()
            Text(DisplayFormat.bytes(registry.diskFree)).monospacedDigit()
          }
          Text(registry.modelsDir).font(.caption).textSelection(.enabled).foregroundStyle(
            .secondary)
          Button("Open models folder", systemImage: "folder") {
            NSWorkspace.shared.open(URL(fileURLWithPath: registry.modelsDir, isDirectory: true))
          }
        }
      }
      if let inventoryError { Text(inventoryError).foregroundStyle(PaddockStyle.caution) }
      if let inventory {
        ForEach(inventory.artifacts) { artifact in
          SettingsGroup(title: artifact.model) {
            HStack {
              Text(artifact.artifact)
              Spacer()
              Text(DisplayFormat.bytes(artifact.bytes)).monospacedDigit()
            }
            if artifact.presentFiles != artifact.totalFiles {
              Text("\(artifact.presentFiles) of \(artifact.totalFiles) files downloaded")
                .foregroundStyle(.secondary)
            }
            HStack {
              if !artifact.configuredPorts.isEmpty {
                Text(
                  "Configured on \(artifact.configuredPorts.map(String.init).joined(separator: ", "))"
                ).font(.caption).foregroundStyle(.secondary)
              }
              Spacer()
              if let path = artifact.path {
                Button("Show files", systemImage: "folder") {
                  NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                }
              }
            }
          }
        }
      }
    }
    .task {
      do { inventory = try await model.client.inspect(.storage, as: StorageInventory.self) } catch {
        if !Task.isCancelled { inventoryError = error.localizedDescription }
      }
    }
  }
}
