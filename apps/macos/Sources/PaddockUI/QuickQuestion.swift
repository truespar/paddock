import AppKit
import Observation
import PaddockClient
import PaddockStudio
import SwiftUI
import UniformTypeIdentifiers

/// Only a bounded unsent draft. The existing Studio owns uploads, page choices,
/// persistence and generation after an acknowledged handoff; never a second WK
/// workspace, conversation database, transport or inference engine.
@MainActor @Observable public final class QuickQuestionModel {
  var text = ""
  var modelID = ""
  var attachments: [QuickAttachment] = []
  var error: String?
  var transferring = false
  private var drops: [UUID: Task<Void, Never>] = [:]
  var loadingDrops: Bool { !drops.isEmpty }
  public var hasContent: Bool { !text.isEmpty || !attachments.isEmpty || loadingDrops }

  static func suggestedModel(_ state: StudioState?) -> String? {
    guard let state else { return nil }
    if let selected = state.selectedModels.first,
      state.models.contains(where: { $0.id == selected && $0.chat && $0.status == "ok" })
    {
      return selected
    }
    // No silent first-cloud-provider selection. A local fallback is account-free;
    // a cloud model must be explicitly selected or already selected in Studio.
    return state.models.first(where: { $0.port != nil && $0.chat && $0.status == "ok" })?.id
  }

  func addDroppedFiles(_ providers: [NSItemProvider]) -> Bool {
    let supported = providers.filter { $0.canLoadObject(ofClass: URL.self) }
    guard !supported.isEmpty else { return false }
    guard !transferring, !loadingDrops, supported.count + attachments.count <= 32 else {
      error = "Wait for the current attachment operation, or attach at most 32 files."
      return false
    }
    let id = UUID()
    drops[id] = Task { [weak self] in
      guard let self else { return }
      defer { drops[id] = nil }
      let deadline = ContinuousClock.now.advanced(by: .seconds(30))
      for provider in supported {
        do {
          let url = try await QuickDropLoad().load(provider, deadline: deadline)
          addFiles([url])
        } catch {
          self.error = error.localizedDescription
          break
        }
      }
    }
    return true
  }

  func addFiles(_ urls: [URL]) {
    guard !transferring else { return }
    guard urls.allSatisfy(\.isFileURL), urls.count + attachments.count <= 32 else {
      error = "Attach at most 32 local files."
      return
    }
    attachments.append(contentsOf: urls.map(QuickAttachment.init))
    error = nil
  }
  func addImage(_ data: Data, png: Bool) {
    guard !transferring else { return }
    let total = attachments.reduce(0) { $0 + ($1.image?.count ?? 0) }
    guard attachments.count < 32, data.count <= 16 * 1024 * 1024,
      total + data.count <= 32 * 1024 * 1024
    else {
      error =
        "Pasted images exceed the Quick Question limit. Attach the image file in Studio instead."
      return
    }
    attachments.append(QuickAttachment(image: data, png: png))
    error = nil
  }

  @discardableResult func handoff(to workspace: WorkspaceModel, openStudio: () -> Void) async
    -> Bool
  {
    guard !transferring, !loadingDrops, !workspace.desktopTransition, !workspace.quitting,
      hasContent
    else {
      return false
    }
    guard text.utf8.count <= 128 * 1024 else {
      error = "The question is too long (128 KiB limit)."
      return false
    }
    guard !workspace.desktopNavigationBlocked else {
      error =
        "Finish the active response, upload or edit in Studio first. Your question is kept here."
      return false
    }
    transferring = true
    workspace.desktopTransition = true
    error = nil
    defer {
      transferring = false
      workspace.desktopTransition = false
    }
    do {
      try await workspace.prepareDesktopChat()
      guard !workspace.desktopNavigationBlocked else {
        throw ManagerError.core("Finish the current Studio work first. Both drafts have been kept.")
      }
      try await workspace.chat.command("refresh")
      let selected = modelID
      guard !selected.isEmpty,
        workspace.chat.state?.models.contains(where: {
          $0.id == selected && $0.chat && $0.status == "ok"
        }) == true
      else {
        throw ManagerError.core(
          "Select a reachable chat model, or start one in Settings > Instances.")
      }
      try await workspace.chat.command("newChat")
      try await workspace.chat.command("models", ["ids": .array([.string(selected)])])
      workspace.draft.message = text
      // Ownership transfers before any send. If upload/generation fails, the
      // full Studio retains the draft and shows its established retry UI.
      let files = attachments
      let question = text
      text = ""
      attachments = []
      workspace.navigation.mode = .studio
      workspace.navigation.studio = .newChat
      openStudio()
      workspace.chat.addFiles(files.compactMap(\.url))
      for file in files {
        if let image = file.image { workspace.chat.addPastedImage(image, png: file.png) }
      }
      if files.isEmpty {
        if await workspace.chat.send(question) { workspace.draft.message = "" }
      } else {
        // PDFs and images are reviewed in the full composer, including page
        // selection and image detail, before the person authorizes a send.
        Task {
          while workspace.chat.uploading { try? await Task.sleep(for: .milliseconds(100)) }
          withExtendedLifetime(files) {}
        }
      }
      withExtendedLifetime(files) {}
      return true
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }
}

/// NSItemProvider callbacks are not guaranteed to arrive promptly or honor
/// cancellation. A main-actor, exactly-once receipt bounds admission and ignores
/// late callbacks without letting a provider retain the entire quick composer.
@MainActor final class QuickDropLoad {
  private var receipt: CheckedContinuation<URL, any Error>?
  private var timer: Task<Void, Never>?
  private var progress: Progress?
  func load(_ provider: NSItemProvider, deadline: ContinuousClock.Instant) async throws -> URL {
    try Task.checkCancellation()
    guard ContinuousClock.now < deadline else {
      throw ManagerError.core("The dropped file timed out. Try attaching it from Finder again.")
    }
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { receipt in
        self.receipt = receipt
        timer = Task { [weak self] in
          do { try await Task.sleep(until: deadline, clock: .continuous) } catch { return }
          self?.finish(
            .failure(
              ManagerError.core("The dropped file timed out. Try attaching it from Finder again.")))
        }
        progress = provider.loadObject(ofClass: URL.self) { [weak self] url, error in
          Task { @MainActor in
            // The timer and callback may both wait behind main-actor work.
            // Admission depends on the deadline, not their eventual queue order.
            if ContinuousClock.now >= deadline {
              self?.finish(
                .failure(
                  ManagerError.core(
                    "The dropped file timed out. Try attaching it from Finder again.")))
            } else if let url {
              self?.finish(.success(url))
            } else {
              self?.finish(
                .failure(error ?? ManagerError.core("The dropped file could not be read.")))
            }
          }
        }
      }
    } onCancel: {
      Task { @MainActor [weak self] in self?.finish(.failure(CancellationError())) }
    }
  }
  private func finish(_ result: Result<URL, any Error>) {
    guard let receipt else { return }
    self.receipt = nil
    timer?.cancel()
    timer = nil
    progress?.cancel()
    progress = nil
    receipt.resume(with: result)
  }
}

@MainActor final class QuickAttachment: Identifiable {
  let id = UUID()
  let url: URL?
  let image: Data?
  let png: Bool
  let scoped: Bool
  var name: String { url?.lastPathComponent ?? "Pasted image" }
  init(_ url: URL) {
    self.url = url
    image = nil
    png = false
    scoped = url.startAccessingSecurityScopedResource()
  }
  init(image: Data, png: Bool) {
    url = nil
    self.image = image
    self.png = png
    scoped = false
  }
  deinit { if scoped { url?.stopAccessingSecurityScopedResource() } }
}
