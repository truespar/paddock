import AppKit
import Foundation
import Observation
import PaddockClient
import PaddockConversationCore
import UniformTypeIdentifiers
import WebKit

/// App-lifetime native Studio. Swift owns state, networking, capture and UI.
/// Independent web views exist only for allowed document/graph viewers.
@MainActor @Observable
public final class StudioWorkspace: NSObject {
  @ObservationIgnored var runtime: NativeStudioRuntime?
  @ObservationIgnored var nativeTransport: NativeConversationTransport?
  @ObservationIgnored private var viewers: [NativeViewerRole: StudioViewerHost] = [:]
  @ObservationIgnored private var preparingGraph = false
  @ObservationIgnored var microphone: NativeMicrophoneSession?
  /// Compatibility for document diagnostics. Shipping slots select a role.
  public var webView: WKWebView { webView(for: .document) }
  public func webView(for role: NativeViewerRole) -> WKWebView {
    viewer(for: role).webView
  }
  private func viewer(for role: NativeViewerRole) -> StudioViewerHost {
    if let viewer = viewers[role] { return viewer }
    let value = StudioViewerHost(owner: self, role: role)
    viewers[role] = value
    Task { [weak self, weak value] in
      guard let self, let value else { return }
      do {
        let root = Bundle.main.resourceURL!.appending(path: "StudioWorkspace")
        let host = try await client.studio(assets: root)
        guard viewers[role] === value, !closed else { return }
        try await value.start(host: host)
        if viewers[role] === value, !value.preparedForAdmission { await updateViewer(role) }
      } catch {
        if viewers[role] === value, !closed { self.error = error.localizedDescription }
      }
    }
    return value
  }
  public var hasWebViewer: Bool { !viewers.isEmpty }
  public func hasWebViewer(for role: NativeViewerRole) -> Bool { viewers[role] != nil }
  var viewerWindow: NSWindow? {
    viewers[.document]?.webView.window ?? viewers[.graph]?.webView.window
  }
  private func closeViewer(_ role: NativeViewerRole) {
    viewers.removeValue(forKey: role)?.close()
  }
  private func closeViewers() {
    for role in NativeViewerRole.allCases { closeViewer(role) }
  }
  /// Native panels belong to the composer window even while the WebKit split
  /// item is collapsed and its view is temporarily outside the window tree.
  @ObservationIgnored public weak var presentationWindow: NSWindow?
  public private(set) var state: StudioState?
  @ObservationIgnored public var onPresentation: ((StudioState) -> Void)?
  public private(set) var ready = false
  /// A ready bridge may still be restoring routes, preferences or attachments.
  /// OS entry points wait for that restoration before creating another chat.
  public private(set) var restoring = false
  public internal(set) var error: String?
  public var messageEdit: StudioMessageEdit?
  public internal(set) var messageMutation = false
  public internal(set) var messageNavigationAnchor: String?
  public var attachments: [StudioAttachment] = []
  public var selectedArtifactId: String?
  /// Public origin only, never the private session descriptor/cookie. HTML
  /// previews use their own ephemeral WebKit store and credential-free shells.
  public var artifactPreviewOrigin: URL? { host?.origin }
  public var artifactPicks: [String: String] = [:]
  /// View-local dismissals, scoped to the conversation being inspected. Keep
  /// the stored artifacts and source drafts intact; Show can reopen each one.
  var dismissedArtifactIDs = Set<String>()
  public var presentedArtifacts: [StudioState.Artifact] {
    (state?.nativeArtifacts ?? []).filter { !dismissedArtifactIDs.contains($0.id) }
  }
  public struct ArtifactDraft {
    public let saved: String
    public var text: String
    public init(saved: String) {
      self.saved = saved
      self.text = saved
    }
  }
  public var artifactDrafts: [String: ArtifactDraft] = [:]
  public var hasArtifactEdits: Bool { artifactDrafts.values.contains { $0.saved != $0.text } }
  public let audioPlayback = StudioAudioPlayback()
  public let audioMedia = StudioAudioMedia()
  public let documentMedia = StudioDocumentMedia()
  public let microphoneMeter = StudioMicrophoneMeter()
  public internal(set) var microphoneStarting = false
  @ObservationIgnored var captureRequested = false
  @ObservationIgnored var microphoneEpoch = 0
  var dictationSession = ""
  var dictatedThrough = -1
  public var microphoneBusy: Bool { microphoneStarting || state?.audio?.busy == true }
  public var onFiles: (([URL]) -> Void)?
  public var conversation: StudioState.Conversation? { state?.conversation }
  public var history: [StudioState.History] { state?.history ?? [] }
  public var busy: Bool {
    sending || messageMutation || microphoneStarting || (ready && state?.busy == true)
  }
  public var uploading: Bool { !uploads.isEmpty || !incomingDrops.isEmpty }
  private var incomingDrops: [UUID: Task<Void, Never>] = [:]
  public var hasAttachments: Bool { !attachments.isEmpty }
  public private(set) var historyMutations = Set<String>()
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var host: StudioHost?
  @ObservationIgnored private var startTask: Task<Void, Never>?
  @ObservationIgnored private let presentationDecoder = StudioPresentationDecoder()
  @ObservationIgnored private var uploads: [String: Task<Void, Never>] = [:]
  @ObservationIgnored private var network = URLSession(
    configuration: .ephemeral, delegate: NoRedirects(), delegateQueue: nil)
  private var sending = false
  private var closed = false
  private var dark = false
  private var composerInset: Double = 0
  private var receivedRevision: UInt64 = 0
  private var recovery: StudioState?

  public init(client: any ManagerLoading) {
    self.client = client
    super.init()
  }
  public func start(assets: URL? = nil) async {
    guard !closed, !ready else { return }
    if let startTask {
      await startTask.value
      return
    }
    let task = Task { await boot(assets: assets) }
    startTask = task
    await task.value
    startTask = nil
  }
  private func boot(assets: URL?) async {
    guard !closed, !Task.isCancelled else { return }
    restoring = true
    defer { restoring = false }
    do {
      if runtime == nil {
        let value = try await client.nativeConversationHost()
        try Task.checkCancellation()
        guard !closed else { return }
        guard Self.validHost(value) else { throw ManagerError.core("Invalid private native host") }
        host = value
        let transport = try NativeConversationTransport(host: value)
        nativeTransport = transport
        let core = NativeStudioRuntime(
          transport: transport,
          prepareGraph: { [weak self] fields in
            guard let self else { throw CancellationError() }
            return try await self.prepareGraph(fields)
          },
          rasterPDF: { bytes, part in try await NativePDFInput.shared.parts(bytes, metadata: part)
          },
          openPDF: { bytes in try await NativePDFInput.shared.source(bytes) },
          publish: { [weak self] fields in
            await self?.receiveNative(fields)
          })
        runtime = core
        do { try await core.start() } catch {
          await core.close()
          runtime = nil
          nativeTransport = nil
          throw error
        }
      } else {
        _ = try await runtime?.command("refresh")
      }
      try Task.checkCancellation()
      guard !closed else { return }
      guard state != nil else {
        throw ManagerError.core(error ?? "The native Studio could not restore its presentation")
      }
      ready = true
      error = nil
      NSLog(
        "Paddock native Studio ready: %d model choices; hidden web workspace: none",
        state?.models.count ?? 0)
    } catch {
      if !closed, !Task.isCancelled { self.error = error.localizedDescription }
    }
  }
  private func receiveNative(_ fields: [String: ConversationValue]) async {
    guard !closed else { return }
    do {
      let value = try await presentationDecoder.decode(fields)
      apply(value)
    } catch {
      guard !closed, !(error is CancellationError) else { return }
      self.error = "Invalid native presentation: \(error)"
      NSLog("Paddock native presentation failed: %@", String(describing: error))
    }
  }
  private func updateViewer(_ role: NativeViewerRole) async {
    guard !(role == .graph && preparingGraph), let viewer = viewers[role], let runtime else {
      return
    }
    let state = await runtime.viewerState()
    guard viewers[role] === viewer else { return }
    do { try await viewer.update(state, dark: dark) } catch {
      if viewers[role] === viewer { self.error = error.localizedDescription }
    }
  }
  private func updateViewers() async {
    // A loading Traverse must not delay an independent document/theme update.
    async let document: Void = updateViewer(.document)
    async let graph: Void = updateViewer(.graph)
    _ = await (document, graph)
  }
  private func prepareGraph(_ fields: [String: ConversationValue]) async throws -> String {
    guard !closed, !Task.isCancelled else { throw CancellationError() }
    preparingGraph = true
    defer { preparingGraph = false }
    let viewer = viewer(for: .graph)
    viewer.preparedForAdmission = true
    try await viewer.waitUntilReady()
    guard viewers[.graph] === viewer else { throw CancellationError() }
    return try await viewer.update(fields, dark: dark)
  }
  public static func validHost(_ host: StudioHost) -> Bool {
    host.isValidPrivateHost
  }
  public func isLocal(_ url: URL?) -> Bool {
    guard let url, let origin = host?.origin else { return false }
    return url.scheme == origin.scheme && url.host == origin.host && url.port == origin.port
  }
  func apply(_ value: StudioState) {
    guard !closed, value.version == 1, value.revision > receivedRevision else { return }
    receivedRevision = value.revision
    if value.conversation?.id != state?.conversation?.id {
      if state?.conversation?.id != nil { closeViewers() }
      selectedArtifactId = nil
      artifactPicks = [:]
      dismissedArtifactIDs = []
      audioMedia.reset()
      documentMedia.reset()
    }
    if value.conversation?.id != state?.conversation?.id || value.audio?.busy == true {
      audioPlayback.reset()
    } else if let id = audioPlayback.clipId,
      value.nativeAudioPreview?.id != id,
      !(value.nativeTranscript?.messages.contains {
        $0.audioClips?.contains { $0.id == id } == true
      } ?? false)
    {
      audioPlayback.reset()
    }
    state = value
    dismissedArtifactIDs.formIntersection((value.nativeArtifacts ?? []).map(\.id))
    if value.nativeArtifactsPaneOpen != false, value.nativeGraph?.visible != true {
      let artifacts = presentedArtifacts
      if !artifacts.contains(where: { $0.id == selectedArtifactId }) {
        selectedArtifactId =
          artifacts.first(where: { ["html", "svg"].contains($0.kind) })?.id ?? artifacts.first?.id
      }
    } else {
      selectedArtifactId = nil
    }
    if !["pdf", "docx"].contains(value.nativeDocument?.kind ?? "") { closeViewer(.document) }
    // Keep the model's graph bridge alive through a turn even if its panel
    // closes. Closing the left pane must never cancel that bridge.
    if !value.busy, !preparingGraph, value.nativeGraph?.visible != true { closeViewer(.graph) }
    if let audio = value.audio {
      if dictationSession != audio.session {
        dictationSession = audio.session
        dictatedThrough = -1
      }
      if let clip = audio.attachment, !attachments.contains(where: { $0.id == clip.attachmentId }) {
        attachments.append(
          .init(
            id: clip.attachmentId, name: clip.name, mime: clip.mime, size: clip.size ?? 0,
            phase: "Ready"))
      }
    }
    onPresentation?(value)
    if !value.error.isEmpty { error = value.error }
  }
  @discardableResult
  public func command(
    _ kind: String, _ payload: [String: StudioValue] = [:], id: String = UUID().uuidString
  ) async throws -> [String: StudioValue] {
    guard ready, !closed else {
      throw ManagerError.core("Wait for the content workspace to finish opening")
    }
    if hasMessageEdit && ["newChat", "open", "openEndpoint", "renderer", "send"].contains(kind) {
      throw ManagerError.core("Send or cancel the message edit first. Nothing was discarded.")
    }
    if hasMessageEdit, kind == "deleteChats",
      payload["ids"]?.array?.contains(.string(messageEdit!.target.conversationId)) == true
    {
      throw ManagerError.core("Send or cancel the message edit before deleting this conversation.")
    }
    if kind == "documentAction" {
      guard let viewer = viewers[.document] else {
        throw ManagerError.core("Open a document first")
      }
      try await viewer.action(payload["action"]?.text ?? "")
      return [:]
    }
    guard let runtime else { throw ManagerError.core("Native conversation core is not ready") }
    let native = try JSONDecoder().decode(
      [String: ConversationValue].self, from: JSONEncoder().encode(payload))
    if kind.hasPrefix("microphone") || kind == "dictationAck" {
      return try await audioCommand(kind, native)
    }
    let reply = try await runtime.command(kind, native, id: id)
    if ["preview", "openDocument", "graphArtifact", "graphPanel", "closePreview", "messageAction"]
      .contains(kind)
    {
      await updateViewers()
    }
    if ["newChat", "open", "openEndpoint"].contains(kind) { closeViewers() }
    return try JSONDecoder().decode([String: StudioValue].self, from: JSONEncoder().encode(reply))
  }
  public func perform(_ kind: String, _ payload: [String: StudioValue] = [:]) async {
    do {
      error = nil
      try await command(kind, payload)
    } catch { self.error = describe(error) }
  }
  public func refreshHistory() async {
    if ready { await perform("refresh") } else { await start() }
  }
  /// Acknowledged metadata operations; callers retain editors/selection on
  /// failure. The shared store owns hydration, write ordering and rollback.
  @discardableResult
  public func changeHistory(_ kind: String, ids: [String], title: String? = nil) async -> Bool {
    guard !ids.isEmpty, historyMutations.isDisjoint(with: ids) else { return false }
    if kind != "generateTitle" { historyMutations.formUnion(ids) }
    defer { if kind != "generateTitle" { historyMutations.subtract(ids) } }
    var payload: [String: StudioValue] =
      kind == "deleteChats"
      ? ["ids": .array(ids.map(StudioValue.string))] : ["id": .string(ids[0])]
    if let title { payload["title"] = .string(title) }
    do {
      error = nil
      try await command(kind, payload)
      return true
    } catch {
      self.error = describe(error)
      return false
    }
  }
  public func newChat() async {
    guard !busy, !uploading else { return }
    do {
      try await command("newChat")
      attachments = []
      error = nil
    } catch { self.error = describe(error) }
  }
  public func open(_ id: String) async {
    guard !busy, !uploading else { return }
    do {
      try await command("open", ["id": .string(id)])
      attachments = []
      error = nil
    } catch { self.error = describe(error) }
  }
  public func send(_ text: String) async -> Bool {
    if let issue = attachmentBudgetIssue(for: text) {
      error = issue
      return false
    }
    guard !busy, !uploading, attachments.allSatisfy({ $0.ready && $0.selectionError == nil }) else {
      return false
    }
    sending = true
    messageNavigationAnchor = nil
    defer { sending = false }
    error = nil
    let ids = Set(attachments.map(\.id))
    do {
      let result = try await command(
        "send", ["text": .string(text), "attachments": .array(attachments.map(\.choices))])
      guard result["accepted"]?.boolean == true else {
        throw ManagerError.core("The message was not accepted; your draft has been kept")
      }
      attachments.removeAll { ids.contains($0.id) }
      return true
    } catch {
      self.error = describe(error)
      return false
    }
  }
  public func cancel() async {
    if microphoneBusy { _ = await stopMicrophone(text: "") } else { await perform("stop") }
  }
  public func updateDraft(_ text: String) async {
    guard ready else { return }
    // Typing must never dismiss a server/upload error. No draft is persisted
    // by this command; only the shared context estimate is updated.
    do { try await command("draft", ["text": .string(text)]) } catch {
      self.error = describe(error)
    }
  }
  public func setDark(_ value: Bool) async {
    dark = value
    await updateViewers()
  }
  public func setComposerInset(_ height: Double) async {
    composerInset = height
    guard ready else { return }
    do { try await command("composerSize", ["height": .number(height)]) } catch {
      self.error = describe(error)
    }
  }
  public func reload() {
    // Never silently interrupt a response or discard an app-owned draft.
    guard !busy, !uploading else {
      error = "Stop the response before reloading content"
      return
    }
    guard state?.unsavedEdits != true, !hasMessageEdit, !hasArtifactEdits else {
      error =
        "Send or cancel message edits and save or revert artifact edits before reloading content"
      return
    }
    recovery = state
    ready = false
    error = nil
    Task { await start() }
  }
  public func addFiles(_ urls: [URL]) {
    guard ready, !closed, !busy else {
      error = "Wait for the current operation before attaching files"
      return
    }
    guard attachments.count + urls.count <= 32 else {
      error = "Attach at most 32 files per turn"
      return
    }
    // Two uploads at once; both file reads and transfer are off the main thread.
    let previous = Array(uploads.values)
    for (offset, url) in urls.enumerated() {
      let id = UUID().uuidString
      let mime =
        UTType(filenameExtension: url.pathExtension)?.preferredMIMEType
        ?? "application/octet-stream"
      attachments.append(
        .init(id: id, name: url.lastPathComponent, mime: mime, size: 0, phase: "Waiting"))
      let preceding = offset >= 2 ? uploads[attachments[attachments.count - 3].id] : nil
      uploads[id] = Task { [weak self] in
        for task in previous { await task.value }
        await preceding?.value
        guard let self, !Task.isCancelled else { return }
        await upload(url, id: id, mime: mime)
        uploads[id] = nil
      }
    }
  }
  /// Keep a drop in admission until its provider resolves: Send/navigation
  /// must not race the asynchronous delivery of a dragged screenshot.
  public func addDroppedItems(_ providers: [NSItemProvider]) -> Bool {
    let supported = providers.filter {
      $0.canLoadObject(ofClass: URL.self)
        || $0.hasItemConformingToTypeIdentifier(UTType.png.identifier)
        || $0.hasItemConformingToTypeIdentifier(UTType.tiff.identifier)
    }
    guard !supported.isEmpty else { return false }
    guard ready, !closed, !busy, incomingDrops.isEmpty, supported.count + attachments.count <= 32
    else {
      error = "Wait for the current operation, or attach at most 32 files per turn"
      return false
    }
    let id = UUID()
    incomingDrops[id] = Task { [weak self] in
      guard let self else { return }
      defer { incomingDrops[id] = nil }
      for provider in supported {
        guard !Task.isCancelled, !closed else { return }
        do {
          if provider.canLoadObject(ofClass: URL.self) {
            let url: URL = try await withCheckedThrowingContinuation { continuation in
              _ = provider.loadObject(ofClass: URL.self) { url, failure in
                if let url {
                  continuation.resume(returning: url)
                } else {
                  continuation.resume(
                    throwing: failure ?? ManagerError.core("The dropped file could not be opened"))
                }
              }
            }
            guard !Task.isCancelled, !closed else { return }
            addFiles([url])
          } else {
            let png = provider.hasItemConformingToTypeIdentifier(UTType.png.identifier)
            let data: Data = try await withCheckedThrowingContinuation { continuation in
              provider.loadDataRepresentation(
                forTypeIdentifier: (png ? UTType.png : .tiff).identifier
              ) { data, failure in
                if let data {
                  continuation.resume(returning: data)
                } else {
                  continuation.resume(
                    throwing: failure ?? ManagerError.core("The dropped image could not be opened"))
                }
              }
            }
            guard !Task.isCancelled, !closed else { return }
            addPastedImage(data, png: png)
          }
        } catch { self.error = describe(error) }
      }
    }
    return true
  }
  private func upload(_ url: URL, id: String, mime: String) async {
    let access = url.startAccessingSecurityScopedResource()
    defer { if access { url.stopAccessingSecurityScopedResource() } }
    do {
      guard url.isFileURL, let host else { throw ManagerError.core("Choose a local file") }
      let size = try await Task.detached {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true, let size = values.fileSize, size <= 100 * 1024 * 1024
        else { throw ManagerError.core("Choose a regular file up to 100 MiB") }
        return size
      }.value
      guard let index = attachments.firstIndex(where: { $0.id == id }) else { return }
      attachments[index] = .init(
        id: id, name: url.lastPathComponent, mime: mime, size: size, phase: "Uploading")
      var destination = URLComponents(
        url: host.origin.appending(path: "api/attachments/\(id)"), resolvingAgainstBaseURL: false)!
      destination.queryItems = [URLQueryItem(name: "name", value: url.lastPathComponent)]
      var request = URLRequest(url: destination.url!)
      request.httpMethod = "PUT"
      request.setValue(mime, forHTTPHeaderField: "Content-Type")
      request.setValue("\(host.cookieName)=\(host.session)", forHTTPHeaderField: "Cookie")
      let (_, response) = try await network.upload(for: request, fromFile: url)
      guard
        (response as? HTTPURLResponse)?.statusCode == 200
          || (response as? HTTPURLResponse)?.statusCode == 204
      else { throw ManagerError.core("The attachment upload failed") }
      try Task.checkCancellation()
      guard attachments.contains(where: { $0.id == id }), !closed else { return }
      var metadata: [String: StudioValue] = [
        "id": .string(id), "name": .string(url.lastPathComponent), "mime": .string(mime),
        "size": .number(Double(size)),
      ]
      metadata.merge(await NativeAttachmentMetadata.describe(url, mime: mime)) { _, new in new }
      let reply = try await command("stage", metadata)
      if let index = attachments.firstIndex(where: { $0.id == id }) {
        attachments[index].phase = "Ready"
        attachments[index].pages = reply["attachment"]?.object?["pages"]?.number.map(Int.init)
        let part = reply["attachment"]?.object
        attachments[index].width = part?["width"]?.number.map(Int.init)
        attachments[index].height = part?["height"]?.number.map(Int.init)
        if let thumbnail = part?["thumbUrl"]?.text,
          thumbnail.hasPrefix("data:image/jpeg;base64,"), thumbnail.utf8.count < 64 * 1024
        {
          attachments[index].thumbnail = Data(base64Encoded: String(thumbnail.dropFirst(23)))
        }
      }
    } catch {
      if let index = attachments.firstIndex(where: { $0.id == id }) {
        attachments[index].phase = "Failed"
        attachments[index].error = describe(error)
      }
    }
  }
  /// Clipboard pixels remain native data. A task-owned temporary original is
  /// streamed once to Rust, then removed even on cancellation or upload failure.
  public func addPastedImage(_ data: Data, png: Bool) {
    guard ready, !closed, !busy else {
      error = "Wait for the current operation before attaching an image"
      return
    }
    guard attachments.count < 32 else {
      error = "Attach at most 32 files per turn"
      return
    }
    guard data.count <= 100 * 1024 * 1024 else {
      error = "Paste an image up to 100 MiB"
      return
    }
    let id = UUID().uuidString
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Pasted-image-\(id).\(png ? "png" : "tiff")")
    let mime = png ? "image/png" : "image/tiff"
    let previous = Array(uploads.values)
    attachments.append(
      .init(id: id, name: url.lastPathComponent, mime: mime, size: data.count, phase: "Waiting"))
    uploads[id] = Task { [weak self] in
      guard let self else { return }
      defer { uploads[id] = nil }
      for task in previous { await task.value }
      guard !Task.isCancelled else { return }
      do {
        try await Task.detached { try data.write(to: url, options: .withoutOverwriting) }.value
        defer { try? FileManager.default.removeItem(at: url) }
        try Task.checkCancellation()
        await upload(url, id: id, mime: mime)
      } catch {
        if let i = attachments.firstIndex(where: { $0.id == id }) {
          attachments[i].phase = "Failed"
          attachments[i].error = describe(error)
        }
      }
    }
  }
  public func removeAttachment(_ id: String) {
    uploads[id]?.cancel()
    uploads[id] = nil
    attachments.removeAll { $0.id == id }
    Task { await perform("removeAttachment", ["id": .string(id)]) }
  }
  /// Downloads to a private temporary original, never into the conversation
  /// store. Playback and Save Original can fail without altering saved bytes.
  func downloadAudio(_ clip: StudioState.AudioClip) async throws -> URL {
    guard !closed, let host,
      clip.id.range(of: "^[a-zA-Z0-9_-]{1,128}$", options: .regularExpression) != nil,
      clip.mime.hasPrefix("audio/"),
      state?.nativeAudioPreview?.id == clip.id
        || state?.nativeTranscript?.messages.contains(where: {
          $0.audioClips?.contains(where: { $0.id == clip.id }) == true
        }) == true
    else { throw ManagerError.core("This audio clip is no longer in the displayed conversation") }
    var request = URLRequest(url: host.origin.appending(path: "api/attachments/\(clip.id)"))
    request.timeoutInterval = 60
    request.setValue("\(host.cookieName)=\(host.session)", forHTTPHeaderField: "Cookie")
    let (temporary, response) = try await network.download(for: request)
    defer { try? FileManager.default.removeItem(at: temporary) }
    try Task.checkCancellation()
    guard (response as? HTTPURLResponse)?.statusCode == 200,
      let size = try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize,
      size > 0, size <= 100 * 1024 * 1024
    else { throw ManagerError.core("The saved recording could not be downloaded") }
    let ext =
      switch clip.mime.components(separatedBy: ";")[0] {
      case "audio/mp4", "audio/x-m4a": "m4a"
      case "audio/webm": "webm"
      case "audio/ogg": "ogg"
      case "audio/mpeg": "mp3"
      case "audio/flac": "flac"
      default: "wav"
      }
    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Paddock-playback-\(UUID().uuidString).\(ext)")
    try FileManager.default.moveItem(at: temporary, to: destination)
    return destination
  }
  public func downloadOriginal(_ id: String) async throws -> URL {
    guard id.range(of: "^[a-zA-Z0-9_-]{1,128}$", options: .regularExpression) != nil,
      attachments.contains(where: { $0.id == id }) || state?.nativeDocument?.id == id
        || state?.nativeTranscript?.messages.contains(where: { m in
          m.attachments?.contains(where: { $0.id == id }) == true
            || m.pictures?.contains(where: { $0.id == id && !$0.preview }) == true
            || m.files?.contains(where: { $0.id == id && $0.stored }) == true
        }) == true
    else { throw ManagerError.core("This attachment is no longer on the displayed branch") }
    if let inline = state?.nativeTranscript?.messages.flatMap({ $0.pictures ?? [] })
      .first(where: { $0.id == id && !$0.preview })?.dataURL
    {
      return try await Task.detached(priority: .userInitiated) {
        guard inline.utf8.count <= 64 * 1024 * 1024,
          ["png", "jpeg", "webp"].contains(where: { inline.hasPrefix("data:image/\($0);base64,") }),
          let comma = inline.firstIndex(of: ","),
          let bytes = Data(base64Encoded: String(inline[inline.index(after: comma)...])),
          !bytes.isEmpty
        else { throw ManagerError.core("The saved image is invalid") }
        try Task.checkCancellation()
        let file = FileManager.default.temporaryDirectory.appending(
          path: "Paddock-preview-\(UUID().uuidString)")
        try bytes.write(to: file, options: [.atomic, .completeFileProtectionUnlessOpen])
        return file
      }.value
    }
    let request = try localRequest("api/attachments/\(id)")
    let (file, response) = try await network.download(for: request)
    defer { try? FileManager.default.removeItem(at: file) }
    try Task.checkCancellation()
    guard (response as? HTTPURLResponse)?.statusCode == 200,
      let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize,
      size > 0, size <= 100 * 1024 * 1024
    else { throw ManagerError.core("The saved attachment could not be downloaded") }
    let owned = FileManager.default.temporaryDirectory.appending(
      path: "Paddock-preview-\(UUID().uuidString)")
    try FileManager.default.moveItem(at: file, to: owned)
    return owned
  }
  func localRequest(_ path: String) throws -> URLRequest {
    guard !closed, let host else { throw ManagerError.core("The content workspace is closed") }
    var request = URLRequest(url: host.origin.appending(path: path))
    request.timeoutInterval = 60
    request.setValue("\(host.cookieName)=\(host.session)", forHTTPHeaderField: "Cookie")
    return request
  }
  func contentData(_ request: URLRequest) async throws -> Data {
    try await contentResponse(request).0
  }
  func contentResponse(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    let (file, response) = try await network.download(for: request)
    defer { try? FileManager.default.removeItem(at: file) }
    try Task.checkCancellation()
    if (response as? HTTPURLResponse)?.statusCode == 412 {
      throw ManagerError.core(
        "The artifact changed while you edited it. Your draft is retained; review the latest version before saving."
      )
    }
    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
      let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 16 * 1024 * 1024
    else { throw ManagerError.core("The content request failed") }
    return (try await Task.detached { try Data(contentsOf: file) }.value, http)
  }
  public func shutdown() async {
    // Close admission before the first suspension. A host request may ignore
    // cancellation; boot and presentation delivery also check this terminal flag.
    guard !closed else { return }
    closed = true
    ready = false
    restoring = false
    startTask?.cancel()
    await presentationDecoder.close()
    audioPlayback.reset()
    audioMedia.reset()
    documentMedia.reset()
    captureRequested = false
    microphoneEpoch += 1
    for task in incomingDrops.values { task.cancel() }
    incomingDrops.removeAll()
    for task in uploads.values { task.cancel() }
    uploads.removeAll()
    await microphone?.cancel()
    microphone = nil
    await runtime?.close()
    await startTask?.value
    startTask = nil
    runtime = nil
    nativeTransport = nil
    network.invalidateAndCancel()
    closeViewers()
  }
  public func reclaimCaches() async {
    audioMedia.reclaim()
    documentMedia.reclaim()
    await presentationDecoder.reclaim()
    await runtime?.reclaimPresentationCaches()
  }
  private func describe(_ error: any Error) -> String { error.localizedDescription }
}

private final class NoRedirects: NSObject, URLSessionTaskDelegate {
  func urlSession(
    _ session: URLSession, task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
    completionHandler: @escaping @Sendable (URLRequest?) -> Void
  ) { completionHandler(nil) }
}
