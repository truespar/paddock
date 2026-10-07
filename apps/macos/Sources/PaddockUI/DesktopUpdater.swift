import AppKit
import Observation
import Sparkle
import SwiftUI

/// The standard signed updater owns download, validation, replacement and
/// relaunch. Unconfigured development builds never fetch an unsigned feed.
@MainActor @Observable public final class DesktopUpdater: NSObject, SPUUpdaterDelegate {
  public static let shared = DesktopUpdater()
  public private(set) var available = false
  public private(set) var canCheck = false
  public private(set) var error: String?
  public private(set) var hasPendingInstallation = false
  public var automaticChecks = false {
    didSet { controller?.updater.automaticallyChecksForUpdates = automaticChecks }
  }
  @ObservationIgnored public var readyToInstall: () -> Bool = { false }
  @ObservationIgnored private var installHandler: (() -> Void)?
  @ObservationIgnored private var controller: SPUStandardUpdaterController?
  @ObservationIgnored private var observation: NSKeyValueObservation?
  @ObservationIgnored private var automaticObservation: NSKeyValueObservation?

  public init(bundle: Bundle = .main) {
    super.init()
    guard Self.validConfiguration(bundle.infoDictionary ?? [:]) else { return }
    let controller = SPUStandardUpdaterController(
      startingUpdater: false, updaterDelegate: self, userDriverDelegate: nil)
    self.controller = controller
    do {
      try controller.updater.start()
      available = true
      automaticChecks = controller.updater.automaticallyChecksForUpdates
      canCheck = controller.updater.canCheckForUpdates
      observation = controller.updater.observe(\.canCheckForUpdates, options: [.new]) {
        [weak self] _, change in
        let value = change.newValue ?? false
        Task { @MainActor [weak self] in self?.canCheck = value }
      }
      automaticObservation = controller.updater.observe(
        \.automaticallyChecksForUpdates, options: [.new]
      ) { [weak self] _, change in
        let value = change.newValue ?? false
        Task { @MainActor [weak self] in
          if self?.automaticChecks != value { self?.automaticChecks = value }
        }
      }
    } catch { self.error = error.localizedDescription }
  }

  static func validConfiguration(_ info: [String: Any]) -> Bool {
    guard let raw = info["SUFeedURL"] as? String, let url = URL(string: raw),
      url.scheme == "https", url.host == "github.com", url.user == nil, url.password == nil,
      url.port == nil, url.query == nil, url.fragment == nil,
      url.path == "/truespar/paddock/releases/latest/download/appcast.xml",
      info["SURequireSignedFeed"] as? Bool == true,
      info["SUVerifyUpdateBeforeExtraction"] as? Bool == true,
      let key = info["SUPublicEDKey"] as? String, Data(base64Encoded: key)?.count == 32
    else { return false }
    return true
  }
  public func updater(_ updater: SPUUpdater, willInstallUpdate item: SUAppcastItem) {
    hasPendingInstallation = true
  }
  public func updater(_ updater: SPUUpdater, didAbortWithError error: Error) {
    hasPendingInstallation = false
    installHandler = nil
    // Sparkle presents its own error/no-update result. Do not leave stale
    // restart controls or duplicate its dialog in Application settings.
    self.error = nil
  }
  public func updater(
    _ updater: SPUUpdater, shouldPostponeRelaunchForUpdate item: SUAppcastItem,
    untilInvokingBlock installHandler: @escaping () -> Void
  ) -> Bool {
    hasPendingInstallation = true
    if readyToInstall() { return false }
    self.installHandler = installHandler
    error = "Finish active work and stop running models before installing the update."
    return true
  }
  public func installPending() {
    guard readyToInstall(), let installHandler else { return }
    self.installHandler = nil
    error = nil
    installHandler()
  }
  public func check() { if canCheck { controller?.checkForUpdates(nil) } }
}

struct DesktopUpdateSettings: View {
  @Bindable var updater = DesktopUpdater.shared
  var body: some View {
    SettingsGroup(title: "Updates") {
      HStack {
        Text(
          "Paddock \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")"
        ).foregroundStyle(.secondary)
        Spacer()
        if updater.available {
          Button("Check for updates…") { updater.check() }.disabled(!updater.canCheck)
        } else {
          Link(
            "View releases…",
            destination: URL(string: "https://github.com/truespar/paddock/releases")!)
        }
      }
      if updater.hasPendingInstallation {
        Button("Install & restart Paddock") { updater.installPending() }
      }
      if updater.available {
        SettingsRow(title: "Check for updates automatically", compact: true) {
          Toggle("Check for updates automatically", isOn: $updater.automaticChecks)
            .labelsHidden().toggleStyle(.switch).controlSize(.small)
        }
      }
      if let error = updater.error {
        Text(error).foregroundStyle(PaddockStyle.caution).font(.caption)
      }
    }
  }
}
