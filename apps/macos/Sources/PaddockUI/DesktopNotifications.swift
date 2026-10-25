import AppKit
import Observation
import PaddockClient
import PaddockStudio
import UserNotifications

@MainActor @Observable
public final class DesktopNotifications: NSObject, UNUserNotificationCenterDelegate {
  public private(set) var authorization: UNAuthorizationStatus?
  public var permissionBlocked: Bool { authorization == .denied }
  public var effectiveEnabled: Bool { enabled && Self.allowsDelivery(authorization) }
  public private(set) var error: String?
  public private(set) var requesting = false
  @ObservationIgnored private let delivery: any DesktopNotificationDelivering
  @ObservationIgnored private let defaults: UserDefaults
  @ObservationIgnored private var tracker = DesktopEventTracker()
  @ObservationIgnored public var open: ((DesktopAction) -> Void)?
  @ObservationIgnored public var isVisible: ((DesktopAction) -> Bool)?
  @ObservationIgnored private var pending: [String: (UUID, Task<Void, Never>)] = [:]
  @ObservationIgnored private var authorizationTask: Task<Void, Never>?
  @ObservationIgnored private var busyConversations = Set<String>()
  public var enabled: Bool {
    didSet {
      defaults.set(enabled, forKey: "desktopNotifications")
      if !enabled {
        cancelPending()
        delivery.removeDelivered(nil)
      }
    }
  }
  public var sounds: Bool { didSet { defaults.set(sounds, forKey: "desktopNotificationSounds") } }
  public var responsePreviews: Bool {
    didSet {
      defaults.set(responsePreviews, forKey: "desktopNotificationPreviews")
      if !responsePreviews {
        cancelPending()
        delivery.removeDelivered(nil)
      }
    }
  }

  public convenience init(defaults: UserDefaults = .standard) {
    let delivery = SystemNotificationDelivery()
    self.init(defaults: defaults, delivery: delivery)
    delivery.center.delegate = self
    let open = UNNotificationAction(
      identifier: "open", title: "Open Paddock", options: [.foreground])
    delivery.center.setNotificationCategories([
      UNNotificationCategory(
        identifier: "paddock.activity", actions: [open], intentIdentifiers: [],
        hiddenPreviewsBodyPlaceholder: "Open Paddock to view this activity.", options: [])
    ])
  }

  init(defaults: UserDefaults, delivery: any DesktopNotificationDelivering) {
    self.defaults = defaults
    // An absent preference is not a refusal. macOS permission still gates
    // every delivery; an explicit in-app opt-out remains off after upgrades.
    enabled =
      defaults.object(forKey: "desktopNotifications") == nil
      || defaults.bool(forKey: "desktopNotifications")
    sounds = defaults.bool(forKey: "desktopNotificationSounds")
    responsePreviews =
      defaults.object(forKey: "desktopNotificationPreviews") == nil
      || defaults.bool(forKey: "desktopNotificationPreviews")
    self.delivery = delivery
    super.init()
  }

  public func refreshAuthorization() async {
    authorization = await delivery.settings().authorization
  }

  static func allowsDelivery(_ authorization: UNAuthorizationStatus?) -> Bool {
    authorization == .authorized || authorization == .provisional
  }

  public func openSettings() {
    if let url = URL(string: "x-apple.systempreferences:com.apple.Notifications-Settings.extension")
    {
      NSWorkspace.shared.open(url)
    }
  }

  public func enable() async {
    guard !requesting else { return }
    requesting = true
    defer { requesting = false }
    do {
      error = nil
      await refreshAuthorization()
      if permissionBlocked {
        // Remember the user's choice; macOS still gates delivery. On return
        // from System Settings the switch reflects the refreshed permission.
        enabled = true
        openSettings()
        return
      }
      try await delivery.requestAuthorization()
      enabled = true
      await refreshAuthorization()
    } catch { self.error = error.localizedDescription }
  }

  public func management(_ value: PaddockClient.ManagerSnapshot) {
    deliver(tracker.management(value))
  }
  public func studio(_ value: PaddockStudio.StudioState) {
    if let id = value.conversation?.id ?? value.composerDraft?.id {
      if value.busy {
        if busyConversations.insert(id).inserted { prepareForBackgroundWork() }
      } else {
        busyConversations.remove(id)
      }
    }
    deliver(tracker.studio(value))
  }

  /// Called by actual work, never startup/history restoration. A completion
  /// waits for the same permission request, including very fast replies.
  func prepareForBackgroundWork() {
    guard enabled, authorizationTask == nil else { return }
    authorizationTask = Task { [weak self] in
      guard let self else { return }
      defer { authorizationTask = nil }
      await refreshAuthorization()
      guard enabled, authorization == .notDetermined, !requesting,
        !defaults.bool(forKey: "desktopNotificationsPrompted")
      else { return }
      defaults.set(true, forKey: "desktopNotificationsPrompted")
      requesting = true
      defer { requesting = false }
      do {
        error = nil
        try await delivery.requestAuthorization()
        await refreshAuthorization()
      } catch { self.error = error.localizedDescription }
    }
  }

  func deliver(_ events: [DesktopEvent]) {
    for event in events where enabled && isVisible?(event.route) != true {
      // Group approvals and compare lanes by conversation. Stable identifiers
      // replace the existing banner; they do not grow an unbounded history.
      let key = Self.identifier(event)
      pending[key]?.1.cancel()
      let generation = UUID()
      let task = Task { [weak self] in
        guard let self else { return }
        defer { if pending[key]?.0 == generation { pending[key] = nil } }
        prepareForBackgroundWork()
        await authorizationTask?.value
        let settings = await delivery.settings()
        guard !Task.isCancelled, enabled, isVisible?(event.route) != true,
          Self.allowsDelivery(settings.authorization)
        else { return }
        let content = UNMutableNotificationContent()
        content.title = event.title
        // macOS handles When Unlocked redaction. Never put a preview in the
        // title, identifiers or routing payload where it could bypass that.
        content.body =
          responsePreviews && settings.showPreviews != .never
          ? event.preview ?? event.body : event.body
        content.categoryIdentifier = "paddock.activity"
        content.threadIdentifier = key
        content.userInfo = Self.payload(event.route)
        if sounds && settings.soundEnabled { content.sound = .default }
        do {
          try await delivery.add(
            UNNotificationRequest(identifier: key, content: content, trigger: nil))
          if !enabled || (!responsePreviews && content.body != event.body) {
            delivery.removeDelivered([key])
          }
        } catch { self.error = "Could not deliver a notification: \(error.localizedDescription)" }
      }
      pending[key] = (generation, task)
    }
  }

  private func cancelPending() {
    for task in pending.values { task.1.cancel() }
    pending.removeAll()
    delivery.removePending()
  }

  func settle() async {
    await authorizationTask?.value
    for (_, task) in pending.values { await task.value }
  }

  func presentation(for route: DesktopAction) -> UNNotificationPresentationOptions {
    guard enabled, isVisible?(route) != true else { return [] }
    return sounds ? [.banner, .list, .sound] : [.banner, .list]
  }

  static func identifier(_ event: DesktopEvent) -> String {
    switch event.route {
    case .conversation(let id): "paddock.\(event.kind == .approval ? "approval" : "reply").\(id)"
    case .chat(let port): "paddock.model.\(port)"
    default: "paddock.models"
    }
  }
  static func payload(_ route: DesktopAction) -> [String: String] {
    switch route {
    case .conversation(let id): ["destination": "conversation", "id": id]
    case .chat(let port): ["destination": "model", "port": String(port)]
    default: ["destination": "manager"]
    }
  }
  nonisolated static func route(_ payload: [AnyHashable: Any]) -> DesktopAction? {
    switch payload["destination"] as? String {
    case "manager": return .manager
    case "model":
      guard let value = payload["port"] as? String, let port = UInt16(value), port >= 1024 else {
        return nil
      }
      return .chat(port: port)
    case "conversation":
      guard let id = payload["id"] as? String, !id.isEmpty, id.utf8.count <= 128,
        id.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_") })
      else { return nil }
      return .conversation(id)
    default: return nil
    }
  }

  public nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse
  ) async {
    guard [UNNotificationDefaultActionIdentifier, "open"].contains(response.actionIdentifier),
      let route = Self.route(response.notification.request.content.userInfo)
    else { return }
    await MainActor.run { open?(route) }
  }
  public nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, willPresent notification: UNNotification
  ) async -> UNNotificationPresentationOptions {
    guard let route = Self.route(notification.request.content.userInfo) else { return [] }
    return await MainActor.run {
      presentation(for: route)
    }
  }
}
