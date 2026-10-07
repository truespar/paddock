import AppKit
import ServiceManagement
import SwiftUI

public struct DesktopSettingsView: View {
  @Bindable var preferences: DesktopPreferences
  @Bindable var notifications: DesktopNotifications
  var embedded: Bool

  public init(
    preferences: DesktopPreferences, notifications: DesktopNotifications, embedded: Bool = false
  ) {
    self.preferences = preferences
    self.notifications = notifications
    self.embedded = embedded
  }

  public var body: some View {
    DesktopSettingsSurface(embedded: embedded) {
      DesktopSettingsContent(
        preferences: preferences,
        notify: Binding(
          get: { notifications.effectiveEnabled },
          set: { value in
            if value {
              Task { await notifications.enable() }
            } else {
              notifications.enabled = false
            }
          }),
        sounds: $notifications.sounds, previews: $notifications.responsePreviews,
        notificationBusy: notifications.requesting,
        notificationBlocked: notifications.permissionBlocked,
        notificationError: notifications.error,
        openNotifications: { notifications.openSettings() })
      DesktopUpdateSettings()
    }
    .task {
      preferences.refreshLogin()
      await notifications.refreshAuthorization()
    }
    .onReceive(
      NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
    ) { _ in
      preferences.refreshLogin()
      Task { await notifications.refreshAuthorization() }
    }
  }
}

/// The same rows serve the embedded page and the Settings scene. System state
/// is supplied separately so layout tests cannot request OS permissions.
struct DesktopSettingsContent: View {
  @Bindable var preferences: DesktopPreferences
  @Binding var notify: Bool
  @Binding var sounds: Bool
  @Binding var previews: Bool
  var notificationBusy: Bool
  var notificationBlocked: Bool
  var notificationError: String?
  var openNotifications: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 28) {
      group("Startup & menu bar") {
        switchRow("Show in menu bar", isOn: $preferences.showMenuBar)
          .accessibilityIdentifier("application-menu-bar")
        VStack(alignment: .leading, spacing: 10) {
          switchRow(
            "Launch at login",
            isOn: Binding(
              get: { preferences.loginEnabled },
              set: { value in Task { await preferences.setLogin(value) } })
          ).disabled(preferences.loginBusy)
            .accessibilityIdentifier("application-login")
          if preferences.loginNeedsApproval {
            Button("Allow in Login Items…") { SMAppService.openSystemSettingsLoginItems() }
              .accessibilityIdentifier("application-login-approval")
          }
          if let error = preferences.loginError { errorText(error) }
        }
      }
      group("Quick Question") {
        switchRow(
          "Keyboard shortcut",
          isOn: Binding(
            get: { preferences.shortcutEnabled },
            set: { preferences.setShortcutEnabled($0) })
        ).accessibilityIdentifier("application-shortcut-toggle")
        if preferences.shortcutEnabled {
          HStack {
            Text("Open Quick Question").foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Dropdown(
              title: "Quick Question shortcut",
              value: QuestionHotKey.choices.first { $0.0 == preferences.shortcut }?.1
                ?? "Choose shortcut"
            ) {
              ForEach(QuestionHotKey.choices, id: \.0) { choice in
                Button(choice.1) { preferences.configureShortcut(choice.0) }
              }
            }.accessibilityIdentifier("application-shortcut-key")
          }
        }
        if let error = preferences.shortcutError { errorText(error) }
      }
      group("Notifications") {
        switchRow("Notify about background work", isOn: $notify)
          .disabled(notificationBusy)
          .accessibilityIdentifier("application-notifications")
        if notificationBlocked {
          Button("Allow in Notification Settings…", action: openNotifications)
            .accessibilityIdentifier("application-notification-permission")
        }
        switchRow("Play sounds", isOn: $sounds)
          .disabled(!notify || notificationBusy)
          .accessibilityIdentifier("application-notification-sounds")
        switchRow("Response previews", isOn: $previews)
          .disabled(!notify || notificationBusy)
          .accessibilityIdentifier("application-notification-previews")
        if let error = notificationError { errorText(error) }
      }
    }
  }

  private func group<Content: View>(_ title: String, @ViewBuilder content: @escaping () -> Content)
    -> some View
  {
    SettingsGroup(title: title, content: content)
  }

  private func switchRow(_ title: String, isOn: Binding<Bool>) -> some View {
    SettingsRow(title: title, compact: true) {
      Toggle(title, isOn: isOn).labelsHidden().toggleStyle(.switch).controlSize(.small)
    }
  }

  private func errorText(_ text: String) -> some View {
    Text(text).font(.caption).foregroundStyle(PaddockStyle.caution)
      .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
  }
}

struct DesktopSettingsSurface<Content: View>: View {
  @AppStorage("workspaceAppearance") private var appearance: WorkspaceAppearance = .system
  var embedded = false
  @ViewBuilder var content: Content

  var body: some View {
    if embedded {
      page.background(PaddockStyle.canvas)
    } else {
      page.frame(width: 530, height: 620)
        .desktopWindowSurface(appearance: appearance)
        .toolbarBackground(PaddockStyle.canvas, for: .windowToolbar)
        .toolbarBackgroundVisibility(.visible, for: .windowToolbar)
        .preferredColorScheme(appearance.colorScheme)
    }
  }

  private var page: some View {
    SettingsPage(title: "Application") { _ in content }
  }
}
