import AppKit
import PaddockUI
import SwiftUI

@main
struct PaddockMacApp: App {
  @NSApplicationDelegateAdaptor(AppLifecycle.self) private var lifecycle
  @Environment(\.openWindow) private var openWindow
  private var workspace: WorkspaceModel { lifecycle.workspace }

  var body: some Scene {
    Window("Paddock", id: "workspace") {
      WorkspaceView(
        model: workspace,
        applicationSettings: AnyView(
          DesktopSettingsView(
            preferences: lifecycle.preferences, notifications: lifecycle.notifications,
            embedded: true))
      )
      .onAppear {
        lifecycle.connect(
          openWindow: { openWindow(id: "workspace") },
          settings: { lifecycle.open(.settings) })
      }
    }
    .defaultSize(width: 1200, height: 820)
    .windowStyle(.hiddenTitleBar)
    .windowToolbarStyle(.unifiedCompact(showsTitle: false))
    .windowResizability(.contentMinSize)
    .commands {
      CommandGroup(replacing: .newItem) {
        Button("New Chat") { lifecycle.open(.newChat) }.keyboardShortcut("n")
        Button("New Question…") { lifecycle.showQuestion() }.keyboardShortcut(
          "n", modifiers: [.command, .shift])
      }
      CommandGroup(replacing: .appSettings) {
        Button("Settings…") { lifecycle.open(.settings) }.keyboardShortcut(",")
      }
      StudioRendererCommands(model: workspace)
      StudioHistoryCommands(model: workspace)
    }
  }
}

@MainActor
final class AppLifecycle: NSObject, NSApplicationDelegate {
  let workspace = WorkspaceModel()
  let preferences = DesktopPreferences()
  let notifications = DesktopNotifications()
  let question = QuickQuestionController()
  private var status: DesktopStatusController?
  private var openWindow: (() -> Void)?
  private var pendingAction: DesktopAction?
  private var terminating = false

  func applicationWillFinishLaunching(_ notification: Notification) {
    DesktopUpdater.shared.readyToInstall = { [weak self] in
      guard let self, self.workspace.snapshot != nil else { return false }
      return self.workspace.managedRunners.isEmpty && !self.workspace.operationInProgress
        && !self.workspace.maintenanceInProgress
        && !self.workspace.studioNeedsQuitConfirmation && !self.workspace.downloads.active
        && !self.question.draft.hasContent
    }
    notifications.open = { [weak self] in self?.open($0) }
    notifications.isVisible = { [weak self] in
      self?.workspace.isDesktopDestinationVisible($0) == true
    }
    workspace.onSnapshot = { [weak self] in self?.notifications.management($0) }
    workspace.onStudioState = { [weak self] in self?.notifications.studio($0) }
    preferences.onQuestion = { [weak self] in self?.showQuestion() }
    preferences.onMenuBarChange = { [weak self] in self?.status?.setVisible($0) }
  }

  func applicationDidFinishLaunching(_ notification: Notification) {
    preferences.start()
    // App lifetime, not window-task lifetime: closing a window mid-startup
    // must not cancel management initialization or discard its result.
    Task {
      await workspace.start()
      workspace.beginMonitoring()
    }
  }

  func connect(openWindow: @escaping () -> Void, settings: @escaping () -> Void) {
    self.openWindow = openWindow
    if status == nil {
      status = DesktopStatusController(
        workspace: workspace, open: { [weak self] in self?.open($0) },
        question: { [weak self] in self?.showQuestion(anchor: $0) }, settings: settings)
    }
    status?.setVisible(preferences.showMenuBar)
    if let pendingAction {
      self.pendingAction = nil
      open(pendingAction)
    }
  }
  func open(_ action: DesktopAction) {
    guard !terminating else { return }
    guard let openWindow else {
      pendingAction = action
      return
    }
    workspace.request(action)
    openWindow()
    NSApp.activate(ignoringOtherApps: true)
  }
  func selectWorkspace(_ mode: WorkspaceMode) {
    guard !terminating else { return }
    workspace.selectWorkspace(mode)
    openWindow?()
    NSApp.activate(ignoringOtherApps: true)
  }
  func showQuestion(anchor: NSRect? = nil) {
    guard !terminating else { return }
    question.show(workspace: workspace, anchor: anchor, open: open)
  }

  func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
    let menu = NSMenu()
    for (title, action): (String, Selector) in [
      ("New Question…", #selector(dockQuestion)), ("Open Paddock", #selector(dockStudio)),
      ("Settings…", #selector(dockSettings)),
    ] {
      let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
      item.target = self
      menu.addItem(item)
    }
    return menu
  }
  @objc private func dockQuestion() { showQuestion() }
  @objc private func dockStudio() { open(.studio) }
  @objc private func dockSettings() { open(.settings) }

  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

  func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
    guard !terminating else { return .terminateLater }
    if workspace.operationInProgress || workspace.desktopTransition
      || workspace.maintenanceInProgress
    {
      let alert = DesktopAlert.make()
      alert.messageText = "An operation is still in progress"
      alert.informativeText =
        workspace.maintenanceInProgress
        ? "Finish or cancel the benchmark, and allow any backup export to complete before quitting. You can close the window while Paddock finishes."
        : "Keep Paddock open until the model operation or settings save finishes. You can close its window; the menu bar remains available."
      alert.addButton(withTitle: "Keep Paddock Open")
      alert.runModal()
      return .terminateCancel
    }
    if workspace.studioNeedsQuitConfirmation || question.draft.hasContent {
      let alert = DesktopAlert.make()
      alert.messageText = "Quit with Studio work in progress?"
      alert.informativeText =
        "Quitting stops active replies and discards unsent text, message edits, attachments, unsaved connection and tool settings, and artifact edits. Sent messages are saved. Closing the window instead keeps your work available."
      alert.addButton(withTitle: "Keep Paddock Open")
      alert.addButton(withTitle: "Quit Paddock")
      guard alert.runModal() == .alertSecondButtonReturn else { return .terminateCancel }
    }
    let runners = workspace.managedRunners
    if workspace.downloads.active {
      let alert = DesktopAlert.make()
      alert.messageText = "Pause downloads and quit?"
      alert.informativeText =
        "Partial model files are kept. Reopen Paddock and choose Resume in Downloads to continue. Closing the window instead keeps downloading."
      alert.addButton(withTitle: "Keep Paddock Open")
      alert.addButton(withTitle: "Pause Downloads and Quit")
      guard alert.runModal() == .alertSecondButtonReturn else { return .terminateCancel }
    }
    var stopModels = false
    if !runners.isEmpty {
      let alert = DesktopAlert.make()
      alert.messageText =
        "Quit with \(runners.count) model\(runners.count == 1 ? "" : "s") running?"
      let installingUpdate = DesktopUpdater.shared.hasPendingInstallation
      alert.informativeText =
        installingUpdate
        ? "Models must stop before Paddock can install an update. Requests drain for up to 30 seconds per model; longer requests may be interrupted."
        : "Keeping models running leaves their API endpoints available to external clients, but Paddock monitoring and notifications stop. Stop Models and Quit drains requests for up to 30 seconds per model; longer requests may be interrupted."
      alert.addButton(withTitle: "Cancel")
      if !installingUpdate { alert.addButton(withTitle: "Keep Models Running and Quit") }
      alert.addButton(withTitle: "Stop Models and Quit")
      let reply = alert.runModal()
      if reply == .alertFirstButtonReturn { return .terminateCancel }
      stopModels = installingUpdate || reply == .alertThirdButtonReturn
    }
    terminating = true
    workspace.quitting = true
    Task {
      guard await workspace.saveTablesBeforeQuit() else {
        terminating = false
        workspace.quitting = false
        sender.reply(toApplicationShouldTerminate: false)
        return
      }
      if stopModels, !(await workspace.stopForQuit(runners)) {
        terminating = false
        workspace.quitting = false
        sender.reply(toApplicationShouldTerminate: false)
        open(.manager)
        return
      }
      await workspace.shutdown()
      sender.reply(toApplicationShouldTerminate: true)
    }
    return .terminateLater
  }

  func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool
  {
    // A Quick Question panel or Settings window must not prevent Dock reopen.
    // Restore the existing main-window destination rather than forcing Studio.
    openWindow?()
    sender.activate(ignoringOtherApps: true)
    return true
  }
}
