import AppKit
import SwiftUI

private final class WorkspaceToolbarAnchor: NSView {
  var onMoveToWindow: (() -> Void)?
  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    onMoveToWindow?()
  }
}

final class WorkspaceToolbarBackdrop: NSView {
  var sidebarWidth: CGFloat = 0 {
    didSet { if oldValue != sidebarWidth { needsDisplay = true } }
  }
  override init(frame: NSRect) {
    super.init(frame: frame)
    autoresizingMask = [.width, .height]
    setAccessibilityElement(false)
  }
  required init?(coder: NSCoder) { nil }
  override var isOpaque: Bool { true }
  override func hitTest(_ point: NSPoint) -> NSView? { nil }
  override func draw(_ dirtyRect: NSRect) {
    PaddockStyle.nsColor("canvas").setFill()
    dirtyRect.fill()
    PaddockStyle.nsColor("sidebar").setFill()
    NSRect(
      x: bounds.minX, y: bounds.minY, width: min(sidebarWidth, bounds.width), height: bounds.height
    )
    .intersection(dirtyRect).fill()
  }
  override func viewDidChangeEffectiveAppearance() {
    super.viewDidChangeEffectiveAppearance()
    needsDisplay = true
  }
}

extension EnvironmentValues {
  @Entry var workspaceWindowChrome: WorkspaceChromeState? = nil
}

/// One retained native split and toolbar own the window geometry. Content
/// does not create a second title bar or recreate the chat when a pane folds.
struct WorkspaceShell<Sidebar: View, Content: View>: NSViewControllerRepresentable {
  @Binding var navigation: WorkspaceNavigation
  var title: String
  var enabled = true
  var onNewChat: () -> Void
  var metrics: AnyView
  @ViewBuilder var sidebar: () -> Sidebar
  @ViewBuilder var content: () -> Content

  func makeNSViewController(context: Context) -> WorkspaceSplitController {
    let controller = WorkspaceSplitController()
    updateNSViewController(controller, context: context)
    return controller
  }

  func updateNSViewController(_ controller: WorkspaceSplitController, context: Context) {
    controller.onSidebar = { navigation.toggleSidebar() }
    controller.onNewChat = onNewChat
    controller.onBack = { navigation.returnToChat() }
    controller.onSettings = { navigation.showSettings() }
    controller.onWidth = { navigation.panelWidth = $0 }
    controller.update(
      navigation: navigation, title: title, enabled: enabled,
      sidebar: AnyView(
        WorkspacePaneSurface(chrome: controller.chrome, sidebar: true, content: AnyView(sidebar()))
          .environment(\.self, context.environment)),
      content: AnyView(
        WorkspacePaneSurface(chrome: controller.chrome, sidebar: false, content: AnyView(content()))
          .environment(\.self, context.environment)),
      metrics: AnyView(metrics.disabled(!enabled).environment(\.self, context.environment)))
  }

  func sizeThatFits(
    _ proposal: ProposedViewSize, nsViewController: WorkspaceSplitController, context: Context
  ) -> CGSize? {
    guard let width = proposal.width, let height = proposal.height,
      width.isFinite, height.isFinite
    else { return nil }
    return CGSize(width: max(0, width), height: max(0, height))
  }
}

@Observable @MainActor final class WorkspaceChromeState {
  var topInset: CGFloat = 0
}

private struct WorkspacePaneSurface: View {
  var chrome: WorkspaceChromeState
  var sidebar: Bool
  var content: AnyView

  var body: some View {
    content
      .padding(.top, sidebar ? chrome.topInset : 0)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .environment(\.workspaceWindowChrome, chrome)
      .environment(\.workspaceLeadingPaneInset, chrome.topInset)
      .environment(\.workspaceCatalogContentInset, chrome.topInset)
      .background(sidebar ? PaddockStyle.sidebar : PaddockStyle.canvas)
  }
}

struct WorkspaceDetailSurface<Content: View>: View {
  var edgeToEdge: Bool
  @ViewBuilder var content: () -> Content
  @Environment(\.workspaceLeadingPaneInset) private var topInset

  var body: some View {
    content().padding(.top, edgeToEdge ? 0 : topInset)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .background(PaddockStyle.canvas)
  }
}

@MainActor final class WorkspaceSplitController: NSSplitViewController, NSToolbarDelegate {
  let sidebarHost = NSHostingController(rootView: AnyView(EmptyView()))
  let contentHost = NSHostingController(rootView: AnyView(EmptyView()))
  let chrome = WorkspaceChromeState()
  private(set) var sidebarItem: NSSplitViewItem!
  private(set) var nativeToolbar: NSToolbar!
  private var navigation = WorkspaceNavigation()
  private var preferredWidth: CGFloat = 260
  private var needsPreferredWidth = true
  private var adjustingWidth = false
  private var insetQueued = false
  private var geometryQueued = false
  private var pendingTitle = ""
  private var windowObservation: NSKeyValueObservation?
  private var windowStyleObservation: NSKeyValueObservation?
  private weak var observedWindow: NSWindow?
  private let titleField = NSTextField(labelWithString: "")
  private let titleContainer = WorkspaceToolbarAnchor()
  private let fullscreenBackdrop = WorkspaceToolbarBackdrop()
  private let metricsHost = NSHostingView(rootView: AnyView(EmptyView()))
  private var buttons: [NSToolbarItem.Identifier: NSButton] = [:]
  private var toolbarEnabled = true
  private var dividerMonitor: Any?
  private var trackingDivider = false

  var onSidebar: (() -> Void)?
  var onNewChat: (() -> Void)?
  var onBack: (() -> Void)?
  var onSettings: (() -> Void)?
  var onWidth: ((CGFloat) -> Void)?

  private enum Item {
    static let sidebar = NSToolbarItem.Identifier("paddock.sidebar")
    static let newChat = NSToolbarItem.Identifier("paddock.new-chat")
    static let settings = NSToolbarItem.Identifier("paddock.settings")
    static let separator = NSToolbarItem.Identifier("paddock.sidebar-separator")
    static let title = NSToolbarItem.Identifier("paddock.title")
    static let metrics = NSToolbarItem.Identifier("paddock.metrics")
  }

  override func viewDidLoad() {
    super.viewDidLoad()
    splitView.isVertical = true
    splitView.dividerStyle = .thin
    splitView.setAccessibilityIdentifier("workspace-split")
    for host in [sidebarHost, contentHost] {
      host.sizingOptions = []
      host.safeAreaRegions = []
    }
    sidebarItem = NSSplitViewItem(viewController: sidebarHost)
    sidebarItem.minimumThickness = 220
    sidebarItem.maximumThickness = 360
    sidebarItem.holdingPriority = .init(260)
    sidebarItem.collapseBehavior = .preferResizingSiblingsWithFixedSplitView
    sidebarItem.canCollapse = false
    addSplitViewItem(sidebarItem)
    let contentItem = NSSplitViewItem(viewController: contentHost)
    contentItem.minimumThickness = WorkspacePanelMetrics.contentMinimum
    addSplitViewItem(contentItem)

    titleField.font = .systemFont(ofSize: 13, weight: .medium)
    titleField.lineBreakMode = .byTruncatingTail
    titleField.maximumNumberOfLines = 1
    titleField.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    titleField.setAccessibilityIdentifier("workspace-header-title")
    titleContainer.onMoveToWindow = { [weak self] in self?.refreshInset() }
    metricsHost.sizingOptions = []
    titleContainer.translatesAutoresizingMaskIntoConstraints = false
    titleContainer.addSubview(titleField)
    titleField.translatesAutoresizingMaskIntoConstraints = false
    metricsHost.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      // AppKit leaves about 8pt after the tracking separator. Give the title
      // a proper gutter instead of letting it hug the sidebar or its controls.
      titleField.leadingAnchor.constraint(equalTo: titleContainer.leadingAnchor, constant: 16),
      titleField.trailingAnchor.constraint(equalTo: titleContainer.trailingAnchor),
      titleField.topAnchor.constraint(equalTo: titleContainer.topAnchor),
      titleField.bottomAnchor.constraint(equalTo: titleContainer.bottomAnchor),
      titleField.widthAnchor.constraint(greaterThanOrEqualToConstant: 80),
      titleField.widthAnchor.constraint(lessThanOrEqualToConstant: 600),
      titleField.heightAnchor.constraint(equalToConstant: 20),
      metricsHost.widthAnchor.constraint(equalToConstant: 32),
      metricsHost.heightAnchor.constraint(equalToConstant: 32),
    ])
    nativeToolbar = NSToolbar(identifier: "paddock.workspace")
    nativeToolbar.delegate = self
    nativeToolbar.allowsUserCustomization = false
    nativeToolbar.autosavesConfiguration = false
    nativeToolbar.displayMode = .iconOnly
  }

  override func viewDidAppear() {
    super.viewDidAppear()
    scheduleGeometry()
  }

  override func viewDidLayout() {
    super.viewDidLayout()
    scheduleGeometry()
  }

  func update(
    navigation: WorkspaceNavigation, title: String, enabled: Bool,
    sidebar: AnyView, content: AnyView, metrics: AnyView
  ) {
    loadViewIfNeeded()
    if self.navigation.mode != navigation.mode || preferredWidth != navigation.panelWidth
      || self.navigation.showsSidebar != navigation.showsSidebar
    {
      needsPreferredWidth = true
    }
    self.navigation = navigation
    preferredWidth = navigation.panelWidth
    sidebarHost.rootView = sidebar
    contentHost.rootView = content
    metricsHost.rootView = metrics
    pendingTitle = title
    toolbarEnabled = enabled
    scheduleGeometry()
  }

  private func scheduleGeometry() {
    guard !geometryQueued else { return }
    geometryQueued = true
    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      geometryQueued = false
      attachWindow()
      let limits = WorkspacePanelMetrics.limits(navigation.mode)
      // Never force AppKit to measure SwiftUI hosts from inside a representable
      // update. This also coalesces repeated window and model-state updates.
      if sidebarItem.maximumThickness < limits.upperBound {
        sidebarItem.maximumThickness = limits.upperBound
      }
      if sidebarItem.minimumThickness != limits.lowerBound {
        sidebarItem.minimumThickness = limits.lowerBound
      }
      if sidebarItem.maximumThickness != limits.upperBound {
        sidebarItem.maximumThickness = limits.upperBound
      }
      if sidebarItem.isCollapsed == navigation.showsSidebar {
        sidebarItem.isCollapsed = !navigation.showsSidebar
      }
      if titleField.stringValue != pendingTitle {
        titleField.stringValue = pendingTitle
        titleField.toolTip = pendingTitle.isEmpty ? nil : pendingTitle
      }
      updateToolbar()
      applyPreferredWidth()
      refreshInset()
    }
  }

  private func attachWindow() {
    guard let window = view.window, observedWindow !== window else { return }
    observedWindow = window
    window.titleVisibility = .hidden
    updateTitlebarSurface(window)
    window.titlebarSeparatorStyle = .line
    window.toolbar = nativeToolbar
    updateToolbar()
    if dividerMonitor == nil {
      dividerMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .leftMouseUp]) {
        [weak self] event in
        MainActor.assumeIsolated { self?.trackDivider(event) }
        return event
      }
    }
    windowObservation = window.observe(\.contentLayoutRect, options: [.new]) { [weak self] _, _ in
      Task { @MainActor [weak self] in self?.refreshInset() }
    }
    windowStyleObservation = window.observe(\.styleMask, options: [.new]) { [weak self] _, _ in
      Task { @MainActor [weak self] in self?.refreshInset() }
    }
  }

  private func refreshInset() {
    guard !insetQueued else { return }
    insetQueued = true
    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      insetQueued = false
      let inset =
        view.window.map {
          updateTitlebarSurface($0)
          return max(0, view.convert(view.bounds, to: nil).maxY - $0.contentLayoutRect.maxY)
        } ?? 0
      if chrome.topInset != inset { chrome.topInset = inset }
    }
  }

  private func updateTitlebarSurface(_ window: NSWindow) {
    if !window.titlebarAppearsTransparent { window.titlebarAppearsTransparent = true }
    // AppKit moves toolbar items to a companion window in full screen. Its
    // transparent title bar has no workspace beneath it, so provide the same
    // opaque pane backgrounds behind the controls. Use the public view/window
    // relationship, without depending on private titlebar classes or subviews.
    guard window.styleMask.contains(.fullScreen), let toolbarWindow = titleContainer.window,
      toolbarWindow !== window, let surface = toolbarWindow.contentView
    else {
      fullscreenBackdrop.removeFromSuperview()
      return
    }
    if fullscreenBackdrop.superview !== surface {
      fullscreenBackdrop.removeFromSuperview()
      fullscreenBackdrop.frame = surface.bounds
      surface.addSubview(fullscreenBackdrop, positioned: .below, relativeTo: nil)
    }
    fullscreenBackdrop.sidebarWidth = sidebarItem.isCollapsed ? 0 : sidebarHost.view.frame.width
  }

  private func applyPreferredWidth() {
    guard !adjustingWidth, !sidebarItem.isCollapsed, splitView.bounds.width > 0 else { return }
    // Native divider tracking determines its width until mouse-up. Window
    // fitting may temporarily shrink it, but does not rewrite the preference.
    guard !trackingDivider else { return }
    let available = splitView.bounds.width - splitView.dividerThickness
    let desired = min(
      preferredWidth,
      max(
        sidebarItem.minimumThickness,
        available - WorkspacePanelMetrics.contentMinimum))
    guard needsPreferredWidth || abs(sidebarHost.view.frame.width - desired) > 0.5 else { return }
    adjustingWidth = true
    needsPreferredWidth = false
    splitView.setPosition(desired, ofDividerAt: 0)
    adjustingWidth = false
  }

  private func trackDivider(_ event: NSEvent) {
    guard event.window === view.window else { return }
    if event.type == .leftMouseDown, !sidebarItem.isCollapsed {
      let point = splitView.convert(event.locationInWindow, from: nil)
      trackingDivider =
        abs(point.x - sidebarHost.view.frame.maxX) <= 5 && splitView.bounds.contains(point)
    } else if event.type == .leftMouseUp, trackingDivider {
      rememberDividerWidth()
      trackingDivider = false
    }
  }

  override func viewDidDisappear() {
    super.viewDidDisappear()
    fullscreenBackdrop.removeFromSuperview()
    if let dividerMonitor { NSEvent.removeMonitor(dividerMonitor) }
    dividerMonitor = nil
    trackingDivider = false
    observedWindow = nil
    windowObservation = nil
    windowStyleObservation = nil
  }

  private func rememberDividerWidth() {
    guard !sidebarItem.isCollapsed else { return }
    preferredWidth = sidebarHost.view.frame.width
    onWidth?(preferredWidth)
  }

  private var itemIdentifiers: [NSToolbarItem.Identifier] {
    [Item.sidebar, Item.newChat]
      + (navigation.showsSidebar ? [Item.separator] : [Item.settings])
      + [Item.title, .flexibleSpace, Item.metrics]
  }

  private func updateToolbar() {
    guard observedWindow != nil else { return }
    let desired = itemIdentifiers
    if nativeToolbar.items.map(\.itemIdentifier) != desired {
      // Retain content and metrics hosts while changing only toolbar items.
      for index in nativeToolbar.items.indices.reversed() { nativeToolbar.removeItem(at: index) }
      for (index, id) in desired.enumerated() {
        nativeToolbar.insertItem(withItemIdentifier: id, at: index)
      }
    }
    for button in buttons.values { button.isEnabled = toolbarEnabled }
    let back = navigation.mode == .manager
    buttons[Item.newChat]?.image = NSImage(
      systemSymbolName: back ? "arrow.left" : "square.and.pencil", accessibilityDescription: nil)
    let actionLabel =
      back ? "Back to chat" : navigation.studio == .tables ? "New table" : "New chat"
    buttons[Item.newChat]?.toolTip = actionLabel
    buttons[Item.newChat]?.setAccessibilityLabel(actionLabel)
    buttons[Item.sidebar]?.toolTip = navigation.showsSidebar ? "Hide sidebar" : "Show sidebar"
  }

  func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
    itemIdentifiers
  }
  func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
    itemIdentifiers
  }

  func toolbar(
    _ toolbar: NSToolbar, itemForItemIdentifier id: NSToolbarItem.Identifier,
    willBeInsertedIntoToolbar flag: Bool
  ) -> NSToolbarItem? {
    if id == Item.separator {
      return NSTrackingSeparatorToolbarItem(identifier: id, splitView: splitView, dividerIndex: 0)
    }
    let item = NSToolbarItem(itemIdentifier: id)
    if id == Item.title {
      item.view = titleContainer
      item.visibilityPriority = .low
    } else if id == Item.metrics {
      item.view = metricsHost
      item.label = "GPU metrics"
    } else if [Item.sidebar, Item.newChat, Item.settings].contains(id) {
      let symbol =
        id == Item.sidebar
        ? "sidebar.left" : id == Item.settings ? "gearshape" : "square.and.pencil"
      let label =
        id == Item.sidebar ? "Toggle sidebar" : id == Item.settings ? "Settings" : "New chat"
      let button = NSButton(
        image: NSImage(systemSymbolName: symbol, accessibilityDescription: nil)!,
        target: self, action: #selector(performToolbarAction(_:)))
      button.identifier = NSUserInterfaceItemIdentifier(id.rawValue)
      button.isBordered = false
      button.imageScaling = .scaleProportionallyDown
      button.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 14, weight: .regular)
      button.contentTintColor = .secondaryLabelColor
      button.setAccessibilityLabel(label)
      button.setAccessibilityIdentifier(
        id == Item.sidebar
          ? "toggle-sidebar" : id == Item.settings ? "open-settings" : "workspace-new-chat")
      button.toolTip = label
      if id == Item.sidebar {
        button.keyEquivalent = "s"
        button.keyEquivalentModifierMask = [.command, .control]
      }
      button.translatesAutoresizingMaskIntoConstraints = false
      NSLayoutConstraint.activate([
        button.widthAnchor.constraint(equalToConstant: 32),
        button.heightAnchor.constraint(equalToConstant: 32),
      ])
      buttons[id] = button
      item.view = button
      item.label = label
      item.visibilityPriority = .high
    } else {
      return nil
    }
    if #available(macOS 26, *) { item.isBordered = false }
    return item
  }

  @objc private func performToolbarAction(_ button: NSButton) {
    switch button.identifier?.rawValue {
    case Item.sidebar.rawValue: onSidebar?()
    case Item.settings.rawValue: onSettings?()
    case Item.newChat.rawValue: navigation.mode == .manager ? onBack?() : onNewChat?()
    default: break
    }
  }
}
