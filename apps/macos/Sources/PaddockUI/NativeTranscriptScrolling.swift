import SwiftUI

/// Scroll observation must not invalidate NativeStudioTranscript's body: that
/// would replace its AppKit hosting root and update every rich-text attachment
/// for each wheel tick. Only this modifier and the Latest control observe the
/// reader's intent, and only discrete transitions publish view changes.
struct NativeTranscriptScrolling: ViewModifier {
  let intent: TranscriptScrollIntent
  let proxy: ScrollViewProxy
  @State private var viewport = CGSize.zero
  @State private var position = ScrollPosition(edge: .top)
  @State private var lastOffset: CGFloat = 0
  @State private var interacting = false
  @State private var restoringTo: CGFloat?
  init(intent: TranscriptScrollIntent, proxy: ScrollViewProxy) {
    self.intent = intent
    self.proxy = proxy
    _position = State(
      initialValue: ScrollPosition(edge: intent.initiallyFollowing ? .bottom : .top))
  }

  func body(content: Content) -> some View {
    content
      .scrollPosition($position)
      .defaultScrollAnchor(intent.initiallyFollowing ? .bottom : .top, for: .initialOffset)
      .onScrollPhaseChange { old, phase, context in
        if phase == .tracking || phase == .interacting {
          intent.readingDisclosure = false
          intent.pinned = false
          intent.reading = nil
          interacting = true
          restoringTo = nil
        } else if phase == .idle && old != .idle && !intent.readingDisclosure {
          interacting = false
          intent.pinned =
            context.geometry.contentSize.height - context.geometry.visibleRect.maxY
            + context.geometry.contentInsets.bottom <= 24
          intent.capture(y: context.geometry.visibleRect.minY)
        }
      }
      .onScrollGeometryChange(for: TranscriptScrollGeometry.self) {
        TranscriptScrollGeometry(
          height: $0.contentSize.height,
          viewport: $0.containerSize,
          offset: $0.contentOffset.y, bottom: $0.visibleRect.maxY - $0.contentInsets.bottom)
      } action: { old, new in
        lastOffset = new.offset
        if viewport != new.viewport { viewport = new.viewport }
        let away = new.height - new.bottom > 24
        if intent.awayFromBottom != away { intent.awayFromBottom = away }
        // Actual message measurements and viewport resizes drive follow, not
        // scroll geometry changes. A user scrolling must never trigger scrollTo.
        guard old.height == new.height, old.viewport == new.viewport else { return }
        let restoring = restoringTo != nil && !interacting
        if !intent.awayFromBottom && !intent.readingDisclosure && !restoring {
          intent.pinned = true
        } else if new.offset < old.offset && old.bottom <= old.height + 1 {
          // Keyboard, accessibility and scrollbar jumps have no wheel phase.
          // Ignore initial out-of-range offsets being clamped during layout.
          intent.pinned = false
        }
        if !restoring {
          // Keyboard/accessibility/AppKit scrolling may have no gesture phase.
          // Release our last explicit point so SwiftUI cannot reapply it on a
          // later toolbar or safe-area change.
          if old.offset != new.offset { position.isPositionedByUser = true }
          intent.capture(y: new.offset)
        }
      }
      .task(id: FollowLayout(viewport: viewport, revision: intent.layoutRevision)) {
        // AppKit resizes the clip and hosting views in separate passes. Resolve
        // the anchor after real row growth or a viewport resize, never for
        // scroll-offset changes.
        await Task.yield()
        guard !Task.isCancelled else { return }
        // A new layout supersedes the previous restoration task, including
        // its short settling window. Never leave input suppression latched.
        restoringTo = nil
        if intent.pinned && !Task.isCancelled && viewport.height > 0 {
          proxy.scrollTo("native-bottom", anchor: .bottom)
        } else if !interacting, !Task.isCancelled, viewport.height > 0,
          let y = intent.restorationY(), abs(y - lastOffset) > 0.5
        {
          restoringTo = y
          position.scrollTo(y: y)
          do {
            try await Task.sleep(for: .milliseconds(80))
            restoringTo = nil
          } catch {}
        }
      }
      .onDisappear {
        if intent.reading == nil || position.isPositionedByUser { intent.capture(y: lastOffset) }
      }
  }

  private struct FollowLayout: Equatable {
    let viewport: CGSize
    let revision: Int
  }
}

struct NativeTranscriptLatestControl: View {
  let intent: TranscriptScrollIntent
  let columnWidth: CGFloat
  let composerHeight: CGFloat
  let proxy: ScrollViewProxy

  var body: some View {
    if !intent.pinned && intent.awayFromBottom {
      NativeTranscriptLatestButton(columnWidth: columnWidth, composerHeight: composerHeight) {
        intent.readingDisclosure = false
        intent.pinned = true
        proxy.scrollTo("native-bottom", anchor: .bottom)
      }
    }
  }
}

private struct TranscriptScrollGeometry: Equatable {
  let height: CGFloat
  let viewport: CGSize
  let offset: CGFloat
  let bottom: CGFloat
}
