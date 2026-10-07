import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI

/// ThinkingFold.vue: one bordered disclosure, trailing chevron, live 220pt
/// window. Auto-collapse when answer text starts; a user's choice wins.
struct NativeThinkingBlock: View {
  let message: StudioState.NativeTranscript.Message
  @Environment(\.transcriptDisclosure) private var readerDisclosure
  @State private var manualOpen: Bool?
  @State private var pinned = true
  @State private var started = Date()
  @State private var reasoningHeight: CGFloat = 1
  private var active: Bool { message.streaming && message.text.isEmpty }
  private var expanded: Bool { manualOpen ?? active }
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      Button {
        readerDisclosure?.reveal()
        manualOpen = !expanded
      } label: {
        HStack(spacing: 7) {
          Image(systemName: "brain").frame(width: 14, height: 14)
          if active {
            TimelineView(.periodic(from: started, by: 0.25)) { context in
              let elapsed = max(0, Int(context.date.timeIntervalSince(started)))
              Text(elapsed >= 1 ? "Thinking... \(elapsed)s" : "Thinking...")
                .foregroundStyle(PaddockStyle.accent)
            }
          } else {
            Text(message.chrome?.thinkingLabel ?? "Thought for a moment")
            if let meta = message.chrome?.thinkingMeta, !meta.isEmpty {
              Text(meta).font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
            }
          }
          Spacer(minLength: 0)
          Image(systemName: expanded ? "chevron.down" : "chevron.right").frame(
            width: 14, height: 14)
        }.font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary)
          .padding(.vertical, 7).padding(.horizontal, 10)
          .contentShape(Rectangle())
      }.buttonStyle(.plain)
        .accessibilityValue(expanded ? "Expanded" : "Collapsed")
        .accessibilityIdentifier("thinking-toggle-\(message.id)")
      if expanded {
        Group {
          if active {
            ScrollViewReader { proxy in
              PaddockScrollView {
                VStack(alignment: .leading, spacing: 0) {
                  reasoning
                    .fixedSize(horizontal: false, vertical: true)
                    .onGeometryChange(for: CGFloat.self) {
                      $0.size.height
                    } action: {
                      reasoningHeight = $0
                    }
                  Color.clear.frame(height: 1).id("thinking-tail")
                }.frame(maxWidth: .infinity, alignment: .leading)
              }.frame(height: min(220, max(1, reasoningHeight + 1)))
                .defaultScrollAnchor(.bottom, for: .initialOffset)
                .defaultScrollAnchor(pinned ? .bottom : nil, for: .sizeChanges)
                .onScrollPhaseChange { _, phase, context in
                  if phase == .tracking || phase == .interacting {
                    pinned = false
                  } else if phase == .idle {
                    pinned =
                      context.geometry.contentSize.height - context.geometry.visibleRect.maxY < 24
                  }
                }
                .onChange(of: message.reasoning) { _, _ in
                  if pinned { proxy.scrollTo("thinking-tail", anchor: .bottom) }
                }
            }
          } else {
            reasoning
          }
        }.padding(.top, 2).padding(.horizontal, 12).padding(.bottom, 12)
      }
    }.frame(maxWidth: .infinity, alignment: .leading)
      .background(PaddockStyle.canvas)
      .clipShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
          .strokeBorder(Color(nsColor: PaddockStyle.nsColor("borderSubtle"))).allowsHitTesting(
            false)
      )
      .onChange(of: active) { _, value in
        if value {
          started = Date()
          pinned = true
        }
      }
      .accessibilityIdentifier("thinking-\(message.id)")
  }
  private var reasoning: some View {
    NativeMarkdown(message.reasoning, streaming: active, textSize: 13).equatable()
      .foregroundStyle(.secondary)
      // An explicitly expanded disclosure remains part of cross-message
      // selection. Its parent already releases it when the disclosure closes.
      .environment(\.nativeMarkdownViewportReclamation, false)
  }
}
