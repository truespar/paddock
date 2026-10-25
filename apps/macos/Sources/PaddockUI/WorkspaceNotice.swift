import SwiftUI

/// Workspace-level recovery/progress, not a second full-width toolbar. Model
/// rows own their routine lifecycle status; only unrepresented work comes here.
struct WorkspaceNotice: View {
  enum Kind { case progress, warning }
  let message: String
  let kind: Kind
  var dismiss: (() -> Void)? = nil
  var retry: (() -> Void)? = nil

  var body: some View {
    HStack(alignment: .top, spacing: 10) {
      Group {
        switch kind {
        case .progress:
          ProgressView().controlSize(.mini).tint(PaddockStyle.secondary)
        case .warning:
          Image(systemName: "exclamationmark.triangle").foregroundStyle(PaddockStyle.caution)
        }
      }.frame(width: 16, height: 18).accessibilityHidden(true)
      Text(message).foregroundStyle(.secondary).textSelection(.enabled)
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
      if let retry {
        Button("Retry", action: retry).buttonStyle(QuietButtonStyle())
      }
      if let dismiss {
        Button("Dismiss", systemImage: "xmark", action: dismiss)
          .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
          .frame(width: 24, height: 24).help("Dismiss message")
          .accessibilityIdentifier("dismiss-workspace-notice")
      }
    }.font(.system(size: 12)).padding(12)
      .background(
        PaddockStyle.surface,
        in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
      )
      .accessibilityIdentifier("workspace-notice")
  }
}
