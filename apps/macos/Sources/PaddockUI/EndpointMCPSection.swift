import PaddockClient
import SwiftUI

struct EndpointMCPSection: View {
  @Bindable var model: IntegrationsModel
  let port: UInt16
  var body: some View {
    EndpointFormCard("MCP servers") {
      if model.rows.isEmpty {
        VStack(alignment: .leading, spacing: 6) {
          Text("No connectors added").fontWeight(.medium)
        }.padding(.vertical, 8)
      } else {
        VStack(spacing: 0) {
          ForEach(model.rows) { row in
            if row.id != model.rows.first?.id { Divider().padding(.leading, 12) }
            connector(row)
          }
        }.background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
      }
      Button("Manage connectors…", systemImage: "puzzlepiece.extension") { model.onManage?() }
        .buttonStyle(FlatButtonStyle()).disabled(model.saving)
      if !model.rows.isEmpty {
        EndpointHint(text: "Enabled servers may run tools without per-call approval.")
      }
    }.accessibilityIdentifier("endpoint-mcp-section").settingsSearchAnchor(.mcpServers)
  }
  private func connector(_ row: NativeConnector) -> some View {
    HStack(alignment: .center, spacing: 16) {
      VStack(alignment: .leading, spacing: 5) {
        Text(row.label).font(.system(size: 12, weight: .medium))
          .fixedSize(horizontal: false, vertical: true)
        Text(row.url).font(.system(size: 11)).foregroundStyle(.secondary)
          .lineLimit(1).truncationMode(.middle).help(row.url).textSelection(.enabled)
        if row.system {
          Text("On for every model").font(.system(size: 10)).foregroundStyle(.secondary)
        } else if row.credentialReady == false {
          Text("Unlock credentials in Connectors before use").font(.system(size: 10))
            .foregroundStyle(PaddockStyle.caution)
        }
      }.frame(maxWidth: .infinity, alignment: .leading)
      Toggle(
        row.label,
        isOn: Binding(
          get: { model.endpointEnabled(row, port: port) },
          set: { model.setEndpoint(row, port: port, enabled: $0) })
      )
      .labelsHidden().toggleStyle(.switch).controlSize(.small)
      .disabled(row.system || model.saving)
      .accessibilityIdentifier("endpoint-mcp-\(row.id)")
      .accessibilityLabel("Enable \(row.label) for this model")
    }.padding(12)
  }
}
