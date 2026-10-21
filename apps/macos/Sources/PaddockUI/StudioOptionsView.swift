import PaddockStudio
import SwiftUI

struct StudioCompareView: View {
  @Bindable var chat: StudioWorkspace
  var body: some View { StudioModelSelectionView(chat: chat, initiallyComparing: true) }
}

/// Single and Compare share identity, search, availability and keyboard
/// navigation. Only Apply commits the Compare draft; filtering never edits it.
struct StudioModelSelectionView: View {
  @Bindable var chat: StudioWorkspace
  var initiallyComparing = false
  @Environment(\.dismiss) private var dismiss
  @State private var selected = Set<String>()
  @State private var search = ""
  @State private var comparing = false
  @State private var highlighted: String?
  @State private var applying = false
  private var models: [StudioState.Model] { chat.state?.models ?? [] }
  private var blocked: Bool { chat.busy || chat.hasMessageEdit || applying || !chat.ready }
  private var visibleIDs: [String] {
    let policy = StudioModelSelectionPolicy(models: models)
    return StudioCompareList.groups(models, search: search, includeUnavailable: !comparing)
      .flatMap(\.models).filter { model in
        model.status == "ok"
          && (!comparing || selected.contains(model.id)
            || policy.canApply(selected.union([model.id])))
      }.map(\.id)
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Picker("Model selection", selection: $comparing) {
        Text("Model").tag(false)
        Text("Compare").tag(true)
      }.pickerStyle(.segmented).labelsHidden().disabled(blocked)
      NativeSearchField(
        text: $search, prompt: "Search models or providers", focusOnAppear: true,
        onMove: { highlighted = SearchNavigation.moved(highlighted, in: visibleIDs, by: $0) },
        onSubmit: {
          if let id = highlighted ?? visibleIDs.first { activate(id) }
        }, onCancel: { dismiss() })
      if comparing { StudioCompareSelectionBar(selected: $selected, busy: blocked) }
      StudioCompareList(
        models: models, selected: $selected, search: search, single: !comparing,
        highlighted: highlighted, onSelect: { activate($0) }
      ).disabled(blocked)
      if let error = chat.error { Text(error).foregroundStyle(.secondary) }
      if comparing {
        HStack {
          Spacer()
          Button("Cancel") { dismiss() }.buttonStyle(FlatButtonStyle())
          Button("Use selection") {
            apply(
              StudioCompareList.orderedSelection(
                selected, existing: chat.state?.selectedModels ?? []))
          }.buttonStyle(FlatButtonStyle(primary: true))
            .disabled(!StudioCompareList.canApply(selected, models: models) || blocked)
        }
      }
    }.font(.system(size: 12)).padding(16).frame(width: 380)
      .onAppear {
        comparing = initiallyComparing
        selected = Set(chat.state?.selectedModels ?? [])
        highlighted = chat.state?.modelHeader?.currentModel
      }
      .onChange(of: visibleIDs) { _, ids in
        if !ids.contains(highlighted ?? "") { highlighted = ids.first }
      }
  }

  private func activate(_ id: String) {
    guard !blocked, visibleIDs.contains(id) else { return }
    highlighted = id
    if comparing {
      if selected.contains(id) { selected.remove(id) } else { selected.insert(id) }
    } else {
      apply([id])
    }
  }
  private func apply(_ ids: [String]) {
    guard !blocked else { return }
    applying = true
    Task {
      await chat.perform("models", ["ids": .array(ids.map(StudioValue.string))])
      applying = false
      if chat.error == nil { dismiss() }
    }
  }
}

/// Resolve the fleet once per projection. Candidate validation is bounded by
/// four lanes, not a full catalog scan for every row/arrow-key press.
struct StudioModelSelectionPolicy {
  private let byID: [String: StudioState.Model]
  init(models: [StudioState.Model]) {
    byID = Dictionary(models.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
  }
  func canApply(_ selected: Set<String>) -> Bool {
    guard (1...4).contains(selected.count) else { return false }
    let choices = selected.compactMap { byID[$0] }.filter { $0.status == "ok" }
    return choices.count == selected.count
      && (choices.allSatisfy(\.chat) || choices.allSatisfy(\.audio)
        || choices.allSatisfy { $0.image == true })
  }
}

/// Edits only the popover draft, independently of the catalog's search filter.
struct StudioCompareSelectionBar: View {
  @Binding var selected: Set<String>
  var busy = false

  func clearSelection() {
    guard !busy else { return }
    selected.removeAll()
  }

  var body: some View {
    HStack {
      Text(selected.isEmpty ? "No models selected" : "\(selected.count) selected")
        .font(.system(size: 11)).foregroundStyle(.secondary)
        .accessibilityIdentifier("compare-selection-count")
      Spacer()
      Button("Clear selection", action: clearSelection)
        .buttonStyle(FlatButtonStyle()).disabled(selected.isEmpty || busy)
        .accessibilityIdentifier("compare-clear-selection")
        .help(
          "Clear all selected models, including ones hidden by search. "
            + "Nothing is applied until you use the selection.")
    }
  }
}

/// An intrinsically bounded list: a maxHeight-only ScrollView can collapse to
/// one checkbox in an NSPopover even when more rows exist in its AX tree.
/// Row/group geometry sets the viewport; larger libraries scroll normally.
struct StudioCompareList: View {
  let models: [StudioState.Model]
  @Binding var selected: Set<String>
  var search = ""
  var single = false
  var highlighted: String?
  var onSelect: ((String) -> Void)?

  struct Group: Identifiable {
    let id: String
    let title: String
    let models: [StudioState.Model]
  }

  static func groups(
    _ models: [StudioState.Model], search: String, includeUnavailable: Bool = false
  ) -> [Group] {
    let terms = search.split(whereSeparator: \.isWhitespace).map(String.init)
    let available = models.filter { model in
      (includeUnavailable || model.status == "ok")
        && terms.allSatisfy { term in
          [model.title, model.id, model.provider, model.vendor].contains {
            $0.localizedStandardContains(term)
          }
        }
    }
    var groups: [Group] = []
    let local = available.filter { $0.port != nil }
    if !local.isEmpty { groups.append(Group(id: "local", title: "Local", models: local)) }
    var providers: [String] = []
    var byProvider: [String: [StudioState.Model]] = [:]
    for model in available where model.port == nil {
      if byProvider[model.provider] == nil { providers.append(model.provider) }
      byProvider[model.provider, default: []].append(model)
    }
    for provider in providers {
      groups.append(
        Group(
          id: "cloud:\(provider)", title: provider,
          models: byProvider[provider] ?? []))
    }
    return groups
  }

  static func canApply(_ selected: Set<String>, models: [StudioState.Model]) -> Bool {
    StudioModelSelectionPolicy(models: models).canApply(selected)
  }

  static func orderedSelection(_ selected: Set<String>, existing: [String]) -> [String] {
    let retained = existing.filter { selected.contains($0) }
    return retained + selected.subtracting(retained).sorted()
  }

  var body: some View {
    let groups = Self.groups(models, search: search, includeUnavailable: single)
    let height = min(300, max(72, CGFloat(groups.reduce(0) { $0 + 26 + $1.models.count * 52 + 8 })))
    let policy = StudioModelSelectionPolicy(models: models)
    ScrollViewReader { proxy in
      PaddockScrollView {
        LazyVStack(alignment: .leading, spacing: 0) {
          if groups.isEmpty {
            Text(
              search.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "No available models. Start a local model or add a cloud model in Settings."
                : "No models match your search."
            )
            .font(.system(size: 12)).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
          }
          ForEach(groups) { group in
            Text(group.title).font(.system(size: 11, weight: .semibold)).foregroundStyle(.secondary)
              .frame(height: 26).accessibilityAddTraits(.isHeader)
            ForEach(group.models) { model in
              SwiftUI.Group {
                if single {
                  StudioSingleModelRow(model: model, selected: selected.contains(model.id)) {
                    onSelect?(model.id)
                  }.disabled(model.status != "ok")
                } else {
                  StudioCompareModelRow(
                    model: model,
                    isSelected: Binding(
                      get: { selected.contains(model.id) },
                      set: { enabled in
                        if enabled { selected.insert(model.id) } else { selected.remove(model.id) }
                      })
                  ).disabled(
                    !selected.contains(model.id)
                      && !policy.canApply(selected.union([model.id])))
                }
              }.background(
                highlighted == model.id ? PaddockStyle.elevated : .clear,
                in: RoundedRectangle(cornerRadius: 6)
              ).id(model.id)
            }
            Color.clear.frame(height: 8).accessibilityHidden(true)
          }
        }.frame(maxWidth: .infinity, alignment: .leading)
      }.onChange(of: highlighted) { _, id in
        if let id { proxy.scrollTo(id) }
      }
    }.frame(height: height).accessibilityIdentifier("compare-model-list")
  }
}

private struct StudioSingleModelRow: View {
  let model: StudioState.Model
  let selected: Bool
  let action: () -> Void
  var body: some View {
    Button(action: action) {
      HStack(spacing: 9) {
        Image(systemName: "checkmark").opacity(selected ? 1 : 0).frame(width: 16)
        StudioPickerIdentity(model: model)
      }.frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
        .contentShape(Rectangle())
    }.buttonStyle(.plain)
      .accessibilityLabel("\(model.title) · \(model.provider)")
      .accessibilityAddTraits(selected ? .isSelected : [])
      .help("\(model.title) · \(model.id)")
  }
}

private struct StudioPickerIdentity: View {
  let model: StudioState.Model
  var body: some View {
    HStack(spacing: 9) {
      ModelAvatar(vendor: model.vendor, size: 26)
      VStack(alignment: .leading, spacing: 4) {
        Text(model.title).font(.system(size: 12, weight: .medium)).lineLimit(1)
        Text(
          model.status == "ok"
            ? (model.port.map { "Port \(String($0))" } ?? model.provider) : "Unavailable"
        )
        .font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
      }
      Spacer(minLength: 0)
      if model.vision { Image(systemName: "photo").help("Supports images") }
      if model.audio { Image(systemName: "waveform").help("Transcribes audio") }
      if model.image == true { Image(systemName: "paintbrush.pointed").help("Generates images") }
    }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
  }
}

/// A checkbox's default compound label aligns to the first text baseline.
/// Lay out the native control separately so its center, the maker mark and
/// the two-line identity share one axis. The label remains a click target,
/// but only the checkbox participates in keyboard/VoiceOver navigation.
struct StudioCompareModelRow: View {
  let model: StudioState.Model
  @Binding var isSelected: Bool

  var body: some View {
    HStack(alignment: .center, spacing: 9) {
      Toggle("Select model", isOn: $isSelected)
        .toggleStyle(.checkbox).labelsHidden()
        .accessibilityLabel("\(model.title) · \(model.provider)")
        .accessibilityIdentifier("compare-model-\(model.id)")
      Button {
        isSelected.toggle()
      } label: {
        StudioPickerIdentity(model: model)
      }.buttonStyle(.plain).focusable(false).accessibilityHidden(true)
    }.frame(height: 52)
      .help("\(model.title) · \(model.provider)")
  }
}
