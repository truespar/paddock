import Foundation

/// Only active-path rows survive a publication. The cache is a rendering
/// projection, never a second durable document or a reason to drop messages.
struct NativePresentationCache {
  struct Message: Equatable {
    let source: NativeStudioRuntime.O
    let parent: NativeStudioRuntime.O?
    let preview: NativeStudioRuntime.O?
    let controls: ConversationMessageControls?
    let fastest: Bool
  }
  var conversation: String?
  var models: [NativeStudioRuntime.O] = []
  var messages: [String: (Message, ConversationValue)] = [:]
  var history: [NativeStudioRuntime.O] = []
  var ordered: [NativeStudioRuntime.O] = []
  var fleet: [NativeStudioRuntime.O] = []
  var capabilities: [String: NativeStudioRuntime.O] = [:]
  var modelRows: [ConversationValue] = []
  var pickerOptions: [ConversationValue] = []
}

extension NativeStudioRuntime {
  func projectedModels() -> (rows: [V], options: [V]) {
    if projectionCache.fleet != models || projectionCache.capabilities != caps {
      projectionCache.fleet = models
      projectionCache.capabilities = caps
      projectionCache.modelRows = models.filter {
        ["chat", "transcriber", "image"].contains($0["kind"]?.string ?? "")
      }.map { model in
        var row = model
        let id = model["id"]!.string!
        row["vision"] = .bool(caps[id]?["vision"]?.bool == true)
        row["audio"] = .bool(
          model["kind"]?.string == "transcriber" || caps[id]?["audio"]?.bool == true)
        row["chat"] = .bool(model["kind"]?.string == "chat")
        row["image"] = .bool(
          model["kind"]?.string == "image" && caps[id]?["image_generation"]?.object != nil)
        return .object(row)
      }
      projectionCache.pickerOptions = projectionCache.modelRows.map { value in
        .object([
          "value": value["id"]!, "label": value["title"]!, "title": value["title"]!,
          "vendor": value["vendor"]!, "hint": value["provider"]!,
          "available": .bool(value["status"]?.string == "ok"),
        ])
      }
    }
    return (projectionCache.modelRows, projectionCache.pickerOptions)
  }
  func projectedMessages(
    _ active: [O], controls: [String: ConversationMessageControls],
    fastest: Set<String>
  ) -> [V] {
    if projectionCache.conversation != document?.id || projectionCache.models != models {
      projectionCache.messages = [:]
      projectionCache.conversation = document?.id
      projectionCache.models = models
    }
    var next: [String: (NativePresentationCache.Message, V)] = [:]
    let projected = active.map { message -> V in
      let id = message["id"]!.string!
      let key = NativePresentationCache.Message(
        source: message,
        parent: message["transcript"] == nil
          ? nil
          : document?.messages.first { $0["id"] == message["parentId"] },
        preview: imagePreviews[id], controls: controls[id], fastest: fastest.contains(id))
      let value: V
      if let cached = projectionCache.messages[id], cached.0 == key {
        value = cached.1
      } else {
        value = messageProjection(message, controls: key.controls, fastest: key.fastest)
      }
      next[id] = (key, value)
      return value
    }
    projectionCache.messages = next
    return projected
  }

  func orderedHistory() -> [O] {
    if projectionCache.history != history {
      projectionCache.history = history
      projectionCache.ordered = history.sorted { a, b in
        if (a["pinned"]?.bool == true) != (b["pinned"]?.bool == true) {
          return a["pinned"]?.bool == true
        }
        let av = a["updatedAt"]?.double ?? 0
        let bv = b["updatedAt"]?.double ?? 0
        return av == bv ? (a["id"]?.string ?? "") < (b["id"]?.string ?? "") : av > bv
      }
    }
    return projectionCache.ordered
  }
}
