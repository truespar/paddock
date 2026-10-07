import Foundation
import PaddockClient

extension NativeStudioRuntime {
  func presentation() -> O {
    let fields = (document?.fields ?? [:]).filter { $0.value != .null }
    let id = selected.first ?? ""
    let cap = capability(id)
    let params = fields["params"]?.object ?? Self.defaultParams
    let (modelRows, options) = projectedModels()
    let ordered = orderedHistory()
    var matches = ordered.filter {
      search.isEmpty || ($0["title"]?.string ?? "").localizedCaseInsensitiveContains(search)
    }
    if sort != "newest" {
      matches.sort { a, b in
        if (a["pinned"]?.bool == true) != (b["pinned"]?.bool == true) {
          return a["pinned"]?.bool == true
        }
        if sort == "title" {
          return (a["title"]?.string ?? "").localizedStandardCompare(b["title"]?.string ?? "")
            == .orderedAscending
        }
        return (a["updatedAt"]?.double ?? 0) < (b["updatedAt"]?.double ?? 0)
      }
    }
    let page = min(max(1, page), max(1, (matches.count + 99) / 100))
    let historyRow: (O) -> V = { raw in
      let row = raw.filter { $0.value != .null }
      return .object([
        "id": row["id"] ?? .string(""), "title": row["title"] ?? .string("New conversation"),
        "model": row["model"] ?? .string(""), "updatedAt": row["updatedAt"] ?? .number(0),
        "pinned": row["pinned"] ?? .bool(false),
        "kind": row["kind"] ?? .string("chat"),
        "busy": .bool(row["id"] == fields["id"] && !self.tasks.isEmpty), "titleState": .string(""),
      ])
    }
    let levels =
      cap["reasoning_levels"]?.array
      ?? (cap["reasoning"]?.string == "effort" ? ["low", "medium", "high"].map(V.string) : [])
    let reasoningOff = cap["reasoning_off"]?.bool == true || cap["reasoning"]?.string == "toggle"
    var choices = levels.map {
      V.object(["value": $0, "label": .string(($0.string ?? "").capitalized)])
    }
    if cap["reasoning"]?.string == "toggle" {
      choices = [.object(["value": .string("on"), "label": .string("On")])]
    }
    if reasoningOff {
      choices.insert(.object(["value": .string("off"), "label": .string("Off")]), at: 0)
    }
    let selectedChoice =
      params["thinking"]?.bool == false
      ? "off"
      : (params["reasoningEffort"]?.string).flatMap { $0.isEmpty ? nil : $0 } ?? cap[
        "reasoning_default"]?.string ?? levels.first?.string ?? "on"
    let capabilities: O = [
      "reasoning": cap["reasoning"] ?? .string("none"), "levels": .array(levels),
      "reasoningDefault": cap["reasoning_default"] ?? levels.first ?? .string(""),
      "reasoningOff": .bool(reasoningOff),
      "preserveThinking": cap["reasoning_preserve"] ?? .bool(false),
      "thinkingBudget": cap["thinking_budget"] ?? .bool(false),
      "webSearch": cap["web_search"] ?? .bool(false), "vision": cap["vision"] ?? .bool(false),
      "context": .number(Decimal(contextLimit)),
      "ocrModes": cap["ocr"]?["modes"] ?? .array([]),
      "ocrGrounding": cap["ocr"]?["grounding"] ?? .bool(false),
      "docParser": cap["document_parser"] ?? .bool(false),
      "hasDocument": .bool(
        document?.activeMessages.contains { !NativeDocumentPlan.rasterParts($0).isEmpty } == true),
      "pdfRaster": cap["pdf"]?["raster"] ?? .bool(false),
      "imageLanes": .array(
        selected.compactMap { id in
          let lane = capability(id)
          guard lane["vision"]?.bool == true else { return nil }
          return .object([
            "id": .string(id),
            "name": models.first { $0["id"]?.string == id }?["title"] ?? .string(id),
            "context": lane["max_ctx"] ?? .number(0),
            "documentParser": lane["document_parser"] ?? .bool(false),
            "taskTags": .array((lane["task_tags"]?.array ?? []).compactMap { $0["tag"] }),
            "budget": NativeVisionBudget(value: lane["vision_budget"]) == nil
              ? .null : lane["vision_budget"]!,
          ])
        }),
    ]
    let dials: [(String, String, Double, Double, Double, Double)] = [
      ("temperature", "Temperature", 0, 2, 0.05, 0.8), ("topP", "Top P", 0.01, 1, 0.01, 0.95),
      ("topK", "Top K", 0, 200, 1, 0), ("minP", "Min P", 0, 1, 0.01, 0),
      ("presencePenalty", "Presence penalty", -2, 2, 0.1, 0),
      ("repeatPenalty", "Repeat penalty", 1, 2, 0.01, 1),
    ]
    let sampling = dials.map { key, label, low, high, step, fallback in
      let n = params[key]?.double ?? fallback
      return V.object([
        "key": .string(key), "label": .string(label), "min": .number(Decimal(low)),
        "max": .number(Decimal(high)),
        "step": .number(Decimal(step)), "value": .number(Decimal(n)),
        "display": .string(params[key]?.double == nil ? "Default" : String(format: "%g", n)),
        "set": .bool(params[key]?.double != nil),
      ])
    }
    let active = document?.activeMessages ?? []
    let controls = document?.messageControls ?? [:]
    let groups = Dictionary(
      grouping: active.filter { $0["group"]?.string != nil }, by: { $0["group"]!.string! })
    let fastest = Set(groups.values.compactMap(NativeComparePresentation.fastest))
    let transcript = projectedMessages(active, controls: controls, fastest: fastest)
    let compare = selected.map { id -> V in
      let model = models.first { $0["id"]?.string == id }
      return .object([
        "id": .string(id), "label": model?["title"] ?? .string(id),
        "vendor": model?["vendor"] ?? .string(""), "spec": .string(Self.recordedSpec(model)),
      ])
    }
    var settings = fields.filter {
      [
        "systemPrompt", "params", "toolSelection", "connectorIds", "webSearchEnabled", "ocrMode",
        "audioLanguage", "ocrRegions", "imageParams",
      ].contains($0.key)
    }
    settings["maxTokens"] = maxTokens
    settings["imageParams"] = .object(
      NativeImageGeneration.defaults.merging(fields["imageParams"]?.object ?? [:]) { _, new in new }
    )
    settings["lastImageSeed"] =
      document?.activeMessages.reversed()
      .compactMap { $0["imageGen"]?["seed"] }.first ?? .null
    settings["summarize"] = .bool(preferenceBool("summarize", fallback: true))
    settings["toolSelection"] = settings["toolSelection"] ?? .object(["mode": .string("all")])
    settings["webSearchEnabled"] = settings["webSearchEnabled"] ?? .bool(true)
    settings["connectorIds"] = settings["connectorIds"] ?? .array([])
    return [
      "version": .number(1), "revision": .number(Decimal(revision)),
      "conversation": draft
        ? .null
        : .object([
          "id": fields["id"]!, "title": fields["title"] ?? .string("New conversation"),
          "model": fields["model"] ?? .string(""),
          "messageCount": .number(Decimal(document?.messages.count ?? 0)),
        ]),
      "history": .array(ordered.prefix(20).map(historyRow)),
      "historyTotal": .number(Decimal(history.count)),
      "library": .object([
        "rows": .array(matches.dropFirst((page - 1) * 100).prefix(100).map(historyRow)),
        "page": .number(Decimal(page - 1)), "pageSize": .number(100),
        "total": .number(Decimal(history.count)), "matched": .number(Decimal(matches.count)),
        "search": .string(search), "sort": .string(sort),
      ]),
      "autoTitle": .bool(preferenceBool("auto_title", fallback: true)), "models": .array(modelRows),
      "selectedModels": .array(selected.map(V.string)),
      "modelHeader": .object([
        "currentModel": .string(id), "pickerOptions": .array(options),
        "compareLanes": .array(compare), "comparing": .bool(selected.count > 1),
        "specLabel": selected.count > 1 ? .string("") : compare.first?["spec"] ?? .string(""),
        "isVision": cap["vision"] ?? .bool(false),
      ]),
      "busy": .bool(busy || (audio["phase"]?.string ?? "idle") != "idle"), "loading": .bool(false),
      "error": .string(error),
      "viewport": .object(["left": .number(0), "width": .number(0)]),
      "previewing": .bool(previewPart != nil), "unsavedEdits": .bool(false),
      "settings": .object(settings), "capabilities": .object(capabilities),
      "tools": .array(projectTools()),
      "composer": .object([
        "imageMode": .bool(!selected.isEmpty && selected.allSatisfy(canImagine)),
        "imageCaps": cap["image_generation"] ?? .null,
        "imageEditing": .bool(
          !selected.isEmpty
            && selected.allSatisfy {
              canImagine($0) && capability($0)["image_generation"]?["edit"]?.bool == true
            }),
        "reasoning": .array(choices), "reasoningChoice": .string(selectedChoice),
        "preserveThinking": capabilities["preserveThinking"]!,
        "thinkingBudget": capabilities["thinkingBudget"]!,
        "webSearch": .bool(selected.contains { caps[$0]?["web_search"]?.bool == true }),
        "audioMode": .bool(!selected.isEmpty && selected.allSatisfy(canAudio)),
        "docParser": capabilities["docParser"]!,
        "inputIssue": .string(inputIssue()), "warnings": .array([]),
        "toolCount": .number(
          Decimal(toolGroups.reduce(0) { $0 + ($1["tools"]?.array?.count ?? 0) })),
        "samplerSet": .bool(sampling.contains { $0["set"]?.bool == true }),
        "samplingSource": cap["sampling"]?["source"] ?? .string("Model defaults"),
        "samplingWarnings": .array([]), "sampling": .array(sampling),
        "cost": .number(
          Decimal(active.reduce(0.0) { $0 + ($1["usage"]?["costUsd"]?.double ?? 0) })),
        "contextUsed": .number(
          Decimal(
            document.map {
              NativeContextPlan.contextTokens($0, draft: draftText)
            } ?? NativeContextPlan.tokens(draftText))),
      ]),
      "nativeArtifactsPaneOpen": fields["artifactsPaneOpen"] ?? .bool(true),
      "nativeTranscript": .object([
        "available": .bool(true), "notice": .string(""),
        "conversationId": draft ? .null : fields["id"]!, "leafId": fields["leafId"] ?? .null,
        "messages": .array(transcript),
        "context": contextPresentation(),
      ]),
      "nativeDocument": previewPart.flatMap(Self.documentBadge).map(V.object) ?? .null,
      "nativeAudioPreview": previewPart.flatMap(Self.audioBadge).map(V.object) ?? .null,
      "nativeGraph": .object([
        "available": .bool(
          active.contains {
            ($0["content"]?.array ?? []).contains { $0["type"]?.string == "graph" }
          }), "visible": .bool(graphVisible),
      ]),
      "nativeArtifacts": .array(
        artifacts.map { artifact in
          .object([
            "id": artifact["id"] ?? .string(""),
            "kind": artifact["kind"]?.string.map(V.string) ?? .string("code"),
            "title": artifact["title"]?.string.map(V.string) ?? .string("Artifact"),
            "language": artifact["language"]?.string.map(V.string) ?? .string(""),
            "model": artifact["model"]?.string.map(V.string) ?? .string(""),
            "versions": artifact["versions"]?.integer.map { .number(Decimal($0)) } ?? .number(0),
            "updatedAt": artifact["updatedAt"]?.double.map { .number(Decimal($0)) } ?? .number(0),
          ])
        }), "markUnsure": .bool(preferenceBool("mark_unsure", fallback: true)),
      "audio": .object(audioPresentation()),
      "activity": .object([
        "completedTurn": completedTurn,
        "replies": .array(
          active.filter { $0["role"]?.string == "assistant" }.map {
            .object([
              "id": $0["id"]!,
              "state": .string(
                $0["streaming"]?.bool == true
                  ? "streaming"
                  : $0["stopped"]?.bool == true
                    ? "stopped" : $0["error"]?.string != nil ? "failed" : "completed"),
            ])
          }),
        "approvals": .array(
          active.flatMap { ($0["toolCalls"]?.array ?? []).compactMap { $0["approvalId"] } }),
      ]),
    ]
  }
  static func documentBadge(_ p: O) -> O? {
    guard let id = p["attachmentId"], let name = p["name"]?.string else { return nil }
    let kind: String
    if p["type"]?.string == "image" {
      kind = "image"
    } else if name.lowercased().hasSuffix(".pdf") {
      kind = "pdf"
    } else if name.lowercased().hasSuffix(".docx") {
      kind = "docx"
    } else {
      return nil
    }
    return [
      "id": id, "name": .string(name), "kind": .string(kind), "pages": p["pages"] ?? .null,
      "pageRange": p["pageRange"] ?? .null, "textOnly": .bool(p["pdfMode"]?.string == "text"),
    ]
  }
  static func audioBadge(_ p: O) -> O? {
    guard p["type"]?.string == "audio", let id = p["attachmentId"] else { return nil }
    return [
      "id": id, "name": p["name"] ?? .string("Recording.wav"),
      "mime": p["mime"] ?? .string("audio/wav"), "size": p["size"] ?? .null,
      "duration": p["durationS"] ?? .null,
    ]
  }
  func messageProjection(_ m: O, controls: ConversationMessageControls?, fastest: Bool = false) -> V
  {
    func textValue(_ value: V?) -> V {
      guard let value, value != .null else { return .string("") }
      if let text = value.string { return .string(text) }
      if let text = value["message"]?.string { return .string(text) }
      return .string((try? String(decoding: JSONEncoder().encode(value), as: UTF8.self)) ?? "")
    }
    let m = m.filter { $0.value != .null }
    let model = models.first { $0["id"] == m["model"] }
    let parts = m["content"]?.array ?? []
    var actions: O = [
      "edit": .bool(controls?.edit == true), "retry": .bool(controls?.retry == true),
      "continueReply": .bool(controls?.continueReply == true),
    ]
    if let branch = controls?.branch {
      actions["branch"] = .object([
        "index": .number(Decimal(branch.index)), "count": .number(Decimal(branch.count)),
        "previous": branch.previous.map(V.string) ?? .null,
        "next": branch.next.map(V.string) ?? .null,
      ])
    }
    var usage = NativeResponseMetrics.presentation(m["usage"]?.object ?? [:])
    // Older native failures saved synthetic zero-token "measurements" even
    // when the provider supplied no usage. Hide those without rewriting data.
    if m["error"] != nil, m["response"]?["usage"] == nil,
      usage["promptTokens"]?.integer == 0, usage["completionTokens"]?.integer == 0,
      usage["costUsd"] == nil
    {
      usage = [:]
    }
    let run = m["run"]?.object
    let author = m["model"]?.string ?? run?["model"]?.string ?? ""
    let modelName =
      model?["title"]?.string ?? run?["modelName"]?.string
      ?? CloudModelIdentity.fallbackName(author)
    let vendor =
      model?["vendor"]?.string.flatMap { $0.isEmpty ? nil : $0 }
      ?? run?["vendor"]?.string.flatMap { $0.isEmpty ? nil : $0 }
      ?? CloudModelIdentity.vendor(CloudModelIdentity.bareModel(author)) ?? ""
    var footer = NativeMessagePresentation.footer(usage)
    if let image = m["imageGen"]?.object {
      footer = NativeImageGeneration.footer(image)
    }
    if let speech = m["transcript"]?.object, let ms = usage["ms"]?.double, ms.isFinite, ms > 0 {
      let parent = document?.messages.first { $0["id"] == m["parentId"] }
      let duration =
        speech["durationS"]?.double ?? speech["duration"]?.double
        ?? parent?["content"]?.array?.first(where: { $0["type"]?.string == "audio" })?["durationS"]?
        .double
      if let duration, duration.isFinite, duration > 0 {
        footer = NativeMessagePresentation.footer(usage, realtime: duration * 1000 / ms)
        if m["run"]?["contended"]?.bool == true { footer += " · shared GPU" }
      }
    }
    // Typed before it is returned: as a bare .object([...]) argument the literal
    // is more than Swift 6.3.3's type checker solves in time.
    let presentation: O = [
      "id": m["id"]!, "role": m["role"]!, "text": .string(ConversationDocument.text(m)),
      "reasoning": m["reasoning"] ?? .string(""),
      "model": m["model"] ?? .string(""), "streaming": m["streaming"] ?? .bool(false),
      "stopped": m["stopped"] ?? .bool(false), "error": textValue(m["error"]),
      "incomplete": .bool(m["incomplete"] != nil),
      "actions": .object(actions), "group": m["group"] ?? .null,
      "contended": run?["contended"] ?? m["contended"] ?? .bool(false),
      "automatic": m["auto"] ?? .bool(false),
      "attachments": .array(
        parts.filter { $0["gen"] == nil }.compactMap {
          $0.object.flatMap(Self.documentBadge).map(V.object)
        }),
      "pictures": .array(
        parts.enumerated().compactMap { index, part in
          guard
            let id = NativeImageGeneration.pictureID(
              part.object ?? [:], messageID: m["id"]!.string!, index: index)
          else { return nil }
          return .object([
            "id": .string(id), "name": part["name"] ?? .string("Generated image"),
            "preview": .bool(false),
            "dataURL": part["dataUrl"] ?? .null,
          ])
        } + (imagePreviews[m["id"]!.string!].map { [.object($0)] } ?? [])),
      "imageGeneration": .bool(m["imageGen"] != nil),
      "audioClips": .array(parts.compactMap { $0.object.flatMap(Self.audioBadge).map(V.object) }),
      "toolCalls": .array(
        (m["toolCalls"]?.array ?? []).filter(Self.visibleToolCall).map { call in
          .object([
            "id": textValue(call["id"]), "name": textValue(call["name"]),
            "server": textValue(call["server"]?.string.map(V.string) ?? call["serverLabel"]),
            "arguments": textValue(call["arguments"]), "output": textValue(call["output"]),
            "status": textValue(call["status"]), "error": textValue(call["error"]),
            "approvalId": call["approvalId"]?.string.map(V.string) ?? .null,
            "artifactId": Self.toolArtifactID(call).map(V.string) ?? .null,
          ])
        }),
      "searches": .array(
        (m["webSearches"]?.array ?? []).map { call in
          .object([
            "id": call["id"] ?? .string(""), "query": call["query"] ?? .string(""),
            "status": call["status"] ?? .string(""), "provider": call["provider"] ?? .string(""),
            "error": call["error"]?.string.map(V.string) ?? .string(""),
            "sources": .array(
              (call["sources"]?.array ?? []).compactMap { source in
                guard let url = source["url"]?.string else { return nil }
                return .object(["title": source["title"] ?? .string(url), "url": .string(url)])
              }),
          ])
        }),
      "files": .array(
        parts.compactMap { part in
          guard let object = part.object, part["type"]?.string != "text",
            Self.documentBadge(object) == nil, Self.audioBadge(object) == nil, part["gen"] == nil
          else { return nil }
          let id = part["attachmentId"]?.string ?? ""
          return .object([
            "id": .string(id.isEmpty ? "legacy-\(m["id"]!.string!)" : id),
            "name": part["name"] ?? .string("Attachment"), "kind": part["type"] ?? .string("file"),
            "mime": part["mime"] ?? .string(""), "stored": .bool(ConversationDocument.validID(id)),
          ])
        }), "documentResult": documentResult(m), "speech": speechProjection(m),
      "audioPending": m["audioPending"] ?? .bool(false),
      "chrome": .object([
        "modelName": .string(modelName), "vendor": .string(vendor),
        "spec": .string(Self.recordedSpec(run)),
        "tools": .array(run?["tools"]?.array?.filter { $0.string != nil } ?? []),
        "fastest": .bool(fastest),
        "footer": .string(footer),
        "footerHint": .string(
          m["imageGen"]?.object.map { NativeImageGeneration.hint($0, usage: usage) }
            ?? NativeMessagePresentation.hint(usage)),
        "thinkingLabel": .string(
          m["streaming"]?.bool == true && ConversationDocument.text(m).isEmpty
            ? "Thinking..."
            : usage["reasoningMs"]?.double.map {
              "Thought for \(NativeMessagePresentation.duration($0))"
            } ?? "Thought for a moment"),
        "thinkingMeta": .string(
          m["streaming"]?.bool == true
            ? ""
            : NativeMessagePresentation.join([
              usage["reasoningTokens"]?.integer.map { "\($0) tokens" },
              NativeMessagePresentation.speed(usage["reasoningTps"]?.double),
            ])),
        "cutNote": .string(Self.tokenLimitNote(usage)),
        "sections": .array(
          m["imageGen"]?.object.map { NativeImageGeneration.sections($0, run: run) }
            ?? NativeMessagePresentation.sections(run: run, usage: usage)),
        "promptText": run?["systemPrompt"] ?? .string(""),
      ]),
    ]
    return .object(presentation)
  }
  func inputIssue() -> String {
    guard !selected.isEmpty else { return "Choose a running model" }
    if selected.contains(where: { id in
      !models.contains { $0["id"]?.string == id && $0["status"]?.string == "ok" }
    }) {
      return "A selected model is not reachable"
    }
    let parts = Array(staged.values)
    if selected.contains(where: canImagine) {
      if !selected.allSatisfy(canImagine) { return "Compare image models with other image models" }
      for id in selected {
        do {
          try NativeImageGeneration.validateReferences(
            parts, caps: capability(id)["image_generation"]?.object ?? [:])
        } catch { return error.localizedDescription }
      }
      return ""
    }
    if selected.contains(where: { capability($0)["document_parser"]?.bool == true }),
      !parts.contains(where: { $0["type"]?.string == "image" || NativeDocumentPlan.isPDF($0) }),
      document?.activeMessages.contains(where: { !NativeDocumentPlan.rasterParts($0).isEmpty })
        != true
    {
      return "Attach an image or PDF to read"
    }
    let clips = parts.filter { $0["type"]?.string == "audio" }
    if clips.count > 1 { return "Attach one recording per turn" }
    if !clips.isEmpty && !selected.allSatisfy(canAudio) {
      return "Every selected model must support audio"
    }
    if selected.contains(where: { !canChat($0) }) && clips.isEmpty {
      return "Attach a recording or use the microphone"
    }
    if !parts.isEmpty && parts.contains(where: { $0["type"]?.string != "audio" })
      && selected.contains(where: { !canChat($0) })
    {
      return "Speech models accept audio, not documents or images"
    }
    return ""
  }
}
