import Foundation

extension NativeStudioRuntime {
  public func audioDraftText() -> String { draftText }
  public func audioPresentation() -> O {
    let audioOK = !selected.isEmpty && selected.allSatisfy(canAudio)
    let audioMode = audioOK && !selected.allSatisfy(canChat)
    let ears = models.filter {
      $0["status"]?.string == "ok" && $0["port"] != nil && canAudio($0["id"]!.string!)
        && NativeSpeechPolicy.liveSupported(capability($0["id"]!.string!))
    }
    let blocked =
      audioMode
      && selected.contains {
        $0.hasPrefix("cloud:") || !NativeSpeechPolicy.liveSupported(capability($0))
      }
    let jobs =
      !audioOK
      ? (ears.isEmpty ? [] : ["dictate"])
      : audioMode
        ? (blocked ? ["record"] : ["live", "record"])
        : selected.count > 1 || ears.isEmpty ? ["record"] : ["record", "dictate"]
    let preferred = audio["mode"]?.string ?? "live"
    let mode = jobs.contains(preferred) ? preferred : jobs.first ?? "dictate"
    let limit = NativeSpeechPolicy.recordingLimit(
      capabilities: selected.map(capability), live: mode != "record")
    var result: O = [
      "mode": .string(mode), "phase": .string("idle"), "retryAvailable": .bool(false),
      "jobs": .array(jobs.map(V.string)), "audioMode": .bool(audioMode), "audioOk": .bool(audioOK),
      "liveBlocked": .bool(blocked),
      "liveReason": .string(blocked ? "A selected model accepts recordings, not live audio" : ""),
      "transcribers": .array(ears.map { .object(["id": $0["id"]!, "label": $0["title"]!]) }),
      "transcriber": preferences["pk_dictate_with"] ?? ears.first?["id"] ?? .string(""),
      "devices": .array([]), "devicesNamed": .bool(false),
      "device": preferences["pk_mic_device"] ?? .string(""),
      "language": document?.fields["audioLanguage"]
        ?? .string(Locale.current.language.languageCode?.identifier ?? "en"),
      "languages": .array(
        [.object(["value": .string("auto"), "label": .string("Detect language")])]
          + Locale.LanguageCode.isoLanguageCodes.compactMap { code in
            guard code.identifier.count == 2,
              let label = Locale.current.localizedString(forLanguageCode: code.identifier)
            else { return nil }
            return .object(["value": .string(code.identifier), "label": .string(label)])
          }),
      "session": .string(""), "dictation": .array([]), "provisional": .string(""),
      "levels": .array([]), "elapsed": .number(0), "remaining": .number(Decimal(limit)),
      "limit": .number(Decimal(limit)),
      "arming": .bool(false), "idle": .bool(false), "shouldStop": .bool(false),
      "error": .string(""), "deviceNote": .string(""),
      "menu": .object([
        "offered": .bool(true), "menu": .bool(true), "needsSetup": .bool(jobs.isEmpty),
        "jobChoice": .bool(jobs.count > 1), "deviceChoice": .bool(true),
        "earChoice": .bool(mode == "dictate" && ears.count > 1),
        "setupMessage": .string(jobs.isEmpty ? "Start a speech model to use dictation" : ""),
        "setupAction": .string("Start speech model"),
      ]),
      "speechModels": .array(
        ears.map {
          .object([
            "port": $0["port"]!, "model": $0["id"]!, "title": $0["title"]!,
            "vendor": $0["vendor"] ?? .string(""), "running": .bool(true), "busy": .bool(false),
            "status": .string("ok"), "canStart": .bool(false), "canStop": .bool(true),
          ])
        }),
    ]
    for (key, value) in audio
    where ![
      "jobs", "audioMode", "audioOk", "liveBlocked", "liveReason", "transcribers", "speechModels",
      "menu", "limit",
    ].contains(key) { result[key] = value }
    result["mode"] = .string(mode)
    return result
  }
  public func configureAudio(_ p: O) async throws {
    guard (audio["phase"]?.string ?? "idle") == "idle" else {
      throw ConversationFailure.invalid("Stop the microphone first")
    }
    var next = preferences
    for (key, value) in p {
      let text = try Self.text(value, limit: 1024)
      switch key {
      case "mode":
        guard audioPresentation()["jobs"]?.array?.contains(.string(text)) == true else {
          throw ConversationFailure.invalid("This microphone mode is unavailable")
        }
        audio["mode"] = .string(text)
      case "transcriber":
        guard
          models.contains(where: {
            $0["id"]?.string == text && $0["status"]?.string == "ok" && $0["port"] != nil
          }), canAudio(text)
        else { throw ConversationFailure.invalid("The selected speech model is not running") }
        next["pk_dictate_with"] = value
        audio[key] = value
      case "device":
        next["pk_mic_device"] = value
        audio[key] = value
      case "language":
        try change { $0["audioLanguage"] = value }
        try await persist()
        audio[key] = value
      default: throw ConversationFailure.invalid("Invalid audio setting")
      }
    }
    if next != preferences {
      _ = try await transport.api(
        "api/settings", method: "PUT", body: .object(["macos_studio_preferences": .object(next)]))
      preferences = next
    }
    schedulePublish()
  }
  public func audioTargets(dictation: Bool) throws -> [O] {
    let state = audioPresentation()
    let ids = dictation ? [state["transcriber"]?.string ?? ""] : selected
    return try ids.map { id in
      let row = try model(id)
      guard canAudio(id), row["port"] != nil, NativeSpeechPolicy.liveSupported(capability(id))
      else {
        throw ConversationFailure.invalid("Start a local speech model for live transcription")
      }
      var target = row
      target["transcription"] = .object(
        NativeSpeechPolicy.transcription(
          capabilities: capability(id),
          dictation: dictation, language: state["language"]?.string))
      target["utteranceLimit"] = .number(
        Decimal(NativeSpeechPolicy.utteranceLimit([capability(id)])))
      target["drain"] = .bool(capability(id)["realtime_transcription"]?["drain"]?.bool == true)
      return target
    }
  }
  func transcribe(_ clip: O, modelID: String, messageID: String) async throws {
    let parentID = document?.messages.first(where: { $0["id"]?.string == messageID })?["parentId"]
    let parent = document?.messages.first { $0["id"] == parentID }
    let instruction = canChat(modelID) ? parent.map(ConversationDocument.text) : nil
    let result = try await transcribeClip(clip, modelID: modelID, instruction: instruction) {
      [weak self] event in
      await self?.transcriptionEvent(event, messageID: messageID)
    }
    // A speech lane does not use the Responses reducer; otherwise a subsequent
    // coalesced flush would replace its transcript with empty response text.
    reducers[messageID] = nil
    let verbose = result["paddock_verbose"]?.object ?? result
    try updateMessage(messageID) {
      $0["content"] = .array([
        .object(["type": .string("text"), "text": result["text"] ?? verbose["text"] ?? .string("")])
      ])
      let meta = NativeSpeechMetadata.file(result)
      $0["transcript"] = .object(meta)
      $0["response"] = .object(result)
      let wire = result["usage"]?.object ?? [:]
      $0["usage"] = .object([
        "promptTokens": wire["input_tokens"] ?? .number(0),
        "completionTokens": wire["output_tokens"] ?? .number(0),
        "ms": result["paddock_client_elapsed_ms"] ?? .null, "costUsd": wire["cost"] ?? .null,
      ])
      $0["run"] = .object([
        "model": .string(modelID), "audioPrompt": .string(instruction ?? ""),
        "contended": .bool(selected.filter { !$0.hasPrefix("cloud:") }.count > 1),
      ])
    }
    try await enrichSpeech(messageID: messageID, clip: clip)
  }
  func transcriptionEvent(_ event: O, messageID: String) {
    if let delta = event["delta"]?.string {
      reducers[messageID] = nil
      try? updateMessage(messageID) { message in
        let text = ConversationDocument.text(message) + delta
        message["content"] = .array([.object(["type": .string("text"), "text": .string(text)])])
      }
      schedulePublish()
    }
  }
  public func transcribeClip(
    _ clip: O, modelID: String, instruction: String? = nil,
    receive: @escaping @Sendable (O) async throws -> Void
  ) async throws -> O {
    let model = try model(modelID)
    let cap = capability(modelID)
    let aid = try Self.id(clip["attachmentId"])
    if let duration = clip["durationS"]?.double,
      let limit = cap["transcription_max_clip_s"]?.double, duration > limit
    {
      throw ConversationFailure.invalid(
        "This model accepts at most \(Int(limit)) seconds per clip. The original recording is retained."
      )
    }
    let original = try await transport.bytes("api/attachments/\(aid)", maximum: 100 * 1024 * 1024)
    let boundary = UUID().uuidString
    var body = Data()
    func field(_ name: String, _ value: String) {
      body.append(
        Data(
          "--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n"
            .utf8))
    }
    let cloud = model["endpoint"]?.string
    let path =
      cloud.map { "api/cloud/\($0)/v1/audio/transcriptions" }
      ?? "api/runners/\(model["port"]!.integer!)/v1/audio/transcriptions"
    field("model", model["wireModel"]?.string ?? modelID)
    for (name, value) in NativeSpeechPolicy.fileFields(
      capabilities: cap, cloud: cloud != nil, instruction: instruction,
      language: clip["language"]?.string)
    { field(name, value) }
    let mime = clip["mime"]?.string ?? "audio/wav"
    guard !mime.contains("\r"), !mime.contains("\n") else {
      throw ConversationFailure.invalid("Invalid audio type")
    }
    body.append(
      Data(
        "--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording\"\r\nContent-Type: \(mime)\r\n\r\n"
          .utf8))
    body.append(original)
    body.append(Data("\r\n--\(boundary)--\r\n".utf8))
    let started = ContinuousClock.now
    var result = try await transport.transcription(
      path: path, body: body, boundary: boundary, receive: receive)
    result["paddock_client_elapsed_ms"] = .number(
      Decimal(Self.seconds(started.duration(to: .now)) * 1000))
    return result
  }
  func speechProjection(_ m: O) -> V {
    guard let raw = m["transcript"]?.object else { return .null }
    let meta = NativeSpeechMetadata.file(raw)
    let parent = document?.messages.first { $0["id"] == m["parentId"] }
    let clip = parent?["content"]?.array?.first { $0["type"]?.string == "audio" }?.object.flatMap(
      Self.audioBadge)
    let streaming = m["streaming"]?.bool == true
    let words = NativeSpeechMetadata.renderWords(
      meta, text: ConversationDocument.text(m), streaming: streaming)
    var facts: [V] =
      meta["language"].map { [.object(["label": .string("Language"), "value": $0])] } ?? []
    if let source = meta["wordsFrom"] {
      facts.append(.object(["label": .string("Word timing"), "value": source]))
    }
    if let warning = meta["alignmentError"] {
      facts.append(.object(["label": .string("Word timing unavailable"), "value": warning]))
    }
    if meta["wordsLangOk"]?.bool == false {
      facts.append(
        .object([
          "label": .string("Alignment language"),
          "value": .string("Outside the aligner's trained language set"),
        ]))
    }
    return .object([
      "messageId": m["id"] ?? .null, "conversationId": document.map { .string($0.id) } ?? .null,
      "diarization": meta["diarization"] ?? .null,
      "diarizers": .array(
        streaming || clip == nil
          ? []
          : models.filter {
            $0["kind"]?.string == "diarizer" && $0["status"]?.string == "ok"
          }.map { .object($0) }),
      "clip": clip.map(V.object) ?? .null, "words": .array(words),
      "segments": streaming ? .array([]) : meta["segments"] ?? .array([]), "differs": .array([]),
      "facts": .array(facts),
      "guards": meta["guards"] ?? .array([]),
      "subtitleExport": .bool(!(meta["segments"]?.array ?? []).isEmpty),
    ])
  }
  func transcriptExport(_ p: O) throws -> O {
    guard p["conversationId"]?.string == document?.id, p["leafId"] == document?.fields["leafId"],
      let message = document?.activeMessages.first(where: { $0["id"] == p["messageId"] }),
      message["streaming"]?.bool != true,
      let meta = message["transcript"]?.object
    else { throw ConversationFailure.stale }
    let format = p["format"]?.string ?? "txt"
    let text: String
    switch format {
    case "txt": text = ConversationDocument.text(message)
    case "json":
      var export = meta
      export["text"] = .string(ConversationDocument.text(message))
      text = String(decoding: try JSONEncoder().encode(export), as: UTF8.self)
    case "srt", "vtt":
      let segments = meta["segments"]?.array ?? []
      guard !segments.isEmpty else {
        throw ConversationFailure.invalid("This model did not supply subtitle times")
      }
      func clock(_ value: V?) -> String {
        let ms = max(0, Int((value?.double ?? 0) * 1000))
        return String(
          format: "%02d:%02d:%02d%@%03d", ms / 3_600_000, ms / 60000 % 60, ms / 1000 % 60,
          format == "srt" ? "," : ".", ms % 1000)
      }
      text =
        (format == "vtt" ? "WEBVTT\n\n" : "")
        + segments.enumerated().map { index, s in
          "\(index + 1)\n\(clock(s["start"])) --> \(clock(s["end"]))\n\(s["text"]?.string ?? "")\n"
        }.joined(separator: "\n")
    default: throw ConversationFailure.invalid("Unknown transcript format")
    }
    return ["export": .object(["name": .string("transcript.\(format)"), "text": .string(text)])]
  }
}
