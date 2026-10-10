import Foundation

/// Request-time, per-lane gates. Persisted preferences are not proof that the
/// currently selected model understands a Paddock extension.
enum NativeDocumentRequestOptions {
  typealias V = ConversationValue
  typealias O = [String: V]

  static func apply(fields: O, capability: O, input: [V], body: inout O) -> Bool {
    let visual = input.contains { item in
      (item["content"]?.array ?? []).contains {
        ["input_image", "input_file"].contains($0["type"]?.string ?? "")
      }
    }
    var ocr: O = [:]
    if visual, let advertised = capability["ocr"]?.object {
      if let mode = fields["ocrMode"]?.string, !mode.isEmpty,
        advertised["modes"]?.array?.contains(.string(mode)) == true
      {
        ocr["mode"] = .string(mode)
      }
      // LightOn trains grounding as a mode. A retained toggle from another
      // OCR family must not override its selected Plain/Automatic mode.
      if fields["ocrRegions"]?.bool == true, advertised["grounding"]?.bool == true,
        advertised["modes"]?.array?.contains(.string("grounding")) != true
      {
        ocr["grounding"] = .bool(true)
      }
    }
    body["ocr"] = ocr.isEmpty ? nil : .object(ocr)
    // Match the web's explicit opt-in, but never send this local extension to
    // an endpoint that does not advertise it (not even forensics: off).
    body["forensics"] =
      capability["forensics"]?.object == nil
      ? nil
      : .string(fields["forensicsEnabled"]?.bool == true ? "on" : "off")
    body["file_metadata"] = fields["fileMetadataEnabled"]?.bool == false ? .string("off") : nil

    // Only an actual extraction request suppresses tools; a stale OCR mode
    // must not silently remove tools after switching to another model.
    let last = input.last { $0["role"]?.string == "user" }
    let text =
      (last?["content"]?.string
      ?? last?["content"]?.array?.first { $0["type"]?.string == "input_text" }?["text"]?.string
      ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    return !ocr.isEmpty || isTask(text, capability: capability)
  }

  static func isTask(_ text: String, capability: O) -> Bool {
    let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
    let curated = [
      "<chart2csv>", "<chart2code>", "<chart2summary>", "<tables_json>", "<tables_html>",
      "<tables_otsl>",
    ]
    return
      curated.contains(text)
      || capability["task_tags"]?.array?.contains { $0["tag"]?.string == text } == true
  }
}
