import Foundation

extension ReadDraft {
  /// A runnable example with a key placeholder, never a credential export.
  /// Image requests use file parts, not megabytes of base64 in the UI.
  public func curl(port: UInt16, model: String) throws -> String {
    let endpoint = "curl http://localhost:\(port)/v1/systemone \\\n"
    let auth = "  -H 'Authorization: Bearer <api key>'"
    if images.isEmpty {
      return endpoint + "  -H 'Content-Type: application/json' \\\n" + auth
        + " \\\n  -d @- <<'JSON'\n" + (try orderedJSON(model: model)) + "\nJSON"
    }
    var text = self
    text.images = []
    let files = images.enumerated().map { index, image in
      // curl's multipart grammar interprets quotes/commas/semicolons itself,
      // independently of shell quoting. Remove those and control characters.
      let clean = String(
        image.name.unicodeScalars.filter {
          !CharacterSet.controlCharacters.contains($0) && !"\";,".unicodeScalars.contains($0)
        })
      let name = clean.isEmpty ? "image-\(index + 1).jpg" : clean
      let part = "image=@" + name
      return "  -F '" + part.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
    return "cat > request.json <<'JSON'\n" + (try text.orderedJSON(model: model))
      + "\nJSON\n" + endpoint + auth
      + " \\\n  -F 'request=<request.json;type=application/json' \\\n"
      + files.joined(separator: " \\\n")
  }
}
