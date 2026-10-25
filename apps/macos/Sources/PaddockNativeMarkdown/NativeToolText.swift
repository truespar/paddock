import Foundation

/// Shared by the visible disclosure and the off-main Find index. A closed
/// tool's JSON is not decoded again on every transcript publication.
public enum NativeToolText {
  public static func pretty(_ value: Any) -> String {
    guard
      let data = try? JSONSerialization.data(
        withJSONObject: value,
        options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed, .withoutEscapingSlashes])
    else { return String(describing: value) }
    return String(decoding: data, as: UTF8.self)
  }
  public static func arguments(_ raw: String) -> String {
    guard
      let value = try? JSONSerialization.jsonObject(
        with: Data(raw.utf8), options: .fragmentsAllowed)
    else { return raw }
    guard let fields = value as? [String: Any] else { return pretty(value) }
    return fields.keys.sorted().map { key in
      if let text = fields[key] as? String {
        return text.contains("\n")
          ? "\(key):\n  " + text.replacingOccurrences(of: "\n", with: "\n  ") : "\(key): \(text)"
      }
      return "\(key): \(pretty(fields[key]!))"
    }.joined(separator: "\n")
  }
  public static func output(_ raw: String) -> String {
    guard
      let value = try? JSONSerialization.jsonObject(
        with: Data(raw.utf8), options: .fragmentsAllowed)
    else { return raw }
    if let parts = value as? [[String: Any]], !parts.isEmpty,
      parts.allSatisfy({ $0["text"] is String })
    {
      return parts.compactMap { $0["text"] as? String }.map { text in
        (try? JSONSerialization.jsonObject(with: Data(text.utf8), options: .fragmentsAllowed)).map(
          pretty) ?? text
      }.joined(separator: "\n")
    }
    return pretty(value)
  }
}
