import Foundation

/// Web Studio's cloudVendor + cloud display contract. Shared by the native
/// catalog and conversation runtime so raw organization slugs never reach
/// the artwork lookup. A hosting provider is not the model's maker.
public enum CloudModelIdentity {
  private static let organizations: [String: String] = [
    "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "meta": "Meta",
    "meta-llama": "Meta", "deepseek": "DeepSeek", "tencent": "Tencent", "xiaomi": "Xiaomi",
    "z-ai": "Z.ai", "nvidia": "NVIDIA", "mistral": "Mistral", "mistralai": "Mistral",
    "moonshotai": "Moonshot", "x-ai": "xAI", "qwen": "Alibaba", "alibaba": "Alibaba",
    "perplexity": "Perplexity", "baidu": "Baidu", "bytedance": "ByteDance", "minimax": "MiniMax",
    "poolside": "Poolside", "ibm": "IBM", "cohere": "Cohere", "huggingface": "Hugging Face",
    "openrouter": "OpenRouter", "aleph-alpha": "Aleph Alpha",
  ]
  public static func vendor(_ id: String) -> String? {
    let s = id.lowercased()
    if let slash = s.firstIndex(of: "/"), slash != s.startIndex {
      let org = String(s[..<slash].drop(while: { $0 == "~" }))
      if let vendor = organizations[org] { return vendor }
    }
    if s.contains("gpt") || s.range(of: #"^o[134]\b"#, options: .regularExpression) != nil {
      return "OpenAI"
    }
    for (needles, vendor) in [
      (["claude"], "Anthropic"), (["gemini", "gemma"], "Google"), (["qwen"], "Alibaba"),
      (["granite"], "IBM"), (["llama"], "Meta"), (["grok"], "xAI"), (["kimi"], "Moonshot"),
      (["glm"], "Z.ai"), (["deepseek"], "DeepSeek"), (["mistral", "mixtral"], "Mistral"),
      (["kolibri"], "Aleph Alpha"),
    ] { if needles.contains(where: s.contains) { return vendor } }
    return nil
  }
  public static func resolve(
    id: String, display: String?, kind: String = "", provider: String? = nil
  ) -> (name: String, vendor: String?) {
    let inferred = vendor(id)
    let maker = inferred ?? (kind == "openai" ? "OpenAI" : kind == "anthropic" ? "Anthropic" : nil)
    var name = display ?? id
    if inferred != nil {
      name = name.replacingOccurrences(
        of: #"^[^:]{2,24}:\s+"#, with: "", options: .regularExpression)
    }
    if let provider, !provider.isEmpty { name += " (\(provider))" }
    return (name, maker)
  }
  public static func bareModel(_ id: String) -> String {
    guard id.hasPrefix("cloud:") else { return id }
    let rest = id.dropFirst(6)
    guard let colon = rest.firstIndex(of: ":") else { return String(rest) }
    return String(rest[rest.index(after: colon)...].split(separator: "@", maxSplits: 1).first ?? "")
  }
  public static func fallbackName(_ id: String) -> String {
    let base =
      bareModel(id).split(whereSeparator: { $0 == "/" || $0 == "\\" }).last.map(String.init) ?? id
    var parts = base.replacingOccurrences(
      of: #"\.gguf$"#, with: "", options: [.regularExpression, .caseInsensitive]
    ).components(separatedBy: "-")
    while parts.count > 1,
      parts.last!.range(
        of: #"^(iq?\d|q\d|f16|bf16|fp\d|gguf|gptq|awq|int\d)"#,
        options: [.regularExpression, .caseInsensitive]) != nil
    { parts.removeLast() }
    return parts.joined(separator: " ")
  }
}
