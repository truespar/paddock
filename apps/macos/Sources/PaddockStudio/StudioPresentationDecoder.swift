import PaddockConversationCore

/// Serial, off-main decoding with structural reuse of unchanged fields/rows.
actor StudioPresentationDecoder {
  private let cache = ConversationDecodingCache()
  private var closed = false
  func decode(_ fields: [String: ConversationValue]) throws -> StudioState {
    guard !closed else { throw ConversationFailure.closed }
    try Task.checkCancellation()
    return try cache.decode(StudioState.self, from: .object(fields))
  }
  func reclaim() { cache.reclaim() }
  func close() {
    closed = true
    cache.reclaim()
  }
}
