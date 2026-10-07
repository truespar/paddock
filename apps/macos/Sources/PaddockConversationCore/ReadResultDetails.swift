import Foundation

extension ReadResponse {
  public struct Skip: Decodable, Sendable {
    public let because: String
    public let was: String?
    public let wanted: [String]
    public var explanation: String {
      let expected = wanted.count == 1 ? wanted[0] : "one of " + wanted.joined(separator: ", ")
      return
        "\(because) \(was.map { "answered " + $0 } ?? "was not asked"); asked only if it is \(expected)"
    }
  }
  public struct Thought: Decodable, Sendable {
    public let text: String
    public let tokens: Int
    public let closed: Bool
    public let ms: Double
  }
  public struct Thoughts: Decodable, Sendable {
    public let values: [Thought]
    public init(from decoder: any Decoder) throws {
      let c = try decoder.singleValueContainer()
      if let array = try? c.decode([Thought].self) {
        values = array
      } else {
        values = [try c.decode(Thought.self)]
      }
    }
  }
}
