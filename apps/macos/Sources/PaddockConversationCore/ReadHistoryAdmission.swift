import Foundation

/// Lossless rollover, using the same document schema as Web Studio. The caller
/// must durably save the previous document before admitting the next result.
public enum ReadHistoryAdmission {
  public static let byteLimit = 16 * 1024 * 1024
  public static let requestLimit = 12 * 1024 * 1024
  public struct Result: Sendable {
    public let document: ReadHistoryDocument
    public let rolledOver: Bool
  }
  public struct Overflow: Error, Sendable {
    public let document: ReadHistoryDocument
  }
  public static func append(
    _ run: ConversationValue, pictures: [ReadPicture], title: String,
    to previous: ReadHistoryDocument?, byteLimit: Int = byteLimit
  ) throws -> Result {
    func document(_ previous: ReadHistoryDocument?) -> ReadHistoryDocument {
      var fields =
        previous?.value.object ?? [
          "id": .string(UUID().uuidString), "title": .string(title),
          "createdAt": run["at"] ?? .number(0),
        ]
      fields["updatedAt"] = run["at"]
      fields["model"] = run["model"]
      fields["runs"] = .array((previous?.runs ?? []) + [run])
      var images = fields["images"]?.object ?? [:]
      for picture in pictures { images[picture.ref] = .string(picture.url) }
      if !images.isEmpty { fields["images"] = .object(images) }
      return ReadHistoryDocument(value: .object(fields))
    }
    let candidate = document(previous)
    if candidate.runs.count <= 20, try candidate.json.utf8.count <= byteLimit {
      return Result(document: candidate, rolledOver: false)
    }
    let single = document(nil)
    guard try single.json.utf8.count <= byteLimit else { throw Overflow(document: single) }
    return Result(document: single, rolledOver: previous != nil)
  }
}
