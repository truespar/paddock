import CoreGraphics
import Foundation
import ImageIO
import PaddockConversationCore
import Testing
import UniformTypeIdentifiers

@testable import PaddockUI

@Suite("Reads file import", .serialized) @MainActor
struct NativeReadImportTests {
  @Test func onePickerUsesTheCorrectFiltersAndSelectionMode() {
    #expect(NativeReadImport.state.contentTypes == [.item])
    #expect(NativeReadImport.questions.contentTypes == [.json])
    #expect(NativeReadImport.images.contentTypes == [.image])
    #expect(!NativeReadImport.state.allowsMultipleSelection)
    #expect(!NativeReadImport.questions.allowsMultipleSelection)
    #expect(NativeReadImport.images.allowsMultipleSelection)
  }

  @Test func cancellationAndEmptySelectionPreserveTheDraft() async {
    let model = NativeReadsTests().model()
    model.draft.state = "Keep my draft"
    model.stateError = "Previous state error"
    model.questionsError = "Previous question error"
    let before = model.draft
    for kind in [NativeReadImport.state, .questions, .images] {
      await model.importSelection(.failure(CocoaError(.userCancelled)), kind: kind)
      await model.importSelection(.success([]), kind: kind)
    }
    #expect(model.draft == before && !model.importing)
    #expect(model.stateError == "Previous state error")
    #expect(model.questionsError == "Previous question error")
  }

  @Test func pickerFailuresAreVisibleAtTheRelevantField() async {
    let model = NativeReadsTests().model()
    let failure = NSError(
      domain: "PickerFixture", code: 1,
      userInfo: [NSLocalizedDescriptionKey: "The file picker could not read this location."])
    await model.importSelection(.failure(failure), kind: .state)
    #expect(model.stateError == failure.localizedDescription && model.questionsError == nil)
    model.stateError = nil
    await model.importSelection(.failure(failure), kind: .questions)
    #expect(model.questionsError == failure.localizedDescription && model.stateError == nil)
    await model.importSelection(.failure(failure), kind: .images)
    #expect(model.stateError == failure.localizedDescription)
  }

  @Test func textAndQuestionFilesReachDifferentImportPaths() async throws {
    let root = try directory()
    defer { try? FileManager.default.removeItem(at: root) }
    let text = root.appending(path: "message.txt")
    let json = root.appending(path: "questions.json")
    try "A local text fixture.".write(to: text, atomically: true, encoding: .utf8)
    try #"{"questions":{"urgent":{"type":"noul","instructions":"Does this need action today?"}}}"#
      .write(to: json, atomically: true, encoding: .utf8)
    let model = NativeReadsTests().model()
    await model.importSelection(.success([text]), kind: .state)
    #expect(model.draft.state == "A local text fixture." && model.fileName == "message.txt")
    await model.importSelection(.success([json]), kind: .questions)
    #expect(model.draft.questions.first?.questionID == "urgent")
    #expect(model.draft.state == "A local text fixture.")
    #expect(model.error == nil && model.questionsError == nil && !model.importing)
    let before = model.draft
    await model.importSelection(.success([text, json]), kind: .state)
    #expect(model.draft == before && model.stateError == "Choose one file.")
  }

  @Test func imageSelectionsAttachInOrderWithoutReplacingTheText() async throws {
    let root = try directory()
    defer { try? FileManager.default.removeItem(at: root) }
    let first = try image(root.appending(path: "first.png"))
    let second = try image(root.appending(path: "second.png"))
    let model = NativeReadsTests().model()
    let originalAPI = model.api
    model.api = { path, method, body, query in
      if path.hasSuffix("1234/server") {
        return .object([
          "structured_read": .object(["canvas_width": .number(256), "images": .bool(true)])
        ])
      }
      return try await originalAPI(path, method, body, query)
    }
    await model.refresh()
    model.draft.state = "Classify these pictures"
    await model.importSelection(.success([first, second]), kind: .images)
    #expect(model.draft.images.map(\.name) == ["first.png", "second.png"])
    #expect(model.draft.state == "Classify these pictures" && model.stateError == nil)
    await model.importSelection(.success([first]), kind: .images)
    #expect(model.draft.images.count == 3 && model.draft.state == "Classify these pictures")
    var extracted = false
    model.api = { path, method, body, _ in
      #expect(path == "api/runners/1234/extract" && method == "POST")
      #expect(body?["filename"] == .string("first.png"))
      #expect(body?["file_metadata"] == .string("off"))
      extracted = true
      return .object(["text": .string("Extracted image text")])
    }
    await model.importSelection(.success([first]), kind: .state)
    #expect(extracted && model.draft.state == "Extracted image text")
    #expect(model.draft.images.map(\.name) == ["first.png", "second.png", "first.png"])
    #expect(model.error == nil && !model.importing)
  }

  private func directory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appending(path: "reads-import-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
  }
  private func image(_ url: URL) throws -> URL {
    let context = try #require(
      CGContext(
        data: nil, width: 8, height: 8, bitsPerComponent: 8,
        bytesPerRow: 32, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    let pixels = try #require(context.makeImage())
    let destination = try #require(
      CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, pixels, nil)
    #expect(CGImageDestinationFinalize(destination))
    return url
  }
}
