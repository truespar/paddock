import AVFoundation
import Foundation
import PaddockClient
import PaddockConversationCore
import Testing

@testable import PaddockUI

@Suite("Native embedding workspace", .serialized) @MainActor
struct NativeEmbeddingsTests {
  @Test func movieImportDecodesOffscreenAndRejectsAnExceededBudget() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("frames.mov")
    let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64,
      ])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64,
      ])
    writer.add(input)
    #expect(writer.startWriting())
    writer.startSession(atSourceTime: .zero)
    var buffer: CVPixelBuffer?
    #expect(
      CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32BGRA, nil, &buffer)
        == kCVReturnSuccess)
    let pixels = try #require(buffer)
    CVPixelBufferLockBaseAddress(pixels, [])
    memset(CVPixelBufferGetBaseAddress(pixels), 127, CVPixelBufferGetDataSize(pixels))
    CVPixelBufferUnlockBaseAddress(pixels, [])
    for i in 0..<4 {
      let deadline = ContinuousClock.now.advanced(by: .seconds(5))
      while !input.isReadyForMoreMediaData && ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(2))
      }
      #expect(input.isReadyForMoreMediaData)
      #expect(adaptor.append(pixels, withPresentationTime: CMTime(value: Int64(i), timescale: 2)))
    }
    writer.endSession(atSourceTime: CMTime(value: 2, timescale: 1))
    input.markAsFinished()
    await writer.finishWriting()
    #expect(writer.status == .completed)
    let frames = try await NativeEmbeddingVideo.decode(url, byteBudget: 1024 * 1024)
    #expect(frames.count == 2)
    #expect(frames.allSatisfy { $0.png.starts(with: [137, 80, 78, 71]) })
    await #expect(throws: (any Error).self) {
      try await NativeEmbeddingVideo.decode(url, byteBudget: 1)
    }
    let cancelled = Task { try await NativeEmbeddingVideo.decode(url, byteBudget: 1024 * 1024) }
    cancelled.cancel()
    await #expect(throws: CancellationError.self) { try await cancelled.value }
  }
  @Test func videoSamplingIsBoundedAndPreservesTheWholeTimeline() throws {
    #expect(try NativeEmbeddingVideo.sampleTimes(duration: 0.3) == [0])
    #expect(try NativeEmbeddingVideo.sampleTimes(duration: 3.9) == [0, 1, 2])
    let long = try NativeEmbeddingVideo.sampleTimes(duration: 3600)
    #expect(long.count == 32)
    #expect(long.first == 0 && long.last == 3599)
    #expect(zip(long, long.dropFirst()).allSatisfy { $0 < $1 })
    #expect(throws: (any Error).self) { try NativeEmbeddingVideo.sampleTimes(duration: .infinity) }
    #expect(throws: (any Error).self) { try NativeEmbeddingVideo.sampleTimes(duration: 0) }
  }
  @Test func videoBodyUsesVideoFramesNotAnImageAlbum() {
    let video = NativeEmbeddingsModel.Medium(
      name: "clip.mov", kind: "video", data: Data(), mime: "video/quicktime", format: "mov",
      frames: [.init(timestamp: 0, png: Data([1, 2, 3])), .init(timestamp: 1, png: Data([4]))])
    #expect(video.byteCount == 4)
    let part = video.item["content"]?.array?.first
    #expect(part?["type"]?.string == "input_video")
    #expect(part?["input_video"]?["add_timestamps"]?.bool == false)
    #expect(part?["input_video"]?["frames"]?.array?.count == 2)
  }
  private func json(_ s: String) throws -> ConversationValue {
    try JSONDecoder().decode(ConversationValue.self, from: Data(s.utf8))
  }
  private func model() -> NativeEmbeddingsModel {
    let m = NativeEmbeddingsModel(client: NativeManager())
    m.api = { path, _, _ in
      if path == "api/runners" {
        return try json(
          #"[{"port":1234,"pid":1,"status":"ok","embedder":"eg2","vendor":"Google"},{"port":1235,"pid":2,"status":"ok","model":"chat"}]"#
        )
      }
      if path.hasSuffix("/server") {
        return try json(
          #"{"reranker":false,"embedder_inputs":["text","image","audio"],"embedder_dimensions":[128,256,512,768],"embedder_tasks":["query","document"]}"#
        )
      }
      return try json(
        #"{"data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}],"usage":{"total_tokens":6}}"#
      )
    }
    return m
  }
  @Test func discoversOnlyEncodersAndUsesFunctionalModalities() async throws {
    let m = model()
    await m.refresh()
    #expect(m.endpoints.count == 1 && m.port == 1234)
    #expect(m.current?.inputs == Set(["text", "image", "audio"]))
    #expect(m.current?.vendor == "Google" && m.current?.dimensions == [128, 256, 512, 768])
    let runner = try JSONDecoder().decode(
      RunnerInfo.self,
      from: Data(
        #"{"port":1234,"pid":1,"status":"ok","embedder":"eg2","endpoint":"http://127.0.0.1:1234"}"#
          .utf8))
    #expect(runner.studioActionTitle == "Open Embeddings")
    m.api = { _, _, _ in .array([]) }
    await m.refresh()
    #expect(m.current == nil && !m.canRun && m.port == 0)
  }
  @Test func sendsTaskAndDimensionsAndRestoresResponseOrder() async throws {
    let m = model()
    await m.refresh()
    m.text = "red\nblue\n"
    m.dimensions = 128
    m.task = "query"
    let previous = m.api
    m.api = { path, method, body in
      #expect(path == "api/runners/1234/v1/embeddings" && method == "POST")
      #expect(body?["dimensions"]?.integer == 128 && body?["task"]?.string == "query")
      #expect(body?["input"] == .array([.string("red"), .string("blue")]))
      return try await previous(path, method, body)
    }
    m.run()
    await m.settle()
    #expect(m.error == nil && m.result?.labels == ["red", "blue"])
    #expect(m.result?.vectors == [[1, 0], [0, 1]] && m.result?.similarity == [[1, 0], [0, 1]])
    #expect(m.result?.tokens == 6)
    m.text = "different draft"
    #expect(m.result?.labels == ["red", "blue"])
  }
  @Test func refusesDuplicateIndicesAndInconsistentDimensions() async throws {
    let m = model()
    await m.refresh()
    let endpoint = try #require(m.current)
    for response in [
      #"{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[1]}]}"#,
      #"{"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[1,0]}]}"#,
      #"{"data":[{"index":2,"embedding":[1]}]}"#,
    ] {
      #expect(throws: (any Error).self) {
        try NativeEmbeddingsModel.decode(
          json(response), endpoint: endpoint, labels: ["a", "b"], milliseconds: 1)
      }
    }
  }
  @Test func rerankerUsesItsOwnEndpointAndQuery() async throws {
    let m = model()
    let previous = m.api
    m.api = { path, method, body in
      if path.hasSuffix("/server") { return try json(#"{"reranker":true}"#) }
      if path.hasSuffix("/rerank") {
        #expect(method == "POST" && body?["query"]?.string == "find blue")
        #expect(body?["documents"] == .array([.string("red"), .string("blue")]))
        #expect(body?["input"] == nil && body?["dimensions"] == nil)
        return try json(
          #"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#)
      }
      return try await previous(path, method, body)
    }
    await m.refresh()
    m.text = "red\nblue"
    m.query = "find blue"
    m.run()
    await m.settle()
    #expect(m.result?.rankings.first?.index == 1 && m.result?.vectors.isEmpty == true)
  }
  @Test func cancelledRequestCannotReplaceTheResult() async throws {
    let m = model()
    await m.refresh()
    m.text = "red\nblue"
    m.api = { _, _, _ in
      try await Task.sleep(for: .seconds(30))
      return .null
    }
    m.run()
    m.cancel()
    await m.settle()
    #expect(!m.busy && m.error == nil && m.result == nil && m.text == "red\nblue")
  }
  @Test func mediaUsesTheSameContentPartShapeAsWebStudio() {
    let image = NativeEmbeddingsModel.Medium(
      name: "a.png", kind: "image", data: Data([1, 2, 3]), mime: "image/png", format: "png")
    #expect(
      image.item["content"]?.array?.first?["image_url"]?["url"]?.string
        == "data:image/png;base64,AQID")
    let audio = NativeEmbeddingsModel.Medium(
      name: "a.wav", kind: "audio", data: Data([1, 2, 3]), mime: "audio/wav", format: "wav")
    #expect(audio.item["content"]?.array?.first?["input_audio"]?["data"]?.string == "AQID")
  }
  @Test func inputsAreBoundedAndMediaImportRefusesNonfiles() async {
    let m = model()
    await m.refresh()
    m.text = Array(repeating: "x", count: 33).joined(separator: "\n")
    #expect(!m.canRun)
    m.text = ""
    m.importFiles([URL(string: "https://example.com/image.png")!])
    await m.settle()
    #expect(m.error != nil && m.media.isEmpty && !m.importing)
  }
}
