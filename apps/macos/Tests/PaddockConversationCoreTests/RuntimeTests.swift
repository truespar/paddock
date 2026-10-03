import Foundation
import PaddockClient
import Synchronization
import Testing

@testable import PaddockConversationCore

@Suite("Native Studio cutover", .serialized)
struct RuntimeTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  func fixture(
    speechFeatures: Bool = false, history: [O] = [], localCapabilities: O? = nil,
    imagePeerCapabilities: O? = nil,
    cloudContext: Int? = 131072, cloudOutput: Int? = 2048,
    openPDF: @escaping @Sendable (Data) async throws -> NativeDocumentSource = { _ in
      throw ConversationFailure.invalid("Test PDF loader not configured")
    }
  )
    async throws -> NativeStudioRuntime
  {
    RuntimeProtocol.state.withLock {
      $0 = .init()
      $0.speechFeatures = speechFeatures
      $0.localCapabilities = localCapabilities
      $0.imagePeerCapabilities = imagePeerCapabilities
      $0.cloudContext = cloudContext
      $0.cloudOutput = cloudOutput
      $0.documents = Dictionary(uniqueKeysWithValues: history.map { ($0["id"]!.string!, $0) })
    }
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43210", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [RuntimeProtocol.self]
    let transport = try NativeConversationTransport(host: host, configuration: configuration)
    let runtime = NativeStudioRuntime(transport: transport, openPDF: openPDF) { _ in }
    try await runtime.start()
    return runtime
  }
  func settled(_ runtime: NativeStudioRuntime) async throws -> O {
    for _ in 0..<500 {
      let state = await runtime.presentation()
      if state["busy"]?.bool == false { return state }
      try await Task.sleep(for: .milliseconds(10))
    }
    throw ConversationFailure.invalid("Native runtime did not settle")
  }
  @Test func nativeImagesUseImageAPIAndSurviveReopening() async throws {
    let runtime = try await fixture(localCapabilities: [
      "image_generation": .object([
        "stream": .bool(true), "max_n": .number(4), "max_steps": .number(100),
        "max_partial_images": .number(3), "default_steps": .number(40),
        "default_size": .string("1024x1024"), "output_formats": .array([.string("png")]),
      ])
    ])
    _ = try await runtime.command("models", ["ids": .array([.string("local")])])
    #expect(await runtime.presentation()["composer"]?["imageMode"] == .bool(true))
    _ = try await runtime.command(
      "preferencesSave",
      [
        "changes": .object(["maxTokens": .number(1)]), "expected": .object(["maxTokens": .null]),
      ])
    _ = try await runtime.command("settings", ["imageParams": .object(["quality": .string("low")])])
    _ = try await runtime.command("send", ["text": .string("A red apple")])
    _ = try await settled(runtime)
    let fields = try #require(await runtime.currentFields())
    let response = try #require(fields["messages"]?.array?.last)
    #expect(response["error"] == nil)
    #expect(response["imageGen"]?["steps"] == .number(20))
    #expect(response["imageGen"]?["previews"] == .number(1))
    let picture = try #require(response["content"]?.array?.first)
    #expect(picture["type"] == .string("image"))
    #expect(picture["attachmentId"]?.string != nil && picture["dataURL"] == nil)
    let requests = RuntimeProtocol.state.withLock { $0.requests }
    #expect(requests.contains { $0["path"]?.string == "/api/runners/12481/v1/images/generations" })
    #expect(
      requests.first { $0["path"]?.string?.hasSuffix("/images/generations") == true }?["body"]?[
        "max_output_tokens"] == nil)
    #expect(!requests.contains { $0["path"]?.string?.hasSuffix("/responses") == true })
    let encoded = try JSONEncoder().encode(fields)
    #expect(!String(decoding: encoded, as: UTF8.self).contains("b64_json"))
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": fields["id"]!])
    #expect(
      await runtime.currentFields()?["messages"]?.array?.last?["content"]?.array?.first == picture)
    _ = try await runtime.command(
      "openDocument", ["messageId": response["id"]!, "attachmentId": picture["attachmentId"]!])
    #expect(await runtime.currentFields()?["activeDocId"] == response["id"])
    #expect(
      await runtime.presentation()["settings"]?["lastImageSeed"] == response["imageGen"]?["seed"])
    _ = try await runtime.command("settings", ["imageParams": .null])
    #expect(await runtime.currentFields()?["imageParams"] == nil)
    #expect(
      await runtime.presentation()["settings"]?["imageParams"]
        == .object(NativeImageGeneration.defaults))
    await runtime.close()
  }
  @Test func completedImageSurvivesAttachmentStoreFailureAndReopens() async throws {
    let runtime = try await fixture(localCapabilities: [
      "image_generation": .object([
        "stream": .bool(true), "max_n": .number(4), "max_steps": .number(100),
        "max_partial_images": .number(3), "default_steps": .number(40),
        "default_size": .string("1024x1024"), "output_formats": .array([.string("png")]),
      ])
    ])
    RuntimeProtocol.state.withLock { $0.failImageStore = true }
    _ = try await runtime.command("models", ["ids": .array([.string("local")])])
    _ = try await runtime.command("send", ["text": .string("A red apple")])
    _ = try await settled(runtime)
    let fields = try #require(await runtime.currentFields())
    let response = try #require(fields["messages"]?.array?.last)
    #expect(response["error"] == nil)
    let part = try #require(response["content"]?.array?.first)
    #expect(part["attachmentId"] == .string(""))
    #expect(part["dataUrl"]?.string?.hasPrefix("data:image/png;base64,") == true)
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": fields["id"]!])
    let projected = await runtime.presentation()
    let picture = try #require(
      projected["nativeTranscript"]?["messages"]?.array?.last?["pictures"]?.array?.first)
    #expect(picture["dataURL"] == part["dataUrl"])
    #expect(picture["id"]?.string?.hasPrefix("inline-") == true)
    _ = try await runtime.command(
      "openDocument",
      [
        "messageId": response["id"]!, "attachmentId": picture["id"]!,
      ])
    #expect(await runtime.presentation()["nativeDocument"]?["id"] == picture["id"])
    #expect(
      await runtime.currentFields()?["messages"]?.array?.last?["content"]?.array?.first == part)
    await runtime.close()
  }
  @Test func repetitiveOCRIsSavedForReviewWithoutLosingText() async throws {
    let runtime = try await fixture(localCapabilities: [
      "document_parser": .bool(true), "vision": .bool(true), "max_ctx": .number(8192),
    ])
    let loop = String(repeating: "INVOICE 123\n", count: 500)
    RuntimeProtocol.state.withLock { $0.documentAnswer = loop }
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("scan"), "name": .string("Scan.jpg"), "mime": .string("image/jpeg"),
        "size": .number(10),
      ])
    _ = try await runtime.command(
      "send", ["attachments": .array([.object(["id": .string("scan")])])])
    _ = try await settled(runtime)
    let page = try #require(
      await runtime.currentFields()?["messages"]?.array?.last?["docRun"]?["pages"]?.array?.first)
    #expect(page["state"] == .string("review"))
    #expect(page["text"] == .string(loop))
    #expect((page["repetitionRatio"]?.double ?? 0) > 4.5)
    #expect(page["note"]?.string?.contains("original") == true)
    await runtime.close()
  }
  @Test func documentImagesFanOutAndSelectedOriginalSurvivesReopening() async throws {
    let runtime = try await fixture(localCapabilities: [
      "document_parser": .bool(true), "vision": .bool(true), "max_ctx": .number(8192),
      "ocr": .object(["modes": .array([.string("markdown")]), "grounding": .bool(true)]),
    ])
    _ = try await runtime.command(
      "settings", ["ocrMode": .string("markdown"), "ocrRegions": .bool(true)])
    for id in ["scan-one", "scan-two"] {
      _ = try await runtime.command(
        "stage",
        [
          "id": .string(id), "mime": .string("image/jpeg"),
          "name": .string("\(id).jpg"), "size": .number(10),
        ])
    }
    _ = try await runtime.command(
      "send",
      [
        "text": .string("Free text cannot enter a fixed-vocabulary OCR decoder"),
        "attachments": .array([
          .object(["id": .string("scan-one")]), .object(["id": .string("scan-two")]),
        ]),
      ])
    _ = try await settled(runtime)
    let first = try #require(await runtime.currentFields())
    let source = try #require(first["messages"]?.array?.first?["id"])
    #expect(first["activeDocId"] == source)
    let pages = try #require(first["messages"]?.array?.last?["docRun"]?["pages"]?.array)
    #expect(pages.count == 2 && pages.allSatisfy { $0["state"] == .string("done") })
    #expect(pages.allSatisfy { $0["text"] == .string("Local final tail 🦊") })
    #expect(first["messages"]?.array?.last?["usage"]?["promptTokens"]?.integer == 20)
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
    }
    #expect(requests.count == 2)
    for request in requests {
      let body = request["body"]
      #expect(body?["input"]?.array?.count == 1)
      #expect(body?["input"]?.array?.first?["content"]?.array?.count == 1)
      #expect(
        body?["input"]?.array?.first?["content"]?.array?.first?["type"] == .string("input_image"))
      #expect(body?["ocr"] == .object(["mode": .string("markdown"), "grounding": .bool(true)]))
      #expect(body?["tools"] == nil && body?["instructions"] == nil)
      #expect(body?["include"] == .array([.string("message.output_text.logprobs")]))
    }
    // A fresh document selects itself, and opening the earlier original makes
    // that document the target of the next text-only reading request.
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("scan-new"), "mime": .string("image/jpeg"),
        "name": .string("New.jpg"), "size": .number(10),
      ])
    _ = try await runtime.command(
      "send", ["attachments": .array([.object(["id": .string("scan-new")])])])
    _ = try await settled(runtime)
    #expect(await runtime.currentFields()?["activeDocId"] != source)
    _ = try await runtime.command(
      "openDocument", ["messageId": source, "attachmentId": .string("scan-one")])
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": first["id"]!])
    #expect(await runtime.currentFields()?["activeDocId"] == source)
    _ = try await runtime.command("send", ["text": .string("Read again")])
    _ = try await settled(runtime)
    let final = try #require(await runtime.currentFields())
    #expect(final["messages"]?.array?.last?["docRun"]?["sourceId"] == source)
    #expect(final["messages"]?.array?.last?["docRun"]?["pages"]?.array?.count == 2)
    #expect(
      RuntimeProtocol.state.withLock {
        $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }.count
      } == 5)
    await runtime.close()
  }
  @Test func selectedPDFPagesRasterizeOneAtATimeAndCancellationKeepsCompletedPages() async throws {
    let rendered = Mutex<[Int]>([])
    let opened = Mutex(0)
    let runtime = try await fixture(
      localCapabilities: [
        "document_parser": .bool(true), "vision": .bool(true), "max_ctx": .number(8192),
        "pdf": .object(["max_pages": .number(40)]),
      ],
      openPDF: { bytes in
        #expect(bytes == Data("{}".utf8))
        opened.withLock { $0 += 1 }
        return NativeDocumentSource(pageCount: 200) { page, maxPixels in
          let previous = rendered.withLock { pages in
            let count = pages.count
            pages.append(page)
            return count
          }
          #expect(maxPixels == 1_500_000)
          #expect(
            RuntimeProtocol.state.withLock {
              $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }.count
            } == previous)
          if page == 198 { try await Task.sleep(for: .seconds(30)) }
          return NativeRasterPage(
            image: [
              "type": .string("input_image"),
              "image_url": .string("data:image/jpeg;base64,cGFnZQ=="),
            ], width: 100, height: 200)
        }
      })
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("book"), "mime": .string("application/pdf"),
        "name": .string("Book.pdf"), "size": .number(2000), "pages": .number(200),
      ])
    _ = try await runtime.command(
      "send",
      [
        "attachments": .array([
          .object([
            "id": .string("book"), "from": .number(197), "to": .number(199),
          ])
        ])
      ])
    for _ in 0..<500 {
      if rendered.withLock({ $0.count }) == 2 { break }
      try await Task.sleep(for: .milliseconds(10))
    }
    #expect(rendered.withLock { $0 } == [197, 198])
    _ = try await runtime.command("stop")
    _ = try await settled(runtime)
    let saved = try #require(await runtime.currentFields())
    let pages = try #require(saved["messages"]?.array?.last?["docRun"]?["pages"]?.array)
    #expect(pages.map { $0["page"]?.integer } == [197, 198, 199])
    #expect(pages.map { $0["state"]?.string } == ["done", "error", "error"])
    #expect(pages[0]["text"] == .string("Local final tail 🦊"))
    #expect(pages[1]["note"] == .string("Stopped") && pages[2]["note"] == .string("Stopped"))
    #expect(saved["messages"]?.array?.last?["usage"] == nil)
    #expect(opened.withLock { $0 } == 1)
    #expect(rendered.withLock { $0 } == [197, 198])
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": saved["id"]!])
    #expect(
      await runtime.currentFields()?["messages"]?.array?.last?["docRun"]?["pages"]?.array == pages)
    await runtime.close()
  }
  @Test func imageOverflowPreservesDraftAndChecksCombinedImagesBeforeAnyRequest() async throws {
    let budget = try #require(VisionBudgetTests.fixture()["budgets"]?["bonsai"])
    let runtime = try await fixture(localCapabilities: [
      "vision": .bool(true), "max_ctx": .number(8192), "vision_budget": budget,
    ])
    let state = await runtime.presentation()
    #expect(state["capabilities"]?["imageLanes"]?.array?.first?["budget"] == budget)
    for id in ["one", "two"] {
      _ = try await runtime.command(
        "stage",
        [
          "id": .string(id), "mime": .string("image/jpeg"), "name": .string("Photo.jpg"),
          "size": .number(1000), "width": .number(6720), "height": .number(4480),
        ])
    }
    for choices: [V] in [
      [.object(["id": .string("one"), "detail": .string("high")])],
      [
        .object(["id": .string("one"), "detail": .string("auto")]),
        .object(["id": .string("two"), "detail": .string("auto")]),
      ],
    ] {
      await #expect(throws: ConversationFailure.self) {
        _ = try await runtime.command(
          "send", ["text": .string("Describe"), "attachments": .array(choices)])
      }
      #expect(await runtime.currentFields()?["messages"]?.array?.isEmpty == true)
      #expect(RuntimeProtocol.state.withLock { $0.documents.isEmpty })
      #expect(
        !RuntimeProtocol.state.withLock {
          $0.requests.contains { $0["path"]?.string?.contains("/responses") == true }
        })
    }
    // The same originals are still staged after refusal; resizing permits send.
    _ = try await runtime.command(
      "send",
      [
        "text": .string("Describe"),
        "attachments": .array([
          .object(["id": .string("one"), "detail": .string("low")]),
          .object(["id": .string("two"), "detail": .string("low")]),
        ]),
      ])
    _ = try await settled(runtime)
    #expect(await runtime.currentFields()?["messages"]?.array?.first?["content"]?.array?.count == 3)
    await runtime.close()
  }
  @Test func documentRefusalsOnlyRetryExplicitUnsupportedLogprobsAndKeepOtherPages() async throws {
    for refusal in [
      (400, "Unsupported include: message.output_text.logprobs", true),
      (429, "Temporarily rate limited", false),
      (400, "Image size exceeds the configured context", false),
    ] {
      let runtime = try await fixture(localCapabilities: [
        "document_parser": .bool(true), "vision": .bool(true), "max_ctx": .number(8192),
      ])
      RuntimeProtocol.state.withLock { $0.documentRefusals = [(refusal.0, refusal.1)] }
      for id in ["one", "two"] {
        _ = try await runtime.command(
          "stage",
          [
            "id": .string(id), "mime": .string("image/jpeg"),
            "name": .string("\(id).jpg"), "size": .number(10),
          ])
      }
      _ = try await runtime.command(
        "send",
        [
          "attachments": .array([
            .object(["id": .string("one")]), .object(["id": .string("two")]),
          ])
        ])
      let shown = try await settled(runtime)
      let doc = try #require(await runtime.currentFields())
      let reply = try #require(doc["messages"]?.array?.last)
      let pages = try #require(reply["docRun"]?["pages"]?.array)
      #expect(pages.map { $0["state"]?.string } == [refusal.2 ? "done" : "error", "done"])
      #expect((reply["usage"] != nil) == refusal.2)
      let calls = RuntimeProtocol.state.withLock {
        $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
      }
      #expect(calls.count == (refusal.2 ? 3 : 2))
      if refusal.2 { #expect(calls[1]["body"]?["include"] == nil) }
      #expect(shown["capabilities"]?["hasDocument"] == .bool(true))
      _ = try await runtime.command("newChat")
      #expect(
        await runtime.presentation()["composer"]?["inputIssue"]
          == .string("Attach an image or PDF to read"))
      await runtime.close()
    }
  }
  @Test func persistedDocumentOptionsAreGatedOnTheWireAfterSwitchingModels() async throws {
    let runtime = try await fixture(localCapabilities: [
      "vision": .bool(true), "max_ctx": .number(8192),
      "ocr": .object(["modes": .array([.string("markdown")]), "grounding": .bool(true)]),
      "forensics": .object(["vision": .bool(true)]),
    ])
    _ = try await runtime.command(
      "settings",
      [
        "ocrMode": .string("markdown"), "ocrRegions": .bool(true), "forensicsEnabled": .bool(true),
      ])
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("photo"), "mime": .string("image/jpeg"), "name": .string("Photo"),
        "size": .number(1000),
      ])
    _ = try await runtime.command(
      "send", ["text": .string("Read"), "attachments": .array([.object(["id": .string("photo")])])])
    _ = try await settled(runtime)
    let local = try #require(
      RuntimeProtocol.state.withLock {
        $0.requests.last { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
      })
    #expect(local["ocr"]?["mode"] == .string("markdown"))
    #expect(local["ocr"]?["grounding"] == .bool(true))
    #expect(local["forensics"] == .string("on"))
    #expect(local["tools"] == nil)
    _ = try await runtime.command("models", ["ids": .array([.string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("Continue as a normal conversation")])
    _ = try await settled(runtime)
    let cloud = try #require(
      RuntimeProtocol.state.withLock {
        $0.requests.last { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
      })
    #expect(cloud["ocr"] == nil && cloud["forensics"] == nil)
    #expect(cloud["tools"]?.array?.isEmpty == false)
    #expect(await runtime.currentFields()?["ocrMode"] == .string("markdown"))
    await runtime.close()
  }
  @Test func imageResizeChoiceSurvivesSavingReopeningAndResponsesInput() async throws {
    for detail in ["auto", "high", "low"] {
      let runtime = try await fixture()
      _ = try await runtime.command(
        "stage",
        [
          "id": .string("fixture-photo"), "name": .string("Photo.jpg"),
          "mime": .string("image/jpeg"), "size": .number(18_171_177),
          "width": .number(6720), "height": .number(4480),
        ])
      _ = try await runtime.command(
        "send",
        [
          "text": .string("Describe this photo"),
          "attachments": .array([
            .object(["id": .string("fixture-photo"), "detail": .string(detail)])
          ]),
        ])
      _ = try await settled(runtime)
      let doc = try #require(await runtime.currentFields())
      let part = try #require(doc["messages"]?.array?.first?["content"]?.array?.last)
      #expect(part["detail"]?.string == detail)
      #expect(part["attachmentId"]?.string == "fixture-photo")
      #expect(part["width"]?.integer == 6720 && part["height"]?.integer == 4480)
      #expect(part["size"]?.integer == 18_171_177)
      #expect(part["modelUrl"] == nil && part["dataUrl"] == nil)
      let body = try #require(
        RuntimeProtocol.state.withLock {
          $0.requests.last { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
        })
      let image = try #require(body["input"]?.array?.first?["content"]?.array?.last)
      #expect(image["type"]?.string == "input_image")
      #expect(image["detail"]?.string == detail)
      // The test transport's attachment bytes are an empty JSON object.
      // Resizing stays in Rust, so the native client preserves those bytes.
      #expect(image["image_url"]?.string == "data:image/jpeg;base64,e30=")
      _ = try await runtime.command("newChat")
      _ = try await runtime.command("open", ["id": doc["id"]!])
      let reopened = await runtime.currentFields()
      #expect(reopened?["messages"]?.array?.first?["content"]?.array?.last == part)
      await runtime.close()
    }
  }
  @Test func cloudHTTPFailureKeepsRecoveryMetadataAndIsNotAutomaticallyRetried() async throws {
    let runtime = try await fixture()
    RuntimeProtocol.state.withLock { $0.responseFailure = true }
    _ = try await runtime.command("models", ["ids": .array([.string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("Provider failure fixture")])
    let state = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    let raw = try #require(messages.last?["error"]?.string)
    let error = try JSONDecoder().decode(V.self, from: Data(raw.utf8))
    #expect(error["code"]?.integer == 429)
    #expect(error["metadata"]?["provider_name"]?.string == "DeepInfra")
    #expect(error["metadata"]?["action"]?.string == "openrouter_integrations")
    #expect(messages.last?["streaming"]?.bool != true)
    #expect(
      RuntimeProtocol.state.withLock {
        $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }.count
      } == 1)
    let id = try #require(state["conversation"]?["id"]?.string)
    #expect(
      RuntimeProtocol.state.withLock {
        $0.documents[id]?["messages"]?.array?.last?["error"]?.string
      } == raw)
    await runtime.close()
  }
  @Test func artifactPanelStateAndAvailableFilesSurviveRevisit() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Make a page")])
    let state = try await settled(runtime)
    let id = try #require(state["conversation"]?["id"]?.string)
    let artifact: V = .object([
      "id": .string("art_012345abcdef"), "kind": .string("html"), "title": .string("Page"),
      "model": .string("local"), "versions": .number(1), "updatedAt": .number(1),
    ])
    RuntimeProtocol.state.withLock { $0.artifacts = [artifact] }
    _ = try await runtime.command("refresh")
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.count == 1)
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == true)
    _ = try await runtime.command(
      "artifactsPane", ["open": .bool(false), "conversationId": .string(id)])
    #expect(
      RuntimeProtocol.state.withLock { $0.documents[id]?["artifactsPaneOpen"]?.bool } == false)
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": .string(id)])
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.count == 1)
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == false)
    _ = try await runtime.command("graphPanel", ["open": .bool(true)])
    _ = try await runtime.command(
      "artifactsPane", ["open": .bool(true), "conversationId": .string(id)])
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == true)
    #expect(await runtime.graphVisible == false)
    await #expect(throws: ConversationFailure.self) {
      _ = try await runtime.command(
        "artifactsPane", ["open": .bool(false), "conversationId": .string("stale-chat")])
    }
    RuntimeProtocol.state.withLock { $0.failSave = true }
    await #expect(throws: (any Error).self) {
      _ = try await runtime.command(
        "artifactsPane", ["open": .bool(false), "conversationId": .string(id)])
    }
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == true)
    await runtime.close()
  }
  @Test func completedArtifactRefreshIsCoalescedAndCannotLeakIntoAnotherChat() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Make a page")])
    _ = try await settled(runtime)
    RuntimeProtocol.state.withLock {
      $0.artifacts = [.object(["id": .string("art_012345abcdef"), "kind": .string("html")])]
    }
    let before = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/artifacts") == true }.count
    }
    for _ in 0..<8 { await runtime.scheduleArtifactRefresh() }
    try await Task.sleep(for: .milliseconds(200))
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.count == 1)
    #expect(
      RuntimeProtocol.state.withLock {
        $0.requests.filter { $0["path"]?.string?.hasSuffix("/artifacts") == true }.count
      } == before + 1)
    await runtime.scheduleArtifactRefresh()
    _ = try await runtime.command("newChat")
    try await Task.sleep(for: .milliseconds(200))
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.isEmpty == true)
    await runtime.close()
  }
  @Test func newChatSurfacesAboveLoadedUnpinnedHistoryWithoutRestartOrRefresh() async throws {
    var history: [O] = (0..<32).map { i in
      [
        "id": .string("old-\(i)"), "title": .string("Old \(i)"), "model": .string("local"),
        "updatedAt": .number(Decimal(i)), "pinned": .bool(false),
      ]
    }
    history.append([
      "id": .string("pinned"), "title": .string("Pinned"), "updatedAt": .number(0),
      "pinned": .bool(true),
    ])
    let runtime = try await fixture(history: history)
    _ = try await runtime.command("send", ["text": .string("Brand new chat")])
    let state = try await settled(runtime)
    let id = try #require(state["conversation"]?["id"]?.string)
    for rows in [state["history"]?.array, state["library"]?["rows"]?.array] {
      #expect(rows?.prefix(2).compactMap { $0["id"]?.string } == ["pinned", id])
    }
    // A follow-up save, title edit and explicit reload must retain that order.
    _ = try await runtime.command(
      "renameChat", ["id": .string(id), "title": .string("Renamed new chat")])
    _ = try await runtime.command("refresh")
    let refreshed = await runtime.presentation()
    #expect(
      refreshed["history"]?.array?.prefix(2).compactMap { $0["id"]?.string } == ["pinned", id])
    await runtime.close()
  }
  @Test func liveProjectionShowsNextUtteranceBeforeItsFinalMetadata() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("models", ["ids": .array([.string("speech")])])
    let mapping = try await runtime.beginLive()
    let mid = try #require(mapping["speech"])
    let meta: O = [
      "words": .array([
        .object([
          "word": .string("Hej"), "start": .number(0), "end": .number(0.5),
          "confidence": .number(0.3),
        ])
      ])
    ]
    try await runtime.liveUpdate(messageID: mid, text: "Hej nästa mening", transcript: meta)
    let state = await runtime.presentation()
    let words = try #require(
      state["nativeTranscript"]?["messages"]?.array?.last?["speech"]?["words"]?.array)
    #expect(words.compactMap { $0["word"]?.string } == ["Hej", "nästa", "mening"])
    #expect(words.allSatisfy { $0["start"] == nil && $0["confidence"] == nil })
    let doc = try #require(await runtime.currentFields())
    #expect(
      doc["messages"]?.array?.last?["transcript"]?.object == meta,
      "Presentation must not erase completed utterance metadata")
    await runtime.close()
  }
  @Test func compareUsesExactLocalAndCloudRoutesAndDurableTerminalTails() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    let id = UUID().uuidString
    let accepted = try await runtime.command("send", ["text": .string("Hello")], id: id)
    #expect(accepted["accepted"]?.bool == true)
    _ = try await runtime.command("send", ["text": .string("MUST NOT DUPLICATE")], id: id)
    let state = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    #expect(messages.count == 3)
    #expect(messages[1]["group"] == messages[2]["group"])
    #expect(state["nativeTranscript"]?["messages"]?.array?.count == 3)
    #expect(messages[1]["content"]?.array?.first?["text"]?.string == "Local final tail 🦊")
    #expect(messages[2]["content"]?.array?.first?["text"]?.string == "Cloud final tail 🦊")
    let requests = RuntimeProtocol.state.withLock { $0.requests }
    #expect(
      requests.contains {
        $0["path"]?.string == "/api/runners/12481/v1/responses"
          && $0["body"]?["model"]?.string == "local"
      })
    #expect(
      requests.contains {
        $0["path"]?.string == "/api/cloud/ep/v1/responses"
          && $0["body"]?["model"]?.string == "remote"
      })
    let saved = RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!] }
    #expect(saved?["messages"] == doc["messages"])
    await runtime.close()
  }
  @Test func runDetailsUseTheActualRequestSnapshotAndSurviveConfigurationChanges() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "settings", ["params": .object(["temperature": .number(0.4), "thinking": .bool(false)])])
    _ = try await runtime.command("send", ["text": .string("Run details")])
    _ = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let assistant = try #require(doc["messages"]?.array?.last)
    let run = try #require(assistant["run"]?.object)
    let request = try #require(
      RuntimeProtocol.state.withLock {
        $0.requests.first { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
      })
    #expect(run["params"]?["maxTokens"] == (request["max_output_tokens"] ?? .null))
    #expect(
      (request["max_output_tokens"]?.integer ?? 0) > 1024,
      "Model maximum must not fall through to the runner's API default")
    #expect(
      request["max_output_tokens"]?.integer == 4096,
      "Local runner clamps exact remaining context; no extra client 1024 reserve")
    #expect(run["params"]?["temperature"]?.double == 0.4)
    #expect(run["tools"]?.array == [.string("artifacts")])
    _ = try await runtime.command("settings", ["params": .object(["temperature": .number(0.9)])])
    let state = await runtime.presentation()
    let chrome = try #require(state["nativeTranscript"]?["messages"]?.array?.last?["chrome"])
    #expect(chrome["sections"]?.array?.count == 2)
    #expect(chrome["footer"]?.string?.contains("5 tokens") == true)
    #expect(
      chrome["sections"]?.array?.first?["rows"]?.array?.contains {
        $0["label"]?.string == "Sampling" && $0["value"]?.string?.contains("temp 0.4") == true
      } == true)
    #expect(await runtime.currentFields()?["messages"]?.array?.last?["run"]?.object == run)
    await runtime.close()
  }

  @Test func explicitReplyLimitSurvivesAndCompareUsesEachProviderCeiling() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "preferencesSave",
      ["changes": .object(["maxTokens": .number(1536)]), "expected": .object(["maxTokens": .null])])
    _ = try await runtime.command("send", ["text": .string("Explicit limit")])
    _ = try await settled(runtime)
    let explicit = RuntimeProtocol.state.withLock {
      $0.requests.first { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
    }
    #expect(explicit?["max_output_tokens"]?.integer == 1536)
    _ = try await runtime.command("newChat")
    _ = try await runtime.command(
      "preferencesSave",
      ["changes": .object(["maxTokens": .null]), "expected": .object(["maxTokens": .number(1536)])])
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("Compare limits")])
    _ = try await settled(runtime)
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }.suffix(2)
    }
    let cloud = requests.first { $0["path"]?.string?.contains("/cloud/") == true }?["body"]
    let local = requests.first { $0["path"]?.string?.contains("/runners/") == true }?["body"]
    #expect(cloud?["max_output_tokens"]?.integer == 2048)
    #expect((local?["max_output_tokens"]?.integer ?? 0) > 2048)
    await runtime.close()
  }

  @Test func compareUsesIndependentOutputCapacityAndDoesNotRewriteCustomPreference() async throws {
    let runtime = try await fixture(cloudOutput: 16384)
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("Independent capacities")])
    _ = try await settled(runtime)
    var requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
    }
    #expect(
      requests.first { $0["path"]?.string?.contains("/cloud/") == true }?["body"]?[
        "max_output_tokens"] == .number(16384))
    #expect(
      requests.first { $0["path"]?.string?.contains("/runners/") == true }?["body"]?[
        "max_output_tokens"] == .number(4096))
    _ = try await runtime.command(
      "preferencesSave",
      [
        "changes": .object(["maxTokens": .number(32768)]),
        "expected": .object(["maxTokens": .null]),
      ])
    _ = try await runtime.command("send", ["text": .string("Custom ceiling")])
    _ = try await settled(runtime)
    requests = RuntimeProtocol.state.withLock {
      Array($0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }.suffix(2))
    }
    #expect(
      requests.first { $0["path"]?.string?.contains("/cloud/") == true }?["body"]?[
        "max_output_tokens"] == .number(16384))
    #expect(
      requests.first { $0["path"]?.string?.contains("/runners/") == true }?["body"]?[
        "max_output_tokens"] == .number(4096))
    let saved = try await runtime.command("preferencesGet")
    #expect(saved["preferences"]?["maxTokens"] == .number(32768))
    await runtime.close()
  }
  @Test func unknownCapacityUsesEndpointDefaultInsteadOfAnInventedReplyCap() async throws {
    let runtime = try await fixture(
      localCapabilities: ["reasoning": .string("none")], cloudContext: nil, cloudOutput: nil)
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("Unknown limits")])
    _ = try await settled(runtime)
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
    }
    #expect(requests.count == 2)
    #expect(requests.allSatisfy { $0["body"]?["max_output_tokens"] == nil })
    await runtime.close()
  }
  @Test func compareMakerProviderAndBadgesSurviveFleetRemovalWithoutChangingTheRoute() async throws
  {
    let runtime = try await fixture()
    let kimi = "cloud:ep:moonshotai/kimi-k3@deepinfra/bf16"
    let muse = "cloud:ep:meta/muse"
    _ = try await runtime.command("models", ["ids": .array([.string(kimi), .string(muse)])])
    _ = try await runtime.command("send", ["text": .string("Compare identity")])
    let state = try await settled(runtime)
    let lanes = try #require(state["nativeTranscript"]?["messages"]?.array).dropFirst()
    #expect(
      lanes.map { $0["chrome"]?["modelName"]?.string } == ["Kimi K3 (deepinfra/bf16)", "Muse"])
    #expect(lanes.map { $0["chrome"]?["vendor"]?.string } == ["Moonshot", "Meta"])
    #expect(lanes.allSatisfy { $0["chrome"]?["tools"]?.array == [.string("artifacts")] })
    #expect(lanes.allSatisfy { $0["chrome"]?["fastest"]?.bool == false })
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
    }
    #expect(requests.count == 2)
    #expect(
      Set(requests.compactMap { $0["body"]?["model"]?.string }) == [
        "moonshotai/kimi-k3@deepinfra/bf16", "meta/muse",
      ])
    #expect(requests.allSatisfy { $0["path"]?.string == "/api/cloud/ep/v1/responses" })
    let saved = try #require(
      await runtime.currentFields()?["messages"]?.array?.dropFirst().first?.object)
    #expect(saved["run"]?["modelName"]?.string == "Kimi K3 (deepinfra/bf16)")
    #expect(saved["run"]?["vendor"]?.string == "Moonshot")

    RuntimeProtocol.state.withLock { $0.cloudEnabled = false }
    _ = try await runtime.command("refresh")
    let restored = await runtime.messageProjection(saved, controls: nil)
    #expect(restored["chrome"]?["modelName"] == lanes.first?["chrome"]?["modelName"])
    #expect(restored["chrome"]?["vendor"] == lanes.first?["chrome"]?["vendor"])
    var legacy = saved
    legacy["run"] = .object(["contended": .bool(true), "tools": .array([.string("filesystem")])])
    legacy["toolCalls"] = .array([])
    let projected = await runtime.messageProjection(legacy, controls: nil)
    #expect(projected["chrome"]?["modelName"]?.string == "kimi k3")
    #expect(projected["chrome"]?["vendor"]?.string == "Moonshot")
    #expect(projected["contended"]?.bool == true)
    #expect(
      projected["chrome"]?["tools"]?.array == [.string("filesystem")],
      "Tools badge describes the run, not whether a tool was called")
    await runtime.close()
  }

  @Test func discoveryDoesNotAppearAsAnArtifactInvocationInNewOrSavedChats() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Tool discovery")])
    _ = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let mid = try #require(doc["leafId"]?.string)
    let discovery: O = [
      "id": .string("discovery"), "type": .string("mcp_list_tools"),
      "server_label": .string("artifacts"),
    ]
    try await runtime.applyOutputItem(discovery, messageID: mid, done: true)
    #expect(await runtime.currentFields()?["messages"]?.array?.last?["toolCalls"] == nil)
    // Previous builds saved discovery. Rendering must filter it without a DB rewrite.
    try await runtime.updateMessage(mid) { $0["toolCalls"] = .array([.object(discovery)]) }
    let search: O = [
      "id": .string("search"), "type": .string("mcp_call"), "name": .string("mcp_search_tools"),
      "status": .string("completed"),
    ]
    try await runtime.applyOutputItem(search, messageID: mid, done: true)
    let state = await runtime.presentation()
    #expect(
      state["nativeTranscript"]?["messages"]?.array?.last?["toolCalls"]?.array?.map {
        $0["name"]?.string
      } == ["mcp_search_tools"])
    #expect(
      await runtime.currentFields()?["messages"]?.array?.last?["toolCalls"]?.array?.count == 2)
    await runtime.close()
  }

  @Test func failedAdmissionKeepsDraftAndDoesNotCallAModel() async throws {
    let runtime = try await fixture()
    let before = await runtime.currentFields()
    RuntimeProtocol.state.withLock { $0.failSave = true }
    await #expect(throws: (any Error).self) {
      try await runtime.command("send", ["text": .string("Keep me")])
    }
    #expect(await runtime.currentFields() == before)
    #expect(
      !RuntimeProtocol.state.withLock {
        $0.requests.contains { $0["path"]?.string?.hasSuffix("/responses") == true }
      })
    await runtime.close()
  }
  @Test func retryEditBranchesAndUnknownFieldsSurvive() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Original question")])
    _ = try await settled(runtime)
    var doc = try #require(await runtime.currentFields())
    doc["future"] = .object(["exact": .number(9_007_199_254_740_993)])
    RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!] = doc }
    _ = try await runtime.command("open", ["id": doc["id"]!])
    let oldLeaf = doc["leafId"]!
    _ = try await runtime.command(
      "messageAction",
      [
        "action": .string("retry"), "conversationId": doc["id"]!, "leafId": oldLeaf,
        "messageId": oldLeaf,
      ])
    _ = try await settled(runtime)
    var next = try #require(await runtime.currentFields())
    #expect(next["messages"]?.array?.count == 3)
    #expect(next["future"] == doc["future"])
    let user = doc["messages"]!.array![0]["id"]!
    _ = try await runtime.command(
      "messageAction",
      [
        "action": .string("edit"), "conversationId": doc["id"]!, "leafId": next["leafId"]!,
        "messageId": user, "originalText": .string("Original question"),
        "text": .string("Edited question"),
      ])
    _ = try await settled(runtime)
    next = try #require(await runtime.currentFields())
    #expect(next["messages"]?.array?.count == 5)
    let active = try ConversationDocument(fields: next).activeMessages
    #expect(ConversationDocument.text(active[0]) == "Edited question")
    #expect(active.count == 2)
    await #expect(throws: ConversationFailure.stale) {
      try await runtime.command(
        "messageAction",
        [
          "action": .string("retry"), "conversationId": doc["id"]!, "leafId": oldLeaf,
          "messageId": oldLeaf,
        ])
    }
    await runtime.close()
  }
  @Test func compareContextNeverReplaysTheOtherLanesAnswer() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("First")])
    _ = try await settled(runtime)
    _ = try await runtime.command("send", ["text": .string("Second")])
    _ = try await settled(runtime)
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
    }
    #expect(requests.count == 4)
    for request in requests.suffix(2) {
      let body = String(decoding: try JSONEncoder().encode(request["body"]!), as: UTF8.self)
      if request["path"]?.string?.contains("/cloud/") == true {
        #expect(!body.contains("Local final"))
      } else {
        #expect(!body.contains("Cloud final"))
      }
    }
    await runtime.close()
  }
  @Test func historyTitleAndSamplingAreNativeAndAcknowledged() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "settings", ["params": .object(["temperature": .number(0.4), "thinking": .bool(false)])])
    _ = try await runtime.command("send", ["text": .string("A question")])
    _ = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    _ = try await runtime.command(
      "renameChat", ["id": doc["id"]!, "title": .string("My full title")])
    _ = try await runtime.command("pinChat", ["id": doc["id"]!])
    let state = await runtime.presentation()
    #expect(state["history"]?.array?.first?["title"]?.string == "My full title")
    #expect(state["history"]?.array?.first?["pinned"]?.bool == true)
    let body = RuntimeProtocol.state.withLock {
      $0.requests.first { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
    }
    #expect(body?["temperature"]?.double == 0.4)
    #expect(body?["chat_template_kwargs"]?["enable_thinking"]?.bool == false)
    await runtime.close()
  }
  @Test func segmentOnlySpeechFinishesWithFinalWordAndOriginalRecording() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("models", ["ids": .array([.string("speech")])])
    let targets = try await runtime.audioTargets(dictation: false)
    #expect(targets.first?["transcription"]?["paddock_verbose"] == nil)
    let mapping = try await runtime.beginLive()
    let messageID = try #require(mapping["speech"])
    try await runtime.liveUpdate(
      messageID: messageID, text: "Testar funktionen med", transcript: [:])
    try await runtime.liveUpdate(
      messageID: messageID, text: "Testar funktionen med ljudinspelning.", transcript: [:])
    let clip: O = [
      "type": .string("audio"), "attachmentId": .string("original-wav"),
      "mime": .string("audio/wav"), "durationS": .number(4.0419375),
    ]
    try await runtime.finishLive(messages: [messageID], clip: clip)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    #expect(messages[0]["content"]?.array == [.object(clip)])
    #expect(
      messages[1]["content"]?.array?.first?["text"]?.string
        == "Testar funktionen med ljudinspelning.")
    #expect(messages[1]["streaming"]?.bool == false)
    #expect(messages[1]["error"] == nil)
    #expect(
      RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!]?["messages"] }
        == doc["messages"])
    await runtime.close()
  }
  @Test func speechTaskInstructionAlignmentAndRealtimeMetricsAreWiredEndToEnd() async throws {
    let runtime = try await fixture(speechFeatures: true)
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("fixture-audio"), "name": .string("Recording.wav"),
        "mime": .string("audio/wav"), "size": .number(44), "durationS": .number(2),
      ])
    _ = try await runtime.command(
      "send",
      [
        "text": .string("Identify speakers"),
        "attachments": .array([.object(["id": .string("fixture-audio")])]),
      ])
    let state = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let assistant = try #require(doc["messages"]?.array?.last)
    #expect(assistant["content"]?.array?.first?["text"]?.string == "Hello world")
    #expect(assistant["transcript"]?["words"]?.array?.last?["end"] == .number(1.5))
    #expect(assistant["transcript"]?["words"]?.array?.first?["confidence"] == .number(0.9))
    #expect(assistant["transcript"]?["wordsFrom"]?.string == "aligner")
    #expect((assistant["usage"]?["ms"]?.double ?? 0) > 0)
    #expect(
      state["nativeTranscript"]?["messages"]?.array?.last?["chrome"]?["footer"]?.string?.contains(
        "× realtime") == true)
    let requests = RuntimeProtocol.state.withLock { $0.requests }
    let form = try #require(
      requests.first { $0["path"]?.string?.hasSuffix("/transcriptions") == true }?["multipart"]?
        .string)
    #expect(form.contains("name=\"prompt\"\r\n\r\nIdentify speakers"))
    #expect(!form.contains("timestamp_granularities[]"))
    #expect(requests.contains { $0["path"]?.string == "/api/runners/11541/v1/audio/alignments" })
    #expect(!requests.contains { $0["path"]?.string?.hasSuffix("/responses") == true })
    let speech = state["nativeTranscript"]?["messages"]?.array?.last?["speech"]
    #expect(speech?["diarizers"]?.array?.first?["id"] == .string("speaker-model"))
    #expect(
      state["models"]?.array?.contains(where: { $0["id"] == .string("speaker-model") }) == false)
    _ = try await runtime.command(
      "identifySpeakers",
      [
        "conversationId": doc["id"]!,
        "messageId": assistant["id"]!, "modelId": .string("speaker-model"), "port": .number(11542),
      ])
    let enriched = try #require(await runtime.currentFields()?["messages"]?.array?.last)
    #expect(enriched["content"] == assistant["content"])
    #expect(enriched["transcript"]?["words"]?.array?.first?["speaker"] == .number(0))
    #expect(enriched["transcript"]?["words"]?.array?.first?["confidence"] == .number(0.9))
    #expect(enriched["transcript"]?["diarization"]?["segments"]?.array?.count == 1)
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": doc["id"]!])
    #expect(
      await runtime.currentFields()?["messages"]?.array?.last?["transcript"]
        == enriched["transcript"])
    await runtime.close()
  }
  @Test func failedLiveLaneDoesNotPoisonSuccessfulTranscriptOrOriginal() async throws {
    let runtime = try await fixture(speechFeatures: true)
    _ = try await runtime.command("models", ["ids": .array([.string("local"), .string("speech")])])
    let lanes = try await runtime.beginLive()
    let successful = try #require(lanes["local"])
    let failed = try #require(lanes["speech"])
    try await runtime.liveUpdate(
      messageID: successful, text: "Hello world",
      transcript: ["language": .string("en"), "durationS": .number(2)])
    try await runtime.liveFailure(messageID: failed, error: "Model disconnected")
    let clip: O = [
      "type": .string("audio"), "attachmentId": .string("fixture-audio"),
      "mime": .string("audio/wav"), "durationS": .number(2),
    ]
    try await runtime.finishLive(messages: [successful, failed], clip: clip)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    let good = try #require(messages.first { $0["id"]?.string == successful })
    let bad = try #require(messages.first { $0["id"]?.string == failed })
    #expect(good["error"] == nil)
    #expect(good["streaming"]?.bool == false && bad["streaming"]?.bool == false)
    #expect(good["content"]?.array?.first?["text"]?.string == "Hello world")
    #expect(good["transcript"]?["wordsFrom"]?.string == "aligner")
    #expect(bad["error"]?.string == "Model disconnected")
    #expect(messages.first?["content"]?.array?.first?["attachmentId"]?.string == "fixture-audio")
    #expect(
      RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!]?["messages"] }
        == doc["messages"])
    await runtime.close()
  }
}

final class RuntimeProtocol: URLProtocol, @unchecked Sendable {
  typealias V = ConversationValue
  typealias O = [String: V]
  struct State: Sendable {
    var documents: [String: O] = [:]
    var requests: [O] = []
    var failSave = false
    var failImageStore = false
    var imageAttachments: [String: Data] = [:]
    var imageEditError: Int?
    var imageJSONCount: Int?
    var speechFeatures = false
    var cloudEnabled = true
    var cloudContext: Int? = 131072
    var cloudOutput: Int? = 2048
    var artifacts: [V] = []
    var responseFailure = false
    var localCapabilities: O?
    var imagePeerCapabilities: O?
    var documentAnswer: String?
    var documentRefusals: [(Int, String)] = []
  }
  static let state = Mutex(State())
  override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "127.0.0.1" }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let path = request.url!.path
    let method = request.httpMethod ?? "GET"
    var bytes = request.httpBody ?? Data()
    if bytes.isEmpty, let stream = request.httpBodyStream {
      stream.open()
      defer { stream.close() }
      var buffer = [UInt8](repeating: 0, count: 4096)
      while stream.hasBytesAvailable {
        let n = stream.read(&buffer, maxLength: buffer.count)
        if n <= 0 { break }
        bytes.append(contentsOf: buffer.prefix(n))
      }
    }
    let body = (try? JSONDecoder().decode(V.self, from: bytes)) ?? .null
    let response: (Int, String, Data) = Self.state.withLock { state in
      state.requests.append([
        "path": .string(path), "body": body,
        "multipart": .string(String(decoding: bytes, as: UTF8.self)),
        "contentType": .string(request.value(forHTTPHeaderField: "Content-Type") ?? ""),
        "method": .string(method),
      ])
      var value: V = .object([:])
      var status = 200
      if state.failImageStore, method == "PUT", path.hasPrefix("/api/attachments/") {
        return (503, "application/json", Data("{\"error\":\"Store unavailable\"}".utf8))
      } else if state.localCapabilities?["image_generation"] != nil,
        path.hasPrefix("/api/attachments/")
      {
        let id = String(path.split(separator: "/").last!)
        if method == "PUT" {
          state.imageAttachments[id] = bytes
        } else if let original = state.imageAttachments[id] {
          return (200, "image/png", original)
        } else {
          return (404, "application/json", Data("{}".utf8))
        }
      } else if state.speechFeatures, path.hasPrefix("/api/attachments/") {
        return (200, "audio/wav", Data(repeating: 0, count: 44))
      } else if state.speechFeatures, path == "/api/runners" {
        value = .array([
          .object([
            "model": .string("local"), "display": .string("Granite Speech Plus"),
            "port": .number(12481), "status": .string("ok"),
          ]),
          .object([
            "asr": .string("speech"), "display": .string("Whisper"), "port": .number(11540),
            "status": .string("ok"),
          ]),
          .object(["aligner": .string("aligner"), "port": .number(11541), "status": .string("ok")]),
          .object([
            "diarization": .string("speaker-model"), "port": .number(11542),
            "status": .string("ok"),
          ]),
        ])
      } else if state.speechFeatures, path == "/api/runners/12481/server" {
        value = .object([
          "audio": .bool(true), "timestamp_granularities": .array([.string("word")]),
          "include": .array([.string("logprobs")]), "transcription_max_clip_s": .number(120),
          "realtime_transcription": .object(["supported": .bool(true), "enrichment": .bool(false)]),
        ])
      } else if state.speechFeatures, path == "/api/runners/11541/server" {
        value = .object(["aligner": .string("aligner"), "alignment_max_clip_s": .number(120)])
      } else if state.speechFeatures, path.hasSuffix("/transcriptions") {
        let terminal =
          "{\"type\":\"transcript.text.done\",\"text\":\"Hello world\",\"paddock_verbose\":{\"duration\":2,\"words\":[{\"word\":\"Hello\",\"paddock_confidence\":0.9},{\"word\":\"world\"}]}}"
        return (
          200, "text/event-stream",
          Data(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"Hello world\"}\n\ndata: \(terminal)\n\n"
              .utf8)
        )
      } else if state.speechFeatures, path.hasSuffix("/alignments") {
        return (
          200, "application/json",
          Data(
            "{\"words\":[{\"word\":\"Hello\",\"start\":0.2,\"end\":0.8},{\"word\":\"world\",\"start\":0.8,\"end\":1.5}],\"language_supported\":true}"
              .utf8)
        )
      } else if state.speechFeatures, path.hasSuffix("/diarizations") {
        return (
          200, "application/json",
          Data(
            #"{"model":"speaker-model","duration":2,"preset":"offline","attribution":"time_overlap_v1","segments":[{"speaker":0,"start":0.1,"end":1.6}],"words":[{"word":"Hello","start":0.2,"end":0.8,"confidence":0.9,"speaker":0,"speakers":[0]},{"word":"world","start":0.8,"end":1.5,"speaker":0,"speakers":[0]}]}"#
              .utf8)
        )
      } else if path == "/api/settings" {
        value = .object(["macos_studio_preferences": .object(["pk_auto_title": .string("off")])])
      } else if path == "/api/runners" {
        value = .array([
          .object([
            (state.localCapabilities?["image_generation"] != nil ? "image" : "model"): .string(
              "local"), "display": .string("Local model"), "port": .number(12481),
            "status": .string("ok"),
          ]),
          .object([
            "asr": .string("speech"), "display": .string("Whisper"), "port": .number(11540),
            "status": .string("ok"),
          ]),
        ])
        if state.imagePeerCapabilities != nil {
          value = .array(
            (value.array ?? []) + [
              .object([
                "image": .string("local-peer"), "display": .string("Other image model"),
                "port": .number(12482), "status": .string("ok"),
              ])
            ])
        }
      } else if path == "/api/cloud" {
        value = .array(
          state.cloudEnabled
            ? [
              .object([
                "id": .string("ep"), "name": .string("Fixture cloud"), "hasKey": .bool(true),
                "models": .array([
                  .object([
                    "id": .string("remote"), "display": .string("Remote model"),
                    "vision": .bool(false),
                    "ctx": state.cloudContext.map { .number(Decimal($0)) } ?? .null,
                    "maxOut": state.cloudOutput.map { .number(Decimal($0)) } ?? .null,
                  ]),
                  .object([
                    "id": .string("moonshotai/kimi-k3"), "display": .string("MoonshotAI: Kimi K3"),
                    "provider": .string("deepinfra/bf16"),
                  ]),
                  .object(["id": .string("meta/muse"), "display": .string("Meta: Muse")]),
                ]),
              ])
            ] : [])
      } else if path == "/api/runners/11540/server" {
        value = .object([
          "asr": .string("speech"), "timestamp_granularities": .array([.string("segment")]),
        ])
      } else if path == "/api/runners/12482/server", let cap = state.imagePeerCapabilities {
        value = .object(cap)
      } else if path.hasSuffix("/server") {
        value = .object(
          state.localCapabilities ?? [
            "max_ctx": .number(4096), "default_max_output_tokens": .number(1024),
            "reasoning": .string("toggle"), "vision": .bool(true),
          ])
      } else if path == "/api/conversations" {
        value = .array(state.documents.values.map(V.object))
      } else if path.hasSuffix("/artifacts") {
        value = .array(state.artifacts)
      } else if path.hasPrefix("/api/conversations/") {
        let id = String(path.split(separator: "/").last!)
        if method == "PUT" {
          if state.failSave { status = 503 } else { state.documents[id] = body.object }
        } else if let doc = state.documents[id] {
          value = .object(doc)
        } else {
          status = 404
        }
      } else if path.hasSuffix("/images/generations") || path.hasSuffix("/images/edits") {
        if path.hasSuffix("/images/edits"), let status = state.imageEditError {
          return (
            status, "application/json",
            Data("{\"error\":{\"message\":\"Reference image rejected by endpoint\"}}".utf8)
          )
        }
        let png =
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        if let count = state.imageJSONCount {
          return (
            200, "application/json",
            try! JSONEncoder().encode(
              V.object([
                "data": .array((0..<count).map { _ in .object(["b64_json": .string(png)]) }),
                "size": .string("1536x1024"), "output_format": .string("png"),
              ]))
          )
        }
        return (
          200, "text/event-stream",
          Data(
            ("data: {\"type\":\"image_generation.partial_image\",\"b64_json\":\"\(png)\",\"partial_image_index\":0}\n\n"
              + "data: {\"type\":\"image_generation.completed\",\"b64_json\":\"\(png)\",\"size\":\"1024x1024\",\"quality\":\"low\",\"background\":\"opaque\",\"usage\":{\"input_tokens\":5,\"output_tokens\":4096}}\n\n")
              .utf8)
        )
      } else if path.hasSuffix("/responses") {
        if !state.documentRefusals.isEmpty {
          let refusal = state.documentRefusals.removeFirst()
          return (
            refusal.0, "application/json",
            try! JSONEncoder().encode(
              V.object([
                "error": .object(["message": .string(refusal.1)])
              ]))
          )
        }
        if state.responseFailure {
          let error =
            #"{"error":{"message":"DeepInfra: model is temporarily rate-limited upstream.","metadata":{"provider_name":"DeepInfra","provider_error_code":"engine_overloaded","action":"openrouter_integrations"}}}"#
          return (429, "application/json", Data(error.utf8))
        }
        let answer =
          state.documentAnswer
          ?? (path.contains("/cloud/") ? "Cloud final tail 🦊" : "Local final tail 🦊")
        let terminal: O = [
          "type": .string("response.completed"),
          "response": .object([
            "status": .string("completed"),
            "output": .array([
              .object([
                "type": .string("message"),
                "content": .array([
                  .object(["type": .string("output_text"), "text": .string(answer)])
                ]),
              ])
            ]), "usage": .object(["input_tokens": .number(10), "output_tokens": .number(5)]),
          ]),
        ]
        let end = String(decoding: try! JSONEncoder().encode(terminal), as: UTF8.self)
        return (
          200, "text/event-stream",
          Data(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\ndata: \(end)\n\n"
              .utf8)
        )
      }
      return (status, "application/json", try! JSONEncoder().encode(value))
    }
    client?.urlProtocol(
      self,
      didReceive: HTTPURLResponse(
        url: request.url!, statusCode: response.0, httpVersion: "HTTP/1.1",
        headerFields: ["Content-Type": response.1])!, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: response.2)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {}
}
