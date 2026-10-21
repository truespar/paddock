import Darwin
import Foundation
import PaddockClient
import PaddockConversationCore

/// No NSApplication, viewer, JavaScript runtime or user model invocation.
/// This executable deliberately does not link PaddockUI/PaddockStudio.
@main struct NativeCoreCheck {
  static func main() async {
    do {
      try await run()
    } catch {
      // Failed checks and an accidental Finder/crash-dialog relaunch are
      // ordinary diagnostic failures. Throwing out of async main turns them
      // into SIGTRAP and a user-visible macOS crash report.
      FileHandle.standardError.write(Data("FAIL: \(error.localizedDescription)\n".utf8))
      exit(EXIT_FAILURE)
    }
  }

  static func run() async throws {
    guard let root = ProcessInfo.processInfo.environment["PADDOCK_DATA"],
      root.hasPrefix("/tmp/paddock-native-core-check.")
    else {
      throw ConversationFailure.invalid("Use isolated native-core-check storage")
    }
    let manager = NativeManager()
    let host = try await manager.nativeConversationHost()
    let transport = try NativeConversationTransport(host: host)
    let repository = ConversationRepository(storage: transport)
    do {
      try require(host.isValidPrivateHost, "private host policy")
      try require(try await transport.listConversations().isEmpty, "isolated store starts empty")
      let data = Data(
        #"{"id":"native-fixture","title":"Before","model":"synthetic","systemPrompt":"","params":{},"createdAt":1,"updatedAt":1,"messages":[{"id":"q","parentId":null,"role":"user","content":[{"type":"file","attachmentId":"retained-reference","future":"keep"}]},{"id":"a","parentId":"q","role":"assistant","content":[{"type":"text","text":"First answer"}]},{"id":"b","parentId":"q","role":"assistant","content":[{"type":"text","text":"Second answer"}]}],"leafId":"a","future":{"exactInteger":9007199254740993,"text":"🦊"}}"#
          .utf8)
      let document = try ConversationDocument(data: data)
      try await transport.saveConversation(document)
      async let renamed = repository.edit(document.id, .rename("Native saved title"))
      async let pinned = repository.edit(document.id, .pin(true))
      _ = try await (renamed, pinned)
      let switched = try await repository.edit(
        document.id, .branch(expectedLeaf: "a", message: "a", target: "b"))
      try require(
        switched.activeMessages.compactMap { $0["id"]?.string } == ["q", "b"],
        "native branch selection")
      let restored = try await transport.loadConversation(document.id)
      try require(
        restored.title == "Native saved title" && restored.fields["pinned"]?.bool == true,
        "concurrent edits survive Rust/SQLite")
      try require(
        restored.fields["future"] == document.fields["future"]
          && restored.messages == document.messages, "all stored message/unknown fields retained")
      try require(try await transport.listConversations().count == 1, "native history list")
      let urlSession = URLSession(configuration: .ephemeral)
      defer { urlSession.invalidateAndCancel() }
      for path in ["/", "/studio", "/index.html"] {
        var request = URLRequest(url: host.origin.appending(path: path))
        request.setValue("\(host.cookieName)=\(host.session)", forHTTPHeaderField: "Cookie")
        let (_, response) = try await urlSession.data(for: request)
        try require((response as? HTTPURLResponse)?.statusCode == 404, "no HTML served at \(path)")
      }
      // HTML artifacts are an explicit isolated viewer, not a web Studio.
      // The shared empty frame is allowed; its network/origin restrictions
      // must survive embedding the Rust host in a native app.
      var frameRequest = URLRequest(url: host.origin.appending(path: "artifact-frame"))
      frameRequest.setValue("\(host.cookieName)=\(host.session)", forHTTPHeaderField: "Cookie")
      let (frame, frameResponse) = try await urlSession.data(for: frameRequest)
      let frameHTTP = frameResponse as? HTTPURLResponse
      let csp = frameHTTP?.value(forHTTPHeaderField: "Content-Security-Policy") ?? ""
      try require(
        frameHTTP?.statusCode == 200
          && String(decoding: frame, as: UTF8.self).contains("paddock:artifact"),
        "isolated artifact frame is available")
      try require(
        csp.contains("default-src 'none'") && csp.contains("connect-src 'none'")
          && csp.contains("sandbox allow-scripts") && !csp.contains("allow-same-origin")
          && csp.contains("frame-ancestors 'self'") && csp.contains("img-src data: blob:;"),
        "artifact frame cannot inherit the app origin or access the network")
      let (_, unauthorized) = try await urlSession.data(
        from: host.origin.appending(path: "api/conversations"))
      try require(
        (unauthorized as? HTTPURLResponse)?.statusCode == 401, "unauthenticated reads denied")
      print(
        "PASS: asset-free Rust host, native save/list/load, serialized metadata and branch edits")
      let runtime = NativeStudioRuntime(transport: transport) { _ in }
      try await runtime.start()
      _ = try await runtime.command("open", ["id": .string(document.id)])
      _ = try await runtime.command(
        "renameChat", ["id": .string(document.id), "title": .string("Native runtime title")])
      let nativeSaved = try await transport.loadConversation(document.id)
      try require(
        nativeSaved.title == "Native runtime title"
          && nativeSaved.fields["future"] == document.fields["future"],
        "running coordinator owns actual Rust documents without HTML")
      print(
        "PASS: native Studio coordinator startup, full-document open and acknowledged history mutation through Rust"
      )
      try await composerDraftChecks(runtime: runtime, transport: transport, host: host)
      try await networkChecks()
      let libraries = (0..<_dyld_image_count()).compactMap { index in
        _dyld_get_image_name(index).map { String(cString: $0) }
      }
      try require(
        !libraries.contains {
          $0.contains("WebKit.framework") || $0.contains("JavaScriptCore.framework")
        }, "no WebKit or JavaScriptCore loaded")
      print("PASS: no WebKit/JavaScriptCore libraries loaded; no UI or microphone opened")
      await runtime.close()
      await repository.close()
      await transport.close()
      await manager.close()
    } catch {
      await repository.close()
      await transport.close()
      await manager.close()
      throw error
    }
  }
  static func require(_ value: Bool, _ description: String) throws {
    guard value else { throw ConversationFailure.invalid("CHECK FAILED: \(description)") }
  }
}
