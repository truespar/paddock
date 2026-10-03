import Foundation
import PaddockClient
import Testing

@testable import PaddockConversationCore

@Suite("Native conversation transport security")
struct TransportPolicyTests {
  @Test func onlyExactLoopbackDescriptorIsAccepted() throws {
    let token = String(repeating: "a", count: 64)
    for origin in [
      "https://example.com", "http://localhost:1234", "http://127.0.0.1:0",
      "http://user:password@127.0.0.1:1234", "http://127.0.0.1:1234/path",
      "http://127.0.0.1:1234?query=x", "http://127.0.0.1:1234#fragment",
    ] {
      let host = try descriptor(origin, token)
      #expect(!host.isValidPrivateHost)
      #expect(throws: (any Error).self) { try NativeConversationTransport(host: host) }
    }
    #expect(try descriptor("http://127.0.0.1:1234", token).isValidPrivateHost)
    #expect(
      try !descriptor("http://127.0.0.1:1234", token + "\r\nInjected: header").isValidPrivateHost)
  }
  @Test func endpointCannotEscapePrivateRoutes() {
    for id in ["", "../settings", "https://evil.example", "x%2f..", "a/b", "a?token=x", "a#b"] {
      #expect(NativeConversationTransport.Endpoint.cloud(id).path == nil)
    }
    #expect(NativeConversationTransport.Endpoint.runner(0).path == nil)
    #expect(
      NativeConversationTransport.Endpoint.cloud("provider-id").path
        == "api/cloud/provider-id/v1/responses")
  }
  @Test func localApiErrorsPreserveActionableMessagesWithoutRawJson() async throws {
    let host = try descriptor("http://127.0.0.1:1234", String(repeating: "a", count: 64))
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [ReadsErrorProtocol.self]
    let transport = try NativeConversationTransport(host: host, configuration: config)
    do {
      _ = try await transport.api("api/reads", method: "POST", body: .object([:]))
      Issue.record("Expected the revision conflict")
    } catch {
      #expect(error.localizedDescription == "This question set changed. Reload it before saving.")
    }
    await transport.close()
  }
  @Test func orderedReadBytesAreNotReencodedByNetworkTransport() async throws {
    let host = try descriptor("http://127.0.0.1:1234", String(repeating: "a", count: 64))
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [ReadsEchoProtocol.self]
    let transport = try NativeConversationTransport(host: host, configuration: config)
    let draft = try ReadDraft.parse(
      Data(
        #"{"state":"test","questions":{"zebra":{"type":"choice","instructions":"Pick","criteria":{"z":"last","a":"first"}},"alpha":{"type":"noul","instructions":"Is it?","criteria":{"true":"Yes","false":"No"}}}}"#
          .utf8))
    let bytes = try draft.requestData(model: "clef-flash")
    let received = try await transport.bytes(
      "api/runners/1234/v1/systemone", method: "POST", body: bytes)
    #expect(received == bytes)
    #expect(try ReadDraft.parse(received).ordering == [["zebra", "z", "a"], ["alpha"]])
    await transport.close()
  }
  private func descriptor(_ origin: String, _ token: String) throws -> StudioHost {
    try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": origin, "cookieName": "paddock_desktop_session", "session": token,
      ]))
  }
}

private final class ReadsEchoProtocol: URLProtocol, @unchecked Sendable {
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    var bytes = request.httpBody ?? Data()
    if let stream = request.httpBodyStream {
      stream.open()
      defer { stream.close() }
      var buffer = [UInt8](repeating: 0, count: 4096)
      while stream.hasBytesAvailable {
        let n = stream.read(&buffer, maxLength: buffer.count)
        if n <= 0 { break }
        bytes.append(contentsOf: buffer.prefix(n))
      }
    }
    let response = HTTPURLResponse(
      url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: bytes)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {}
}

private final class ReadsErrorProtocol: URLProtocol, @unchecked Sendable {
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let response = HTTPURLResponse(
      url: request.url!, statusCode: 409, httpVersion: nil,
      headerFields: ["Content-Type": "application/json"])!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(
      self,
      didLoad: Data(
        #"{"error":{"message":"This question set changed. Reload it before saving."}}"#.utf8))
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {}
}
