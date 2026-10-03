@preconcurrency import AVFoundation
import AppKit
import CoreImage
import ImageIO
import PaddockClient
import SwiftUI
import UniformTypeIdentifiers

/// Device-independent clock seam: startup and a stalled established stream
/// have different deadlines. In particular, a camera that never produces its
/// first buffer must not leave the inference loop polling nil forever.
struct NativeReadFrameWatchdog {
  private var started: ContinuousClock.Instant?
  private var lastFrame: ContinuousClock.Instant?
  mutating func start(at now: ContinuousClock.Instant) {
    started = now
    lastFrame = nil
  }
  mutating func received(at now: ContinuousClock.Instant) { lastFrame = now }
  func failure(at now: ContinuousClock.Instant, isRunning: Bool) -> String? {
    guard let started else { return nil }
    guard isRunning else { return "The camera stopped. Reconnect it and try again." }
    if let lastFrame {
      return lastFrame.duration(to: now) >= .seconds(2)
        ? "The camera stopped delivering frames. Reconnect it and try again." : nil
    }
    return started.duration(to: now) >= .seconds(5)
      ? "The camera did not deliver a frame. Check the camera connection and try again." : nil
  }
}

/// One serial owner for configuration and delivery. Retain only the newest
/// buffer; resize/JPEG runs here on demand after inference has consumed one.
/// The preview is AVFoundation's layer, not a stream of SwiftUI image updates.
final class NativeReadCameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
  @unchecked Sendable
{
  let session = AVCaptureSession()
  private let queue = DispatchQueue(label: "io.truespar.paddock.reads.camera", qos: .userInitiated)
  private let output = AVCaptureVideoDataOutput()
  private let context = CIContext(options: [.cacheIntermediates: false])
  private var latest: CVPixelBuffer?
  private var watchdog = NativeReadFrameWatchdog()
  private var closed = false

  static func devices() -> [AVCaptureDevice] {
    AVCaptureDevice.DiscoverySession(
      deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
      mediaType: .video, position: .unspecified
    ).devices
  }

  func start(deviceID: String) async throws {
    try await withCheckedThrowingContinuation { (reply: CheckedContinuation<Void, Error>) in
      queue.async { [self] in
        do {
          guard !closed, AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
            throw ManagerError.core("Camera access is not granted.")
          }
          let device =
            deviceID.isEmpty
            ? AVCaptureDevice.default(for: .video)
            : Self.devices().first { $0.uniqueID == deviceID }
          guard let device else { throw ManagerError.core("The selected camera is not connected.") }
          let input = try AVCaptureDeviceInput(device: device)
          session.beginConfiguration()
          session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
          guard session.canAddInput(input), session.canAddOutput(output) else {
            session.commitConfiguration()
            throw ManagerError.core("This camera could not be configured.")
          }
          session.addInput(input)
          output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
          ]
          output.alwaysDiscardsLateVideoFrames = true
          session.addOutput(output)
          output.setSampleBufferDelegate(self, queue: queue)
          session.commitConfiguration()
          session.startRunning()
          guard session.isRunning else { throw ManagerError.core("The camera did not start.") }
          watchdog.start(at: .now)
          reply.resume()
        } catch { reply.resume(throwing: error) }
      }
    }
  }

  func stop() {
    queue.async { [self] in
      closed = true
      output.setSampleBufferDelegate(nil, queue: nil)
      session.stopRunning()
      latest = nil
    }
  }

  func captureOutput(
    _ output: AVCaptureOutput, didOutput buffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard !closed else { return }
    guard let frame = CMSampleBufferGetImageBuffer(buffer) else { return }
    latest = frame
    watchdog.received(at: .now)
  }

  func jpeg(longSide: Int) async throws -> Data? {
    try await withCheckedThrowingContinuation { reply in
      queue.async { [self] in
        autoreleasepool {
          guard !closed else {
            reply.resume(returning: nil)
            return
          }
          if let error = watchdog.failure(at: .now, isRunning: session.isRunning) {
            reply.resume(throwing: ManagerError.core(error))
            return
          }
          guard let latest else {
            reply.resume(returning: nil)
            return
          }
          let image = CIImage(cvPixelBuffer: latest)
          let scale = min(1, CGFloat(longSide) / max(image.extent.width, image.extent.height))
          let resized = image.applyingFilter(
            "CILanczosScaleTransform",
            parameters: [kCIInputScaleKey: scale, kCIInputAspectRatioKey: 1])
          guard let cg = context.createCGImage(resized, from: resized.extent.integral) else {
            reply.resume(throwing: ManagerError.core("The camera frame could not be resized."))
            return
          }
          let data = NSMutableData()
          guard
            let dest = CGImageDestinationCreateWithData(
              data, UTType.jpeg.identifier as CFString, 1, nil)
          else {
            reply.resume(throwing: ManagerError.core("The camera frame could not be encoded."))
            return
          }
          CGImageDestinationAddImage(
            dest, cg, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
          guard CGImageDestinationFinalize(dest) else {
            reply.resume(throwing: ManagerError.core("The camera frame could not be encoded."))
            return
          }
          reply.resume(returning: data as Data)
        }
      }
    }
  }
}

struct NativeReadCameraPreview: NSViewRepresentable {
  let capture: NativeReadCameraCapture
  final class View: NSView {
    let preview = AVCaptureVideoPreviewLayer()
    override init(frame: NSRect) {
      super.init(frame: frame)
      wantsLayer = true
      preview.videoGravity = .resizeAspect
      layer?.addSublayer(preview)
    }
    required init?(coder: NSCoder) { nil }
    override func layout() {
      super.layout()
      CATransaction.begin()
      CATransaction.setDisableActions(true)
      preview.frame = bounds
      CATransaction.commit()
    }
  }
  func makeNSView(context: Context) -> View { View(frame: .zero) }
  func updateNSView(_ view: View, context: Context) {
    if view.preview.session !== capture.session { view.preview.session = capture.session }
    if let connection = view.preview.connection, connection.isVideoMirroringSupported {
      connection.automaticallyAdjustsVideoMirroring = false
      connection.isVideoMirrored = true
    }
  }
  static func dismantleNSView(_ view: View, coordinator: ()) { view.preview.session = nil }
}
