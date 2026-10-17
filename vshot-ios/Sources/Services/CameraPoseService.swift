import AVFoundation
import CoreImage
import UIKit
import Vision

final class CameraPoseService: NSObject, ObservableObject {
    let session = AVCaptureSession()
    @Published private(set) var isReady = false
    @Published private(set) var permissionDenied = false
    @Published private(set) var position: AVCaptureDevice.Position = .back
    @Published private(set) var joints: [String: CGPoint] = [:]
    @Published private(set) var bodyBox: CGRect = .zero
    @Published private(set) var guidance = "Step into frame ✨"
    @Published private(set) var brightness: CGFloat = 0.55

    private let sessionQueue = DispatchQueue(label: "app.vshot.camera.session", qos: .userInitiated)
    private let outputQueue = DispatchQueue(label: "app.vshot.camera.output", qos: .userInteractive)
    private let visionQueue = DispatchQueue(label: "app.vshot.vision", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let videoOutput = AVCaptureVideoDataOutput()
    private var currentInput: AVCaptureDeviceInput?
    private var latestImageData: Data?
    private var lastVisionTime: CFTimeInterval = 0
    private var lastJPEGTime: CFTimeInterval = 0
    private var isVisionBusy = false

    override init() { super.init(); requestPermissionAndConfigure() }
    deinit { session.stopRunning() }

    func start() {
        sessionQueue.async { [weak self] in
            guard let self, self.isReady, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    func toggleCamera() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            let target: AVCaptureDevice.Position = self.position == .back ? .front : .back
            guard let device = Self.camera(position: target), let input = try? AVCaptureDeviceInput(device: device) else { return }
            self.session.beginConfiguration()
            let old = self.currentInput
            if let old { self.session.removeInput(old) }
            if self.session.canAddInput(input) {
                self.session.addInput(input); self.currentInput = input
                DispatchQueue.main.async { self.position = target }
            } else if let old, self.session.canAddInput(old) { self.session.addInput(old) }
            self.session.commitConfiguration()
        }
    }

    func captureCurrentFrame() async -> Data? {
        await withCheckedContinuation { continuation in
            outputQueue.async { [weak self] in continuation.resume(returning: self?.latestImageData) }
        }
    }

    private func requestPermissionAndConfigure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: configure()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                granted ? self?.configure() : DispatchQueue.main.async { self?.permissionDenied = true }
            }
        default: permissionDenied = true
        }
    }

    private func configure() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration(); self.session.sessionPreset = .high
            guard let device = Self.camera(position: .back), let input = try? AVCaptureDeviceInput(device: device), self.session.canAddInput(input) else {
                self.session.commitConfiguration(); return
            }
            self.session.addInput(input); self.currentInput = input
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
            self.videoOutput.setSampleBufferDelegate(self, queue: self.outputQueue)
            if self.session.canAddOutput(self.videoOutput) { self.session.addOutput(self.videoOutput) }
            if let connection = self.videoOutput.connection(with: .video), connection.isVideoOrientationSupported { connection.videoOrientation = .portrait }
            self.session.commitConfiguration()
            DispatchQueue.main.async { self.isReady = true; self.start() }
        }
    }

    private static func camera(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInTripleCamera], mediaType: .video, position: position).devices.first
    }

    private func makeJPEG(from pixelBuffer: CVPixelBuffer) -> Data? {
        var image = CIImage(cvPixelBuffer: pixelBuffer).oriented(position == .front ? .leftMirrored : .right)
        let maxSide = max(image.extent.width, image.extent.height)
        if maxSide > 1800 {
            let scale = 1800 / maxSide
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        guard let cg = context.createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.92)
    }

    private func analyze(pixelBuffer: CVPixelBuffer) {
        guard !isVisionBusy else { return }
        isVisionBusy = true
        let front = position == .front
        visionQueue.async { [weak self] in
            guard let self else { return }
            defer { self.outputQueue.async { self.isVisionBusy = false } }
            let request = VNDetectHumanBodyPoseRequest()
            let orientation: CGImagePropertyOrientation = front ? .leftMirrored : .right
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
            do {
                try handler.perform([request])
                guard let observation = request.results?.first else {
                    DispatchQueue.main.async { self.joints = [:]; self.bodyBox = .zero; self.guidance = "Step into frame ✨" }
                    return
                }
                let parsed = self.extractPoints(observation)
                let box = self.boundingBox(parsed)
                let guide = self.makeGuidance(box: box, points: parsed)
                DispatchQueue.main.async { self.joints = parsed; self.bodyBox = box; self.guidance = guide }
            } catch { }
        }
    }

    private func extractPoints(_ observation: VNHumanBodyPoseObservation) -> [String: CGPoint] {
        let mapping: [(String, VNHumanBodyPoseObservation.JointName)] = [
            ("nose", .nose), ("leftShoulder", .leftShoulder), ("rightShoulder", .rightShoulder),
            ("leftElbow", .leftElbow), ("rightElbow", .rightElbow), ("leftWrist", .leftWrist), ("rightWrist", .rightWrist),
            ("leftHip", .leftHip), ("rightHip", .rightHip), ("leftKnee", .leftKnee), ("rightKnee", .rightKnee),
            ("leftAnkle", .leftAnkle), ("rightAnkle", .rightAnkle)
        ]
        var result: [String: CGPoint] = [:]
        for (key, name) in mapping {
            if let point = try? observation.recognizedPoint(name), point.confidence > 0.25 {
                result[key] = CGPoint(x: point.location.x, y: 1 - point.location.y)
            }
        }
        return result
    }

    private func boundingBox(_ points: [String: CGPoint]) -> CGRect {
        guard points.count >= 4 else { return .zero }
        let xs = points.values.map(\.x), ys = points.values.map(\.y)
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else { return .zero }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private func makeGuidance(box: CGRect, points: [String: CGPoint]) -> String {
        guard box != .zero else { return "Step into frame ✨" }
        if box.midX < 0.42 { return "Move a little right →" }
        if box.midX > 0.58 { return "← Move a little left" }
        if box.height < 0.42 { return "Come closer ♡" }
        if box.height > 0.90 { return "Step back a touch" }
        if points["leftAnkle"] == nil && points["rightAnkle"] == nil { return "Tilt camera down ↓" }
        if let nose = points["nose"], nose.y < 0.08 { return "Give yourself more headroom" }
        return "Frame looks cute ✦"
    }
}

extension CameraPoseService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let now = CACurrentMediaTime()
        if now - lastJPEGTime > 0.12 { lastJPEGTime = now; latestImageData = makeJPEG(from: pixelBuffer) }
        if now - lastVisionTime > 0.18 {
            lastVisionTime = now
            let offset = CGFloat(currentInput?.device.exposureTargetOffset ?? 0)
            let estimate = min(1, max(0, 0.55 + offset * 0.12))
            DispatchQueue.main.async { self.brightness = estimate }
            analyze(pixelBuffer: pixelBuffer)
        }
    }
}
