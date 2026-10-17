import AVFoundation
import CoreImage
import UIKit

final class CameraService: NSObject, ObservableObject {
    let session = AVCaptureSession()

    @Published private(set) var isReady = false
    @Published private(set) var isCapturing = false
    @Published private(set) var permissionDenied = false
    @Published private(set) var position: AVCaptureDevice.Position = .back

    private let sessionQueue = DispatchQueue(label: "app.tiltphoto.camera.session", qos: .userInitiated)
    private let outputQueue = DispatchQueue(label: "app.tiltphoto.camera.frames", qos: .userInteractive)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let videoOutput = AVCaptureVideoDataOutput()

    private var currentInput: AVCaptureDeviceInput?
    private var rollingFrames: [Data] = []
    private var captureFrames: [Data] = []
    private var remainingPostFrames = 0
    private var completion: (([Data]) -> Void)?
    private var lastAcceptedTime: Double = -1

    private let preFrameCount = 7
    private let postFrameCount = 8
    private let frameInterval = 1.0 / 12.0

    override init() {
        super.init()
        requestPermissionAndConfigure()
    }

    deinit {
        session.stopRunning()
    }

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
            guard let device = Self.camera(position: target),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }

            self.session.beginConfiguration()
            if let old = self.currentInput { self.session.removeInput(old) }
            if self.session.canAddInput(input) {
                self.session.addInput(input)
                self.currentInput = input
                DispatchQueue.main.async { self.position = target }
            } else if let old = self.currentInput, self.session.canAddInput(old) {
                self.session.addInput(old)
            }
            self.session.commitConfiguration()
        }
    }

    func captureTiltFrames() async -> [Data] {
        guard !isCapturing else { return [] }
        await MainActor.run { self.isCapturing = true }

        return await withCheckedContinuation { continuation in
            outputQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(returning: [])
                    return
                }
                self.captureFrames = self.rollingFrames
                self.remainingPostFrames = self.postFrameCount
                self.completion = { frames in
                    DispatchQueue.main.async { self.isCapturing = false }
                    continuation.resume(returning: frames)
                }
            }
        }
    }

    private func requestPermissionAndConfigure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configure()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted { self?.configure() }
                else { DispatchQueue.main.async { self?.permissionDenied = true } }
            }
        default:
            permissionDenied = true
        }
    }

    private func configure() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .high

            guard let device = Self.camera(position: .back),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }

            self.session.addInput(input)
            self.currentInput = input

            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
            ]
            self.videoOutput.setSampleBufferDelegate(self, queue: self.outputQueue)

            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
            }
            if let connection = self.videoOutput.connection(with: .video), connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }

            self.session.commitConfiguration()
            DispatchQueue.main.async {
                self.isReady = true
                self.start()
            }
        }
    }

    private static func camera(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .builtInDualWideCamera, .builtInDualCamera],
            mediaType: .video,
            position: position
        )
        return discovery.devices.first
    }

    private func makeJPEG(from sampleBuffer: CMSampleBuffer) -> Data? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        var image = CIImage(cvPixelBuffer: pixelBuffer)

        if position == .front {
            image = image.oriented(.leftMirrored)
        } else {
            image = image.oriented(.right)
        }

        let extent = image.extent
        let maxSide = max(extent.width, extent.height)
        if maxSide > 1440 {
            let scale = 1440 / maxSide
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }

        guard let cg = context.createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.86)
    }
}

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard lastAcceptedTime < 0 || pts - lastAcceptedTime >= frameInterval else { return }
        lastAcceptedTime = pts
        guard let data = makeJPEG(from: sampleBuffer) else { return }

        rollingFrames.append(data)
        if rollingFrames.count > preFrameCount {
            rollingFrames.removeFirst(rollingFrames.count - preFrameCount)
        }

        guard remainingPostFrames > 0 else { return }
        captureFrames.append(data)
        remainingPostFrames -= 1

        if remainingPostFrames == 0 {
            let frames = captureFrames
            captureFrames = []
            let done = completion
            completion = nil
            done?(frames)
        }
    }
}
