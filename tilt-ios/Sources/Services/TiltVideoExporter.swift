import AVFoundation
import UIKit

struct TiltVideoExporter {
    static func exportLoop(images: [UIImage], fps: Int32 = 12) async throws -> URL {
        guard let first = images.first, images.count > 1 else { throw ExportError.notEnoughFrames }

        let forward = images
        let backward = Array(images.dropFirst().dropLast().reversed())
        let sequence = forward + backward + forward + backward

        let target = fittedSize(for: first.size, maxLongEdge: 1080)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("TILT-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: url)

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Int(target.width),
            AVVideoHeightKey: Int(target.height),
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 5_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
        ])
        input.expectsMediaDataInRealTime = false

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32ARGB),
                kCVPixelBufferWidthKey as String: Int(target.width),
                kCVPixelBufferHeightKey as String: Int(target.height)
            ]
        )

        guard writer.canAdd(input) else { throw ExportError.writerSetup }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? ExportError.writerSetup }
        writer.startSession(atSourceTime: .zero)

        guard let pool = adaptor.pixelBufferPool else { throw ExportError.pixelBuffer }
        let frameDuration = CMTime(value: 1, timescale: fps)

        for (index, image) in sequence.enumerated() {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 2_000_000)
            }
            guard let pixel = pixelBuffer(from: image, size: target, pool: pool) else {
                throw ExportError.pixelBuffer
            }
            let time = CMTimeMultiply(frameDuration, multiplier: Int32(index))
            guard adaptor.append(pixel, withPresentationTime: time) else {
                throw writer.error ?? ExportError.appendFailed
            }
        }

        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        if writer.status == .failed { throw writer.error ?? ExportError.appendFailed }
        return url
    }

    private static func fittedSize(for size: CGSize, maxLongEdge: CGFloat) -> CGSize {
        let long = max(size.width, size.height)
        guard long > 0 else { return CGSize(width: 1080, height: 1440) }
        let scale = min(1, maxLongEdge / long)
        func even(_ x: CGFloat) -> CGFloat { CGFloat(max(2, Int(x * scale) / 2 * 2)) }
        return CGSize(width: even(size.width), height: even(size.height))
    }

    private static func pixelBuffer(from image: UIImage, size: CGSize, pool: CVPixelBufferPool) -> CVPixelBuffer? {
        var optional: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optional) == kCVReturnSuccess,
              let buffer = optional else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: base,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return nil }

        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(origin: .zero, size: size))
        context.interpolationQuality = .high

        guard let cg = image.cgImage else { return nil }
        context.draw(cg, in: CGRect(origin: .zero, size: size))
        return buffer
    }

    enum ExportError: LocalizedError {
        case notEnoughFrames, writerSetup, pixelBuffer, appendFailed
        var errorDescription: String? {
            switch self {
            case .notEnoughFrames: "Not enough frames to export."
            case .writerSetup: "Could not prepare the video exporter."
            case .pixelBuffer: "Could not render a video frame."
            case .appendFailed: "Could not write the video."
            }
        }
    }
}
