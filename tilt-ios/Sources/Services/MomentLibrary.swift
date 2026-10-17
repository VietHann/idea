import Foundation
import UIKit

@MainActor
final class MomentLibrary: ObservableObject {
    @Published private(set) var moments: [TiltMoment] = []

    private let fm = FileManager.default
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    init() {
        try? fm.createDirectory(at: rootURL, withIntermediateDirectories: true)
        reload()
    }

    private var rootURL: URL {
        let base = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("TiltMoments", isDirectory: true)
    }

    func reload() {
        let dirs = (try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil)) ?? []
        moments = dirs.compactMap { dir in
            let manifest = dir.appendingPathComponent("manifest.json")
            guard let data = try? Data(contentsOf: manifest) else { return nil }
            return try? decoder.decode(TiltMoment.self, from: data)
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    @discardableResult
    func save(frames: [Data]) throws -> TiltMoment {
        guard !frames.isEmpty else { throw LibraryError.noFrames }
        let moment = TiltMoment(frameCount: frames.count)
        let dir = directory(for: moment)
        let framesDir = dir.appendingPathComponent("frames", isDirectory: true)
        try fm.createDirectory(at: framesDir, withIntermediateDirectories: true)

        for (index, data) in frames.enumerated() {
            try data.write(to: frameURL(momentID: moment.id, index: index), options: .atomic)
        }
        try encoder.encode(moment).write(to: dir.appendingPathComponent("manifest.json"), options: .atomic)
        moments.insert(moment, at: 0)
        return moment
    }

    func setCover(_ index: Int, for moment: TiltMoment) {
        guard let arrayIndex = moments.firstIndex(where: { $0.id == moment.id }) else { return }
        let safe = min(max(0, index), max(0, moment.frameCount - 1))
        moments[arrayIndex].coverIndex = safe
        let updated = moments[arrayIndex]
        let url = directory(for: updated).appendingPathComponent("manifest.json")
        try? encoder.encode(updated).write(to: url, options: .atomic)
    }

    func frameData(for moment: TiltMoment, index: Int) -> Data? {
        guard moment.frameCount > 0 else { return nil }
        let safe = min(max(index, 0), moment.frameCount - 1)
        return try? Data(contentsOf: frameURL(momentID: moment.id, index: safe))
    }

    func image(for moment: TiltMoment, index: Int) -> UIImage? {
        guard let data = frameData(for: moment, index: index) else { return nil }
        return UIImage(data: data)
    }

    func coverImage(for moment: TiltMoment) -> UIImage? {
        image(for: moment, index: moment.coverIndex)
    }

    func allImages(for moment: TiltMoment) -> [UIImage] {
        (0..<moment.frameCount).compactMap { image(for: moment, index: $0) }
    }

    func delete(_ moment: TiltMoment) {
        try? fm.removeItem(at: directory(for: moment))
        moments.removeAll { $0.id == moment.id }
    }

    private func directory(for moment: TiltMoment) -> URL {
        rootURL.appendingPathComponent(moment.id.uuidString, isDirectory: true)
    }

    private func frameURL(momentID: UUID, index: Int) -> URL {
        rootURL
            .appendingPathComponent(momentID.uuidString, isDirectory: true)
            .appendingPathComponent("frames", isDirectory: true)
            .appendingPathComponent(String(format: "%03d.jpg", index))
    }

    enum LibraryError: LocalizedError {
        case noFrames
        var errorDescription: String? { "No frames were captured." }
    }
}
