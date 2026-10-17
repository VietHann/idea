import Foundation

struct TiltMoment: Identifiable, Codable, Hashable {
    let id: UUID
    let createdAt: Date
    var frameCount: Int
    var coverIndex: Int

    init(id: UUID = UUID(), createdAt: Date = .now, frameCount: Int, coverIndex: Int? = nil) {
        self.id = id
        self.createdAt = createdAt
        self.frameCount = frameCount
        self.coverIndex = coverIndex ?? max(0, frameCount / 2)
    }
}
