import Foundation

enum FrameIndexMapper {
    static func index(progress: Double, frameCount: Int) -> Int {
        guard frameCount > 1 else { return 0 }
        let clamped = min(max(progress, 0), 1)
        return Int((clamped * Double(frameCount - 1)).rounded())
    }

    static func progress(roll: Double, maxRoll: Double = 0.62) -> Double {
        guard maxRoll > 0 else { return 0.5 }
        let normalized = min(max(roll / maxRoll, -1), 1)
        return (normalized + 1) / 2
    }
}
