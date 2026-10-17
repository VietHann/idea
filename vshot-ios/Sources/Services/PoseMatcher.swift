import CoreGraphics
import Foundation

struct PoseMatchResult {
    let score: Int
    let matchedJointCount: Int
}

enum PoseMatcher {
    static func clampScore(_ value: Int) -> Int { min(100, max(0, value)) }

    static func match(current: [String: CGPoint], template: PoseTemplate) -> PoseMatchResult {
        let usableKeys = template.points.keys.filter { current[$0] != nil }
        guard usableKeys.count >= 5 else { return .init(score: 0, matchedJointCount: usableKeys.count) }
        let currentSubset = Dictionary(uniqueKeysWithValues: usableKeys.compactMap { key in current[key].map { (key, $0) } })
        let normalized = normalize(currentSubset)
        var total: CGFloat = 0
        var weight: CGFloat = 0
        for key in usableKeys {
            guard let a = normalized[key], let b = template.points[key] else { continue }
            let d = hypot(a.x - b.x, a.y - b.y)
            let jointWeight: CGFloat = ["nose", "leftWrist", "rightWrist", "leftAnkle", "rightAnkle"].contains(key) ? 1.25 : 1.0
            total += d * jointWeight
            weight += jointWeight
        }
        let avg = total / max(weight, 0.001)
        let raw = Int((1.0 - min(avg / 0.34, 1.0)) * 100.0)
        return .init(score: clampScore(raw), matchedJointCount: usableKeys.count)
    }

    static func normalize(_ points: [String: CGPoint]) -> [String: CGPoint] {
        guard !points.isEmpty else { return [:] }
        let xs = points.values.map(\.x), ys = points.values.map(\.y)
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else { return points }
        let width = max(maxX - minX, 0.001), height = max(maxY - minY, 0.001)
        return points.mapValues { CGPoint(x: ($0.x - minX) / width, y: ($0.y - minY) / height) }
    }
}
