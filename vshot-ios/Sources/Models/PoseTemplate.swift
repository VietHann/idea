import CoreGraphics
import Foundation

struct PoseTemplate: Identifiable, Hashable {
    let id: String
    let name: String
    let subtitle: String
    let emoji: String
    let category: PoseCategory
    let accent: String
    let points: [String: CGPoint]
    static func == (lhs: PoseTemplate, rhs: PoseTemplate) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

enum PoseCategory: String, CaseIterable, Identifiable {
    case everyday = "Everyday"
    case cafe = "Café"
    case street = "Street"
    case outfit = "Outfit"
    case cute = "Cute"
    var id: String { rawValue }
    var icon: String {
        switch self {
        case .everyday: return "sparkles"
        case .cafe: return "cup.and.saucer.fill"
        case .street: return "figure.walk"
        case .outfit: return "tshirt.fill"
        case .cute: return "heart.fill"
        }
    }
}

enum PoseLibrary {
    static let all: [PoseTemplate] = [
        PoseTemplate(id: "soft-s", name: "Soft S", subtitle: "one hip out · relaxed shoulders", emoji: "🌷", category: .everyday, accent: "pink", points: [
            "nose": .init(x: 0.52, y: 0.08), "leftShoulder": .init(x: 0.40, y: 0.23), "rightShoulder": .init(x: 0.62, y: 0.22), "leftElbow": .init(x: 0.33, y: 0.39), "rightElbow": .init(x: 0.71, y: 0.39), "leftWrist": .init(x: 0.39, y: 0.51), "rightWrist": .init(x: 0.62, y: 0.51), "leftHip": .init(x: 0.43, y: 0.51), "rightHip": .init(x: 0.59, y: 0.50), "leftKnee": .init(x: 0.38, y: 0.73), "rightKnee": .init(x: 0.65, y: 0.75), "leftAnkle": .init(x: 0.33, y: 0.96), "rightAnkle": .init(x: 0.69, y: 0.94)]),
        PoseTemplate(id: "tiny-heart", name: "Tiny Heart", subtitle: "hands up · soft smile", emoji: "🫶", category: .cute, accent: "lavender", points: [
            "nose": .init(x: 0.50, y: 0.09), "leftShoulder": .init(x: 0.39, y: 0.24), "rightShoulder": .init(x: 0.61, y: 0.24), "leftElbow": .init(x: 0.38, y: 0.36), "rightElbow": .init(x: 0.62, y: 0.36), "leftWrist": .init(x: 0.47, y: 0.31), "rightWrist": .init(x: 0.53, y: 0.31), "leftHip": .init(x: 0.43, y: 0.52), "rightHip": .init(x: 0.57, y: 0.52), "leftKnee": .init(x: 0.41, y: 0.75), "rightKnee": .init(x: 0.60, y: 0.74), "leftAnkle": .init(x: 0.39, y: 0.96), "rightAnkle": .init(x: 0.62, y: 0.96)]),
        PoseTemplate(id: "coffee-side", name: "Coffee Side", subtitle: "turn 30° · cup near face", emoji: "☕️", category: .cafe, accent: "peach", points: [
            "nose": .init(x: 0.56, y: 0.10), "leftShoulder": .init(x: 0.42, y: 0.25), "rightShoulder": .init(x: 0.61, y: 0.23), "leftElbow": .init(x: 0.36, y: 0.40), "rightElbow": .init(x: 0.66, y: 0.35), "leftWrist": .init(x: 0.43, y: 0.51), "rightWrist": .init(x: 0.59, y: 0.22), "leftHip": .init(x: 0.44, y: 0.52), "rightHip": .init(x: 0.58, y: 0.51), "leftKnee": .init(x: 0.41, y: 0.74), "rightKnee": .init(x: 0.62, y: 0.77), "leftAnkle": .init(x: 0.38, y: 0.95), "rightAnkle": .init(x: 0.67, y: 0.94)]),
        PoseTemplate(id: "street-step", name: "Street Step", subtitle: "mid-step · look away", emoji: "🪩", category: .street, accent: "mint", points: [
            "nose": .init(x: 0.47, y: 0.09), "leftShoulder": .init(x: 0.39, y: 0.24), "rightShoulder": .init(x: 0.59, y: 0.22), "leftElbow": .init(x: 0.30, y: 0.39), "rightElbow": .init(x: 0.70, y: 0.39), "leftWrist": .init(x: 0.24, y: 0.55), "rightWrist": .init(x: 0.74, y: 0.54), "leftHip": .init(x: 0.42, y: 0.51), "rightHip": .init(x: 0.57, y: 0.51), "leftKnee": .init(x: 0.34, y: 0.72), "rightKnee": .init(x: 0.63, y: 0.70), "leftAnkle": .init(x: 0.26, y: 0.93), "rightAnkle": .init(x: 0.77, y: 0.91)]),
        PoseTemplate(id: "model-cross", name: "Model Cross", subtitle: "cross legs · hand on waist", emoji: "🎀", category: .outfit, accent: "lemon", points: [
            "nose": .init(x: 0.52, y: 0.08), "leftShoulder": .init(x: 0.40, y: 0.23), "rightShoulder": .init(x: 0.62, y: 0.23), "leftElbow": .init(x: 0.30, y: 0.38), "rightElbow": .init(x: 0.67, y: 0.39), "leftWrist": .init(x: 0.24, y: 0.54), "rightWrist": .init(x: 0.58, y: 0.50), "leftHip": .init(x: 0.43, y: 0.51), "rightHip": .init(x: 0.58, y: 0.51), "leftKnee": .init(x: 0.59, y: 0.72), "rightKnee": .init(x: 0.46, y: 0.73), "leftAnkle": .init(x: 0.67, y: 0.95), "rightAnkle": .init(x: 0.39, y: 0.94)]),
        PoseTemplate(id: "peace-pop", name: "Peace Pop", subtitle: "V-sign · weight on one leg", emoji: "✌️", category: .cute, accent: "pink", points: [
            "nose": .init(x: 0.49, y: 0.09), "leftShoulder": .init(x: 0.39, y: 0.24), "rightShoulder": .init(x: 0.61, y: 0.23), "leftElbow": .init(x: 0.30, y: 0.38), "rightElbow": .init(x: 0.67, y: 0.25), "leftWrist": .init(x: 0.24, y: 0.54), "rightWrist": .init(x: 0.61, y: 0.12), "leftHip": .init(x: 0.43, y: 0.52), "rightHip": .init(x: 0.58, y: 0.51), "leftKnee": .init(x: 0.40, y: 0.75), "rightKnee": .init(x: 0.63, y: 0.72), "leftAnkle": .init(x: 0.38, y: 0.96), "rightAnkle": .init(x: 0.70, y: 0.91)])
    ]
    static func forCategory(_ category: PoseCategory) -> [PoseTemplate] {
        let matches = all.filter { $0.category == category }
        return matches.isEmpty ? all : matches
    }
}
