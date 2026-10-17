import Foundation
import CoreGraphics

struct EntitySpec {
    let texture: String
    let mass: CGFloat
    let points: Int
    let display: String
    let mutation: Mutation?
    let isNPC: Bool
    let baseSize: CGFloat
}

enum Mutation: String, CaseIterable {
    case freeWifi = "FREE WIFI"
    case gymBro = "GYM BRO"
    case caffeine = "CAFFEINE"
    case roadAuthority = "ROAD AUTHORITY"

    var emoji: String {
        switch self {
        case .freeWifi: return "📶"
        case .gymBro: return "💪"
        case .caffeine: return "⚡️"
        case .roadAuthority: return "🚧"
        }
    }

    var subtitle: String {
        switch self {
        case .freeWifi: return "everyone suddenly needs you"
        case .gymBro: return "bro is built different"
        case .caffeine: return "movement speed absolutely illegal"
        case .roadAuthority: return "zero qualifications. full authority."
        }
    }
}

enum ChaosEvent: CaseIterable {
    case monday
    case freeWifiPanic
    case taxSeason
    case npcPatch
    case absoluteCinema

    var title: String {
        switch self {
        case .monday: return "MONDAY DETECTED"
        case .freeWifiPanic: return "FREE WIFI PANIC"
        case .taxSeason: return "TAX SEASON 💀"
        case .npcPatch: return "NPC PATCH FAILED"
        case .absoluteCinema: return "ABSOLUTE CINEMA"
        }
    }

    var subtitle: String {
        switch self {
        case .monday: return "motivation -80%"
        case .freeWifiPanic: return "the whole city wants your password"
        case .taxSeason: return "they found you"
        case .npcPatch: return "walking logic has left the chat"
        case .absoluteCinema: return "✋😐🤚"
        }
    }
}
