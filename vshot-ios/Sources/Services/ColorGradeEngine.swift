import CoreImage
import UIKit

struct GradePreset: Identifiable, Hashable {
    let id: String
    let name: String
    let emoji: String
}

enum ColorGradeEngine {
    static let presets: [GradePreset] = [
        .init(id: "ai", name: "AI Pick", emoji: "✨"), .init(id: "clean", name: "Clean", emoji: "🫧"),
        .init(id: "peach", name: "Peach", emoji: "🍑"), .init(id: "cool", name: "Cool", emoji: "🧊"),
        .init(id: "film", name: "Film", emoji: "🎞️"), .init(id: "night", name: "Night", emoji: "🌙")
    ]
    private static let context = CIContext(options: [.cacheIntermediates: false])

    static func recommendedPreset(brightness: CGFloat) -> String {
        if brightness < 0.32 { return "night" }
        if brightness > 0.72 { return "clean" }
        return "peach"
    }

    static func render(_ image: UIImage, presetID: String, sourceBrightness: CGFloat) -> UIImage {
        guard let ci = CIImage(image: image) else { return image }
        let resolved = presetID == "ai" ? recommendedPreset(brightness: sourceBrightness) : presetID
        let out: CIImage
        switch resolved {
        case "peach": out = ci.applyingFilter("CITemperatureAndTint", parameters: ["inputNeutral": CIVector(x: 6500, y: 0), "inputTargetNeutral": CIVector(x: 5600, y: 12)]).applyingFilter("CIColorControls", parameters: [kCIInputBrightnessKey: 0.05, kCIInputContrastKey: 0.96, kCIInputSaturationKey: 0.96])
        case "cool": out = ci.applyingFilter("CITemperatureAndTint", parameters: ["inputNeutral": CIVector(x: 6500, y: 0), "inputTargetNeutral": CIVector(x: 7600, y: -5)]).applyingFilter("CIColorControls", parameters: [kCIInputBrightnessKey: 0.02, kCIInputContrastKey: 1.03, kCIInputSaturationKey: 0.90])
        case "film": out = ci.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.82, kCIInputContrastKey: 0.93, kCIInputBrightnessKey: 0.025]).applyingFilter("CIPhotoEffectProcess")
        case "night": out = ci.applyingFilter("CIHighlightShadowAdjust", parameters: ["inputShadowAmount": 0.75, "inputHighlightAmount": 0.82]).applyingFilter("CIColorControls", parameters: [kCIInputBrightnessKey: 0.07, kCIInputContrastKey: 1.10, kCIInputSaturationKey: 0.88])
        default: out = ci.applyingFilter("CIColorControls", parameters: [kCIInputBrightnessKey: 0.03, kCIInputContrastKey: 1.04, kCIInputSaturationKey: 1.04])
        }
        guard let cg = context.createCGImage(out, from: out.extent) else { return image }
        return UIImage(cgImage: cg)
    }
}
