import SwiftUI

struct PoseOverlayView: View {
    let current: [String: CGPoint]
    let target: PoseTemplate
    let showCurrent: Bool
    private let bones: [(String, String)] = [
        ("nose", "leftShoulder"), ("nose", "rightShoulder"), ("leftShoulder", "rightShoulder"),
        ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"), ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
        ("leftShoulder", "leftHip"), ("rightShoulder", "rightHip"), ("leftHip", "rightHip"),
        ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"), ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle")
    ]

    var body: some View {
        GeometryReader { _ in
            Canvas { context, size in
                let targetRect = CGRect(x: size.width * 0.15, y: size.height * 0.09, width: size.width * 0.70, height: size.height * 0.82)
                drawSkeleton(context: &context, points: target.points, transform: { p in CGPoint(x: targetRect.minX + p.x * targetRect.width, y: targetRect.minY + p.y * targetRect.height) }, color: .white.opacity(0.68), lineWidth: 5, dotRadius: 5, dashed: true)
                if showCurrent {
                    drawSkeleton(context: &context, points: current, transform: { p in CGPoint(x: p.x * size.width, y: p.y * size.height) }, color: Color.vsMint.opacity(0.95), lineWidth: 3, dotRadius: 4, dashed: false)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func drawSkeleton(context: inout GraphicsContext, points: [String: CGPoint], transform: (CGPoint) -> CGPoint, color: Color, lineWidth: CGFloat, dotRadius: CGFloat, dashed: Bool) {
        for (a, b) in bones {
            guard let pa = points[a], let pb = points[b] else { continue }
            var path = Path(); path.move(to: transform(pa)); path.addLine(to: transform(pb))
            context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, dash: dashed ? [8, 8] : []))
        }
        for p in points.values {
            let c = transform(p)
            context.fill(Path(ellipseIn: CGRect(x: c.x - dotRadius, y: c.y - dotRadius, width: dotRadius * 2, height: dotRadius * 2)), with: .color(color))
        }
    }
}
