import SwiftUI

extension Color {
    static let tiltInk = Color(red: 0.12, green: 0.10, blue: 0.16)
    static let tiltCream = Color(red: 1.00, green: 0.97, blue: 0.93)
    static let tiltPink = Color(red: 1.00, green: 0.66, blue: 0.76)
    static let tiltLavender = Color(red: 0.76, green: 0.70, blue: 1.00)
    static let tiltLemon = Color(red: 1.00, green: 0.90, blue: 0.48)
    static let tiltMint = Color(red: 0.66, green: 0.91, blue: 0.79)
}

struct SoftBackground: View {
    var body: some View {
        ZStack {
            Color.tiltCream.ignoresSafeArea()
            Circle()
                .fill(Color.tiltPink.opacity(0.24))
                .frame(width: 310, height: 310)
                .blur(radius: 26)
                .offset(x: 170, y: -300)
            Circle()
                .fill(Color.tiltLavender.opacity(0.20))
                .frame(width: 280, height: 280)
                .blur(radius: 30)
                .offset(x: -170, y: 300)
        }
    }
}

struct StickerLabel: View {
    let text: String
    let symbol: String
    var color: Color = .white

    var body: some View {
        Label(text, systemImage: symbol)
            .font(.system(size: 13, weight: .bold, design: .rounded))
            .foregroundStyle(Color.tiltInk)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(color.opacity(0.94), in: Capsule())
            .overlay(Capsule().stroke(Color.tiltInk.opacity(0.09), lineWidth: 1))
    }
}

struct BouncyButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
            .rotationEffect(.degrees(configuration.isPressed ? -1.2 : 0))
            .animation(.spring(response: 0.22, dampingFraction: 0.58), value: configuration.isPressed)
    }
}
