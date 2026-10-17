import SwiftUI

extension Color {
    static let vsInk = Color(red: 0.10, green: 0.09, blue: 0.14)
    static let vsCream = Color(red: 1.00, green: 0.98, blue: 0.95)
    static let vsPeach = Color(red: 1.00, green: 0.73, blue: 0.68)
    static let vsPink = Color(red: 1.00, green: 0.79, blue: 0.88)
    static let vsLavender = Color(red: 0.78, green: 0.72, blue: 1.00)
    static let vsMint = Color(red: 0.72, green: 0.94, blue: 0.84)
    static let vsLemon = Color(red: 1.00, green: 0.91, blue: 0.58)
}

struct VShotBackground: View {
    var body: some View {
        LinearGradient(colors: [.vsCream, .vsPink.opacity(0.22), .vsLavender.opacity(0.18)], startPoint: .topLeading, endPoint: .bottomTrailing)
            .ignoresSafeArea()
            .overlay(alignment: .topTrailing) {
                Circle().fill(Color.vsLemon.opacity(0.33)).frame(width: 190, height: 190).blur(radius: 2).offset(x: 70, y: -70)
            }
            .overlay(alignment: .bottomLeading) {
                Circle().fill(Color.vsMint.opacity(0.34)).frame(width: 230, height: 230).offset(x: -85, y: 80)
            }
    }
}

struct PressScaleStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.spring(response: 0.22, dampingFraction: 0.72), value: configuration.isPressed)
    }
}

struct Pill: View {
    let text: String
    let icon: String
    let color: Color
    var body: some View {
        HStack(spacing: 6) { Image(systemName: icon); Text(text) }
            .font(.system(size: 12, weight: .black, design: .rounded))
            .foregroundStyle(Color.vsInk)
            .padding(.horizontal, 11)
            .padding(.vertical, 8)
            .background(color.opacity(0.92), in: Capsule())
    }
}
