import SwiftUI
import SpriteKit

struct RootView: View {
    @State private var started = false
    @State private var pulse = false

    var body: some View {
        ZStack {
            if started {
                GameHostView(onExit: { started = false })
                    .transition(.opacity.combined(with: .scale(scale: 1.02)))
            } else {
                HomeView(started: $started, pulse: $pulse)
                    .transition(.opacity)
            }
        }
        .animation(.spring(response: 0.55, dampingFraction: 0.86), value: started)
    }
}

private struct HomeView: View {
    @Binding var started: Bool
    @Binding var pulse: Bool

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.10, green: 0.08, blue: 0.14), Color(red: 0.16, green: 0.11, blue: 0.20)], startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea()

            Circle().fill(Color(red: 0.69, green: 0.95, blue: 0.43).opacity(0.13)).frame(width: 360, height: 360).blur(radius: 2).offset(x: -150, y: -250)
            Circle().fill(Color.pink.opacity(0.12)).frame(width: 320, height: 320).offset(x: 180, y: 280)

            VStack(spacing: 22) {
                Spacer()

                ZStack {
                    Circle().fill(Color.black.opacity(0.22)).frame(width: 210, height: 54).offset(y: 98).blur(radius: 8)
                    Image("blob_base")
                        .resizable().scaledToFit().frame(width: 210, height: 210)
                        .scaleEffect(pulse ? 1.035 : 0.985)
                        .offset(y: pulse ? -4 : 4)
                        .animation(.easeInOut(duration: 1.05).repeatForever(autoreverses: true), value: pulse)
                }

                VStack(spacing: 6) {
                    Text("BLOB CITY")
                        .font(.system(size: 46, weight: .black, design: .rounded))
                        .tracking(-1.8)
                    Text("eat everything. regret nothing.")
                        .font(.system(size: 17, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white.opacity(0.68))
                }

                HStack(spacing: 8) {
                    Tag(text: "SQUISHY")
                    Tag(text: "CHAOTIC")
                    Tag(text: "OFFLINE")
                }

                Button {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    started = true
                } label: {
                    HStack(spacing: 10) {
                        Text("CAUSE PROBLEMS")
                            .font(.system(size: 18, weight: .black, design: .rounded))
                        Image(systemName: "arrow.right")
                    }
                    .foregroundStyle(Color(red: 0.10, green: 0.08, blue: 0.14))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 18)
                    .background(Color(red: 0.68, green: 0.95, blue: 0.42), in: Capsule())
                    .shadow(color: Color(red: 0.68, green: 0.95, blue: 0.42).opacity(0.28), radius: 22, y: 8)
                }
                .padding(.horizontal, 30)

                Text("drag to move  •  eat smaller stuff  •  get huge")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.42))
                    .multilineTextAlignment(.center)

                Spacer().frame(height: 34)
            }
            .padding(.top, 24)
        }
        .onAppear { pulse = true }
    }
}

private struct Tag: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .black, design: .rounded))
            .foregroundStyle(.white.opacity(0.82))
            .padding(.horizontal, 11).padding(.vertical, 7)
            .background(.white.opacity(0.08), in: Capsule())
            .overlay(Capsule().stroke(.white.opacity(0.10), lineWidth: 1))
    }
}

private struct GameHostView: View {
    let onExit: () -> Void
    @State private var scene: BlobCityScene = {
        let s = BlobCityScene(size: CGSize(width: 430, height: 932))
        s.scaleMode = .resizeFill
        return s
    }()

    var body: some View {
        ZStack(alignment: .topTrailing) {
            SpriteView(scene: scene, options: [.ignoresSiblingOrder])
                .ignoresSafeArea()
                .background(Color.black)

            Button(action: onExit) {
                Image(systemName: "xmark")
                    .font(.system(size: 15, weight: .black))
                    .foregroundStyle(.white.opacity(0.92))
                    .frame(width: 42, height: 42)
                    .background(.black.opacity(0.34), in: Circle())
                    .overlay(Circle().stroke(.white.opacity(0.12), lineWidth: 1))
            }
            .padding(.top, 8)
            .padding(.trailing, 14)
        }
    }
}
