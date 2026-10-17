import SwiftUI

struct IntroView: View {
    let done: () -> Void

    var body: some View {
        ZStack {
            SoftBackground()
            VStack(spacing: 26) {
                Spacer()
                ZStack {
                    RoundedRectangle(cornerRadius: 44, style: .continuous)
                        .fill(Color.tiltLavender.opacity(0.72))
                        .frame(width: 210, height: 245)
                        .rotationEffect(.degrees(-7))
                    RoundedRectangle(cornerRadius: 44, style: .continuous)
                        .fill(Color.tiltPink.opacity(0.90))
                        .frame(width: 210, height: 245)
                        .rotationEffect(.degrees(7))
                    VStack(spacing: 10) {
                        Image(systemName: "move.3d")
                            .font(.system(size: 64, weight: .black))
                        Text("TILT")
                            .font(.system(size: 34, weight: .black, design: .rounded))
                    }
                    .foregroundStyle(Color.tiltInk)
                }

                VStack(spacing: 8) {
                    Text("a photo you can lean into")
                        .font(.system(size: 28, weight: .black, design: .rounded))
                        .multilineTextAlignment(.center)
                    Text("Tap once. We keep the tiny moment around the shutter. Tilt your phone later to move through it — all on-device.")
                        .font(.system(size: 15, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 28)
                }

                HStack(spacing: 10) {
                    StickerLabel(text: "offline", symbol: "airplane", color: .tiltMint)
                    StickerLabel(text: "private", symbol: "lock.fill", color: .tiltLemon)
                    StickerLabel(text: "weird", symbol: "sparkles", color: .tiltPink)
                }

                Spacer()
                Button(action: done) {
                    Text("okay, show me ✦")
                        .font(.system(size: 17, weight: .black, design: .rounded))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 58)
                        .background(Color.tiltInk, in: Capsule())
                }
                .buttonStyle(BouncyButtonStyle())
                .padding(.horizontal, 24)
                .padding(.bottom, 12)
            }
        }
    }
}
