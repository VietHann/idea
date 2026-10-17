import SwiftUI

struct RootView: View {
    @EnvironmentObject private var library: MomentLibrary
    @AppStorage("didSeeTiltIntro") private var didSeeIntro = false
    @State private var showCamera = false

    var body: some View {
        NavigationStack {
            ZStack {
                SoftBackground()
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 22) {
                        header
                        hero
                        recent
                    }
                    .padding(.horizontal, 18)
                    .padding(.bottom, 120)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .overlay(alignment: .bottom) {
                captureBar
            }
            .fullScreenCover(isPresented: $showCamera) {
                TiltCameraView()
                    .environmentObject(library)
            }
            .sheet(isPresented: Binding(get: { !didSeeIntro }, set: { if !$0 { didSeeIntro = true } })) {
                IntroView { didSeeIntro = true }
                    .presentationDetents([.large])
                    .interactiveDismissDisabled()
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("TILT")
                    .font(.system(size: 36, weight: .black, design: .rounded))
                    .tracking(-1.5)
                Text("photos that move when you do")
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            ZStack {
                Circle().fill(Color.tiltLemon)
                Image(systemName: "sparkles")
                    .font(.system(size: 20, weight: .black))
            }
            .frame(width: 48, height: 48)
            .rotationEffect(.degrees(7))
        }
        .padding(.top, 16)
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                StickerLabel(text: "NO FILTER", symbol: "wand.and.stars.inverse", color: .white)
                Spacer()
                StickerLabel(text: "100% OFFLINE", symbol: "airplane", color: .tiltMint)
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("Don't freeze it.\nTilt through it.")
                    .font(.system(size: 34, weight: .black, design: .rounded))
                    .tracking(-1)
                Text("TILT captures the tiny moment around the shutter — then your phone becomes the scrubber.")
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.tiltInk.opacity(0.68))
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 9) {
                Image(systemName: "iphone.gen3.radiowaves.left.and.right")
                Text("lean left  •  lean right")
            }
            .font(.system(size: 13, weight: .bold, design: .rounded))
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.white.opacity(0.75), in: Capsule())
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 32, style: .continuous)
                .fill(Color.tiltPink.opacity(0.60))
                .overlay(alignment: .topTrailing) {
                    Text("✦")
                        .font(.system(size: 72, weight: .black))
                        .foregroundStyle(.white.opacity(0.55))
                        .offset(x: -14, y: 8)
                }
        )
        .overlay(RoundedRectangle(cornerRadius: 32).stroke(.white.opacity(0.72), lineWidth: 1.5))
    }

    @ViewBuilder
    private var recent: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("your tilts")
                    .font(.system(size: 20, weight: .black, design: .rounded))
                Spacer()
                Text("\(library.moments.count)")
                    .font(.system(size: 13, weight: .black, design: .rounded))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(Color.tiltLavender.opacity(0.35), in: Capsule())
            }

            if library.moments.isEmpty {
                VStack(spacing: 10) {
                    Text("🫧")
                        .font(.system(size: 42))
                    Text("nothing here yet")
                        .font(.system(size: 17, weight: .black, design: .rounded))
                    Text("Your first moving photo will live here.")
                        .font(.system(size: 13, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 34)
                .background(.white.opacity(0.66), in: RoundedRectangle(cornerRadius: 26))
            } else {
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    ForEach(library.moments) { moment in
                        NavigationLink(value: moment) {
                            MomentCard(moment: moment)
                        }
                        .buttonStyle(BouncyButtonStyle())
                    }
                }
            }
        }
        .navigationDestination(for: TiltMoment.self) { moment in
            TiltReviewView(moment: moment, showsCloseButton: false)
        }
    }

    private var captureBar: some View {
        Button {
            showCamera = true
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "camera.fill")
                Text("make a tilt")
                    .font(.system(size: 17, weight: .black, design: .rounded))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 28)
            .frame(height: 58)
            .background(Color.tiltInk, in: Capsule())
            .shadow(color: Color.tiltInk.opacity(0.24), radius: 18, y: 8)
        }
        .buttonStyle(BouncyButtonStyle())
        .padding(.bottom, 16)
    }
}

private struct MomentCard: View {
    @EnvironmentObject private var library: MomentLibrary
    let moment: TiltMoment

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Group {
                if let image = library.coverImage(for: moment) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Color.tiltLavender.opacity(0.3)
                }
            }
            .frame(height: 205)
            .clipped()
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(alignment: .topTrailing) {
                Image(systemName: "move.3d")
                    .font(.system(size: 12, weight: .black))
                    .padding(8)
                    .background(.ultraThinMaterial, in: Circle())
                    .padding(8)
            }

            Text(moment.createdAt.formatted(date: .abbreviated, time: .shortened))
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundStyle(Color.tiltInk.opacity(0.65))
        }
    }
}
