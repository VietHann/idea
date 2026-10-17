import SwiftUI

struct RootView: View {
    @State private var category: PoseCategory = .everyday
    @State private var showCamera = false
    @State private var selectedPose = PoseLibrary.all[0]
    @AppStorage("vshot.didOnboard") private var didOnboard = false

    var body: some View {
        NavigationStack {
            ZStack {
                VShotBackground()
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 24) { header; hero; vibePicker; poseIdeas; coachCard }
                        .padding(.horizontal, 18).padding(.bottom, 110)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .overlay(alignment: .bottom) { bottomCTA }
            .fullScreenCover(isPresented: $showCamera) { VShotCameraView(initialPose: selectedPose) }
            .sheet(isPresented: Binding(get: { !didOnboard }, set: { if !$0 { didOnboard = true } })) {
                OnboardingView { didOnboard = true }.interactiveDismissDisabled()
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 0) {
                Text("VShot").font(.system(size: 37, weight: .black, design: .rounded)).tracking(-1.6)
                Text("your pocket photographer").font(.system(size: 13, weight: .semibold, design: .rounded)).foregroundStyle(Color.vsInk.opacity(0.58))
            }
            Spacer()
            ZStack { Circle().fill(Color.vsLavender.opacity(0.72)); Image(systemName: "camera.aperture").font(.system(size: 21, weight: .black)).foregroundStyle(Color.vsInk) }
                .frame(width: 50, height: 50).rotationEffect(.degrees(8))
        }.padding(.top, 14)
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack { Pill(text: "ON-DEVICE AI", icon: "sparkles", color: .vsLemon); Spacer(); Pill(text: "OFFLINE", icon: "airplane", color: .vsMint) }
            Text("No more\n“how do I pose?”").font(.system(size: 36, weight: .black, design: .rounded)).tracking(-1.4).foregroundStyle(Color.vsInk)
            Text("Match the ghost pose, fix your framing live, then let VShot pick a color vibe that fits the light.")
                .font(.system(size: 15, weight: .medium, design: .rounded)).foregroundStyle(Color.vsInk.opacity(0.66)).fixedSize(horizontal: false, vertical: true)
            Button { selectedPose = PoseLibrary.forCategory(category).first ?? PoseLibrary.all[0]; showCamera = true } label: {
                HStack(spacing: 9) { Image(systemName: "viewfinder"); Text("start a guided shot") }
                    .font(.system(size: 16, weight: .black, design: .rounded)).foregroundStyle(.white).frame(maxWidth: .infinity).frame(height: 56).background(Color.vsInk, in: Capsule())
            }.buttonStyle(PressScaleStyle())
        }
        .padding(20)
        .background(RoundedRectangle(cornerRadius: 32, style: .continuous).fill(Color.vsPink.opacity(0.56)).overlay(alignment: .topTrailing) {
            Text("✦").font(.system(size: 80, weight: .black)).foregroundStyle(.white.opacity(0.56)).offset(x: -8, y: 4)
        })
        .overlay(RoundedRectangle(cornerRadius: 32).stroke(.white.opacity(0.72), lineWidth: 1.4))
    }

    private var vibePicker: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("what's the vibe?").font(.system(size: 20, weight: .black, design: .rounded))
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 9) {
                    ForEach(PoseCategory.allCases) { item in
                        Button { category = item; if let pose = PoseLibrary.forCategory(item).first { selectedPose = pose } } label: {
                            HStack(spacing: 7) { Image(systemName: item.icon); Text(item.rawValue) }
                                .font(.system(size: 13, weight: .black, design: .rounded)).foregroundStyle(category == item ? .white : Color.vsInk)
                                .padding(.horizontal, 14).frame(height: 40).background(category == item ? Color.vsInk : Color.white.opacity(0.72), in: Capsule())
                        }.buttonStyle(PressScaleStyle())
                    }
                }
            }
        }
    }

    private var poseIdeas: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text("pose ideas").font(.system(size: 20, weight: .black, design: .rounded)); Spacer(); Text("tap to shoot").font(.system(size: 12, weight: .bold, design: .rounded)).foregroundStyle(.secondary) }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(PoseLibrary.forCategory(category)) { pose in
                        Button { selectedPose = pose; showCamera = true } label: { PoseIdeaCard(pose: pose) }.buttonStyle(PressScaleStyle())
                    }
                }
            }
        }
    }

    private var coachCard: some View {
        HStack(spacing: 14) {
            ZStack { RoundedRectangle(cornerRadius: 18).fill(Color.vsLavender.opacity(0.48)); Image(systemName: "figure.stand.line.dotted.figure.stand").font(.system(size: 25, weight: .bold)) }.frame(width: 62, height: 62)
            VStack(alignment: .leading, spacing: 4) {
                Text("VShot coaches the photographer too").font(.system(size: 15, weight: .black, design: .rounded))
                Text("Move left, step closer, lower the camera — live, before the shutter.").font(.system(size: 12.5, weight: .medium, design: .rounded)).foregroundStyle(Color.vsInk.opacity(0.60))
            }
            Spacer(minLength: 0)
        }.padding(16).background(.white.opacity(0.68), in: RoundedRectangle(cornerRadius: 24))
    }

    private var bottomCTA: some View {
        Button { showCamera = true } label: {
            HStack(spacing: 10) { Image(systemName: "camera.fill"); Text("open VShot") }
                .font(.system(size: 17, weight: .black, design: .rounded)).foregroundStyle(.white).padding(.horizontal, 28).frame(height: 58)
                .background(Color.vsInk, in: Capsule()).shadow(color: Color.vsInk.opacity(0.22), radius: 18, y: 8)
        }.buttonStyle(PressScaleStyle()).padding(.bottom, 14)
    }
}

private struct PoseIdeaCard: View {
    let pose: PoseTemplate
    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            ZStack {
                RoundedRectangle(cornerRadius: 23, style: .continuous).fill(cardColor)
                Text(pose.emoji).font(.system(size: 48)).rotationEffect(.degrees(-4))
                Image(systemName: "figure.stand").font(.system(size: 52, weight: .ultraLight)).foregroundStyle(Color.vsInk.opacity(0.22)).offset(x: 44, y: 25)
            }.frame(width: 166, height: 150)
            Text(pose.name).font(.system(size: 16, weight: .black, design: .rounded))
            Text(pose.subtitle).font(.system(size: 11.5, weight: .semibold, design: .rounded)).foregroundStyle(Color.vsInk.opacity(0.54)).lineLimit(1)
        }.padding(10).background(.white.opacity(0.72), in: RoundedRectangle(cornerRadius: 28))
    }
    private var cardColor: Color {
        switch pose.accent { case "peach": return .vsPeach.opacity(0.62); case "lavender": return .vsLavender.opacity(0.58); case "mint": return .vsMint.opacity(0.62); case "lemon": return .vsLemon.opacity(0.62); default: return .vsPink.opacity(0.62) }
    }
}

private struct OnboardingView: View {
    let done: () -> Void
    var body: some View {
        ZStack {
            VShotBackground()
            VStack(spacing: 24) {
                Spacer()
                ZStack {
                    RoundedRectangle(cornerRadius: 42).fill(Color.vsPink.opacity(0.62)).frame(width: 220, height: 220).rotationEffect(.degrees(-5))
                    Image(systemName: "camera.viewfinder").font(.system(size: 74, weight: .black)).foregroundStyle(Color.vsInk)
                    Text("✦").font(.system(size: 54, weight: .black)).offset(x: 82, y: -78)
                }
                VStack(spacing: 8) {
                    Text("Meet VShot").font(.system(size: 38, weight: .black, design: .rounded))
                    Text("A tiny photographer in your pocket — pose coach, framing coach, and color coach. All on your iPhone.")
                        .font(.system(size: 16, weight: .medium, design: .rounded)).foregroundStyle(Color.vsInk.opacity(0.62)).multilineTextAlignment(.center).padding(.horizontal, 28)
                }
                VStack(spacing: 11) { onboardingLine("Match a ghost pose", "figure.stand"); onboardingLine("Fix framing before you shoot", "viewfinder"); onboardingLine("Get an automatic color vibe", "camera.filters") }
                Button(action: done) {
                    Text("let's shoot ♡").font(.system(size: 17, weight: .black, design: .rounded)).foregroundStyle(.white).frame(maxWidth: .infinity).frame(height: 58).background(Color.vsInk, in: Capsule())
                }.buttonStyle(PressScaleStyle()).padding(.horizontal, 24)
                Spacer(minLength: 30)
            }
        }
    }
    private func onboardingLine(_ text: String, _ icon: String) -> some View {
        HStack(spacing: 12) { Image(systemName: icon).frame(width: 28); Text(text).font(.system(size: 14, weight: .bold, design: .rounded)); Spacer() }
            .padding(.horizontal, 18).frame(height: 48).background(.white.opacity(0.72), in: Capsule()).padding(.horizontal, 24)
    }
}
