import SwiftUI
import UIKit

struct VShotCameraView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var camera = CameraPoseService()
    @State private var selectedPose: PoseTemplate
    @State private var capturedImage: UIImage?
    @State private var capturedBrightness: CGFloat = 0.55
    @State private var showReview = false
    @State private var showCurrentSkeleton = true
    @State private var shutterScale: CGFloat = 1

    init(initialPose: PoseTemplate) { _selectedPose = State(initialValue: initialPose) }
    private var match: PoseMatchResult { PoseMatcher.match(current: camera.joints, template: selectedPose) }
    private var lightScore: Int { PoseMatcher.clampScore(Int(100 - abs(camera.brightness - 0.55) * 120)) }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: camera.session).ignoresSafeArea()
            LinearGradient(colors: [.black.opacity(0.58), .clear, .clear, .black.opacity(0.72)], startPoint: .top, endPoint: .bottom).ignoresSafeArea()
            PoseOverlayView(current: camera.joints, target: selectedPose, showCurrent: showCurrentSkeleton).ignoresSafeArea()
            VStack(spacing: 0) { topBar; liveCoach; Spacer(); poseStrip; bottomBar }
                .padding(.horizontal, 15).padding(.top, 8).padding(.bottom, 10)
            if camera.permissionDenied { permissionView }
        }
        .statusBarHidden()
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
        .fullScreenCover(isPresented: $showReview, onDismiss: { camera.start() }) {
            if let image = capturedImage { PhotoReviewView(image: image, sourceBrightness: capturedBrightness, pose: selectedPose) }
        }
    }

    private var topBar: some View {
        HStack {
            roundButton("xmark") { dismiss() }
            Spacer()
            VStack(spacing: 1) {
                Text("VShot").font(.system(size: 17, weight: .black, design: .rounded))
                Text("AI POSE COACH").font(.system(size: 8.5, weight: .black, design: .rounded)).tracking(1.5).opacity(0.7)
            }.foregroundStyle(.white)
            Spacer()
            roundButton("arrow.triangle.2.circlepath.camera.fill") { camera.toggleCamera() }
        }
    }

    private var liveCoach: some View {
        VStack(spacing: 9) {
            HStack(spacing: 8) {
                scoreChip(title: "POSE", score: camera.joints.isEmpty ? nil : match.score, color: .vsPink)
                scoreChip(title: "LIGHT", score: lightScore, color: .vsLemon)
                Button { showCurrentSkeleton.toggle() } label: {
                    HStack(spacing: 5) { Image(systemName: "figure.stand"); Text(showCurrentSkeleton ? "BODY ON" : "BODY OFF") }
                        .font(.system(size: 10, weight: .black, design: .rounded)).foregroundStyle(.white).padding(.horizontal, 10).frame(height: 31)
                        .background(.black.opacity(0.42), in: Capsule())
                }
            }
            Text(camera.guidance)
                .font(.system(size: 15, weight: .black, design: .rounded)).foregroundStyle(Color.vsInk).padding(.horizontal, 16).frame(height: 39)
                .background(.white.opacity(0.90), in: Capsule()).shadow(color: .black.opacity(0.12), radius: 12, y: 4)
                .animation(.easeInOut(duration: 0.18), value: camera.guidance)
        }.padding(.top, 14)
    }

    private var poseStrip: some View {
        VStack(spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(selectedPose.emoji)  \(selectedPose.name)").font(.system(size: 15, weight: .black, design: .rounded))
                    Text(selectedPose.subtitle).font(.system(size: 10.5, weight: .bold, design: .rounded)).opacity(0.66)
                }.foregroundStyle(.white)
                Spacer()
                Button { shufflePose() } label: {
                    HStack(spacing: 6) { Image(systemName: "sparkles"); Text("suggest") }
                        .font(.system(size: 11, weight: .black, design: .rounded)).foregroundStyle(Color.vsInk).padding(.horizontal, 12).frame(height: 34).background(Color.vsLemon, in: Capsule())
                }.buttonStyle(PressScaleStyle())
            }
            .padding(.horizontal, 14).padding(.vertical, 11).background(.black.opacity(0.40), in: RoundedRectangle(cornerRadius: 18))

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(PoseLibrary.all) { pose in
                        Button { withAnimation(.spring(response: 0.28, dampingFraction: 0.78)) { selectedPose = pose } } label: {
                            Text(pose.emoji).font(.system(size: 23)).frame(width: 46, height: 46)
                                .background(selectedPose.id == pose.id ? Color.white : Color.black.opacity(0.38), in: RoundedRectangle(cornerRadius: 15))
                                .overlay(RoundedRectangle(cornerRadius: 15).stroke(selectedPose.id == pose.id ? Color.vsPink : .white.opacity(0.18), lineWidth: 2))
                        }.buttonStyle(PressScaleStyle())
                    }
                }
            }
        }
    }

    private var bottomBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(match.score >= 82 ? "nailed it ✦" : match.score >= 60 ? "almost there" : "match the ghost").font(.system(size: 12, weight: .black, design: .rounded))
                Text(camera.joints.isEmpty ? "find a person" : "pose \(match.score)%").font(.system(size: 10, weight: .bold, design: .rounded)).opacity(0.65)
            }.foregroundStyle(.white).frame(width: 100, alignment: .leading)
            Spacer()
            Button { capture() } label: {
                ZStack {
                    Circle().stroke(.white, lineWidth: 5).frame(width: 78, height: 78)
                    Circle().fill(Color.vsPink).frame(width: 62, height: 62)
                    Image(systemName: "sparkles").font(.system(size: 20, weight: .black)).foregroundStyle(Color.vsInk)
                }.scaleEffect(shutterScale)
            }.buttonStyle(PressScaleStyle())
            Spacer()
            VStack(alignment: .trailing, spacing: 3) {
                Text("AI color").font(.system(size: 12, weight: .black, design: .rounded))
                Text(ColorGradeEngine.recommendedPreset(brightness: camera.brightness).uppercased()).font(.system(size: 10, weight: .black, design: .rounded)).opacity(0.68)
            }.foregroundStyle(.white).frame(width: 100, alignment: .trailing)
        }.padding(.top, 12)
    }

    private func capture() {
        shutterScale = 0.86
        withAnimation(.spring(response: 0.25, dampingFraction: 0.5)) { shutterScale = 1 }
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        Task {
            if let data = await camera.captureCurrentFrame(), let image = UIImage(data: data) {
                await MainActor.run { capturedImage = image; capturedBrightness = camera.brightness; camera.stop(); showReview = true }
            }
        }
    }

    private func shufflePose() {
        if let next = PoseLibrary.all.filter({ $0.id != selectedPose.id }).randomElement() {
            withAnimation(.spring(response: 0.32, dampingFraction: 0.76)) { selectedPose = next }
        }
    }

    private func scoreChip(title: String, score: Int?, color: Color) -> some View {
        HStack(spacing: 5) { Circle().fill(color).frame(width: 7, height: 7); Text(title); Text(score.map(String.init) ?? "--") }
            .font(.system(size: 10, weight: .black, design: .rounded)).foregroundStyle(.white).padding(.horizontal, 10).frame(height: 31).background(.black.opacity(0.42), in: Capsule())
    }

    private func roundButton(_ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) { Image(systemName: icon).font(.system(size: 15, weight: .black)).foregroundStyle(.white).frame(width: 42, height: 42).background(.black.opacity(0.40), in: Circle()) }
            .buttonStyle(PressScaleStyle())
    }

    private var permissionView: some View {
        VStack(spacing: 12) {
            Image(systemName: "camera.fill").font(.system(size: 38))
            Text("Camera access is off").font(.system(size: 22, weight: .black, design: .rounded))
            Text("Enable Camera for VShot in Settings to use pose guidance.").font(.system(size: 14, weight: .medium, design: .rounded)).multilineTextAlignment(.center).opacity(0.7)
        }.foregroundStyle(.white).padding(28).background(.black.opacity(0.82), in: RoundedRectangle(cornerRadius: 28)).padding(24)
    }
}
