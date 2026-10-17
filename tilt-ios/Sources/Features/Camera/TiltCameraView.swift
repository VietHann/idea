import AVFoundation
import SwiftUI

struct TiltCameraView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var library: MomentLibrary
    @StateObject private var camera = CameraService()
    @State private var captured: TiltMoment?
    @State private var errorText: String?
    @State private var flash = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if camera.permissionDenied {
                permissionView
            } else {
                CameraPreview(session: camera.session)
                    .ignoresSafeArea()
                    .overlay {
                        LinearGradient(colors: [.black.opacity(0.38), .clear, .black.opacity(0.52)], startPoint: .top, endPoint: .bottom)
                            .ignoresSafeArea()
                    }
                controls
            }

            if flash {
                Color.white.ignoresSafeArea().transition(.opacity)
            }

            if let captured {
                TiltReviewView(moment: captured, showsCloseButton: true) {
                    self.captured = nil
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .zIndex(5)
            }
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
        .alert("oops", isPresented: Binding(get: { errorText != nil }, set: { if !$0 { errorText = nil } })) {
            Button("OK", role: .cancel) { errorText = nil }
        } message: {
            Text(errorText ?? "Something went wrong.")
        }
    }

    private var controls: some View {
        VStack {
            HStack {
                Button { dismiss() } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 16, weight: .black))
                        .frame(width: 46, height: 46)
                        .background(.ultraThinMaterial, in: Circle())
                }
                Spacer()
                StickerLabel(text: camera.isCapturing ? "CATCHING IT..." : "TILT MODE", symbol: "move.3d", color: camera.isCapturing ? .tiltLemon : .white)
                Spacer()
                Button { camera.toggleCamera() } label: {
                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                        .font(.system(size: 17, weight: .black))
                        .frame(width: 46, height: 46)
                        .background(.ultraThinMaterial, in: Circle())
                }
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.top, 10)

            Spacer()

            VStack(spacing: 18) {
                Text("hold still-ish. the messy bit is the point.")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.88))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 9)
                    .background(.black.opacity(0.22), in: Capsule())

                Button {
                    Task { await capture() }
                } label: {
                    ZStack {
                        Circle().fill(.white).frame(width: 86, height: 86)
                        Circle().stroke(Color.white.opacity(0.55), lineWidth: 5).frame(width: 100, height: 100)
                        Circle().fill(Color.tiltPink).frame(width: camera.isCapturing ? 34 : 68, height: camera.isCapturing ? 34 : 68)
                    }
                }
                .buttonStyle(BouncyButtonStyle())
                .disabled(camera.isCapturing || !camera.isReady)

                Text(camera.isReady ? "tap once — we grab before + after" : "warming up camera…")
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.72))
            }
            .padding(.bottom, 28)
        }
    }

    private var permissionView: some View {
        VStack(spacing: 16) {
            Text("📸")
                .font(.system(size: 58))
            Text("camera pls?")
                .font(.system(size: 30, weight: .black, design: .rounded))
                .foregroundStyle(.white)
            Text("TILT only works if it can see the tiny moment around your shutter. Frames stay on your phone.")
                .font(.system(size: 15, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
    }

    @MainActor
    private func capture() async {
        let frames = await camera.captureTiltFrames()
        guard !frames.isEmpty else {
            errorText = "The camera didn't catch enough frames. Try once more."
            return
        }
        withAnimation(.easeOut(duration: 0.09)) { flash = true }
        try? await Task.sleep(nanoseconds: 90_000_000)
        withAnimation(.easeIn(duration: 0.16)) { flash = false }

        do {
            let moment = try library.save(frames: frames)
            withAnimation(.spring(response: 0.5, dampingFraction: 0.85)) {
                captured = moment
            }
        } catch {
            errorText = error.localizedDescription
        }
    }
}
