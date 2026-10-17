import Photos
import SwiftUI

struct TiltReviewView: View {
    @EnvironmentObject private var library: MomentLibrary
    @Environment(\.dismiss) private var dismiss
    @StateObject private var motion = MotionService()

    let moment: TiltMoment
    var showsCloseButton: Bool = false
    var onClose: (() -> Void)? = nil

    @State private var manualProgress: Double = 0.5
    @State private var manualOverride = false
    @State private var exportURL: URL?
    @State private var exporting = false
    @State private var toast: String?
    @State private var showShare = false

    private var progress: Double { manualOverride ? manualProgress : motion.progress }
    private var index: Int { FrameIndexMapper.index(progress: progress, frameCount: moment.frameCount) }

    var body: some View {
        ZStack {
            Color.tiltInk.ignoresSafeArea()
            VStack(spacing: 0) {
                topBar
                Spacer(minLength: 10)
                photo
                Spacer(minLength: 12)
                scrubber
                actions
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)

            if let toast {
                Text(toast)
                    .font(.system(size: 13, weight: .black, design: .rounded))
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(.white, in: Capsule())
                    .foregroundStyle(Color.tiltInk)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .frame(maxHeight: .infinity, alignment: .top)
                    .padding(.top, 72)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .onAppear { motion.start() }
        .onDisappear { motion.stop() }
        .sheet(isPresented: $showShare) {
            if let exportURL { ActivityView(items: [exportURL]) }
        }
    }

    private var topBar: some View {
        HStack {
            Button {
                if let onClose { onClose() } else { dismiss() }
            } label: {
                Image(systemName: showsCloseButton ? "xmark" : "chevron.left")
                    .font(.system(size: 16, weight: .black))
                    .frame(width: 44, height: 44)
                    .background(.white.opacity(0.12), in: Circle())
            }
            Spacer()
            VStack(spacing: 1) {
                Text("TILT")
                    .font(.system(size: 16, weight: .black, design: .rounded))
                Text("\(index + 1) / \(moment.frameCount)")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.5))
            }
            Spacer()
            Button(role: .destructive) {
                library.delete(moment)
                if let onClose { onClose() } else { dismiss() }
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 15, weight: .bold))
                    .frame(width: 44, height: 44)
                    .background(.white.opacity(0.12), in: Circle())
            }
        }
        .foregroundStyle(.white)
        .padding(.top, 8)
    }

    private var photo: some View {
        GeometryReader { geo in
            Group {
                if let image = library.image(for: moment, index: index) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Color.black
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .clipped()
            .clipShape(RoundedRectangle(cornerRadius: 32, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 32).stroke(.white.opacity(0.12), lineWidth: 1))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        manualOverride = true
                        manualProgress = min(max(value.location.x / max(1, geo.size.width), 0), 1)
                    }
                    .onEnded { _ in
                        withAnimation(.easeOut(duration: 0.25)) { manualOverride = false }
                    }
            )
            .overlay(alignment: .topLeading) {
                StickerLabel(text: manualOverride ? "DRAGGING" : (motion.isAvailable ? "TILT ME" : "DRAG ME"), symbol: "arrow.left.and.right", color: .tiltLemon)
                    .padding(12)
            }
        }
        .aspectRatio(3/4, contentMode: .fit)
    }

    private var scrubber: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "arrow.turn.up.left")
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(.white.opacity(0.10)).frame(height: 5)
                        Capsule().fill(Color.tiltPink).frame(width: max(12, geo.size.width * progress), height: 5)
                        Circle().fill(.white).frame(width: 14, height: 14).offset(x: max(0, min(geo.size.width - 14, geo.size.width * progress - 7)))
                    }
                    .frame(maxHeight: .infinity)
                }
                .frame(height: 18)
                Image(systemName: "arrow.turn.up.right")
            }
            .foregroundStyle(.white.opacity(0.64))

            Text("lean your phone left ↔ right  •  or drag the photo")
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundStyle(.white.opacity(0.50))
        }
        .padding(.vertical, 10)
    }

    private var actions: some View {
        HStack(spacing: 10) {
            Button { saveCurrentFrame() } label: {
                actionLabel("save frame", "square.and.arrow.down", color: .white)
            }
            .buttonStyle(BouncyButtonStyle())

            Button { library.setCover(index, for: moment); showToast("cover set ✦") } label: {
                actionLabel("set cover", "heart.fill", color: .tiltPink)
            }
            .buttonStyle(BouncyButtonStyle())

            Button { Task { await export() } } label: {
                actionLabel(exporting ? "making…" : "share loop", exporting ? "hourglass" : "arrowshape.turn.up.right.fill", color: .tiltLemon)
            }
            .buttonStyle(BouncyButtonStyle())
            .disabled(exporting)
        }
    }

    private func actionLabel(_ title: String, _ symbol: String, color: Color) -> some View {
        VStack(spacing: 6) {
            Image(systemName: symbol).font(.system(size: 18, weight: .black))
            Text(title).font(.system(size: 10, weight: .black, design: .rounded)).lineLimit(1)
        }
        .foregroundStyle(Color.tiltInk)
        .frame(maxWidth: .infinity)
        .frame(height: 62)
        .background(color, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func saveCurrentFrame() {
        guard let image = library.image(for: moment, index: index) else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            } completionHandler: { success, _ in
                if success { DispatchQueue.main.async { showToast("saved to photos ♡") } }
            }
        }
    }

    @MainActor
    private func export() async {
        exporting = true
        defer { exporting = false }
        do {
            let images = library.allImages(for: moment)
            exportURL = try await TiltVideoExporter.exportLoop(images: images)
            showShare = true
        } catch {
            showToast("export failed :(")
        }
    }

    private func showToast(_ text: String) {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { toast = text }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) {
            withAnimation { toast = nil }
        }
    }
}
