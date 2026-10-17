import Photos
import SwiftUI
import UIKit

struct PhotoReviewView: View {
    @Environment(\.dismiss) private var dismiss
    let image: UIImage
    let sourceBrightness: CGFloat
    let pose: PoseTemplate
    @State private var presetID = "ai"
    @State private var saved = false
    @State private var showShare = false
    @State private var shareImage: UIImage?

    private var rendered: UIImage { ColorGradeEngine.render(image, presetID: presetID, sourceBrightness: sourceBrightness) }

    var body: some View {
        ZStack {
            Color.vsCream.ignoresSafeArea()
            VStack(spacing: 0) { header; imageCard; gradePicker; aiNote; actions }
        }
        .statusBarHidden()
        .sheet(isPresented: $showShare) { if let shareImage { ShareSheet(items: [shareImage]) } }
    }

    private var header: some View {
        HStack {
            Button { dismiss() } label: { Image(systemName: "chevron.left").font(.system(size: 15, weight: .black)).frame(width: 42, height: 42).background(.white, in: Circle()) }.foregroundStyle(Color.vsInk)
            Spacer()
            VStack(spacing: 1) {
                Text("your shot").font(.system(size: 19, weight: .black, design: .rounded))
                Text("\(pose.emoji) \(pose.name)").font(.system(size: 10, weight: .bold, design: .rounded)).opacity(0.56)
            }
            Spacer()
            ZStack { Circle().fill(saved ? Color.vsMint : Color.vsPink).frame(width: 42, height: 42); Image(systemName: saved ? "checkmark" : "sparkles").font(.system(size: 14, weight: .black)) }
        }.padding(.horizontal, 16).padding(.vertical, 10)
    }

    private var imageCard: some View {
        Image(uiImage: rendered).resizable().scaledToFill().frame(maxWidth: .infinity).frame(height: UIScreen.main.bounds.height * 0.56).clipped()
            .clipShape(RoundedRectangle(cornerRadius: 30, style: .continuous))
            .overlay(alignment: .topLeading) {
                HStack(spacing: 6) { Image(systemName: "camera.filters"); Text(presetName.uppercased()) }
                    .font(.system(size: 10, weight: .black, design: .rounded)).foregroundStyle(.white).padding(.horizontal, 11).frame(height: 31)
                    .background(.black.opacity(0.44), in: Capsule()).padding(12)
            }
            .padding(.horizontal, 14).shadow(color: Color.vsInk.opacity(0.15), radius: 20, y: 8)
    }

    private var gradePicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 9) {
                ForEach(ColorGradeEngine.presets) { preset in
                    Button { withAnimation(.easeInOut(duration: 0.18)) { presetID = preset.id } } label: {
                        VStack(spacing: 5) { Text(preset.emoji).font(.system(size: 20)); Text(preset.name).font(.system(size: 10.5, weight: .black, design: .rounded)) }
                            .foregroundStyle(presetID == preset.id ? .white : Color.vsInk).frame(width: 72, height: 58)
                            .background(presetID == preset.id ? Color.vsInk : Color.white, in: RoundedRectangle(cornerRadius: 17))
                    }.buttonStyle(PressScaleStyle())
                }
            }.padding(.horizontal, 15)
        }.padding(.top, 12)
    }

    private var aiNote: some View {
        HStack(spacing: 9) {
            Image(systemName: "sparkles")
            Text("AI Pick likes **\(ColorGradeEngine.recommendedPreset(brightness: sourceBrightness).capitalized)** for this light.").font(.system(size: 11.5, weight: .semibold, design: .rounded))
            Spacer(minLength: 0)
        }.foregroundStyle(Color.vsInk.opacity(0.70)).padding(.horizontal, 18).padding(.top, 10)
    }

    private var actions: some View {
        HStack(spacing: 10) {
            Button { save() } label: {
                HStack(spacing: 8) { Image(systemName: saved ? "checkmark" : "square.and.arrow.down.fill"); Text(saved ? "saved" : "save") }
                    .font(.system(size: 15, weight: .black, design: .rounded)).foregroundStyle(Color.vsInk).frame(maxWidth: .infinity).frame(height: 54)
                    .background(saved ? Color.vsMint : Color.vsPink, in: Capsule())
            }.buttonStyle(PressScaleStyle())
            Button { shareImage = rendered; showShare = true } label: {
                HStack(spacing: 8) { Image(systemName: "square.and.arrow.up"); Text("share") }
                    .font(.system(size: 15, weight: .black, design: .rounded)).foregroundStyle(.white).frame(maxWidth: .infinity).frame(height: 54).background(Color.vsInk, in: Capsule())
            }.buttonStyle(PressScaleStyle())
        }.padding(.horizontal, 15).padding(.vertical, 12)
    }

    private var presetName: String {
        if presetID == "ai" { return "AI · \(ColorGradeEngine.recommendedPreset(brightness: sourceBrightness))" }
        return ColorGradeEngine.presets.first(where: { $0.id == presetID })?.name ?? "Clean"
    }

    private func save() {
        let finalImage = rendered
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            PHPhotoLibrary.shared().performChanges { PHAssetChangeRequest.creationRequestForAsset(from: finalImage) } completionHandler: { success, _ in
                if success { DispatchQueue.main.async { withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) { saved = true }; UINotificationFeedbackGenerator().notificationOccurred(.success) } }
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController { UIActivityViewController(activityItems: items, applicationActivities: nil) }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) { }
}
