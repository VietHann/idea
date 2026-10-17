import SwiftUI

@main
struct TILTApp: App {
    @StateObject private var library = MomentLibrary()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(library)
                .preferredColorScheme(.light)
        }
    }
}
