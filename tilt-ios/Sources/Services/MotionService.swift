import CoreMotion
import Foundation

final class MotionService: ObservableObject {
    @Published private(set) var progress: Double = 0.5
    @Published private(set) var isAvailable = false

    private let manager = CMMotionManager()
    private let queue = OperationQueue()
    private var smoothedRoll: Double = 0

    init() {
        queue.qualityOfService = .userInteractive
    }

    func start() {
        guard manager.isDeviceMotionAvailable else {
            DispatchQueue.main.async { self.isAvailable = false }
            return
        }

        isAvailable = true
        manager.deviceMotionUpdateInterval = 1.0 / 60.0
        manager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let raw = motion.attitude.roll
            self.smoothedRoll = (self.smoothedRoll * 0.82) + (raw * 0.18)
            let mapped = FrameIndexMapper.progress(roll: self.smoothedRoll)
            DispatchQueue.main.async {
                self.progress = mapped
            }
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
    }
}
