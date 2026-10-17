import XCTest
@testable import TILT

final class TILTTests: XCTestCase {
    func testFrameIndexClamps() {
        XCTAssertEqual(FrameIndexMapper.index(progress: -3, frameCount: 15), 0)
        XCTAssertEqual(FrameIndexMapper.index(progress: 2, frameCount: 15), 14)
    }

    func testFrameIndexCenter() {
        XCTAssertEqual(FrameIndexMapper.index(progress: 0.5, frameCount: 15), 7)
    }

    func testRollMapping() {
        XCTAssertEqual(FrameIndexMapper.progress(roll: 0), 0.5, accuracy: 0.0001)
        XCTAssertEqual(FrameIndexMapper.progress(roll: -9), 0, accuracy: 0.0001)
        XCTAssertEqual(FrameIndexMapper.progress(roll: 9), 1, accuracy: 0.0001)
    }
}
