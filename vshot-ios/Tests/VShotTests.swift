import XCTest
@testable import VShot

final class VShotTests: XCTestCase {
    func testScoreClamp() {
        XCTAssertEqual(PoseMatcher.clampScore(108), 100)
        XCTAssertEqual(PoseMatcher.clampScore(-5), 0)
    }
}
