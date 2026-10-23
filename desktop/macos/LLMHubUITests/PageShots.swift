import XCTest

/// Real window screenshots of chosen pages (Liquid Glass is composited by the window server, so the app's own
/// capture cannot show it). Pages come from SHOT_PAGES, set with TEST_RUNNER_SHOT_PAGES=design,studio when running
/// xcodebuild test; extra app arguments (for example a --designtest run first) from SHOT_ARGS, split on "|".
final class PageShots: XCTestCase {
    func testPageShots() throws {
        let env = ProcessInfo.processInfo.environment
        let pages = (env["SHOT_PAGES"] ?? "chat,agent,design,studio").split(separator: ",").map(String.init)
        let extra = env["SHOT_ARGS"].map { $0.split(separator: "|").map(String.init) } ?? []
        let wait = Double(env["SHOT_WAIT"] ?? "") ?? 4
        for page in pages {
            let app = XCUIApplication()
            // --capture makes DevTools switch to the page; the file itself is not used here.
            app.launchArguments = ["--page", page, "--capture", "/tmp/llmhub-devcapture.png", "--wait", "600000"] + extra
            app.launch()
            XCTAssertTrue(app.windows.firstMatch.waitForExistence(timeout: 30))
            Thread.sleep(forTimeInterval: wait)
            let shot = XCTAttachment(screenshot: app.windows.firstMatch.screenshot())
            shot.name = page
            shot.lifetime = .keepAlways
            add(shot)
            app.terminate()
        }
    }
}
