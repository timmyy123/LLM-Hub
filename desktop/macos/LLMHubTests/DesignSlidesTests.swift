import XCTest
@testable import LLMHub

/// What Export › PPTX puts on each slide (the same rules as Windows and Linux).
final class DesignSlidesTests: XCTestCase {
    /// Shaped like a page Gemma 4 12B wrote for "menu highlights and opening hours": the menu and the hours are <div> and
    /// <span> rows, and the hours card is the second heading of its section. Both were missing from the deck.
    private let cafe = """
    <html><head><style>body { color: #222 }</style></head><body>
    <header class="hero"><h1>Morning ritual, refined.</h1><p>Small-batch roasting meets artisanal pastry.</p>
      <a href="#menu" class="cta-button">View Today's Menu</a></header>
    <section class="menu-preview" id="menu">
      <div class="section-title"><h2>Daily Offerings</h2><p>Crafted with seasonal ingredients.</p></div>
      <div class="menu-grid">
        <div class="menu-category"><h3>The Brew</h3>
          <div class="menu-item"><div><span class="item-name">Pour Over Selection</span>
            <span class="item-desc">Single origin beans, hand-poured.</span></div><span class="item-price">$6.00</span></div>
          <div class="menu-item"><div><span class="item-name">The Hearth Latte</span></div><span class="item-price">$7.50</span></div>
        </div>
        <div class="menu-category"><h3>The Bakery</h3>
          <div class="menu-item"><div><span class="item-name">Morning Croissant</span></div><span class="item-price">$5.50</span></div>
        </div>
      </div>
    </section>
    <section class="info-grid" id="hours">
      <div class="info-card"><h2>Visit Us</h2><p>124 Willow Lane<br>Old Town District</p></div>
      <div class="info-card"><h2>Opening Hours</h2>
        <div class="hours-list"><span>Mon &mdash; Fri</span> <span>7:00 AM - 4:00 PM</span></div>
        <div class="hours-list"><span>Sunday</span> <span>8:00 AM - 2:00 PM</span></div>
      </div>
    </section>
    <footer><p>&copy; 2024 The Hearth &amp; Husk.</p></footer>
    </body></html>
    """

    func testMenuAndHoursReachTheSlides() {
        let slides = DesignDocument.slides(cafe, folder: FileManager.default.temporaryDirectory, name: "The Hearth & Husk")
        XCTAssertEqual(slides.map(\.title), ["Morning ritual, refined.", "Daily Offerings", "Visit Us", "The Hearth & Husk"])
        XCTAssertEqual(slides[0].lines, ["Small-batch roasting meets artisanal pastry.", "View Today's Menu"])
        XCTAssertEqual(slides[1].lines, ["Crafted with seasonal ingredients.", "The Brew", "Pour Over Selection Single origin beans, hand-poured.",
                                         "$6.00", "The Hearth Latte", "$7.50", "The Bakery", "Morning Croissant", "$5.50"])
        XCTAssertEqual(slides[2].lines, ["124 Willow Lane", "Old Town District", "Opening Hours", "Mon — Fri 7:00 AM - 4:00 PM", "Sunday 8:00 AM - 2:00 PM"])
        XCTAssertEqual(slides[3].lines, ["© 2024 The Hearth & Husk."])
    }

    func testLongSectionContinuesOnMoreSlides() {
        let items = (1...20).map { "<li>Item \($0)</li>" }.joined()
        let slides = DesignDocument.slides("<section><h2>Menu</h2><ul>\(items)</ul></section>", folder: FileManager.default.temporaryDirectory, name: "Cafe")
        XCTAssertEqual(slides.map(\.title), ["Menu", "Menu", "Menu"])
        XCTAssertEqual(slides.map(\.lines.count), [9, 9, 2])
        XCTAssertEqual(slides[2].lines.last, "Item 20")
    }

    func testPlaceholdersAreLeftOut() {
        let slides = DesignDocument.slides("<section><h2>Team</h2><p>[Photo of the team]</p><p>Six bakers.</p></section>",
                                           folder: FileManager.default.temporaryDirectory, name: "Cafe")
        XCTAssertEqual(slides.first?.lines, ["Six bakers."])
    }
}
