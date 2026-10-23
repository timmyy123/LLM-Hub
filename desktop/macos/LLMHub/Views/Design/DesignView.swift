import SwiftUI

struct DesignView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        PlaceholderView(section: "Design", symbol: "sparkles")
    }
}
