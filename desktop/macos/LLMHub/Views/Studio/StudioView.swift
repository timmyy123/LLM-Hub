import SwiftUI

struct StudioView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        PlaceholderView(section: "Studio", symbol: "sparkles")
    }
}
