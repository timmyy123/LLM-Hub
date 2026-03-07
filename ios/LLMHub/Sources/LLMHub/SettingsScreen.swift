import SwiftUI

struct SettingsScreen: View {
    @AppStorage("huggingFaceToken") private var hfToken: String = ""
    @AppStorage("enableTokenStreaming") private var enableStreaming: Bool = true
    @AppStorage("showResultStatus") private var showStatus: Bool = true
    @AppStorage("contextLength") private var contextLength: Double = 2048.0
    
    var body: some View {
        Form {
            Section(header: Text("Model Configuration")) {
                Toggle("Token Streaming", isOn: $enableStreaming)
                Toggle("Show Result Status", isOn: $showStatus)
                
                VStack(alignment: .leading) {
                    Text("Max Context Length: \(Int(contextLength))")
                    Slider(value: $contextLength, in: 512...8192, step: 256)
                }
            }
            
            Section(header: Text("Authentication")) {
                SecureField("Hugging Face Profile Token", text: $hfToken)
            }
        }
        .navigationTitle("Settings")
    }
}
