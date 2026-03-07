import SwiftUI

struct ChatScreen: View {
    @State private var messageText = ""
    @AppStorage("enableTokenStreaming") private var enableStreaming: Bool = true
    @AppStorage("showResultStatus") private var showStatus: Bool = true
    @State private var resultStatus = "5.32 tok/s • 42 tokens • 8.1s"
    
    var body: some View {
        VStack {
            if showStatus {
                HStack {
                    Image(systemName: "bolt.fill")
                        .foregroundColor(.green)
                    Text("Result Status: \(resultStatus)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 8)
                
                Divider()
            }
            
            ScrollView {
                VStack(alignment: .leading, spacing: 15) {
                    ChatBubble(message: "Hello! I am Gemma 3 running directly on your iPhone using MLX. How can I help you?", isUser: false)
                }
                .padding()
            }
            
            Spacer()
            
            HStack {
                TextField("Type a message...", text: $messageText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding(.horizontal)
                
                Button(action: {
                    // Start generation
                }) {
                    Image(systemName: "paperplane.fill")
                        .foregroundColor(.white)
                        .padding(10)
                        .background(Color.blue)
                        .clipShape(Circle())
                }
                .padding(.trailing)
            }
            .padding(.bottom, 10)
        }
        .navigationTitle("AI Chat")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    // Open model config
                }) {
                    Image(systemName: "slider.horizontal.3")
                }
            }
        }
    }
}

struct ChatBubble: View {
    var message: String
    var isUser: Bool
    
    var body: some View {
        HStack {
            if isUser { Spacer() }
            
            Text(message)
                .padding(12)
                .background(isUser ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(isUser ? .white : .primary)
                .cornerRadius(15)
                .frame(maxWidth: 250, alignment: isUser ? .trailing : .leading)
            
            if !isUser { Spacer() }
        }
    }
}
