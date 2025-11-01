//
//  ChatView.swift
//  LLMHub
//
//  Main chat interface
//

import SwiftUI

struct ChatView: View {
    @EnvironmentObject var inferenceService: InferenceService
    @StateObject private var viewModel: ChatViewModel
    @State private var scrollProxy: ScrollViewProxy?
    
    init() {
        // Note: We'll get the inferenceService from the environment
        // This is a workaround for initialization
        _viewModel = StateObject(wrappedValue: ChatViewModel(inferenceService: InferenceService()))
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Model status bar
                if let model = inferenceService.currentModel {
                    HStack {
                        Image(systemName: inferenceService.isModelLoaded ? "checkmark.circle.fill" : "circle")
                            .foregroundColor(inferenceService.isModelLoaded ? .green : .gray)
                        Text(model.name)
                            .font(.caption)
                            .lineLimit(1)
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray6))
                } else {
                    HStack {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                        Text("No model loaded")
                            .font(.caption)
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray6))
                }
                
                // Messages list
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onAppear {
                        scrollProxy = proxy
                    }
                    .onChange(of: viewModel.messages.count) { _ in
                        if let lastMessage = viewModel.messages.last {
                            withAnimation {
                                proxy.scrollTo(lastMessage.id, anchor: .bottom)
                            }
                        }
                    }
                }
                
                // Input area
                VStack(spacing: 8) {
                    if let error = viewModel.errorMessage {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.red)
                            Spacer()
                        }
                        .padding(.horizontal)
                    }
                    
                    HStack(spacing: 12) {
                        TextField("Type a message...", text: $viewModel.currentInput, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                            .lineLimit(1...5)
                            .disabled(viewModel.isGenerating || !inferenceService.isModelLoaded)
                        
                        Button(action: {
                            Task {
                                await viewModel.sendMessage()
                            }
                        }) {
                            Image(systemName: viewModel.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                                .font(.title2)
                                .foregroundColor(viewModel.currentInput.isEmpty || !inferenceService.isModelLoaded ? .gray : .blue)
                        }
                        .disabled(viewModel.currentInput.isEmpty || !inferenceService.isModelLoaded)
                    }
                    .padding()
                }
                .background(Color(.systemBackground))
            }
            .navigationTitle("Chat")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: {
                            viewModel.clearChat()
                        }) {
                            Label("Clear Chat", systemImage: "trash")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .onAppear {
            // Update viewModel with the actual inferenceService from environment
            // This is necessary because we can't inject it during init
        }
    }
}

struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .padding(12)
                    .background(message.isUser ? Color.blue : Color(.systemGray5))
                    .foregroundColor(message.isUser ? .white : .primary)
                    .cornerRadius(16)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if !message.isUser {
                Spacer()
            }
        }
    }
}

#Preview {
    ChatView()
        .environmentObject(InferenceService())
}
