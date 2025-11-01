//
//  ContentView.swift
//  LLMHub
//
//  Main navigation view
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var inferenceService: InferenceService
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            ChatView()
                .tabItem {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                }
                .tag(0)
            
            ModelSelectorView()
                .tabItem {
                    Label("Models", systemImage: "cube.box")
                }
                .tag(1)
            
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(2)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(InferenceService())
}
