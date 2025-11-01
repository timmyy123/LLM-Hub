//
//  LLMHubApp.swift
//  LLMHub
//
//  iOS version of LLM Hub
//

import SwiftUI

@main
struct LLMHubApp: App {
    @StateObject private var inferenceService = InferenceService()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(inferenceService)
        }
    }
}
