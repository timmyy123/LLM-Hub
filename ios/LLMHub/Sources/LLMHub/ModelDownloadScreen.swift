import SwiftUI

struct ModelDownloadScreen: View {
    @State private var downloadProgress: Double = 0.0
    @State private var downloadStatus = "Ready"
    @State private var isDownloading = false
    
    var body: some View {
        VStack {
            List {
                Section(header: Text("Available Models")) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Gemma 3 2B Instruct 4-bit")
                            .font(.headline)
                        Text("mlx-community/gemma-3n-E4B-it-4bit")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Divider()
                        
                        if isDownloading {
                            ProgressView(value: downloadProgress, total: 1.0)
                                .progressViewStyle(LinearProgressViewStyle())
                        }
                        
                        HStack {
                            Text("Status: \(downloadStatus)")
                                .font(.footnote)
                                .foregroundColor(isDownloading ? .orange : (downloadStatus == "Downloaded" ? .green : .gray))
                            
                            Spacer()
                            
                            Button(action: {
                                isDownloading = true
                                downloadStatus = "Downloading (10%)"
                                downloadProgress = 0.1
                                
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    downloadProgress = 1.0
                                    downloadStatus = "Downloaded"
                                    isDownloading = false
                                }
                            }) {
                                Text(downloadStatus == "Downloaded" ? "Delete" : (isDownloading ? "Pause" : "Download"))
                                    .font(.subheadline)
                                    .padding(.horizontal, 15)
                                    .padding(.vertical, 8)
                                    .background(downloadStatus == "Downloaded" ? Color.red : Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(8)
                            }
                        }
                    }
                    .padding(.vertical, 5)
                }
            }
        }
        .navigationTitle("Model Downloads")
    }
}
