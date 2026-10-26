import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// Studio › Image and Video: the generate form on the left, the gallery on the right.
struct ImageStudioView: View {
    var body: some View {
        HStack(spacing: 0) {
            GenerateForm()
                .frame(width: 340)
            Divider()
            GalleryPane()
        }
    }
}

private struct GenerateForm: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var studio = app.studio
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    FieldLabel(L.s("StudioWorkflow"))
                    Picker("", selection: $studio.workflow) {
                        ForEach(studio.workflows) { Text($0.name).tag($0) }
                    }
                    .labelsHidden()
                    .controlSize(.large)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("WorkflowPicker")
                    if !studio.downloadNote.isEmpty {
                        Text(studio.downloadNote).font(.system(size: 12)).foregroundStyle(Theme.textFaint)
                            .accessibilityIdentifier("DownloadNote")
                    }
                }

                // Upscaling takes only a picture: no prompt, size, steps or seed.
                if studio.workflow.id != "upscale" {
                    VStack(alignment: .leading, spacing: 6) {
                        FieldLabel(L.s("StudioPrompt"))
                        TextEditor(text: $studio.prompt)
                            .font(.system(size: 14))
                            .scrollContentBackground(.hidden)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 8)
                            .frame(minHeight: 100, maxHeight: 180)
                            .glassEffect(.regular, in: .rect(cornerRadius: 10))
                            .accessibilityIdentifier("PromptBox")
                        Button {
                            studio.improvePrompt()
                        } label: {
                            HStack(spacing: 6) {
                                if studio.isImproving { ProgressView().controlSize(.mini) } else { Image(systemName: "sparkles") }
                                Text(L.s("StudioImprove"))
                            }
                            .font(.system(size: 12))
                        }
                        .buttonStyle(.glass)
                        .disabled(studio.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || studio.isImproving || studio.isGenerating)
                        .accessibilityIdentifier("ImproveButton")
                    }

                    Grid(horizontalSpacing: 10, verticalSpacing: 10) {
                        GridRow {
                            field(L.s("StudioAspect")) {
                                Picker("", selection: $studio.aspect) {
                                    ForEach(StudioService.aspects, id: \.self) { Text($0).tag($0) }
                                }
                                .labelsHidden()
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .accessibilityIdentifier("AspectPicker")
                            }
                            field(L.s("StudioImages")) {
                                Picker("", selection: $studio.count) {
                                    ForEach(StudioService.counts, id: \.self) { Text("\($0)").tag($0) }
                                }
                                .labelsHidden()
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .disabled(studio.workflow.kind == "video" || studio.workflow.needsImage)
                                .accessibilityIdentifier("CountPicker")
                            }
                        }
                        GridRow {
                            field(L.s("StudioSteps")) {
                                TextField("", value: $studio.steps, format: .number)
                                    .textFieldStyle(.roundedBorder)
                                    .accessibilityIdentifier("StepsBox")
                            }
                            field(L.s("StudioSeed")) {
                                TextField(L.s("StudioRandom"), text: $studio.seed)
                                    .textFieldStyle(.roundedBorder)
                                    .accessibilityIdentifier("SeedBox")
                            }
                        }
                    }
                    .controlSize(.large)
                }

                if studio.needsReference {
                    ReferencePicker()
                }

                if !studio.status.isEmpty {
                    Text(studio.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                        .accessibilityIdentifier("StudioStatus")
                }
                if !studio.comfy.status.isEmpty {
                    Text(studio.comfy.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("ComfyProgress")
                }

                if studio.isGenerating {
                    Button {
                        studio.cancel()
                    } label: {
                        Text(L.s("Cancel")).frame(maxWidth: .infinity, minHeight: 30)
                    }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                    .accessibilityIdentifier("CancelGenerateButton")
                } else {
                    Button {
                        studio.generate()
                    } label: {
                        Text(L.s("StudioGenerate")).fontWeight(.semibold).frame(maxWidth: .infinity, minHeight: 30)
                    }
                    .buttonStyle(.glassProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.return, modifiers: .command)
                    .accessibilityIdentifier("GenerateButton")
                }
            }
            .padding(20)
        }
    }

    private func field<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            FieldLabel(title)
            content().frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// The dashed "Reference image" button; shows the chosen picture once there is one.
private struct ReferencePicker: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let studio = app.studio
        let path = studio.referenceImage
        let has = !path.isEmpty && FileManager.default.fileExists(atPath: path)
        Button {
            pick()
        } label: {
            HStack(spacing: 10) {
                if has {
                    Thumbnail(url: URL(fileURLWithPath: path), maxPixel: 120)
                        .frame(width: 44, height: 30)
                        .clipShape(.rect(cornerRadius: 5))
                } else {
                    Image(systemName: "plus").font(.system(size: 12, weight: .medium))
                }
                Text(has ? URL(fileURLWithPath: path).lastPathComponent : L.s("StudioReference"))
                    .font(.system(size: 13))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12)
            .frame(minHeight: 40)
            .contentShape(Rectangle())
            .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(Theme.controlBorder, style: StrokeStyle(lineWidth: 1, dash: has ? [] : [4, 3])))
        }
        .buttonStyle(.plain)
        .foregroundStyle(.primary.opacity(0.85))
        .accessibilityIdentifier("ReferenceButton")
    }

    private func pick() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP]
        panel.allowsMultipleSelection = false
        panel.directoryURL = ComfyService.gallery
        if panel.runModal() == .OK, let url = panel.url { app.studio.referenceImage = url.path }
    }
}

private struct GalleryPane: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var studio = app.studio
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Text(L.s("StudioGallery")).font(.system(size: 15, weight: .medium))
                Text(L.f("StudioSavedTo", studio.galleryPathLabel)).font(.system(size: 12)).foregroundStyle(Theme.textFaint)
                    .accessibilityIdentifier("SavedToNote")
                Spacer()
                if studio.isGenerating {
                    HStack(spacing: 10) {
                        Text(studio.status).font(.system(size: 12)).foregroundStyle(.secondary).lineLimit(1)
                        ProgressView(value: studio.progress).frame(width: 140)
                    }
                    .accessibilityIdentifier("GenerationProgress")
                }
            }

            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 12)], spacing: 12) {
                    if studio.isGenerating {
                        ForEach(0..<max(1, studio.runningCount), id: \.self) { index in
                            PendingTile(text: index == 0 ? studio.status : L.s("StudioQueued"))
                        }
                    }
                    ForEach(studio.gallery) { item in
                        GalleryTile(item: item, selected: studio.selected == item)
                            .onTapGesture(count: 2) { NSWorkspace.shared.open(item.url) }
                            .onTapGesture { studio.selected = item }
                            .contextMenu {
                                Button(L.s("StudioOpen")) { NSWorkspace.shared.open(item.url) }
                                Button(L.s("StudioShowInFinder")) { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                                Divider()
                                Button(L.s("StudioUseInChat")) { studio.useInChat(item) }.disabled(item.isVideo)
                                Button(L.s("StudioAnimate")) { studio.animate(item) }.disabled(item.isVideo)
                                Button(L.s("StudioUpscale")) { studio.upscale(item) }.disabled(item.isVideo)
                            }
                    }
                }
                .padding(2)
            }
            .accessibilityIdentifier("GalleryGrid")

            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 8) {
                    let item = studio.selected
                    Button(L.s("StudioUseInChat")) { if let item { studio.useInChat(item) } }
                        .disabled(item == nil || item?.isVideo == true)
                        .accessibilityIdentifier("UseInChatButton")
                    Button(L.s("StudioAnimate")) { if let item { studio.animate(item) } }
                        .disabled(item == nil || item?.isVideo == true)
                        .accessibilityIdentifier("AnimateButton")
                    Button(L.s("StudioUpscale")) { if let item { studio.upscale(item) } }
                        .disabled(item == nil || item?.isVideo == true)
                        .accessibilityIdentifier("UpscaleButton")
                    Button(L.s("StudioOpenComfy")) { studio.openInComfy() }
                        .accessibilityIdentifier("OpenComfyButton")
                    Spacer()
                    if let item {
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([item.url])
                        } label: {
                            Label(L.s("StudioShowInFinder"), systemImage: "folder")
                        }
                    }
                }
                .buttonStyle(.glass)
                .controlSize(.large)
            }
        }
        .padding(20)
        .onAppear { studio.loadGallery() }
    }
}

private struct PendingTile: View {
    let text: String
    var body: some View {
        RoundedRectangle(cornerRadius: 12)
            .strokeBorder(Theme.controlBorder, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            .aspectRatio(3 / 2, contentMode: .fit)
            .overlay {
                Text(text).font(.system(size: 12)).foregroundStyle(Theme.textMuted).lineLimit(2).multilineTextAlignment(.center).padding(8)
            }
            .glassEffect(.regular, in: .rect(cornerRadius: 12))
    }
}

private struct GalleryTile: View {
    let item: GalleryItem
    let selected: Bool

    var body: some View {
        Color.clear
            .aspectRatio(3 / 2, contentMode: .fit)
            .overlay { Thumbnail(url: item.url, maxPixel: 480) }
            .clipShape(.rect(cornerRadius: 12))
            .overlay(alignment: .bottomLeading) {
                if item.isVideo {
                    Image(systemName: "play.fill")
                        .font(.system(size: 11))
                        .padding(7)
                        .glassEffect(.regular, in: .circle)
                        .padding(8)
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 12).strokeBorder(selected ? Color.accentColor : .clear, lineWidth: 2)
            }
            .contentShape(.rect(cornerRadius: 12))
            .help(item.name)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(item.name)
            .accessibilityAddTraits(selected ? [.isImage, .isSelected] : .isImage)
            .accessibilityIdentifier("GalleryItem")
    }
}

/// A downsampled image (first frame for animated WebP), decoded off the main thread and cached.
struct Thumbnail: View {
    let url: URL
    let maxPixel: Int
    @State private var image: NSImage?

    private static let cache = NSCache<NSString, NSImage>()

    var body: some View {
        ZStack {
            Rectangle().fill(.fill.quaternary)
            if let image {
                Image(nsImage: image).resizable().scaledToFill()
            }
        }
        .task(id: url) {
            let key = "\(url.path)#\(maxPixel)" as NSString
            if let cached = Self.cache.object(forKey: key) {
                image = cached
                return
            }
            let url = url, maxPixel = maxPixel
            let loaded = await Task.detached(priority: .utility) { () -> NSImage? in
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                          kCGImageSourceCreateThumbnailFromImageAlways: true,
                          kCGImageSourceThumbnailMaxPixelSize: maxPixel,
                          kCGImageSourceCreateThumbnailWithTransform: true,
                      ] as CFDictionary) else { return nil }
                return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
            }.value
            if let loaded { Self.cache.setObject(loaded, forKey: key) }
            image = loaded
        }
    }
}
