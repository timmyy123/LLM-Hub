#if DEBUG
import AppKit

/// `--modelstest [--addlocal path] [--removelocal] [--filter key] [--hfquery text] [--hfformat all|gguf|litertlm]
/// [--hfsort trendingScore|downloads|lastModified] [--hfrepo id] [--hfdownload quant] [--modelstab catalog|huggingface|installed]`:
/// drives Settings › Models the way a click would and logs what it shows ("MODELSTEST ..."). Pair with
/// `--page models --capture out.png` for a screenshot.
extension DevTools {
    static func modelsTest(_ args: [String]) async {
        guard args.contains("--modelstest") else { return }
        // As in chatTest: changing state while SwiftUI is still creating the window can leave the app without one.
        for i in 0..<40 where !NSApp.windows.contains(where: { $0.isVisible && $0.frame.width > 600 }) {
            if i == 8 { NSApp.activate() }
            try? await Task.sleep(for: .milliseconds(250))
        }
        let app = AppModel.shared
        let vm = app.models
        if let path = value("--addlocal", args) {
            vm.addLocalFile(URL(fileURLWithPath: path))
            Log.info("MODELSTEST add local: \(vm.localFileMessage)")
            for model in app.modelStore.installed where model.isLocalFile {
                Log.info("MODELSTEST local \(model.id) name='\(model.name)' quant=\(model.quant) format=\(model.format) tags=\(model.tags) "
                    + "size=\(Fmt.bytes(model.sizeBytes)) primary=\(model.primaryPath.path) mmproj=\(model.mmprojPath?.path ?? "-")")
            }
        }
        if args.contains("--removelocal") {
            for model in app.modelStore.installed where model.isLocalFile {
                let path = model.primaryPath.path
                app.modelStore.delete(model, engine: app.engines)
                Log.info("MODELSTEST removed \(model.id): record gone=\(app.modelStore.find(model.id) == nil), "
                    + "folder gone=\(!FileManager.default.fileExists(atPath: model.folder.path)), file kept=\(FileManager.default.fileExists(atPath: path))")
            }
        }
        Log.info("MODELSTEST engines: llama.cpp '\(vm.llamaEngineLine)', LiteRT-LM '\(vm.liteRtEngineLine)'")
        if let filter = value("--filter", args) {
            await vm.loadCatalog()
            for key in ModelsViewModel.filters.map(\.key) {
                vm.filter = key
                Log.info("MODELSTEST filter \(key): \(vm.visibleCatalog.map { "\($0.model.id)[\($0.selected?.fitText ?? "-")]" }.joined(separator: ", "))")
            }
            vm.filter = filter
        }
        if let query = value("--hfquery", args) {
            vm.tab = .huggingFace
            vm.query = query
            vm.hfFormat = value("--hfformat", args) ?? vm.hfFormat
            vm.hfSort = value("--hfsort", args) ?? vm.hfSort
            vm.search()
            await vm.searchTask?.value
            Log.info("MODELSTEST hf '\(query)' format=\(vm.hfFormat) sort=\(vm.hfSort): \(vm.results.count) results")
            for repo in vm.results.prefix(8) {
                let r = repo.repo
                Log.info("MODELSTEST   \(r.id) [\(repo.formatLabel)] \(repo.meta) gated=\(r.gated) downloads=\(r.downloads) "
                    + "trending=\(r.trending) modified=\(r.lastModified.map { ISO8601DateFormatter().string(from: $0) } ?? "-")")
            }
            if let id = value("--hfrepo", args) {
                if !vm.results.contains(where: { $0.id == id }) {
                    vm.results.insert(RepoItem(HfRepo(id: id, downloads: 0, likes: 0, gated: false,
                                                      format: id.lowercased().contains("litert") ? "litertlm" : "gguf"), app: app), at: 0)
                }
                vm.selectedRepoId = id
            }
            if let repo = vm.selectedRepo {
                // The file list loads in the background once the repo is selected.
                var waited = 0
                while waited < 120, repo.isLoading || repo.status == L.s("LoadingVariants") || (repo.rows.isEmpty && repo.status.isEmpty) {
                    try? await Task.sleep(for: .milliseconds(250))
                    waited += 1
                }
                Log.info("MODELSTEST detail \(repo.repo.id): '\(repo.detail)' gated=\(repo.repo.gated) url=\(repo.url) status='\(repo.status)'")
                for row in repo.rows {
                    Log.info("MODELSTEST   \(row.file) | \(row.option.variant.quant) | \(row.size) | \(row.fitText)\(row.isBest ? " (best)" : "")")
                }
                // `--hfdownload <quant>`: the row's Download button, then wait for it to install.
                if let quant = value("--hfdownload", args), let row = repo.rows.first(where: { $0.option.variant.quant == quant }) {
                    repo.downloadRow(row)
                    while app.downloads.find(row.id) != nil { try? await Task.sleep(for: .milliseconds(500)) }
                    let installed = app.modelStore.find(row.id)
                    Log.info("MODELSTEST downloaded \(row.id): installed=\(installed != nil) error=\(repo.error ?? "-") "
                        + "best now=\(repo.rows.first(where: \.isBest)?.option.variant.quant ?? "-")")
                }
            }
        }
        switch value("--modelstab", args) {
        case "catalog": vm.tab = .catalog
        case "huggingface": vm.tab = .huggingFace
        case "installed": vm.tab = .installed
        default: break
        }
    }
}
#endif
