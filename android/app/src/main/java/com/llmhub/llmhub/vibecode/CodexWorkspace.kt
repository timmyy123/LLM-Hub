package com.llmhub.llmhub.vibecode

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract as Docs
import android.util.Base64
import org.json.JSONObject

/** A private Termux working copy. Check every source file before publishing any edits. */
internal class CodexWorkspace(
    private val context: Context,
    val tree: Uri,
    val home: String
) {
    private val resolver = context.contentResolver
    private data class Source(val uri: Uri, val bytes: ByteArray)
    private val original = linkedMapOf<String, Source>()
    private val ignored = setOf(".git", "node_modules", ".gradle", ".build", "build", "dist", "__pycache__", ".venv")
    private var total = 0

    var isDirect: Boolean = false
        private set
    var remote: String = "$home/workspaces/project"
        private set

    companion object {
        fun resolveRealPath(context: Context? = null, treeUri: Uri): String? {
            val uriString = treeUri.toString()
            if (uriString.startsWith("/")) {
                return runCatching { java.io.File(uriString).canonicalPath }.getOrDefault(uriString)
            }
            if (treeUri.scheme == "file") {
                val path = treeUri.path ?: return null
                return runCatching { java.io.File(path).canonicalPath }.getOrDefault(path)
            }
            if (treeUri.authority == "com.android.externalstorage.documents") {
                val docId = runCatching { Docs.getTreeDocumentId(treeUri) }.getOrNull()
                    ?: runCatching { Docs.getDocumentId(treeUri) }.getOrNull()
                    ?: treeUri.pathSegments.lastOrNull()
                    ?: return null
                val decoded = runCatching { java.net.URLDecoder.decode(docId, "UTF-8") }.getOrDefault(docId)
                val cleanDocId = decoded.substringAfter("tree/").substringAfter("document/")
                val parts = cleanDocId.split(':', limit = 2)
                val storageId = parts[0]
                val relative = if (parts.size > 1) parts[1].trim('/') else ""
                val basePath = when {
                    storageId.equals("primary", ignoreCase = true) || storageId == "0" || storageId.equals("emulated", ignoreCase = true) -> "/storage/emulated/0"
                    else -> "/storage/$storageId"
                }
                return if (relative.isNotEmpty()) "$basePath/$relative" else basePath
            }
            if (treeUri.authority == "com.android.providers.downloads.documents") {
                val docId = runCatching { Docs.getTreeDocumentId(treeUri) }.getOrNull()
                    ?: runCatching { Docs.getDocumentId(treeUri) }.getOrNull()
                    ?: return null
                val decoded = runCatching { java.net.URLDecoder.decode(docId, "UTF-8") }.getOrDefault(docId)
                if (decoded.startsWith("raw:")) {
                    return decoded.removePrefix("raw:")
                }
                if (decoded.equals("downloads", ignoreCase = true) || decoded.equals("my_downloads", ignoreCase = true)) {
                    return "/storage/emulated/0/Download"
                }
            }
            return null
        }

        fun resolveFolderName(context: Context? = null, treeUri: Uri): String {
            val real = resolveRealPath(context, treeUri)
            if (real != null) {
                val name = real.trimEnd('/').substringAfterLast('/').trim()
                if (name.isNotEmpty() && name != "0" && name != "emulated") return name
            }
            val docId = runCatching { Docs.getTreeDocumentId(treeUri) }.getOrNull()
                ?: runCatching { Docs.getDocumentId(treeUri) }.getOrNull()
            if (docId != null) {
                val decoded = runCatching { java.net.URLDecoder.decode(docId, "UTF-8") }.getOrDefault(docId)
                val name = decoded.trimEnd('/').substringAfterLast(':').substringAfterLast('/').trim()
                if (name.isNotEmpty() && name != "primary") return name
            }
            return "project"
        }
    }

    private suspend fun ensureDirectory(client: CodexClient, path: String) {
        val clean = path.trimEnd('/')
        val parts = clean.split('/').filter { it.isNotEmpty() }
        var current = ""
        for (part in parts) {
            current += "/$part"
            if (current == "/data" || current == "/data/data" || current == "/data/data/com.termux" ||
                current == "/data/data/com.termux/files" || current == "/data/data/com.termux/files/home" ||
                current == "/storage" || current == "/storage/emulated" || current == "/storage/emulated/0") {
                continue
            }
            runCatching {
                client.request("fs/createDirectory", JSONObject().put("path", current))
            }
        }
    }

    private fun children(parent: Uri): List<Triple<String, String, Uri>> {
        val query = Docs.buildChildDocumentsUriUsingTree(tree, Docs.getDocumentId(parent))
        return buildList {
            val cursor = checkNotNull(resolver.query(query, arrayOf(Docs.Document.COLUMN_DOCUMENT_ID,
                Docs.Document.COLUMN_DISPLAY_NAME, Docs.Document.COLUMN_MIME_TYPE), null, null, null))
            cursor.use {
                while (it.moveToNext()) {
                    val name = it.getString(1)
                    require(name.isNotEmpty() && name != "." && name != ".." && '/' !in name && '\\' !in name)
                    add(Triple(name, it.getString(2), Docs.buildDocumentUriUsingTree(tree, it.getString(0))))
                }
            }
        }
    }

    private fun read(uri: Uri): ByteArray = checkNotNull(resolver.openInputStream(uri)).use {
        val out = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(8192)
        while (true) {
            val n = it.read(buffer); if (n < 0) break
            require(out.size() + n <= 4 * 1024 * 1024) { "Workspace file exceeds 4 MiB" }
            out.write(buffer, 0, n)
        }
        out.toByteArray()
    }

    private fun takeSnapshot(parent: Uri) {
        original.clear()
        total = 0
        snapshot(parent, "")
    }

    private fun snapshot(parent: Uri, prefix: String = "") {
        for ((name, mime, uri) in children(parent)) {
            if (name in ignored) continue
            val path = prefix + name
            if (mime == Docs.Document.MIME_TYPE_DIR) snapshot(uri, "$path/")
            else {
                val bytes = read(uri); total += bytes.size
                require(original.size < 2000 && total <= 32 * 1024 * 1024) { "Workspace exceeds 2000 files or 32 MiB" }
                original[path] = Source(uri, bytes)
            }
        }
    }

    suspend fun stage(client: CodexClient, editorUri: String?, editorName: String?, editorText: String, editorDirty: Boolean = false) {
        val realPath = resolveRealPath(context, tree)
        val targetPath = realPath ?: "/storage/emulated/0/Codex"
        val targetDir = java.io.File(targetPath)
        runCatching { targetDir.mkdirs() }
        isDirect = true
        remote = runCatching { targetDir.canonicalPath }.getOrDefault(targetPath)

        // A clean editor may be stale after an interrupted run. Never overwrite an agent's
        // actual edits with that snapshot on the next prompt; only publish unsaved user edits.
        if (editorDirty && !editorName.isNullOrBlank()) {
            client.request("fs/writeFile", JSONObject().put("path", "$remote/$editorName")
                .put("dataBase64", Base64.encodeToString(editorText.toByteArray(), Base64.NO_WRAP)))
        }
    }

    suspend fun publish(client: CodexClient): Map<String, Uri> {
        val dir = java.io.File(remote)
        val result = linkedMapOf<String, Uri>()
        if (dir.exists() && dir.isDirectory) {
            dir.walkTopDown().filter { it.isFile && it.name !in ignored }.forEach { file ->
                val rel = file.relativeTo(dir).path
                result[rel] = Uri.fromFile(file)
            }
        }
        val rootDocId = runCatching { Docs.getTreeDocumentId(tree) }.getOrNull()
            ?: runCatching { Docs.getDocumentId(tree) }.getOrNull()
        if (rootDocId != null) {
            runCatching {
                takeSnapshot(Docs.buildDocumentUriUsingTree(tree, rootDocId))
                for ((path, src) in original) {
                    result[path] = src.uri
                }
            }
        }
        return result
    }
}
