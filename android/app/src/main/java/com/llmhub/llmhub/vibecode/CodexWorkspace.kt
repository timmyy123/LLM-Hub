package com.llmhub.llmhub.vibecode

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract as Docs
import android.util.Base64
import org.json.JSONObject

/** A private Termux working copy. Check every source file before publishing any edits. */
internal class CodexWorkspace(private val context: Context, private val tree: Uri, val remote: String) {
    private val resolver = context.contentResolver
    private data class Source(val uri: Uri, val bytes: ByteArray)
    private val original = linkedMapOf<String, Source>()
    private val ignored = setOf(".git", "node_modules", ".gradle", ".build", "build", "dist", "__pycache__", ".venv")
    private var total = 0

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

    suspend fun stage(client: CodexClient, editorUri: String?, editorName: String?, editorText: String) {
        snapshot(Docs.buildDocumentUriUsingTree(tree, Docs.getTreeDocumentId(tree)))
        client.request("fs/createDirectory", JSONObject().put("path", remote))
        for ((path, source) in original) {
            client.request("fs/createDirectory", JSONObject().put("path", "$remote/${path.substringBeforeLast('/', "")}"))
            val bytes = if (source.uri.toString() == editorUri) editorText.toByteArray() else source.bytes
            client.request("fs/writeFile", JSONObject().put("path", "$remote/$path")
                .put("dataBase64", Base64.encodeToString(bytes, Base64.NO_WRAP)))
        }
        if (editorUri == null && !editorName.isNullOrBlank()) {
            require('/' !in editorName && '\\' !in editorName && editorName != "." && editorName != "..")
            client.request("fs/writeFile", JSONObject().put("path", "$remote/$editorName")
                .put("dataBase64", Base64.encodeToString(editorText.toByteArray(), Base64.NO_WRAP)))
        }
    }

    suspend fun publish(client: CodexClient): Map<String, Uri> {
        val result = linkedMapOf<String, ByteArray>()
        var bytesRead = 0
        suspend fun scan(path: String, prefix: String) {
            val entries = client.request("fs/readDirectory", JSONObject().put("path", path)).getJSONArray("entries")
            for (i in 0 until entries.length()) {
                val entry = entries.getJSONObject(i)
                val name = entry.getString("fileName")
                require(name.isNotEmpty() && name != "." && name != ".." && '/' !in name && '\\' !in name)
                if (name in ignored) continue
                val absolute = "$path/$name"
                val meta = client.request("fs/getMetadata", JSONObject().put("path", absolute))
                require(!meta.getBoolean("isSymlink")) { "Workspace symbolic links are unsupported" }
                if (meta.getBoolean("isDirectory")) scan(absolute, "$prefix$name/")
                else if (meta.getBoolean("isFile")) {
                    val encoded = client.request("fs/readFile", JSONObject().put("path", absolute)).getString("dataBase64")
                    require(encoded.length <= 5_592_408)
                    val bytes = Base64.decode(encoded, Base64.DEFAULT); bytesRead += bytes.size
                    require(result.size < 2000 && bytesRead <= 32 * 1024 * 1024)
                    result[prefix + name] = bytes
                }
            }
        }
        scan(remote, "")
        // Re-scan to detect source additions as well as modifications and deletions.
        val current = CodexWorkspace(context, tree, remote)
        current.snapshot(Docs.buildDocumentUriUsingTree(tree, Docs.getTreeDocumentId(tree)))
        require(original.keys == current.original.keys && original.all { (path, src) ->
            src.bytes.contentEquals(current.original.getValue(path).bytes)
        }) { "workspace_conflict" }
        val uris = linkedMapOf<String, Uri>()
        val root = Docs.buildDocumentUriUsingTree(tree, Docs.getTreeDocumentId(tree))
        for ((path, bytes) in result) {
            val source = original[path]
            var uri = source?.uri
            if (uri == null) {
                var parent = root
                val parts = path.split('/')
                for (dir in parts.dropLast(1)) {
                    parent = children(parent).firstOrNull { it.first == dir && it.second == Docs.Document.MIME_TYPE_DIR }?.third
                        ?: checkNotNull(Docs.createDocument(resolver, parent, Docs.Document.MIME_TYPE_DIR, dir))
                }
                uri = checkNotNull(Docs.createDocument(resolver, parent, "application/octet-stream", parts.last()))
            }
            if (source == null || !source.bytes.contentEquals(bytes)) {
                checkNotNull(resolver.openOutputStream(uri, "wt")).use { it.write(bytes) }
            }
            uris[path] = uri
        }
        for ((path, source) in original) if (path !in result) check(Docs.deleteDocument(resolver, source.uri))
        return uris
    }
}
