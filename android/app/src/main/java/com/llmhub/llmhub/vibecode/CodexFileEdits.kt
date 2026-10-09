package com.llmhub.llmhub.vibecode

import org.json.JSONObject

/** Transport literal edits without shell escaping, accepting unique indentation-only matches. */
internal object CodexFileEdits {
    fun writtenFile(command: String): Pair<String, String>? {
        val match = Regex("(?:mkdir -p -- '[^']*' && )?printf '%s' '([A-Za-z0-9+/=]*)' \\| base64 -d > ('(?:[^']|'\"'\"')*')").matchEntire(command) ?: return null
        val path = match.groupValues[2].removeSurrounding("'").replace("'\"'\"'", "'")
        return runCatching { path to String(java.util.Base64.getDecoder().decode(match.groupValues[1]), Charsets.UTF_8) }.getOrNull()
    }
    fun commandSummary(command: String): String? {
        if (!command.startsWith("node -e ") || !command.contains("const p=JSON.parse(Buffer.from(process.argv[1]")) return null
        val encoded = Regex("'([A-Za-z0-9+/=]+)'$").find(command)?.groupValues?.get(1) ?: return null
        return runCatching {
            val payload = JSONObject(String(java.util.Base64.getDecoder().decode(encoded), Charsets.UTF_8))
            "replace_in_file(" + JSONObject().put("path", payload.getString("path"))
                .put("old_text_bytes", payload.getString("old").toByteArray(Charsets.UTF_8).size)
                .put("replacement_bytes", payload.getString("replacement").toByteArray(Charsets.UTF_8).size) + ")"
        }.getOrNull()
    }
    fun replaceCommand(path: String, old: String, replacement: String): String {
        require(path.isNotBlank() && old.isNotEmpty()) { "replace_in_file requires a path and nonempty old_text" }
        val payload = JSONObject().put("path", path).put("old", old).put("replacement", replacement)
        val encoded = java.util.Base64.getEncoder().encodeToString(payload.toString().toByteArray(Charsets.UTF_8))
        val script = """
            const fs=require('fs'),path=require('path');
            const p=JSON.parse(Buffer.from(process.argv[1],'base64').toString('utf8'));
            const s=fs.readFileSync(p.path,'utf8');let i=s.indexOf(p.old),length=p.old.length;
            if(i>=0&&s.indexOf(p.old,i+length)>=0)throw Error('Text matches more than once; use a larger exact fragment');
            if(i<0&&/\.(html?|css|[jt]sx?|kts?|java|c|cpp|h|cs|swift|go|rs)$/i.test(path.extname(p.path))){
                const escape=x=>x.replace(/[.*+?^${'$'}{}()|[\]\\]/g,'\\${'$'}&');
                const lines=p.old.split(/\r?\n/).map(x=>x.replace(/^[ \t]+/,''));
                const pattern=lines.map(escape).join('\\r?\\n[ \\t]*');
                const matches=[...s.matchAll(new RegExp(pattern,'g'))];
                if(matches.length>1)throw Error('Indentation-only text matches more than once; use a larger exact fragment');
                if(matches.length===1){i=matches[0].index;length=matches[0][0].length;}
            }
            if(i<0)throw Error('Exact text not found; no file was changed. Use a smaller fragment from the returned file or write_file with complete content.');
            const updated=s.slice(0,i)+p.replacement+s.slice(i+length);
            if(updated===s)throw Error('Replacement makes no change; file was not modified');
            fs.writeFileSync(p.path,updated,'utf8');console.log('Updated '+p.path);
        """.trimIndent()
        return "node -e ${CodexConfig.shellQuote(script)} ${CodexConfig.shellQuote(encoded)}"
    }
}
