package com.llmhub.llmhub.vibecode

import org.json.JSONObject

/** Vibecode runs unattended, as requested by the user. */
internal object CodexApprovals {
    const val POLICY = "never"

    fun response(method: String, params: JSONObject): JSONObject? = when (method) {
        "item/commandExecution/requestApproval", "item/fileChange/requestApproval" ->
            JSONObject().put("decision", "accept")
        "item/permissions/requestApproval" -> JSONObject()
            .put("permissions", params.optJSONObject("permissions") ?: JSONObject())
            .put("scope", "session")
        "execCommandApproval", "applyPatchApproval" -> JSONObject().put("decision", "approved")
        else -> null
    }
}
