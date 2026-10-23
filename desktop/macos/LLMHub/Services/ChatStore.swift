import Foundation
import SQLite3

struct ChatRecord {
    let id: String
    var title: String
    var modelId: String?
    var updatedAt: Date
}

struct MessageRecord {
    let id: String
    let chatId: String
    let role: String
    let content: String
    let reasoning: String?
    let modelId: String?
    let createdAt: Date
}

/// Chat sessions and messages in a local SQLite database (same schema as the Windows app).
@MainActor
final class ChatStore {
    private var db: OpaquePointer?
    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    init() {
        if sqlite3_open(AppPaths.database.path, &db) != SQLITE_OK { Log.error("Open database failed") }
        exec("""
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, model_id TEXT,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, starred INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL, reasoning TEXT, model_id TEXT, created_at INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, created_at);
            """)
    }

    func listChats() -> [ChatRecord] {
        query("SELECT id, title, model_id, updated_at FROM chats ORDER BY updated_at DESC", []) { s in
            ChatRecord(id: Self.text(s, 0)!, title: Self.text(s, 1)!, modelId: Self.text(s, 2), updatedAt: Self.date(s, 3))
        }
    }

    func createChat(title: String, modelId: String?) -> ChatRecord {
        let chat = ChatRecord(id: UUID().uuidString.lowercased(), title: title, modelId: modelId, updatedAt: Date())
        run("INSERT INTO chats (id, title, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            [chat.id, title, modelId, Self.millis(chat.updatedAt), Self.millis(chat.updatedAt)])
        return chat
    }

    func deleteChat(_ id: String) {
        run("DELETE FROM messages WHERE chat_id = ?", [id])
        run("DELETE FROM chats WHERE id = ?", [id])
    }

    func messages(_ chatId: String) -> [MessageRecord] {
        query("SELECT id, chat_id, role, content, reasoning, model_id, created_at FROM messages WHERE chat_id = ? ORDER BY created_at", [chatId]) { s in
            MessageRecord(id: Self.text(s, 0)!, chatId: Self.text(s, 1)!, role: Self.text(s, 2)!, content: Self.text(s, 3) ?? "",
                          reasoning: Self.text(s, 4), modelId: Self.text(s, 5), createdAt: Self.date(s, 6))
        }
    }

    func add(_ m: MessageRecord) {
        run("INSERT INTO messages (id, chat_id, role, content, reasoning, model_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [m.id, m.chatId, m.role, m.content, m.reasoning, m.modelId, Self.millis(m.createdAt)])
        run("UPDATE chats SET updated_at = ?, model_id = COALESCE(?, model_id) WHERE id = ?", [Self.millis(m.createdAt), m.modelId, m.chatId])
    }

    // MARK: SQLite helpers

    private func exec(_ sql: String) {
        if sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK { Log.error("SQL failed: \(String(cString: sqlite3_errmsg(db)))") }
    }

    private func prepare(_ sql: String, _ args: [Any?]) -> OpaquePointer? {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            Log.error("SQL prepare failed: \(String(cString: sqlite3_errmsg(db)))")
            return nil
        }
        for (i, arg) in args.enumerated() {
            let index = Int32(i + 1)
            switch arg {
            case let v as String: sqlite3_bind_text(statement, index, v, -1, Self.transient)
            case let v as Int64: sqlite3_bind_int64(statement, index, v)
            default: sqlite3_bind_null(statement, index)
            }
        }
        return statement
    }

    private func run(_ sql: String, _ args: [Any?]) {
        guard let statement = prepare(sql, args) else { return }
        defer { sqlite3_finalize(statement) }
        if sqlite3_step(statement) != SQLITE_DONE { Log.error("SQL step failed: \(String(cString: sqlite3_errmsg(db)))") }
    }

    private func query<T>(_ sql: String, _ args: [Any?], _ row: (OpaquePointer) -> T) -> [T] {
        guard let statement = prepare(sql, args) else { return [] }
        defer { sqlite3_finalize(statement) }
        var rows: [T] = []
        while sqlite3_step(statement) == SQLITE_ROW { rows.append(row(statement)) }
        return rows
    }

    private static func text(_ s: OpaquePointer, _ i: Int32) -> String? {
        sqlite3_column_text(s, i).map { String(cString: $0) }
    }

    private static func date(_ s: OpaquePointer, _ i: Int32) -> Date {
        Date(timeIntervalSince1970: Double(sqlite3_column_int64(s, i)) / 1000)
    }

    private static func millis(_ d: Date) -> Int64 { Int64(d.timeIntervalSince1970 * 1000) }
}
