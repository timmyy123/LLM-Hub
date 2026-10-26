using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace LLMHub.Services;

public sealed record ChatRecord(string Id, string Title, string? ModelId, DateTimeOffset UpdatedAt, bool Starred);

public sealed record MessageRecord(string Id, string ChatId, string Role, string Content, string? Reasoning, string? ModelId, DateTimeOffset CreatedAt)
{
    public IReadOnlyList<Attachment> Attachments { get; init; } = [];
    public IReadOnlyList<WebSource> Sources { get; init; } = [];
}

/// <summary>Chat sessions and messages in a local SQLite database.</summary>
public sealed class ChatStore
{
    private readonly string _connectionString = new SqliteConnectionStringBuilder { DataSource = AppPaths.Database }.ToString();

    public ChatStore()
    {
        using var db = Open();
        Execute(db, """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model_id TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                starred INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                reasoning TEXT,
                model_id TEXT,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, created_at);
            """);
        // Columns added after the first release: attached files and web sources, as JSON.
        foreach (var column in new[] { "attachments", "sources" })
        {
            try
            {
                Execute(db, $"ALTER TABLE messages ADD COLUMN {column} TEXT");
            }
            catch (SqliteException e) when (e.Message.Contains("duplicate column", StringComparison.OrdinalIgnoreCase))
            {
            }
        }
    }

    public void DeleteMessage(string id)
    {
        using var db = Open();
        Execute(db, "DELETE FROM messages WHERE id = $id", ("$id", id));
    }

    public List<ChatRecord> ListChats()
    {
        using var db = Open();
        using var cmd = db.CreateCommand();
        cmd.CommandText = "SELECT id, title, model_id, updated_at, starred FROM chats ORDER BY updated_at DESC";
        using var reader = cmd.ExecuteReader();
        var list = new List<ChatRecord>();
        while (reader.Read())
            list.Add(new ChatRecord(reader.GetString(0), reader.GetString(1), reader.IsDBNull(2) ? null : reader.GetString(2),
                DateTimeOffset.FromUnixTimeMilliseconds(reader.GetInt64(3)), reader.GetInt64(4) != 0));
        return list;
    }

    public ChatRecord CreateChat(string title, string? modelId)
    {
        var now = DateTimeOffset.Now;
        var chat = new ChatRecord(Guid.NewGuid().ToString("N"), title, modelId, now, false);
        using var db = Open();
        Execute(db, "INSERT INTO chats (id, title, model_id, created_at, updated_at) VALUES ($id, $title, $model, $now, $now)",
            ("$id", chat.Id), ("$title", title), ("$model", modelId), ("$now", now.ToUnixTimeMilliseconds()));
        return chat;
    }

    public void RenameChat(string id, string title)
    {
        using var db = Open();
        Execute(db, "UPDATE chats SET title = $title WHERE id = $id", ("$id", id), ("$title", title));
    }

    public void SetStarred(string id, bool starred)
    {
        using var db = Open();
        Execute(db, "UPDATE chats SET starred = $s WHERE id = $id", ("$id", id), ("$s", starred ? 1 : 0));
    }

    public void DeleteChat(string id)
    {
        using var db = Open();
        Execute(db, "DELETE FROM messages WHERE chat_id = $id; DELETE FROM chats WHERE id = $id", ("$id", id));
    }

    public List<MessageRecord> GetMessages(string chatId)
    {
        using var db = Open();
        using var cmd = db.CreateCommand();
        cmd.CommandText = "SELECT id, chat_id, role, content, reasoning, model_id, created_at, attachments, sources FROM messages WHERE chat_id = $id ORDER BY created_at";
        cmd.Parameters.AddWithValue("$id", chatId);
        using var reader = cmd.ExecuteReader();
        var list = new List<MessageRecord>();
        while (reader.Read())
            list.Add(new MessageRecord(reader.GetString(0), reader.GetString(1), reader.GetString(2), reader.GetString(3),
                reader.IsDBNull(4) ? null : reader.GetString(4), reader.IsDBNull(5) ? null : reader.GetString(5),
                DateTimeOffset.FromUnixTimeMilliseconds(reader.GetInt64(6)))
            {
                Attachments = reader.IsDBNull(7) ? [] : JsonSerializer.Deserialize<List<Attachment>>(reader.GetString(7)) ?? [],
                Sources = reader.IsDBNull(8) ? [] : JsonSerializer.Deserialize<List<WebSource>>(reader.GetString(8)) ?? [],
            });
        return list;
    }

    public void AddMessage(MessageRecord m)
    {
        using var db = Open();
        Execute(db, """
            INSERT INTO messages (id, chat_id, role, content, reasoning, model_id, created_at, attachments, sources)
            VALUES ($id, $chat, $role, $content, $reasoning, $model, $at, $attachments, $sources);
            UPDATE chats SET updated_at = $at, model_id = COALESCE($model, model_id) WHERE id = $chat;
            """,
            ("$id", m.Id), ("$chat", m.ChatId), ("$role", m.Role), ("$content", m.Content), ("$reasoning", m.Reasoning),
            ("$model", m.ModelId), ("$at", m.CreatedAt.ToUnixTimeMilliseconds()),
            ("$attachments", m.Attachments.Count > 0 ? JsonSerializer.Serialize(m.Attachments) : null),
            ("$sources", m.Sources.Count > 0 ? JsonSerializer.Serialize(m.Sources) : null));
    }

    private SqliteConnection Open()
    {
        var db = new SqliteConnection(_connectionString);
        db.Open();
        return db;
    }

    private static void Execute(SqliteConnection db, string sql, params (string Name, object? Value)[] args)
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = sql;
        foreach (var (name, value) in args)
            cmd.Parameters.AddWithValue(name, value ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }
}
