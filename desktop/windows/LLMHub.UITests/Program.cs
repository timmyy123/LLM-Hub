using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.Core.Capturing;
using FlaUI.Core.Definitions;
using FlaUI.Core.Input;
using FlaUI.Core.WindowsAPI;
using FlaUI.UIA3;

// End-to-end UI tests for LLM Hub on Windows. Drives the real app through UI Automation.
// Usage: LLMHub.UITests.exe <path to LLMHub.exe> <output folder> [test name filter]
// Needs: Gemma 4 12B installed (the --selftest run does that) and an interactive desktop session.

var exe = args[0];
var outDir = args[1];
var filter = args.Length > 2 ? args[2] : null;
Directory.CreateDirectory(outDir);

var dataDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LLMHub");
var settingsPath = Path.Combine(dataDir, "settings.json");
var results = new List<(string Name, bool Pass, string Detail)>();
var automation = new UIA3Automation();
Application? app = null;
Window? window = null;

void Log(string line)
{
    Console.WriteLine(line);
    File.AppendAllText(Path.Combine(outDir, "results.txt"), line + Environment.NewLine);
}

void Shot(string name)
{
    try
    {
        if (window is not null)
            Capture.Element(window).ToFile(Path.Combine(outDir, $"{name}.png"));
    }
    catch (Exception e)
    {
        Log($"   (screenshot {name} failed: {e.Message})");
    }
}

void Test(string name, Action body)
{
    if (filter is not null && !name.Contains(filter, StringComparison.OrdinalIgnoreCase))
        return;
    var clock = Stopwatch.StartNew();
    try
    {
        body();
        results.Add((name, true, ""));
        Log($"PASS {name} ({clock.Elapsed.TotalSeconds:F1}s)");
    }
    catch (Exception e)
    {
        results.Add((name, false, e.Message));
        Log($"FAIL {name} ({clock.Elapsed.TotalSeconds:F1}s): {e.Message}");
        Shot("FAIL-" + string.Concat(name.Where(char.IsLetterOrDigit)));
    }
}

void Check(bool condition, string message)
{
    if (!condition)
        throw new Exception(message);
}

T WaitFor<T>(Func<T?> probe, TimeSpan timeout, string what) where T : class
{
    var deadline = DateTime.UtcNow + timeout;
    while (DateTime.UtcNow < deadline)
    {
        try
        {
            if (probe() is { } value)
                return value;
        }
        catch
        {
            // Element tree changing under us; retry.
        }
        Thread.Sleep(250);
    }
    throw new TimeoutException($"Timed out after {timeout.TotalSeconds:F0}s waiting for {what}");
}

void WaitUntil(Func<bool> condition, TimeSpan timeout, string what) =>
    WaitFor(() => condition() ? "ok" : null, timeout, what);

AutomationElement? Find(string id) => window!.FindFirstDescendant(cf => cf.ByAutomationId(id));
AutomationElement[] FindAll(string id) => window!.FindAllDescendants(cf => cf.ByAutomationId(id));
AutomationElement Get(string id, int seconds = 15) => WaitFor(() => Find(id), TimeSpan.FromSeconds(seconds), id);

bool Visible(string id)
{
    var e = Find(id);
    return e is not null && !e.Properties.IsOffscreen.ValueOrDefault;
}

void Invoke(string id) => Get(id).AsButton().Invoke();

void Select(string id)
{
    var e = Get(id);
    if (e.Patterns.SelectionItem.IsSupported)
        e.Patterns.SelectionItem.Pattern.Select();
    else
        e.Click();
    Thread.Sleep(400);
}

/// Keyboard input goes to whatever window is in front, so put LLM Hub there first.
void Front()
{
    window!.SetForeground();
    Thread.Sleep(150);
}

void SetText(string id, string text)
{
    Front();
    var box = Get(id).AsTextBox();
    box.Focus();
    box.Text = text;
    Thread.Sleep(150);
}

void Launch(string extraArgs = "", Dictionary<string, string>? env = null)
{
    Close();
    var info = new ProcessStartInfo(exe, extraArgs) { UseShellExecute = false };
    foreach (var (k, v) in env ?? [])
        info.Environment[k] = v;
    app = Application.Launch(info);
    window = WaitFor(() => app.GetMainWindow(automation, TimeSpan.FromSeconds(2)), TimeSpan.FromSeconds(40), "main window");
    window.SetForeground();
    Get("NewChatButton", 30);
}

void Close()
{
    if (app is null)
        return;
    try
    {
        app.Close();
        app.WaitWhileMainHandleIsMissing(TimeSpan.FromSeconds(1));
        if (!app.HasExited)
            Thread.Sleep(2000);
        if (!app.HasExited)
            app.Kill();
    }
    catch
    {
    }
    app.Dispose();
    app = null;
    window = null;
    Thread.Sleep(1500);
}

JsonObject ReadSettings() => File.Exists(settingsPath) ? JsonNode.Parse(File.ReadAllText(settingsPath))!.AsObject() : new JsonObject();

void WriteSetting(string key, JsonNode? value)
{
    var s = ReadSettings();
    s[key] = value;
    Directory.CreateDirectory(Path.GetDirectoryName(settingsPath)!); // a fresh PC has no LLMHub folder before the first launch
    File.WriteAllText(settingsPath, s.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
}

/// Generation is over once Send is back and Stop is gone for several consecutive checks
/// (single reads can miss elements while the tree is being rebuilt during streaming).
void WaitIdle(int timeoutSeconds)
{
    var stable = 0;
    WaitUntil(() =>
    {
        stable = Visible("SendButton") && !Visible("StopButton") ? stable + 1 : 0;
        return stable >= 6;
    }, TimeSpan.FromSeconds(timeoutSeconds), "generation to finish");
}

string AssistantText()
{
    var all = FindAll("AssistantMessage");
    return all.Length == 0 ? "" : SafeName(all[^1]);
}

/// <summary>An element's name, or "" if it went away while being read (a reply re-rendering).</summary>
string SafeName(AutomationElement e)
{
    try
    {
        return e.Name;
    }
    catch (Exception ex) when (ex is FlaUI.Core.Exceptions.PropertyNotSupportedException or System.Runtime.InteropServices.COMException)
    {
        return "";
    }
}

/// <summary>Types a message, sends it and waits for the reply to finish. Returns the reply text.</summary>
string Send(string text, int timeoutSeconds = 240, bool useEnter = false)
{
    var before = FindAll("AssistantMessage").Length;
    SetText("Composer", text);
    if (useEnter)
    {
        Front();
        Get("Composer").Focus();
        Keyboard.Press(VirtualKeyShort.ENTER);
        Keyboard.Release(VirtualKeyShort.ENTER);
    }
    else
        Invoke("SendButton");
    WaitUntil(() => FindAll("AssistantMessage").Length > before, TimeSpan.FromSeconds(20), "reply bubble");
    // A loaded model can answer a short prompt before the Stop button is ever seen.
    WaitUntil(() => Visible("StopButton") || AssistantText().Length > 0, TimeSpan.FromSeconds(30), "generation to start");
    WaitIdle(timeoutSeconds);
    var status = FindAll("MessageStatus").Select(SafeName).LastOrDefault(n => !string.IsNullOrEmpty(n));
    var reply = AssistantText();
    Check(reply.Length > 0, $"empty reply (status: {status})");
    return reply;
}

void PickModel(string namePart)
{
    var combo = Get("ModelPicker").AsComboBox();
    combo.Expand();
    Thread.Sleep(400);
    var item = combo.Items.FirstOrDefault(i => i.Text.Contains(namePart, StringComparison.OrdinalIgnoreCase))
               ?? throw new Exception($"model '{namePart}' not in picker: {string.Join(", ", combo.Items.Select(i => i.Text))}");
    item.Select();
    combo.Collapse();
    Thread.Sleep(300);
}

/// Settings replace the app navigation in the single sidebar; Back returns to it.
void GoToChat()
{
    if (Visible("BackButton"))
        Invoke("BackButton");
    Select("NavChat");
}

void OpenSettings(string section)
{
    Invoke("SettingsButton");
    Select(section);
}

AutomationElement FindCatalogRow(string modelName)
{
    var list = Get("CatalogList", 20);
    for (var attempt = 0; attempt < 40; attempt++)
    {
        var name = list.FindAllDescendants(cf => cf.ByAutomationId("CatalogName")).FirstOrDefault(e => e.Name == modelName);
        if (name is not null)
        {
            var row = name;
            while (row.Parent is { } parent && row.ControlType != ControlType.ListItem)
                row = parent;
            if (row.Patterns.ScrollItem.IsSupported)
                row.Patterns.ScrollItem.Pattern.ScrollIntoView();
            return row;
        }
        if (list.Patterns.Scroll.IsSupported && list.Patterns.Scroll.Pattern.VerticallyScrollable)
            list.Patterns.Scroll.Pattern.Scroll(ScrollAmount.NoAmount, ScrollAmount.LargeIncrement);
        Thread.Sleep(300);
    }
    throw new Exception($"catalog row '{modelName}' not found");
}

double Progress(AutomationElement row) =>
    row.FindFirstDescendant(cf => cf.ByAutomationId("DownloadProgress"))?.Patterns.RangeValue.Pattern.Value.Value ?? -1;

int LlamaServers() => Process.GetProcessesByName("llama-server").Length;

// A word unique to this run, so chats left by earlier runs never match the searches below.
var run = "kiwi" + DateTime.Now.ToString("HHmmss");

File.Delete(Path.Combine(outDir, "results.txt"));
Log($"LLM Hub UI tests {DateTime.Now:yyyy-MM-dd HH:mm}");
WriteSetting("Language", "");
WriteSetting("HuggingFaceToken", "");

// ---------------------------------------------------------------- Shell and navigation
Launch();
Shot("01-launch");

Test("Window opens with sidebar navigation", () =>
{
    foreach (var id in new[] { "NavChat", "NavAgent", "NavDesign", "NavStudio", "NewChatButton", "ChatSearch", "SettingsButton" })
        Check(Find(id) is not null, $"{id} missing");
});

Test("Agent, Design and Studio pages open", () =>
{
    Select("NavAgent");
    Check(Find("AgentEmptyTitle") is not null || Find("AgentComposer") is not null, "Agent page did not open");
    Shot("02-NavAgent");
    Select("NavDesign");
    Get("KindPrototype");
    Get("DesignComposer");
    Shot("02-NavDesign");
    Select("NavStudio");
    Get("TabImage");
    Get("WorkflowPicker");
    Shot("02-NavStudio");
    Get("TabSearch").AsToggleButton().Toggle();
    Get("SearchQueryBox");
    Get("SearchIndexButton");
    Check(Find("SearchFolder")?.Name.Length > 0, "search folder shown");
    Shot("02-StudioSearch");
    Get("TabImage").AsToggleButton().Toggle();
    Select("NavChat");
    Get("Composer");
});

Test("Studio Music says live music is not on Windows yet", () =>
{
    Select("NavStudio");
    Get("TabMusic").AsToggleButton().Toggle();
    var note = Get("MusicUnavailable").Name;
    Check(note.Contains("Windows") && note.Contains("Stable Audio 3"), $"music page note: '{note}'");
    Shot("02-StudioMusic");
    Get("TabImage").AsToggleButton().Toggle();
    OpenSettings("SectionStudio");
    var line = Get("MusicLine", 30).Name;
    Check(line == note, $"Settings › Studio music line: '{line}'");
    Shot("02-SettingsStudioMusic");
    Invoke("BackButton");
    Select("NavChat");
    Get("Composer");
});

Test("Design and Studio collapse the sidebar to the icon rail", () =>
{
    Select("NavDesign");
    WaitUntil(() => Get("NavDesign").BoundingRectangle.Width < 90, TimeSpan.FromSeconds(5), "rail width");
    Check(!Visible("NewChatButton"), "New button still shown on the rail");
    Check(!Visible("ChatSearch"), "chat search still shown on the rail");
    Select("NavChat");
    WaitUntil(() => Get("NavChat").BoundingRectangle.Width > 150, TimeSpan.FromSeconds(5), "sidebar width back");
    Check(Visible("NewChatButton"), "New chat button missing after leaving the rail");
});

Test("Engine status card shows the GPU", () =>
{
    var settings = Get("SettingsButton");
    Check(settings.FindAllDescendants().Any(e => e.Name.Contains("4090") || e.Name.Contains("GPU") || e.Name.Contains("RAM")), "no hardware line");
});

Test("Settings use the same single sidebar", () =>
{
    Invoke("SettingsButton");
    Get("BackButton");
    Check(!Visible("NavAgent"), "app navigation still shown next to settings");
    foreach (var id in new[] { "SectionGeneral", "SectionModels", "SectionEngines", "SectionApi", "SectionSpeech", "SectionStudio", "SectionAgent", "SectionWebSearch", "SectionStorage" })
        Check(Visible(id), $"{id} missing");
    Check(!Visible("NewChatButton"), "New chat button shown in settings");
    Shot("02-settings-sidebar");
    Invoke("BackButton");
    WaitUntil(() => Visible("NavAgent") && !Visible("BackButton"), TimeSpan.FromSeconds(5), "app navigation back");
});

// ---------------------------------------------------------------- Chat
Test("Model picker lists installed models", () =>
{
    PickModel("Gemma 4 12B");
    Check(Get("ModelPicker").AsComboBox().SelectedItem?.Text.Contains("Gemma 4 12B") == true, "Gemma 4 12B not selected");
});

Test("Send with Enter, reply streams and finishes", () =>
{
    Invoke("NewChatButton");
    var reply = Send($"{run}: reply with exactly one word: pineapple", useEnter: true);
    Check(reply.Contains("pineapple", StringComparison.OrdinalIgnoreCase), $"unexpected reply: {reply}");
    Check(Get("EngineStatus").Name.Contains("Gemma"), $"engine status: {Get("EngineStatus").Name}");
    Shot("03-chat-reply");
});

Test("New chat appears in sidebar under Today", () =>
{
    var titles = WaitFor(() => FindAll("ChatTitle").Select(e => e.Name).ToList() is { Count: > 0 } t ? t : null, TimeSpan.FromSeconds(10), "chat titles");
    Check(titles.Any(t => t.Contains(run)), $"titles: {string.Join(" | ", titles)}");
    Check(Get("ChatHeaderTitle").Name.Contains(run), "header title not set");
});

Test("Multi-turn conversation keeps context", () =>
{
    Send("Remember this: my favourite colour is teal. Just answer OK.");
    var reply = Send("What is my favourite colour? Answer with one word.");
    Check(reply.Contains("teal", StringComparison.OrdinalIgnoreCase), $"reply: {reply}");
});

Test("Stop button ends generation early", () =>
{
    Invoke("NewChatButton");
    SetText("Composer", $"{run} count from 1 to 400, one number per line, no other text.");
    Invoke("SendButton");
    WaitUntil(() => Visible("StopButton"), TimeSpan.FromSeconds(30), "generation start");
    WaitUntil(() => AssistantText().Length > 20, TimeSpan.FromSeconds(60), "some output");
    Invoke("StopButton");
    WaitIdle(15);
    var text = AssistantText();
    Check(!text.Contains("400"), "reached 400, stop did not cut it short");
    Thread.Sleep(2000);
    Check(AssistantText() == text, "text kept growing after Stop");
});

Test("Search filters the chat list", () =>
{
    // This run made two chats: the multi-turn one and the stopped "count" one.
    SetText("ChatSearch", run);
    WaitUntil(() => FindAll("ChatTitle") is { Length: 2 } t && t.All(e => e.Name.Contains(run)), TimeSpan.FromSeconds(5), "this run's two chats");
    SetText("ChatSearch", "zzzz-no-such-chat");
    WaitUntil(() => FindAll("ChatTitle").Length == 0, TimeSpan.FromSeconds(5), "no matches");
    SetText("ChatSearch", "");
    WaitUntil(() => FindAll("ChatTitle").Length >= 2, TimeSpan.FromSeconds(5), "all chats back");
});

Test("Opening an old chat loads its messages", () =>
{
    // Also guards a regression: rebuilding the sidebar after a reply used to detach the open chat, so each follow-up
    // was saved as a new chat and this one held a single message.
    var item = FindAll("ChatTitle").First(e => e.Name.StartsWith($"{run}:"));
    var row = item;
    while (row.Parent is { } p && row.ControlType != ControlType.ListItem)
        row = p;
    row.Patterns.SelectionItem.Pattern.Select();
    WaitUntil(() => FindAll("UserMessage").Length >= 3, TimeSpan.FromSeconds(10), "3 user messages");
    Check(FindAll("AssistantMessage").Any(e => e.Name.Contains("teal", StringComparison.OrdinalIgnoreCase)), "teal reply missing");
});

Test("New chat button clears the conversation", () =>
{
    Invoke("NewChatButton");
    WaitUntil(() => FindAll("UserMessage").Length == 0, TimeSpan.FromSeconds(5), "empty conversation");
    Check(Find("EmptyTitle") is not null, "empty state missing");
});

Test("Chats persist after restart", () =>
{
    Close();
    WaitUntil(() => LlamaServers() == 0, TimeSpan.FromSeconds(20), "llama-server to exit");
    Launch();
    WaitUntil(() => FindAll("ChatTitle").Any(e => e.Name.Contains(run)), TimeSpan.FromSeconds(10), "saved chat");
});

Test("Delete chat from the context menu", () =>
{
    bool Present() => FindAll("ChatTitle").Any(e => e.Name.StartsWith($"{run} count", StringComparison.OrdinalIgnoreCase));
    var item = FindAll("ChatTitle").First(e => e.Name.StartsWith($"{run} count", StringComparison.OrdinalIgnoreCase));
    Front();
    item.RightClick();
    var menu = WaitFor(() => window!.FindFirstDescendant(cf => cf.ByAutomationId("DeleteChatMenuItem"))
                             ?? automation.GetDesktop().FindFirstDescendant(cf => cf.ByAutomationId("DeleteChatMenuItem")), TimeSpan.FromSeconds(5), "delete menu");
    menu.AsMenuItem().Invoke();
    WaitUntil(() => !Present(), TimeSpan.FromSeconds(5), "chat removed");
    Close();
    Launch();
    Check(!FindAll("ChatTitle").Any(e => e.Name.StartsWith($"{run} count", StringComparison.OrdinalIgnoreCase)), "deleted chat came back after restart");
});

// ---------------------------------------------------------------- Models
Test("Catalog loads quants from Hugging Face, only ones that fit", () =>
{
    OpenSettings("SectionModels");
    Select("TabCatalog");
    var row = FindCatalogRow("Gemma 4 12B");
    WaitUntil(() => row.FindFirstDescendant(cf => cf.ByAutomationId("VariantPicker")) is not null, TimeSpan.FromSeconds(60), "variants");
    var picker = row.FindFirstDescendant(cf => cf.ByAutomationId("VariantPicker"))!.AsComboBox();
    picker.Expand();
    Thread.Sleep(500);
    var items = picker.Items.Select(i => i.Text + " " + string.Join(" ", i.FindAllDescendants().Select(d => d.Name))).ToList();
    picker.Collapse();
    Check(items.Count > 0, "no variants");
    Check(items.All(i => !i.Contains("Too large")), $"too-large variant offered: {string.Join(" | ", items)}");
    Log($"   Gemma 4 12B variants: {string.Join(" | ", items)}");
    Shot("04-catalog");
});

Test("Filter chips narrow the catalog", () =>
{
    var list = Get("CatalogList", 20);
    Get("Filter_embedding").AsToggleButton().Toggle();
    WaitUntil(() => list.FindAllDescendants(cf => cf.ByAutomationId("CatalogName")).Select(e => e.Name).All(n => n.StartsWith("EmbeddingGemma 2")),
        TimeSpan.FromSeconds(10), "Embedding chip shows only EmbeddingGemma");
    Get("Filter_vision").AsToggleButton().Toggle();
    WaitUntil(() => list.FindAllDescendants(cf => cf.ByAutomationId("CatalogName")).Any(e => e.Name == "Gemma 4 12B")
                    && list.FindAllDescendants(cf => cf.ByAutomationId("CatalogName")).All(e => e.Name != "Llama 3.2 3B"), TimeSpan.FromSeconds(10), "Vision chip");
    Shot("04-chips");
    Get("Filter_all").AsToggleButton().Toggle();
    WaitUntil(() => list.FindAllDescendants(cf => cf.ByAutomationId("CatalogName")).Length > 3, TimeSpan.FromSeconds(10), "All chip");
    Check(Find("LlamaEngineLine")?.Name.StartsWith("GGUF") == true, "llama.cpp engine card");
});

Test("Download, pause and resume a catalog model", () =>
{
    var row = FindCatalogRow("LFM2.5 1.2B Thinking");
    WaitFor(() => row.FindFirstDescendant(cf => cf.ByAutomationId("DownloadButton")) ?? row.FindFirstDescendant(cf => cf.ByAutomationId("DeleteModelButton")), TimeSpan.FromSeconds(60), "row actions");
    if (row.FindFirstDescendant(cf => cf.ByAutomationId("DeleteModelButton")) is { } delete)
        delete.AsButton().Invoke(); // Installed by an earlier run; start clean.
    WaitFor(() => row.FindFirstDescendant(cf => cf.ByAutomationId("DownloadButton")), TimeSpan.FromSeconds(30), "Download button").AsButton().Invoke();
    WaitUntil(() => Progress(row) > 15, TimeSpan.FromSeconds(120), "15% progress");
    var paused = Progress(row);
    row.FindFirstDescendant(cf => cf.ByAutomationId("CancelDownloadButton"))!.AsButton().Invoke();
    WaitFor(() => row.FindFirstDescendant(cf => cf.ByAutomationId("DownloadButton")), TimeSpan.FromSeconds(15), "Download button after pause").AsButton().Invoke();
    WaitUntil(() => Progress(row) > 0, TimeSpan.FromSeconds(30), "progress after resume");
    var resumed = Progress(row);
    Check(resumed >= paused - 1, $"restarted from {resumed:F1}% instead of resuming from {paused:F1}%");
    Log($"   paused at {paused:F1}%, resumed at {resumed:F1}%");
    WaitFor(() => row.FindFirstDescendant(cf => cf.ByAutomationId("InstalledLabel")), TimeSpan.FromMinutes(10), "Installed");
});

Test("Gated model shows a clear token message", () =>
{
    Select("TabHuggingFace");
    SetText("HfQuery", "google/gemma-3-1b-it-qat-q4_0-gguf");
    Invoke("HfSearchButton");
    // Results mix GGUF and LiteRT-LM repos by trend: pick the gated Google one by name.
    WaitFor(() => window!.FindAllDescendants(cf => cf.ByAutomationId("HfRepoId")).FirstOrDefault(e => e.Name == "google/gemma-3-1b-it-qat-q4_0-gguf"),
        TimeSpan.FromSeconds(60), "google/gemma-3-1b-it-qat-q4_0-gguf in results").Click();
    WaitFor(() => window!.FindAllDescendants(cf => cf.ByAutomationId("HfFileDownload")).FirstOrDefault(b => b.IsEnabled && !b.IsOffscreen), TimeSpan.FromSeconds(60), "Download in the file table")
        .AsButton().Invoke();
    Check(Find("HfGatedBar") is { IsOffscreen: false }, "gated notice with sign-in missing");
    WaitUntil(() => Find("RepoStatus")?.Name.Contains("gated", StringComparison.OrdinalIgnoreCase) == true, TimeSpan.FromSeconds(60), "gated message");
    Shot("05-gated");
    Select("TabCatalog");
});

Test("Hugging Face search finds repos and lists fitting quants", () =>
{
    if (Find("TabHuggingFace") is null)
        OpenSettings("SectionModels");
    Select("TabHuggingFace");
    SetText("HfQuery", "LFM2.5-1.2B-Instruct-GGUF");
    Invoke("HfSearchButton");
    // The tab opens on trending repos; wait for this search's own result and open it.
    WaitFor(() => window!.FindAllDescendants(cf => cf.ByAutomationId("HfRepoId")).FirstOrDefault(e => e.Name == "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"),
        TimeSpan.FromSeconds(60), "LiquidAI/LFM2.5-1.2B-Instruct-GGUF in results").Click();
    WaitUntil(() => Find("RepoDetail")?.Name.Contains("LFM2.5") == true, TimeSpan.FromSeconds(30), "LFM2.5 repo detail");
    WaitFor(() => Find("HfFileFit"), TimeSpan.FromSeconds(30), "file table with fit");
    var fits = window!.FindAllDescendants(cf => cf.ByAutomationId("HfFileFit")).Select(e => e.Name).ToList();
    Check(fits.Contains("Best fit"), $"no Best fit row: {string.Join(" | ", fits)}");
    Shot("06-huggingface");
});

Test("Download from the Hugging Face tab", () =>
{
    // Only the small LFM2.5 repo: never start a download on whatever else is showing (a trending repo can be 10 GB).
    Check(Find("RepoDetail")?.Name.Contains("LFM2.5") == true, "LFM2.5 repo is not the one shown");
    // The first on-screen Download (the file table recycles rows; recycled ones stay in the tree, offscreen).
    var download = WaitFor(() => window!.FindAllDescendants(cf => cf.ByAutomationId("HfFileDownload")).FirstOrDefault(e => e.IsEnabled && !e.IsOffscreen),
        TimeSpan.FromSeconds(30), "Download in the file table");
    download.AsButton().Invoke();
    WaitFor(() => Find("HfFileInstalled"), TimeSpan.FromMinutes(10), "Installed in the file table");
});

Test("Installed tab lists models and deletes one", () =>
{
    Select("TabInstalled");
    var list = Get("InstalledList");
    WaitUntil(() => list.FindAllDescendants(cf => cf.ByAutomationId("InstalledName")).Length >= 3, TimeSpan.FromSeconds(10), "3 installed models");
    var names = list.FindAllDescendants(cf => cf.ByAutomationId("InstalledName")).Select(e => e.Name).ToList();
    Log($"   installed: {string.Join(" | ", names)}");
    Shot("07-installed");
    var target = list.FindAllDescendants(cf => cf.ByAutomationId("InstalledName")).First(e => e.Name.StartsWith("LFM2.5-1.2B-Instruct", StringComparison.OrdinalIgnoreCase));
    var row = target;
    while (row.Parent is { } p && row.ControlType != ControlType.ListItem)
        row = p;
    row.FindFirstDescendant(cf => cf.ByAutomationId("DeleteInstalledButton"))!.AsButton().Invoke();
    WaitUntil(() => list.FindAllDescendants(cf => cf.ByAutomationId("InstalledName")).Length == names.Count - 1, TimeSpan.FromSeconds(10), "model removed");
    var quant = target.Name.Split('·').Last().Trim().ToLowerInvariant();
    var folder = Directory.EnumerateDirectories(Path.Combine(dataDir, "models")).Any(d => Path.GetFileName(d) == "liquidai_lfm2.5-1.2b-instruct-gguf__" + quant);
    Check(!folder, "model folder still on disk");
});

// ---------------------------------------------------------------- Second model, thinking, switching
Test("Switch model and show thinking", () =>
{
    GoToChat();
    Invoke("NewChatButton");
    PickModel("LFM2.5 1.2B Thinking");
    var reply = Send("What is 17 multiplied by 3? Answer with the number.");
    Check(reply.Contains("51"), $"reply: {reply}");
    Check(Get("EngineStatus").Name.Contains("LFM2.5"), $"engine did not switch: {Get("EngineStatus").Name}");
    var thinking = Find("ThinkingExpander");
    Log($"   thinking section shown: {thinking is not null && !thinking.Properties.IsOffscreen.ValueOrDefault}");
    Check(thinking is not null && !thinking.Properties.IsOffscreen.ValueOrDefault, "no thinking section for a thinking model");
    Shot("08-thinking");
});

Test("Web search answers with sources", () =>
{
    GoToChat();
    Invoke("NewChatButton");
    PickModel("Gemma 4 12B");
    var toggle = Get("WebSearchToggle").AsToggleButton();
    if (toggle.ToggleState != ToggleState.On)
        toggle.Toggle();
    try
    {
        var reply = Send("Search the web: what is the latest stable version of the Rust programming language? Answer in one sentence and cite the source.", 300);
        Check(reply.Length > 10, $"reply: {reply}");
        WaitUntil(() => Find("SourceLink") is not null, TimeSpan.FromSeconds(10), "source links");
        Check(Find("ActivityPill") is not null, "no activity pill for the search");
        Log($"   reply: {reply[..Math.Min(reply.Length, 160)]}");
        Shot("08-web-search");
    }
    finally
    {
        if (toggle.ToggleState == ToggleState.On)
            toggle.Toggle();
    }
});

// ---------------------------------------------------------------- Settings
Test("Every settings section opens with its content", () =>
{
    Invoke("SettingsButton");
    foreach (var (section, content) in new[] { ("SectionEngines", "EngineCard"), ("SectionSpeech", "TtsLine"), ("SectionStudio", "ComfyLine"),
                                                ("SectionAgent", "ClaudeVersion"), ("SectionWebSearch", "WebSearchToggle"), ("SectionStorage", "StorageTotal") })
    {
        Select(section);
        Get(content, 30);
        Shot($"09-{section}");
    }
    WaitUntil(() => Find("StorageTotal")?.Name.Contains("GB") == true, TimeSpan.FromSeconds(60), "storage total in GB");
    Invoke("BackButton");
});

Test("Context length setting is saved", () =>
{
    OpenSettings("SectionGeneral");
    var box = Get("ContextBox").AsComboBox();
    box.Select("32K");
    Thread.Sleep(500);
    Check(ReadSettings()["ContextLength"]?.GetValue<int>() == 32768, "ContextLength not 32768");
    box.Select("16K");
    Thread.Sleep(500);
    Check(ReadSettings()["ContextLength"]?.GetValue<int>() == 16384, "ContextLength not restored");
});

Test("Hugging Face token is saved", () =>
{
    Front();
    var box = Get("TokenBox");
    box.Click();
    box.Focus();
    Keyboard.TypeSimultaneously(VirtualKeyShort.CONTROL, VirtualKeyShort.KEY_A);
    Keyboard.Type(VirtualKeyShort.BACK);
    Keyboard.Type("hf_uitest");
    Thread.Sleep(500);
    Check(ReadSettings()["HuggingFaceToken"]?.GetValue<string>() == "hf_uitest", "token not saved");
    for (var i = 0; i < 9; i++)
        Keyboard.Type(VirtualKeyShort.BACK);
    Thread.Sleep(500);
    Check(ReadSettings()["HuggingFaceToken"]?.GetValue<string>() == "", "token not cleared");
});

Test("Language picker saves and asks for restart", () =>
{
    var box = Get("LanguageBox").AsComboBox();
    box.Select("Deutsch");
    Thread.Sleep(500);
    Check(ReadSettings()["Language"]?.GetValue<string>() == "de-DE", "language not saved");
    Check(Visible("RestartHint"), "restart hint not shown");
});

var expected = new Dictionary<string, string>
{
    ["es-ES"] = "Nuevo chat",
    ["pt-BR"] = "Nova conversa",
    ["fr-FR"] = "Nouvelle discussion",
    ["de-DE"] = "Neuer Chat",
    ["it-IT"] = "Nuova chat",
    ["ru-RU"] = "Новый чат",
    ["en-US"] = "New chat",
};
foreach (var (tag, label) in expected)
{
    Test($"UI is translated: {tag}", () =>
    {
        WriteSetting("Language", tag);
        Launch();
        var button = Get("NewChatButton");
        Check(button.FindAllDescendants().Any(e => e.Name == label) || button.Name == label, $"expected '{label}'");
        OpenSettings("SectionModels");
        Thread.Sleep(1500);
        Shot($"09-lang-{tag}");
    });
}
WriteSetting("Language", "");

Test("Dark theme renders", () =>
{
    Launch("--theme dark");
    Select("NavChat");
    Shot("10-dark-chat");
    OpenSettings("SectionModels");
    Thread.Sleep(3000);
    Shot("10-dark-models");
});

// ---------------------------------------------------------------- LiteRT-LM, MTP and the local API
// LiteRT-LM has no Windows on Arm build, so these two only run on x64.
if (System.Runtime.InteropServices.RuntimeInformation.OSArchitecture == System.Runtime.InteropServices.Architecture.X64)
{
    Test("LiteRT-LM model chats and shows its speed", () =>
    {
        GoToChat();
        Invoke("NewChatButton");
        PickModel("Gemma 4 E2B");
        var reply = Send($"{run} Name the capital of France in one word.");
        Check(reply.Contains("Paris", StringComparison.OrdinalIgnoreCase), $"reply: {reply}");
        var footer = FindAll("MessageFooter").Select(e => e.Name).LastOrDefault() ?? "";
        Check(footer.Contains("tok/s"), $"footer: {footer}");
        Shot("11-litert");
    });

    Test("MTP toggle is saved per model and reloads the engine", () =>
    {
        var toggle = WaitFor(() => Visible("MtpToggle") ? Find("MtpToggle") : null, TimeSpan.FromSeconds(5), "MTP toggle").AsToggleButton();
        var id = ReadSettings()["LastChatModelId"]?.GetValue<string>() ?? "";
        bool Off() => ReadSettings()["MtpOff"]?.AsArray().Any(n => n?.GetValue<string>() == id) == true;
        var wasOff = Off();
        toggle.Toggle();
        WaitUntil(() => Off() != wasOff, TimeSpan.FromSeconds(3), "MTP choice saved");
        var reply = Send($"{run} Reply with exactly one word: cherry");
        Check(reply.Contains("cherry", StringComparison.OrdinalIgnoreCase), $"reply after switching MTP: {reply}");
        toggle.Toggle();
        WaitUntil(() => Off() == wasOff, TimeSpan.FromSeconds(3), "MTP choice restored");
        PickModel("Gemma 4 12B");
        WaitUntil(() => !Visible("MtpToggle"), TimeSpan.FromSeconds(3), "MTP toggle hidden for GGUF");
    });
}

Test("Local API switch starts and stops the server", () =>
{
    OpenSettings("SectionApi");
    var toggle = Get("ApiToggle").AsToggleButton();
    var port = ReadSettings()["ApiPort"]?.GetValue<int>() ?? 8642;
    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
    bool Reachable()
    {
        try { return (int)http.GetAsync($"http://127.0.0.1:{port}/v1/models").Result.StatusCode is 200 or 401; }
        catch { return false; }
    }
    var wasOn = toggle.ToggleState == ToggleState.On;
    if (!wasOn)
        toggle.Toggle();
    WaitUntil(Reachable, TimeSpan.FromSeconds(10), "API answering");
    Shot("12-local-api");
    toggle.Toggle();
    WaitUntil(() => !Reachable(), TimeSpan.FromSeconds(10), "API stopped");
    if (wasOn)
        toggle.Toggle();
    GoToChat();
});

// ---------------------------------------------------------------- Studio and speech on real hardware
// Flux.1 schnell is a 16 GB download on first use, so this runs only where Studio has it already.
var fluxCheckpoint = Path.Combine(dataDir, "engines", "comfyui", "data", "models", "checkpoints", "flux1-schnell-fp8.safetensors");
if (File.Exists(fluxCheckpoint))
{
    Test("Studio makes an image with Flux.1 schnell", () =>
    {
        GoToChat();
        Select("NavStudio");
        if (Get("TabImage").AsToggleButton().ToggleState != ToggleState.On)
            Get("TabImage").AsToggleButton().Toggle();
        var picker = Get("WorkflowPicker").AsComboBox();
        picker.Expand();
        Thread.Sleep(400);
        var flux = picker.Items.FirstOrDefault(i => i.Text.Contains("Flux.1 schnell")) ?? throw new Exception($"no Flux.1 schnell in {string.Join(", ", picker.Items.Select(i => i.Text))}");
        flux.Select();
        picker.Collapse();
        SetText("PromptBox", "a red fox sitting in fresh snow at sunrise, detailed fur, photo");
        SetText("SeedBox", "42");
        var gallery = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "LLM Hub");
        var started = DateTime.Now;
        Invoke("GenerateButton");
        WaitUntil(() => Visible("CancelGenerateButton"), TimeSpan.FromSeconds(10), "generation to start");
        WaitUntil(() => Visible("GenerateButton") && !Visible("CancelGenerateButton"), TimeSpan.FromMinutes(15), "generation to finish");
        var status = Find("StudioStatus")?.Name ?? "";
        var made = Directory.EnumerateFiles(gallery, "*flux-schnell*.png").Select(f => new FileInfo(f)).Where(f => f.LastWriteTime >= started).ToList();
        Check(made.Count == 1 && made[0].Length > 100_000, $"no new Flux picture in {gallery} (status: {status})");
        WaitUntil(() => FindAll("GalleryItem").FirstOrDefault()?.Name == made[0].Name, TimeSpan.FromSeconds(10), "new picture first in the gallery");
        Log($"   {status}: {made[0].Name} ({made[0].Length / 1024} KB)");
        File.Copy(made[0].FullName, Path.Combine(outDir, "13-flux.png"), overwrite: true);
        Shot("13-studio-flux");
        Select("NavChat");
    });
}
else
    Log("   (Flux.1 schnell is not installed: Studio test not run)");

// Needs the default playback device to be speakers the default microphone can hear (not headphones); runs when asked
// for by name or with LLMHUB_UITEST_MIC=1.
if (filter is not null || Environment.GetEnvironmentVariable("LLMHUB_UITEST_MIC") == "1")
{
    Test("Dictation from the microphone", () =>
    {
        // Windows' own voice reads a sentence aloud through the speakers while the chat mic records.
        const string sentence = "The lighthouse keeper rowed across the quiet harbor at dawn.";
        var wav = Path.Combine(outDir, "dictation-source.wav");
        var speak = Process.Start(new ProcessStartInfo("powershell.exe",
            $"-NoProfile -Command \"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.SetOutputToWaveFile('{wav}'); $s.Speak('{sentence}'); $s.Dispose()\"")
            { UseShellExecute = false, CreateNoWindow = true })!;
        speak.WaitForExit(30_000);
        Check(File.Exists(wav), "could not make the spoken sentence");
        GoToChat();
        Invoke("NewChatButton");
        SetText("Composer", "");
        Invoke("DictateButton");
        WaitUntil(() => Get("DictateButton").Name.StartsWith("Listening"), TimeSpan.FromSeconds(10), "recording");
        Thread.Sleep(800);
        Sound.PlaySync(wav);
        Thread.Sleep(800);
        Invoke("DictateButton");
        // Whisper downloads on first use.
        WaitUntil(() => Get("Composer").AsTextBox().Text.Length > 0, TimeSpan.FromMinutes(15), "transcript in the composer");
        var heard = Get("Composer").AsTextBox().Text;
        static string[] Words(string s) => s.ToLowerInvariant().Split(s.Where(c => !char.IsLetterOrDigit(c)).Distinct().ToArray(), StringSplitOptions.RemoveEmptyEntries);
        var expected = Words(sentence);
        var share = expected.Count(Words(heard).Contains) / (double)expected.Length;
        Log($"   heard ({share:P0} of the words): {heard}");
        Shot("14-dictation");
        Check(share >= 0.7, $"transcript: {heard}");
        SetText("Composer", "");
    });
}

// ---------------------------------------------------------------- Engine
Test("Engine falls back when a GPU backend fails", () =>
{
    var working = Path.Combine(dataDir, "engines", "llama.cpp", "working.txt");
    File.Delete(working);
    Launch("", new() { ["LLMHUB_FAIL_VARIANT"] = "win-cuda-13.4-x64" });
    Invoke("NewChatButton");
    PickModel("LFM2.5 1.2B Thinking");
    var reply = Send("Say hi in one word.", timeoutSeconds: 600);
    Check(File.ReadAllText(working).Trim() == "win-vulkan-x64", $"working build: {File.ReadAllText(working)}");
    Log($"   fallback reply: {reply}");
    Close();
    File.Delete(working);
    Launch();
});

Test("llama-server stops when the app closes", () =>
{
    Invoke("NewChatButton");
    PickModel("Gemma 4 12B");
    Send("Say OK.");
    Check(LlamaServers() == 1, $"expected 1 llama-server, found {LlamaServers()}");
    Close();
    WaitUntil(() => LlamaServers() == 0, TimeSpan.FromSeconds(20), "llama-server to exit");
});

Close();
var passed = results.Count(r => r.Pass);
Log($"DONE {passed}/{results.Count} passed");
return passed == results.Count ? 0 : 1;

/// <summary>Plays a WAV through the default speakers and returns when it ends.</summary>
static class Sound
{
    private const uint SndSync = 0x0000, SndFilename = 0x00020000, SndNoDefault = 0x0002;

    public static void PlaySync(string wav) => PlaySound(wav, IntPtr.Zero, SndFilename | SndSync | SndNoDefault);

    [System.Runtime.InteropServices.DllImport("winmm.dll", EntryPoint = "PlaySoundW", CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
    private static extern bool PlaySound(string? sound, IntPtr module, uint flags);
}
