using LLMHub.Services;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;

namespace LLMHub.Views;

/// <summary>Stands in for sections that are not built yet (Agent, Design, Studio).</summary>
public sealed partial class PlaceholderPage : Page
{
    public PlaceholderPage() => InitializeComponent();

    protected override void OnNavigatedTo(NavigationEventArgs e)
    {
        var section = e.Parameter as string ?? "";
        Icon.Glyph = section switch
        {
            "Agent" => "",
            "Design" => "",
            _ => "",
        };
        Heading.Text = Loc.F("ComingSoonTitle", Loc.S($"Nav{section}"));
        Body.Text = Loc.S($"{section}Description");
    }
}
