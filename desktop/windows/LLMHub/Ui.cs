using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;

namespace LLMHub;

/// <summary>Small helpers for x:Bind function bindings.</summary>
public static class Ui
{
    public static Visibility Show(bool value) => value ? Visibility.Visible : Visibility.Collapsed;
    public static Visibility Hide(bool value) => value ? Visibility.Collapsed : Visibility.Visible;
    public static Visibility ShowText(string? value) => string.IsNullOrEmpty(value) ? Visibility.Collapsed : Visibility.Visible;
    public static Visibility ShowAll(bool a, bool b) => a && b ? Visibility.Visible : Visibility.Collapsed;
    public static bool Not(bool value) => !value;
    /// <summary>Looks up a brush from Themes/Theme.xaml for the element's current light/dark theme.</summary>
    public static Brush? ThemeBrush(FrameworkElement element, string key)
    {
        var theme = element.ActualTheme == ElementTheme.Light ? "Light" : "Default";
        foreach (var dictionary in Application.Current.Resources.MergedDictionaries)
        {
            if (dictionary.ThemeDictionaries.TryGetValue(theme, out var themed)
                && themed is ResourceDictionary resources
                && resources.TryGetValue(key, out var value))
                return value as Brush;
        }
        return null;
    }

    public static HorizontalAlignment MessageAlign(bool isUser) => isUser ? HorizontalAlignment.Right : HorizontalAlignment.Stretch;
}
