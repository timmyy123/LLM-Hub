using System.Net.Sockets;

namespace LLMHub.Services;

/// <summary>
/// Retries GET and HEAD requests that fail before reaching the server: a name lookup that briefly fails or a connection
/// that is refused or reset (home routers drop the odd DNS answer). Three tries, half a second then a second and a half
/// apart; anything the server actually answered, and every other method, goes through untouched.
/// </summary>
public sealed class RetryHandler(HttpMessageHandler inner) : DelegatingHandler(inner)
{
    private static readonly TimeSpan[] Delays = [TimeSpan.FromMilliseconds(500), TimeSpan.FromMilliseconds(1500)];

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        if (request.Method != HttpMethod.Get && request.Method != HttpMethod.Head)
            return await base.SendAsync(request, ct);
        for (var attempt = 0; ; attempt++)
        {
            try
            {
                return await base.SendAsync(request, ct);
            }
            catch (HttpRequestException e) when (attempt < Delays.Length && IsTransient(e) && !ct.IsCancellationRequested)
            {
                Log.Info($"Network: {request.RequestUri?.Host} failed ({e.InnerException?.Message ?? e.Message}), retrying");
                await Task.Delay(Delays[attempt], ct);
            }
        }
    }

    private static bool IsTransient(HttpRequestException e) => e.InnerException is SocketException
    {
        SocketErrorCode: SocketError.HostNotFound or SocketError.TryAgain or SocketError.NoData or SocketError.ConnectionRefused
            or SocketError.ConnectionReset or SocketError.NetworkUnreachable or SocketError.HostUnreachable or SocketError.TimedOut,
    } || e.HttpRequestError is HttpRequestError.NameResolutionError or HttpRequestError.ConnectionError;
}
