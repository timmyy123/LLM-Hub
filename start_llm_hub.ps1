# LLM Hub Startup Script
# Uses iTunes/Apple Devices + port forwarding (NOT usbipd) for reliable xtool dev.
# See ios/XTOL_WINDOWS_SETUP.md for one-time setup (firewall, port forwarding).

Write-Host "Starting LLM Hub Dev Environment..." -ForegroundColor Magenta
Write-Host "Ensure iPhone is connected via USB and trusted on Windows (iTunes or Apple Devices app)" -ForegroundColor Cyan

# Get WSL gateway IP for port forwarding
$routeLine = wsl ip route list default 2>$null | Select-Object -First 1
if ($routeLine) {
    $parts = $routeLine -split ' '
    $wslGateway = $parts[2]
    if ($wslGateway) {
        Write-Host "Port 27015 forwarded via WSL gateway: $wslGateway" -ForegroundColor Cyan
        netsh interface portproxy set v4tov4 listenport=27015 connectaddress=127.0.0.1 connectport=27015 listenaddress=$wslGateway 2>$null
    }
}
if (-not $wslGateway) {
    Write-Host "Could not get WSL gateway. Run one-time setup from ios/XTOL_WINDOWS_SETUP.md" -ForegroundColor Yellow
}

Write-Host "Starting xtool dev (build + install to iPhone)..." -ForegroundColor Magenta
$bashCmd = "export USBMUXD_SOCKET_ADDRESS=`"`$(ip route list default | cut -d' ' -f3):27015`" && cd /mnt/c/Users/timmy/Downloads/LLM-Hub/ios/LLMHub && xtool dev"
wsl -- bash --login -c $bashCmd
