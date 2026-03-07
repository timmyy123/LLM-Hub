# LLM Hub Startup Script
# This script handles iPhone connection to WSL and starts the dev server.

Write-Host "🚀 Starting LLM Hub Dev Environment..." -ForegroundColor Indigo

# 1. Attach iPhone to WSL
$busid = "2-5" # Your iPhone Bus ID
Write-Host "📱 Attaching iPhone (Bus $busid) to WSL..." -ForegroundColor Cyan
& "C:\Program Files\usbipd-win\usbipd.exe" attach --wsl --busid $busid 2>$null

# 2. Check if attached
$status = & "C:\Program Files\usbipd-win\usbipd.exe" list
if ($status -match "$busid.*Attached") {
    Write-Host "✅ iPhone attached successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Warning: iPhone might already be attached or failed to attach. Continuing anyway..." -ForegroundColor Yellow
}

# 3. Start xtool dev in WSL
Write-Host "🛠️ Starting xtool dev server..." -ForegroundColor Magenta
$bashCommand = "export USBMUXD_SOCKET_ADDRESS=127.0.0.1:27015 && cd /mnt/c/Users/timmy/Downloads/LLM-Hub/ios/LLMHub && xtool dev"
wsl -- bash --login -c $bashCommand
