# xtool + iPhone on Windows/WSL – Setup Guide

Use **iTunes/Apple Devices + port forwarding** (not usbipd) for reliable `xtool dev` installation. This is the method that works per [xtool#19](https://github.com/xtool-org/xtool/issues/19) and [usbmuxd#126](https://github.com/libimobiledevice/usbmuxd/issues/126).

---

## Prerequisites

1. **WSL2** with Ubuntu (or similar)
2. **xtool** installed and configured (`xtool setup` done)
3. **iTunes** or **Apple Devices** (from Microsoft Store)
4. **Apple ID** signed into the app on your iPhone

---

## One-Time Setup

### 1. Connect iPhone and trust the computer

1. Connect the iPhone to the PC via USB
2. Open **iTunes** or **Apple Devices** on Windows
3. When prompted on iPhone: **Trust this computer**
4. Leave the cable connected and iTunes/Apple Devices can stay closed afterward

### 2. Add Windows Firewall rule (run once as Administrator)

Open **PowerShell as Administrator**:

```powershell
# Try default WSL interface name first
New-NetFirewallRule -DisplayName "WSL to AppleMobileDevice" -Direction Inbound -InterfaceAlias "vEthernet (WSL)" -Action Allow
```

If that fails, run `ipconfig` and find your WSL adapter (e.g. "vEthernet (WSL)" or "vEthernet (WSL (Hyper-V firewall))"), then use its exact name:

```powershell
New-NetFirewallRule -DisplayName "WSL to AppleMobileDevice" -Direction Inbound -InterfaceAlias "vEthernet (WSL (Hyper-V firewall))" -Action Allow
```

### 3. Configure port forwarding

In **PowerShell (can be normal user)**:

```powershell
# Get WSL default gateway (host IP as seen from WSL)
$host = $(wsl -- ip route list default | ForEach-Object { ($_ -split '\s+')[2] })
Write-Host "WSL gateway: $host"

# Forward port 27015 from WSL to Windows AppleMobileDeviceService
netsh interface portproxy set v4tov4 listenport=27015 connectaddress=127.0.0.1 connectport=27015 listenaddress=$host
```

To remove it later:

```powershell
netsh interface portproxy delete v4tov4 listenport=27015 listenaddress=$host
```

---

## Every Time You Want to Install

### 1. Ensure iPhone is connected on Windows

- iPhone connected via USB to Windows
- Device trusted (no need to open iTunes again unless re-trusting)

### 2. Run from WSL

```bash
export USBMUXD_SOCKET_ADDRESS="$(ip route list default | awk '{print $3}'):27015"
cd /mnt/c/Users/timmy/Downloads/LLM-Hub/ios/LLMHub
xtool dev
```

Or use the startup script (see below).

---

## Optional: Use `start_llm_hub.ps1`

If you've done the one-time setup above, run:

```powershell
.\start_llm_hub.ps1
```

**Note:** The script currently uses usbipd. For the iTunes/port-forward method, use the WSL commands directly instead of the script, or update the script to skip usbipd.

---

## Troubleshooting

### "No device found" or "muxError"

- **Don’t** use `usbipd attach` with this setup; keep the iPhone attached only to Windows.
- Open iTunes/Apple Devices and confirm the device is detected.
- On iPhone: Settings → General → Reset → Reset Location & Privacy, then reconnect and choose Trust again.
- Run `ideviceinfo` in WSL (with `USBMUXD_SOCKET_ADDRESS` set) to verify connection before `xtool dev`.

### Verify connection from WSL

```bash
export USBMUXD_SOCKET_ADDRESS="$(ip route list default | awk '{print $3}'):27015"
ideviceinfo
```

If `ideviceinfo` lists your device, `xtool dev` should work.

### Install ideviceinfo if needed

```bash
sudo apt-get install usbmuxd libimobiledevice6 libimobiledevice-utils
```

### Port forwarding after reboot

Port forwarding rules may reset. Re-run the `netsh interface portproxy` command when needed.

### xtool.yml bundle ID

For installation to your own device, set a unique bundle ID in `ios/LLMHub/xtool.yml`, for example:

```yaml
bundleID: com.yourname.LLMHub
```
