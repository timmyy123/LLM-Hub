> **Outdated.** This guide was only used at the very start of the project, to build the iOS app with xtool on Windows (WSL) before there was a Mac. The iOS app is now built with Xcode on a Mac, so don't follow it.

# xtool – What to Do Each Time

## 1. Open WSL

Open **Ubuntu** from the Start menu, or run `wsl` in PowerShell.

---

## 2. Go to Your Project

```bash
cd ~/MyApp
```

(Or wherever your app lives. Use `xtool new AppName` to create a new one.)

---

## 3. Build an IPA

```bash
xtool dev build --ipa
```

The `.ipa` is in `xtool/AppName.ipa`. Install it with AltStore, Sideloadly, or another sideloading tool.

---

## Quick Commands

| Command | What it does |
|--------|--------------|
| `xtool new AppName` | Create new app |
| `xtool dev build --ipa` | Build IPA for sideloading |
| `xtool dev` | Build and install via USB (if usbipd configured) |
