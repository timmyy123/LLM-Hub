---
name: remote-test-lab
description: Rent any Samsung Remote Test Lab device (about 1,300 Galaxy phones, tablets, watches and TVs), connect it to adb, and test LLM Hub on it by command, without clicking the website. Use when asked to test on a Samsung device the owner doesn't have, reproduce a Play review or vitals report on a specific Galaxy model, or check speed, loading, Stop, New Chat and leaving the chat on real hardware.
---

# Samsung Remote Test Lab for LLM Hub

Tools live in `android/tools/remote_test_lab/`:
- `rtl.mjs`: rents devices through the site's own API and keeps the web viewer open in a **headless Chrome**
  (its own profile in `~/.llmhub-rtl/`). The viewer must stay open for the rental, and headless Chrome keeps it
  running even when the Mac is locked (a normal Chrome window is "hidden" then and the viewer never starts).
- `device.py`: drives LLM Hub on any adb device (lab phone or the owner's Tab S9+).

## One-time setup (the owner, at the Mac)
`node android/tools/remote_test_lab/rtl.mjs login` opens a Chrome window with the tool's profile; the owner signs in
to Samsung there. Never type the owner's password or copy cookies from their browser. If any command says
"Not signed in", ask the owner to run `login` again.

## Rent and connect
```
node android/tools/remote_test_lab/rtl.mjs devices [--model "Tab S9"] [--free]   # every model, free = true can be rented
node android/tools/remote_test_lab/rtl.mjs rent --model "Galaxy A56" --minutes 30 [--location USA] [--os 17]
node android/tools/remote_test_lab/rtl.mjs rent --device SM-S948U-US01           # one specific unit (name or id)
node android/tools/remote_test_lab/rtl.mjs rent                                  # any free device
node android/tools/remote_test_lab/rtl.mjs status     # minutes left, viewer open, adb state
node android/tools/remote_test_lab/rtl.mjs release    # return the device when done (stops spending credits)
```
- `--model` matches part of the product name (exact name first, so "Galaxy S26" doesn't pick an S26 Ultra).
- Credits: 1 per 15 minutes (`--minutes` rounds up). Check `status` before long work; rent again if it runs out.
- `rent` prints the adb serial (`localhost:<rdb port>`). An `offline` entry is often a stale transport from an earlier
  rental: `adb disconnect localhost:<port>` then `adb connect localhost:<port>` (the tool now does this itself).
  If adb still stays `offline`, run `rtl.mjs viewer` once, then
  `rtl.mjs screenshot` to see the viewer.
- The `rdb` bridge (`~/Downloads/rdb/rdb`) must be running; `rent` starts it if not.

## Test LLM Hub on it
Install the side-by-side preview build (add `applicationIdSuffix = ".preview"` to the debug build type only for the
build, never commit it), then:
```
D="python3 -I android/tools/remote_test_lab/device.py -s localhost:<port>"
$D install android/app/build/outputs/apk/debug/app-debug.apk android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
$D wake; $D launch; $D chat
$D model "LFM-2.5 VL 3B (Q4_0)" "NPU (Hexagon)"     # model must be downloaded first (Download screen, `tap`)
$D send "Answer only in Korean. Explain how rainbows form."; $D wait-reply
$D send "Write a 600 word story."; sleep 5; $D stop; $D stats      # Stop mid-reply
$D send "Write a 600 word story."; sleep 5; $D new-chat; $D stats  # New Chat mid-reply
$D send "Write a 600 word story."; sleep 5; $D back; $D stats      # leave the chat mid-reply
$D crashes
```
`stats` prints the engine's own numbers (`llama.cpp ... generation completed: N tokens at X tok/s`,
`LiteRT-LM reply: ... decode N tokens at X tok/s`) next to the chat's saved badge (`Saving stats ... N tokens, X tok/sec`).
`shot NAME` and `ui` show the screen; `tap LABEL` presses anything by its text or content description.

## Rules (from the owner)
- Never send `svc power`, `settings put` or `KEYCODE_WAKEUP` to a lab device: it drops the session. `device.py wake` is safe.
- Phones download models themselves through the app; never `adb push` a model.
- `adb install` over the lab bridge takes about 5 minutes for the 280 MB debug APK: run it in the background.
- Release the device when done. Ask the owner before using their own Tab S9+ or iPhone.
