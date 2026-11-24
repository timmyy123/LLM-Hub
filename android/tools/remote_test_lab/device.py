#!/usr/bin/env python3
"""Drive LLM Hub on one Android device over adb, for agents: wake, install, open chat, send a prompt,
stop or leave mid-reply, and compare the chat's tok/s badge with the engine's own log line.

usage: device.py [-s SERIAL] [--app com.llmhub.llmhub.preview] <command> [args]

  info                      model, chip, Android, battery, installed LLM Hub apps
  wake                      turn the screen on without power commands (Remote Test Lab safe), dismiss the lock screen
  install APK [TEST_APK]    adb install -r (keeps app data); also installs the test APK for `wake`
  launch                    bring the app to the front
  shot NAME                 half-size screenshot, prints its path
  ui                        list labeled elements (text / content-desc) with tap coordinates
  tap LABEL                 tap the element whose text or content-desc is LABEL (trailing * = prefix)
  back                      press Back
  chat                      open AI Chat from the home screen
  model NAME [BACKEND]      pick the chat model (and CPU / GPU / NPU (Hexagon)) in chat settings and load it
  send TEXT                 type TEXT in the chat box and send it (ASCII; ask for other languages in words)
  wait-reply [SECONDS]      wait until the reply finishes (no Cancel button), then print stats
  stop                      tap Cancel generation
  new-chat                  Menu, then New Chat
  stats [N]                 the last N engine / chat speed lines from logcat
  crashes                   FATAL / SIGSEGV / ANR lines for the app since it started

Default serial: the only device adb lists, else the Remote Test Lab bridge (localhost:<rdb port>).
Never sends svc power, settings put or KEYCODE_WAKEUP: those drop Remote Test Lab sessions.
"""
import argparse, os, re, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("-s", dest="serial")
ap.add_argument("--app", default="com.llmhub.llmhub.preview")
ap.add_argument("cmd", nargs="?")
ap.add_argument("rest", nargs="*")
a = ap.parse_args()
if not a.cmd or a.cmd in ("-h", "help"):
    print(__doc__); sys.exit(0)
OUT = os.path.join(tempfile.gettempdir(), "llmhub-device")
os.makedirs(OUT, exist_ok=True)


def run(*args, timeout=40, check=False):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode: sys.exit(p.stderr or p.stdout)
    return p.stdout


def pick_serial():
    if a.serial: return a.serial
    lines = [l.split()[0] for l in run("adb", "devices").splitlines()[1:] if l.strip().endswith("device")]
    if len(lines) == 1: return lines[0]
    lab = [l for l in lines if l.startswith("localhost:")]
    if lab: return lab[0]
    sys.exit(f"pick a device with -s: {lines}")


S = pick_serial()
adb = lambda *x, **k: run("adb", "-s", S, *x, **k)
sh = lambda cmd, **k: adb("shell", cmd, **k)


def size():
    m = re.search(r"(\d+)x(\d+)", sh("wm size"));
    return int(m.group(1)), int(m.group(2))


def nodes():
    sh("uiautomator dump /sdcard/llmhub-ui.xml >/dev/null 2>&1", timeout=60)
    xml = sh("cat /sdcard/llmhub-ui.xml")
    out = []
    for n in re.findall(r"<node [^>]*>", xml):
        g = lambda k: (re.search(k + r'="([^"]*)"', n) or [None, ""])[1]
        b = list(map(int, re.findall(r"\d+", g("bounds")))) or [0, 0, 0, 0]
        out.append({"text": g(" text"), "desc": g("content-desc"), "pkg": g("package"),
                    "click": 'clickable="true"' in n, "x": (b[0] + b[2]) // 2, "y": (b[1] + b[3]) // 2})
    return out


def find(label):
    for n in nodes():
        for v in (n["text"], n["desc"]):
            if v == label or (label.endswith("*") and v.startswith(label[:-1])): return n
    return None


def tap(label, required=True):
    n = find(label)
    if not n:
        if required: sys.exit(f"no element '{label}' on screen")
        return False
    sh(f"input tap {n['x']} {n['y']}"); time.sleep(1); return True


def pid(): return sh(f"pidof {a.app}").strip()


def stats(n=6):
    p = pid()
    if not p: return "app not running"
    log = adb("logcat", "-d", "-v", "time", f"--pid={p}", timeout=60)
    keep = re.compile(r"Starting completion|generation completed|LiteRT-LM reply|Saving stats|Exception caught|Generation was cancelled")
    return "\n".join([l for l in log.splitlines() if keep.search(l)][-n:])


c, r = a.cmd, a.rest
if c == "info":
    print(sh("getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.release; dumpsys battery | grep -m1 level; pm list packages | grep llmhub"))
elif c == "wake":
    if "mWakefulness=Awake" not in sh("dumpsys power | grep -m1 mWakefulness="):
        sh("am start -W -n com.llmhub.llmhub.preview.test/com.llmhub.llmhub.KeepAwakeActivity"); time.sleep(5)
    sh("wm dismiss-keyguard"); time.sleep(1)
    print(sh("dumpsys power | grep -m1 mWakefulness=").strip())
elif c == "install":
    for apk in r: print(adb("install", "-r", apk, timeout=1800).strip().splitlines()[-1:])
elif c == "launch":
    sh(f"am start -n {a.app}/com.llmhub.llmhub.MainActivity"); time.sleep(3); print(sh("dumpsys window | grep -m1 mCurrentFocus").strip())
elif c == "shot":
    path = os.path.join(OUT, (r[0] if r else "shot") + ".png")
    with open(path, "wb") as f: f.write(subprocess.run(["adb", "-s", S, "exec-out", "screencap", "-p"], capture_output=True, timeout=40).stdout)
    w, _ = size(); subprocess.run(["sips", "--resampleWidth", str(min(w // 2, 700)), path], capture_output=True)
    print(path)
elif c == "ui":
    for n in nodes():
        if (n["text"] or n["desc"]) and "llmhub" in n["pkg"]: print(f"{n['x']:>5} {n['y']:>5}  {n['text'] or n['desc']!r}{'  (clickable)' if n['click'] else ''}")
elif c == "tap": tap(" ".join(r))
elif c == "back": sh("input keyevent KEYCODE_BACK")
elif c == "chat": tap("Chat Now")
elif c == "model":
    tap("Settings"); name = r[0]
    current = [n for n in nodes() if n["text"] and n["y"] < size()[1] // 3 and n["text"] not in ("Settings", "Select Model", "Select model", "Select Backend")]
    if current: sh(f"input tap {current[0]['x']} {current[0]['y']}"); time.sleep(1)
    tap(name)
    if len(r) > 1:
        b = find("CPU") or find("GPU") or find("NPU (Hexagon)")
        if b: sh(f"input tap {b['x']} {b['y']}"); time.sleep(1); tap(r[1])
    tap("Load Model"); time.sleep(15); print(stats(2))
elif c == "send":
    w, h = size(); box = find("Type a message*")
    sh(f"input tap {box['x'] if box else w // 2} {box['y'] if box else h - 120}"); time.sleep(1)
    sh("input text " + "'" + " ".join(r).replace(" ", "%s").replace("'", "") + "'"); time.sleep(1)
    tap("Send"); print(time.strftime("%H:%M:%S sent"))
elif c == "wait-reply":
    end = time.time() + int(r[0]) if r else time.time() + 180
    time.sleep(3)
    while time.time() < end and find("Cancel generation"): time.sleep(3)
    print(stats(3))
elif c == "stop": tap("Cancel generation")
elif c == "new-chat": tap("Menu"); tap("New Chat")
elif c == "stats": print(stats(int(r[0]) if r else 6))
elif c == "crashes":
    log = adb("logcat", "-d", "-v", "time", "-b", "crash,main", timeout=60)
    print("\n".join(l for l in log.splitlines() if a.app in l and re.search(r"FATAL|SIGSEGV|ANR in|Fatal signal", l))[-3000:] or "no crashes")
else:
    sys.exit(f"unknown command {c}\n{__doc__}")
