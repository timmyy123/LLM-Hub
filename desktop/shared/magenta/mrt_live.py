"""Live Magenta RealTime 2 for LLM Hub on Linux with an NVIDIA GPU (JAX + CUDA).

The app starts this with the private Python it installed and talks to it over pipes:
  stdin   one JSON command per line:
            {"cmd": "styles", "styles": [{"text": "jazz trio", "weight": 0.7}, ...]}
            {"cmd": "controls", "temperature": 1.3, "top_k": 40, "guidance": 3.0}
            {"cmd": "play"} | {"cmd": "stop"} | {"cmd": "quit"}
  stdout  audio blocks: b"AUD0", uint32 little-endian byte count, then float32 interleaved stereo at 48 kHz
  stderr  "EVT {json}" lines (ready, playing, error) mixed with the libraries' own logging
Style changes take effect on the next block, so the music follows a slider within about a second.
"""

import json
import logging
import struct
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 48000
FRAMES_PER_BLOCK = 10  # 25 frames per second: 0.4 s of audio per block
LEAD_SECONDS = 1.2  # how far generation may run ahead of the clock


def event(**fields):
    sys.stderr.write("EVT " + json.dumps(fields) + "\n")
    sys.stderr.flush()


class Controls:
    def __init__(self):
        self.lock = threading.Condition()
        self.styles = []
        self.temperature = 1.3
        self.top_k = 40
        self.guidance = 3.0
        self.playing = False
        self.generation = 0  # bumped on play so a new session starts from a fresh state
        self.quit = False


def read_commands(controls):
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        with controls.lock:
            name = command.get("cmd")
            if name == "styles":
                controls.styles = [(s["text"].strip(), float(s["weight"])) for s in command.get("styles", [])
                                   if s.get("text", "").strip() and float(s.get("weight", 0)) > 0]
            elif name == "controls":
                controls.temperature = float(command.get("temperature", controls.temperature))
                controls.top_k = max(1, int(command.get("top_k", controls.top_k)))
                controls.guidance = float(command.get("guidance", controls.guidance))
            elif name == "play":
                controls.playing = True
                controls.generation += 1
            elif name == "stop":
                controls.playing = False
            elif name == "quit":
                controls.quit = True
            controls.lock.notify_all()
    with controls.lock:
        controls.quit = True
        controls.lock.notify_all()


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    model = sys.argv[1] if len(sys.argv) > 1 else "mrt2_small"
    try:
        import jax
        from magenta_rt import MagentaRT2Jax
        from magenta_rt.config import MUSICCOCA

        devices = [d.device_kind for d in jax.devices() if d.platform == "gpu"]
        mrt = MagentaRT2Jax(size=model)
    except Exception as error:  # noqa: BLE001 - reported to the app as text
        event(type="error", message=f"{type(error).__name__}: {error}")
        return 1
    event(type="ready", device=devices[0] if devices else "CPU", sampleRate=SAMPLE_RATE)

    controls = Controls()
    threading.Thread(target=read_commands, args=(controls,), daemon=True).start()
    out = sys.stdout.buffer
    embeddings = {}
    blend_key, blend = None, None
    state, generation = None, -1
    started, made = 0.0, 0.0

    while True:
        with controls.lock:
            while not controls.quit and not (controls.playing and controls.styles):
                controls.lock.wait()
            if controls.quit:
                return 0
            styles = list(controls.styles)
            temperature, top_k, guidance = controls.temperature, controls.top_k, controls.guidance
            if controls.generation != generation:
                generation, state = controls.generation, None
                started, made = time.monotonic(), 0.0

        # A weighted average of the styles' MusicCoCa embeddings; each text is embedded once.
        key = tuple(styles)
        if key != blend_key:
            total = sum(w for _, w in styles)
            for text, _ in styles:
                if text not in embeddings:
                    embeddings[text] = np.asarray(mrt.embed_style(text, use_mapper=True), dtype=np.float32)
            blend = sum(embeddings[text] * (w / total) for text, w in styles)
            blend_key = key

        ahead = made - (time.monotonic() - started)
        if ahead > LEAD_SECONDS:
            time.sleep(min(ahead - LEAD_SECONDS, 0.05))
            continue

        try:
            wav, state = mrt.generate(conditioning={MUSICCOCA.key: blend}, cfg_scales={"musiccoca": guidance},
                                      temperature=temperature, top_k=top_k, frames=FRAMES_PER_BLOCK, state=state)
        except Exception as error:  # noqa: BLE001
            event(type="error", message=f"{type(error).__name__}: {error}")
            with controls.lock:
                controls.playing = False
            continue
        samples = np.ascontiguousarray(wav.samples, dtype="<f4")
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        data = samples[:, :2].tobytes()
        try:
            out.write(b"AUD0" + struct.pack("<I", len(data)) + data)
            out.flush()
        except BrokenPipeError:
            return 0
        made += samples.shape[0] / SAMPLE_RATE


if __name__ == "__main__":
    sys.exit(main())
