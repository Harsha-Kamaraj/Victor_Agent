# Victor — build plan

Nine phases, each with an exit gate. A phase is done when its gate demonstrably
passes, not when the code is written. Gates are written as things you can watch
happen, because that is the only definition that survives a demo.

Phases are units of execution, not days. The original sketch was six days; the
sequencing below is what actually matters.

---

## Dependency graph

```
P0 ─┬─> P1 ──> P2 ──> P3 ──> P5 ──> P6 ──> P7 ──> P8
    └─> P4 ───────────────────┘
```

P4 is read-only — it observes the screen and changes nothing — so it is the one
phase safe to build out of order or in parallel. Everything that *acts* (P5)
waits behind the safety layer (P3), never the other way round.

---

## P0 · Skeleton & Plumbing ✅

Nothing here is interesting on its own. It exists so that every later phase has
somewhere to put its config, its spending, and its evidence.

**Built**
- `config.py` — env + `.env`, secrets held as `SecretStr` so they can't be
  logged by accident.
- `quota.py` — the free-tier ledger. Normalises requests/min, requests/day,
  tokens/min, tokens/day and audio-seconds/day across providers whose daily
  windows reset in different timezones. Persisted, atomic, thread-safe.
- `providers/` — the routing table plus a `Router` that picks the best model a
  workload can currently afford. Selection is pure: no network, fully testable.
- `tracing.py` — one JSONL file per session, append-only, flushed per event so
  a killed process still leaves a readable trace.
- `doctor.py` / `cli.py` — `victor doctor`, `quota`, `route`, `models`, `trace`.

**Exit gate** — `victor doctor` reports honestly on a machine with no keys;
`victor route vision` visibly falls through from Gemini to Groq once the ledger
says 250/250 is spent. ✅

**Design note.** The ledger reserves a request *before* the call and reconciles
real token counts *after*. A crash mid-call therefore over-counts by one
request. That direction of error is deliberate: over-counting costs a little
headroom, under-counting costs money.

---

## P1 · Voice I/O ✅

**Built**
- `voice/audio.py` — one representation end to end: int16 mono PCM at a
  declared rate, plus WAV encode/decode, dBFS and resampling.
- `voice/sources.py` — `AudioSource` protocol. `MicrophoneSource` for
  PortAudio, `ArraySource` for tests and benchmarks. This split is why the
  entire endpointing and transcription stack is testable with no audio
  hardware.
- `voice/vad.py` — `EnergyVad` (default) and an optional `WebRtcVad`, under an
  `Endpointer` state machine with hysteresis, pre-roll and guard rails.
- `voice/stt.py` — Groq Whisper, routed and metered.
- `voice/tts.py` — Piper ONNX with streaming playback; system and null backends.
- `voice/pipeline.py`, `voice/bench.py` — composition and measurement.
- CLI: `victor listen`, `say`, `voice devices`, `voice install`, `bench voice`.

**Exit gate** — real microphone capture (2.00 s at 16 kHz, verified), real
Piper synthesis played through real speakers, and measured p50/p95 published
below rather than asserted. ✅

### Three decisions that changed the design

**WebRTC VAD was dropped as a dependency.** `webrtcvad` 2.0.10 imports
`pkg_resources`, which setuptools 81+ removed, so it fails at import on a
current install. Putting that on the critical path of a voice agent is not
worth it. Victor ships an adaptive energy detector instead — a running noise
floor that only adapts on non-speech frames, so a long utterance cannot drag
the threshold up over itself. WebRTC remains available as an opt-in backend.

**The noise floor is clamped.** If the mic opens while the user is already
talking — routine in push-to-talk — calibration measures speech, the floor
lands at speech level, and the VAD goes permanently deaf. Capping the floor at
-35 dBFS trades a slightly trigger-happy threshold for never failing to hear.

**The end-of-turn silence is trimmed before upload.** Ending a turn takes
700 ms of silence, and Whisper bills by audio duration. Uploading the pause
that proved the user stopped talking would spend ~15% of the audio budget on
nothing. Only a 200 ms tail is kept, enough to avoid clipping a final
consonant.

### Measured on this machine

MacBook Air (Apple Silicon), Python 3.13.14, `victor bench voice --runs 7`:

```
stage                                  runs        p50        p95  unit
vad endpointing                           7        0.1        0.1  ms/s audio
tts model load (once)                     1      525.6      525.6  ms
tts time-to-first-audio (1 sentence)      7       75.5      132.9  ms
tts synthesis (1 sentence)                7       75.5      132.9  ms
tts time-to-first-audio (3 sentences)     7       42.1       46.8  ms
tts synthesis (3 sentences)               7      108.5      112.8  ms
tts realtime factor (3 sentences)         7      0.036      0.037  x
```

**Piper chunks per sentence.** For a one-sentence reply, time-to-first-audio
*is* the full synthesis time and streaming buys nothing; across three
sentences it drops from 108 ms to 42 ms. The benchmark reports both rather
than averaging the distinction away. The practical consequence for P2: replies
should be written as several short sentences, not one long one.

**Not yet measured:** STT round trip and the full voice→voice loop, both of
which need a live `GROQ_API_KEY`. `victor bench voice --stt` measures them and
spends real audio quota doing it. Windows numbers are also outstanding — the
figures above are macOS.

---

## P2 · Agent Core ✅

**Built**
- `tools/base.py` — tool contract, registry, and the interceptor seam.
- `tools/shell.py`, `tools/git.py` — `shell`, `read_file`, `git`.
- `agent/llm.py` — chat client for the text tier, with token reconciliation
  and fall-through on 429.
- `agent/loop.py` — the ReAct loop and its budgets.
- `agent/prompts.py` — system prompt, plus the STT vocabulary bias.
- CLI: `victor do`, `victor converse`, `victor tools`.

**Exit gate** — "what branch am I on and what changed?" drove three real `git`
invocations (`rev-parse`, `log`, `status`) against this repository in sequence,
each result fed back before the next decision, answered in 4 steps and 580
tokens. ✅

### Budgets are the design

A free tier of ~1,000 requests a day and 8,000 tokens a minute means an
unbounded loop does not hang, it ends the day. Three limits, all reported in
the result rather than silently applied:

- **Step cap** — default 8 think-act cycles.
- **Token budget** — default 20,000 per task.
- **Repetition guard** — identical consecutive tool calls are refused with a
  message telling the model to try something else. A stuck model will
  otherwise spend the entire allowance repeating one mistake, and this is the
  single cheapest protection against that.

Tool output is truncated at the tool boundary, head and tail kept, because one
noisy `git log` can exceed the per-minute token budget on its own. The failure
mode without it is not a clean error — it is the agent losing the earlier half
of its own conversation.

### A 429 outranks the ledger

Declared free-tier numbers are conservative but providers change them silently.
When the API returns 429, the client marks that model spent for the day and
retries down the chain *mid-task*. Covered by test: first request goes to
`gpt-oss-120b`, the 429 sends the second to `llama-3.3-70b`, and the run still
answers.

### Model mistakes are results, not exceptions

Unknown tool names, malformed JSON arguments, wrong kwargs, blocked calls — all
come back as failed `ToolResult`s the model can read and correct on the next
step. An exception would end a run that was one message away from recovering.

### Still missing, and stated in the product

`victor tools` and `victor doctor` both say out loud that the P3 interceptor
does not exist yet and that mutating tools currently run behind only a
denylist. That denylist (`rm -rf`, `mkfs`, `dd` to a device, fork bombs,
curl-pipe-to-shell, force push) is a guard against an obvious model mistake,
**not** a security boundary, and it is documented as such in
`src/victor/tools/shell.py`.

---

## P3 · Safety & Reversibility

The gate for everything that touches the machine. Built *before* actuation, not
bolted on after.

**Build**
- Interceptor classifying every action `safe` / `confirm` / `deny`.
- Dry-run mode that prints the exact command and stops.
- Spoken confirmation for destructive actions.
- Kill switch that aborts mid-task from a global hotkey.
- Action journal with undo recipes where an inverse exists.

**Exit gate** — `rm -rf` is refused without confirmation, confirmed once spoken,
recorded in the journal, and the kill switch stops a running task inside 200 ms.

---

## P4 · Screen Perception *(parallelizable)*

**Build** — Windows UI Automation tree reader producing indexed, filtered
elements with names, control types and bounding boxes. Screen capture via `mss`.
Vision only as a fallback for surfaces with no usable tree (canvas apps, remote
desktops, images).

**Exit gate** — dump the tree of File Explorer, Edge, Settings and VS Code in
under 100 ms each, with every actionable element addressable by index.

Read-only by construction: this phase cannot click anything.

---

## P5 · Desktop Actuation

**Build** — click, type, focus, scroll, driven by UIA element handles rather
than coordinates. Every action routed through P3's interceptor.

**Exit gate** — "open Settings and turn on dark mode" completes hands-free,
with no coordinate guessing in the trace.

---

## P6 · Memory

**Build** — FAISS index over `fastembed` vectors. Error tracebacks and their
eventual fixes are captured automatically from the agent loop, not typed in by
hand. Recall injects prior fixes into context when a traceback resembles one
already seen.

**Exit gate** — hit the same error twice; the second run cites the first fix.

---

## P7 · Scout

**Build** — GitHub portfolio gap analysis reusing P6's embedding stack.

**Exit gate** — point it at a profile, get back specific, non-generic gaps.

---

## P8 · Surface & Ship

**Build** — HUD, the real benchmark table, test pass, demo recording.

**Exit gate** — README's latency numbers are measured on this machine and match
what the demo visibly does.

---

## Deliberately cut

Each of these was considered and dropped, with the reason, so they don't get
silently re-added later:

- **Always-on wake word.** 250 vision requests/day cannot support a continuous
  loop, and a wake word without a safety layer is how you get an agent that
  acts on the television. Push-to-talk instead.
- **Pixel-coordinate clicking as the primary path.** It is the thing this
  project exists to avoid. Vision stays a fallback.
- **Multi-agent planner/critic split.** Doubles token spend against a 1,000
  req/day budget to fix a problem a single-model ReAct loop has not yet shown.
- **Cross-platform support.** UI Automation is a Windows API. macOS AX and
  Linux AT-SPI would be a second perception backend, not a port.
- **A fine-tuned local model.** No GPU is part of the promise. Everything is
  ONNX or a free API.
- **Browser automation via CDP.** Would work better than UIA inside the browser,
  but it is a second actuation stack serving one app family.

---

## Development environment

Victor targets Windows. The core — config, quota, routing, tracing, agent loop,
memory — is platform-neutral and developed and tested on macOS and Linux too.
P4/P5 import `uiautomation`, which is Windows-only and declared behind a
`sys_platform == 'win32'` marker; on other platforms `victor doctor` reports
desktop control as unavailable rather than pretending.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

**macOS + Homebrew Python caveat.** Homebrew's `python@3.13`/`3.14` link
`pyexpat` against keg-only `expat`, which breaks `pip` with a missing
`_XML_SetAllocTrackerActivationThreshold` symbol. Work around it with
`export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`. Separately, macOS may
set the `UF_HIDDEN` flag on files in `site-packages`, and Python 3.13 skips
hidden `.pth` files — which silently breaks editable installs. `pytest` is
configured with `pythonpath = ["src"]` so the suite is immune; for the CLI, run
`PYTHONPATH=src .venv/bin/python -m victor` or `chflags nohidden` the `.pth`.
