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

## P1 · Voice I/O

**Build** — mic capture, WebRTC VAD for endpointing, Groq
`whisper-large-v3-turbo` for STT, Piper ONNX for TTS. Push-to-talk first,
because always-listening is a demo liability before the safety layer exists.

**Exit gate** — speak a sentence, see the transcript, hear a spoken reply.
Publish measured p50/p95 for mic-close → first audio.

---

## P2 · Agent Core

**Build** — ReAct loop over the router's TEXT workload, a tool registry with
JSON-schema'd tools, and the first two tools: `shell` and `git`. Loop state,
step cap, and every step traced.

**Exit gate** — "what branch am I on and what changed?" runs the right tools in
sequence and answers correctly.

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
