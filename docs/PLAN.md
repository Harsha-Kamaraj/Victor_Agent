# Victor Agent — Implementation Plan

> **This document is the plan of record**, written by [@Gagan-1718](https://github.com/Gagan-1718)
> before implementation began. It is kept as authored, with phase headings ticked
> as their exit gates pass and short *as-built* notes where reality diverged.
>
> **Build status: P0–P5 complete, P6–P8 outstanding.**
> What was actually built, why it differs, and the measured numbers live in
> [BUILD-LOG.md](BUILD-LOG.md) — that file is the record of execution, this one
> is the record of intent. Read this first.
>
> Implementation lives at
> [Harsha-Kamaraj/Victor_Agent](https://github.com/Harsha-Kamaraj/Victor_Agent).

## Context

Build **Victor**, an autonomous voice-driven Computer-Use & Developer AI agent, at `C:\Users\jssps\victor-agent`.

**Locked decisions:**
- Budget: **strictly $0** — free tiers only, no credit card
- Purpose: **portfolio / resume showcase** — optimize for a credible demo + README, not general robustness
- Desktop control: **UIA accessibility tree first, VLM fallback**
- Scope: voice, dev/shell + HITL safety, desktop GUI, RAG memory, plus `victor scout` as a secondary feature
- Cross-cutting: session tracing + replay, global kill switch + reversible action journal

**Verified environment** (checked, not assumed): Python 3.14.5, Node 24.11, git 2.51, Intel Core Ultra 7 255H (16 cores), 31.5 GB RAM, Intel Arc 140T iGPU (**no CUDA** — CPU/ONNX only). All wheels resolve on 3.14: `faiss-cpu` 1.14.3, `fastembed` 0.8.0, `onnxruntime` 1.28.0, `google-genai` 2.14.0, `groq` 1.6.0, `mss` 10.2.0, `uiautomation` 2.0.29, `piper-tts` 1.6.0, `webrtcvad` 2.0.10, `typer`, `pydantic` 2.13. No `pipx`/`uv`, but `Python314\Scripts` is on PATH.

---

## How this plan is organized

Phases are **units of execution and integration**, not calendar days. Each phase is a
thing that either works or doesn't — you can stop between phases and the system is in a
coherent state.

Every phase declares:

- **Consumes** — what must already exist for this phase to be buildable
- **Exposes** — the contract later phases depend on
- **Exit gate** — a concrete, verifiable check. **Do not start the next phase until it passes.** This is the single most important discipline in the plan; it's what keeps a five-pillar project from collapsing into five half-finished ones.

Effort is marked **S / M / L** (a few hours / half a day / a full day or more) rather than
assigned to dates. The whole thing is roughly 5–6 days of focused work, but the phase is
the unit of progress — pace them however your week actually goes.

**Dependency graph:**

```
P0 Skeleton ──┬──> P1 Voice I/O ──┐
              │                   ├──> P2 Agent Core ──> P3 Safety ──┐
              │                   │                                  │
              │   (P1 optional — P2 works headless via --text)       ├──> P5 Actuation
              │                                                      │        │
              └──> P4 Perception ────────────────────────────────────┘        │
                                                                              ▼
                                                    P6 Memory ──> P7 Scout ──> P8 Ship
```

**P4 (Perception) is read-only and depends only on P0** — build it in parallel with P1–P3
if you want a change of pace, or pull it forward if voice tuning gets frustrating. It is
the only phase with that freedom; everything else is genuinely sequential.

---

## Honest note on scope

The original estimates summed to 11–17 days of solo human work. Compressing to ~6 comes
from two places, and only one is free:

1. **Code generation speed** — scaffolding, API clients, schemas, CLI wiring. Real, but covers maybe half the gap.
2. **Scope cuts** — the rest, listed explicitly under "What the compressed scope costs you." Nothing here is achieved by optimism.

**P5 is the schedule risk.** UIA behaviour on real applications is unpredictable and is the
one thing that can consume a day without warning. Mitigation is baked in: a fixed set of
known-good target apps rather than "any app."

---

## Reality corrections baked into this plan

These override the original architecture sketch. Each was verified against current sources.

| Original claim | Reality | Design response |
|---|---|---|
| "Groq Whisper via WebSockets" | Groq STT is **HTTP file upload only**; no streaming socket exists | VAD segments utterances locally, POSTs each chunk. WebSockets are internal (HUD ↔ core) only |
| "Sub-500ms voice pipeline" | Voice→shell ≈ 600–900 ms. Voice→**vision**→act→speak ≈ **2–6 s** | Publish a real measured latency table. Never claim 500 ms end-to-end |
| "Continuous live screen capture" | Gemini Flash free tier ≈ **10 RPM / ~250 requests per day** (per-account now; Google cut free quotas 50–80% in Dec 2025) | **On-demand capture only.** Hard quota ledger. UIA handles most actions at zero API cost |
| "edge-tts for voice output" | Recurring **403 blocks** from Microsoft through 2026 | **Piper** (local ONNX neural TTS) primary, `pyttsx3`/SAPI5 fallback. Fully offline |
| VLM predicts click coordinates | Pixel-coordinate clicking on Windows is unreliable (off-by-30px, wrong control) — this is where these projects die | **UIA gives exact element names + rects**, locally, ~20 ms, free. Agent picks by ID |

### Further corrections, found while building

The table above was verified before implementation. These only surfaced once code
ran, and each changed a decision above:

| Plan said | Building it showed | Response |
|---|---|---|
| `webrtcvad` 2.0.10 resolves on 3.14 | The wheel installs but **fails at import**: it does `import pkg_resources`, which setuptools 81+ removed | Victor ships its own adaptive energy VAD. WebRTC stays an opt-in backend. An unmaintained C extension is not worth putting on a voice agent's critical path |
| Use the `groq` and `google-genai` SDKs | Groq is OpenAI-compatible, so one thin `httpx` client covers chat *and* STT and swaps provider by URL | Raw `httpx`. Two fewer dependencies, and `MockTransport` makes the whole provider layer testable without a network |
| `victor = "victor.cli:app"` | Typer's `app` object cannot catch `VictorError` for a tidy exit code | `victor.cli:main`, a wrapper that maps expected failures to exit codes |
| Actuation is "click the rect centre" | Both platforms can perform the control's *own* action — `AXPress`, UIA `Invoke`/`Toggle`/`SelectionItem` — which is invisible, needs no cursor move, and cannot miss | Handles first, synthetic click only when a control offers no action. Calculator drove entirely through `AXPress`; `method` is reported so the ratio is visible |
| An index from the last `screen_read` is safe to click | A list that re-sorts between the read and the click hands index 7 to a different button | Every action re-reads the tree (~20 ms) and refuses if the label moved, naming where the target went |
| One Quartz event can carry a whole string | Text fields accept it; anything handling `keyDown:` itself takes the first character and drops the rest — Calculator typed `8*8` and showed `8` | One event per character. A silently truncated string is a worse failure than a slow one |
| A window with no listed controls has none | macOS stops reporting window geometry while the screen is locked, so every rect is empty and every element is filtered out | `session.py` detects a locked screen and a secure desktop on both platforms; snapshots carry a `note` distinguishing "nothing to click" from "nothing measurable" |
| Python 3.14 verified | True on the target Windows box. On macOS, Homebrew's 3.13/3.14 link `pyexpat` against keg-only `expat` and **pip does not work at all** | Development on macOS uses 3.13 with `DYLD_LIBRARY_PATH` set — see BUILD-LOG. Windows is unaffected |

---

## The $0 architecture: split-brain routing

The core engineering idea that makes a strict-free-tier build viable. **Free tiers differ
wildly in generosity — route each workload to whichever can afford it.**

| Workload | Provider | Free allowance | Why |
|---|---|---|---|
| **Text reasoning** (ReAct loop, tool choice, safety adjudication) | Groq `openai/gpt-oss-120b` | ~1,000–14,400 req/day, 30 RPM, ~300 TPS | Fast and abundant. Carries ~90% of all calls |
| **Vision** (only when UIA is insufficient) | Gemini 2.5 Flash → fallback Groq Llama-4-Scout | ~250/day → +~1,000/day | Scarcest resource. Spent only when earned |
| **STT** | Groq `whisper-large-v3-turbo` | 28,800 audio sec/day, 2,000 req/day | Separate quota pool from chat — effectively free |
| **TTS** | Piper (local ONNX) | Unlimited | Offline, ~100 ms, nothing to break |
| **Embeddings** | `fastembed` (local ONNX) | Unlimited | Offline |
| **UI perception** | Windows UIA tree / macOS AX tree | Unlimited | Local, ~20 ms, zero cost |
| **Repo data** | GitHub REST API | 5,000 req/hr authenticated | Ample |

**Zero-torch stack.** `fastembed` + `piper` + `webrtcvad` are ONNX or native C, avoiding a
~2.5 GB CPU-only torch download that buys nothing without CUDA.

```
[Mic] → webrtcvad → utterance .wav → Groq Whisper (STT)
                                          │
                                          ▼
                    ┌──── ReAct loop (Groq gpt-oss-120b) ────┐
                    │                                        │
             [UIA tree: free]                        [FAISS RAG recall]
                    │                                        │
          tree sufficient? ──no──> [mss capture → Gemini Flash]  ← quota ledger
                    │ yes                                     (fallback: Groq Scout)
                    ▼
          [Safety Interceptor: ALLOW / CONFIRM / DENY]   ←── kill switch aborts any stage
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   [ EXECUTE via journal ]   [ spoken confirm + dry-run preview ]
        └───────────┬───────────┘
                    ▼
          [Piper TTS] + [HUD]        ── every step appended to session trace ──
```

---

## Repo layout

```
victor-agent/
  pyproject.toml            .env.example       README.md
  victor/
    cli.py                  # typer entry point
    config.py               # pydantic-settings from .env
    bus.py                  # asyncio event bus + state machine
    llm/
      router.py             # split-brain provider routing
      budget.py             # persistent daily quota ledger
      schemas.py            # pydantic tool-call contracts
    voice/  mic.py  vad.py  stt.py  tts.py  hotkey.py
    agent/  loop.py  prompts.py  tools.py
    desktop/ uia.py  capture.py  actions.py  vision.py
    dev/    shell.py  git_tools.py
    safety/
      interceptor.py        # ALLOW / CONFIRM / DENY, pre-execution gate
      rules.py  dryrun.py
      killswitch.py         # global panic hotkey, aborts mid-action
      journal.py            # reversible file ops + undo
    trace/  recorder.py  replay.py
    rag/    store.py  embed.py  ingest.py  recall.py
    scout/  github.py  corpus.py  analyze.py
    ui/     hud.py           # minimal always-on-top status strip
  tests/                    scripts/
```

### As built

The architecture landed as planned; the naming drifted. Recorded here so the two
documents agree rather than quietly disagreeing:

| Planned | As built | Why |
|---|---|---|
| `victor/` at repo root | `src/victor/` | src-layout: tests import the installed package, not the working tree, so a broken `pyproject.toml` fails loudly instead of passing on a stale path |
| `llm/router.py`, `llm/budget.py`, `llm/schemas.py` | `providers/router.py`, `quota.py`, `providers/base.py` | The ledger is not LLM-specific — it meters audio seconds too, so it sits above the provider layer |
| `dev/shell.py`, `dev/git_tools.py` | `tools/shell.py`, `tools/git.py` | `tools/` also holds the registry and the interceptor seam, which are not dev-specific |
| `agent/tools.py` | `tools/` package | Outgrew one module once the registry, contract and safety seam were in it |
| `trace/recorder.py`, `trace/replay.py` | `tracing.py` | One file, 195 lines. Splitting it would be structure without content |
| `safety/rules.py` | `safety/classify.py` | Same role; the name says what it produces |
| `safety/dryrun.py` | folded into `safety/interceptor.py` | Dry-run is one branch of the same decision, not a separate policy |
| `voice/mic.py` | `voice/sources.py` | Holds the `AudioSource` protocol plus mic *and* array sources — the split that makes the stack testable without hardware |
| `voice/hotkey.py` | `safety/killswitch.py` | The hotkey exists to abort; it belongs with the thing it aborts |
| `bus.py` (asyncio event bus + state machine) | **not built** | The loop is synchronous and call-stack-shaped. An event bus would be indirection with no second consumer yet; revisit if the P8 HUD needs to observe live state |

Unbuilt directories from the plan — `desktop/`, `rag/`, `scout/`, `ui/` — belong to
P4–P8 and are absent rather than stubbed, so `victor doctor` can report them as
`PENDING` instead of half-present.

---

## Install & CLI surface

```toml
[project.scripts]
victor = "victor.cli:app"
```

```powershell
cd C:\Users\jssps\victor-agent
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
victor doctor
```

**Global access without activating the venv.** No `pipx`/`uv` present, but
`C:\Users\jssps\AppData\Local\Programs\Python\Python314\Scripts` is already on PATH.
`victor install-shim` writes a `.cmd` there pointing at the venv interpreter.

| Command | Does | Arrives in |
|---|---|---|
| `victor doctor` | Verify keys, mic, speakers, UIA access, quota | P0 |
| `victor quota` | Today's remaining free-tier budget per provider | P0 |
| `victor install-shim` | Put `victor` on the global PATH | P0 |
| `victor bench [--voice]` | Measured p50/p95 latency table | P1 |
| `victor run` | Start the agent — push-to-talk voice loop + HUD | P2 |
| `victor run --text "..."` | Skip the mic, drive by typed prompt (**used constantly**) | P2 |
| `victor run --dry-run` | Full loop, nothing executes | P2 |
| `victor undo [--last N]` | Revert file ops from the action journal | P3 |
| `victor sessions` / `victor replay <id>` | List / step through recorded traces | P3 |
| `victor uia --dump` | Print the focused window's element tree, zero API calls | P4 |
| `victor index <path>` / `victor recall "<q>"` | FAISS memory ingest / query | P6 |
| `victor scout --user <handle>` | GitHub portfolio gap report | P7 |

**Kill switch:** `Ctrl+Alt+Esc` (configurable) — a global hook, live whenever `victor run` is active, not a CLI command.

---

# Phases

## Prerequisites (~30 min, your side)

Not a build phase, but P0 stalls without it.

1. **Groq API key** — console.groq.com, free, no card
2. **Gemini API key** — aistudio.google.com, free, no card. While there, **note your actual rate limits** on the AI Studio rate-limit page; they're per-account and this plan's numbers are typical, not guaranteed
3. **GitHub PAT** — classic token, `public_repo` + `read:user`
4. A working mic and speakers

Keys go in `.env` (git-ignored).

---

## P0 — Skeleton & Plumbing · **M**  ✅ **shipped**

**Goal:** nothing user-visible works yet, but everything later hangs off this. Get it right and the rest is assembly.

**Consumes:** nothing.
**Exposes:** `config`, `budget.can_spend()`, `router.complete()`, `trace.record()`, a CLI that runs.

**Build:**
- `pyproject.toml`, venv, editable install, `install-shim`
- `config.py` — pydantic-settings. **Every free-tier limit is config, never hardcoded** — these numbers change without notice
- `llm/budget.py` — JSON ledger at `~/.victor/quota.json`; per-provider daily counters, midnight reset, `can_spend(provider) -> bool`
- `llm/router.py` — split-brain routing with fallback chain. **The single choke point for every API call**, so quota accounting can't be bypassed by a later module
- `llm/schemas.py` — pydantic tool-call contracts
- **`trace/recorder.py`** — built now, not later. Every run appends to `logs/sessions/<id>.jsonl`: utterance, ReAct steps, tool calls + args, safety verdicts, per-step latency, provider/tokens/quota. Benchmarks, replay, debugging, and the README's cost numbers all read this one format. Retrofitting instrumentation across five modules later costs far more than writing it now
- `bus.py` — asyncio event bus + state machine (IDLE / LISTENING / THINKING / ACTING / CONFIRMING / SPEAKING)
- `cli.py` — `doctor`, `quota`, `install-shim`, stubs for the rest

**Exit gate:** `victor doctor` runs from a fresh terminal and prints an all-green table — three keys valid, mic and speakers enumerated, UIA reachable, quota ledger readable. `victor quota` shows today's budget.

---

## P1 — Voice I/O · **L**  ✅ **shipped**

**Goal:** you speak, Victor speaks back. A closed loop with no intelligence in it yet.

**Consumes:** P0 (config, router, trace).
**Exposes:** `listen() -> str`, `speak(text)`. Everything downstream treats voice as a solved I/O layer and never touches audio again.

**Build:**
- `voice/mic.py` — `sounddevice`, 16 kHz mono, 20 ms frames, ring buffer
- `voice/vad.py` — `webrtcvad`, ~300 ms hangover, emits complete utterances
- `voice/stt.py` — Groq `whisper-large-v3-turbo`, retry + backoff
- `voice/tts.py` — Piper ONNX (`en_US-lessac-medium`, one-time download), `pyttsx3`/SAPI5 fallback
- `voice/hotkey.py` — push-to-talk (the default; a hot mic burns quota and invites false triggers)
- `victor bench --voice`

**Exit gate:** speak a sentence, hear it echoed back. `victor bench --voice` prints p50/p95 per leg — expect STT 300–600 ms, TTS ~100 ms. Numbers come from real traces, not stopwatch guesses.

**Watch for:** VAD threshold tuning against your room noise is the likely time sink here. Timebox it — push-to-talk means it never blocks progress, and `--text` mode (P2) means voice is never on the critical path for development.

---

## P2 — Agent Core & Tool Execution · **L**  ✅ **shipped**

**Goal:** Victor first becomes an *agent* — it reasons, picks a tool, and runs it.

**Consumes:** P0 (router, trace). P1 optional — `--text` mode works headless.
**Exposes:** the tool registry. **Every later capability registers here**, which is why the registry contract matters more than any individual tool.

**Build:**
- `agent/loop.py` — ReAct loop on Groq structured/JSON tool calling; max-step cap, full transcript, cancel-on-interrupt
- `agent/tools.py` — registry auto-generating JSON schemas from pydantic models
- `agent/prompts.py` — system prompt, tool descriptions, few-shot examples
- `dev/shell.py` — `subprocess` with timeout, captured streams, cwd management
- `dev/git_tools.py` — status, diff, branch, commit, push
- `victor run`, `--text`, `--dry-run`

**Exit gate:** `victor run --text "list the files in my Downloads folder and tell me the largest"` completes a multi-step ReAct loop and answers correctly. Then the same by voice.

**Spike this first — before writing the loop:** confirm Groq `openai/gpt-oss-120b` handles strict JSON tool-calling reliably. If it doesn't, fall back to Llama-4-Scout or Qwen3-32B. A 30-minute check that prevents discovering a foundational problem three phases later.

---

## P3 — Safety & Reversibility · **M**  ✅ **shipped**

**Goal:** Victor becomes safe to actually let loose. This is not a feature — it's a wrapper around everything P2 can do, and everything P5 will add.

**Consumes:** P2 (wraps the tool registry).
**Exposes:** an execution gate every tool call passes through, plus abort and undo.
**P5 depends on this existing first.** Do not give an agent mouse and keyboard control before the kill switch works.

**Build:**
- **`safety/interceptor.py`** — three layers, evaluated in order:
  1. **DENY** — `rm -rf /`, `format`, `diskpart`, fork bombs, writes to `C:\Windows`, force-push to a default branch
  2. **CONFIRM** — any delete, admin elevation, network writes, `git push`, repo deletion, >N file mutations
  3. **LLM adjudication** (Groq, cheap) for anything unmatched — **fails closed to CONFIRM** on error or timeout
- `safety/dryrun.py` — renders what *would* happen (files matched, diff preview) and speaks a summary before executing
- **`safety/killswitch.py`** — the interceptor gates actions *before* they run; this stops one *already running*. Global `Ctrl+Alt+Esc` hook on its own thread sets an abort event polled by the ReAct loop, subprocess runner, and desktop executor. Kills the child process **tree**, releases held keys/mouse buttons, cancels pending API calls, speaks "stopped". `pyautogui.FAILSAFE` (mouse to a screen corner) is a second, dependency-free trigger
- **`safety/journal.py`** — deletes become moves to `~/.victor/trash/<session>/`; overwrites snapshot the original first. Each entry records its inverse, so `victor undo` replays backward. 7-day retention, size-capped
- `trace/replay.py` + `victor sessions` / `victor replay`
- `tests/test_safety.py` — the highest-value tests in the repo

**Exit gate:** all four must pass.
1. *"Delete every log file in Downloads"* → speaks the file count, waits for a spoken "yes"
2. Long-running task + `Ctrl+Alt+Esc` → stops in ~200 ms, no orphaned processes, no stuck modifier keys
3. Delete files, then `victor undo` → restored byte-identical
4. `pytest tests/test_safety.py` green — every DENY rule blocks, unmatched input fails closed, abort reaps the process tree

**Verify in the first 30 minutes:** that the global hotkey registers **without elevation**. A kill switch that needs admin is not a kill switch. If `keyboard` misbehaves, swap to `pynput` immediately.

---

## P4 — Screen Perception · **L** *(parallelizable — needs only P0)*  ✅ **shipped**

**Goal:** Victor can *see*. Read-only, so it's safe to build and test in isolation at any point.

**Consumes:** P0 only.
**Exposes:** `get_elements() -> list[Element]` and `capture()`. P5 consumes both.

**Build:**
- **`desktop/uia.py`** — walk the UI Automation tree of the focused window, filter to interactable controls, emit a compact numbered list; cache per window handle, invalidate on focus change

```
[3]  Button  "Compose"      (24,180)-(140,220)
[7]  Edit    "Search mail"  (300,60)-(900,100)
[12] Button  "Settings"     (1400,60)-(1440,100)
```

- `desktop/capture.py` — `mss` capture, downscale to ~768 px longest edge, **perceptual-hash cache** so an unchanged screen never re-bills a VLM call
- `desktop/vision.py` — **fallback only**, for surfaces with no usable tree (canvas, games, poor Electron trees). Annotates the screenshot with numbered boxes over UIA rects (Set-of-Mark) and asks the VLM to pick a **number, not a coordinate**. Checks `budget.can_spend()` first; degrades gracefully with a spoken *"I'm out of vision quota for today"*

**Exit gate:** `victor uia --dump` prints a usable element list for each target app — File Explorer, Edge/Chrome, Windows Settings, VS Code — with **zero API calls**. Separately, force the vision fallback on a canvas surface and confirm it returns a valid element choice and decrements the quota ledger.

---

## P5 — Desktop Actuation · **L** ⚠️ *schedule risk — the big integration*  ✅ **shipped**

**Goal:** *"Victor, open Gmail and search for invoices from last month"* works. This is where P2, P3, and P4 meet.

**Consumes:** P2 (registry), P3 (safety gate — **mandatory**, not optional), P4 (perception).
**Exposes:** the demo.

**Build:**
- `desktop/actions.py` — `click_element(id)`, `type_text`, `hotkey`, `scroll`, `focus_window`, `open_app`. All operate on UIA handles, **never raw pixels**; every file-touching action routes through the journal
- Register desktop tools in the P2 registry, behind the P3 interceptor
- Instrument API calls per task so the trace shows the zero-cost ratio

**Exit gate:** two multi-step GUI tasks completed by voice, end to end, with `victor replay` showing how many steps used **zero** API calls. The kill switch must still abort cleanly mid-click.

**Scope guard — this is what keeps P5 from swallowing the project:** target **File Explorer, Edge/Chrome, Windows Settings, VS Code**. These have good trees. Do not chase universal app support — that's an open-ended research problem, not a phase. If a target app's tree is poor, **swap the app rather than fight it**.

---

## P6 — Memory · **M**

**Goal:** Victor remembers past fixes and stops repeating diagnostic work.

**Consumes:** P2 (hooks the shell error path), P0 (trace).
**Exposes:** `recall(query)` injected into agent context; the embedding stack P7 reuses.

**Build:**
- `rag/embed.py` — `fastembed` with `BAAI/bge-small-en-v1.5` (~130 MB ONNX, CPU)
- `rag/store.py` — FAISS `IndexFlatIP` + SQLite sidecar for metadata/text
- `rag/ingest.py` — `victor index <path>` chunks project files. **Auto-capture hook:** when a shell command exits non-zero and a later one succeeds, store the `(traceback → fix)` pair. This is what makes the memory grow by itself instead of needing to be curated
- `rag/recall.py` — top-k retrieval injected into agent context on every error

**Exit gate:** trigger the same traceback twice. The second time Victor recalls the prior fix instantly, **offline, with zero API calls** — confirmed in the trace.

---

## P7 — Scout · **S**

**Goal:** GitHub portfolio gap analysis. A secondary feature, deliberately — not a second product.

**Consumes:** P6 (embedding + store, reused wholesale — no new infrastructure).
**Exposes:** `victor scout`.

**Build:**
- `scout/github.py` — authenticated REST client; user repos, languages, topics, READMEs
- `scout/corpus.py` — comparison corpus via **GitHub Search API** (`stars:>N pushed:>date` across topics). *Honest framing: GitHub has no official "trending" API — this is a heuristic, and the README says so plainly rather than dressing it up as science*
- `scout/analyze.py` — cosine distance, ranked gaps, **each row citing the specific repos that produced it** so the output is checkable rather than vague

**Exit gate:** `victor scout --user <handle>` prints and speaks a ranked gap report where every row names its supporting evidence.

**Cut-line:** if P5 overran, **this is the first thing to drop.** P6 stays — it carries the "learns from its errors" story.

---

## P8 — Surface & Ship · **M**

**Goal:** it looks finished, and every claim is backed by a number.

**Consumes:** everything. Reads P0's traces for all published figures.

**Build:**
- `ui/hud.py` — minimal always-on-top status strip: state, live transcript, **live quota counter**. The quota counter *is* the story. A status strip, not a UI framework — do not rabbit-hole into PyQt
- **README** — architecture diagram, real measured benchmark table, GIFs, and an explicit **"What this can't do"** section. That section buys more credibility than any feature
- `victor bench` full-pipeline table, regenerated from session traces
- Round out `tests/` — interceptor, kill switch, journal, UIA parsing, quota rollover
- **Demo video, 90 seconds, 4 scenarios:** (1) voice → shell, HITL block, then `victor undo`; (2) desktop GUI navigation with the quota counter barely moving; (3) error → RAG recall, offline; (4) kill switch stopping a task mid-action

Session traces make recording easy: rehearse until one run is clean, then record that run.

**Exit gate:** a stranger can clone the repo, follow the README, and reach `victor doctor` all-green without asking you anything.

---

## What the compressed scope costs you

Stated plainly so these stay decisions, not surprises:

- **Targeted app support, not universal** — UIA tuned for 4 named apps. Others may work; not guaranteed
- **Demo-grade, not production-grade** — the four scenarios are solid; the long tail isn't hardened
- **Tests concentrated on safety** — interceptor, kill switch, journal, quota. Not broad coverage
- **Minimal HUD** — status strip only
- **`victor scout` is single-shot** — no caching or iterative refinement
- **No eval harness** — deferred; P0's tracing is its prerequisite, so it can be added any time later
- **No streaming TTS** — ~300 ms of felt latency left on the table
- **No wake word** — push-to-talk only

---

## Verification

Per-phase gates, cumulative — later phases must not break earlier ones:

```powershell
victor doctor                          # P0 — env, keys, devices, quota green
victor quota                           # P0
victor bench --voice                   # P1 — p50/p95 per leg
victor run --text "..."                # P2 — multi-step ReAct completes
victor run --dry-run                   # P2 — full loop, nothing executes
pytest tests/test_safety.py -v         # P3 — DENY blocks; unmatched fails closed
victor undo --last 1                   # P3 — byte-identical restore
victor uia --dump                      # P4 — element list, zero API calls
victor index . ; victor recall "ModuleNotFoundError"   # P6
victor scout --user <handle>           # P7
victor sessions ; victor replay <id>   # any phase — trace complete and readable
pytest -v                              # P8 — full suite
```

**Manual end-to-end (run after P5, repeat after P8):**
1. *"Create a folder called victor-test on my desktop"* → executes, confirms by voice
2. *"Delete everything in my Documents folder"* → **must** stop, request spoken confirmation, show a dry-run count
3. **Kill switch:** multi-step task + `Ctrl+Alt+Esc` → stops ~200 ms, no orphaned processes, no stuck modifier keys
4. **Undo:** delete, then `victor undo` → files restored byte-identical
5. Kill network mid-task → degrades gracefully, speaks the failure, no crash
6. Exhaust Gemini quota deliberately → falls back to Groq vision, then UIA-only, announcing each step
7. All of the above reproduce under `victor replay` — **if a failure isn't reproducible from its trace, the tracing is incomplete**

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **P5 UIA quirks eat the schedule** | **High** | Fixed target app list; swap apps rather than fight trees; P7 is the designated cut |
| Gemini ~250/day quota hit during a heavy session | High | Split-brain routing, pHash cache, Groq vision fallback, UIA-first. The ledger makes the ceiling visible instead of surprising |
| Global hotkey needs elevation → kill switch silently dead | Medium | Verify in P3's first 30 min; `pyautogui.FAILSAFE` is the backup trigger; swap to `pynput` if needed |
| Kill switch fires but a child process survives | Medium | Kill the process **tree**, not the direct child; asserted in tests |
| Groq model's JSON tool-calling too weak for ReAct | Medium | P2 opening spike; fall back to Llama-4-Scout / Qwen3-32B |
| VAD false triggers on background noise | Medium | Push-to-talk default; `--text` mode means voice never blocks development |
| Piper download / ONNX issue on 3.14 | Low | `pyttsx3`/SAPI5 fallback always present |
| Free-tier terms change mid-build | Medium | All limits in `config.py`; providers swappable behind `llm/router.py` |

---

## What makes this a blockbuster

Most "computer use agent" repos are flaky VLM-coordinate demos that break on the second
run and quietly cost money. Victor's real differentiators:

1. **Runs on $0/day and proves it** — live quota counter in the HUD, documented split-brain routing table
2. **UIA-first hybrid control** — deterministic clicks, ~20 ms, no pixel guessing; VLM spend only when earned
3. **Honest measured benchmarks** — a real latency table beats an inflated "sub-500ms" claim every time
4. **Safety that fails closed** — DENY rules, a kill switch that stops mid-action, reversible file ops, tests proving all three
5. **Observable** — every run is a replayable trace. Almost no portfolio agent repo has this, and it's what an experienced reviewer notices first

**One-sentence pitch:** *A voice-driven computer-use agent for Windows that runs entirely on free API tiers — because it reads the accessibility tree instead of guessing pixels.*

---

## Deliberately deferred

Recorded so these stay decisions rather than oversights:

- **Eval harness (`victor eval`)** — scripted task suite producing a success-rate number. High credibility value; unblocked by P0's tracing, so it can be added any time after P8
- **Streaming TTS** — speak sentence one while the rest generates
- **Gemini Live API** — would collapse STT+VLM+TTS into one stream, but its free tier is too thin
- **Wake word** — an always-hot mic burns quota and invites false triggers
- **Gemini built-in computer-use tool** — check free-tier availability later; an optional enhancement, never a dependency
