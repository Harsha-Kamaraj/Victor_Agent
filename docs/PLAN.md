# Victor Agent — 6-Day Implementation Plan

## Context

Build **Victor**, an autonomous voice-driven Computer-Use & Developer AI agent, from scratch at `C:\Users\jssps\victor-agent` (greenfield — nothing exists yet).

**Locked decisions:**
- Budget: **strictly $0** — free tiers only, no credit card
- Purpose: **portfolio / resume showcase** — optimize for a credible demo + README, not general robustness
- Desktop control: **UIA accessibility tree first, VLM fallback**
- Timeline: **6 focused days**
- Scope: voice, dev/shell + HITL safety, desktop GUI, RAG memory, plus `victor scout` as a secondary feature
- Cross-cutting: session tracing + replay, global kill switch + reversible action journal

**Verified environment** (checked, not assumed): Python 3.14.5, Node 24.11, git 2.51, Intel Core Ultra 7 255H (16 cores), 31.5 GB RAM, Intel Arc 140T iGPU (**no CUDA** — CPU/ONNX only). All wheels resolve on 3.14: `faiss-cpu` 1.14.3, `fastembed` 0.8.0, `onnxruntime` 1.28.0, `google-genai` 2.14.0, `groq` 1.6.0, `mss` 10.2.0, `uiautomation` 2.0.29, `piper-tts` 1.6.0, `webrtcvad` 2.0.10, `typer`, `pydantic` 2.13. No `pipx`/`uv`, but `Python314\Scripts` is on PATH. No API keys currently set.

---

## Honest note on the timeline

The original phase estimates summed to 11–17 days of solo human work. Hitting 6 comes from two places, and only one of them is free:

1. **Code generation speed** — scaffolding, API clients, schemas, and CLI wiring are fast to produce. Real, but it only covers maybe half the gap.
2. **Scope cuts** — the rest. These are listed explicitly under "What 6 days costs you" below. Nothing here is achieved by optimism.

**Day 4 is the schedule risk.** UIA behaviour on real applications is unpredictable and is the one thing that can eat a day without warning. Its mitigation is baked in: target a fixed set of known-good apps rather than "any app."

---

## Reality corrections baked into this plan

These override the original architecture diagram. Each was verified against current sources.

| Original claim | Reality | Design response |
|---|---|---|
| "Groq Whisper via WebSockets" | Groq STT is **HTTP file upload only**; no streaming socket exists | VAD segments utterances locally, POSTs each chunk. WebSockets are internal (HUD ↔ core) only |
| "Sub-500ms voice pipeline" | Voice→shell ≈ 600–900ms. Voice→**vision**→act→speak ≈ **2–6s** | Publish a real measured latency table. Never claim 500ms end-to-end |
| "Continuous live screen capture" | Gemini Flash free tier ≈ **10 RPM / ~250 requests per day** (Google now sets these per-account and cut free quotas 50–80% in Dec 2025) | **On-demand capture only.** Hard quota ledger. UIA handles most actions at zero API cost |
| "edge-tts for voice output" | Recurring **403 blocks** from Microsoft through 2026 | **Piper** (local ONNX neural TTS) primary, `pyttsx3`/SAPI5 fallback. Fully offline |
| VLM predicts click coordinates | Pixel-coordinate clicking on Windows is unreliable (off-by-30px, wrong control) — this is where these projects die | **UIA gives exact element names + rects**, locally, ~20ms, free. Agent picks by ID |

---

## The $0 architecture: split-brain routing

The core engineering idea that makes a strict-free-tier build viable. **Different free tiers differ wildly in generosity — route each workload to whichever can afford it.**

| Workload | Provider | Free allowance | Why |
|---|---|---|---|
| **Text reasoning** (ReAct loop, tool choice, safety adjudication) | Groq `openai/gpt-oss-120b` | ~1,000–14,400 req/day, 30 RPM, ~300 TPS | Fast and abundant. Carries ~90% of all calls |
| **Vision** (only when UIA is insufficient) | Gemini 2.5 Flash → fallback Groq Llama-4-Scout | ~250/day → +~1,000/day | Scarcest resource. Spent only when earned |
| **STT** | Groq `whisper-large-v3-turbo` | 28,800 audio sec/day, 2,000 req/day | Separate quota pool from chat — effectively free |
| **TTS** | Piper (local ONNX) | Unlimited | Offline, ~100ms, nothing to break |
| **Embeddings** | `fastembed` (local ONNX) | Unlimited | Offline |
| **UI perception** | Windows UIA tree | Unlimited | Local, ~20ms, zero cost |
| **Repo data** | GitHub REST API | 5,000 req/hr authenticated | Ample |

**Zero-torch stack.** `fastembed` + `piper` + `webrtcvad` are ONNX or native C, avoiding a ~2.5 GB CPU-only torch download that buys nothing without CUDA.

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

---

## Install & CLI surface

`pyproject.toml` declares a console-script entry point:

```toml
[project.scripts]
victor = "victor.cli:app"
```

**Install (one time):**
```powershell
cd C:\Users\jssps\victor-agent
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
victor doctor
```

**Global access without activating the venv.** No `pipx`/`uv` present, but `C:\Users\jssps\AppData\Local\Programs\Python\Python314\Scripts` is already on PATH. `victor install-shim` writes a `.cmd` there pointing at the venv interpreter, so `victor ...` works from any PowerShell window while dependencies stay isolated.

| Command | Does |
|---|---|
| `victor run` | Start the agent — push-to-talk voice loop + HUD |
| `victor run --text "..."` | Skip the mic, drive by typed prompt (**used constantly during the build**) |
| `victor run --dry-run` | Full loop, nothing executes |
| `victor doctor` | Verify keys, mic, speakers, UIA access, quota |
| `victor bench [--voice]` | Measured p50/p95 latency table |
| `victor index <path>` / `victor recall "<q>"` | FAISS memory ingest / query |
| `victor scout --user <handle>` | GitHub portfolio gap report |
| `victor sessions` / `victor replay <id>` | List / step through recorded traces |
| `victor undo [--last N]` | Revert file ops from the action journal |
| `victor quota` | Today's remaining free-tier budget per provider |
| `victor install-shim` | Put `victor` on the global PATH |

**Kill switch:** `Ctrl+Alt+Esc` (configurable) — a global hook, live whenever `victor run` is active, not a CLI command.

---

## Day 0 — Prerequisites (~30 min, your side, before Day 1)

Not a build day. Do this first or Day 1 stalls.

1. **Groq API key** — console.groq.com, free, no card
2. **Gemini API key** — aistudio.google.com, free, no card. While there, **note your actual rate limits** on the AI Studio rate-limit page; they're per-account and the plan's numbers are typical, not guaranteed
3. **GitHub PAT** — classic token, `public_repo` + `read:user` scope
4. Confirm a working mic and speakers

Keys go in `.env` (git-ignored). `victor doctor` validates all three on Day 1.

---

## Day 1 — Foundation + voice spine

**Goal:** you speak, Victor speaks back, with real measured latency.

**Build:**
- `pyproject.toml`, venv, `.gitignore`, `.env.example`, editable install, `install-shim`
- `config.py` — pydantic-settings; every free-tier limit lives here as config, never hardcoded
- `llm/budget.py` — JSON ledger at `~/.victor/quota.json`; per-provider daily counters, midnight reset, `can_spend(provider) -> bool`
- `llm/router.py` — split-brain routing with fallback chain
- **`trace/recorder.py`** — built now, not later. Every run appends to `logs/sessions/<id>.jsonl`: utterance, ReAct steps, tool calls + args, safety verdicts, per-step latency, provider/tokens/quota. Benchmarks, replay, debugging, and the README's cost numbers all read this one format. Retrofitting it across five modules later costs far more than writing it today
- `voice/mic.py` — `sounddevice`, 16 kHz mono, 20 ms frames, ring buffer
- `voice/vad.py` — `webrtcvad`, ~300 ms hangover, emits complete utterances
- `voice/stt.py` — Groq `whisper-large-v3-turbo`, retry + backoff
- `voice/tts.py` — Piper ONNX (`en_US-lessac-medium`, one-time download), `pyttsx3` fallback
- `voice/hotkey.py` — push-to-talk (default; avoids hot-mic quota burn)
- `cli.py` — `doctor`, `bench`, `run` skeleton

**Deliverable:** `victor doctor` all-green from any terminal; `victor bench --voice` prints p50/p95 per leg (expect STT 300–600 ms, TTS ~100 ms).

**Watch for:** VAD threshold tuning against your room noise is the likely time sink. Timebox it — push-to-talk makes it non-blocking.

---

## Day 2 — Agent core + shell tools + HITL safety

**Goal:** voice → shell command, with destructive actions blocked.

**Build:**
- `agent/loop.py` — ReAct loop on Groq structured/JSON tool calling; max-step cap, full transcript, cancel-on-interrupt
- `agent/tools.py` — registry auto-generating JSON schemas from pydantic models
- `agent/prompts.py` — system prompt, tool descriptions, few-shot examples
- `dev/shell.py` — `subprocess` with timeout, captured streams, cwd management
- `dev/git_tools.py` — status, diff, branch, commit, push (push always CONFIRM-gated)
- **`safety/interceptor.py`** — three layers, in order:
  1. **DENY** — `rm -rf /`, `format`, `diskpart`, fork bombs, writes to `C:\Windows`, force-push to default branch
  2. **CONFIRM** — any delete, admin elevation, network writes, `git push`, repo deletion, >N file mutations
  3. **LLM adjudication** (Groq, cheap) for anything unmatched — **fails closed to CONFIRM** on error or timeout
- `safety/dryrun.py` — renders what *would* happen (files matched, diff preview), speaks a summary first

**Deliverable:** *"Victor, delete every log file in Downloads"* → speaks the file count, waits for spoken "yes".

**First hour spike:** confirm Groq `openai/gpt-oss-120b` handles strict JSON tool-calling well enough. If not, fall back to Llama-4-Scout or Qwen3-32B — decide inside the first hour, don't discover it on Day 4.

---

## Day 3 — Kill switch, journal, undo + UIA foundation

**Goal:** Victor is safe to let loose, and can read any window's UI.

**Build:**
- **`safety/killswitch.py`** — the interceptor gates actions *before* they run; this stops one *already running*. Global `Ctrl+Alt+Esc` hook on its own thread sets an abort event polled by the ReAct loop, subprocess runner, and desktop executor. Kills the child process **tree**, releases held keys/mouse buttons, cancels pending API calls, speaks "stopped". `pyautogui.FAILSAFE` (mouse to corner) is a second, dependency-free trigger
- **`safety/journal.py`** — deletes become moves to `~/.victor/trash/<session>/`; overwrites snapshot the original first. Each entry records its inverse, so `victor undo` replays backward. 7-day retention, size-capped
- **`desktop/uia.py`** — walk the UI Automation tree of the focused window, filter to interactable controls, emit a compact numbered list; cache per window handle, invalidate on focus change

```
[3]  Button  "Compose"      (24,180)-(140,220)
[7]  Edit    "Search mail"  (300,60)-(900,100)
[12] Button  "Settings"     (1400,60)-(1440,100)
```

- `trace/replay.py` + `victor sessions` / `victor replay`
- `tests/test_safety.py` — every DENY rule blocks; unmatched input fails closed; abort reaps the process tree; journal round-trip is byte-identical

**Deliverable:** start a long task, hit `Ctrl+Alt+Esc` → stops in ~200 ms, nothing orphaned. Delete files → `victor undo` restores them. `python -m victor.desktop.uia --dump` lists the focused window's elements with zero API calls.

**Verify in the first 30 minutes:** that the global hotkey registers without elevation. **A kill switch that needs admin is not a kill switch** — if `keyboard` misbehaves, swap to `pynput` immediately.

---

## Day 4 — Desktop control end-to-end ⚠️ schedule risk

**Goal:** *"Victor, open Gmail and search for invoices from last month"* works.

**Build:**
- `desktop/actions.py` — `click_element(id)`, `type_text`, `hotkey`, `scroll`, `focus_window`, `open_app`, all on UIA handles, never raw pixels; every file-touching action routed through the journal
- `desktop/capture.py` — `mss` capture, downscale to ~768 px longest edge, **perceptual-hash cache** so an unchanged screen never re-bills a VLM call
- `desktop/vision.py` — **fallback only**, when the tree returns nothing useful (canvas, game, poor Electron tree). Annotates the screenshot with numbered boxes over UIA rects (Set-of-Mark) and asks the VLM to pick a **number, not a coordinate**. Checks `budget.can_spend()` first; degrades gracefully with a spoken *"I'm out of vision quota for today"*
- Wire desktop tools into the agent registry; instrument API calls per task

**Deliverable:** two multi-step GUI tasks completed by voice, with the trace showing how many steps used **zero** API calls.

**Scope guard — this is what keeps Day 4 to one day:** target **File Explorer, Edge/Chrome, Windows Settings, VS Code**. These have good trees. Do not chase universal app support; that is an open-ended research problem, not a day of work. If a target app's tree is poor, swap the app rather than fighting it.

---

## Day 5 — RAG memory + `victor scout`

**Goal:** Victor remembers past fixes, and can analyze a GitHub portfolio.

**Build:**
- `rag/embed.py` — `fastembed` with `BAAI/bge-small-en-v1.5` (~130 MB ONNX, CPU)
- `rag/store.py` — FAISS `IndexFlatIP` + SQLite sidecar for metadata/text
- `rag/ingest.py` — `victor index <path>` chunks project files; **auto-capture hook**: when a shell command exits non-zero and a later one succeeds, store the `(traceback → fix)` pair
- `rag/recall.py` — top-k retrieval injected into agent context on every error
- `scout/github.py` — authenticated REST client; user repos, languages, topics, READMEs
- `scout/corpus.py` — comparison corpus via **GitHub Search API** (`stars:>N pushed:>date` across topics). *Honest framing: GitHub has no official "trending" API — this is a heuristic and the README says so plainly*
- `scout/analyze.py` — reuses `rag/embed.py` and `rag/store.py` wholesale; cosine distance, ranked gaps, **each row citing the specific repos that produced it** so the output is checkable

**Deliverable:** trigger the same traceback twice — the second time Victor recalls the fix instantly, offline, zero API calls. `victor scout` prints and speaks a ranked gap report.

**Cut-line:** if Day 4 overran, `victor scout` is the first thing to drop. RAG stays — it's load-bearing for the "learns from errors" story.

---

## Day 6 — Polish, HUD, README, demo

**Goal:** it looks finished.

**Build:**
- `ui/hud.py` — minimal always-on-top status strip: state (LISTENING / THINKING / ACTING / CONFIRMING), live transcript, **live quota counter**. The quota counter *is* the story. A status strip, not a UI framework — do not rabbit-hole into PyQt
- **README** — architecture diagram, real measured benchmark table, GIFs, `victor scout` as a section (not video time), and an explicit **"What this can't do"** section. That section buys more credibility than any feature
- `victor bench` full-pipeline table, regenerated from session traces
- Round out `tests/` — safety interceptor, kill switch, journal, UIA parsing, quota rollover
- **Demo video, 90 seconds, 4 scenarios:**
  1. Voice → shell, HITL block, then `victor undo`
  2. Desktop GUI navigation, quota counter barely moving
  3. Error → RAG recall, offline
  4. Kill switch stopping a task mid-action

Session traces make recording easy: rehearse until one run is clean, then record that run.

---

## What 6 days costs you

Stated plainly so these are decisions, not surprises:

- **Targeted app support, not universal** — UIA tuned for 4 named apps. Others may work; they aren't guaranteed
- **Demo-grade, not production-grade** — the four scenarios are solid; the long tail isn't hardened
- **Tests concentrated on safety** — interceptor, kill switch, journal, quota. Not broad coverage
- **Minimal HUD** — status strip only
- **`victor scout` is single-shot** — no caching or iterative refinement
- **No eval harness** — deferred; session tracing (Day 1) is its prerequisite, so it can be added any time later
- **No streaming TTS** — ~300 ms of felt latency left on the table
- **No wake word** — push-to-talk only

---

## Verification

```powershell
victor doctor                          # Day 1 — env, keys, devices, quota green
victor bench --voice                   # Day 1 — p50/p95 per leg
pytest tests/test_safety.py -v         # Day 3 — DENY blocks; unmatched fails closed
victor run --dry-run                   # Day 2 — full loop, nothing executes
python -m victor.desktop.uia --dump    # Day 3 — element list, zero API calls
victor index . ; victor recall "ModuleNotFoundError"   # Day 5
victor scout --user <handle>           # Day 5
victor sessions ; victor replay <id>   # any day — trace complete and readable
pytest -v                              # Day 6 — full suite
```

**Manual end-to-end:**
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
| **Day 4 UIA quirks eat the schedule** | **High** | Fixed target app list; swap apps rather than fight trees; Day 5 is the buffer, `scout` is the cut |
| Gemini 250/day quota hit during a heavy session | High | Split-brain routing, pHash cache, Groq vision fallback, UIA-first. Ledger makes the ceiling visible, not surprising |
| Global hotkey needs elevation → kill switch silently dead | Medium | Verify in Day 3's first 30 min; `pyautogui.FAILSAFE` is the backup trigger; swap to `pynput` if needed |
| Kill switch fires but a child process survives | Medium | Kill the process **tree**, not the direct child; asserted in tests |
| Groq model's JSON tool-calling too weak for ReAct | Medium | Day 2 first-hour spike; fall back to Llama-4-Scout / Qwen3-32B |
| VAD false triggers on background noise | Medium | Push-to-talk default; `--text` mode means voice never blocks development |
| Piper download / ONNX issue on 3.14 | Low | `pyttsx3`/SAPI5 fallback always present |
| Free-tier terms change mid-build | Medium | All limits in `config.py`; providers swappable behind `llm/router.py` |

---

## What makes this a blockbuster

Most "computer use agent" repos are flaky VLM-coordinate demos that break on the second run and quietly cost money. Victor's real differentiators:

1. **Runs on $0/day and proves it** — live quota counter in the HUD, documented split-brain routing table
2. **UIA-first hybrid control** — deterministic clicks, ~20 ms, no pixel guessing; VLM spend only when earned
3. **Honest measured benchmarks** — a real latency table beats an inflated "sub-500ms" claim every time
4. **Safety that fails closed** — DENY rules, a kill switch that stops mid-action, reversible file ops, tests proving all three
5. **Observable** — every run is a replayable trace. Almost no portfolio agent repo has this, and it's what an experienced reviewer notices first

**One-sentence pitch:** *A voice-driven computer-use agent for Windows that runs entirely on free API tiers — because it reads the accessibility tree instead of guessing pixels.*

---

## Deliberately deferred

Recorded so these stay decisions rather than oversights:

- **Eval harness (`victor eval`)** — scripted task suite producing a success-rate number. High credibility value; unblocked by Day 1's tracing, add any time after Day 6
- **Streaming TTS** — speak sentence one while the rest generates
- **Gemini Live API** — would collapse STT+VLM+TTS into one stream, but its free tier is too thin
- **Wake word** — always-hot mic burns quota and invites false triggers
- **Gemini built-in computer-use tool** — check free-tier availability later; an optional enhancement, never a dependency
