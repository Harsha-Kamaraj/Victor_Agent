# Victor Agent

A voice-driven computer-use agent for **Windows and macOS** that runs entirely on free API tiers — because it reads the accessibility tree instead of guessing pixels.

**Status: all eight phases complete.** 835 tests, and `victor doctor` reports what is genuinely unavailable on your machine rather than a green tick for something that does not work.

Two documents, deliberately separate: [docs/PLAN.md](docs/PLAN.md) is the plan of record — what was intended, why, and what was deliberately cut. [docs/BUILD-LOG.md](docs/BUILD-LOG.md) is what actually happened, including the decisions that changed during implementation and the measured numbers.

## What it will do

Victor listens, looks at your screen, and acts — hands-free.

- **Everyday desktop** — open apps, navigate UI, drive browser tasks
- **Developer workflows** — run shell commands, inspect and fix on-screen code, manage git
- **Remembers** — a local FAISS index of past error tracebacks and their fixes
- **Refuses to do anything stupid** — destructive commands are gated behind spoken confirmation, a dry-run preview, and a kill switch

## Why this is different

Most computer-use agent projects ask a vision model to predict click coordinates. On a real Windows desktop that means off-by-30px clicks, wrong buttons, and silent failures — and it burns a paid API call on every single step.

Victor reads the operating system's own accessibility tree instead — UI Automation on Windows, the Accessibility API on macOS. That gives exact element names and bounding boxes, locally, in tens of milliseconds, for free:

```
[3]  Button  "Compose"      (24,180)-(140,220)
[7]  Edit    "Search mail"  (300,60)-(900,100)
[12] Button  "Settings"     (1400,60)-(1440,100)
```

The agent picks element `[3]`. No pixel guessing, no API call. The vision model is a fallback, used only for surfaces with no usable tree.

## The $0 architecture: split-brain routing

Free tiers differ wildly in generosity, so each workload is routed to whichever provider can actually afford it. Vision is by far the scarcest resource, so the design spends it last.

| Workload | Provider | Free allowance |
| --- | --- | --- |
| Text reasoning (ReAct loop, tool choice) | Groq `gpt-oss-120b` | ~1,000–14,400 req/day |
| Vision (fallback only) | Gemini Flash → Groq Llama-4-Scout | ~250/day → +~1,000/day |
| Speech-to-text | Groq `whisper-large-v3-turbo` | 28,800 audio sec/day |
| Text-to-speech | Piper (local ONNX) | unlimited, offline |
| Embeddings | fastembed (local ONNX) | unlimited, offline |
| UI perception | OS accessibility tree | unlimited, local |

The whole stack is ONNX or native C — no torch, no CUDA, no GPU required.

Routing is not a diagram; it is code you can interrogate:

```console
$ victor route vision
   gemini:gemini-flash-latest daily request limit reached (250/250)
     Primary vision. Scarcest resource in the stack - spend it last.
-> groq:meta-llama/llama-4-scout-17b-16e-instruct
     Vision fallback once Gemini's 250/day is gone.

selected groq:meta-llama/llama-4-scout-17b-16e-instruct
```

Spending is tracked in a persistent ledger that understands each provider's metering — including that Groq's day rolls at UTC midnight and Google's at midnight Pacific — so the free-tier promise survives a reboot.

## How it fits together

```
   microphone ──▶ VAD ──▶ Groq Whisper ──▶ ┌─────────────┐
                          (STT)            │   ReAct     │
                                           │   loop      │──▶ Groq gpt-oss-120b
   ┌──────────────────────────────────────▶│             │    (text reasoning)
   │                                       └──────┬──────┘
   │   accessibility tree                         │ picks a tool
   │   UIA (Windows) / AX (macOS)                 ▼
   │   ~20 ms, local, free            ┌───────────────────────┐
   │        ▲                         │  safety interceptor   │  DENY  ─▶ refused
   │        │  no usable tree?        │  classify → confirm   │  CONFIRM ─▶ ask
   │        └── Gemini Flash          │  → journal            │  SAFE  ─▶ run
   │            (vision, last resort) └───────────┬───────────┘
   │                                              ▼
   │                          shell · git · click · type · press · read
   │                                              │
   └───────────── local memory ◀──────────────────┘  error → fix pairs
                  fastembed + FAISS + SQLite         captured automatically
                  offline, no quota
                                                  ▼
                                    Piper TTS ──▶ speaker  (local, offline)
```

Everything on the left of that diagram is free and local. The only boxes that spend a request are the three model calls, and the design spends the scarcest one — vision — last.

## Roadmap

Phases are units of execution and integration, not a calendar. Each has an exit gate it must pass before the next one starts — a box gets ticked only when that gate is green.

- [x] **P0 · Skeleton & Plumbing** — config, quota ledger, provider router, session tracing, CLI
- [x] **P1 · Voice I/O** — mic → VAD → STT → TTS, push-to-talk, latency benchmarks
- [x] **P2 · Agent Core** — ReAct loop, tool registry, shell and git tools
- [x] **P3 · Safety & Reversibility** — interceptor, dry-run, kill switch, action journal + undo
- [x] **P4 · Screen Perception** — UIA tree reader, screen capture, vision fallback (parallelizable)
- [x] **P5 · Desktop Actuation** — clicks and typing driven by accessibility handles, gated by P3, using P4
- [x] **P6 · Memory** — FAISS + fastembed, auto-captured error/fix pairs, recall injection
- [x] **P7 · Scout** — GitHub portfolio gap analysis, reusing P6's embedding stack
- [x] **P8 · Surface & Ship** — status strip, trace-derived benchmarks, tests

Dependencies: P0 → P1 → P2 → P3 → P5 → P6 → P7 → P8, with P4 branching off P0 and merging into P5. P4 is read-only, so it's the one phase that can be built out of order.

## Measured latency

Not projected — run `victor bench voice` and get these on your own machine. MacBook Air (Apple Silicon), Python 3.13.14, 7 runs:

| stage | p50 | p95 |
| --- | --- | --- |
| VAD endpointing | 0.1 ms per second of audio | 0.1 ms |
| Piper model load (once per session) | 526 ms | — |
| TTS time-to-first-audio, 1 sentence | 76 ms | 133 ms |
| TTS time-to-first-audio, 3 sentences | 42 ms | 47 ms |
| TTS full synthesis, 3 sentences | 109 ms | 113 ms |
| TTS realtime factor | 0.036× | 0.037× |
| STT round trip (Groq Whisper, live) | 246 ms | 533 ms |

Piper emits one chunk per sentence, so a one-sentence reply gets no benefit from streaming playback — time-to-first-audio *is* the synthesis time. Across three sentences it drops from 109 ms to 42 ms. Both are reported because averaging them would hide the effect.

The STT figure is a real network round trip, measured with `victor bench --voice --stt`, which spends real audio quota. The full voice→voice loop is still unmeasured: it needs a person speaking into a microphone, and this table would rather have a gap than an estimate.

## What this can't do

Stated upfront rather than discovered later. This section is longer than most projects' and that is deliberate — it is the part a reader can check.

**Nothing has gone voice → model → tool → speech end to end.** That is the last gap, and it needs a person at a microphone rather than more code. Every individual path has now met a real model: `victor selftest --live` runs each phase's exit gate against the real thing and reports **16 passed, 0 failed, 0 skipped** on this machine — the ReAct loop, tool schemas, prompts, speech-to-text, vision and memory recall all exercised by something other than `httpx.MockTransport`.

That retires the gap this section used to open with, *"never run with a live API key"*. Vision was the last to fall: asked for *"the folder icon for Projects"* against a real screenshot, Gemini answered `[3] Projects`.

- **Not sub-500ms.** Voice → shell is ~600–900 ms. Voice → vision → act → speak is 2–6 s. Both remain estimates: the model half is now real but the voice half has not been measured against it. The voice-stack numbers below are real.
- **Push-to-talk is terminal-scoped.** `victor listen --mode ptt` starts and stops on Enter. A system-wide hotkey needs an OS-level hook and lands with the HUD in P8.
- **Not always-on.** The free vision tier is ~250 requests/day, so screen capture happens on demand, never as a continuous stream.
- **Targeted app support.** UIA is tuned for File Explorer, Edge/Chrome, Windows Settings, and VS Code. Other apps may work but aren't guaranteed.
- **`--app` finds windows, not tabs.** Windows 11 Notepad is one window hosting tabbed documents, so `--app "notes"` targets the window and acts on whatever tab is active — which may not be the one you meant. Victor notices when the foreground it got is not the one it asked for and refuses rather than typing into the wrong document, so the failure is safe; it is still a limit. Browsers and Explorer tabs are the same shape.
- **Explorer saturates the element cap.** The walk stops at 200 elements and a real Explorer window has more, so every read of it reports the tree as truncated. Breadth-first ordering means the toolbar and the first rows survive, which is the part the agent acts on — but `screen_read` on Explorer is a window onto the window, not the whole of it.
- **Windows and macOS, not Linux.** Perception needs an accessibility backend: UI Automation on Windows, the Accessibility API on macOS. Linux (AT-SPI) is not implemented. Everything else — voice, shell, git, safety, memory — is platform-neutral and runs anywhere.
- **macOS needs permission.** Grant Accessibility to your terminal in System Settings → Privacy & Security → Accessibility, or the tree comes back empty. `victor doctor` says so plainly when it is missing.
- **Actuation is off by default.** Clicking and typing exist on both platforms, but the agent only gets them when you pass `--desktop`. These tools act on whatever window is in front rather than inside a directory Victor was pointed at, which is a different kind of permission.
- **Actuation is verified on both, at different depths.** macOS is verified end to end, including a two-task GUI run. Windows has had two review passes against real windows — the Pane climb, focus verification, modifier release and the capture region convention are all confirmed there, and each pass found defects that no macOS test could see. The agent loop itself has still never run on Windows, because that needs an API key on that machine. See the [build log](docs/BUILD-LOG.md#p5--desktop-actuation-) for what has and has not been exercised.
- **A locked screen looks like an empty one.** Both platforms stop reporting window geometry when locked, so Victor checks and says so rather than reporting a window with no controls.
- **Memory is semantic only with the extra installed.** `pip install -e '.[memory]'` pulls a ~130 MB ONNX model that matches paraphrases. Without it, recall falls back to a hashed bag of words that finds a traceback it has seen almost verbatim and nothing more. `victor doctor` reports which one is live, because "Victor remembers" means two different things.
- **Vision has never produced an answer.** Screen capture works, but the one live attempt hit a locked screen, so no image has reached a vision model. The request path is tested against a mock and the capture path is now real; the join between them is not.
- **Free-tier numbers are declared, not discovered.** Providers change allowances without notice. The routing table in [src/victor/providers/registry.py](src/victor/providers/registry.py) states them conservatively, so Victor under-uses a generous tier rather than hitting a 429 mid-demo.
- **Scout is a heuristic and says so on every run.** GitHub has no trending API, stars measure attention rather than quality, and the comparison set is seeded from your own topics — so it finds gaps adjacent to what you already do, not a view of the industry. Corpus results skew toward tutorials and awesome-lists, which outstar production code.
- **The status strip polls; it does not stream.** It reads the ledger and the newest trace four times a second. A task that starts and finishes between two polls will not appear.
- **No demo video.** The plan asks for a 90-second recording of four scenarios. That needs a person at the machine, and this section would rather admit the gap than describe a video nobody has made.

## Setup

Requires Python 3.13+ (developed on 3.13 and 3.14).

**Windows**

```powershell
git clone https://github.com/Harsha-Kamaraj/Victor_Agent.git
cd Victor_Agent
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[voice,desktop,memory]"

copy .env.example .env    # then add GROQ_API_KEY, free, no card
victor doctor             # verifies keys, storage, deps, quota
```

**macOS**

```console
git clone https://github.com/Harsha-Kamaraj/Victor_Agent.git
cd Victor_Agent
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[voice,desktop,memory]"

cp .env.example .env      # then add GROQ_API_KEY, free, no card
victor doctor
```

macOS also needs Accessibility permission for your terminal, in System Settings → Privacy & Security → Accessibility. `victor doctor` says so if it is missing.

Only `GROQ_API_KEY` is required — it serves both text reasoning and speech-to-text. `GEMINI_API_KEY` adds the larger half of the vision budget and `GITHUB_TOKEN` raises Scout from 60 requests an hour to 5,000; `victor doctor` reports each as SKIP rather than FAIL when absent, because a missing optional key is not a broken install.

The extras are optional and named after the phase that needs them: `voice` for the microphone and speech, `desktop` for reading and driving the screen, `memory` for semantic recall. Everything degrades with a stated reason rather than crashing when one is absent.

Then:

```console
victor selftest        # run every phase's exit gate; spends nothing
victor selftest --live  # ...including the ones that need a real model
victor models          # the routing table and every free allowance in it
victor route text      # which model serves a workload right now, and why
victor quota           # what's left today
victor trace show      # replay the last session, event by event

victor voice install   # download the Piper voice (~63 MB, once)
victor voice devices   # list microphones and speakers
victor say "hello"     # local synthesis, no network
victor listen          # record one utterance, transcribe it, read it back
victor bench voice     # measure VAD and TTS latency here

victor uia --dump                   # the focused window's elements, 0 API calls
victor uia --apps                   # applications you can target by name
victor uia --app Finder             # read a specific app's window
victor uia --demo                   # the same output shape on any platform
victor tools                        # what the agent can call
victor do "what changed on main?"   # one task, printed
victor do "..." --dry-run           # preview every action, execute none
victor converse                     # hold a spoken conversation

victor click "Compose" --dry-run     # what would be clicked, and its verdict
victor click "Compose"              # click it, gated and journalled
victor press "mod+s"                # mod is Ctrl on Windows, Command on macOS

victor check "rm -rf build"         # how the safety layer grades a command
victor journal list                 # what has been done, and what can be undone
victor journal undo last            # reverse it, if an inverse exists

victor index src/                   # read project files into local memory
victor recall "connection refused"  # search it, offline and free
victor memory                       # what Victor remembers, and how

victor scout --user <handle>        # portfolio gap analysis, with citations
victor hud                          # live status strip with the quota counter
victor bench --traces               # the measured table, from real sessions
```

`victor do` runs the ReAct loop: the model picks a tool, reads the result, and decides again, up to a step and token budget it reports at the end. `victor converse` wires that between the microphone and the speaker.

Budgets exist because the free tier is real. A run stops at 8 steps or 20,000 tokens, identical consecutive tool calls are refused, and tool output is truncated before it reaches the model — one noisy `git log` can otherwise exceed the 8,000 tokens-per-minute allowance by itself.

## Refusing to do anything stupid

Every action is graded before it runs — whether the agent chose it or you typed it — and you can ask without running anything:

```console
$ victor check "ls -la"          safe     reads only
$ victor check "rm -rf build"    confirm  rm deletes files
$ victor check "rm -rf /"        deny     recursive delete of a root, home or drive path
```

That includes clicks. A file manager is not a menu: UI Automation's Invoke on a file *opens* it, so `victor click "setup.exe"` would install something. Clicks are graded by label, so documents stay silent and executables ask.

Two rules govern this, and they pull against each other. **Fail closed:** anything not recognised as read-only needs confirmation, because a classifier that guesses "probably fine" teaches you the prompt means nothing. **Avoid alarm fatigue:** if everything prompts, you stop reading and start saying yes — so the read-only set is deliberately generous and a confirmation is remembered for the rest of the session.

Confirmation fails closed everywhere. Silence is no, an unparseable answer is no, no terminal to ask on is no. Spoken confirmation matters most here: "no" misheard as "go" would run a delete you just refused, so the affirmative set is small and explicit and anything outside it is a refusal.

**The journal is honest about undo.** Most side effects have no inverse, so the prompt tells you which *before* you answer:

```
Victor wants to run:
  rm notes.txt
  rm deletes files
  This cannot be undone: deleted files cannot be restored.
Continue? [y/N]
```

A plausible-looking undo would be worse than none — it would encourage approving a delete on the belief it can be walked back. Confirmation is the protection for irreversible actions; `victor journal undo` is a convenience for the reversible ones.

**The kill switch is cooperative, not a `SIGKILL`** — killing mid-write would lose the journal entry for the action in flight, which is the one you would most want. Ctrl-C, or saying "stop" during `victor converse`, trips a flag that three checkpoints observe: between loop steps, before a tool runs, and inside the shell wait loop. Measured abort latency on a running `sleep 30`: **26 ms**.

Developing on macOS is supported for everything including P4/P5; Linux for everything except them — see the [development environment notes](docs/BUILD-LOG.md#development-environment), which include the Homebrew `pyexpat` and macOS hidden-`.pth` workarounds.

## Remembering what it already worked out

Nobody curates a knowledge base of their own mistakes, so Victor builds one by watching. When something fails and the *same thing* later succeeds, whatever ran in between is stored as the fix:

```console
$ python3 app.py                     ok=False  ModuleNotFoundError: No module named 'helper'
$ ls                                 ok=True   ← ignored, this is looking around
$ printf 'def greet...' > helper.py  ok=True   ← recorded as the intervention
$ python3 app.py                     ok=True   → remembered how python3 was fixed
```

Hit the same error again in a later session and the fix comes back from a local ONNX model and a SQLite file, with **zero API calls** and no quota spent. `victor recall "<anything>"` searches it by hand, and `victor index <path>` adds project files.

Measured against a live model, on the same `ModuleNotFoundError` twice:

| | first run | second run |
| --- | --- | --- |
| steps | 7 | **3** |
| tool calls | 8 | **4** |
| wall clock | 24.8 s | **2.6 s** |

The first run explored — `ls -R`, `read_file`, `ls` — before finding the fix. The second recalled it at 0.956 similarity in 195 ms and went straight to the two commands that worked, character for character. The trace records `cost: 0` on that recall, because the claim is worth nothing unless it is counted.

"A later command succeeded" would have been the easier rule and a worse one: `pytest` fails, `ls` succeeds, and *"the fix is ls"* gets recalled confidently next time. Requiring the failing command itself to recover makes the pair verifiable rather than inferred — and when it never recovers, nothing is stored, because nothing has been learned.

This is not only about the shell. The desktop fails in the most repetitive ways of anything here, so the same rule covers it:

```console
click Save          ok=False  no element at index 3
screen_read         ok=True   ← ignored, this is looking around
open_app Notepad    ok=True   ← recorded as the intervention
click Save          ok=True   → remembered how click Save was fixed
```

The identity is the tool plus its **target** — `click Save`, not `click`, and the label rather than the index, because the index moves as a list re-sorts while the button stays the same button. Sharing an identity across two controls would let a successful click on Cancel "prove" that the failed click on Save had been fixed.

Whether something counts as a fix rather than a look is never guessed twice: `shell` asks the safety classifier, `git` asks its own list of mutating subcommands, and every other tool uses the flag its own definition already declares. And a call the safety layer blocked is not recorded as a failure at all — nothing ran, so nothing can be its fix.

Recall stays quiet below a relevance floor. A vector store always returns its nearest neighbour, and nearest is not relevant; an injected wrong memory is worse than none, because it arrives as prior experience and gets treated as evidence.

## Credits

Architecture and implementation plan by [@Gagan-1718](https://github.com/Gagan-1718) — the split-brain routing idea, the UIA-over-pixels bet, and the phase structure with exit gates all come from [docs/PLAN.md](docs/PLAN.md). Implementation by [@Harsha-Kamaraj](https://github.com/Harsha-Kamaraj).

## License

[MIT](LICENSE) — Copyright (c) 2026 Harsha Kamaraj and Gagandeep.

Use it, fork it, ship it. The dependencies keep their own licences; `pip install -e ".[voice,desktop,memory]"` pulls Piper, fastembed, FAISS and pyobjc, all under permissive terms of their own.
