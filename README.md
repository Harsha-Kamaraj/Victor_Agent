# Victor Agent

A voice-driven computer-use agent for Windows that runs entirely on free API tiers — because it reads the accessibility tree instead of guessing pixels.

**Status: P0 complete.** Config, quota ledger, provider router, session tracing and CLI are built and tested. Everything below the P0 line is not implemented yet, and `victor doctor` says so out loud rather than reporting a green tick for a pipeline that does not exist. See [docs/PLAN.md](docs/PLAN.md) for the full plan, including the parts that were deliberately cut.

## What it will do

Victor listens, looks at your screen, and acts — hands-free.

- **Everyday desktop** — open apps, navigate UI, drive browser tasks
- **Developer workflows** — run shell commands, inspect and fix on-screen code, manage git
- **Remembers** — a local FAISS index of past error tracebacks and their fixes
- **Refuses to do anything stupid** — destructive commands are gated behind spoken confirmation, a dry-run preview, and a kill switch

## Why this is different

Most computer-use agent projects ask a vision model to predict click coordinates. On a real Windows desktop that means off-by-30px clicks, wrong buttons, and silent failures — and it burns a paid API call on every single step.

Victor reads the Windows UI Automation tree instead. That gives exact element names and bounding boxes, locally, in ~20 ms, for free:

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
| Vision (fallback only) | Gemini 2.5 Flash → Groq Llama-4-Scout | ~250/day → +~1,000/day |
| Speech-to-text | Groq `whisper-large-v3-turbo` | 28,800 audio sec/day |
| Text-to-speech | Piper (local ONNX) | unlimited, offline |
| Embeddings | fastembed (local ONNX) | unlimited, offline |
| UI perception | Windows UIA tree | unlimited, local |

The whole stack is ONNX or native C — no torch, no CUDA, no GPU required.

Routing is not a diagram; it is code you can interrogate:

```console
$ victor route vision
   gemini:gemini-2.5-flash daily request limit reached (250/250)
     Primary vision. Scarcest resource in the stack - spend it last.
-> groq:meta-llama/llama-4-scout-17b-16e-instruct
     Vision fallback once Gemini's 250/day is gone.

selected groq:meta-llama/llama-4-scout-17b-16e-instruct
```

Spending is tracked in a persistent ledger that understands each provider's metering — including that Groq's day rolls at UTC midnight and Google's at midnight Pacific — so the free-tier promise survives a reboot.

## Roadmap

Phases are units of execution and integration, not a calendar. Each has an exit gate it must pass before the next one starts — a box gets ticked only when that gate is green.

- [x] **P0 · Skeleton & Plumbing** — config, quota ledger, provider router, session tracing, CLI
- [ ] **P1 · Voice I/O** — mic → VAD → STT → TTS, push-to-talk, latency benchmarks
- [ ] **P2 · Agent Core** — ReAct loop, tool registry, shell and git tools
- [ ] **P3 · Safety & Reversibility** — interceptor, dry-run, kill switch, action journal + undo
- [ ] **P4 · Screen Perception** — UIA tree reader, screen capture, vision fallback (parallelizable)
- [ ] **P5 · Desktop Actuation** — UIA-driven clicks and typing, gated by P3, using P4
- [ ] **P6 · Memory** — FAISS + fastembed, auto-captured error/fix pairs, recall injection
- [ ] **P7 · Scout** — GitHub portfolio gap analysis, reusing P6's embedding stack
- [ ] **P8 · Surface & Ship** — HUD, benchmarks, tests, demo

Dependencies: P0 → P1 → P2 → P3 → P5 → P6 → P7 → P8, with P4 branching off P0 and merging into P5. P4 is read-only, so it's the one phase that can be built out of order.

## Honest limitations

Stated upfront rather than discovered later:

- **Not sub-500ms.** Voice → shell is ~600–900 ms. Voice → vision → act → speak is 2–6 s. Real measured benchmarks will be published here; there is no point claiming a number the demo visibly misses.
- **Not always-on.** The free vision tier is ~250 requests/day, so screen capture happens on demand, never as a continuous stream.
- **Targeted app support.** UIA is tuned for File Explorer, Edge/Chrome, Windows Settings, and VS Code. Other apps may work but aren't guaranteed.
- **Windows only.** UI Automation is a Windows API. The core (config, quota, routing, tracing, memory) is platform-neutral and runs anywhere; only perception and actuation are Windows-bound.
- **Free-tier numbers are declared, not discovered.** Providers change allowances without notice. The routing table in [src/victor/providers/registry.py](src/victor/providers/registry.py) states them conservatively, so Victor under-uses a generous tier rather than hitting a 429 mid-demo.

## Setup

Requires Python 3.13+ (developed on 3.13 and 3.14).

```powershell
git clone https://github.com/Gagan-1718/Victor_Agent.git
cd Victor_Agent
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

copy .env.example .env    # then add your three free API keys
victor doctor             # verifies keys, storage, deps, quota
```

Then:

```console
victor models        # the routing table and every free allowance in it
victor route text    # which model serves a workload right now, and why
victor quota         # what's left today
victor trace show    # replay the last session, event by event
```

Developing on macOS or Linux is supported for everything except P4/P5 — see the [development environment notes](docs/PLAN.md#development-environment), which include the Homebrew `pyexpat` and macOS hidden-`.pth` workarounds.

## License

MIT
