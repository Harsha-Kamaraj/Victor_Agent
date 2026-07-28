# Victor Agent

A voice-driven computer-use agent for Windows that runs entirely on free API tiers — because it reads the accessibility tree instead of guessing pixels.

**Status: P0–P1 complete.** Plumbing (config, quota ledger, provider router, tracing, CLI) and voice I/O (mic → VAD → STT → TTS) are built, tested and measured. Everything below the P1 line is not implemented yet, and `victor doctor` says so out loud rather than reporting a green tick for a pipeline that does not exist. See [docs/PLAN.md](docs/PLAN.md) for the full plan, including the parts that were deliberately cut.

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
- [x] **P1 · Voice I/O** — mic → VAD → STT → TTS, push-to-talk, latency benchmarks
- [ ] **P2 · Agent Core** — ReAct loop, tool registry, shell and git tools
- [ ] **P3 · Safety & Reversibility** — interceptor, dry-run, kill switch, action journal + undo
- [ ] **P4 · Screen Perception** — UIA tree reader, screen capture, vision fallback (parallelizable)
- [ ] **P5 · Desktop Actuation** — UIA-driven clicks and typing, gated by P3, using P4
- [ ] **P6 · Memory** — FAISS + fastembed, auto-captured error/fix pairs, recall injection
- [ ] **P7 · Scout** — GitHub portfolio gap analysis, reusing P6's embedding stack
- [ ] **P8 · Surface & Ship** — HUD, benchmarks, tests, demo

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

Piper emits one chunk per sentence, so a one-sentence reply gets no benefit from streaming playback — time-to-first-audio *is* the synthesis time. Across three sentences it drops from 109 ms to 42 ms. Both are reported because averaging them would hide the effect.

STT round trip and the full voice→voice loop are **not measured yet** — they need a live `GROQ_API_KEY`. `victor bench voice --stt` measures them, and spends real audio quota to do it.

## Honest limitations

Stated upfront rather than discovered later:

- **Not sub-500ms.** Voice → shell is ~600–900 ms. Voice → vision → act → speak is 2–6 s. Those two remain estimates until P2 exists to measure them end to end; the voice-stack numbers above are real.
- **Push-to-talk is terminal-scoped.** `victor listen --mode ptt` starts and stops on Enter. A system-wide hotkey needs an OS-level hook and lands with the HUD in P8.
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
victor models          # the routing table and every free allowance in it
victor route text      # which model serves a workload right now, and why
victor quota           # what's left today
victor trace show      # replay the last session, event by event

victor voice install   # download the Piper voice (~63 MB, once)
victor voice devices   # list microphones and speakers
victor say "hello"     # local synthesis, no network
victor listen          # record one utterance, transcribe it, read it back
victor bench voice     # measure VAD and TTS latency here
```

`victor listen` has no agent behind it until P2, so it acknowledges what it heard rather than answering. It exists to prove the mic → STT → TTS round trip works before anything is built on top of it.

Developing on macOS or Linux is supported for everything except P4/P5 — see the [development environment notes](docs/PLAN.md#development-environment), which include the Homebrew `pyexpat` and macOS hidden-`.pth` workarounds.

## License

MIT
