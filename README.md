# Victor Agent

**A voice-driven computer-use agent for Windows that runs entirely on free API tiers — because it reads the accessibility tree instead of guessing pixels.**

> **Status: Day 0 — planning complete, implementation starts now.**
> Nothing below is built yet. This README tracks the build honestly; features get
> ticked off only once they demo end to end. See [`docs/PLAN.md`](docs/PLAN.md)
> for the full 6-day plan, including the parts that were deliberately cut.

---

## What it will do

Victor listens, looks at your screen, and acts — hands-free.

- **Everyday desktop** — open apps, navigate UI, drive browser tasks
- **Developer workflows** — run shell commands, inspect and fix on-screen code, manage git
- **Remembers** — a local FAISS index of past error tracebacks and their fixes
- **Refuses to do anything stupid** — destructive commands are gated behind spoken confirmation, a dry-run preview, and a kill switch

---

## Why this is different

Most computer-use agent projects ask a vision model to predict click coordinates. On a real Windows desktop that means off-by-30px clicks, wrong buttons, and silent failures — and it burns a paid API call on every single step.

Victor reads the **Windows UI Automation tree** instead. That gives exact element names and bounding boxes, locally, in ~20 ms, for free:

```
[3]  Button  "Compose"      (24,180)-(140,220)
[7]  Edit    "Search mail"  (300,60)-(900,100)
[12] Button  "Settings"     (1400,60)-(1440,100)
```

The agent picks element `[3]`. No pixel guessing, no API call. The vision model is a *fallback*, used only for surfaces with no usable tree.

## The $0 architecture: split-brain routing

Free tiers differ wildly in generosity, so each workload is routed to whichever provider can actually afford it. Vision is by far the scarcest resource, so the design spends it last.

| Workload | Provider | Free allowance |
|---|---|---|
| Text reasoning (ReAct loop, tool choice) | Groq `gpt-oss-120b` | ~1,000–14,400 req/day |
| Vision (fallback only) | Gemini 2.5 Flash → Groq Llama-4-Scout | ~250/day → +~1,000/day |
| Speech-to-text | Groq `whisper-large-v3-turbo` | 28,800 audio sec/day |
| Text-to-speech | Piper (local ONNX) | unlimited, offline |
| Embeddings | fastembed (local ONNX) | unlimited, offline |
| UI perception | Windows UIA tree | unlimited, local |

The whole stack is ONNX or native C — no torch, no CUDA, no GPU required.

---

## Roadmap

- [ ] **Day 1** — Foundation, quota ledger, session tracing, voice spine (mic → VAD → STT → TTS)
- [ ] **Day 2** — ReAct agent core, shell tools, HITL safety interceptor
- [ ] **Day 3** — Kill switch, reversible action journal + undo, UIA tree reader
- [ ] **Day 4** — Desktop control end to end, vision fallback
- [ ] **Day 5** — RAG memory, `victor scout` portfolio analysis
- [ ] **Day 6** — HUD, benchmarks, tests, demo

---

## Honest limitations

Stated upfront rather than discovered later:

- **Not sub-500ms.** Voice → shell is ~600–900 ms. Voice → vision → act → speak is **2–6 s**. Real measured benchmarks will be published here; there is no point claiming a number the demo visibly misses.
- **Not always-on.** The free vision tier is ~250 requests/day, so screen capture happens on demand, never as a continuous stream.
- **Targeted app support.** UIA is tuned for File Explorer, Edge/Chrome, Windows Settings, and VS Code. Other apps may work but aren't guaranteed.
- **Windows only.** UI Automation is a Windows API.

---

## Setup

Requires Python 3.13+ (developed on 3.14).

```powershell
git clone https://github.com/Gagan-1718/Victor_Agent.git
cd Victor_Agent
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

copy .env.example .env    # then add your three free API keys
victor doctor             # verifies keys, mic, speakers, UIA access, quota
```

## License

MIT
