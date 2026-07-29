# Victor — build log

> The companion to [PLAN.md](PLAN.md). That document is the **plan of record**:
> what was intended and why, written before implementation. This one is the
> **record of execution**: what was actually built, the decisions that changed
> once code ran, the measured numbers, and the exit-gate evidence.
>
> Where the two disagree, PLAN.md describes the intent and this file describes
> reality. Neither is retro-edited to hide the gap — the gap is the useful part.

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

## P3 · Safety & Reversibility ✅

The gate for everything that touches the machine. Built *before* actuation, not
bolted on after.

**Built**
- `safety/classify.py` — every call graded safe / confirm / deny.
- `safety/confirm.py` — typed and spoken confirmation, both failing closed.
- `safety/journal.py` — append-only record with undo recipes where an inverse
  exists, and an explicit reason where one does not.
- `safety/killswitch.py` — cooperative abort with three checkpoints.
- `safety/interceptor.py` — the piece that slots into P2's seam.
- CLI: `victor check`, `victor journal list`, `victor journal undo`,
  `--dry-run` and `--yes` on `victor do`.

**Exit gate**, measured rather than asserted:

| claim | result |
| --- | --- |
| `rm -rf` refused without confirmation | refused; the target file survived |
| confirmed once approved | asked once, ran, file removed |
| recorded in the journal | both the refusal and the execution, with undo status |
| kill switch stops a task inside 200 ms | **26 ms** — a `sleep 30` returned in 0.43 s |

### Two rules that pull against each other

**Fail closed.** Anything not recognised as read-only needs confirmation. A
classifier that guesses "probably fine" is worse than none, because it teaches
the user that the prompt means nothing.

**Avoid alarm fatigue.** If everything prompts, users stop reading and start
saying yes, and the safety layer becomes a latency tax that protects nobody. So
the read-only set is generous and specific — the commands a developer runs
dozens of times an hour pass silently — and confirmations within one session
are remembered rather than re-asked on every retry.

### The denylist was too wide, and that was a safety bug

The first implementation refused *every* `rm -rf`. Running the exit gate showed
why that is wrong: `rm -rf build` and `rm -rf node_modules` are routine, so a
permanent block makes ordinary cleanup impossible and pushes users toward
`--yes`, which disables confirmation for everything else too. A refusal users
route around is worse than a confirmation they read.

`DENY` is now reserved for damage with no recovery path — the filesystem root,
the whole home directory, a drive letter, a bare `*`, `mkfs`, `dd` to a device.
Everything else destructive is confirmed and journalled. `rm -rf ~` and
`rm -rf ~/` are both caught: a pattern that catches only one spelling of a
disaster catches neither in practice.

### The journal is honest about undo

Most side effects have no inverse. `rm` does not, and neither does anything
that reached the network. Entries carry either a real recipe or an explicit
reason there is none, and the confirmation prompt says which *before* the user
answers:

```
This cannot be undone: deleted files cannot be restored.
```

Presenting a plausible-looking undo would be worse than none, because it would
encourage approving a delete on the belief it can be walked back. Confirmation
is the protection for irreversible actions; undo is a convenience for the rest.

One bug found by test and worth recording: the `git add` inverse was
`git reset HEAD <paths>`, which fails with *ambiguous argument 'HEAD'* in a
repository with no commits — exactly the `git init && git add .` case. It is
now `git reset -- <paths>`, which works in both. An undo offered in a prompt
that would not have worked is the same class of failure the design exists to
prevent.

### The kill switch is cooperative

A `SIGKILL` mid-run would lose the journal entry for the action in flight,
which is the one you would most want to keep. So tripping the switch sets a
flag and three checkpoints observe it: between loop steps, before a tool runs,
and inside the shell tool's wait loop — the last is what bounds abort latency
by the 50 ms poll interval rather than by whatever the command decided to do.

The first Ctrl-C stops the run cleanly; a second one interrupts for real, so a
wedged process is still escapable. In `victor converse`, saying "stop" trips
the same switch. A truly global hotkey needs an OS-level hook and a macOS
permissions prompt; it is opt-in via `pynput` and, like push-to-talk, the
system-wide binding lands with the HUD in P8.

---

## P4 · Screen Perception ✅

**Built** — `desktop/elements.py`, `desktop/uia.py`, `desktop/capture.py`,
`desktop/vision.py`, and `victor uia --dump` / `--demo`.

**Exit gate** — `victor uia --demo` prints the README's example element list
and reports **0 API calls, 0 quota spent**, verified against the ledger
afterwards. ✅ The four-app timing check (Explorer, Edge, Settings, VS Code)
is outstanding and needs Windows.

### The index is the whole point

An `Element` is addressed by index, never by position. The model picks an
integer and *cannot* invent a coordinate — the rectangle comes from the OS
either way. That keeps the failure mode "chose the wrong button", which is
visible and recoverable, rather than "clicked 30 pixels off one", which is
neither. A vision answer naming an index the snapshot does not contain is
rejected before it can reach P5.

### The walk is bounded, and says when it truncated

A web page in Edge is thousands of nodes deep. An unbounded walk turns "20 ms"
into eight seconds on the one window you most wanted to read, so depth, element
count and wall-clock time are all capped — and the snapshot reports that it was
cut short rather than pretending it saw everything.

It is breadth-first for the same reason: if the walk *is* cut short, the
controls a user would actually reach for are near the top of the tree and are
already collected.

Filtering lives in one testable function rather than inside the walk. A window
contains hundreds of anonymous panes and groups; listing them buries the six
things you can act on and spends the context budget the model needs to reason.

### A fake backend is why this is testable at all

The `Backend` protocol has two implementations: `UIABackend` (Windows) and
`FakeBackend` (a literal tree). Everything above the backend — filtering,
indexing, bounding, caching, rendering, the vision request shapes — is verified
on macOS. **Only the thin `UIABackend` is unrun**, and it is the one piece that
genuinely cannot be exercised off Windows.

### Vision is metered by a hash, not by discipline

Screenshots downscale to 768 px and carry a difference hash. An unchanged
screen is served from cache, so an agent that looks twice while deciding — the
normal case, not the exception — pays once. The hash ignores a blinking caret
and notices a dialog opening, which is exactly the distinction worth paying
for.

### Both provider shapes, because the chain crosses providers

Gemini takes `inline_data`; Groq speaks the OpenAI `image_url` shape. Both are
implemented, so when Gemini's ~250/day is spent, vision continues on Groq's
separate allowance instead of stopping. This also closes a gap flagged earlier
in the build: before P4 the routing table listed a vision chain that **no code
could call**.

---

## P5 · Desktop Actuation ✅

**Build** — `desktop/keys.py` (one key vocabulary, two code tables),
`desktop/actions.py` (`Actuator` protocol, `MacActuator`, `WindowsActuator`,
`FakeActuator`, and the `Desktop` façade), `desktop/session.py`,
`tools/desktop.py` (seven tools), classification rules for clicks and
shortcuts, `victor click` and `victor press`.

**Exit gate** — two multi-step GUI tasks completed end to end on macOS, through
the real tool registry, the P3 interceptor and the journal, with the model taken
out of the loop. **12 of 12 tool calls spent zero quota; 0 API calls, 0 tokens.**
The voice leg still needs an API key.

```
TASK 1 - Calculator: 12 x 12, by clicking
  click  ok=True cost=0 accessibility  pressed '1'
  ... six clicks, every one via AXPress ...
  display -> ['12×12', '144']

TASK 2 - TextEdit: select all, replace, save
  screen_read ok=True cost=0            7 elements
  press_keys  ok=True cost=0 synthetic  pressed cmd+a
  type_text   ok=True cost=0 synthetic  typed 28 characters
  press_keys  ok=True cost=0 synthetic  pressed cmd+s
  disk after: 'Victor typed this during P5.'   <- verified on disk

SAFETY     one confirmation, on 'Delete': "delete is not something I can undo"
KILL SWITCH tripped mid-task -> next click blocked before it ran
```

### Act on the control, not on its pixels

Both platforms can perform a control's own action — `AXPress` on macOS, UI
Automation's `Invoke` / `Toggle` / `SelectionItem` on Windows. That is better
than clicking the rectangle's centre in three ways: it cannot miss, it needs no
cursor movement so the screen does not visibly twitch while the agent works,
and it is what the control actually *is* rather than a gesture that usually
triggers it. A synthetic click at the OS-reported centre is the fallback, used
only when a control offers no action at all.

Every `ActionResult` reports which path it took, so the ratio is visible rather
than assumed. Driving Calculator through seven buttons used `accessibility`
seven times and `synthetic` zero.

### The index is re-verified before it is used

A snapshot is a photograph. A list that re-sorts between the photograph and the
click hands index 7 to a different button, and the agent has no way to know.
So `click` and `type_text` take the *label* as well as the index, re-read the
tree (~20 ms), and refuse if they no longer match:

```
element 22 is '1', not 'Equals' - the screen changed since you looked.
'Equals' is now element 29.
```

Naming where the target went means the model recovers in one step instead of
retrying the same wrong index. This removes an entire class of failure for the
cost of one tree walk.

### Verified live, on real windows

macOS Calculator, driven end to end with no model in the loop:

| What | How | Result |
|---|---|---|
| 7 × 6 by clicking | 5 clicks, all via `AXPress` | display reads `42` |
| 8 × 8 by typing | `type_text("8*8")`, `press_keys("return")` | display reads `64` |
| Stale index | `click(22, "Equals")` after the tree moved | refused, correct index named |

### One event per character

The first typing implementation put the whole string on a single Quartz event
with `CGEventKeyboardSetUnicodeString`. It looks right, and it works in text
fields — but anything handling `keyDown:` itself sees one keystroke and drops
the rest. Calculator was sent `8*8` and displayed `8`. Now each character is its
own event, 4 ms apart. A silently truncated string is a worse failure than a
slow one, and this one was silent.

### Modifier flags leak onto the next keystroke

The one that cost the most to find, because every layer reported success while
nothing happened.

`press_keys("mod+a")` worked. `type_text("Victor…")` immediately afterwards
typed nothing, and reported "typed 28 characters" — which was true; the events
were posted. On macOS a newly created `CGEvent` **inherits the window server's
current modifier state**, so once a chord has set the Command flag, every event
made afterwards carries it. Each letter of "Victor" arrived as ⌘V, ⌘i, ⌘c, ⌘t,
⌘o, ⌘r — six menu shortcuts instead of a word.

```
inherited flags on a new event: 0x20100000     <- Command, still set
type WITHOUT clearing flags:  'before\n'       <- nothing
type WITH CGEventSetFlags(0): 'B'              <- typed
```

`release_modifiers` was originally a no-op on macOS, with a comment explaining
that flags ride on each event so nothing can be held. That was exactly backwards
— nothing is *held*, and the state persists anyway. It now posts a key-up for
each modifier keycode with no flags set, chord key-ups carry no flags, and
`type_text` clears them on every event. Three places, because the failure is
silent and any one of them being missed brings it back.

This also explains the earlier "it works on Calculator but not TextEdit"
confusion: the Calculator runs typed *before* pressing any chord.

### The terminal hole

P3 classifies shell commands before they run. None of that applies to a
keystroke — an agent that can type into a Terminal window has a shell with the
same privileges, reached by a path with no classification, no confirmation and
no journal entry. Typing into a terminal emulator is therefore **refused**, not
confirmed, with a pointer to the tool that does get read:

> refused: the focused window is a terminal (harshak — zsh — 80x24), and typing
> there would run a command that Victor's safety layer never sees. Use the shell
> tool instead.

`open_app` is restricted to plain application names for the same reason, and
opening a terminal is itself a confirmation.

### Clicking had to get its own classifier

Falling back on the `mutating` flag would have meant confirming every click,
which is precisely the alarm fatigue the safety layer exists to avoid. Clicks
are classified by their label instead, on the reasoning that interfaces are
designed so a *person* can recognise a consequential button. Matching is on
whole words, so the **Delete** button asks and the **Deleted Items** folder does
not; **Send** asks and **Sent Mail** does not.

Writing the tests changed one rule: a right click never performs what its label
names, it opens a menu, and whatever gets picked from that menu arrives as its
own click with its own label. So the button check moved above the label check.

### The zero-cost claim, counted

The plan asks P5 to instrument API calls per task. Tools now report what they
spent in `metadata["cost"]`, and `AgentResult` exposes `api_calls`,
`free_tool_calls` and `zero_cost_ratio`. That made the vision fallback worth
wiring as a tool: P4 built `VisionClient` and nothing consumed it, so the ratio
would have been a constant 1.0 — true, and meaningless. `find_on_screen` is the
only tool here that costs anything, it says so in its own description, and
running out of vision quota leaves a working agent rather than a crashed one.

### The screen was locked, and nothing said so

Trying to run the two-task exit gate, both TextEdit and Calculator reported
zero elements. The tree walk was fine. macOS had locked the screen, and a locked
screen keeps answering accessibility queries while quietly refusing to report
window geometry — so every rectangle came back empty and every element was
filtered out for having no visible area.

`desktop/session.py` now detects this on both platforms: `CGSessionCopyCurrentDictionary`
on macOS, `OpenInputDesktop` on Windows, which also catches a UAC prompt holding
the secure desktop. It fails open — a probe that cannot answer must not block
work on a guess. Snapshots also carry a `note` separating "nothing to click"
from "nothing measurable", because those need different responses from both the
user and the model.

### The Windows smoke test, and what it found

Gagan ran the smoke test this section used to ask for — `victor click --dry-run`,
then `victor click`, on File Explorer. The headline passed: perception read
**147 real elements in ~235 ms**, the click fired through UI Automation's Invoke
pattern and reported `via accessibility` rather than a synthetic click, and
`victor quota` still read zero afterwards.

It also found four defects, which is roughly the number that prediction implied
and is why the section was written. Every one was invisible from macOS.

**The environment did not even start clean.** `tzdata` was undeclared, and
Windows ships no IANA database — so the quota ledger's `ZoneInfo("America/Los_Angeles")`,
which exists because Groq's day rolls at UTC and Google's at Pacific, took ~57
tests down with it. Three further failures were test portability: two used POSIX
shell syntax against a tool that runs PowerShell, and the 200 ms abort budget
was written from a macOS measurement. Windows aborts in ~412 ms because
PowerShell's spawn and teardown dominate; the budget is now per-platform, and
the 26 ms figure in the README is labelled as the macOS one.

**Perception saw about half of File Explorer.** The climb from the focused
control to its owning window stopped at the first Pane ancestor, on the theory
that the desktop root is a Pane. It is — but so is six levels of Explorer's
internal scaffolding:

```
ListControl   'Items View'                <- climb stopped here
PaneControl   'Shell Folder View'
PaneControl   'Folder Layout Pane'
PaneControl   'Explorer Pane'
PaneControl   ''
PaneControl   'Downloads'
WindowControl 'Downloads - File Explorer' <- wanted this
PaneControl   'Desktop 1'                 <- avoiding this
```

147 elements instead of 248. The 101 missing ones were Back, Forward, the
address bar, Search, Cut, Copy, Paste, Rename, Delete, Sort, View and the window
buttons — the entire actionable toolbar of an app the plan names as a target.
The visible symptom was `victor uia --dump` reporting the window title as `Items
View`. It now climbs to the first `WindowControl` and falls back to the
foreground window if there is none.

**`focus_app` reported success without focusing.** `SetActive` cannot beat the
Windows foreground lock — when the caller is not already in front, the request
is downgraded to flashing a taskbar button, and it returns no error:

```
focus_app('Downloads') -> ok=True detail='focused Downloads'
actual foreground:      'Clone and smoke test Vic… - Visual Studio Code'
```

This is the same shape as the macOS modifier-flag bug two sections up: every
layer reporting success while nothing happened. The fix is the verification, not
a workaround for the lock — a workaround that is not verified is how this got
here. `launch_app` now checks whether the app is running *before* focusing, so a
focus that is merely blocked no longer opens a second window.

**A modifier could be left physically held.** The Windows chord pressed each key
before recording it, leaving a one-statement window where a key is down and
untracked — so the `finally` released nothing. Gagan injected a failure mid-chord
and confirmed a stuck physical Ctrl via `GetAsyncKeyState`. Reachable through a
COM error or a `KeyboardInterrupt`, which is how the kill switch is triggered,
and on Windows a stuck modifier affects the whole machine rather than just
Victor. Intent is recorded before acting now, and the release is unconditional,
matching macOS.

**`victor click` reached execution without passing the safety layer.** The CLI
called `Desktop.click` directly: no classifier, no confirmation, no journal.
That is structurally the same hole P5 closed by refusing to type into terminals,
and it matters more on Windows, where Invoke on an Explorer list item *opens* the
file — Gagan's click on a document launched it into Chrome. The classifier also
read `setup.exe` as "clicking navigates the interface", so `victor click` on an
installer would have run it unconfirmed.

Both halves are fixed. `victor click` and `victor press` now build the same
interceptor and journal the agent builds, with `--yes` for the smoke-test
workflow; and clicks whose label names a file that *runs* ask first, while
documents stay silent. "A person typed it" is not the same as "a person
understood what it would do", and the classifier is the part that knows the
difference between `notes.txt` and `setup.exe`.

Two smaller things: `--app` was accepted and silently ignored by the Windows
backend, so every command read the foreground window regardless of what was
asked for — silently targeting the wrong window being precisely the failure
class this project exists to avoid. And `--apps` was dead on Windows because the
enumeration lived in the macOS backend. Both implemented.

### Still outstanding

- **Windows has had one smoke test, not a verification.** Perception, a click,
  the quota ledger and the fixes above are exercised; typing, chords, scrolling
  and the vision fallback are not. Every fix has a regression test, but the
  tests drive a fake `uiautomation` — they prove the logic, not the binding.
- The exit gate ran through the tool registry, not through the model — there is
  still no API key, so no run has gone voice → LLM → click end to end.
- The re-verification Gagan asked for (`victor uia --dump` naming the real
  window, then a gated click on a `.txt`) has **not** been run: it needs a
  Windows machine, and this pass was written on macOS.

---

## P6 · Memory ✅

**Build** — `rag/embed.py` (two embedders behind one protocol), `rag/store.py`
(SQLite plus a FAISS cache), `rag/ingest.py` (chunking and the auto-capture
watcher), `rag/recall.py` (the `Memory` facade), the agent's error path, and
`victor index` / `victor recall` / `victor memory`.

**Exit gate** — passed. Same traceback, fresh process, recalled offline:

```
=== SESSION ONE ===
  $ python3 app.py            ok=False  ModuleNotFoundError: No module named 'helper'
  $ ls                        ok=True             <- ignored, diagnostic
  $ cat app.py                ok=True             <- ignored, diagnostic
  $ printf 'def greet...' > helper.py  ok=True    <- recorded as the intervention
  $ python3 app.py            ok=True   -> remembered how python3 was fixed

=== SESSION TWO (fresh process) ===
  $ python3 app.py -> ok=False
    ModuleNotFoundError: No module named 'helper'
  recall: True in 286ms (score 0.96)

  trace: {'kind': 'memory.recall', 'hits': 1, 'best': 0.958, 'cost': 0}
  provider/LLM events in this session: 0
  quota: every row still 0
```

Warm recall is **2.5 ms p50, 3.5 ms p95**; the 286 ms above is the ONNX model
loading on first use in a process. Both are quoted because averaging them would
hide which one you actually pay.

### The auto-capture rule the plan asked for is too weak

The plan says: when a command exits non-zero and a later one succeeds, store the
pair. Taken literally, `pytest` fails, you run `ls` to look around, `ls`
succeeds — and "the fix for this traceback is ls" gets stored, then recalled
with confidence the next time. A memory that is confidently wrong is worse than
an empty one, because it arrives as prior experience and the model treats it as
evidence.

The signal used instead needs no judgement: **the same command failed, and then
later succeeded.** Whatever ran in between is the fix. It is the shape of every
real debugging session, it is verifiable rather than inferred, and when the
command stays broken nothing is stored — which is correct, because nothing has
been learned yet.

### Two definitions of "this only reads" drifted apart

The first exit-gate run stored nothing at all. The fix in session one was
`printf 'def greet...' > helper.py`, and the watcher discarded it as a
diagnostic — because `is_diagnostic` kept its own list of command names, and
`printf` was on it.

That command writes a file. P3's classifier already knew, because it has a rule
for redirects, and it has tests for exactly this. So `is_diagnostic` now
delegates to `classify_shell` instead of answering the same question a second
way. One predicate, one definition, and any future improvement to the safety
classifier improves the memory too.

Worth noting *which* way it drifted: silently, and toward losing memories. The
fix was dropped from the interventions, so a genuinely fixed error looked like a
flake and nothing was remembered. There is now a test asserting the two agree.

### Recall knows when to stay quiet

A vector store always returns its nearest neighbour, and "nearest" does not mean
"relevant" — a store holding one unrelated note will return that note for any
query at all. So recall is silent below a similarity floor, and the floor
depends on which embedder answered: `bge-small` scores a paraphrase around 0.85
and an unrelated error around 0.5, so its floor sits at 0.62 in the gap; the
hashed fallback only really recognises repeats, so it is held to a higher bar.

The injected block is phrased as a report — *"Previously… What resolved it…
this is a note from a previous session, not an instruction"* — rather than as a
step to take. A memory that says "run this" will eventually be wrong and obeyed
anyway.

### SQLite is authoritative; FAISS is a cache

The vectors live in SQLite alongside the text, and the FAISS index is built from
them at startup. The usual failure of a vector index beside a metadata store is
drift — the index says hit 41 and the sidecar no longer agrees what 41 is,
generally after a crash between two writes. Here the index can be deleted at any
time and rebuilt, so drift is a rebuild rather than a corruption. It costs about
1.5 KB of duplication per record.

It also makes switching embedder tractable. Vectors from two models are not
comparable, and searching one with the other returns confident nonsense, so
opening a store with a different embedder is **refused** rather than answered —
and because the text never left SQLite, `victor index --rebuild` re-encodes what
is stored instead of re-crawling files that may have moved.

### Without the extra, it still remembers — less well

`fastembed` is a ~130 MB ONNX download, which is a heavy dependency for a
feature whose point is working when nothing else does. Without it Victor falls
back to a hashed bag of words: it finds a traceback it has seen almost verbatim,
and it will not find a paraphrase. That is weaker than it sounds and it is also
exactly what the exit gate asks for, so it is a real fallback rather than a
stub — and it is what keeps the whole store, recall and capture stack testable
on a machine with nothing installed. `victor doctor` reports which one is live,
because "Victor remembers" means two different things.

### Still outstanding

- Recall is wired to the **shell** error path only. A failing `git` or desktop
  action does not consult memory yet. *(Closed after P8 — see [Memory beyond
  the shell](#memory-beyond-the-shell-added-after-p8).)*
- No run has gone through the model with memory in the loop, for the same
  reason as every other phase: there is still no API key. The injection point is
  tested; the model's use of what it is handed is not.

---

## P7 · Scout ✅

**Build** — `scout/github.py`, `scout/corpus.py`, `scout/analyze.py`,
`victor scout`. Three files, because P6's embedder and similarity maths are
reused wholesale — which is what the plan asked for, this being the designated
cut-line.

**Exit gate** — passed against two real accounts. Ranked, spoken, every row
citing the repositories that produced it:

```
topic          distance                              seen in  evidence
claude         further from your work than 75%          5      affaan-m/ECC (234946★),
                                                               NousResearch/hermes-agent
                                                               your nearest: Harsha-Kamaraj/Portfolio
design-system  further from your work than 88%          2      donnemartin/system-design-primer,
                                                               mui/material-ui
                                                               your nearest: Cache-Me-If-You-Can

covered  style-guide (0.71 via Gagan-1718/portfolio)
```

### An empty corpus, and a filter that was too clever

The first run produced nothing at all: *"your repositories carry no topics or
languages to search on"*, against an account with ten public repositories.

`GENERIC_TOPICS` exists so a corpus is not seeded on `topic:python` — which
returns everything, and every portfolio has a gap against everything. But the
language fallback ran through the same filter, so `Python` and `C` were
discarded too and nothing was left.

Topics and languages are not interchangeable, which is the thing the first
version missed. `topic:python` is a corpus of everything; `language:Python` is
"active, well-starred Python work", which is a narrower and perfectly reasonable
comparison set. They are now separate qualifiers, and a portfolio with no topics
at all still gets a corpus.

### Absolute thresholds decide nothing here

With a corpus in hand, the second run said every topic was already covered.

Measuring the actual distribution explained why. Across 48 corpus repositories,
the nearest-user similarity ran from **0.53 to 0.73, median 0.63**. A sentence
embedder puts nearly all software writing in that band, so the 0.55 floor
marked everything covered — and 0.65 would have marked half of it a gap,
arbitrarily.

Rows are now ranked within *this run's own* distribution. That is robust to the
compression and it is the more honest claim: "further from your work than 80% of
the comparison set" is checkable, while "coverage 0.58" reads as a measurement
and is not one. The report says so under the table.

### Saying what the corpus is not

GitHub has no public trending API — the trending page comes from an internal
service, and scraping it would make Scout depend on somebody's HTML. So the
corpus is a Search API heuristic, and that is stated **in the output**, not only
in a docstring: the person reading a ranked list is the one who needs to know
how it was made.

Two biases are named for the same reason. Stars measure attention, not quality —
`google/styleguide` and `docker/awesome-compose` came out as the closest
matches to a portfolio README, which says more about documentation repositories
than about the portfolio. And seeding from the user's own topics finds gaps
*adjacent* to what they already do, which is the useful question but a narrower
one than it appears.

---

## P8 · Surface & Ship ✅

**Build** — `ui/hud.py`, `bench.py`, `victor hud`, `victor bench --traces`, the
README rewrite, and 57 more tests.

**Exit gate** — passed. A clone with only the required key reaches:

```
ready: 25 ok, 5 warn, 0 fail, 0 not yet built     (exit 0)
```

The five warnings are two unset optional keys, an unset GitHub token, a voice
model not yet downloaded, and a screen that was locked at the time — every one
naming what it is and how to change it.

### The HUD reads state off disk

The obvious design is for the agent to push updates into the strip, and it is
wrong twice over. Tk insists on owning the main thread on macOS, so an agent
that also wants it acquires a threading problem that has nothing to do with the
feature. And a HUD wired into the agent can only watch runs it was started with.

Both problems disappear if the strip is a monitor. The quota ledger and the
session traces are already files the agent maintains, so it polls them — start
it before or after a task, in another terminal, same result. The coupling is a
directory.

tkinter, because it ships with Python. The plan said *status strip, not a UI
framework* and specifically did not want PyQt.

### "Today" needed the routing table

Summing the ledger's buckets showed the wrong number. The ledger keys buckets by
each provider's reset timezone — Groq at UTC midnight, Google at midnight
Pacific — so for several hours a day those two disagree, and comparing against a
single date would make half the ledger look like yesterday. The strip would read
zero during a run that was spending, which is precisely the number it exists to
be trusted about.

`_today_keys()` derives the valid dates from the routing table, so a new
provider in a new timezone is counted without anyone remembering to. It
currently returns two dates, which is the bug reproducing itself on demand.

### A locked screen is not a broken install

`victor doctor` exited non-zero because a screen saver had kicked in. That is a
false alarm, and false alarms are how a preflight check teaches people to ignore
it. Transient session states are WARN now; FAIL is reserved for things that are
actually broken.

The same pass retired the PENDING convention, which existed so a green tick
could never stand for a pipeline that did not exist. Every phase has one now, so
an empty PENDING set is the correct state — and there is a test asserting no
check still claims "not implemented".

### The benchmark table is regenerated, not typed

`victor bench --traces` folds recorded sessions into p50/p95 per stage. A number
in the README that nobody can reproduce is a claim; one that falls out of a
command is a measurement. Rows carry their sample count, and any row with n < 20
says outright that its "p95" is the worst single sample rather than a
percentile — a p95 over three observations is not a p95, and a table that does
not admit it invites its reader to believe it.

### Still outstanding

- **No demo video.** The plan asks for 90 seconds across four scenarios. That
  needs a person at the machine.
- **Still no live API key**, which remains the largest gap in the whole project
  and is now the first line of the README's "What this can't do".

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

---

## Cross-platform perception *(added after P4)*

The plan targeted Windows only, and the build log said so. That changed: the
project is now developed on macOS and Windows at once, so perception had to
work on both.

It cost one new file. `desktop/ax.py` implements the same `Backend` protocol
against the macOS Accessibility API, and `select_backend()` picks by platform.
Nothing above the backend changed — indexing, filtering, bounding, caching,
Set-of-Mark prompting and every test are untouched. That protocol was put in
during P4 precisely so a second OS would be an addition rather than a rewrite,
and it was.

**Verified on real macOS windows**, not fakes: Chrome, Finder and System
Settings all read correctly, with real names, rectangles and disabled states —
Chrome's Forward button is correctly reported disabled when there is no forward
history.

Three things surfaced only by running it against real applications:

**Real trees contain duplicates.** Chrome reports its bookmark bar and New Tab
button under two parents, so the raw walk produced 124 elements where 49 were
distinct. Two identical rows with different indices waste the context budget
and give the model a choice with no right answer. The walk now dedupes on
(control type, name, rectangle) — which helps Windows equally, since UIA does
the same thing.

**macOS names controls differently.** Where Windows gives a Name, macOS often
gives an empty AXTitle and puts the label in AXDescription, AXHelp or the
value. Window buttons have none of those — but they do have a subrole, so a
close button is now "Close" rather than "<Button>".

**Subrole must beat description.** Finder's zoom button describes itself as
"this button also has an action to zoom the window". The canonical subrole name
is what a person would say and what the model should match on, so it is tried
first.

macOS also differs in two ways worth knowing operationally: Accessibility
permission is explicit and must be granted in System Settings, and the literal
frontmost process is sometimes a helper with no windows — hence `--app` to
target an application by name, and `--apps` to list them.

---

## Memory beyond the shell *(added after P8)*

P6 shipped with recall wired to the shell error path and nothing else, and I
wrote that down as outstanding rather than describing it as finished. This
closes it.

The early return was one line — `if self.memory is None or call.name != "shell"`
— but the reason it was there is not trivial. The watcher's whole rule depends
on identity: *the same thing failed, and later succeeded, so what ran in
between is the fix.* A command line has an obvious identity, `command_head`
extracts it, and two runs of `pytest` are two attempts at the same thing.
A click does not come with one.

**Identity is the tool plus its target.** `describe_call` answers that for
every tool: `click Save`, `git push`, `press_keys mod+s`, `open_app Notepad`.
Getting the target argument right matters more than it looks — for a click it
is the **label**, not the index, because the index shifts as the tree re-sorts
while the button is still the same button. Choosing wrong here does not fail
loudly; it merges two attempts into one, and then a successful click on Cancel
"proves" that the failed click on Save was fixed. There is a test named after
that failure.

For a tool nobody has taught the function about, identity falls back to the
whole argument list. That splits too finely, deliberately: an over-split
identity remembers nothing, an over-merged one remembers the wrong fix and
recalls it with confidence.

**"Did this change anything" has the same shape as before.** P6 already learned
this lesson once — `is_diagnostic` kept its own list of command names, called
`printf 'x' > helper.py` diagnostic, and dropped the fix. The answer now comes
from whoever knows best per call: `shell` asks the P3 classifier, `git` asks
its own `MUTATING` set, and everything else uses the `mutating` flag its
`ToolSpec` already declares. Nothing gets a second opinion invented for it.

That mapping is what makes the desktop work at all. `screen_read` and `scroll`
are declared non-mutating, so re-reading the screen after a failed click is
recognised as looking around rather than fixing — the desktop's version of the
`ls` problem, and it falls out of the existing declaration instead of needing a
new rule.

### A block is not a failure

Extending the hook surfaced a defect that had been live on the shell path since
P6. A call the safety layer denied, or one the loop refused as a repeat, comes
back as `ok=False` — and the watcher recorded it as a failure. Nothing had run.
The next success then looked like its fix, and the store would fill with advice
for problems nobody had.

Both are now marked in `metadata` (`decision` for a block, `refused` for a
turned-away call — a convention the desktop tools already used for refusing to
type into a terminal) and skipped. Four regression tests; all four fail against
the previous code, which is how I know they test something.

**713 tests**, up from 692.

### Still outstanding

- The same one as everything else: no run has gone through a live model with
  memory in the loop. The injection point is tested; what a model does with a
  recalled fix is not.

---

## First live model *(added after P8)*

Every phase up to here shipped with the same caveat attached: no API key on the
development machine, so every provider path was proven against
`httpx.MockTransport` and nothing else. That is now closed for the text tier.

Groq, Gemini and GitHub keys are set; `victor doctor` authenticates all three
against the real services — **29 ok, 2 warn, 0 fail**. The two warnings are a
Piper voice not yet downloaded, and no TTY to ask for confirmation on, which is
an artefact of how the check was invoked rather than a fault.

**The P2 exit gate, re-run against a live model rather than a fake:**

```console
$ victor do "what branch am I on and what changed?"
  ok git(subcommand='status')
       On branch main
You are on the main branch. The working tree has unstaged changes in two
files: src/victor/doctor.py and tests/test_config_and_cli.py.
answered in 2 steps, 2 API calls, 1778 tokens, 1417ms; 1/1 tool calls free
```

Correct, and correct about the two files that were actually modified. The tool
schema, the argument shapes and the result-feedback path all survived contact
with a real model, which until now was an assumption.

Two things this does **not** establish, and the README says so: speech-to-text
and vision have still only met a mock, and no run has gone voice → model →
tool → speech end to end.

### Two environment defects found on the way

**`victor` would not start at all.** The editable install's `.pth` file was
flagged `UF_HIDDEN`, and Python 3.13 skips hidden `.pth` files — a failure
already noted under [development environment](#development-environment). It was
worse than documented: the flag was set on the *entire* `.venv` tree, so
clearing it on one file was undone as soon as anything else touched the
directory, and the file was also missing its trailing newline. `chflags -R
nohidden .venv` fixes it for good. `pytest` never saw any of this because it is
configured with `pythonpath = ["src"]` — which is exactly why that line exists,
and also why a green suite was not evidence the CLI worked.

**`doctor` reported SKIP beside a detail reading "set".** An optional key that
is present was still given `Status.SKIP`, because the status was chosen once
for both branches while only the detail string varied. A contradiction inside a
single line is the fastest way to teach someone to stop reading the output, and
it was reporting a key the user had just gone and fetched as though it had been
ignored. Both branches now pick their own status, with a test for each.

---

## Running the paths that had never been run *(added after P8)*

With keys in place, the remaining unverified paths were worth walking one at a
time. Voice went first and cost nothing but a download; vision produced three
defects, none of which any test could have found.

### Voice, measured live

`victor voice install` fetched the Piper voice (63 MB), `victor say` spoke
through real speakers, and `victor bench --voice --stt` finally put a number on
the leg that had been marked "not measured yet" since P1:

```
stt round trip                            3      245.6      532.9  ms
```

That is Groq Whisper over the network, five runs, and it is the last figure the
README was carrying as an estimate. The full voice → voice loop still needs a
person to speak into a microphone, so it stays unmeasured and stays declared.

`victor doctor` also went from 2 warnings to 0 under a real terminal. One was
the missing voice; the other was "no terminal to ask on", which turned out to
be an artefact of running the check from a non-TTY shell rather than a fault -
worth confirming rather than assuming, since a confirmation prompt that cannot
be shown is exactly the failure that would make the safety layer useless.

### Vision was broken three ways

**`mss` cannot start on macOS 26.** It parses the OS version with
`float(platform.mac_ver()[0])`, and `mac_ver()` returns `''` there, so it raises
`ValueError: could not convert string to float: ''` before taking a frame. The
capture layer now prefers Quartz on macOS, which is already a dependency for
the accessibility tree and for synthesising events. A portable library that
breaks on a new OS release is no longer on the critical path.

**The region convention disagreed with itself.** `find_on_screen` passed the
window rectangle as `(left, top, width, height)`; the grabber unpacked it as
`(left, top, right, bottom)` and derived width by subtraction. Every windowed
capture was therefore the wrong size, and any window not at the origin would
have been captured from the wrong place entirely. It survived this long because
no test ever captured a region and the backend was too broken to run. One
convention now, written down, with a test at both layers.

**A blank screen was going to be sent to the model.** Without screen recording
permission macOS does not raise and does not return nil - it hands back a
perfectly valid image of nothing. That would have been downscaled, hashed and
uploaded, spending one of ~250 daily vision requests to ask which button is on
a black rectangle. Captures are now checked for uniformity and refused with the
System Settings path in the message.

Worth noting how this was found: `CGPreflightScreenCaptureAccess()` returned
**True** while every capture came back blank, and macOS's own `screencapture`
CLI failed with "could not create image from rect". The permission API agreed
the permission was granted; the screen disagreed. Only looking at the pixels
told the truth, which is why the check is on the image rather than on the API.

### And `doctor` was reporting a green tick for it

`ScreenCapture.available()` answered "is `mss` importable" - a different
question from "can this machine take a screenshot", and it answered yes on a
machine where capture was completely broken. It now takes one. That is the
exact failure the README's opening claim is about, sitting inside the tool
whose job is to catch it.

**719 tests.**

### Memory, with a model actually in the loop

The P6 write-up ended with "no run has gone through the model with memory in
the loop, for the same reason as every other phase: there is still no API key."
That is now done, and it is the most convincing thing in this log.

A scratch project with `from helper import greet` and no `helper` module. The
task: *run it, work out why it fails, create whatever is missing, run it again.*

**First run — 7 steps, 8 tool calls, 24.8 s.** It failed, tried `git status`
(not a repository), then `ls -R`, `read_file app.py`, `ls`, before writing the
module and succeeding. Ordinary exploration.

The watcher stored the pair without being asked, and stored the right half of
it: `mkdir helper` and the `echo "def greet..." > helper/__init__.py`, with the
three diagnostics excluded. That `echo` redirect is precisely the case that
broke the first version of `is_diagnostic` - it starts with a read-only command
name and writes a file - so the delegation to the safety classifier is what
made this capture correct rather than empty.

**Second run, same error, `helper/` deleted — 3 steps, 4 tool calls, 2.6 s.**

```json
{"kind": "memory.recall",   "duration_ms": 195.43, "hits": 1, "best": 0.956, "cost": 0}
{"kind": "memory.recalled", "action": "python3 app.py", "score": 0.956, "cost": 0}
```

It ran the script, failed, and went straight to the two commands from the first
session - character for character, including the escaped newline in the `echo`.
No `ls`, no `read_file`, no exploring. Seven steps became three, twenty-five
seconds became under three, and the recall that caused it spent nothing.

The `action` field in that trace line is new today: the hook used to record
`command` and fire only for the shell.
