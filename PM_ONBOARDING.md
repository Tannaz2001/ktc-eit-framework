# Project Knowledge Transfer — For Our New PM

Welcome aboard. This doc is written the way I'd explain the project to you
over coffee on your first day — no code required to read it. Anywhere I say
"see X.md," that's a deeper technical doc for when you want to go further;
you don't need to read those to do your job.

---

## 1. What this project actually is

We're building a **benchmarking framework for EIT (Electrical Impedance
Tomography) image reconstruction algorithms**, built around the **KTC 2023
Challenge** dataset (a real academic competition dataset).

In plain terms: researchers submit different mathematical/ML "methods" that
try to reconstruct an image (basically an X-ray-like scan, but from
electrical measurements) from raw sensor data. Our tool:

1. Runs each method against the same real test data (7 difficulty levels ×
   3 samples each = 21 test cases per method),
2. Scores how good each reconstruction is against the known correct answer
   (ground truth), using standardized metrics,
3. Shows the results side-by-side in a dashboard and an HTML report, so
   it's obvious which method is actually best — and where each one breaks
   down.

Think of it like a leaderboard + test harness for image-reconstruction
algorithms, with a Streamlit web dashboard on top.

**Why it matters:** without this, comparing methods was manual and
inconsistent — different people ran different scripts with different
scoring. This gives one shared, reproducible pipeline everyone trusts.

---

## 2. The three moving parts

```
data (real KTC scans)  →  framework (runs methods, scores them)  →  dashboard (shows results)
```

| Part | What it is | Where it lives |
|---|---|---|
| **Data** | Real KTC voltage measurements + ground-truth images | `Codes_Matlab/`, `EvaluationData/` (not in git — see below) |
| **Framework** | The engine: loads data, runs each method, computes scores | `src/ktc_framework/` |
| **Dashboard** | The Streamlit web app people actually look at | `app.py` |

There's also a **CLI mode** (`run.py`) for running the full benchmark
without opening a browser — useful for batch/overnight runs — and it writes
its results to `outputs/` and `reports/report.html`, which the dashboard
then reads.

**Important:** the actual scan dataset is *not* checked into git (too
large, and it's the official competition dataset with its own license). New
team members have to download it separately and drop it into the right
folder. This is the single biggest "why isn't this working" cause for
anyone new — see [RUN_GUIDE.md](RUN_GUIDE.md).

---

## 3. Glossary (the words you'll hear in standup)

- **Method / Plugin** — one reconstruction algorithm (e.g. `BackProjection`,
  `GaussNewton`, `CompetitionCNN`). Each is a plug-in module; anyone can add
  a new one without touching the core framework.
- **Level 1–7** — difficulty tiers in the KTC dataset. Level 1 is easiest
  (least noisy data), level 7 is hardest. A method's score across all 7
  levels is its "degradation curve."
- **Sample A/B/C** — 3 test cases per level.
- **KTC score / Composite score / Grade** — the standardized scoring metrics
  (borrowed from the official KTC competition scoring script) that let us
  rank methods on a single leaderboard, with a letter grade (A/B/C…) for
  quick reading.
- **Hull analysis / Qualitative detection** — a secondary check on top of
  the numeric score: "did the method actually detect the object in the
  right place at all," expressed as a plain-language summary and a
  percentage (e.g. "detected in 19/21 runs — 90.5%").
- **In-process vs out-of-process methods** — some methods are plain Python
  and run directly inside our app (frictionless to add). Others (like the
  competition-winning CNN) are external programs in a different Python
  environment that we call as a subprocess — much more fragile to wire up.
  This distinction is the source of most "adding a new method is hard"
  complaints. Full writeup: [ARCHITECTURE_GAP.md](ARCHITECTURE_GAP.md).
- **Runner / BatchRunner** — the internal component that loops over every
  method × level × sample combination and executes them.
- **`external_methods/`** — folder where new/experimental methods get
  dropped in (uploaded via the dashboard or manually) before being
  registered into the framework.

---

## 4. Current state of the project

- Actively developed, real commits landing multiple times a week.
- Core pipeline (data → run → score → dashboard) **works end-to-end** and
  has for a while — this isn't a prototype, it's in daily use by the team
  for evaluating methods.
- CI runs the automated test suite on every push to `main`/`final-sprint`
  and every PR into `main` (`.github/workflows/test.yml`, pytest).
- We're currently on the branch **`final-sprint`**, which tracks close to
  `main` and gets synced from it regularly (there's a recurring
  upstream-merge task each week — normal, not a sign of trouble).
- Known active gap: **adding a new out-of-process (e.g. ML/CNN) method
  still requires manual engineering work** — hardcoded subprocess glue,
  path detection, Python-version juggling. There's an agreed design
  (manifest-based method bundles, described in
  [ARCHITECTURE_GAP.md](ARCHITECTURE_GAP.md)) to fix this, tracked as a
  multi-sprint roadmap item. If a stakeholder asks "why can't we just drag
  and drop any ML model in," this is the honest answer — it's on the
  roadmap, not yet built.
- A "fresh clone" friction pass has already been done once (see
  [FRICTION_REPORT.md](FRICTION_REPORT.md)) — the known first-run gotchas
  for new developers are documented and mostly fixed.

---

## 5. Who's who (based on recent contribution history)

We're a small team and everyone touches the dashboard (`app.py`) and the
runner, so don't read this as strict ownership — more like "who to ping
first":

- **Tannaz** — repo maintainer for `main`, merges most PRs, focus on the
  runner/config validation and reporting pipeline.
- **Sahil (me)** — dashboard app, reconstruction methods, and scoring
  metrics.
- **Areeba** — dashboard UI/UX and the HTML report.
- **Seerat** — reconstruction methods and the external-method adapter
  (the in-process/out-of-process wiring mentioned above).

For day-to-day "who do I ask about X," these four are the whole active
dev team right now.

---

## 6. How work flows (so you can track status without reading code)

- Work happens on short-lived feature branches, merged into `main` (or
  occasionally `final-sprint`) via GitHub Pull Requests.
- Sprint branches exist historically (`sprint-6`, `sprint-7`, `sprint-8`)
  but the team has moved to a simpler PR-into-main flow recently — don't
  expect a strict sprint-branch-per-sprint pattern going forward, ask
  Tannaz for the current cadence.
- Every PR into `main` runs the automated test suite (GitHub Actions) —
  green check = tests pass, that's your merge gate signal.
- The best low-effort way for you to see "is it working right now": run
  the dashboard locally (`streamlit run app.py`, or see
  [RUN_GUIDE.md](RUN_GUIDE.md)) and look at the leaderboard tab — it's the
  single view that answers "which method is winning and by how much."

---

## 7. Documentation map (what to read, when)

We already over-document this repo — you won't need to write onboarding
docs from scratch, just know where to point people:

| Doc | Read this when you need to... |
|---|---|
| [README.md](README.md) | Get the overall project layout and quick setup |
| [RUN_GUIDE.md](RUN_GUIDE.md) | Actually run the pipeline/dashboard yourself, including dataset download |
| [PLUGINS.md](PLUGINS.md) | Understand how someone adds a new reconstruction method |
| [ARCHITECTURE_GAP.md](ARCHITECTURE_GAP.md) | Understand the ML-method-upload roadmap/gap mentioned above |
| [FRICTION_REPORT.md](FRICTION_REPORT.md) | See what pain points new devs hit and what's already fixed |
| [COMMANDS_REFERENCE.md](COMMANDS_REFERENCE.md) | Look up a specific CLI command |
| `docs/` folder | Deep-dive technical specs (method adapters, hull plugin, dashboard internals) |

---

## 8. Risks / things to watch as PM

- **Dataset dependency**: the framework is only as good as the real KTC
  dataset being present locally. It's not in git, isn't versioned with the
  code, and new environments (new laptop, new CI runner, a demo machine)
  need it re-provisioned manually. If we ever need this to run somewhere
  new (a demo, a new hire's machine), budget setup time for this step.
- **Out-of-process method friction** (Section 4) is the main scaling risk
  if the plan is "let researchers submit their own ML methods" — right now
  that's a manual engineering task per method, not self-serve.
- **Shared ownership of `app.py`** — it's a large, single dashboard file
  that basically everyone edits, which is a natural merge-conflict
  hotspot (we resolve these routinely, it's manageable, but worth knowing
  if you see repeated "merge conflict in app.py" in PR history).
