# Candidate Evaluation Pipeline — Architecture Notes

A multi-agent recruiting pipeline bolted onto the existing HR Assist Flask app.
Given a JD + resume, it produces a 0–100 fit score, a shortlist recommendation,
and on-demand interview questions / candidate emails. Written up here so the
design decisions and their reasoning are easy to explain and defend, not just
demo.

## Why a custom orchestrator instead of LangGraph / CrewAI

Evaluated both. Chose to hand-build a small `Agent function + Orchestrator`
layer instead, for three reasons:

1. **Transparency.** Every agent is a plain Python function with an explicit
   input/output contract. Nothing is hidden behind a framework's internal
   state machine — if asked "what exactly happens when you click Run", the
   honest answer is "read `orchestrator.py` top to bottom."
2. **This pipeline doesn't need dynamic agent handoff.** The 7 automatic
   agents run in a fixed, known dependency order (see below) — there's no
   case where an agent needs to autonomously decide which agent to call next.
   A graph/handoff framework earns its complexity when the control flow is
   itself uncertain; here it isn't.
3. **Lower risk on a short timeline.** No new heavyweight dependency, no
   framework-specific debugging, and the whole thing is inspectable by
   reading five files.

## Pipeline shape

The 9 requested agents split into three groups by how they're actually used,
not just alphabetically:

```
Stage 1a (parallel, independent)      Stage 1b (depends on 1a)
┌─────────────────┐                   ┌──────────────────────┐
│  JD Analyzer     │                   │  Candidate Screener   │
│  (fast model)    │──┐                │  (deep model)         │
└─────────────────┘  │                │  needs JD must-have   │
┌─────────────────┐  ├──────────────► │  list from Stage 1a   │
│ Skills Extractor │──┘                └──────────────────────┘
│  (fast model)    │
└─────────────────┘

Stage 2 (parallel, depends on Stage 1)
┌──────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│ Semantic Matcher  │  │ Experience Evaluator   │  │ Culture Fit Analyzer   │
│ local embeddings  │  │ (deep model)           │  │ (deep model)           │
│ (not an LLM call)  │  └───────────────────────┘  └───────────────────────┘
└──────────────────┘

Stage 3 (sequential, depends on everything above)
┌───────────────────────────────────────────────────────────┐
│ Ranking Agent - deterministic weighted score (Python)       │
│               + LLM-written rationale (deep model)          │
└───────────────────────────────────────────────────────────┘

On-demand (triggered from the results page, not run automatically)
┌───────────────────┐        ┌──────────────────────────┐
│ Question Generator  │        │ Candidate Communicator     │
│ targets the gaps    │        │ writes the recruiter's     │
│ Stage 1-3 found      │        │ chosen email type          │
└───────────────────┘        └──────────────────────────┘
```

`orchestrator.py` implements exactly this: `ThreadPoolExecutor` for the two
parallel stages, plain sequential calls elsewhere. Each stage's outputs are
collected into a single `EvaluationContext` dataclass that flows through the
rest of the pipeline and gets cached (see Known Limitations).

## Model routing: fast tier vs deep tier

Two Groq model tiers, each configured as the other's automatic failover:

- **`llama-3.1-8b-instant` (fast tier):** JD Analyzer, Skills Extractor,
  Question Generator — structured extraction and generation tasks where the
  fast model is reliably accurate.
- **`openai/gpt-oss-120b` (deep tier):** Candidate Screener, Experience
  Evaluator, Culture Fit Analyzer, Ranking Agent, Candidate Communicator —
  evaluative/judgment tasks, or anything that produces candidate-facing or
  decision-facing output.

This is a resource-allocation call, not a blanket rule — worth stating that
explicitly if asked. The initial version routed Candidate Communicator to the
fast tier under "it's just prose generation, low stakes." Testing surfaced
the opposite: the fast model inconsistently followed the HTML-formatting
instruction (produced markdown asterisks instead) and occasionally appended
unprompted commentary about its own output. Since this text is what actually
gets sent to a candidate, it was moved to deep-tier-first. The lesson worth
saying out loud: "low stakes" should be judged by the audience of the output,
not by how mechanical the task looks.

## Ranking Agent: hybrid deterministic + LLM design

The final 0–100 score is computed in plain Python, not asked of an LLM:

```
weighted_score = 0.30 * skills_coverage        (deterministic token-overlap
                + 0.20 * semantic_similarity      match, not an LLM call)
                + 0.30 * experience_progression
                + 0.20 * culture_fit

final_score = weighted_score - red_flag_penalty
red_flag_penalty = 20 pts per screener red flag, capped at 60 total
                    (a failed screen with no itemized flag still counts as one)
```

The LLM call in the Ranking Agent is used only to turn these numbers into a
human-readable rationale — it's told the score is already final and not to
be changed. This was a deliberate choice: an LLM asked to "give a score" will
produce a plausible-sounding number that isn't reproducible, auditable, or
guaranteed consistent across two runs of the same input. Keeping the
arithmetic in code and reserving the LLM for language is the difference
between "an AI feature" and "a system a hiring decision can be traced through."

**The math is also surfaced in the UI, not just in code.** The results
dashboard shows the weighted average before any penalty, the points deducted
and why, and the resulting final score as one visible line — the first
version applied a red-flag penalty *and* a hard cap at 45 whenever the
screener failed, but only displayed the small per-flag deduction, so a
candidate scoring ~87 across the board could land at 45 with no visible
explanation for the other ~40-point drop. That's a transparency bug in a
system whose whole pitch is auditability: a cap or penalty that isn't shown
is functionally the same as a black-box score.

**The proportional penalty (vs. a hard cap) was also a deliberate revision.**
The original design hard-capped the final score at 45 on any failed screen,
treating a missing mandatory requirement as a near-veto regardless of how
strong the other four signals were. In practice that made a single gap (e.g.
one missing certification) force an automatic "Reject" even for an otherwise
excellent candidate, which doesn't match how a human recruiter actually
weighs a borderline-but-strong profile. Replacing the cap with a larger
proportional deduction (20 pts/flag) keeps red flags consequential without
making them absolute - worth stating as a considered trade-off, not just
"we made the number bigger."

## Semantic Matcher: local embeddings, not an API

Groq's API is chat/completion only — no embeddings endpoint. Rather than add
a second paid API (OpenAI, Cohere, etc.) for one metric, the Semantic Matcher
runs `BAAI/bge-small-en-v1.5` locally via `fastembed` (ONNX runtime, CPU-only,
no `torch` dependency). Free, offline, no extra API key, and light enough not
to bloat the Docker image. Cosine similarity is rescaled from an observed
~0.2–0.9 real-world range into 0–100 so scores read intuitively (see
`embeddings.py` for the calibration rationale).

## Responsible-AI note: Culture Fit Analyzer

This is the most subjective, most bias-prone agent in the pipeline —
inferring "culture fit" from resume language can reward familiar phrasing or
background over substance. Two mitigations, both visible in the UI, not just
in a comment:

- Weighted at only 20% in the final score, same as semantic similarity and
  below skills/experience.
- Always rendered with a visible caveat and its supporting evidence quotes,
  so a human reviewer can sanity-check the signal instead of trusting a bare
  number.

Worth raising unprompted in an interview — it signals product judgment, not
just the ability to wire up a prompt.

## File map

```
groq_client.py          Shared Groq client, model tiers, generate_completion
                         (free text) / generate_json (structured, with JSON
                         extraction fallback if a model ignores JSON mode)
embeddings.py            Local fastembed wrapper for the Semantic Matcher
orchestrator.py          EvaluationContext + run_pipeline() + PIPELINE_STORE
agents/
  jd_analyzer.py          Stage 1a
  skills_extractor.py      Stage 1a
  candidate_screener.py     Stage 1b
  semantic_matcher.py       Stage 2 (local)
  experience_evaluator.py   Stage 2
  culture_fit_analyzer.py    Stage 2
  ranking_agent.py            Stage 3 (hybrid scoring)
  question_generator.py        on-demand
  candidate_communicator.py     on-demand
app.py                   Flask routes only - no business logic
templates/pipeline.html   Upload form + results dashboard
```

## Known limitations (and what production would need)

- **`PIPELINE_STORE` is an in-memory dict**, process-local, lost on restart.
  Fine for a single-user POC/demo; a real deployment would back it with
  Redis or a database, and that's also the natural place to add persistence
  for the `Ranking Agent`'s stated goal of "shortlisting" across *many*
  candidates rather than one CV vs. one JD at a time.
- **No auth** — acceptable for a POC per the project's own scope, not for
  production.
- **Groq dev/free-tier rate limits** apply; the fast/deep failover partially
  absorbs transient errors but isn't a substitute for real rate-limit
  handling under load.
- **Culture Fit Analyzer is a heuristic**, explicitly labeled as such in the
  UI — it should never be the deciding signal on its own.
