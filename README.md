# HR Assist

An HR/recruitment platform built on Flask and Groq. Started as a small set of
single-shot AI writing tools (JD generation, interview questions, candidate
emails) and grew into a multi-agent **Candidate Evaluation Pipeline** —
9 specialist agents, orchestrated end to end, that screen and score a
candidate against a job description.

For the deep-dive on *why* the pipeline is built the way it is (framework
choice, model routing, the hybrid scoring design, known limitations), see
**[ARCHITECTURE.md](ARCHITECTURE.md)**. This file covers what the app does
and how to run it.

## Features

### AI Recruiting Pipeline — `/pipeline`

Upload a job description and a resume (PDF, DOCX or TXT); nine agents run
across three stages and produce a full evaluation:

| Agent | What it does |
|---|---|
| **JD Analyzer** | Extracts role title, seniority, must-have vs. nice-to-have requirements, responsibilities |
| **Skills Extractor** | Parses the resume for technical skills, certifications, years of experience, role history |
| **Candidate Screener** | Fast pass/fail gate — flags unexplained employment gaps, job-hopping, missing mandatory certifications |
| **Semantic Matcher** | Embedding-based cosine similarity between the JD and resume (local model, not an LLM call) |
| **Experience Evaluator** | Judges years-of-experience fit and career progression/trajectory |
| **Culture Fit Analyzer** | Soft signal on values alignment from resume language, with visible evidence and a bias caveat |
| **Ranking Agent** | Combines everything into a 0–100 score and a Shortlist / Maybe / Reject recommendation |
| **Question Generator** *(on-demand)* | Interview questions targeted at the specific gaps the pipeline found |
| **Candidate Communicator** *(on-demand)* | Drafts a status/scheduling email reflecting the pipeline's actual findings |

The results dashboard shows every agent's output, a score breakdown with the
full weighted-average → penalty → final-score math (not just the end number),
and buttons to generate targeted interview questions or draft a candidate
email on demand.

### Quick Tools

Four single-purpose AI tools that predate the pipeline and remain as fast,
lightweight alternatives:

- **JD Creation** (`/jd`) — generate a job description from role, tone, skills, and qualifications
- **JD-Resume Comparison** (`/jdresume`) — TF-IDF match score plus an AI-written comparison summary
- **Interview Questions** (`/iqs`) — generate questions from role, tone, skills, and experience level
- **Email Creation** (`/email`) — freeform candidate email drafting by type and tone

## Architecture at a glance

```
Flask app (app.py) — routes only, no business logic
        │
        ├── groq_client.py     shared Groq client, two model tiers, JSON-mode
        │                      helper with fallback parsing
        ├── embeddings.py       local fastembed wrapper (Semantic Matcher)
        ├── orchestrator.py     EvaluationContext + run_pipeline() + in-memory
        │                      PIPELINE_STORE for the on-demand agents
        └── agents/             one file per agent, single responsibility
```

**Model routing:** two Groq tiers, each configured as the other's automatic
failover — `llama-3.1-8b-instant` for structured extraction/generation tasks,
`openai/gpt-oss-120b` for evaluative or candidate-facing output where
reliability matters more than latency.

**Scoring is hybrid, not LLM-guessed:** the final score is a deterministic
weighted sum of the four sub-scores computed in Python, minus a penalty for
screener red flags. The LLM is only used to write the human-readable
rationale — never to invent the number. Full reasoning in ARCHITECTURE.md.

**No heavy multi-agent framework.** The orchestration is a hand-built
`ThreadPoolExecutor`-based pipeline instead of LangGraph/CrewAI — the agent
dependency graph is fixed and known ahead of time, so a framework built for
dynamic agent handoff wasn't the right tool. See ARCHITECTURE.md for the
full trade-off discussion.

## Tech stack

- **Backend:** Flask, Python 3.12
- **LLM:** Groq API (`openai/gpt-oss-120b` + `llama-3.1-8b-instant`, with automatic failover between them)
- **Embeddings:** `fastembed` (local ONNX model, `BAAI/bge-small-en-v1.5`) — Groq has no embeddings endpoint
- **Document parsing:** `pypdf`, `python-docx`
- **Legacy JD/Resume matching:** `scikit-learn` (TF-IDF + cosine similarity), `nltk`
- **Frontend:** server-rendered Jinja templates, vanilla CSS/JS (no frontend framework or build step)

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com)

### 2. Install

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and add your Groq key:

```
GROQ_API_KEY=your_groq_api_key_here
```

`.env` is gitignored and excluded from Docker/Cloud builds — never commit it.

### 4. Run

```bash
python app.py
```

The app runs at `http://127.0.0.1:8080`. First run downloads the local
embedding model (~130MB, one-time, cached afterward) and NLTK's tokenizer
data.

### 5. Docker (optional)

```bash
docker build -t hr-assist .
docker run -p 8080:8080 -e GROQ_API_KEY=your_key hr-assist
```

## Project structure

```
app.py                        Flask routes only
groq_client.py                 Shared Groq client + model tiers + JSON helper
embeddings.py                   Local embedding wrapper for Semantic Matcher
orchestrator.py                 Pipeline sequencing + EvaluationContext + store
agents/
  jd_analyzer.py                 JD requirement extraction
  skills_extractor.py             Resume skill/experience extraction
  candidate_screener.py            Red-flag / pass-fail gate
  semantic_matcher.py               Embedding similarity (local, not LLM)
  experience_evaluator.py            Career progression assessment
  culture_fit_analyzer.py             Resume-language values signal
  ranking_agent.py                     Hybrid deterministic + LLM scoring
  question_generator.py                 On-demand, gap-targeted questions
  candidate_communicator.py              On-demand candidate email drafting
templates/
  base.html                       App shell: sidebar nav, mobile drawer
  index.html, jd.html, iqs.html,
  email.html, jdresume.html        Quick Tools pages
  pipeline.html                     Pipeline upload form + results dashboard
static/styles.css                Design system: tokens, sidebar, dashboard,
                                 meters, badges, chips, forms
ARCHITECTURE.md                 Design rationale and trade-off discussion
```

## Known limitations

- **`PIPELINE_STORE` is in-memory** (a Python dict), process-local, and lost
  on restart — fine for a single-user demo/POC, not for production. A real
  deployment would back it with Redis or a database, which is also where
  multi-candidate ranking/shortlisting across a pool would live.
- **No authentication.** Intentional for this stage of the project.
- **Culture Fit Analyzer is a heuristic**, not a determination — it's
  weighted low in the final score and always shown with its supporting
  evidence so a human can sanity-check it.
- Groq's free/dev-tier rate limits apply; the fast/deep model failover
  absorbs some transient errors but isn't a substitute for real rate-limit
  handling under sustained load.

## License

Released under the [MIT License](LICENSE).
