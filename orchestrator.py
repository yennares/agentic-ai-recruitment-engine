"""Orchestrator - sequences the specialist agents into a candidate evaluation
pipeline, based on their actual data dependencies (not just "run everything
in parallel for show").

Stage 1a (parallel): JD Analyzer and Skills Extractor read only the raw JD
  and resume text respectively - fully independent, run concurrently.
Stage 1b: Candidate Screener needs the JD Analyzer's must-have list, so it
  runs right after Stage 1a rather than alongside it.
Stage 2 (parallel): Semantic Matcher (local embeddings, near-instant),
  Experience Evaluator and Culture Fit Analyzer all depend on Stage 1
  output but not on each other - run concurrently.
Stage 3: Ranking Agent depends on every prior stage's output, so it runs last
  and combines them into the final score.

Question Generator and Candidate Communicator are NOT part of this automatic
pipeline - they're triggered on demand from the results dashboard, reading a
completed evaluation back out of PIPELINE_STORE.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

from agents.jd_analyzer import analyze_jd
from agents.skills_extractor import extract_skills
from agents.candidate_screener import screen_candidate
from agents.semantic_matcher import match_semantics
from agents.experience_evaluator import evaluate_experience
from agents.culture_fit_analyzer import analyze_culture_fit
from agents.ranking_agent import rank_candidate


@dataclass
class EvaluationContext:
    jd_text: str
    resume_text: str
    jd_analysis: dict = field(default_factory=dict)
    skills_extraction: dict = field(default_factory=dict)
    screener_result: dict = field(default_factory=dict)
    semantic_result: dict = field(default_factory=dict)
    experience_result: dict = field(default_factory=dict)
    culture_result: dict = field(default_factory=dict)
    ranking_result: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# In-memory store keyed by eval_id, so the on-demand agents (Question
# Generator, Candidate Communicator) can reuse a completed evaluation without
# re-running the whole pipeline or round-tripping large text through hidden
# form fields. POC-scoped: process-local and lost on restart - a production
# deployment would back this with Redis or a database instead.
PIPELINE_STORE = {}


def run_pipeline(jd_text, resume_text):
    context = EvaluationContext(jd_text=jd_text, resume_text=resume_text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        jd_future = executor.submit(analyze_jd, jd_text)
        skills_future = executor.submit(extract_skills, resume_text)
        context.jd_analysis = jd_future.result()
        context.skills_extraction = skills_future.result()

    context.screener_result = screen_candidate(resume_text, context.jd_analysis.get("must_have", []))

    with ThreadPoolExecutor(max_workers=3) as executor:
        semantic_future = executor.submit(match_semantics, jd_text, resume_text)
        experience_future = executor.submit(evaluate_experience, context.jd_analysis, context.skills_extraction)
        culture_future = executor.submit(analyze_culture_fit, resume_text, context.jd_analysis)
        context.semantic_result = semantic_future.result()
        context.experience_result = experience_future.result()
        context.culture_result = culture_future.result()

    context.ranking_result = rank_candidate(
        jd_analysis=context.jd_analysis,
        skills_extraction=context.skills_extraction,
        screener_result=context.screener_result,
        semantic_result=context.semantic_result,
        experience_result=context.experience_result,
        culture_result=context.culture_result
    )

    eval_id = uuid.uuid4().hex
    PIPELINE_STORE[eval_id] = context
    return eval_id, context


def get_context(eval_id):
    return PIPELINE_STORE.get(eval_id)
