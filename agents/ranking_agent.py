"""Ranking Agent - aggregates every other agent's output into a final score.

Deliberately hybrid: the numeric score is computed deterministically in
Python from the other agents' structured outputs, not asked of the LLM. An
LLM asked to "give a score" will happily produce a plausible-looking number
that isn't reproducible or auditable. The LLM call here is used only for what
LLMs are actually good at - turning the structured evidence into a clear,
human-readable rationale for the recruiter.
"""

import re

from groq_client import generate_json, MODEL_FAST, MODEL_DEEP

# Filler words stripped before token-overlap matching so "PostgreSQL
# experience" (requirement) still matches "PostgreSQL" (skill) even though
# neither string is a substring of the other.
_STOPWORDS = {
    "experience", "experienced", "years", "year", "with", "and", "or", "in",
    "of", "the", "a", "an", "to", "for", "using", "strong", "solid",
    "proficiency", "proficient", "knowledge", "skills", "skill", "background",
    "ability", "familiarity", "familiar", "working", "hands", "on",
}

WEIGHTS = {
    "skills_coverage": 0.30,
    "semantic_similarity": 0.20,
    "experience_progression": 0.30,
    "culture_fit": 0.20,
}

SYSTEM_PROMPT = """You are a Ranking Agent in an HR recruiting pipeline. You are
given the structured findings from several specialist agents plus a final score
that has already been calculated deterministically. Do not change the score.
Write a clear, concise rationale a hiring manager can read in 15 seconds,
referencing the strongest and weakest points of the evidence provided.

Return a JSON object with exactly this key:
{
  "rationale": "string - 3-5 sentences explaining the score and recommendation"
}"""


def _tokenize(text):
    words = re.findall(r"[a-z0-9+#]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _skills_coverage(jd_must_have, candidate_skills):
    """A requirement counts as covered if either signal fires:

    1. Word-token overlap (drops filler words) - catches phrase-level
       differences like "PostgreSQL experience" vs. skill "PostgreSQL",
       where neither string is a substring of the other but they share the
       meaningful word.
    2. Whole-string substring containment - catches compound tokens like
       requirement "React.js frontend framework" vs. skill "React", where
       tokenizing on non-alphanumerics would split "React.js" away from
       "React" and miss the match that a raw substring check still catches.

    Either alone under-counts realistic JD/resume phrasing; combined they're
    more forgiving without needing an LLM call for what's fundamentally a
    string-matching problem.
    """
    if not jd_must_have:
        return 100.0
    skills_lower = [s.lower() for s in candidate_skills]
    skill_tokens = set()
    for skill in skills_lower:
        skill_tokens |= _tokenize(skill)
    if not skill_tokens:
        return 0.0
    matched = 0
    for requirement in jd_must_have:
        requirement_lower = requirement.lower()
        token_hit = bool(_tokenize(requirement) & skill_tokens)
        substring_hit = any(skill in requirement_lower or requirement_lower in skill for skill in skills_lower)
        if token_hit or substring_hit:
            matched += 1
    return round(matched / len(jd_must_have) * 100, 2)


def _shortlist_label(score):
    if score >= 75:
        return "Shortlist"
    if score >= 50:
        return "Maybe"
    return "Reject"


def rank_candidate(jd_analysis, skills_extraction, screener_result, semantic_result, experience_result, culture_result):
    skills_coverage = _skills_coverage(
        jd_analysis.get("must_have", []),
        skills_extraction.get("technical_skills", [])
    )
    semantic_score = semantic_result.get("semantic_similarity_score", 0)
    progression_score = experience_result.get("progression_score", 0)
    culture_score = culture_result.get("values_alignment_score", 0)

    weighted_score = (
        skills_coverage * WEIGHTS["skills_coverage"] +
        semantic_score * WEIGHTS["semantic_similarity"] +
        progression_score * WEIGHTS["experience_progression"] +
        culture_score * WEIGHTS["culture_fit"]
    )

    # Each screener red flag costs 20 points, capped at a 60-point total
    # deduction - a proportional penalty rather than a hard veto. A failed
    # screen with no itemized red flag still costs at least one flag's worth,
    # so "failed the screen" always costs something even if the screener
    # didn't break the reason into a list. This replaces an earlier design
    # that hard-capped the score at 45 on any failed screen: that made a
    # single missing requirement an automatic "Reject" regardless of how
    # strong everything else was, which didn't match how a human recruiter
    # would actually weigh a borderline-but-otherwise-excellent candidate.
    red_flags = screener_result.get("red_flags", [])
    flag_count = len(red_flags)
    screener_failed = not screener_result.get("pass_initial_screen", True)
    if screener_failed and flag_count == 0:
        flag_count = 1
    penalty = min(flag_count * 20, 60)
    final_score = round(max(0, weighted_score - penalty), 1)

    score_breakdown = {
        "skills_coverage": skills_coverage,
        "semantic_similarity": semantic_score,
        "experience_progression": progression_score,
        "culture_fit": culture_score,
    }

    user_prompt = (
        f"Final score (already calculated, do not change): {final_score}/100\n"
        f"Weighted score from the four sub-scores before any penalty: {weighted_score:.1f}/100\n"
        f"Score breakdown: {score_breakdown}\n"
        f"Screener red flags: {red_flags}\n"
        f"Screener passed initial screen: {screener_result.get('pass_initial_screen')}\n"
        f"Points deducted for red flags: {penalty}\n"
        f"Experience notes: {experience_result.get('notes')}\n"
        f"Culture fit notes: {culture_result.get('notes')}\n"
        f"JD must-have requirements: {jd_analysis.get('must_have')}\n"
        f"Candidate skills: {skills_extraction.get('technical_skills')}"
    )
    llm_result = generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        models=(MODEL_DEEP, MODEL_FAST)
    )

    return {
        "final_score": final_score,
        "score_breakdown": score_breakdown,
        "shortlist_recommendation": _shortlist_label(final_score),
        "weighted_score_before_penalty": round(weighted_score, 1),
        "red_flag_penalty": penalty,
        "screener_failed": screener_failed,
        "rationale": llm_result.get("rationale", "")
    }
