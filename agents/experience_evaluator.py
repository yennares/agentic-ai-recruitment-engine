"""Experience Evaluator agent - judges years-of-experience fit and career
progression. Deep model tier: this requires nuanced reasoning about career
trajectory, not just extraction.
"""

from groq_client import generate_json, MODEL_FAST, MODEL_DEEP

SYSTEM_PROMPT = """You are an Experience Evaluator agent in an HR recruiting pipeline.
Given the job's seniority level and the candidate's role history, assess whether
their experience fits the role and how their career has progressed.

Return a JSON object with exactly these keys:
{
  "years_experience_match": true or false,
  "progression_score": number from 0 to 100 - how strong the candidate's career
    trajectory is (increasing responsibility, relevant role continuity, no
    unexplained regressions),
  "notes": "string - 2-3 sentence explanation of the assessment"
}"""


def evaluate_experience(jd_analysis, skills_extraction):
    user_prompt = (
        f"Job seniority level required: {jd_analysis.get('seniority_level')}\n"
        f"Job title: {jd_analysis.get('role_title')}\n\n"
        f"Candidate total years of experience: {skills_extraction.get('total_years_experience')}\n"
        f"Candidate role history: {skills_extraction.get('roles')}"
    )
    return generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        models=(MODEL_DEEP, MODEL_FAST)
    )
