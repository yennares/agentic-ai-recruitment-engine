"""Culture Fit Analyzer agent - heuristic signal from resume language.

This is the most subjective agent in the pipeline and the most bias-prone:
inferring "culture fit" from resume wording can easily reward familiar
phrasing or background over substance. It is treated as a soft, low-weight
signal in the Ranking Agent (see ranking_agent.py) and is always returned
with supporting evidence so a human reviewer can sanity-check it rather than
trust it blindly. The UI surfaces this limitation next to the score.
"""

from groq_client import generate_json, MODEL_FAST, MODEL_DEEP

SYSTEM_PROMPT = """You are a Culture Fit Analyzer agent in an HR recruiting pipeline.
Based ONLY on the language, tone and values expressed in the resume (initiative,
collaboration, ownership, communication style) and how they relate to the
responsibilities in the job description, give a soft alignment signal.

This is a supporting signal, not a hiring decision - be conservative and cite
concrete evidence rather than inferring personality traits.

Return a JSON object with exactly these keys:
{
  "values_alignment_score": number from 0 to 100,
  "evidence": ["list of strings - short quotes or paraphrases from the resume that informed the score"],
  "notes": "string - 1-2 sentence caveat-aware summary"
}"""


def analyze_culture_fit(resume_text, jd_analysis):
    user_prompt = (
        f"Job responsibilities: {jd_analysis.get('responsibilities')}\n\n"
        f"Resume:\n{resume_text}"
    )
    return generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        models=(MODEL_DEEP, MODEL_FAST)
    )
