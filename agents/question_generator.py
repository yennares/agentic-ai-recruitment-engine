"""Question Generator agent - produces role-specific interview questions.

On-demand (triggered from the results dashboard, not run automatically).
Deliberately targeted at the gaps the pipeline already found (missing
must-have skills, experience concerns, red flags) rather than generic
questions - this is the pipeline's agents feeding one candidate's specific
findings into a downstream agent. Fast model tier: generative, not evaluative.
"""

from groq_client import generate_json, MODEL_FAST, MODEL_DEEP

SYSTEM_PROMPT = """You are a Question Generator agent in an HR recruiting pipeline.
Generate 8-12 targeted interview questions for this specific candidate and role,
prioritizing questions that probe the gaps and open concerns identified by the
pipeline, not generic questions the resume already answers.

Return a JSON object with exactly this key:
{
  "questions": [
    {"question": "string", "targets": "string - short phrase naming the gap/skill/concern this probes"}
  ]
}"""


def generate_questions(jd_analysis, skills_extraction, screener_result, experience_result):
    candidate_skills = [s.lower() for s in skills_extraction.get("technical_skills", [])]
    gaps = [
        req for req in jd_analysis.get("must_have", [])
        if not any(req.lower() in skill or skill in req.lower() for skill in candidate_skills)
    ]
    user_prompt = (
        f"Role: {jd_analysis.get('role_title')} ({jd_analysis.get('seniority_level')})\n"
        f"Responsibilities: {jd_analysis.get('responsibilities')}\n"
        f"Must-have requirements not clearly covered by the resume: {gaps or 'None identified'}\n"
        f"Screener red flags: {screener_result.get('red_flags', [])}\n"
        f"Experience notes: {experience_result.get('notes', '')}\n"
        f"Candidate's listed skills: {skills_extraction.get('technical_skills', [])}"
    )
    return generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        models=(MODEL_FAST, MODEL_DEEP)
    )
