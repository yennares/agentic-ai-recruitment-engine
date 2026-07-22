"""Skills Extractor agent - parses a resume for technical skills, certifications,
and work history. Fast model tier: structured extraction, not judgment.
"""

from groq_client import generate_json, MODEL_FAST, MODEL_DEEP

SYSTEM_PROMPT = """You are a Skills Extractor agent in an HR recruiting pipeline.
Read the resume and extract structured facts about the candidate.

Return a JSON object with exactly these keys:
{
  "technical_skills": ["list of strings - technical skills, tools, languages, frameworks"],
  "certifications": ["list of strings - named certifications or qualifications, empty list if none"],
  "total_years_experience": number - best estimate of total professional experience in years,
  "roles": [
    {"title": "string", "company": "string", "duration_years": number}
  ]
}
If a field cannot be determined, use an empty list or 0. Do not invent information not present in the resume."""


def extract_skills(resume_text):
    return generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Resume:\n{resume_text}",
        models=(MODEL_FAST, MODEL_DEEP)
    )
