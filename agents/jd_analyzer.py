"""JD Analyzer agent - extracts structured requirements from a job description.

Runs on the fast model tier: this is a straightforward extraction task where
the fast model is reliably accurate, so there's no need to pay the latency/
cost premium of the deep model for the primary attempt.
"""

from groq_client import generate_json, MODEL_FAST, MODEL_DEEP

SYSTEM_PROMPT = """You are a JD Analyzer agent in an HR recruiting pipeline.
Read the job description and extract its structured requirements.

Return a JSON object with exactly these keys:
{
  "role_title": "string - the job title",
  "seniority_level": "one of: Entry, Junior, Mid, Senior, Lead, Executive",
  "must_have": ["list of strings - mandatory skills, qualifications or experience"],
  "nice_to_have": ["list of strings - preferred but optional skills or qualifications"],
  "responsibilities": ["list of strings - key day-to-day responsibilities"]
}
Be concise - each list item should be a short phrase, not a sentence."""


def analyze_jd(jd_text):
    return generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Job Description:\n{jd_text}",
        models=(MODEL_FAST, MODEL_DEEP)
    )
