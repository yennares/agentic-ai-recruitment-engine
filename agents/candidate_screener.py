"""Candidate Screener agent - fast initial pass/fail gate and red-flag detection.

Runs on the deep model tier first: false negatives here (missing a real red
flag) are costlier than the latency/cost of the stronger model, so accuracy
is prioritized over speed for this one agent.
"""

from groq_client import generate_json, MODEL_FAST, MODEL_DEEP

SYSTEM_PROMPT = """You are a Candidate Screener agent in an HR recruiting pipeline.
Perform an initial screen of the resume against the job description's mandatory
requirements. Look for concrete red flags only - do not speculate.

Red flags to check for (only report ones you can actually see evidence of):
- Unexplained employment gaps longer than 12 months
- Very short average tenure across multiple recent roles (job hopping)
- Missing mandatory certifications or qualifications explicitly required by the JD
- Clear mismatches between stated seniority and actual experience

Return a JSON object with exactly these keys:
{
  "pass_initial_screen": true or false,
  "red_flags": ["list of strings - each a specific, evidence-based concern; empty list if none"],
  "reasons": ["list of strings - brief justification for the pass/fail decision"]
}"""


def screen_candidate(resume_text, jd_must_have):
    must_have_text = ", ".join(jd_must_have) if jd_must_have else "Not specified"
    user_prompt = (
        f"Job Description mandatory requirements: {must_have_text}\n\n"
        f"Resume:\n{resume_text}"
    )
    return generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        models=(MODEL_DEEP, MODEL_FAST)
    )
