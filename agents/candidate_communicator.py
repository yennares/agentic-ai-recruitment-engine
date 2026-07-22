"""Candidate Communicator agent - drafts candidate-facing status/scheduling
emails using the pipeline's actual findings rather than freeform typed input,
so the email reflects the real evaluation (role, recommendation, rationale)
instead of being generic. On-demand, deep model tier: this is candidate-facing
prose sent externally, so reliable formatting and instruction-following matter
more than latency here - testing showed the fast tier producing markdown
instead of HTML and occasionally appending meta-commentary about its own
output, which the deep tier does not do.
"""

from groq_client import generate_completion, MODEL_FAST, MODEL_DEEP

SYSTEM_PROMPT = "You are an assistant who drafts clear, professional candidate emails on behalf of a recruiter."


def draft_email(email_type, tone, jd_analysis, ranking_result, candidate_name=""):
    name_clause = f" for {candidate_name}" if candidate_name else ""
    prompt = (
        f"Draft a {email_type} email{name_clause} for the {jd_analysis.get('role_title', 'the')} role, "
        f"in a {tone} tone. The recruiter has explicitly chosen to send a '{email_type}' email - write "
        f"that email type regardless of the pipeline's recommendation below; a human recruiter can "
        f"choose to proceed with a candidate even if the automated evaluation was mixed, and the email "
        f"must not contradict the recruiter's chosen action (e.g. do not write a rejection if the "
        f"recruiter selected 'move to next round' or 'schedule interview').\n\n"
        f"Evaluation context (for tone/detail only, not for deciding the outcome): "
        f"pipeline recommendation was '{ranking_result.get('shortlist_recommendation')}' "
        f"with a fit score of {ranking_result.get('final_score')}/100. Rationale: {ranking_result.get('rationale')}. "
        f"Output ONLY the email itself as valid HTML using tags like <h2>, <p>, <strong> - no markdown "
        f"syntax (no **, no #), and no explanation, notes, or commentary about the email before or after it."
    )
    return generate_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        models=(MODEL_DEEP, MODEL_FAST)
    )
