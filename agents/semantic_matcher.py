"""Semantic Matcher agent - embedding-based similarity between JD and resume.

Not an LLM call: this runs a local embedding model (see embeddings.py) since
similarity scoring is a well-defined numeric task better suited to vector
math than to an LLM's judgment - it's also deterministic and free to run.
"""

from embeddings import semantic_similarity


def match_semantics(jd_text, resume_text):
    score = semantic_similarity(jd_text, resume_text)
    return {"semantic_similarity_score": score}
