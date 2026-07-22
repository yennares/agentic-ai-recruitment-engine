"""Local embedding-based semantic similarity.

Groq's API is chat/completion only - it does not serve an embeddings endpoint.
Rather than add a second paid API (OpenAI, Cohere, etc.) just for this one
metric, this runs a small ONNX embedding model on CPU via fastembed: free,
offline, no extra API key, and light enough not to bloat the Docker image
(no torch dependency, unlike sentence-transformers).
"""

import numpy as np
from fastembed import TextEmbedding

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None

# BAAI/bge-small cosine similarities for two *related but non-identical*
# documents (e.g. a well-matched resume and JD) typically land in the
# ~0.55-0.85 band rather than spanning the full [-1, 1] range. Clamping to
# that observed band before rescaling to 0-100 keeps the score intuitive
# (a strong match reads as a high number) instead of every real-world pair
# compressing into the 60-75 range.
_LOW = 0.2
_HIGH = 0.9


def _get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def _truncate(text, max_chars=6000):
    return text[:max_chars]


def semantic_similarity(text_a, text_b):
    """Cosine similarity between two texts' embeddings, rescaled to 0-100."""
    model = _get_model()
    vectors = list(model.embed([_truncate(text_a), _truncate(text_b)]))
    a, b = vectors[0], vectors[1]
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    clamped = max(_LOW, min(_HIGH, cosine))
    score = (clamped - _LOW) / (_HIGH - _LOW) * 100
    return round(score, 2)
