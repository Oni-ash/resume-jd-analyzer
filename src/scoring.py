"""
Answer scoring: overlap coefficient vs model answer + keyword coverage + length.
Returns per-question scores plus an aggregate.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import clean_text, tokenize, remove_stopwords, lemmatize

# Weights (must sum to 1.0)
W_OVERLAP = 0.5
W_KEYWORD = 0.3
W_LENGTH = 0.2

# Length scoring parameters (words)
MIN_WORDS = 15
TARGET_WORDS = 40

# Verdict thresholds
GOOD_THRESHOLD = 0.55
PARTIAL_THRESHOLD = 0.30


def _normalize_for_scoring(text: str) -> str:
    """Clean + tokenize + stopwords + lemmatize, return as string."""
    t = clean_text(text).lower()
    toks = tokenize(t)
    toks = remove_stopwords(toks)
    toks = lemmatize(toks)
    return " ".join(toks)


def _keyword_set(text: str) -> set[str]:
    return set(_normalize_for_scoring(text).split())


def _cosine(student: str, model: str) -> float:
    """TF-IDF cosine similarity (no IDF since corpus is only 2 docs). Info only."""
    if not student or not model:
        return 0.0
    a = _normalize_for_scoring(student)
    b = _normalize_for_scoring(model)
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(use_idf=False)
        matrix = vec.fit_transform([a, b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def _overlap_coefficient(student: str, model: str) -> float:
    """
    |A ∩ B| / min(|A|, |B|)
    How much of the shorter text is covered by the longer.
    Better than cosine for short asymmetric comparisons.
    """
    s = set(_normalize_for_scoring(student).split())
    m = set(_normalize_for_scoring(model).split())
    if not s or not m:
        return 0.0
    return len(s & m) / min(len(s), len(m))


def _keyword_coverage(student: str, model: str) -> float:
    """Fraction of model answer keywords present in student answer."""
    m = _keyword_set(model)
    if not m:
        return 0.0
    s = _keyword_set(student)
    return len(s & m) / len(m)


def _length_score(student: str) -> float:
    """
    Reward reasonable length. Penalize very short answers.
    - 0 words      -> 0.0
    - 15 words     -> 0.5
    - 40+ words    -> 1.0
    """
    wc = len(student.split())
    if wc <= 0:
        return 0.0
    if wc >= TARGET_WORDS:
        return 1.0
    if wc < MIN_WORDS:
        return 0.5 * (wc / MIN_WORDS)
    return 0.5 + 0.5 * (wc - MIN_WORDS) / (TARGET_WORDS - MIN_WORDS)


def _verdict(score: float) -> str:
    if score >= GOOD_THRESHOLD:
        return "good"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "poor"


def score_answer(student_answer: str, model_answer: str) -> dict:
    """
    Score one answer. Returns dict with component scores + verdict.
    """
    overlap = _overlap_coefficient(student_answer, model_answer)
    cosine = _cosine(student_answer, model_answer)  # info only
    kw = _keyword_coverage(student_answer, model_answer)
    length = _length_score(student_answer)

    total = W_OVERLAP * overlap + W_KEYWORD * kw + W_LENGTH * length

    return {
        "overlap": round(overlap, 4),
        "cosine": round(cosine, 4),
        "keyword_coverage": round(kw, 4),
        "length_score": round(length, 4),
        "score": round(total, 4),
        "verdict": _verdict(total),
    }


def score_interview(interview_results: list[dict]) -> dict:
    """
    Score all answers from run_interview().
    Returns dict with per-question scores, per-skill averages, and aggregate.
    """
    scored: list[dict] = []

    for r in interview_results:
        s = score_answer(r["student_answer"], r["model_answer"])
        scored.append({
            "skill": r["skill"],
            "difficulty": r["difficulty"],
            "question": r["question"],
            "student_answer": r["student_answer"],
            "model_answer": r["model_answer"],
            **s,
        })

    if not scored:
        return {
            "per_question": [],
            "by_skill": {},
            "aggregate": {"avg_score": 0.0, "verdict": "poor"},
        }

    avg = sum(q["score"] for q in scored) / len(scored)

    by_skill: dict[str, float] = {}
    counts: dict[str, int] = {}
    for q in scored:
        by_skill[q["skill"]] = by_skill.get(q["skill"], 0.0) + q["score"]
        counts[q["skill"]] = counts.get(q["skill"], 0) + 1
    by_skill_avg = {k: round(v / counts[k], 4) for k, v in by_skill.items()}

    return {
        "per_question": scored,
        "by_skill": by_skill_avg,
        "aggregate": {
            "avg_score": round(avg, 4),
            "verdict": _verdict(avg),
        },
    }


if __name__ == "__main__":
    # Quick manual test
    student = (
        "Dropout is a regularization technique where a random fraction "
        "of neurons are turned off during training to prevent overfitting."
    )
    model = (
        "Dropout randomly zeroes a fraction of neurons during training. "
        "This prevents co-adaptation and acts as regularization reducing overfitting. "
        "At inference time dropout is disabled."
    )
    print(score_answer(student, model))