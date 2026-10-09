"""
Mock interview: pick questions from a bank based on gap analysis,
ask the student in the terminal, collect answers, and score them.
"""

import os
import csv
import random

_DEFAULT_QUESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "questions.csv",
)

DEFAULT_N_MISSING = 3
DEFAULT_N_MATCHED = 2


def load_questions(path: str = _DEFAULT_QUESTIONS_PATH) -> dict[str, list[dict]]:
    """
    Load question bank grouped by skill.
    Returns {skill: [{difficulty, question, model_answer}, ...]}
    """
    if not os.path.exists(path):
        return {}

    bank: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill = (row.get("skill") or "").strip().lower()
            if not skill:
                continue
            bank.setdefault(skill, []).append({
                "skill": skill,
                "difficulty": (row.get("difficulty") or "").strip().lower(),
                "question": (row.get("question") or "").strip(),
                "model_answer": (row.get("model_answer") or "").strip(),
            })
    return bank


def pick_questions(
    missing_skills: list[str],
    matched_skills: list[str],
    bank: dict[str, list[dict]],
    n_missing: int = DEFAULT_N_MISSING,
    n_matched: int = DEFAULT_N_MATCHED,
    seed: int | None = None,
) -> list[dict]:
    """
    Pick questions:
    - Up to n_missing from missing skills (in order = highest weight first)
    - Up to n_matched from matched skills
    Skills without questions are skipped.
    """
    rng = random.Random(seed)
    picked: list[dict] = []
    used_skills: set[str] = set()

    def _pick_from(skill: str) -> dict | None:
        candidates = bank.get(skill)
        if not candidates:
            return None
        return rng.choice(candidates)

    missing_count = 0
    for s in missing_skills:
        if missing_count >= n_missing:
            break
        if s in used_skills:
            continue
        q = _pick_from(s)
        if q:
            picked.append(q)
            used_skills.add(s)
            missing_count += 1

    matched_count = 0
    for s in matched_skills:
        if matched_count >= n_matched:
            break
        if s in used_skills:
            continue
        q = _pick_from(s)
        if q:
            picked.append(q)
            used_skills.add(s)
            matched_count += 1

    return picked


def run_interview(questions: list[dict], input_fn=input) -> list[dict]:
    """
    Ask each question and collect answers.
    input_fn allows tests to inject fake input.
    Returns list of {skill, question, model_answer, student_answer}.
    """
    results: list[dict] = []
    total = len(questions)

    print("\n" + "=" * 60)
    print(" MOCK INTERVIEW")
    print("=" * 60)
    print(f"{total} questions. Answer as best you can.\n")

    for i, q in enumerate(questions, 1):
        print(f"Q{i}/{total}  [{q['skill']} - {q['difficulty']}]")
        print(q["question"])
        try:
            answer = input_fn("Your answer: ").strip()
        except EOFError:
            answer = ""

        results.append({
            "skill": q["skill"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "model_answer": q["model_answer"],
            "student_answer": answer,
        })
        print()

    return results


if __name__ == "__main__":
    import sys
    from src.extract import extract_text
    from src.preprocess import clean_text, fix_glued_words
    from src.skill_extract import (
        extract_skills,
        flatten_skills,
        load_skills,
        expand_implied,
    )
    from src.gap_analysis import (
        compute_gap,
        find_or_groups,
        load_weights,
    )
    from src.scoring import score_interview

    if len(sys.argv) < 3:
        print("Usage: python -m src.interview <resume> <jd>")
        sys.exit(1)

    skills_dict = load_skills()
    weights = load_weights()

    resume_text = fix_glued_words(clean_text(extract_text(sys.argv[1])))
    jd_text = fix_glued_words(clean_text(extract_text(sys.argv[2])))

    resume_skills_raw = flatten_skills(extract_skills(resume_text, skills_dict))
    resume_skills, _ = expand_implied(resume_skills_raw)

    jd_skills_raw = flatten_skills(
        extract_skills(jd_text, skills_dict), apply_aliases=False
    )

    or_groups = find_or_groups(jd_text, jd_skills_raw)
    result = compute_gap(resume_skills, jd_skills_raw, weights, or_groups)

    print(f"Gap score: {result['match_score'] * 100:.1f}%")
    print(f"Missing: {', '.join(result['missing_skills'][:5])}")
    print(f"Matched: {', '.join(result['matched_skills'][:5])}")

    bank = load_questions()
    questions = pick_questions(
        result["missing_skills"],
        result["matched_skills"],
        bank,
    )

    if not questions:
        print("No questions available for these skills. Add to data/questions.csv")
        sys.exit(0)

    answers = run_interview(questions)
    scored = score_interview(answers)

    print("=" * 60)
    print(" SCORING")
    print("=" * 60)

    for i, q in enumerate(scored["per_question"], 1):
        print(
            f"\nQ{i} [{q['skill']}] — {q['verdict'].upper()}  "
            f"({q['score'] * 100:.0f}%)"
        )
        print(
            f"  overlap: {q['overlap']:.2f}  "
            f"keyword: {q['keyword_coverage']:.2f}  "
            f"length: {q['length_score']:.2f}"
        )

    print(
        f"\nAggregate: {scored['aggregate']['avg_score'] * 100:.1f}%  "
        f"({scored['aggregate']['verdict']})"
    )

    print("\nBy skill:")
    for skill, sc in sorted(scored["by_skill"].items(), key=lambda x: -x[1]):
        print(f"  {skill:30s} {sc * 100:5.1f}%")