"""
Skill extraction using a CSV skill dictionary + substring matching.
"""

import os
import re
import csv

_DEFAULT_SKILLS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "skills.csv",
)

# Alternate name -> canonical name
SKILL_ALIASES: dict[str, str] = {
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "llm": "large language models",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "pentest": "penetration testing",
    "pentesting": "penetration testing",
    "k8s": "kubernetes",
}

# Skill A implies skill B (soft match)
SKILL_IMPLIES: dict[str, set[str]] = {
    "mysql": {"sql"},
    "postgresql": {"sql"},
    "sqlite": {"sql"},
    "oracle": {"sql"},
    "scikit-learn": {"machine learning"},
    "tensorflow": {"deep learning", "machine learning"},
    "pytorch": {"deep learning", "machine learning"},
    "keras": {"deep learning", "machine learning"},
    "xgboost": {"machine learning"},
    "lightgbm": {"machine learning"},
    "bert": {"nlp", "natural language processing"},
    "gpt": {"nlp", "natural language processing"},
    "pandas": {"python"},
    "numpy": {"python"},
    "django": {"python", "web"},
    "flask": {"python", "web"},
    "fastapi": {"python", "web"},
    "react": {"javascript", "web"},
    "angular": {"javascript", "typescript", "web"},
    "vue": {"javascript", "web"},
    "node.js": {"javascript", "web"},
    "express": {"javascript", "web"},
}


def canonicalize_skill(skill: str) -> str:
    return SKILL_ALIASES.get(skill, skill)


def canonicalize_set(skills: set[str]) -> set[str]:
    return {canonicalize_skill(s) for s in skills}


def expand_implied(skills: set[str]) -> tuple[set[str], dict[str, set[str]]]:
    """
    Expand skill set with implied skills.
    Returns (expanded_set, {implied_skill: {sources that implied it}}).
    """
    expanded = set(skills)
    implied_by: dict[str, set[str]] = {}

    for s in skills:
        for implied in SKILL_IMPLIES.get(s, set()):
            if implied not in skills:
                expanded.add(implied)
                implied_by.setdefault(implied, set()).add(s)

    return expanded, implied_by


def load_skills(path: str = _DEFAULT_SKILLS_PATH) -> dict[str, str]:
    skills = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill = row["skill"].strip().lower()
            category = row["category"].strip().lower()
            if skill:
                skills[skill] = category
    return skills


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_skills(text: str, skills: dict[str, str] | None = None) -> dict[str, list[str]]:
    if skills is None:
        skills = load_skills()

    normalized = _normalize(text)
    glued_candidates = [
        tok for tok in re.findall(r"[a-z]+", normalized) if len(tok) >= 12
    ]

    found: dict[str, list[str]] = {}

    for skill, category in skills.items():
        matched = False

        if " " in skill:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, normalized):
                matched = True
        else:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, normalized):
                matched = True
            else:
                for tok in glued_candidates:
                    if skill in tok:
                        matched = True
                        break

        if matched:
            found.setdefault(category, []).append(skill)

    return found


def flatten_skills(
    found: dict[str, list[str]], apply_aliases: bool = True
) -> set[str]:
    flat = {s for skills in found.values() for s in skills}
    if apply_aliases:
        flat = canonicalize_set(flat)
    return flat


if __name__ == "__main__":
    import sys
    from src.extract import extract_text
    from src.preprocess import clean_text, fix_glued_words

    if len(sys.argv) < 2:
        print("Usage: python -m src.skill_extract <file_path>")
        sys.exit(1)

    raw = extract_text(sys.argv[1])
    cleaned = fix_glued_words(clean_text(raw))

    found = extract_skills(cleaned)
    flat = flatten_skills(found)

    print(f"--- {len(flat)} skills found ---")
    for category, skills in sorted(found.items()):
        print(f"\n[{category}]")
        for s in sorted(skills):
            print(f"  - {s}")