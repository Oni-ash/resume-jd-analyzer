"""
Gap analysis: compare resume skills vs JD skills, weighted by importance.
Handles OR-groups like 'AWS or GCP' as single requirements.
Applies skill implication (e.g., MySQL implies SQL).
"""

import os
import csv
import re

from src.skill_extract import (
    canonicalize_set,
    canonicalize_skill,
    expand_implied,
)

_DEFAULT_PLACEMENT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "placement_data.csv",
)

DEFAULT_WEIGHT = 0.5


def load_weights(path: str = _DEFAULT_PLACEMENT_PATH) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    weights: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill = (row.get("skill") or "").strip().lower()
            weight_str = (row.get("weight") or "").strip()
            if not skill or not weight_str:
                continue
            try:
                weights[skill] = float(weight_str)
            except ValueError:
                continue
    return weights


def _weight(skill: str, weights: dict[str, float]) -> float:
    return weights.get(skill, DEFAULT_WEIGHT)


def find_or_groups(text: str, skills: set[str]) -> list[set[str]]:
    """
    Find skills connected by 'or' or '/' in the text.
    E.g. 'TensorFlow or PyTorch' -> {tensorflow, pytorch}
    Returns list of skill-sets. Each set is one OR group.
    """
    text_lower = text.lower()
    groups: list[set[str]] = []

    for m in re.finditer(r"\s+(?:or|/)\s+", text_lower):
        left = text_lower[:m.start()]
        right = text_lower[m.end():]

        left_words = re.findall(r"[a-z][a-z\.\+\-#]*", left)
        right_words = re.findall(r"[a-z][a-z\.\+\-#]*", right)

        left_match = None
        for n in range(1, 4):
            if len(left_words) >= n:
                candidate = " ".join(left_words[-n:])
                if candidate in skills:
                    left_match = candidate
                    break

        right_match = None
        for n in range(1, 4):
            if len(right_words) >= n:
                candidate = " ".join(right_words[:n])
                if candidate in skills:
                    right_match = candidate
                    break

        if left_match and right_match and left_match != right_match:
            groups.append({left_match, right_match})

    # Merge overlapping groups (e.g. X or Y or Z)
    merged: list[set[str]] = []
    for g in groups:
        merged_into = None
        for mg in merged:
            if mg & g:
                merged_into = mg
                break
        if merged_into is not None:
            merged_into |= g
        else:
            merged.append(set(g))

    return merged


def compute_gap(
    resume_skills: set[str],
    jd_skills: set[str],
    weights: dict[str, float] | None = None,
    or_groups: list[set[str]] | None = None,
) -> dict:
    """
    Compare resume skills to JD skills, collapsing OR-groups.
    Each OR group = ONE requirement. Satisfied if any member present.
    """
    if weights is None:
        weights = load_weights()
    if or_groups is None:
        or_groups = []

    # Canonicalize so 'gcp' and 'google cloud' merge
    resume_skills = canonicalize_set(resume_skills)
    jd_skills = canonicalize_set(jd_skills)
    or_groups = [{canonicalize_skill(s) for s in g} for g in or_groups]

    grouped_members: set[str] = set()
    for g in or_groups:
        grouped_members |= g

    matched: set[str] = set()
    missing: set[str] = set()
    group_info: list[dict] = []

    for g in or_groups:
        # Representative = highest-weight member
        rep = max(g, key=lambda s: weights.get(s, DEFAULT_WEIGHT))
        satisfied = bool(g & resume_skills)

        group_info.append({
            "group": sorted(g),
            "representative": rep,
            "weight": weights.get(rep, DEFAULT_WEIGHT),
            "satisfied": satisfied,
            "resume_has": sorted(g & resume_skills),
        })

        if satisfied:
            matched.add(rep)
        else:
            missing.add(rep)

    ungrouped = jd_skills - grouped_members
    matched |= (ungrouped & resume_skills)
    missing |= (ungrouped - resume_skills)

    total_weight = sum(_weight(s, weights) for s in (matched | missing))
    matched_weight = sum(_weight(s, weights) for s in matched)
    match_score = matched_weight / total_weight if total_weight > 0 else 0.0

    missing_ranked = sorted(missing, key=lambda s: _weight(s, weights), reverse=True)
    extra = resume_skills - jd_skills

    return {
        "match_score": round(match_score, 4),
        "matched_skills": sorted(matched),
        "missing_skills": missing_ranked,
        "extra_skills": sorted(extra),
        "or_groups": group_info,
        "counts": {
            "total_jd_requirements": len(matched) + len(missing),
            "total_resume_skills": len(resume_skills),
            "total_matched": len(matched),
            "total_missing": len(missing),
            "total_extra": len(extra),
        },
    }


if __name__ == "__main__":
    import sys
    from src.extract import extract_text
    from src.preprocess import clean_text, fix_glued_words
    from src.skill_extract import (
        extract_skills,
        flatten_skills,
        load_skills,
    )

    if len(sys.argv) < 3:
        print("Usage: python -m src.gap_analysis <resume> <jd>")
        sys.exit(1)

    skills_dict = load_skills()
    weights = load_weights()

    resume_text = fix_glued_words(clean_text(extract_text(sys.argv[1])))
    jd_text = fix_glued_words(clean_text(extract_text(sys.argv[2])))

    resume_skills_raw = flatten_skills(extract_skills(resume_text, skills_dict))
    resume_skills, implied_by = expand_implied(resume_skills_raw)

    jd_skills_raw = flatten_skills(
        extract_skills(jd_text, skills_dict), apply_aliases=False
    )

    or_groups = find_or_groups(jd_text, jd_skills_raw)
    result = compute_gap(resume_skills, jd_skills_raw, weights, or_groups)

    print(f"Match score: {result['match_score'] * 100:.1f}%")
    print(
        f"Matched {result['counts']['total_matched']} / "
        f"{result['counts']['total_jd_requirements']} JD requirements"
    )

    if result["or_groups"]:
        print("\nOR groups detected:")
        for info in result["or_groups"]:
            status = "SATISFIED" if info["satisfied"] else "MISSING"
            opts = " or ".join(info["group"])
            have = (
                f" (resume has: {', '.join(info['resume_has'])})"
                if info["resume_has"] else ""
            )
            print(f"  - [{status}] {opts}{have}")

    print("\nMissing skills (ranked by importance):")
    for s in result["missing_skills"]:
        w = weights.get(s, DEFAULT_WEIGHT)
        print(f"  - {s}  (w={w:.2f})")

    print("\nMatched skills:")
    for s in result["matched_skills"]:
        w = weights.get(s, DEFAULT_WEIGHT)
        note = ""
        if s in implied_by:
            sources = ", ".join(sorted(implied_by[s]))
            note = f"  [implied by: {sources}]"
        print(f"  - {s}  (w={w:.2f}){note}")

    if result["extra_skills"]:
        print("\nExtra skills on resume (not in JD):")
        for s in result["extra_skills"]:
            print(f"  - {s}")