"""Candidate-job matching + Boolean search generator for healthcare recruiting.
Usage: python matcher.py --jd sample_jd.txt --top 5
"""
import argparse, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROLES = {"RN": ["rn", "registered nurse"], "LPN": ["lpn", "licensed practical nurse"],
         "Respiratory Therapist": ["respiratory therapist", "rrt"],
         "Physical Therapist": ["physical therapist", "dpt"]}
SPECIALTIES = {"ICU": ["icu", "critical care", "intensive care"],
               "ER": ["er", "emergency", "trauma"], "Med-Surg": ["med-surg", "med surg", "medical surgical"],
               "OR": ["operating room", " or ", "perioperative"], "Telemetry": ["telemetry", "cardiac"],
               "Pediatrics": ["pediatric", "peds", "nicu"]}
CERTS = ["BLS", "ACLS", "PALS", "NIHSS", "TNCC"]
SHIFTS = ["day", "night", "rotating", "flexible"]

def parse_jd(text: str) -> dict:
    t = " " + text.lower() + " "
    find = lambda d: next((k for k, v in d.items() if any(w in t for w in v)), None)
    yrs = re.search(r"(\d+)\+?\s*(?:years|yrs)", t)
    return {"role": find(ROLES), "specialty": find(SPECIALTIES),
            "certs": [c for c in CERTS if c.lower() in t],
            "shift": next((s for s in SHIFTS if f"{s} shift" in t), None),
            "min_years": int(yrs.group(1)) if yrs else 0, "text": text}

def boolean_string(jd: dict) -> str:
    parts = []
    if jd["role"]:
        parts.append("(" + " OR ".join(f'"{w}"' for w in ROLES[jd["role"]]) + ")")
    if jd["specialty"]:
        syn = [w.strip() for w in SPECIALTIES[jd["specialty"]] if len(w.strip()) > 2] + [jd["specialty"]]
        syn = list({w.lower(): w.lower() for w in syn}.values())
        parts.append("(" + " OR ".join(f'"{w}"' for w in syn) + ")")
    if jd["certs"]:
        parts.append("(" + " OR ".join(jd["certs"]) + ")")
    q = " AND ".join(parts)
    return q + ' NOT ("student" OR "intern")' if q else ""

def screen(cand: pd.Series, jd: dict) -> list:
    flags = []
    if jd["role"] and cand["role"] != jd["role"]: flags.append(f"role mismatch ({cand['role']})")
    if jd["specialty"] and cand["specialty"] != jd["specialty"]: flags.append(f"specialty mismatch ({cand['specialty']})")
    missing = [c for c in jd["certs"] if c not in cand["certs"].split("|")]
    if missing: flags.append("missing certs: " + ", ".join(missing))
    if cand["years_exp"] < jd["min_years"]: flags.append(f"only {cand['years_exp']} yrs (needs {jd['min_years']})")
    if jd["shift"] and cand["shift"] not in (jd["shift"], "flexible"): flags.append(f"shift preference: {cand['shift']}")
    return flags

def rank(jd: dict, df: pd.DataFrame) -> pd.DataFrame:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform(list(df["summary"]) + [jd["text"]])
    df = df.copy()
    df["text_sim"] = cosine_similarity(m[-1], m[:-1]).ravel()
    checks = []
    for _, c in df.iterrows():
        n_checks = sum(bool(x) for x in [jd["role"], jd["specialty"], jd["certs"], jd["shift"], jd["min_years"]])
        checks.append(1.0 if n_checks == 0 else 1 - min(len(screen(c, jd)), n_checks) / n_checks)
    df["req_coverage"] = checks
    df["score"] = (0.5 * df["text_sim"] / max(df["text_sim"].max(), 1e-9) + 0.5 * df["req_coverage"])
    if jd["role"]:  # wrong licence type is a hard mismatch in healthcare staffing
        df.loc[df["role"] != jd["role"], "score"] *= 0.5
    df["score"] = df["score"].round(3)
    df["flags"] = [("; ".join(screen(c, jd)) or "none") for _, c in df.iterrows()]
    return df.sort_values("score", ascending=False)

def screening_questions(jd: dict) -> list:
    q = ["Are you currently licensed and in good standing? In which states?",
         "What is your earliest start date and preferred assignment length?"]
    if jd["specialty"]: q.append(f"Tell me about your most recent {jd['specialty']} experience.")
    if jd["certs"]: q.append(f"Are your {', '.join(jd['certs'])} certifications current? Expiry dates?")
    if jd["shift"]: q.append(f"Are you comfortable with {jd['shift']} shifts?")
    return q

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jd", required=True)
    ap.add_argument("--candidates", default="candidates_synthetic.csv"); ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    jd = parse_jd(open(a.jd).read()); df = pd.read_csv(a.candidates)
    print("PARSED JD:", {k: v for k, v in jd.items() if k != "text"})
    print("\nBOOLEAN SEARCH STRING:\n" + boolean_string(jd))
    print(f"\nTOP {a.top} CANDIDATES:")
    print(rank(jd, df).head(a.top)[["candidate_id", "role", "specialty", "years_exp", "score", "flags"]].to_string(index=False))
    print("\nSCREENING QUESTIONS:"); [print(f"  {i+1}. {x}") for i, x in enumerate(screening_questions(jd))]
