# Candidate–Job Matching Tool with Boolean Search Generator

Healthcare-recruiting assistant: parses a job description, ranks candidates, flags gaps,
generates a Boolean sourcing string and screening-call questions.

**Data:** 200 fully synthetic candidate profiles (`generate_candidates.py`). No real people.

## How it works
1. `parse_jd` extracts role, specialty, certifications, shift and minimum experience (rule-based).
2. `rank` scores = 50% TF-IDF (1–2 grams) cosine similarity of JD vs profile + 50% requirement coverage; wrong role type is penalised.
3. `screen` lists per-candidate gaps (missing certs, shift, experience).
4. `boolean_string` builds a LinkedIn/job-board search string; `screening_questions` drafts a call script.

## Run
```
pip install -r requirements.txt
python generate_candidates.py
python matcher.py --jd sample_jd.txt --top 5
streamlit run app.py      # optional UI
```
## Limitations / next steps
Rule-based JD parsing; could be replaced with an NER/LLM extractor and evaluated against recruiter-labelled matches.
