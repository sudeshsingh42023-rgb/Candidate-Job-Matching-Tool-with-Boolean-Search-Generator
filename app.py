"""Optional UI: streamlit run app.py"""
import pandas as pd, streamlit as st
from matcher import parse_jd, boolean_string, rank, screening_questions
st.title("Candidate–Job Matcher")
jd_text = st.text_area("Paste job description", open("sample_jd.txt").read(), height=150)
if st.button("Match"):
    jd = parse_jd(jd_text); df = pd.read_csv("candidates_synthetic.csv")
    st.subheader("Boolean search string"); st.code(boolean_string(jd))
    st.subheader("Top candidates")
    st.dataframe(rank(jd, df).head(10)[["candidate_id", "role", "specialty", "years_exp", "score", "flags"]])
    st.subheader("Screening questions"); [st.write("• " + q) for q in screening_questions(jd)]
