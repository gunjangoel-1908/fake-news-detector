import json
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from utils import clean

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown("""
<style>
.stButton>button {border-radius:10px; font-weight:600;}
.result-card {padding:1.1rem; border-radius:14px; font-size:1.4rem; font-weight:600; margin:1rem 0 0.5rem;}
.fake {background:#3b1219; color:#ff6b6b;}
.real {background:#10311f; color:#51cf66;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model.pkl")
    tfidf = joblib.load("artifacts/tfidf.pkl")
    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)
    return model, tfidf, metrics

model, tfidf, metrics = load_artifacts()

def predict(text):
    X = tfidf.transform([clean(text)])
    proba = model.predict_proba(X)[0]
    label = "REAL" if proba[1] >= 0.5 else "FAKE"
    row = X.toarray()[0] * model.coef_[0]
    names = tfidf.get_feature_names_out()
    idx = np.argsort(-np.abs(row))[:6]
    words = [names[i] for i in idx if row[i] != 0]
    return label, float(max(proba)), words

def get_api_key():
    try:
        return st.secrets["FACTCHECK_KEY"]
    except Exception:
        return None

def search_factchecks(query, lang):
    key = get_api_key()
    if not key:
        return None, "API key not configured."
    try:
        r = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query, "key": key, "languageCode": lang, "pageSize": 10},
            timeout=15)
        if r.status_code != 200:
            return None, f"API error ({r.status_code}). Check your API key."
        return r.json().get("claims", []), None
    except requests.RequestException:
        return None, "Network error. Try again."

FAKE_SAMPLE = ("SHOCKING: Scientists confirm that drinking hot water with lemon cures all diseases. "
               "Doctors do not want you to know this secret. Share this before it gets deleted, "
               "because the government is trying to hide the truth from everyone!")
REAL_SAMPLE = ("The Reserve Bank of India kept its repo rate unchanged on Friday, saying inflation "
               "remains within the target range and the growth outlook is stable. The central bank "
               "said it will continue to monitor global conditions before deciding on any change.")

def set_text(t):
    st.session_state.article = t

with st.sidebar:
    st.header("About")
    st.write("TF-IDF + Logistic Regression trained on about 38K news articles, "
             "plus Google Fact Check API lookup.")

st.title("Fake News Detector")
tab1, tab2, tab3, tab4 = st.tabs(["Check", "Fact-check", "Performance", "About"])

with tab1:
    st.write("Paste a news article and check if it looks real or fake.")
    c1, c2 = st.columns(2)
    c1.button("Try fake example", on_click=set_text, args=(FAKE_SAMPLE,))
    c2.button("Try real example", on_click=set_text, args=(REAL_SAMPLE,))
    text = st.text_area("Article", key="article", height=220)

    if st.button("Analyze", type="primary"):
        if len(text.split()) < 20:
            st.warning("Paste a longer article (at least 20 words).")
        else:
            with st.spinner("Analyzing..."):
                label, conf, words = predict(text)
            if conf < 0.75:
                st.warning(f"Not sure ({conf:.0%}). This model is trained on 2016-17 US news, "
                           "so other topics may be unreliable. Try the Fact-check tab.")
            else:
                css = "real" if label == "REAL" else "fake"
                st.markdown(f'<div class="result-card {css}">{label} ({conf:.0%} confident)</div>',
                            unsafe_allow_html=True)
            st.progress(conf)
            st.caption("Words that influenced the result: " + ", ".join(words))

with tab2:
    st.write("Search published fact-checks (by fact-checking organizations) for a claim or headline.")
    lang_name = st.selectbox("Language", ["English", "Hindi"])
    lang = "en" if lang_name == "English" else "hi"
    query = st.text_input("Claim or headline", placeholder="5G towers spread coronavirus")
    if st.button("Search fact-checks"):
        if not query.strip():
            st.warning("Enter a claim first.")
        else:
            with st.spinner("Searching..."):
                claims, err = search_factchecks(query.strip(), lang)
            if err:
                st.error(err)
            elif not claims:
                st.info("No published fact-checks found for this claim. "
                        "That does not mean it is true or false.")
            else:
                for i, c in enumerate(claims[:6]):
                    reviews = c.get("claimReview", [])
                    if not reviews:
                        continue
                    rv = reviews[0]
                    st.markdown(f"*{c.get('text', 'Claim')}*")
                    st.write(f"{rv.get('publisher', {}).get('name', 'Unknown')}: "
                             f"{rv.get('textualRating', 'n/a')}")
                    if rv.get("url"):
                        st.link_button("Read fact-check", rv["url"], key=f"fc_{i}")
                    st.divider()

with tab3:
    df = pd.DataFrame(metrics["models"]).T
    st.subheader("Model comparison")
    st.dataframe(df)
    st.bar_chart(df[["accuracy", "f1"]], stack=False)
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                              index=["Actual fake", "Actual real"],
                              columns=["Pred fake", "Pred real"]))

with tab4:
    st.write("Built with Python, scikit-learn and Streamlit. "
             "The classifier is trained on 2016-17 US news, so it learns writing style rather than "
             "factual truth and may fail on other topics or short headlines. "
             "The fact-check tab only finds claims that fact-checkers have already reviewed.")