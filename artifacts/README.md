# Fake News Detector

NLP classifier that labels news articles as real or fake, with a Streamlit app and a Google Fact Check API lookup.

*Live demo:* (add after deploy)

## Results
Dataset: ISOT Fake and Real News (about 38.5K articles after removing duplicates).

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 98.17% | 98.35% |
| Naive Bayes | 94.48% | 94.95% |
| Linear SVM | 98.79% | 98.91% |

## Approach
- Text cleaning, including removal of source-name leakage ("Reuters")
- TF-IDF features (unigrams and bigrams, 50K vocabulary)
- 3 models compared on a stratified 80/20 split; the app uses Logistic Regression for confidence scores
- "Not sure" message below 75% confidence
- Fact-check tab searches published fact-checks via Google Fact Check Tools API

## Limitations
The dataset is 2016-17 US news, so the model learns writing style and topic rather than factual truth. It can fail on other topics and short headlines. The fact-check tab only finds claims that fact-checkers have already reviewed.

## Run locally
    pip install -r requirements.txt
    python train.py
    streamlit run app.py

Add your key in .streamlit/secrets.toml:

    FACTCHECK_KEY = "AIzaSyDOSvEP2v7BX8CoFiXbm7YKJh2wLZFFles"