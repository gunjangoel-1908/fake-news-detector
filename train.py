import json, os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils import clean

fake = pd.read_csv("data/Fake.csv.zip"); fake["label"] = 0
real = pd.read_csv("data/True.csv.zip"); real["label"] = 1
df = pd.concat([fake, real], ignore_index=True)

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df.drop_duplicates(subset="content")
df = df[df["content"].str.split().str.len() > 20]
df["clean"] = df["content"].apply(clean)
print(df["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2,
    stratify=df["label"], random_state=42)

tfidf = TfidfVectorizer(stop_words="english", max_features=50000,
                        ngram_range=(1, 2), min_df=3)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
}
results = {}
for name, m in models.items():
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    print(name, results[name])
    lr = models["Logistic Regression"]
cm = confusion_matrix(y_test, lr.predict(Xte)).tolist()

os.makedirs("artifacts", exist_ok=True)
joblib.dump(lr, "artifacts/model.pkl")
joblib.dump(tfidf, "artifacts/tfidf.pkl")
with open("artifacts/metrics.json", "w") as f:
    json.dump({"models": results, "confusion_matrix": cm}, f)
print("Saved.")
