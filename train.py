import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Text cleaning function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['text'] = df['text'].apply(clean_text)


# -----------------------------
# 3. Train-test split
# -----------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 4. Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# 5. Train model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# -----------------------------
# 6. Evaluation (optional but good)
# -----------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")


# -----------------------------
# 7. Save model and vectorizer
# -----------------------------
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(model, open("model/model.pkl", "wb"))

print("Model and vectorizer saved successfully.")
