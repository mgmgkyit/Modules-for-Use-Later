import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load your intent classification data
df = pd.read_csv("it_support_questions_500.csv")

# Vectorize user inputs
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Question'])
y = df['Intent']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Save model
with open("intent_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)

print("✅ Intent classifier trained and saved.")
