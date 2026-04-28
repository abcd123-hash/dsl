import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------
# Load CSV Dataset
# ---------------------------
# FIX: Added encoding to prevent UnicodeDecodeError
df = pd.read_csv("spam.csv", encoding='latin-1')

# ---------------------------
# Preprocessing
# ---------------------------
X = df["v2"]
y = df["v1"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ---------------------------
# Model Training
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Prediction & Evaluation
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# Test Custom Input
# ---------------------------
test = ["Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005."]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)

# FIX: Comparing against the string 'spam' because y contains 'ham'/'spam' strings
if prediction[0] == 'spam':
    print("Prediction: spam")
else:
    print("Prediction: ham")
