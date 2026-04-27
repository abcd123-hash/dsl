# ---------------------------
# Import Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score, mean_squared_error,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("data.csv")

# ---------------------------
# 1. EDA
# ---------------------------
print(df.head())
print(df.describe())

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# Histogram
df.hist(figsize=(8,5))
plt.suptitle("Histogram of Features")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df)
plt.title("Boxplot for Outlier Detection")
plt.show()

# Covariance
cov_matrix = df.cov()
print("Covariance Matrix:\n", cov_matrix)

# ---------------------------
# 2. Data Preprocessing
# ---------------------------
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Study Hours', 'Marks']] = imputer.fit_transform(
    df[['Age', 'Study Hours', 'Marks']]
)

# Features and Target
X = df[['Age', 'Study Hours']]
y = df['Marks']

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# 3. Visualization
# ---------------------------
plt.scatter(df['Study Hours'], df['Marks'])
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Study Hours vs Marks')
plt.show()

# ---------------------------
# 4. Classification (Pass/Fail)
# ---------------------------
y_class = (df['Marks'] >= 50).astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# Scaling for classification
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# Model
clf = LogisticRegression()
clf.fit(X_train_c, y_train_c)

# Prediction
y_pred = clf.predict(X_test_c)

# Metrics
print("----- Classification Metrics -----")
print('Accuracy :', accuracy_score(y_test_c, y_pred))
print('Precision:', precision_score(y_test_c, y_pred))
print('Recall   :', recall_score(y_test_c, y_pred))
print('F1 Score :', f1_score(y_test_c, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ---------------------------
# 5. Regression (Predict Marks)
# ---------------------------
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

print("\n----- Regression -----")
print('MSE:', mean_squared_error(y_test, y_pred_reg))

# ---------------------------
# 6. Clustering
# ---------------------------
# Scale before clustering (IMPORTANT)
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.scatter(df['Study Hours'], df['Marks'], c=clusters)
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Clustering Students')
plt.show()