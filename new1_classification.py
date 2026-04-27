# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("data1.csv")

print(df.head())

# ---------------------------
# 2. Preprocessing
# ---------------------------

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['age','studyhours','marks']] = imputer.fit_transform(df[['age','studyhours','marks']])

# Encode target
le = LabelEncoder()
df['result'] = le.fit_transform(df['result'])

# Features & Target
X = df[['age','studyhours','marks']]
y = df['result']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# 3. Visualization
# ---------------------------

# Histogram
df.hist(figsize=(8,6))
plt.suptitle("Histogram of Features")
plt.show()

# Boxplot
sns.boxplot(data=df[['age','studyhours','marks']])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Heatmap (Correlation)
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# Covariance Matrix
cov_matrix = df[['age','studyhours','marks']].cov()
print("\nCovariance Matrix:\n", cov_matrix)

# ---------------------------
# 4. Classification
# ---------------------------
clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Metrics
print("\n--- Classification Metrics ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# ---------------------------
# 5. Regression
# ---------------------------
X_reg = df[['age','studyhours']]
y_reg = df['marks']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(Xr_train, yr_train)

y_pred_reg = reg.predict(Xr_test)

print("\nRegression MSE:", mean_squared_error(yr_test, y_pred_reg))

# ---------------------------
# 6. Clustering
# ---------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df[['age','studyhours','marks']])

# Plot clusters
plt.scatter(df['studyhours'], df['marks'], c=clusters)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Clusters")
plt.show()