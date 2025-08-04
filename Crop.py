# Crop Recommendation System using Random Forest Classifier

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset from URL (no local file needed)
url = "https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/crop_recommendation.csv"
df = pd.read_csv(url)

print("✅ Dataset Loaded Successfully!")
print(df.head())

# 2. Visualize Correlation (Optional)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title("📊 Feature Correlation Heatmap")
plt.show()

# 3. Split into Features and Target
X = df.drop('label', axis=1)
y = df['label']

# 4. Encode Target Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# 6. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predict and Evaluate
y_pred = model.predict(X_test)
print("\n🎯 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Predict for New Input Data
# Format: [N, P, K, temperature, humidity, pH, rainfall]
sample_input = np.array([[90, 42, 43, 20.8, 82, 6.5, 200]])
prediction = model.predict(sample_input)
predicted_crop = le.inverse_transform(prediction)

print("\n🌱 Recommended Crop for given values:", predicted_crop[0])
