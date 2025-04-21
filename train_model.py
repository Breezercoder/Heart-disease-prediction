import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#from google.colab import files
#ploaded = files.upload()
#print(uploaded)

# Load CSV file
file_path = os.path.join(os.path.dirname(__file__), "heart_disease_dataset.csv")
df = pd.read_csv(file_path)

# ✅ Check columns and drop any unrelated ones
print("📋 Columns in dataset:", df.columns.tolist())

# ✅ Drop rows with missing values
df.dropna(inplace=True)

# ✅ Split data into features and target
if 'target' not in df.columns:
    raise ValueError("❌ 'target' column not found in dataset!")

X = df.drop("target", axis=1)
y = df["target"]

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Use GridSearchCV to tune hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}

clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
clf.fit(X_train, y_train)

# ✅ Evaluate the model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Best Parameters:", clf.best_params_)
print("🎯 Accuracy:", acc)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Save the best model
with open("model.pkl", "wb") as f:
    pickle.dump(clf.best_estimator_, f)

print("✅ Model training complete and saved as model.pkl")
