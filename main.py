import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

file_path = "cumulative.csv"
data = pd.read_csv(file_path)

print(data.info())
print(data.head())

missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

data_cleaned = data.dropna(axis=1, thresh=int(0.5 * len(data)))
data_cleaned = data_cleaned.dropna()

columns_to_drop = ['rowid', 'kepoi_name', 'kepler_name', 'koi_pdisposition']
data_cleaned = data_cleaned.drop([col for col in columns_to_drop if col in data_cleaned.columns], axis=1)

data_cleaned['koi_disposition'] = data_cleaned['koi_disposition'].map({
    'CONFIRMED': 1,
    'FALSE POSITIVE': 0,
    'CANDIDATE': 2
})

print("Cleaned Dataset Shape:", data_cleaned.shape)

X = data_cleaned.drop('koi_disposition', axis=1)
y = data_cleaned['koi_disposition']

non_numeric_columns = X.select_dtypes(include=['object']).columns
print("Non-Numeric Columns:\n", non_numeric_columns)
if 'koi_tce_delivname' in X.columns:
    X = X.drop(columns=['koi_tce_delivname'])
print("Remaining Non-Numeric Columns:", X.select_dtypes(include=['object']).columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_importance = model.feature_importances_
important_features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
print("Feature Importance:\n", important_features)

y_pred = model.predict(X_test)
print("Initial Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['False Positive', 'Confirmed', 'Candidate'],
            yticklabels=['False Positive', 'Confirmed', 'Candidate'])
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Initial Accuracy Score:", accuracy)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, verbose=2, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print("Best Parameters for Random Forest:", grid_search_rf.best_params_)

best_rf_model = grid_search_rf.best_estimator_
best_rf_model.fit(X_train, y_train)
y_pred_rf_optimized = best_rf_model.predict(X_test)

print("Optimized Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf_optimized))
optimized_rf_accuracy = accuracy_score(y_test, y_pred_rf_optimized)
print("Optimized Random Forest Accuracy Score:", optimized_rf_accuracy)

xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',
            xticklabels=['False Positive', 'Confirmed', 'Candidate'],
            yticklabels=['False Positive', 'Confirmed', 'Candidate'])
plt.title("XGBoost Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy Score:", xgb_accuracy)

print(f"Initial Random Forest Accuracy: {accuracy}")
print(f"Optimized Random Forest Accuracy: {optimized_rf_accuracy}")
print(f"XGBoost Accuracy: {xgb_accuracy}")
