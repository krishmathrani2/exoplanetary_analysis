import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

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

print("Cleaned Dataset Shape:", data_cleaned.shape)

data_cleaned['koi_disposition'] = data_cleaned['koi_disposition'].map({
    'CONFIRMED': 1,
    'FALSE POSITIVE': 0,
    'CANDIDATE': 2
})

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
print(important_features)


y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False Positive', 'Confirmed', 'Candidate'],
            yticklabels=['False Positive', 'Confirmed', 'Candidate'])
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

y_pred_optimized = best_model.predict(X_test)

print("Optimized Classification Report:\n", classification_report(y_test, y_pred_optimized))

optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
print("Optimized Accuracy Score:", optimized_accuracy)



