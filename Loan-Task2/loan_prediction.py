import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

#Load dataset
df = pd.read_csv("/loan_prediction.csv")

#Basic EDA
print("=== Basic Data Overview ===")
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nColumns:", df.columns.tolist())
print("Shape:", df.shape)

# Target detection
target = "Loan_Status"  # adjust if needed
X = df.drop(columns=[target])
y = df[target].map({'Y': 1, 'N': 0})  # encode target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Define numerical & categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

#Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Model training (compare Logistic vs RandomForest)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

for name, model in models.items():
    clf = Pipeline([('pre', preprocessor), ('model', model)])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print("ROC AUC:", roc_auc_score(y_test, probs))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Save final model (RandomForest)
final_model = Pipeline([('pre', preprocessor), 
                        ('model', RandomForestClassifier(n_estimators=200, random_state=42))])
final_model.fit(X, y)
joblib.dump(final_model, "loan_model_rf.pkl")

print("\n Model training complete. Saved as 'loan_model_rf.pkl'")