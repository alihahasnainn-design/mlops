import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🚀 SCRIPT STARTED")

# ====================== CONFIG ======================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "mlopzz",
    "port": 3306
}

TABLE_NAME = "german_credit"
# ===================================================

print("🔌 Connecting to XAMPP MySQL...")

try:
    import pymysql
    conn = pymysql.connect(**DB_CONFIG)
    print("✅ Connected Successfully!")

    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    
    print(f"✅ Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())

except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ====================== ML PIPELINE ======================

target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode target if categorical
if y.dtype == "object":
    from sklearn.preprocessing import LabelEncoder
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)
    print(f"✅ Target encoded. Classes: {list(y_encoder.classes_)}")

# Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"Features: {len(num_cols)} Numerical + {len(cat_cols)} Categorical")

# Preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
print("\n🔄 Training models...")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

best_model = None
best_score = -1
best_name = ""

for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    cv_mean = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    print(f"{name:20} → Test: {acc:.4f} | CV: {cv_mean:.4f}")
    
    if cv_mean > best_score:
        best_score = cv_mean
        best_model = pipeline
        best_name = name

# Final Results
print("\n" + "="*65)
print(f"🏆 BEST MODEL: {best_name}")
print(f"BEST CV ACCURACY: {best_score:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

# Save Model
model_info = {
    "model": best_model,
    "model_name": best_name,
    "accuracy": best_score,
    "feature_names": X.columns.tolist(),
    "target_column": target_column
}

with open("best_model.pkl", "wb") as f:
    pickle.dump(model_info, f)

print(f"\n✅ Model saved successfully as 'best_model.pkl'")