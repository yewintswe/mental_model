import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("Descriptivecleaned_capstone_data.csv")

# ── 2. Drop unnecessary columns (from model.py) ────────────────────────────────
df = df.drop(columns=[
    'gender', 'education', 'employment_status', 'financial_struggle',
    'bill_difficulty', 'income_stability', 'financial_stress',
    'healthcare_access', 'housing_situation', 'conflict_impact',
    'service_access', 'gad1', 'gad2', 'phq1', 'phq2',
    'ptsd1', 'ptsd2', 'ptsd3', 'ptsd4', 'ptsd5',
    'gad_total', 'phq_total',
    'ptsd1_clean', 'ptsd1_num', 'ptsd2_clean', 'ptsd2_num',
    'ptsd3_clean', 'ptsd3_num', 'ptsd4_clean', 'ptsd4_num',
    'ptsd5_clean', 'ptsd5_num',
    'anxiety_status', 'depression_status', 'ptsd_status',
    'household_income', 'resident_status', 'ptsd_binary'
])

# ── 3. Separate cat / num ──────────────────────────────────────────────────────
df_cat = df.select_dtypes(include="object")
df_num = df.select_dtypes(include=["int64", "float64"])

# ── 4. Impute ──────────────────────────────────────────────────────────────────
cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer  = SimpleImputer(strategy="median")

df_cat_imputed = pd.DataFrame(
    cat_imputer.fit_transform(df_cat),
    columns=df_cat.columns, index=df_cat.index
)
df_num_imputed = pd.DataFrame(
    num_imputer.fit_transform(df_num),
    columns=df_num.columns, index=df_num.index
)
df = pd.concat([df_num_imputed, df_cat_imputed], axis=1)

# ── 5. Create ptsd_binary, drop ptsd_total ─────────────────────────────────────
df["ptsd_binary"] = np.where(df["ptsd_total"] <= 4, "no", "yes")
df = df.drop(columns=["ptsd_total"])

# ── 6. Re-split after new target column ───────────────────────────────────────
df_cat = df.select_dtypes(include="object")
df_num = df.select_dtypes(include=["int64", "float64"])

# ── 7. Label-encode categoricals; save per-column encoders ────────────────────
label_encoders = {}
df_cat = df_cat.copy()
for col in df_cat.columns:
    le = LabelEncoder()
    df_cat[col] = le.fit_transform(df_cat[col])
    label_encoders[col] = le

df = pd.concat([df_num, df_cat], axis=1)

# ── 8. X / y split ────────────────────────────────────────────────────────────
TARGET_COLS = ["anxiety_binary", "depression_binary", "ptsd_binary"]
X = df.drop(columns=TARGET_COLS)
y = df[TARGET_COLS]

feature_names = X.columns.tolist()
print(f"\nFeatures ({len(feature_names)}): {feature_names}")

# ── 9. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)

# ── 10a. ANXIETY → Logistic Regression + SMOTE ───────────────────────────────
print("\n--- Training Logistic Regression for ANXIETY with SMOTE ---")
X_tr_s, y_tr_s = smote.fit_resample(X_train, y_train["anxiety_binary"])
anxiety_model = LogisticRegression(random_state=42, solver="liblinear")
anxiety_model.fit(X_tr_s, y_tr_s)
y_pred_anx = anxiety_model.predict(X_test)
print(classification_report(y_test["anxiety_binary"], y_pred_anx))

# ── 10b. DEPRESSION → Random Forest + SMOTE ──────────────────────────────────
print("\n--- Training Random Forest for DEPRESSION with SMOTE ---")
X_tr_s, y_tr_s = smote.fit_resample(X_train, y_train["depression_binary"])
depression_model = RandomForestClassifier(random_state=42)
depression_model.fit(X_tr_s, y_tr_s)
y_pred_dep = depression_model.predict(X_test)
print(classification_report(y_test["depression_binary"], y_pred_dep))

# ── 10c. PTSD → Logistic Regression + SMOTE ──────────────────────────────────
print("\n--- Training Logistic Regression for PTSD with SMOTE ---")
X_tr_s, y_tr_s = smote.fit_resample(X_train, y_train["ptsd_binary"])
ptsd_model = LogisticRegression(random_state=42, solver="liblinear")
ptsd_model.fit(X_tr_s, y_tr_s)
y_pred_ptsd = ptsd_model.predict(X_test)
print(classification_report(y_test["ptsd_binary"], y_pred_ptsd))

# ── 11. Save pickle ───────────────────────────────────────────────────────────
bundle = {
    "anxiety_model":    anxiety_model,     # LogisticRegression
    "depression_model": depression_model,  # RandomForestClassifier
    "ptsd_model":       ptsd_model,        # LogisticRegression
    "feature_names":    feature_names,
    "label_encoders":   label_encoders,
}

with open("models.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n✅  models.pkl saved successfully.")
print(f"    Anxiety     accuracy : {accuracy_score(y_test['anxiety_binary'],    y_pred_anx):.3f}")
print(f"    Depression  accuracy : {accuracy_score(y_test['depression_binary'], y_pred_dep):.3f}")
print(f"    PTSD        accuracy : {accuracy_score(y_test['ptsd_binary'],       y_pred_ptsd):.3f}")
