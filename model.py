import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:\lumi\Descriptivecleaned_capstone_data.csv")
#Dropping Unnecessary columns
df = df.drop(columns=['gender', 'education', 'employment_status','financial_struggle', 'bill_difficulty', 'income_stability',
       'financial_stress','healthcare_access', 'housing_situation',
       'conflict_impact', 'service_access','gad1', 'gad2', 'phq1', 'phq2',
       'ptsd1', 'ptsd2', 'ptsd3', 'ptsd4', 'ptsd5', 'gad_total', 'phq_total',
       'ptsd1_clean', 'ptsd1_num', 'ptsd2_clean', 'ptsd2_num', 'ptsd3_clean',
       'ptsd3_num', 'ptsd4_clean', 'ptsd4_num', 'ptsd5_clean', 'ptsd5_num','anxiety_status', 'depression_status',
       'ptsd_status', 'household_income', 'resident_status','ptsd_binary'], axis = 1)
df_cat = df.select_dtypes(include="object")
df_num = df.select_dtypes(include=["int64", "float64"])
from sklearn.impute import SimpleImputer

# Imputers
cat_imputer = SimpleImputer(strategy="most_frequent") #filling with mode value
num_imputer = SimpleImputer(strategy="median") # filling with mean value

# Fit and transform
df_cat_imputed = pd.DataFrame(
    cat_imputer.fit_transform(df_cat),
    columns=df_cat.columns,
    index=df_cat.index
)

df_num_imputed = pd.DataFrame(
    num_imputer.fit_transform(df_num),
    columns=df_num.columns,
    index=df_num.index
)

# Combine back
df = pd.concat([df_num_imputed, df_cat_imputed], axis=1)
# if ptsd_total is less than or equal to 4, then yes, otherwise no
# adding a new column called ptsd_binary and drop ptsd_total
df['ptsd_binary'] = np.where(df['ptsd_total'] <= 4, 'no', 'yes')
df = df.drop(columns=['ptsd_total'], axis=1)
df_cat = df.select_dtypes(include="object")
df_num = df.select_dtypes(include=["int64", "float64"])

# use label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df_cat.columns:
    df_cat[col] = le.fit_transform(df_cat[col])
#concat the numerical and categorical dataframes back together
df = pd.concat([df_num, df_cat], axis=1)
X = df.drop(columns = df[['anxiety_binary', 'depression_binary','ptsd_binary']])
y = df[['anxiety_binary', 'depression_binary','ptsd_binary']]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Initialize a dictionary to store models and predictions for each target
logistic_models = {}
logistic_predictions = {}

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Loop through each target variable (anxiety, depression, ptsd)
for target_col in y_train.columns:
    print(f"\n--- Training Logistic Regression for {target_col.upper()} with SMOTE ---")

    # Apply SMOTE to the training data for the current target
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train[target_col])

    # Instantiate Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear') # liblinear is good for small datasets

    # Train the model with SMOTE-resampled data
    model.fit(X_train_smote, y_train_smote)

    # Store the trained model
    logistic_models[target_col] = model

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    logistic_predictions[target_col] = y_pred

from sklearn.ensemble import RandomForestClassifier

# Initialize a dictionary to store Random Forest models and predictions for each target
rf_models = {}
rf_predictions = {}
rf_feature_importances = {}

# Initialize SMOTE (already imported in previous cells)
smote = SMOTE(random_state=42)

# Loop through each target variable (anxiety, depression, ptsd)
for target_col in y_train.columns:
    print(f"\n--- Training Random Forest Classifier for {target_col.upper()} with SMOTE ---")

    # Apply SMOTE to the training data for the current target
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train[target_col])

    # Instantiate Random Forest Classifier model
    # Using default parameters for now, can be tuned later
    model = RandomForestClassifier(random_state=42)

    # Train the model with SMOTE-resampled data
    model.fit(X_train_smote, y_train_smote)

    # Store the trained model
    rf_models[target_col] = model

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    rf_predictions[target_col] = y_pred

    # Store feature importances
    rf_feature_importances[target_col] = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)