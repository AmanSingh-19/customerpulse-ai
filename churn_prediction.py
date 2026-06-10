import pandas as pd

rfm = pd.read_csv(
    "processed_data/customer_segments.csv"
)

print(rfm.head())

rfm["Churn"] = (
    rfm["Recency"] > 90
).astype(int)

print(
    rfm["Churn"]
    .value_counts()
)

X = rfm[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

y = rfm["Churn"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)

y_pred = model.predict(
    X_test_scaled
)

y_prob = model.predict_proba(
    X_test_scaled
)[:,1]

from sklearn.metrics import accuracy_score

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

X_scaled = scaler.transform(X)

rfm["churn_probability"] = (
    model.predict_proba(
        X_scaled
    )[:,1]
)

def risk_level(prob):

    if prob >= 0.80:
        return "High Risk"

    elif prob >= 0.50:
        return "Medium Risk"

    else:
        return "Low Risk"

rfm["Risk_Level"] = (
    rfm["churn_probability"]
    .apply(risk_level)
)

print(
    rfm["Risk_Level"]
    .value_counts()
)

high_risk_customers = (
    rfm.sort_values(
        "churn_probability",
        ascending=False
    )
)

print(
    high_risk_customers.head(20)
)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient":
        model.coef_[0]
})

print(importance)

rfm.to_csv(
    "processed_data/customer_intelligence_final.csv",
    index=False
)

print(
    "Final Customer Intelligence Dataset Saved"
)

import joblib

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
