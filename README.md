# E-Commerce Customer Intelligence Project

End-to-end customer analytics project using the Olist e-commerce dataset.

## What This Project Covers

- PostgreSQL table loading
- Table audit and data quality reporting
- Master dataset creation
- EDA summaries and charts
- RFM customer feature engineering
- KMeans customer segmentation
- Churn prediction with logistic regression
- Streamlit dashboard for business exploration and prediction

## Run Order

Use the local virtual environment:

* Use Powershell
.\myvenv\Scripts\python.exe A_Data_loader.py
.\myvenv\Scripts\python.exe B_Audit.py
.\myvenv\Scripts\python.exe C_Data_quality_report.py
.\myvenv\Scripts\python.exe F_Master_dataset.py
.\myvenv\Scripts\python.exe G_eda.py
.\myvenv\Scripts\python.exe H_RFM_features.py
.\myvenv\Scripts\python.exe I_KMeans_segmentation.py
.\myvenv\Scripts\python.exe J_Churn_prediction.py


## Main Outputs

- `processed_data/master_dataset.csv`
- `processed_data/rfm_dataset.csv`
- `processed_data/customer_segments.csv`
- `processed_data/customer_intelligence_final.csv`
- `models/churn_model.pkl`
- `models/scaler.pkl`
- `models/feature_columns.pkl`
- `reports/cluster_summary.csv`
- `reports/churn_model_metrics.csv`
- `reports/churn_feature_importance.csv`
- `reports/charts/`

## Dashboard Pages

- Overview: customer, revenue, and geography KPIs
- Segments: RFM and KMeans customer segmentation
- Churn: risk distribution and high-risk customer list
- Customer Lookup: search customer intelligence records
- Prediction: predict churn probability for a new customer profile
