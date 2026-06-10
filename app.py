import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ------------------------
# PAGE CONFIG
# ------------------------

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)

st.title("🛒 E-Commerce Customer Intelligence Dashboard")

# ------------------------
# LOAD DATA
# ------------------------

df = pd.read_csv(
    "processed_data/customer_intelligence_final.csv"
)

# ------------------------
# SIDEBAR
# ------------------------

st.sidebar.header("Filters")

segment_filter = st.sidebar.multiselect(
    "Select Segment",
    options=df["Segment"].unique(),
    default=df["Segment"].unique()
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["Risk_Level"].unique(),
    default=df["Risk_Level"].unique()
)

filtered_df = df[
    (df["Segment"].isin(segment_filter))
    &
    (df["Risk_Level"].isin(risk_filter))
]

# ------------------------
# KPI SECTION
# ------------------------

st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Customers",
        filtered_df["customer_unique_id"].nunique()
    )

with col2:
    st.metric(
        "Revenue",
        f"${filtered_df['Monetary'].sum():,.0f}"
    )

with col3:
    st.metric(
        "Avg Monetary",
        f"${filtered_df['Monetary'].mean():.2f}"
    )

with col4:
    st.metric(
        "Avg Frequency",
        round(filtered_df["Frequency"].mean(),2)
    )

# ------------------------
# SEGMENT DISTRIBUTION
# ------------------------

st.subheader("Customer Segments")

segment_count = (
    filtered_df["Segment"]
    .value_counts()
    .reset_index()
)

segment_count.columns = [
    "Segment",
    "Count"
]

fig = px.bar(
    segment_count,
    x="Segment",
    y="Count",
    title="Customer Segment Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------
# REVENUE BY SEGMENT
# ------------------------

segment_revenue = (
    filtered_df.groupby("Segment")
    ["Monetary"]
    .sum()
    .reset_index()
)

fig2 = px.pie(
    segment_revenue,
    names="Segment",
    values="Monetary",
    title="Revenue Contribution by Segment"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ------------------------
# CHURN ANALYSIS
# ------------------------

st.subheader("Churn Risk Analysis")

risk_count = (
    filtered_df["Risk_Level"]
    .value_counts()
    .reset_index()
)

risk_count.columns = [
    "Risk Level",
    "Count"
]

fig3 = px.bar(
    risk_count,
    x="Risk Level",
    y="Count",
    title="Risk Distribution"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ------------------------
# CUSTOMER SEARCH
# ------------------------

st.subheader("Customer Lookup")

customer_id = st.text_input(
    "Enter Customer ID"
)

if customer_id:

    result = filtered_df[
        filtered_df["customer_unique_id"]
        == customer_id
    ]

    st.dataframe(result)

# ------------------------
# TOP CHURN CUSTOMERS
# ------------------------

st.subheader("Top High Risk Customers")

top_risk = (
    filtered_df
    .sort_values(
        "churn_probability",
        ascending=False
    )
    .head(20)
)

st.dataframe(top_risk)

# ------------------------
# CHURN PREDICTION TOOL
# ------------------------

st.subheader("Predict New Customer Churn")

model = joblib.load(
    "models/churn_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

recency = st.number_input(
    "Recency",
    min_value=0
)

frequency = st.number_input(
    "Frequency",
    min_value=1
)

monetary = st.number_input(
    "Monetary",
    min_value=0.0
)

if st.button("Predict Churn"):

    sample = pd.DataFrame({
        "Recency":[recency],
        "Frequency":[frequency],
        "Monetary":[monetary]
    })

    sample_scaled = scaler.transform(
        sample
    )

    prob = (
        model.predict_proba(
            sample_scaled
        )[0][1]
    )

    st.success(
        f"Churn Probability: {prob:.2%}"
    )
