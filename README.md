# 🛒 CustomerPulse AI
### E-Commerce Customer Intelligence System | RFM Segmentation · KMeans Clustering · Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Business Problem

E-commerce companies lose significant revenue because they treat all customers the same. A Champion customer who spends $500/month and a Hibernating customer who hasn't ordered in 6 months should not receive the same marketing message — yet most businesses lack the analytical infrastructure to tell them apart.

**CustomerPulse AI** solves this by building a complete customer intelligence pipeline that:
- Segments 93,000+ customers into behavioural groups using RFM analysis and KMeans clustering
- Predicts individual churn probability with 99.85% model accuracy
- Delivers an interactive executive dashboard with real-time filtering by segment and risk level
- Enables data-driven retention strategy per customer segment

---

## 📊 Dashboard Preview

> **E-Commerce Customer Intelligence Dashboard** — Live Streamlit app with Executive Overview KPIs, segment distribution charts, churn risk heatmaps, and an individual customer churn prediction interface.

![Dashboard Overview](assets/dashboard_overview.png)

**Key metrics visible on dashboard (Loyal Customers segment):**
| Metric | Value |
|---|---|
| Total Customers | 9,871 |
| Total Revenue | $2,903,557 |
| Avg Monetary Value | $294.15 |
| Avg Purchase Frequency | 1.14 |

---

## 🔍 Key Results

| Metric | Value |
|---|---|
| Total customers analysed | 93,358 |
| Dataset (transactions) | 500K+ records |
| KMeans clusters | 4 (elbow method validated) |
| Churn model accuracy | **99.85%** |
| RFM segments identified | 6 |
| Pipeline scripts | 10 (A → J) |

### Customer Segment Distribution

| Segment | Customers | Business Action |
|---|---|---|
| 🔴 Need Attention | 27,488 | Re-engagement campaigns — highest priority |
| 🟡 Potential Loyalists | 17,530 | Loyalty programme onboarding |
| 🟠 At Risk | 16,353 | Personalised win-back offers |
| ⚫ Hibernating | 14,986 | Low-cost reactivation emails |
| 🟢 Loyal Customers | 9,897 | Upsell and referral programmes |
| 🏆 Champions | 7,104 | VIP treatment, brand ambassadors |

---

## 🏗️ System Architecture

```
Raw Data (OLIST Brazil E-Commerce)
         │
         ▼
┌─────────────────────────────────────────────┐
│           10-Stage ML Pipeline              │
│                                             │
│  A_Data_loader.py    → PostgreSQL ingestion │
│  B_Audit.py          → Schema audit         │
│  C_Data_quality.py   → Quality report       │
│  D_Data_cleaning.py  → EDA + cleaning       │
│  E_Relationship.py   → Table validation     │
│  F_Master_dataset.py → Feature engineering  │
│  G_EDA.py            → Exploratory analysis │
│  H_RFM_features.py   → RFM scoring          │
│  I_KMeans_seg.py     → Clustering (k=4)     │
│  J_Churn_prediction.py → Churn model        │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│     Streamlit Dashboard (app.py)            │
│  • Executive Overview  • Segment Explorer  │
│  • Churn Risk Analysis • Customer Lookup   │
│  • Live Prediction Interface               │
└─────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Details

### RFM Scoring
Each customer is scored on three dimensions derived from transaction history:
- **Recency** — Days since last purchase (lower = better)
- **Frequency** — Number of unique orders placed
- **Monetary** — Total spend across all orders

Scores are calculated using percentile-based binning (1–5 scale) and combined into a composite RFM Total Score used for segment assignment.

### KMeans Clustering
- Optimal k=4 selected via **Elbow Method** (inertia plot) and **Silhouette Analysis**
- Features scaled using `StandardScaler` before clustering
- Cluster labels mapped to business-meaningful segment names (Champions, Loyal, At Risk, Hibernating)

### Churn Prediction Model
- **Algorithm:** Random Forest Classifier
- **Accuracy:** 99.85% on held-out test set
- **Features:** RFM scores, purchase recency, frequency deviation, monetary percentile, cluster label
- **Output:** Binary churn flag + probability score (0–1)
- **Saved model:** `models/churn_model.pkl` + `models/scaler.pkl`

---

## 🗄️ Database & SQL

Data is loaded into **PostgreSQL** across 8 relational tables from the OLIST Brazil E-Commerce dataset:

```
olist_orders  ──────┐
olist_order_items ──┤
olist_customers ────┼──► Master Dataset (F_Master_dataset.py)
olist_products ─────┤
olist_sellers ──────┤
olist_payments ─────┤
olist_reviews ──────┘
```

SQL operations include:
- Multi-table JOINs across order, customer, product, and payment tables
- Window functions for RFM recency calculation
- Aggregate queries for monetary and frequency features
- Data quality audit queries (null counts, duplicate detection, referential integrity checks)

---

## 📁 Project Structure

```
customerpulse-ai/
│
├── A_Data_loader.py          # PostgreSQL data ingestion
├── B_Audit.py                # Schema and data audit
├── C_Data_quality_report.py  # Data quality validation
├── D_Data_cleaning.ipynb     # Exploratory cleaning notebook
├── E_Relationship_validation.ipynb  # Table relationship checks
├── F_Master_dataset.py       # Feature engineering & master join
├── G_eda.py                  # Exploratory data analysis
├── H_RFM_features.py         # RFM scoring pipeline
├── I_KMeans_segmentation.py  # Customer clustering
├── J_Churn_prediction.py     # Churn prediction model
│
├── app.py                    # Streamlit dashboard
├── churn_prediction.py       # Prediction utility module
├── kmeans_segmentation.py    # Segmentation utility module
│
├── processed_data/
│   ├── rfm_dataset.csv        # RFM features (93,358 rows)
│   ├── customer_segments.csv  # Cluster assignments
│   └── master_dataset.csv     # Final feature-engineered dataset
│
├── models/
│   ├── churn_model.pkl        # Trained Random Forest model
│   └── scaler.pkl             # StandardScaler for inference
│
├── reports/
│   └── data_quality_report.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- 4GB+ RAM recommended for full dataset processing

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AmanSingh-19/customerpulse-ai.git
cd customerpulse-ai

# 2. Create virtual environment
python -m venv myvenv
source myvenv/bin/activate        # Linux/Mac
myvenv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Database Setup

```bash
# Create PostgreSQL database
createdb ecommerce_db

# Update connection string in A_Data_loader.py
# DB_CONFIG = {"host": "localhost", "database": "ecommerce_db", "user": "your_user", "password": "your_password"}
```

### Run the Pipeline

Run scripts in order (A → J):

```bash
python A_Data_loader.py          # Load raw data to PostgreSQL
python B_Audit.py                # Audit schema
python C_Data_quality_report.py  # Generate quality report
# Run D and E as Jupyter notebooks
python F_Master_dataset.py       # Build master dataset
python G_eda.py                  # EDA
python H_RFM_features.py         # Generate RFM scores
python I_KMeans_segmentation.py  # Cluster customers
python J_Churn_prediction.py     # Train churn model
```

### Launch Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL |
| Machine Learning | Scikit-learn (KMeans, Random Forest, StandardScaler) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git |

---

## 💡 Business Recommendations by Segment

Based on the model outputs, the following retention strategies are recommended:

**🔴 Need Attention (27,488 customers — largest segment)**
> These customers were once active but engagement is declining. Trigger: personalised "We miss you" campaigns with a time-limited discount (10–15%). Expected uplift: 8–12% reactivation rate.

**🟠 At Risk (16,353 customers)**
> High historical value but showing churn signals. Trigger: loyalty points bonus offer or free shipping threshold reduction. Prioritise top 20% by Monetary score.

**🏆 Champions (7,104 customers)**
> Highest RFM scores. Enrol in VIP programme, offer early access to new products, and use as referral seed audience. Cost to retain: low. Value at stake: high.

---

## 📈 Dataset

**Source:** [OLIST Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle

- 8 relational tables
- 100,000+ orders (2016–2018)
- 93,358 unique customers after cleaning
- Features: order timestamps, product categories, payment values, review scores, geolocation

---

## 🔮 Future Enhancements

- [ ] **GenAI Layer** — Integrate Gemini API to auto-generate personalised retention email copy per segment
- [ ] **FastAPI Deployment** — Expose churn prediction as REST endpoint (`POST /predict`)
- [ ] **Docker containerisation** — Package full app for cloud deployment
- [ ] **MLflow tracking** — Log model experiments and version the churn model
- [ ] **Real-time scoring** — Stream new transactions via Kafka for live RFM updates
- [ ] **A/B testing module** — Measure campaign effectiveness per segment

---

## 👨‍💻 Author

**Aman Singh**
B.Tech Data Science | Amity University, Greater Noida
📧 [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/AmanSingh-19)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built as part of a professional data science portfolio targeting analytics and ML engineering roles in 2026.*
