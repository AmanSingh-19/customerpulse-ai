import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

master = pd.read_csv("processed_data/master_dataset.csv")

master["order_purchase_timestamp"] = pd.to_datetime(
    master["order_purchase_timestamp"]
)

master["order_month"] = master["order_purchase_timestamp"].dt.to_period("M").astype(str)
master["payment_value"] = master["payment_value"].fillna(0)

REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "charts"
REPORT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


def save_bar(data, x, y, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=150)
    plt.close()


monthly_revenue = (
    master.groupby("order_month", as_index=False)
    .agg(
        orders=("order_id", "nunique"),
        customers=("customer_unique_id", "nunique"),
        revenue=("payment_value", "sum"),
    )
)

state_revenue = (
    master.groupby("customer_state", as_index=False)
    .agg(
        orders=("order_id", "nunique"),
        customers=("customer_unique_id", "nunique"),
        revenue=("payment_value", "sum"),
    )
    .sort_values("revenue", ascending=False)
)

top_cities = (
    master.groupby("customer_city", as_index=False)
    .agg(
        orders=("order_id", "nunique"),
        customers=("customer_unique_id", "nunique"),
        revenue=("payment_value", "sum"),
    )
    .sort_values("revenue", ascending=False)
    .head(20)
)

overall_summary = pd.DataFrame(
    [
        {
            "total_orders": master["order_id"].nunique(),
            "total_customers": master["customer_unique_id"].nunique(),
            "total_revenue": master["payment_value"].sum(),
            "average_order_value": master["payment_value"].mean(),
            "average_items_per_order": master["total_items"].mean(),
            "first_order_date": master["order_purchase_timestamp"].min(),
            "last_order_date": master["order_purchase_timestamp"].max(),
        }
    ]
)

monthly_revenue.to_csv(REPORT_DIR / "monthly_revenue.csv", index=False)
state_revenue.to_csv(REPORT_DIR / "state_revenue.csv", index=False)
top_cities.to_csv(REPORT_DIR / "top_cities.csv", index=False)
overall_summary.to_csv(REPORT_DIR / "overall_summary.csv", index=False)

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_revenue, x="order_month", y="revenue", marker="o")
plt.title("Monthly Revenue Trend")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "monthly_revenue_trend.png", dpi=150)
plt.close()

save_bar(
    state_revenue.head(10),
    "customer_state",
    "revenue",
    "Top States by Revenue",
    "top_states_by_revenue.png",
)

save_bar(
    top_cities.head(10),
    "customer_city",
    "revenue",
    "Top Cities by Revenue",
    "top_cities_by_revenue.png",
)

print("EDA summary files saved in reports/")
print(overall_summary.to_string(index=False))
