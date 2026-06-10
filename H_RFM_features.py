from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("processed_data")
MASTER_PATH = PROCESSED_DIR / "master_dataset.csv"
RFM_PATH = PROCESSED_DIR / "rfm_dataset.csv"


def score_series(series, high_is_good=True, bins=5):
    """Create stable quantile scores even when many customers share values."""
    labels = list(range(1, bins + 1))
    if not high_is_good:
        labels = labels[::-1]

    ranked = series.rank(method="first")
    unique_values = ranked.nunique()
    bin_count = min(bins, unique_values)

    if bin_count < 2:
        return pd.Series(np.full(len(series), bins), index=series.index)

    active_labels = labels[-bin_count:] if high_is_good else labels[:bin_count]
    return pd.qcut(ranked, q=bin_count, labels=active_labels).astype(int)


def assign_segment(row):
    if row["RFM_Total"] >= 13 and row["R_Score"] >= 4:
        return "Champions"
    if row["F_Score"] >= 4 and row["M_Score"] >= 4:
        return "Loyal Customers"
    if row["R_Score"] >= 4 and row["RFM_Total"] >= 10:
        return "Potential Loyalists"
    if row["R_Score"] <= 2 and row["F_Score"] >= 3:
        return "At Risk"
    if row["R_Score"] <= 2:
        return "Hibernating"
    return "Need Attention"


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing {MASTER_PATH}. Run F_Master_dataset.py first.")

    master = pd.read_csv(MASTER_PATH)
    master["order_purchase_timestamp"] = pd.to_datetime(
        master["order_purchase_timestamp"],
        errors="coerce",
    )
    master["payment_value"] = master["payment_value"].fillna(0)

    snapshot_date = master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        master.groupby("customer_unique_id")
        .agg(
            Last_Purchase_Date=("order_purchase_timestamp", "max"),
            First_Purchase_Date=("order_purchase_timestamp", "min"),
            Frequency=("order_id", "nunique"),
            Monetary=("payment_value", "sum"),
            Total_Items=("total_items", "sum"),
            Total_Freight=("total_freight", "sum"),
            Customer_State=("customer_state", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            Customer_City=("customer_city", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        )
        .reset_index()
    )

    rfm["Recency"] = (snapshot_date - rfm["Last_Purchase_Date"]).dt.days
    rfm["Average_Order_Value"] = rfm["Monetary"] / rfm["Frequency"].replace(0, np.nan)
    rfm["Average_Order_Value"] = rfm["Average_Order_Value"].fillna(0)

    rfm["R_Score"] = score_series(rfm["Recency"], high_is_good=False)
    rfm["F_Score"] = score_series(rfm["Frequency"], high_is_good=True)
    rfm["M_Score"] = score_series(rfm["Monetary"], high_is_good=True)
    rfm["RFM_Total"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    output_columns = [
        "customer_unique_id",
        "Customer_State",
        "Customer_City",
        "First_Purchase_Date",
        "Last_Purchase_Date",
        "Recency",
        "Frequency",
        "Monetary",
        "Average_Order_Value",
        "Total_Items",
        "Total_Freight",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Total",
        "RFM_Score",
        "Segment",
    ]

    PROCESSED_DIR.mkdir(exist_ok=True)
    rfm[output_columns].to_csv(RFM_PATH, index=False)

    print("RFM dataset saved:", RFM_PATH)
    print("Rows:", rfm.shape[0])
    print(rfm["Segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
