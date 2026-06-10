from A_Data_loader import dataframes
import pandas as pd

# Extracting table

customers = dataframes["olist_customers_dataset"]

orders = dataframes["olist_orders_dataset"]

payments = dataframes["olist_order_payments_dataset"]

items = dataframes["olist_order_items_dataset"]

products = dataframes["olist_products_dataset"]

orders = orders[
    orders["order_status"] == "delivered"
].copy()

print("Delivered Orders:", orders.shape)

# Aggregate Payments

payment_summary = (
    payments
    .groupby("order_id", as_index=False)
    .agg({
        "payment_value":"sum"
    })
)

print(payment_summary.head())

# Aggregate order items

item_summary = (
    items
    .groupby("order_id", as_index=False)
    .agg(
        total_items=("order_item_id","count"),
        total_freight=("freight_value","sum")
    )
)

print(item_summary.head())

# Customer + Orders

master = pd.merge(
    customers,
    orders,
    on="customer_id",
    how="inner"
)

print(master.shape)

# Add Payments

master = pd.merge(
    master,
    payment_summary,
    on="order_id",
    how="left"
)

print(master.shape)

# Add Item Summary

master = pd.merge(
    master,
    item_summary,
    on="order_id",
    how="left"
)

print(master.shape)

# Add Product Information

product_summary = (
    items
    .groupby("order_id")
    ["product_id"]
    .count()
    .reset_index()
)

product_summary.rename(
    columns={
        "product_id":"product_count"
    },
    inplace=True
)

print(product_summary.head())

master = pd.merge(
    master,
    product_summary,
    on="order_id",
    how="left"
)

print(master.shape)

print(master.head())

print(master.columns)

print(
    "Original Revenue:",
    payments["payment_value"].sum()
)

print(
    "Master Revenue:",
    master["payment_value"].sum()
)

master.to_csv(
    "processed_data/master_dataset.csv",
    index=False
)

print("Master Dataset Saved Successfully")

print("Rows:", master.shape[0])

print(
    "Customers:",
    master["customer_unique_id"].nunique()
)

print(
    "Orders:",
    master["order_id"].nunique()
)

print(
    "Revenue:",
    master["payment_value"].sum()
)