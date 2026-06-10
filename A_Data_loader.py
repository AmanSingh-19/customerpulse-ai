import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://" \
    "postgres:AMANsingh%4012@localhost:5432/Olist_db"
)

tables = [
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_orders_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
    "product_category_name_translation"
]

dataframes = {}

for table in tables:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    dataframes[table] = df

    print(f"\n{'='*50}")
    print(table)
    print(df.head())
    print(df.shape)