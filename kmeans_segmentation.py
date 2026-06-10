import pandas as pd

rfm = pd.read_csv(
    "processed_data/rfm_dataset.csv"
)

print(rfm.head())

features = rfm[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    features
)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for k in range(1,11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_features)

    wcss.append(
        kmeans.inertia_
    )

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()

K = 4

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = (
    kmeans.fit_predict(
        scaled_features
    )
)

print(
    rfm["Cluster"]
    .value_counts()
)

cluster_summary = (
    rfm.groupby("Cluster")
    [[
        "Recency",
        "Frequency",
        "Monetary"
    ]]
    .mean()
)

print(cluster_summary)

cluster_names = {
    2:"Champions",
    0:"Loyal Customers",
    3:"Potential Customers",
    1:"At Risk"
}

rfm["Cluster_Name"] = (
    rfm["Cluster"]
    .map(cluster_names)
)

print(
    rfm["Cluster_Name"]
    .value_counts()
)

cluster_revenue = (
    rfm.groupby(
        "Cluster_Name"
    )
    ["Monetary"]
    .sum()
)

print(cluster_revenue)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster_Name"
)

plt.title(
    "Customer Segments"
)

plt.show()

pd.crosstab(
    rfm["Segment"],
    rfm["Cluster_Name"]
)

rfm.to_csv(
    "processed_data/customer_segments.csv",
    index=False
)

print(
    "Customer Segmentation Saved"
)

print(cluster_summary)

print(
    rfm["Cluster"]
    .value_counts()
)
