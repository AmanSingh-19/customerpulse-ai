import pandas as pd
from A_Data_loader import dataframes

report = []

for table_name, df in dataframes.items():

    report.append({
        "Table Name": table_name,
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum()
    })

quality_report = pd.DataFrame(report)

print("\nDATA QUALITY REPORT")
print(quality_report)

quality_report.to_csv(
    "data_quality_report.csv",
    index=False
)

print("\nReport saved successfully!")