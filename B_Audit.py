from A_Data_loader import dataframes

def audit_table(df, table_name):

    print("\n" + "="*60)
    print(f"TABLE : {table_name}")

    print("\nRows, Columns")
    print(df.shape)

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())


for table_name, df in dataframes.items():

    audit_table(df, table_name)