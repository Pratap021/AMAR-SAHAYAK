import pandas as pd

# Correct file path with raw string to avoid \ errors
csv_path = r"C:\Users\LENOVO\Desktop\data\hospital_school_dataset.csv"

try:
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Show first 5 rows
    print("First 5 rows:")
    print(df.head())

    # Show basic info
    print(f"\nRows: {len(df)}, Columns: {len(df.columns)}")

except FileNotFoundError:
    print(f"❌ File not found at: {csv_path}")
    print("💡 Check if the file name or path is correct.")
