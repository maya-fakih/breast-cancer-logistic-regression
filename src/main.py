from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_info(df):
    print("----------DataFrame Info-------\n")
    print(df.info())
    print("\n---------Data shape----------\n")
    print(df.shape)
    print("\n------Data describtion-------\n")
    print(df.describe().to_string())
    print("\n-------Show Corelations------\n")
    print(show_correlations(df))

def show_correlations(df):
    # Exclude the target: compare input features only.
    corr = df.drop(columns="Label").corr()

    # Check each pair once, excluding self-correlations.
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]

            if value > 0.9:
                print(
                    f"{corr.columns[i]} — {corr.columns[j]}: "
                    f"{value:.4f}"
                )

def main():
    data_path = Path(__file__).parent.resolve().parent /"data"/"WDBC_data.xlsx"

    df = pd.read_excel(data_path)
    show_info(df)


if __name__=="__main__":
    main()
    