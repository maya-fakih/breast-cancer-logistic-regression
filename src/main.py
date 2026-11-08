from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def show_info(df):
    print("----------DataFrame Info----------\n")
    df.info()

    print("\n----------Data shape----------\n")
    print(df.shape)

    print("\n----------Data description----------\n")
    print(df.describe().to_string())


def show_correlations(df):
    corr = df.drop(columns="Label").corr()

    print("\n----------Full correlation matrix----------\n")
    print(corr.to_string())

    print("\n----------Pairs with correlation > 0.9----------\n")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            if value > 0.9:
                print(
                    f"{corr.columns[i]} — {corr.columns[j]}: "
                    f"{value:.4f}"
                )

    return corr


def plot_correlations(corr, title, output_path):
    fig, ax = plt.subplots(figsize=(15, 12))

    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"label": "Pearson correlation"},
    )

    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def drop_correlated_features(df, threshold=0.9):
    corr = df.drop(columns="Label").corr()
    kept = []
    dropped = []

    # Keep a feature if it is not highly correlated with any kept feature.
    for feature in corr.columns:
        if any(corr.loc[feature, other] > threshold for other in kept):
            dropped.append(feature)
        else:
            kept.append(feature)

    print("\nDropped features:", dropped)
    print("Remaining feature count:", len(kept))

    return df[kept + ["Label"]].copy()


def plot_feature_labels(df, output_dir):
    # Exercise 5: one joint histogram per (feature, Label).
    for feature in df.drop(columns="Label").columns:
        grid = sns.jointplot(
            data=df,
            x=feature,
            y="Label",
            kind="hist",
            height=6,
            joint_kws={"bins": (20, 2)},
        )

        grid.ax_joint.set_yticks([0, 1])
        grid.ax_joint.set_yticklabels(["Benign (0)", "Malignant (1)"])
        grid.fig.suptitle(f"{feature} against Label")
        grid.fig.subplots_adjust(top=0.92)

        grid.fig.savefig(
            output_dir / f"{feature.replace(' ', '_')}_label.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(grid.fig)

def prepare_data(df):
    X = df.drop(columns="Label")
    y = df["Label"]

    # Assignment: 500 training patients and 69 testing patients.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=69,
        random_state=42,
    )

    # Learn scaling from training data, then apply it to both sets.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()

def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "WDBC_data.xlsx"

    results_dir = root / "results"
    jointplots_dir = results_dir / "jointplots"
    jointplots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path)
    show_info(df)

    # Step 3: full matrix before removing features.
    corr_before = show_correlations(df)
    plot_correlations(
        corr_before,
        "Feature Correlations Before Removal",
        results_dir / "correlations_before.png",
    )
    corr_before.to_csv(results_dir / "correlations_before.csv")

    # Step 4: remove redundant features and repeat the matrix.
    df = drop_correlated_features(df)
    corr_after = show_correlations(df)
    plot_correlations(
        corr_after,
        "Feature Correlations After Removal",
        results_dir / "correlations_after.png",
    )
    corr_after.to_csv(results_dir / "correlations_after.csv")

    # Step 5: joint histograms of retained features against Label.
    plot_feature_labels(df, jointplots_dir)

    print(f"\nFigures and correlation tables saved in: {results_dir}")

    # Step 6: prepare data for training and testing.
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)


if __name__ == "__main__":
    main()