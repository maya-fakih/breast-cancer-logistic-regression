from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without opening windows.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 42


def drop_correlated_features(df, threshold=0.9):
    """Keep a feature only if its correlation with kept features is <= 0.9."""
    corr = df.drop(columns="Label").corr()
    kept, dropped = [], []

    for feature in corr.columns:
        if any(corr.loc[feature, other] > threshold for other in kept):
            dropped.append(feature)
        else:
            kept.append(feature)

    print("Dropped features:", dropped)
    print("Remaining feature count:", len(kept))

    pd.Series(dropped, name="dropped_feature").to_csv(
        RESULTS / "dropped_features.csv", index=False
    )

    return df[kept + ["Label"]].copy()


def save_exploratory_results(df, reduced_df):
    """Questions 2–5: statistics, correlations, and feature/label plots."""
    df.describe().to_csv(RESULTS / "describe.csv")

    for name, frame in [("before", df), ("after", reduced_df)]:
        corr = frame.drop(columns="Label").corr()
        corr.to_csv(RESULTS / f"correlations_{name}.csv")

        fig, ax = plt.subplots(figsize=(15, 12))
        sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Feature correlations {name} removal")
        fig.tight_layout()
        fig.savefig(RESULTS / f"correlations_{name}.png", dpi=160)
        plt.close(fig)

    corr = df.drop(columns="Label").corr()
    pairs = []

    for i, first in enumerate(corr.columns):
        for second in corr.columns[i + 1:]:
            if corr.loc[first, second] > 0.9:
                pairs.append({
                    "feature_1": first,
                    "feature_2": second,
                    "correlation": corr.loc[first, second],
                })

    pd.DataFrame(
        pairs, columns=["feature_1", "feature_2", "correlation"]
    ).to_csv(RESULTS / "high_correlations.csv", index=False)

    plot_dir = RESULTS / "jointplots"
    plot_dir.mkdir(exist_ok=True)

    for feature in reduced_df.drop(columns="Label").columns:
        grid = sns.jointplot(
            data=reduced_df,
            x=feature,
            y="Label",
            kind="hist",
            height=5,
        )
        grid.fig.savefig(
            plot_dir / f"{feature}_label.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(grid.fig)

    # Class-specific descriptive statistics help interpret the histograms.
    reduced_df.groupby("Label").mean().T.to_csv(
        RESULTS / "feature_means_by_label.csv"
    )


def prepare_data(df):
    X = df.drop(columns="Label")
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=69,
        random_state=SEED,
    )

    # Learn scaling from the training set only.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def loss(X, y, w, b, lam):
    """Assignment objective: summed cross-entropy + L2 penalty."""
    z = X @ w + b
    return (
        np.sum(np.logaddexp(0, z) - y * z)
        + (lam / 2) * np.dot(w, w)
    )


def cross_entropy(X, y, w, b):
    """Mean cross-entropy for comparing training and testing sets."""
    z = X @ w + b
    return np.mean(np.logaddexp(0, z) - y * z)


def accuracy(X, y, w, b):
    predictions = (expit(X @ w + b) > 0.5).astype(int)
    return np.mean(predictions == y)


def gradient_norm(X, y, w, b, lam):
    error = expit(X @ w + b) - y
    grad_w = X.T @ error + lam * w
    grad_b = np.sum(error)
    return np.sqrt(np.dot(grad_w, grad_w) + grad_b**2)


def record_metrics(history, iteration, X_train, y_train,
                   X_test, y_test, w, b, lam):
    history.append({
        "iteration": iteration,
        "train_objective": loss(X_train, y_train, w, b, lam),
        "train_loss": cross_entropy(X_train, y_train, w, b),
        "test_loss": cross_entropy(X_test, y_test, w, b),
        "train_accuracy": accuracy(X_train, y_train, w, b),
        "test_accuracy": accuracy(X_test, y_test, w, b),
    })


def gradient_descent(X_train, y_train, X_test, y_test,
                     step_size=0.001, lam=0.1,
                     max_iter=20000, tolerance=1e-5):
    w = np.zeros(X_train.shape[1])
    b = 0.0
    history = []
    updates = 0

    record_metrics(
        history, 0, X_train, y_train, X_test, y_test, w, b, lam
    )

    for _ in range(max_iter):
        error = expit(X_train @ w + b) - y_train

        grad_w = X_train.T @ error + lam * w
        grad_b = np.sum(error)

        norm = np.sqrt(np.dot(grad_w, grad_w) + grad_b**2)

        if norm < tolerance:
            break

        w -= step_size * grad_w
        b -= step_size * grad_b
        updates += 1

        if updates % 100 == 0:
            record_metrics(
                history, updates, X_train, y_train,
                X_test, y_test, w, b, lam
            )

    if history[-1]["iteration"] != updates:
        record_metrics(
            history, updates, X_train, y_train,
            X_test, y_test, w, b, lam
        )

    return w, b, pd.DataFrame(history), updates


def stochastic_gradient_descent(
    X_train, y_train, X_test, y_test,
    step_size=0.0001, lam=0.1,
    max_updates=200000, diminishing=False,
):
    rng = np.random.default_rng(SEED)
    n, dimension = X_train.shape
    w = np.zeros(dimension)
    b = 0.0
    history = []

    record_metrics(
        history, 0, X_train, y_train, X_test, y_test, w, b, lam
    )

    for update in range(1, max_updates + 1):
        # Sample one patient uniformly at random.
        i = rng.integers(n)
        error = expit(X_train[i] @ w + b) - y_train[i]

        # Multiplication by n gives an unbiased gradient estimate
        # of the assignment's SUMMED objective.
        grad_w = n * error * X_train[i] + lam * w
        grad_b = n * error

        # Question 9: i means the global update count, starting at 1.
        eta = 0.1 / update if diminishing else step_size

        w -= eta * grad_w
        b -= eta * grad_b

        # Record every 500 updates: approximately one data pass.
        if update % n == 0:
            record_metrics(
                history, update, X_train, y_train,
                X_test, y_test, w, b, lam
            )

    if history[-1]["iteration"] != max_updates:
        record_metrics(
            history, max_updates, X_train, y_train,
            X_test, y_test, w, b, lam
        )

    return w, b, pd.DataFrame(history), max_updates


def result_row(method, X_train, y_train, X_test, y_test,
               w, b, lam, step_size, updates):
    norm = gradient_norm(X_train, y_train, w, b, lam)

    return {
        "method": method,
        "lambda": lam,
        "step_size": step_size,
        "updates": updates,
        "train_objective": loss(X_train, y_train, w, b, lam),
        "train_loss": cross_entropy(X_train, y_train, w, b),
        "test_loss": cross_entropy(X_test, y_test, w, b),
        "train_accuracy": accuracy(X_train, y_train, w, b),
        "test_accuracy": accuracy(X_test, y_test, w, b),
        "gradient_norm": norm,
        "gradient_tolerance_met": norm < 1e-5,
    }


def plot_histories(histories, training_size):
    """Question 10: training/testing loss and accuracy for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panels = [
        ("train_loss", "Training cross-entropy"),
        ("test_loss", "Testing cross-entropy"),
        ("train_accuracy", "Training accuracy"),
        ("test_accuracy", "Testing accuracy"),
    ]

    for ax, (column, title) in zip(axes.flat, panels):
        for name, history in histories.items():
            # One GD update uses n patients; one SGD update uses one.
            passes = (
                history["iteration"]
                if name == "GD"
                else history["iteration"] / training_size
            )
            ax.plot(passes, history[column], label=name)

        ax.set_title(title)
        ax.set_xlabel("Equivalent passes through training data")
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTS / "training_comparison.png", dpi=180)
    plt.close(fig)

    # Separate plots show fluctuations more clearly.
    for name, history in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        for split in ["train", "test"]:
            axes[0].plot(
                history["iteration"],
                history[f"{split}_loss"],
                label=split,
            )
            axes[1].plot(
                history["iteration"],
                history[f"{split}_accuracy"],
                label=split,
            )

        axes[0].set_title(f"{name}: cross-entropy")
        axes[1].set_title(f"{name}: accuracy")

        for ax in axes:
            ax.set_xlabel("Updates")
            ax.legend()
            ax.grid(alpha=0.25)

        fig.tight_layout()
        filename = name.lower().replace(" ", "_")
        fig.savefig(RESULTS / f"{filename}_curves.png", dpi=160)
        plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_excel(ROOT / "data" / "WDBC_data.xlsx")

    # Ensure malignant = 1 and benign = 0.
    if not pd.api.types.is_numeric_dtype(df["Label"]):
        df["Label"] = (
            df["Label"].astype(str).str.strip().str.upper()
            .map({"M": 1, "B": 0, "1": 1, "0": 0})
        )

    if df["Label"].isna().any() or not df["Label"].isin([0, 1]).all():
        raise ValueError("Label must contain 0/1 or B/M.")

    df["Label"] = df["Label"].astype(int)

    print(df.info())
    print(df.describe())

    reduced_df = drop_correlated_features(df)
    save_exploratory_results(df, reduced_df)

    X_train, X_test, y_train, y_test = prepare_data(reduced_df)
    data = (X_train, y_train, X_test, y_test)

    print("\nTraining shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # Parameter trials are reported individually.
    # We do not choose a winner using the test set.
    trials = []

    print("\nRunning GD parameter trials...")
    for lam in [0.01, 0.1, 1.0]:
        for step in [0.0005, 0.001, 0.002]:
            w, b, _, updates = gradient_descent(
                *data, step_size=step, lam=lam
            )
            trials.append(result_row(
                "GD", *data, w, b, lam, step, updates
            ))

    print("Running diminishing-step SGD lambda trials...")
    for lam in [0.01, 0.1, 1.0]:
        w, b, _, updates = stochastic_gradient_descent(
            *data, lam=lam, diminishing=True
        )
        trials.append(result_row(
            "SGD diminishing", *data, w, b, lam, "0.1/i", updates
        ))

    pd.DataFrame(trials).to_csv(
        RESULTS / "parameter_trials.csv", index=False
    )

    # Use fixed settings for the main comparison.
    lam = 0.1
    summary = []
    histories = {}

    configurations = [
        ("GD", gradient_descent, {"step_size": 0.001}, 0.001),
        (
            "SGD constant",
            stochastic_gradient_descent,
            {"step_size": 0.0001},
            0.0001,
        ),
        (
            "SGD diminishing",
            stochastic_gradient_descent,
            {"diminishing": True},
            "0.1/i",
        ),
    ]

    for name, algorithm, options, step_label in configurations:
        print(f"\nRunning {name}...")

        w, b, history, updates = algorithm(*data, lam=lam, **options)

        histories[name] = history
        filename = name.lower().replace(" ", "_")

        history.to_csv(
            RESULTS / f"{filename}_history.csv", index=False
        )
        np.savez(RESULTS / f"{filename}_parameters.npz", w=w, b=b)

        row = result_row(
            name, *data, w, b, lam, step_label, updates
        )
        summary.append(row)

        print(f"Training objective: {row['train_objective']:.6f}")
        print(f"Training accuracy: {row['train_accuracy']:.2%}")
        print(f"Testing accuracy: {row['test_accuracy']:.2%}")
        print(f"Final gradient norm: {row['gradient_norm']:.6e}")
        print(f"Tolerance met: {row['gradient_tolerance_met']}")

    # sklearn minimizes MEAN loss + alpha/2 * ||w||^2.
    # alpha = lambda/n matches our summed objective up to scaling.
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=lam / len(y_train),
        max_iter=100,
        tol=None,
        random_state=SEED,
        learning_rate="constant",
        eta0=0.01,
    )
    classifier.fit(X_train, y_train)

    w = classifier.coef_.ravel()
    b = float(classifier.intercept_[0])

    summary.append(result_row(
        "sklearn SGDClassifier",
        *data, w, b, lam, 0.01,
        int(classifier.t_ - 1),
    ))

    comparison = pd.DataFrame(summary)
    comparison.to_csv(RESULTS / "comparison.csv", index=False)

    plot_histories(histories, len(y_train))

    print("\n---------- FINAL COMPARISON ----------")
    print(comparison.to_string(index=False))
    print(f"\nResults saved in: {RESULTS}")
    print(
        "\nUse the histories to discuss convergence. "
        "Reaching the update limit does not establish convergence."
    )


if __name__ == "__main__":
    main()