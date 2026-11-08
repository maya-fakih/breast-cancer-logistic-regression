from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def drop_correlated_features(df, threshold=0.9):
    corr = df.drop(columns="Label").corr()
    kept = []
    dropped = []

    for feature in corr.columns:
        if any(corr.loc[feature, other] > threshold for other in kept):
            dropped.append(feature)
        else:
            kept.append(feature)

    print("Dropped features:", dropped)
    print("Remaining feature count:", len(kept))

    return df[kept + ["Label"]].copy()


def prepare_data(df):
    X = df.drop(columns="Label")
    y = df["Label"]

    # Assignment: 500 training patients and 69 testing patients.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=69,
        random_state=42,
    )

    # Fit on training data only.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def loss(X, y, w, b, lam):
    z = X @ w + b

    # Stable evaluation of the assignment's summed objective.
    return (
        np.sum(np.logaddexp(0, z) - y * z)
        + (lam / 2) * np.dot(w, w)
    )


def cross_entropy(X, y, w, b):
    z = X @ w + b

    # Mean loss allows comparison between differently sized datasets.
    return np.mean(np.logaddexp(0, z) - y * z)


def accuracy(X, y, w, b):
    probabilities = expit(X @ w + b)
    predictions = (probabilities > 0.5).astype(int)
    return np.mean(predictions == y)


def gradient_descent(
    X_train,
    y_train,
    X_test,
    y_test,
    step_size=0.001,
    lam=0.1,
    max_iter=20000,
    tolerance=1e-5,
):
    w = np.zeros(X_train.shape[1])
    b = 0.0
    history = []

    def record(iteration):
        history.append({
            "iteration": iteration,
            "train_objective": loss(X_train, y_train, w, b, lam),
            "train_loss": cross_entropy(X_train, y_train, w, b),
            "test_loss": cross_entropy(X_test, y_test, w, b),
            "train_accuracy": accuracy(X_train, y_train, w, b),
            "test_accuracy": accuracy(X_test, y_test, w, b),
        })

    record(0)
    updates = 0

    for _ in range(max_iter):
        probabilities = expit(X_train @ w + b)
        error = probabilities - y_train

        # Gradient of summed cross-entropy + L2 regularization.
        grad_w = X_train.T @ error + lam * w
        grad_b = np.sum(error)

        grad_norm = np.sqrt(np.dot(grad_w, grad_w) + grad_b**2)

        if grad_norm < tolerance:
            break

        w -= step_size * grad_w
        b -= step_size * grad_b
        updates += 1

        if updates % 100 == 0:
            record(updates)

    if history[-1]["iteration"] != updates:
        record(updates)

    # Check the gradient at the final parameters.
    error = expit(X_train @ w + b) - y_train
    grad_w = X_train.T @ error + lam * w
    grad_b = np.sum(error)
    grad_norm = np.sqrt(np.dot(grad_w, grad_w) + grad_b**2)

    return w, b, pd.DataFrame(history), updates, grad_norm


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(root / "data" / "WDBC_data.xlsx")
    df = drop_correlated_features(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    step_size = 0.001
    lam = 0.1

    w, b, history, updates, grad_norm = gradient_descent(
        X_train,
        y_train,
        X_test,
        y_test,
        step_size=step_size,
        lam=lam,
    )

    print("\n----------Gradient descent----------")
    print(f"Step size: {step_size}")
    print(f"Lambda: {lam}")
    print(f"Updates: {updates}")
    print(f"Final gradient norm: {grad_norm:.6e}")
    print(f"Converged to tolerance: {grad_norm < 1e-5}")
    print(f"Training objective: {loss(X_train, y_train, w, b, lam):.6f}")
    print(f"Training accuracy: {accuracy(X_train, y_train, w, b):.2%}")
    print(f"Testing accuracy: {accuracy(X_test, y_test, w, b):.2%}")

    history.to_csv(results_dir / "gd_history.csv", index=False)
    np.savez(results_dir / "gd_parameters.npz", w=w, b=b)

    print(f"\nResults saved in: {results_dir}")


if __name__ == "__main__":
    main()