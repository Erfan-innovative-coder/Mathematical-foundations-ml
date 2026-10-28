import numpy as np


def fit_linear_regression(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    X_design = np.c_[np.ones(X.shape[0]), X]

    theta = np.linalg.pinv(X_design) @ y

    return theta

def predict(X, theta):
    X = np.asarray(X)

    X_design = np.c_[np.ones(X.shape[0]), X]

    return X_design @ theta

def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = (
        1 - np.sum((y_true - y_pred) ** 2) / denominator
        if denominator > 0
        else float("nan")
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }