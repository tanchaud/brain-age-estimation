import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    """
    Calculates Root Mean Squared Error.

    Args:
        y_true (np.array or list): True target values.
        y_pred (np.array or list): Predicted values.

    Returns:
        float: The Root Mean Squared Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Formula: sqrt(mean((y_true - y_pred)^2))
    # return np.sqrt(np.mean((y_true - y_pred)**2))
    # Or using scikit-learn for a robust implementation:
    return np.sqrt(mean_squared_error(y_true, y_pred))


if __name__ == '__main__':
    y_true_example = np.array([3, -0.5, 2, 7])
    y_pred_example = np.array([2.5, 0.0, 2, 8])
    rmse_val = rmse(y_true_example, y_pred_example)
    # MSE = ( (0.5)^2 + (0.5)^2 + (0)^2 + (-1)^2 ) / 4 = (0.25 + 0.25 + 0 + 1) / 4 = 1.5 / 4 = 0.375
    # RMSE = sqrt(0.375) approx 0.612
    print(f"RMSE: {rmse_val}")

    y_true_example_2 = [10, 20, 30, 25]
    y_pred_example_2 = [12, 18, 33, 28]
    rmse_val_2 = rmse(y_true_example_2, y_pred_example_2)
    # MSE = ( (-2)^2 + (2)^2 + (-3)^2 + (-3)^2 ) / 4 = (4+4+9+9)/4 = 26/4 = 6.5
    # RMSE = sqrt(6.5) approx 2.5495
    print(f"RMSE: {rmse_val_2}")