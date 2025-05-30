import numpy as np
from sklearn.metrics import mean_absolute_error as sklearn_mae

def mae(y_true, y_pred):
    """
    Calculates Mean Absolute Error.

    Args:
        y_true (np.array or list): True target values.
        y_pred (np.array or list): Predicted values.

    Returns:
        float: The Mean Absolute Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Formula: sum(abs(y_pred - y_true)) / len(y_true)
    # return np.mean(np.abs(y_pred - y_true))
    # Or using scikit-learn for a robust implementation:
    return sklearn_mae(y_true, y_pred)

if __name__ == '__main__':
    y_true_example = np.array([3, -0.5, 2, 7])
    y_pred_example = np.array([2.5, 0.0, 2, 8])
    mae_val = mae(y_true_example, y_pred_example)
    print(f"MAE: {mae_val}") # Expected: (0.5 + 0.5 + 0 + 1) / 4 = 0.5

    y_true_example_2 = [10, 20, 30, 25]
    y_pred_example_2 = [12, 18, 33, 28]
    mae_val_2 = mae(y_true_example_2, y_pred_example_2)
    print(f"MAE: {mae_val_2}") # Expected: (2 + 2 + 3 + 3) / 4 = 10/4 = 2.5