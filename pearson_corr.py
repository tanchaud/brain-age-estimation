import numpy as np
from scipy.stats import pearsonr

def pearson_correlation(y_true, y_pred):
    """
    Calculates Pearson's correlation coefficient.

    Args:
        y_true (np.array or list): First variable (e.g., true values).
        y_pred (np.array or list): Second variable (e.g., predicted values).

    Returns:
        float: Pearson's r.
    """
    y_true = np.asarray(y_true).flatten() # Ensure 1D
    y_pred = np.asarray(y_pred).flatten() # Ensure 1D

    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan # Not defined for less than 2 points

    # Using scipy.stats for a robust implementation that also handles p-value
    r, _ = pearsonr(y_true, y_pred)
    return r

    # Manual implementation based on the MATLAB code:
    # mean_y_true = np.mean(y_true)
    # mean_y_pred = np.mean(y_pred)
    #
    # t1 = y_true - mean_y_true
    # t2 = y_pred - mean_y_pred
    #
    # sum_t1_t2 = np.sum(t1 * t2)
    #
    # sum_t1_sq = np.sum(t1**2)
    # sum_t2_sq = np.sum(t2**2)
    #
    # denominator = np.sqrt(sum_t1_sq) * np.sqrt(sum_t2_sq)
    #
    # if denominator == 0:
    #     return np.nan # Avoid division by zero; correlation is undefined or perfect if std dev is zero
    #
    # r_manual = sum_t1_t2 / denominator
    # return r_manual

if __name__ == '__main__':
    y_true_example = np.array([1, 2, 3, 4, 5])
    y_pred_example_pos = np.array([2, 4, 5, 4, 6]) # Positive correlation
    y_pred_example_neg = np.array([5, 4, 3, 2, 1]) # Perfect negative correlation
    y_pred_example_no_corr = np.array([3, 3, 3, 3, 3]) # No correlation (or NaN if using manual due to 0 std dev for pred)


    r_pos = pearson_correlation(y_true_example, y_pred_example_pos)
    print(f"Pearson correlation (positive): {r_pos}") # Expected: around 0.87

    r_neg = pearson_correlation(y_true_example, y_pred_example_neg)
    print(f"Pearson correlation (negative): {r_neg}") # Expected: -1.0

    r_no_corr = pearson_correlation(y_true_example, y_pred_example_no_corr)
    print(f"Pearson correlation (no correlation): {r_no_corr}") # Expected: nan or 0, scipy handles it as nan if one variable is constant

    # Test with example from MATLAB script context
    Y_example = np.array([20, 25, 30, 35, 40, 45, 50])
    Y_pred_example = np.array([22, 23, 32, 33, 38, 48, 52])
    r_context = pearson_correlation(Y_example, Y_pred_example)
    print(f"Pearson correlation (context example): {r_context}")