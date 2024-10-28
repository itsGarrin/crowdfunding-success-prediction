import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.metrics import roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(
    data: pd.DataFrame, threshold: float = 5, removed_variables: list = None
) -> pd.DataFrame:
    """
    Calculate and remove variables with high Variance Inflation Factor (VIF) from a DataFrame.

    This function recursively calculates the VIF for each variable in the provided DataFrame.
    Variables with a VIF greater than the specified threshold are removed to reduce multicollinearity.
    The function prints the names of the removed variables and their VIF values once the recursion is complete.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the variables to be evaluated.
    threshold : float, optional
        The VIF threshold above which variables will be removed. Default is 5.
    removed_variables : list, optional
        A list to collect the names and VIF values of removed variables during recursion. Default is None.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with variables having VIF greater than the threshold removed.
    """
    if removed_variables is None:
        removed_variables = []

    vif = pd.DataFrame()
    vif["variables"] = data.columns
    vif["VIF"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    vif = vif.sort_values(by="VIF", ascending=False)

    if vif["VIF"].max() > threshold:
        removed_variable = vif.iloc[0]
        removed_variables.append(
            (removed_variable["variables"], removed_variable["VIF"])
        )
        data = data.drop(removed_variable["variables"], axis=1)
        return calculate_vif(data, threshold, removed_variables)

    if removed_variables:
        print("Removed variables with high VIF:")
        for var, vif_value in removed_variables:
            print(f"{var}: {vif_value:.2f}")

    return data


def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot the ROC curve for a given set of true labels and predicted probabilities.

    Parameters:
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    model_name : str, optional
        The name of the model (used for the plot title). Default is "Model".

    Returns:
    -------
    None
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--"
    )  # Diagonal line for random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {model_name}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def evaluate_models(
        models: list, predictions_base: list, predictions_hyper: list, X: pd.DataFrame, y_test: list,
        task: str = 'classification'
) -> pd.DataFrame:
    """
    Compare the performance of base models and models with hyperparameter tuning.
    Returns a DataFrame with detailed metrics for each model.

    Parameters:
    ----------
    models : list.
        List of model names.
    predictions_base : list.
        List of predicted values from the base models.
    predictions_hyper : list.
        List of predicted values from hyperparameter-tuned models.
    y_test : array-like.
        Ground truth (correct) labels for the test set.
    task : str.
        Type of evaluation ('classification' or 'regression').

    Returns:
    -------
    pd.DataFrame.
        A DataFrame with detailed metrics for both base and hyperparameter-tuned models.
    """

    def compute_classification_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "Positive Precision": precision_score(y_true, y_pred, pos_label=1),
            "Negative Precision": precision_score(y_true, y_pred, pos_label=0),
            "Positive Recall": recall_score(y_true, y_pred, pos_label=1),
            "Negative Recall": recall_score(y_true, y_pred, pos_label=0),
            "Positive F1 Score": f1_score(y_true, y_pred, pos_label=1),
            "Negative F1 Score": f1_score(y_true, y_pred, pos_label=0),
        }

    def compute_regression_metrics(y_true, y_pred):
        return {
            "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
            "Mean Squared Error": mean_squared_error(y_true, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R-squared": r2_score(y_true, y_pred),
            "Adjusted R-squared": 1 - (1-r2_score(y_true, y_pred))*(len(y_true)-1)/(len(y_true)-X.shape[1]-1)
        }

    # Check for input validity
    if len(predictions_base) != len(models) or len(predictions_hyper) != len(models):
        raise ValueError("The number of predictions must match the number of models.")

    if len(y_test) == 0:
        raise ValueError("y_test cannot be empty.")

    # Compute metrics for both types of models
    all_metrics_base = {}
    all_metrics_hyper = {}

    for model, y_pred_base, y_pred_hyper in zip(models, predictions_base, predictions_hyper):
        if task == 'classification':
            all_metrics_base[model] = compute_classification_metrics(y_test, y_pred_base)
            all_metrics_hyper[model] = compute_classification_metrics(y_test, y_pred_hyper)
        elif task == 'regression':
            all_metrics_base[model] = compute_regression_metrics(y_test, y_pred_base)
            all_metrics_hyper[model] = compute_regression_metrics(y_test, y_pred_hyper)
        else:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    # Initialize the DataFrame structure
    metrics = list(all_metrics_base[models[0]].keys())

    results_base = pd.DataFrame(index=metrics, columns=models)
    results_hyper = pd.DataFrame(index=metrics, columns=models)

    # Fill the DataFrames with metrics
    for model in models:
        for metric in metrics:
            results_base.loc[metric, model] = all_metrics_base[model][metric]
            results_hyper.loc[metric, model] = all_metrics_hyper[model][metric]

    # Combine the base and hyperparameter-tuning results by concatenating vertically
    results_base["Type"] = "Base"
    results_hyper["Type"] = "Hyperparameter Tuning"

    results_combined = pd.concat([results_base, results_hyper])

    results_combined.reset_index(inplace=True)
    results_combined.rename(columns={"index": "Metric"}, inplace=True)

    results_combined.set_index(["Metric", "Type"], inplace=True)

    # Create a single summary column for both base and hyperparameter tuning
    summary_list = []

    for metric in metrics:
        base_values = results_combined.xs("Base", level=1).loc[metric]
        hyper_values = results_combined.xs("Hyperparameter Tuning", level=1).loc[metric]

        if task == 'regression' and metric == "R-squared" or metric == "Adjusted R-squared":
            best_value = max(base_values.max(), hyper_values.max())
            best_model = base_values.idxmax() if base_values.max() >= hyper_values.max() else hyper_values.idxmax()
        else:
            best_value = min(base_values.min(), hyper_values.min())
            best_model = base_values.idxmin() if base_values.min() <= hyper_values.min() else hyper_values.idxmin()

        summary_list.append(
            {"Metric": metric, "Best Model": best_model, "Best Value": best_value}
        )

    summary_df = pd.DataFrame(summary_list)

    summary_df.set_index("Metric", inplace=True)

    # Combine the results_combined DataFrame with the summary DataFrame
    final_results = results_combined.join(summary_df)

    return final_results


def plot_coefficients(ols_results: sm, highlight_vars=None, significance_level=0.05):
    """
    Plots the coefficients of an OLS model with p-values annotated.

    Parameters:
    ols_results: The fitted OLS model results from statsmodels.
    highlight_vars: List of variable names to highlight in the plot.
    significance_level: The p-value threshold for statistical significance.
    """

    # Create a DataFrame for coefficients and p-values
    coef_df = pd.DataFrame({
        'coefficients': ols_results.params.drop('const'),
        'pvalues': ols_results.pvalues.drop('const')
    })

    # Sort by absolute value of coefficients
    coef_df['abs_coeff'] = coef_df['coefficients'].abs()
    coef_df = coef_df.sort_values(by='abs_coeff')

    # Highlight significant coefficients
    significant = coef_df['pvalues'] < significance_level

    plt.figure(figsize=(10, 7))
    bars = plt.bar(coef_df.index, coef_df['coefficients'], color=['red' if sig else 'blue' for sig in significant])
    plt.axhline(0, color='k', linestyle='--')
    plt.title('OLS Coefficients with Significance Highlighted')

    # Annotate p-values on the bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{coef_df['pvalues'].iloc[i]:.3f}", ha='center', va='bottom')

    # Highlight specific variables
    if highlight_vars is not None:
        for i, bar in enumerate(bars):
            if coef_df.index[i] in highlight_vars:
                bar.set_color('orange')

    plt.xticks(rotation=45)
    plt.ylabel('Coefficient Value')
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

