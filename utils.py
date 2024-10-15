import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    models: list, predictions_base: list, predictions_hyper: list, y_test: list
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

    Returns:
    -------
    pd.DataFrame.
        A DataFrame with detailed metrics including accuracy, precision, recall, and F1 score for both base and hyperparameter-tuned models.
    """

    def compute_metrics(y_true, y_pred):
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

    # Compute metrics for both types of models
    all_metrics_base = {
        model: compute_metrics(y_test, y_pred)
        for model, y_pred in zip(models, predictions_base)
    }
    all_metrics_hyper = {
        model: compute_metrics(y_test, y_pred)
        for model, y_pred in zip(models, predictions_hyper)
    }

    # Initialize the DataFrame structure
    metrics = list(compute_metrics(y_test, predictions_base[0]).keys())
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

        max_base_value = base_values.max()
        max_hyper_value = hyper_values.max()

        if max_base_value >= max_hyper_value:
            best_model = base_values.idxmax()
            best_value = max_base_value
        else:
            best_model = hyper_values.idxmax()
            best_value = max_hyper_value

        summary_list.append(
            {"Metric": metric, "Best Model": best_model, "Highest Value": best_value}
        )

    summary_df = pd.DataFrame(summary_list)
    summary_df.set_index("Metric", inplace=True)

    # Combine the results_combined DataFrame with the summary DataFrame
    final_results = results_combined.join(summary_df)

    return final_results
