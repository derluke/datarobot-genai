import logging

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from ..utils import get_or_request_training_predictions

log = logging.getLogger(__name__)


def _get_predictions(
    project: dr.Project, model: dr.Model, partition: dr.enums.DATA_SUBSET
):

    if not project.holdout_unlocked and partition == dr.enums.DATA_SUBSET.HOLDOUT:  # type: ignore
        log.info("Unlocking holdout partition")
        project.unlock_holdout()

    training_predictions = get_or_request_training_predictions(model, partition)

    return training_predictions


def _generate_customized_lift_chart(
    plot_data: pd.DataFrame, title: str = "", std_multiplier: int = 1
) -> Figure:
    """
    This function creates a lift chart with a custom color scheme and displays the prediction standard deviation in each bin.

    Parameters
    ----------
    plot_data: output of the function "prepare_lift_chart_data()"
    title: what to use for the chart's title
    std_multipler: how much to multply the standard deviation about the prediction mean
    """

    # Color scheme
    predicted_color = "purple"
    actual_color = "green"
    background_color = "white"

    # Defining bins
    bin_labels = range(1, plot_data.shape[0] + 1)

    # Initialize
    fig, ax = plt.subplots(figsize=(20, 7))

    # Plots
    ax.plot(
        bin_labels,
        plot_data["actual"],
        lw=2,
        color=actual_color,
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Actual",
    )
    ax.plot(
        bin_labels,
        plot_data["predicted"],
        lw=2,
        color=predicted_color,
        marker="+",
        markersize=8,
        markeredgewidth=2,
        label="Predicted",
    )

    # Specifying standard deviation boundary
    lower_bound = plot_data["predicted"] - std_multiplier * plot_data["predicted_std"]
    upper_bound = plot_data["predicted"] + std_multiplier * plot_data["predicted_std"]

    # If binary target, clip at [0, 1]
    if plot_data["actual"].between(0, 1).all():
        lower_bound = np.clip(lower_bound, 0, 1)  # type: ignore
        upper_bound = np.clip(upper_bound, 0, 1)

    # Shaded region for standard deviation
    ax.fill_between(
        bin_labels,
        lower_bound,  # type: ignore
        upper_bound,  # type: ignore
        alpha=0.3,
        color=predicted_color,
    )

    # Settings
    ax.set_facecolor(background_color)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Bins based on predicted value")
    ax.set_ylabel("Average target value")
    ax.grid(visible=True, color="black", linewidth=0.25)
    plt.margins(0.025)  # to reduce white space

    # To ensure the xticks aren't too crowded
    if plot_data.shape[0] <= 60:
        plt.xticks(np.arange(1, plot_data.shape[0] + 1, 1))

    plt.title(title)
    # plot_writer = MatplotlibWriter(
    #     filepath=f"data/08_reporting/lift_chart_plot_{n_bins}_bins.png"
    # )
    # plt.close()
    # plot_writer.save(fig)

    return fig


def _prepare_lift_chart_data(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    This function aggregates data into specified bins for easier plotting.

    Parameters
    ----------
    preds: dataframe consisting of a columns "predicted" and "actual" representing the predicted and target values, respectively
    n_bins: how many groups to use for splitting the data

    Returns
    ----------
    Dataframe containing aggregated data needed to create DataRobot's lift chart
    """

    # First define the bins based on the predictions
    preds = preds.copy()
    preds["bins"] = pd.qcut(
        preds["predicted"],
        q=n_bins,
        labels=np.arange(1, n_bins + 1, 1).tolist(),
        retbins=False,
    ).values

    # Aggregate
    plot_data = (
        preds.groupby(["bins"], observed=True)
        .agg(
            actual=("actual", "mean"),
            predicted=("predicted", "mean"),
            bin_weight=("bins", "size"),
            predicted_std=("predicted", "std"),
        )
        .reset_index(drop=True)
    )

    return plot_data


def create_custom_lift_chart(
    model: dr.Model, lift_chart_data: pd.DataFrame, params
) -> dict[str, Figure]:
    """
    This function creates a lift chart with a custom color scheme and displays the prediction standard deviation in each bin.

    Parameters
    ----------
    model: the model to use for generating predictions
    lift_chart_data: dataframe containing the actuals and predictions
    params: dictionary containing the parameters for the lift chart

    Returns
    ----------
    Dictionary containing the lift chart plots for each bin size
    """

    project = model.project
    if project is None:
        raise ValueError("Model must be associated with a project")
    holdout_preds = _get_predictions(project, model, params["PARTITION"])
    target = project.target

    # Join in the actuals using the "row_id" column
    holdout_preds_with_actuals = (
        holdout_preds.set_index("row_id")
        .join(lift_chart_data[target])
        .rename(columns={"prediction": "predicted", target: "actual"})
        .drop("partition_id", axis=1)
        .reset_index()
    )

    # Preprocess the output if you have a binary classification problem
    if project.target_type == "Binary":
        # If a string, turn target into binary 1/0
        if pd.api.types.is_object_dtype(holdout_preds_with_actuals["actual"]):
            holdout_preds_with_actuals["actual"] = (
                holdout_preds_with_actuals["actual"] == project.positive_class
            ).astype(int)

            holdout_preds_with_actuals["predicted"] = holdout_preds_with_actuals[
                f"class_{project.positive_class}"
            ]

        else:
            # check if positive class str is numeric:
            try:
                positive_class = float(project.positive_class)  # type: ignore
            except ValueError:
                positive_class = project.positive_class
            # Take the positive class probability predictions only
            holdout_preds_with_actuals["predicted"] = holdout_preds_with_actuals[
                f"class_{positive_class}"
            ]

        # Keep only relevant columns
        holdout_preds_with_actuals = holdout_preds_with_actuals[["predicted", "actual"]]

    lift_chart_plots = {}
    for bins in params["BINS"]:
        # Prep and plot
        lift_chart_data = _prepare_lift_chart_data(
            preds=holdout_preds_with_actuals, n_bins=bins
        )
        lf_plot = _generate_customized_lift_chart(
            plot_data=lift_chart_data,
            title=f"Lift Chart with +/- {params['STANDARD_DEVIATIONS']} Standard Deviation ({bins} bins)",
            std_multiplier=params["STANDARD_DEVIATIONS"],
        )
        lift_chart_plots[bins] = lf_plot

    return lift_chart_plots
