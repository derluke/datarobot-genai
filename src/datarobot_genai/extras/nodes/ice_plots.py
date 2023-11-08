# typing: ignore
import concurrent.futures
import logging

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedShuffleSplit

from ..utils import get_or_request_training_predictions, upload_dataset

log = logging.getLogger(__name__)


def partial_dependence(
    project: dr.Project,
    model: dr.Model,
    data: pd.DataFrame,
    feature: str,
    size_of_grid: int = 25,
    sample_size: int = 1000,
):

    np.random.seed(42)
    random_state = np.random.RandomState(42)
    data = data.copy()
    if project.target_type == "Multiclass":
        raise ValueError("Feature Effects is not support for Multiclass yet.")

    if feature not in data.columns:
        raise ValueError("Specified feature is not found in dataset.")

    if sample_size:
        # Random sample if regression or if target is not included in dataset
        if (project.target_type == "Regression") | (project.target not in data.columns):
            # Random sample
            data = data.sample(
                n=min(len(data), sample_size), replace=False, random_state=random_state
            )
        else:
            # Stratified random sample
            max_sample_size = min(len(data) - 2, sample_size)
            sampler = StratifiedShuffleSplit(
                n_splits=1,
                train_size=max_sample_size,
                test_size=len(data) - max_sample_size,
                random_state=random_state,
            )
            idx, _ = list(
                sampler.split(
                    np.zeros_like(data[project.target]), y=data[project.target]
                )
            )[0]
            data = data.iloc[idx]

    if np.issubdtype(data[feature].dtype, np.number):  # type: ignore
        feature_type = "numeric"
    else:
        feature_type = "categorical"

    cats = data[feature].unique()

    if len(cats) > size_of_grid:
        if feature_type == "numeric":
            sampled_values = np.nanquantile(
                data[feature], q=np.linspace(start=0.05, stop=0.95, num=size_of_grid)
            )
        else:
            sampled_values = np.random.choice(cats, size=size_of_grid, replace=False)
    else:
        sampled_values = cats

    data["rowID"] = range(len(data))
    df = []
    for x in sampled_values:
        data_temp = data.copy()
        data_temp[feature] = x
        df.append(data_temp)
    augmented_dataset = pd.concat(df)
    log.info(f"uploading augmented dataset with {len(augmented_dataset)} rows.")
    augmented_dataset_id = upload_dataset(project, augmented_dataset)

    log.info("Requesting predictions on augmented dataset")
    pred_job_id = model.request_predictions(augmented_dataset_id.id)
    predictions = pred_job_id.get_result_when_complete()
    assert isinstance(predictions, pd.DataFrame)
    if project.target_type == "Regression":
        augmented_dataset["predictions"] = predictions["prediction"].values
    else:
        augmented_dataset["predictions"] = predictions["positive_probability"].values
    ice_plot_data = augmented_dataset[[feature, "rowID", "predictions"]].reset_index(
        drop=True
    )

    pd_plot_data = ice_plot_data.groupby(feature).apply(
        lambda df: pd.Series(
            {
                "mean_pred": df["predictions"].mean(),
                "sd": df["predictions"].std(),
            }
        )
    )
    if "mean_pred" not in pd_plot_data.columns:
        pd_plot_data["mean_pred"] = np.nan
    if "sd" not in pd_plot_data.columns:
        pd_plot_data["sd"] = np.nan

    pd_plot_data["mean_minus_sd"] = pd_plot_data["mean_pred"] - pd_plot_data["sd"]
    pd_plot_data["mean_plus_sd"] = pd_plot_data["mean_pred"] + pd_plot_data["sd"]

    return pd_plot_data, ice_plot_data, feature_type


def get_plot(
    pd_plot_data: pd.DataFrame,
    ice_plot_data: pd.DataFrame,
    feature: str,
    feature_type: str,
    ice_plot: bool = True,
    std_dev_plot: bool = False,
    target_name: str = "target",
):

    log.info(f"Preparing plots for feature {feature}")

    # Color scheme
    predicted_color = "purple"
    background_color = "white"

    fig, ax = plt.subplots(figsize=(20, 7))
    # log.info(ice_plot_data.head())
    if ice_plot:
        if feature_type == "numeric":
            g = sns.lineplot(
                x=feature,
                y="predictions",
                data=ice_plot_data,
                units="rowID",
                estimator=None,  # type: ignore
                linewidth=0.2,
                alpha=0.2,
                color="blue",
                ax=ax,
                legend="brief"
                # label="ICE",
            )
        else:
            g = sns.stripplot(
                x=feature,
                y="predictions",
                data=ice_plot_data,
                color="blue",
                size=1,
                jitter=True,  # type: ignore
                alpha=0.3,
                order=pd_plot_data.sort_values(by="mean_pred").index,
                zorder=1,
                ax=ax,
                legend="brief",
            )
    if feature_type == "numeric":
        g = sns.lineplot(
            x=feature,
            y="mean_pred",
            data=pd_plot_data.reset_index(),
            linewidth=3,
            color=predicted_color,
            ax=ax,
            label="Mean",
        )
        # g = sns.lineplot(
        #     x=feature,
        #     y="mean_pred",
        #     data=pd_plot_data.reset_index(),
        #     linewidth=2,
        #     color="gold",
        #     ax=ax,
        # )
    else:
        g = sns.pointplot(
            x=feature,
            y="mean_pred",
            data=pd_plot_data.reset_index(),
            color=predicted_color,
            linestyle="none",
            order=pd_plot_data.sort_values(by="mean_pred").index,
            markersize=4,
            ax=ax,
            label="Mean",
        )
        # g = sns.pointplot(
        #     x=feature,
        #     y="mean_pred",
        #     data=pd_plot_data.reset_index(),
        #     color="gold",
        #     linestyle="none",
        #     order=pd_plot_data.sort_values(by="mean_pred").index,
        #     markersize=5,
        #     ax=ax,
        # )
        # g.set_yticks(g.get_yticks().tolist())
        g.set_xticklabels(g.get_xticklabels(), rotation=30, ha="right")
        # g.tick_params(axis="x", labelrotation=30, ha="right")

    if std_dev_plot:
        if feature_type == "numeric":
            g = sns.lineplot(
                x=feature,
                y="mean_plus_sd",
                data=pd_plot_data.reset_index(),
                linewidth=1.5,
                linestyle="--",
                color=predicted_color,
            )
            g = sns.lineplot(
                x=feature,
                y="mean_minus_sd",
                data=pd_plot_data.reset_index(),
                linewidth=1.5,
                linestyle="--",
                color=predicted_color,
                label="Mean +/- SD",
            )
        else:
            g = sns.pointplot(
                x=feature,
                y="mean_plus_sd",
                data=pd_plot_data.reset_index(),
                color=predicted_color,
                linestyle="none",
                # linestyle="--",
                order=pd_plot_data.sort_values(by="mean_pred").index,
                markersize=4.0,
                # markerstyle="o",
                markerfacecolor="none",
                markeredgewidth=1,
                # edgecolors=predicted_color,
                # alpha=0.8,
                label="Mean +/- SD",
            )
            g = sns.pointplot(
                x=feature,
                y="mean_minus_sd",
                data=pd_plot_data.reset_index(),
                color=predicted_color,
                linestyle="none",
                # markerstyle="o",
                markerfacecolor="none",
                markeredgewidth=1,
                # edgecolors=predicted_color,
                order=pd_plot_data.sort_values(by="mean_pred").index,
                markersize=4.0,
                # alpha=0.8,
            )
    g.set_title("Partial Dependence")
    g.yaxis.set_label_text(f"Target ({target_name})")
    ax.set_facecolor(background_color)
    ax.legend()
    # ax.set_xlabel("Bins based on predicted value")
    # ax.set_ylabel("Average target value")
    ax.grid(visible=True, color="black", linewidth=0.25)
    plt.margins(0.025)  # to reduce white space

    # plot_writer = MatplotlibWriter(filepath=f"data/08_reporting/ice_plot_{feature}.png")
    # plt.close()
    # plot_writer.save(fig)
    return fig


def create_ice_plots(model: dr.Model, training_data: pd.DataFrame) -> dict[str, Figure]:
    """
    Create ICE plots for each feature used in the model.

    Args:
        model: DataRobot model object.
        training_data: DataRobot training data.
        params: Kedro parameters.

        Returns:
            List of ICE plots.
    """
    if model.project_id is None:
        raise ValueError("Model is not associated with a project.")
    project = dr.Project.get(model.project_id)
    assert isinstance(project, dr.Project)

    if project.is_datetime_partitioned:
        validation_predictions = get_or_request_training_predictions(
            model, dr.enums.DATA_SUBSET.ALL_BACKTESTS  # type: ignore
        )
    else:
        validation_predictions = get_or_request_training_predictions(
            model, dr.enums.DATA_SUBSET.VALIDATION_AND_HOLDOUT  # type: ignore
        )
    validation_data = training_data.iloc[validation_predictions["row_id"]]

    features = model.get_features_used()

    ice_plots = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                partial_dependence,
                project,
                model,
                validation_data,
                feature,
            )
            for feature in features
        ]
        for feature, future in zip(features, futures):
            pd_plot_data, ice_plot_data, feature_type = future.result()
            plot = get_plot(
                pd_plot_data,
                ice_plot_data,
                feature,
                feature_type,
                ice_plot=True,
                std_dev_plot=True,
                target_name=project.target,  # type: ignore
            )
            ice_plots[feature] = plot

        # for feature in features:
        #     try:
        #         plot = partial_dependence(
        #             project,
        #             model,
        #             validation_data,
        #             feature,
        #             ice_plot=True,
        #             std_dev_plot=True,
        #         )

        #         ice_plots[feature] = plot
        #     except Exception as e:
        #         log.info(e)

    return ice_plots
