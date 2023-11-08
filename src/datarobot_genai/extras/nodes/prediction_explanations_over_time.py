# type: ignore
import re
import logging
import pandas as pd
import altair as alt

# pyright: reportPrivateImportUsage=false
import datarobot as dr
from ..utils.utils import upload_dataset

log = logging.getLogger(__name__)


def get_prediction_explanations(
    scoring_data: pd.DataFrame,
    model: dr.Model,
    max_explanations: int = 5,
    date_col: str = "date_col",
) -> pd.DataFrame:
    if model.project_id is None:
        raise RuntimeError("Model has no project ID")
    project = dr.Project.get(model.project_id)
    _ = model.get_or_request_feature_impact()

    dataset = upload_dataset(project, scoring_data)

    try:
        pes = dr.PredictionExplanations.list(project.id, model.id)
        pe = [
            pe
            for pe in pes
            if pe.dataset_id == dataset.id
            and pe.max_explanations == max_explanations
            and pe.threshold_low is None
            and pe.threshold_high is None
        ][0]
        log.info("PE: Prediction Explanations already generated")
    except Exception:  # pylint: disable=broad-exception-caught
        log.info("PE: Generating Prediction Explanations")

        log.info("PE: Generating Predictions")
        pred_job = model.request_predictions(dataset.id)
        pred_job.wait_for_completion(max_wait=60 * 120)

        log.info("PE: Initialising Prediction Explanations")
        pei_job = dr.PredictionExplanationsInitialization.create(
            model.project_id, model.id
        )
        pei_job.wait_for_completion(max_wait=60 * 120)

        log.info("PE: Creating Prediction Explanations")
        pe_job = dr.PredictionExplanations.create(
            project_id=project.id,
            model_id=model.id,
            dataset_id=dataset.id,
            max_explanations=max_explanations,
        )

        pe = pe_job.get_result_when_complete(max_wait=60 * 120)

    if not isinstance(pe, dr.PredictionExplanations):
        raise RuntimeError("Prediction Explanations failed to generate")

    log.info("PE: Downloading Prediction Explanations")
    raw_explanations = pe.get_all_as_dataframe()

    log.info("PE: Transforming Prediction Explanations")
    raw_explanations[date_col] = scoring_data[date_col].values
    dfs = {}
    for col in [
        "feature",
        "feature_value",
        "label",
        "qualitative_strength",
        "strength",
    ]:
        pattern = re.compile(f"explanation_(\d)_{col}$")
        tmp = (
            raw_explanations[[c for c in raw_explanations.columns if pattern.match(c)]]
            .stack()
            .reset_index()
        )
        tmp["explanation_n"] = tmp["level_1"].apply(
            lambda x: int(re.findall(pattern, x)[0])
        )
        tmp = tmp.rename(columns={"level_1": "col", 0: f"{col}_values"})
        tmp = tmp.set_index(["level_0", "explanation_n"])
        dfs[col] = tmp

    dfs_stacked = pd.concat(list(dfs.values()), axis=1, join="inner")
    df_explanations = (
        dfs_stacked.reset_index()
        .set_index("level_0")
        .join(raw_explanations[[date_col, "prediction"]])
    )

    explanations = df_explanations.groupby([date_col, "feature_values"])[
        "strength_values"
    ].aggregate(
        ("mean", "count")  # type: ignore
    )
    predictions = df_explanations.groupby([date_col])["prediction"].mean()

    explanations = explanations.reset_index().join(predictions, on=date_col)

    explanations.columns = [
        date_col,
        "Feature",
        "Mean Explanation Strength",
        "Number Observations",
        "Mean Prediction",
    ]

    return explanations


def get_prediction_explanation_over_time_chart(
    prediction_explanation_df: pd.DataFrame,
    date_aggregation: str = "W",
    max_prediction_explanation_threshold: float | None = None,
    date_col: str = "date_col",
) -> alt.LayerChart:

    groupby_window = pd.Grouper(key=date_col, axis=0, freq=date_aggregation)

    prediction_explanation_df[date_col] = pd.to_datetime(
        prediction_explanation_df[date_col]
    )

    prediction_explanation_df.head()

    prediction_explanation_df["strength_values_sum"] = (
        prediction_explanation_df["Mean Explanation Strength"]
        * prediction_explanation_df["Number Observations"]
    )

    bar_plot_df = prediction_explanation_df.groupby([groupby_window, "Feature"])[
        ["strength_values_sum", "Number Observations"]
    ].sum()

    bar_plot_df["Mean Explanation Strength"] = (
        bar_plot_df["strength_values_sum"] / bar_plot_df["Number Observations"]
    )

    bar_plot_df = bar_plot_df.reset_index()

    bar_plot_df.columns = [
        "date",
        "Feature",
        "x",
        "Number Observations",
        "Mean Explanation Strength",
    ]

    line_plot_df = (
        prediction_explanation_df.groupby(groupby_window)[["Mean Prediction"]]
        .mean()
        .reset_index()
    )
    line_plot_df.columns = ["date", "Mean Prediction"]

    # bar_plot_df = bar_plot_df.set_index("date")
    # bar_plot_df["Mean Explanation Strength"] += line_plot_df.set_index("date")[
    #     "Mean Prediction"
    # ]
    # bar_plot_df = bar_plot_df.reset_index()

    line_plot = (
        alt.Chart(line_plot_df)
        .mark_line()
        .encode(x=alt.X("date:T"), y=alt.Y("Mean Prediction:Q"))
    )

    if max_prediction_explanation_threshold is not None:
        bar_plot_df = bar_plot_df[
            bar_plot_df["Mean Explanation Strength"]
            < max_prediction_explanation_threshold
        ]
    bar_plot = (
        alt.Chart(bar_plot_df)
        .mark_bar(size=20)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y(
                "Mean Explanation Strength:Q",
                axis=alt.Axis(title="Average Feature Strength", titleColor="#57A44C"),
            ),
            color=alt.Color("Feature"),
            tooltip=[
                "date",
                "Feature",
                "Number Observations",
                "Mean Explanation Strength",
            ],
        )
    )

    line_bar = (
        alt.layer(bar_plot, line_plot)
        .resolve_scale(y="independent")
        .properties(width=800, height=300)
        .add_params(alt.selection_interval(bind="scales", encodings=["x"]))
    )

    return line_bar
