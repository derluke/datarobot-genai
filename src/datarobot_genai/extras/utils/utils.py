# type: ignore
#
# This file contains utility functions that are used in the nodes.


import hashlib
import logging
import time
from collections import Counter
from pathlib import Path
import tempfile
import requests

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd

log = logging.getLogger(__name__)


def get_list_as_dict(list_of_dr_objects, index):
    """
    Returns a dictionary with the index of the list_of_dr_objects as key and the object as value.
    :param list_of_dr_objects: list of objects
    :param index: index of the list_of_dr_objects to be used as key
    :return: dictionary with the index of the list_of_dr_objects as key and the object as value
    """
    return {vars(o)[index]: vars(o) for o in list_of_dr_objects}


def get_featurelist_by_name(project: dr.Project, name: str):
    """
    Returns the featurelist object for a given project and name.
    :param project: dr.Project object
    :param name: name of the featurelist
    :return: featurelist object
    """
    return [fl for fl in dr.Featurelist.list(project.id) if fl.name == name][0]  # type: ignore


def get_or_request_training_predictions(
    model: dr.Model, data_subset: dr.enums.DATA_SUBSET
) -> pd.DataFrame:
    """
    Returns the training predictions for a given model and data subset.
    :param model: model object
    :param data_subset: data subset to get the training predictions for
    :return: training predictions as pandas dataframe
    """
    try:
        training_prediction_job = model.request_training_predictions(
            data_subset=data_subset
        )
        training_prediction_job.wait_for_completion()
        training_predictions = training_prediction_job.get_result_when_complete()
        if not isinstance(training_predictions, dr.TrainingPredictions):
            raise ValueError(
                "Training predictions are not a dr.TrainingPredicitons instance: "
                + f"{training_predictions}"
            )
    except dr.errors.ClientError as e:
        if e.status_code == 422:
            training_predictions = [
                tp
                for tp in dr.TrainingPredictions.list(model.project_id)
                if tp.data_subset == data_subset and tp.model_id == model.id
            ][0]
        else:
            raise e
    return training_predictions.get_all_as_dataframe()


def _hash_pandas(df: pd.DataFrame):
    """
    Returns the hash of a pandas dataframe.
    :param self: pandas dataframe
    :return: hash of the pandas dataframe
    """
    return (
        str(
            int(
                hashlib.sha256(
                    pd.util.hash_pandas_object(df, index=True).values  # type: ignore
                ).hexdigest(),
                16,
            )
        )
        + ".csv.gz"
    )


# dr.Project._upload_dataset = dr.Project.upload_dataset


def upload_dataset(project: dr.Project, sourcedata: pd.DataFrame, **kwags):
    """
    Uploads a pandas dataframe to a project as a dataset. This function checks if the
    dataset has already been uploaded and returns the dataset object if it has.

    :param self: project object
    :param sourcedata: pandas dataframe to be uploaded
    :return: dataset object
    """

    # check all uploaded dataset's filenames:
    datasets = project.get_datasets()
    filenames = {d.name: d for d in datasets}

    hashed_df = _hash_pandas(sourcedata)
    if hashed_df in filenames:  # type: ignore
        return filenames[hashed_df]  # type: ignore
    else:
        # pop filename from kwargs
        _ = kwags.pop("dataset_filename", None)
        with tempfile.TemporaryDirectory() as d:
            temp_csv = Path(d) / hashed_df
            sourcedata.to_csv(temp_csv, index=False, compression="gzip")
            try:
                dataset = project.upload_dataset(temp_csv, **kwags)
            except TimeoutError:
                time.sleep(5)
                dataset = project.upload_dataset(temp_csv, **kwags)
        return dataset


# dr.Project.upload_dataset = upload_dataset


def predict(model: dr.Model, df: pd.DataFrame):
    """
    Returns the predictions for a given model and data.

    :param model: dr.Model object
    :param df: data to get the predictions for
    :return: predictions as pandas dataframe
    """
    if model.project_id is None:
        raise ValueError("Model has no project associated with it.")
    project = dr.Project.get(model.project_id)
    ds = upload_dataset(project, df)
    pred_job = model.request_predictions(ds.id)
    pred_job.wait_for_completion()
    return pred_job.get_result_when_complete()


def _get_multiseries_id_columns(deployment: dr.Deployment):
    """
    Returns the multiseries id columns for a given deployment.
    :param self: deployment object
    :return: list of multiseries id columns
    """
    if deployment.model is None:
        raise ValueError("Deployment has no model associated with it.")
    project_id = deployment.model["project_id"]

    datetime_partitioning = dr.DatetimePartitioning.get(project_id)
    multiseries_id_columns = datetime_partitioning.multiseries_id_columns
    if not multiseries_id_columns or len(multiseries_id_columns) == 0:
        raise ValueError("Project is not a time series project.")
    return multiseries_id_columns[0].replace(" (actual)", "")


def _get_date_format(deployment: dr.Deployment):
    """
    Returns the date format for a given deployment.
    :param self: deployment object
    :return: date format
    """
    model = deployment.model
    if model is None:
        raise ValueError("Deployment has no model associated with it.")

    return dr.DatetimePartitioning.get(model["project_id"]).date_format


def _get_datetime_partition_column(deployment: dr.Deployment):
    """
    Returns the datetime partition column for a given deployment.
    :param self: deployment object
    :return: datetime partition column
    """
    model = deployment.model
    if model is None:
        raise ValueError("Deployment has no model associated with it.")

    datetime_partitioning = dr.DatetimePartitioning.get(model["project_id"])
    datetime_partition_column = datetime_partitioning.datetime_partition_column
    if not datetime_partition_column:
        raise ValueError("Project is not a time series project.")
    return datetime_partition_column.replace(" (actual)", "")


def deployment_predict(
    deployment: dr.Deployment,
    data: pd.DataFrame,
    forecast_point=None,
    predictions_start_date=None,
    predictions_end_date=None,
    max_explanations: int | None = None,
    threshold_high: float | None = None,
    threshold_low: float | None = None,
):
    """
    Returns the predictions for a given deployment and data. If the deployment is a
    time series deployment, the forecast point, predictions start date and predictions
    end date need to be provided.

    This function uses the real time API

    :param deployment: dr.Deployment object
    :param data: data to get the predictions for
    :param forecast_point: forecast point for time series deployments
    :param predictions_start_date: start date for time series deployments
    :param predictions_end_date: end date for time series deployments
    :param max_explanations: maximum number of explanations to return
    :param threshold_high: high threshold for prediction explanations to be calculated
    :param threshold_low: low threshold for prediction explanations to be calculated
    :return: predictions as pandas dataframe
    """
    default_prediction_server = deployment.default_prediction_server
    if default_prediction_server is None:
        raise ValueError(
            "Deployment has no default prediction server associated with it."
        )
    url = (
        default_prediction_server["url"]
        + f"/predApi/v1.0/deployments/{deployment.id}/predictions"
    )
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        # pylint: disable=protected-access
        "Authorization": deployment._client.headers["Authorization"],
        "DataRobot-Key": default_prediction_server["datarobot-key"],
    }
    params = {
        "forecastPoint": forecast_point,
        "predictionsStartDate": predictions_start_date,
        "predictionsEndDate": predictions_end_date,
        # If explanations are required, uncomment the line below
        # 'maxExplanations': 3,
        # 'thresholdHigh': 0.5,
        # 'thresholdLow': 0.15,
        # Uncomment this for Prediction Warnings, if enabled for your deployment.
        # 'predictionWarningEnabled': 'true',
    }
    if max_explanations:
        params["maxExplanations"] = max_explanations
        params["thresholdHigh"] = threshold_high
        params["thresholdLow"] = threshold_low
    model = deployment.model
    if model is None:
        raise ValueError("Deployment has no model associated with it.")
    project = dr.Project.get(model["project_id"])
    if project.use_time_series:
        date_column = _get_datetime_partition_column(deployment)
        date_format = _get_date_format(deployment)

        if pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = data[date_column].dt.strftime(date_format)
        else:
            data[date_column] = pd.to_datetime(data[date_column]).dt.strftime(
                date_format
            )
    predictions_response = requests.post(
        url,
        data=data.to_json(orient="records"),
        headers=headers,
        params=params,
        timeout=3600,
    )
    preds = pd.DataFrame.from_records(predictions_response.json()["data"])
    if project.use_time_series:
        preds["forecastPoint"] = pd.to_datetime(preds["forecastPoint"]).dt.tz_convert(
            None
        )
        preds["timestamp"] = pd.to_datetime(preds["timestamp"]).dt.tz_convert(None)
        try:
            multiseries_id_columns = _get_multiseries_id_columns(deployment)
            preds = preds.rename(columns={"seriesId": multiseries_id_columns})
        except Exception:  # pylint: disable=broad-exception-caught
            log.info("No multiseries id columns found, assuming single series.")
    if project.positive_class:
        prediction_values_temp = pd.DataFrame.from_records(
            preds.predictionValues.values
        )
        for value in prediction_values_temp.columns:
            prediction_values_df = pd.json_normalize(prediction_values_temp[value])  # type: ignore
            if (prediction_values_df["label"] == project.positive_class).all():
                positive_preds = prediction_values_df["value"]
                positive_preds.name = "positiveClassPrediction"
                break
        preds = pd.concat(
            [
                preds.drop(columns=["predictionValues"]).reset_index(drop=True),
                positive_preds.reset_index(  # pyright: ignore[reportUnboundVariable]
                    drop=True
                ),
            ],
            axis=1,
        )
    if max_explanations:
        explanation_temp = pd.DataFrame.from_records(
            preds.predictionExplanations.values
        )
        explanation_dfs = []
        for explanation in explanation_temp.columns:
            explanation_df = pd.json_normalize(explanation_temp[explanation])  # type: ignore
            explanation_df.columns = [
                f"Explanation {int(explanation)+1} {c}" for c in explanation_df.columns
            ]
            explanation_dfs.append(explanation_df)
        explanation_dfs = pd.concat(explanation_dfs, axis=1)
        preds = pd.concat(
            [
                preds.drop(columns=["predictionExplanations"]).reset_index(drop=True),
                explanation_dfs.reset_index(drop=True),
            ],
            axis=1,
        )
    return preds


def calculate_stats(project: dr.Project, models=5, verbose=False):
    """
    Calculate stats for the first n_models models in the project
    :param self: project object
    :param n_models: int number of models to calculate stats for
    :return: None
    """

    def wait_for_jobs(jobs: list[dr.Job]):
        """
        Wait for a list of jobs to complete
        """
        jobs = [j for j in jobs if j]
        while True:
            # summarise the statuses by count of the jobs in the list
            job_status = Counter([j.status for j in jobs if j])
            # print the summary
            print(job_status)
            # if all jobs are complete, break out of the loop
            if not (job_status["queue"] > 0 or job_status["inprogress"] > 0):
                break
            else:
                time.sleep(5)
                for j in jobs:
                    j.refresh()

    if isinstance(models, int):
        if project.is_datetime_partitioned:
            models = project.get_datetime_models()[:models]
        else:
            models = project.get_models()[:models]

    def score_backtests(m: dr.DatetimeModel):
        try:
            return m.score_backtests()
        except Exception as e:  # pylint: disable=broad-exception-caught
            if verbose:
                print(e)
            return None

    def cross_validate(m: dr.Model):
        try:
            return m.cross_validate()
        except Exception as e:  # pylint: disable=broad-exception-caught
            if verbose:
                print(e)
            return None

    def request_feature_impact(m: dr.Model):
        try:
            return m.request_feature_impact()
        except Exception as e:  # pylint: disable=broad-exception-caught
            if verbose:
                print(e)
            return None

    def request_feature_effect(m: dr.Model, backtest: str | None = None):
        try:
            if backtest is None:
                return m.request_feature_effect()
            else:
                return m.request_feature_effect(backtest)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if verbose:
                print(e)
            return None

    def compute_datetime_trend_plots(m: dr.DatetimeModel, backtest, source):
        try:
            return m.compute_datetime_trend_plots(backtest, source)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if verbose:
                print(e)
            return None

    # calculate FI for all models
    jobs = []
    for m in models:
        jobs.append(request_feature_impact(m))
    wait_for_jobs(jobs)
    if project.id is None:
        raise ValueError("Project has no id associated with it.")
    if project.is_datetime_partitioned:
        dtp = dr.DatetimePartitioning.get(project.id)
        jobs = []
        jobs += [score_backtests(m) for m in models]  # type: ignore
        wait_for_jobs(jobs)

        jobs = []
        for m in models:
            for i in list(range(dtp.number_of_backtests)) + [  # type: ignore
                dr.enums.DATA_SUBSET.HOLDOUT  # type: ignore
            ]:
                jobs.append(request_feature_effect(m, str(i)))
                if project.use_time_series:
                    for source in ["training", "validation"]:
                        try:
                            jobs.append(
                                compute_datetime_trend_plots(
                                    m, backtest=str(i), source=source  # type: ignore
                                )
                            )
                        except:  # pylint: disable=bare-except
                            if verbose:
                                log.info(f"{m.id}, {i}, {source} failed")
    else:
        jobs = []
        for m in models:
            jobs.append(cross_validate(m))
            jobs.append(request_feature_effect(m))

    wait_for_jobs(jobs)


# dr.Project.get_featurelist_by_name = get_featurelist_by_name
# dr.Project.calculate_stats = calculate_stats
# dr.Deployment.predict = deployment_predict
