# type: ignore
# pyright: reportPrivateImportUsage=false
import datarobot as dr
from .utils import (
    calculate_stats,
    predict,
    get_featurelist_by_name,
    get_or_request_training_predictions,
    deployment_predict,
    upload_dataset,
)
from .fire import FIRE as FIRE
from .time_series_ion_cannon import TimeSeriesIonCannon as TimeSeriesIonCannon


def patch_datarobot():
    global dr
    dr.Model.predict = predict
    dr.Deployment.predict = deployment_predict
    dr.Model.get_or_request_training_predictions = get_or_request_training_predictions
    dr.Project.calculate_stats = calculate_stats
    dr.Project.get_featurelist_by_name = get_featurelist_by_name
    dr.Project.upload_dataset = upload_dataset
