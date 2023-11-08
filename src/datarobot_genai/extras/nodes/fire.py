import logging

# pyright: reportPrivateImportUsage=false
import datarobot as dr

from ..utils import FIRE

log = logging.getLogger(__name__)


def run_fire(project: dr.Project, fire_parameters) -> dr.Project:

    fire = FIRE.aim(project.id)
    fire.main_feature_reduction(reduction_method="Rank Aggregation", **fire_parameters)

    return fire
