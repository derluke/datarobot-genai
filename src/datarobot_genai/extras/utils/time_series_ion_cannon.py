# type: ignore

# pyright: reportPrivateImportUsage=false
import datarobot as dr
from datarobot.errors import ClientError
from .fire import FIRE


class TimeSeriesIonCannon(dr.Project):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        all_models = self.get_datetime_models()
        self.supported_metrics = all_models[0].metrics.keys()
        self.spec = dr.DatetimePartitioning.get(self.id)

        if self.spec.windows_basis_unit == "ROW":
            self.training_duration = None
            self.training_row_count = [m for m in all_models if m.training_row_count][
                0
            ].training_row_count
        else:
            self.training_duration = [m for m in all_models if m.training_duration][
                0
            ].training_duration
            self.training_row_count = None

    sort_order = {
        "MASE": False,
        "FVE Poisson": True,
        "Theil's U": False,
        "RMSE": False,
        "FVE Gamma": True,
        "R Squared": True,
        "Gamma Deviance": False,
        "FVE Tweedie": True,
        "MAE": False,
        "SMAPE": True,
        "MAPE": True,
        "Gini Norm": True,
        "Tweedie Deviance": False,
        "Poisson Deviance": False,
        "RMSLE": False,
    }

    @classmethod
    def aim(cls, *args, **kwargs):
        return super().get(*args, **kwargs)

    def get_models_sorted(
        self, partition="validation", metric=None, model_type_filter=None
    ):
        if model_type_filter is None:
            model_type_filter = [""]
        if not metric:
            metric = self.metric
        if partition not in ["backtesting", "holdout", "validation"]:
            raise ValueError(
                f"Partition {partition} not in ['backtesting', 'holdout', 'validation']"
            )
        if partition == "holdout" and not self.holdout_unlocked:
            print("Holdout not unlocked!")
            return []
        if metric not in self.supported_metrics:
            raise ValueError(f"Metric {metric} not supported")
        reverse = self.sort_order.get(metric)
        return sorted(
            [
                m
                for m in self.get_datetime_models()
                if metric in m.metrics
                and m.metrics[metric][partition]
                and m.training_duration
                and any(f in m.model_type for f in model_type_filter)
            ],
            key=lambda m: m.metrics[metric][partition],
            reverse=reverse,
        )

    def delete_models(self):
        models = self.get_models_sorted()

        def delete(model: dr.Model):
            try:
                model.delete()
            except Exception:
                pass

        _ = [delete(m) for m in models[100:]]

    def calculate_backtests(self, models):
        def score_backtests(m):
            try:
                return m.score_backtests()
            except ClientError:
                return None

        jobs = [score_backtests(m) for m in models]
        _ = [job.wait_for_completion() for job in jobs if job]

    def identify_best_featurelist(self):
        best_models = self.get_models_sorted("backtesting")
        if not best_models:
            print("calculate some backtests")
        featurelists = [
            m.featurelist for m in best_models[:20] if "Blender" not in m.model_type
        ]
        reduced_fl = [fl for fl in featurelists if "DR Reduced" in fl.name]
        seen = set()
        seen_add = seen.add
        unique_reduced_featurelists = [
            x for x in reduced_fl if not (x.id in seen or seen_add(x.id))
        ]
        other_fl = [fl for fl in featurelists if "DR Reduced" not in fl.name]
        seen = set()
        seen_add = seen.add
        unique_featurelists = [
            x for x in other_fl if not (x.id in seen or seen_add(x.id))
        ]
        return unique_reduced_featurelists + unique_featurelists[:2]

    def run_blueprints(
        self,
        featurelist,
        training_duration=None,
        model_type_filter=None,
        blueprint_ids=None,
    ):
        if not blueprint_ids:
            blueprint_ids = self.get_blueprints()
        if model_type_filter is None:
            model_type_filter = ["Mean", "Eureqa", "VARMAX", "seasonal", "Keras"]
        if model_type_filter:
            blueprint_ids = [
                bp.id
                for bp in blueprint_ids
                if all(f not in bp.model_type for f in model_type_filter)
            ]
        if not training_duration:
            training_duration = (
                self.training_row_count
                if self.spec.windows_basis_unit == "ROW"
                else self.training_duration
            )

        def train_blueprint(blueprint_id, fl):
            try:
                if self.spec.windows_basis_unit == "ROW":
                    return self.train_datetime(
                        blueprint_id, fl.id, training_row_count=training_duration
                    )
                else:
                    return self.train_datetime(
                        blueprint_id, fl.id, training_duration=training_duration
                    )
            except ClientError:
                # print(e)
                return None

        def wait_for_job(job: dr.Job):
            try:
                job.wait_for_completion(max_wait=10000)
            except Exception:
                pass

        jobs = [train_blueprint(bp, featurelist) for bp in blueprint_ids]
        _ = [wait_for_job(job) for job in jobs if job]

    def run_blenders(self):
        def blend(model_ids, blender_method):
            try:
                return self.blend(model_ids, blender_method)
            except ClientError as e:
                print(e)
                return None

        best_models = self.get_models_sorted("backtesting")
        best_models = [m for m in best_models if "Blender" not in m.model_type]
        jobs = []
        for n in [3, 5, 7]:
            for blender_method in [
                dr.enums.BLENDER_METHOD.FORECAST_DISTANCE_AVG,
                dr.enums.BLENDER_METHOD.AVERAGE,
                dr.enums.BLENDER_METHOD.FORECAST_DISTANCE_ENET,
            ]:
                jobs.append(
                    blend(
                        [m.id for m in best_models[:n]], blender_method=blender_method
                    )
                )
        blender_models = [j.get_result_when_complete() for j in jobs if j]
        blender_models = [dr.DatetimeModel.get(self.id, bm.id) for bm in blender_models]
        return blender_models

    def shoot(self):
        self.calculate_backtests(self.get_models_sorted("validation")[:20])
        fls = self.identify_best_featurelist()
        for fl in fls:
            self.run_blueprints(fl)
        self.calculate_backtests(self.get_models_sorted("validation")[:20])
        self.run_blenders()
        self.calculate_backtests(self.get_models_sorted("validation")[:20])

    def shoot_with_fire(self):
        # init
        print("Identifying best Feature lists...")
        featurelists = self.identify_best_featurelist()
        print("found: " + ", ".join([fl.name for fl in featurelists]))

        print("Finding good seed Blueprints:")
        good_blueprints = self.find_good_blueprints()
        print(
            ", ".join(
                [dr.Blueprint.get(self.id, bp).model_type for bp in good_blueprints]
            )
        )

        print("running seed blueprints for feature list")
        for fl in featurelists:
            print(fl.name)
            self.run_blueprints(
                fl, model_type_filter=False, blueprint_ids=good_blueprints
            )
        fire = FIRE.aim(self.id)
        # first reduction
        print("running FIRE once on")
        for fl in featurelists:
            print(fl.name)
            fire.main_feature_reduction(
                reduction_method="Rank Aggregation",
                start_featurelist_name=fl.name,
                max_iterations=1,
            )
        self.delete_models()
        # ion cannon
        print("shoot the ion cannon first time")
        self.shoot()
        self.delete_models()
        # second reduction
        print("Identifying best feature lists now...")
        featurelists = self.identify_best_featurelist()

        print("found: " + ", ".join([fl.name for fl in featurelists]))
        print("running FIRE again on")
        for fl in featurelists:
            print(fl.name)
            fire.main_feature_reduction(
                reduction_method="Rank Aggregation",
                start_featurelist_name=fl.name,
                max_iterations=3,
            )
        # ion cannon
        print("Shoot the ion cannon again")
        self.shoot()

    def find_good_blueprints(self):
        models = [
            m
            for m in self.get_models_sorted("validation")
            if "Baseline" not in m.model_type and "Blender" not in m.model_type
        ][:15]
        return set([m.blueprint_id for m in models])
