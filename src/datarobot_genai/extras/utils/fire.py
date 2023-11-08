# type: ignore

import logging
from typing import List

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FIRE(dr.Project):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def aim(cls, *args, **kwargs):
        return super().get(*args, **kwargs)

    def main_feature_reduction(
        self,
        reduction_method="Simple",
        start_model_id: str | None = None,
        start_featurelist_name=None,
        restart_autopilot=False,
        lifes=3,
        max_iterations=5,
        top_n_models=5,
        remove_redundant=True,
        partition="validation",
        main_scoring_metric=None,
        initial_impact_reduction_ratio=0.95,
        remove_negative_only_first=False,
        best_model_search_params=None,
    ):
        """
        Main function. Meant to get the optimal shortest feature list.
        Run before choosing (retraining/blending/freezing) the final bluelog.info.
        Currently supports and tested on Binary, Regression, Multiclass projects. Testing for OTV is
        TO DO. Support for TS is TO DO
        Has 3 methods of reducing feature list:
        - 'Simple':
        - 'DR Reduced':replicates 'DR Reduced Feature List' logic. Set to initial_impact_reduction_ratio=0.95 to
        get the exact match
        - 'Rank Aggregation': median aggregation of ranks of features by their impact over top N models

        Example usage:
        >> import datarobot as dr
        >> from Automatic_Feature_Reduction import *

        >> dr.Client(config_path='PATH_TO_DR_CONFIG/drconfig.yaml')
        TIP: set best_model_search_params = {'sample_pct__lte': 65} to avoid using models trained on a higher sample
        size than 3rd stage of autopilot

        >> main_feature_reduction('INSERT_PROJECT_ID',
                                reduction_method='Simple',
                                start_model_id=None,
                                start_featurelist_name=None,
                                restart_autopilot=False,
                                lifes=3,
                                top_n_models=5,
                                remove_redundant=True,
                                partition='crossValidation',
                                main_scoring_metric=None,
                                initial_impact_reduction_ratio=0.95,
                                remove_negative_only_first=False,
                                best_model_search_params=None)

        Parameters:
        -----------
        project_id: str, id of DR project,
        reduction_method: str, can be 'Simple', 'DR Reduced' or 'Rank Aggregation'.
            Default 'Simple'
        start_model_id: str, id of DR model to start iterating from. for
            'Rank Aggregation' keep default. Default None
        start_featurelist_name: str, name of feature list to start iterating from. Default None
        restart_autopilot: boolean, if True will restart current setup of autopilot
            after each iteration. Always True
        for 'Rank Aggregation' method. Default False
        lifes: int, stopping criteria, if no best model produced after lifes
            iterations, stop feature reduction.
        Default 3
        top_n_models: int, only for 'Rank Aggregation' method, get top N best
            models on the leaderboard. Default 5
        remove_redundant: boolean, if every iteration removes redundant features.
            Default True
        partition: str, whether to use 'validation' or 'crossValidation' partition
            to get the best model on. Default
        'validation'
        main_scoring_metric: str, DR metric to check performance against, Default None
        initial_impact_reduction_ratio: float, ratio of total feature impact that new
            feature list will contain.
        Default 0.95
        remove_negative_only_first: boolean, if True will drop only those features
            with 0 or negative FI per iteration.
        Default False
        best_model_search_params: dict, dictonary of parameters to search the best
            model. See official DR python
        api docs.
        Default None

        Returns:
        ----------
        dr.Model object of the best model on the leaderboard
        """
        project = self

        # TO DO: Implement support for TS, test OTV
        # if not project.is_datetime_partitioned:
        #     raise NotImplementedError("This is for TS only")

        assert reduction_method in [
            "Simple",
            "DR Reduced",
            "Rank Aggregation",
        ], "Argument error: reduction_method must be 'Simple', 'DR Reduced' or 'Rank Aggregation'"

        assert self.id is not None, "Project id is not found"

        if project.partition is None:
            raise NotImplementedError("No partioning found")
        project_partition = project.partition["cv_method"]

        ratio = initial_impact_reduction_ratio
        model_search_params = best_model_search_params

        runs = 0
        # main function loop
        while lifes >= 0 and max_iterations > 0:
            max_iterations -= 1
            if reduction_method in ["Simple", "DR Reduced"]:
                # make it a separate function

                # #########################
                # #### GET BEST MODEL #####
                # #########################
                if start_model_id and runs == 0:
                    best_model = dr.DatetimeModel.get(self.id, start_model_id)
                else:
                    best_model = self.get_best_models(
                        metric=main_scoring_metric,
                        by_partition=partition,
                        start_featurelist_name=start_featurelist_name,
                        model_search_params=model_search_params,
                    ).values[0]

                # #############################
                # #### GET FEATURE IMPACT #####
                # #############################
                # wait for 10 mins
                feature_impacts = best_model.get_or_request_feature_impact(600)

                assert isinstance(feature_impacts, List), (
                    "Check if the project is partitioned by datetime and "
                    "if the best model is datetime model"
                )

                # make sure features are sorted by feature impact
                feature_impacts.sort(key=lambda x: x["impactNormalized"], reverse=True)  # type: ignore
                feature_impacts = pd.DataFrame(feature_impacts)

                # ##############################################
                # #### GET NEW REDUCED FEATURE LIST IMPACT #####
                # ##############################################

                new_feature_list = self.get_new_feature_list(
                    reduction_method=reduction_method,
                    feature_impacts=feature_impacts,
                    remove_redundant=remove_redundant,
                    remove_negative_only_first=remove_negative_only_first,
                    ratio=ratio,
                )

                len_new_feature_list = len(new_feature_list)
                # if new feature list length is the same as before -> reduce more by reducing the ratio
                if feature_impacts.shape[0] == len_new_feature_list:
                    log.info(
                        "new feature list is the same as previous one... Use new ratio=ratio^2"
                    )
                    ratio *= ratio
                    new_feature_list = self.get_new_feature_list(
                        reduction_method=reduction_method,
                        feature_impacts=feature_impacts,
                        remove_redundant=remove_redundant,
                        remove_negative_only_first=remove_negative_only_first,
                        ratio=ratio,
                    )

                assert len(feature_impacts) > len_new_feature_list, (
                    "Check number of features with negative feature impact OR new feature list "
                    "is the same as previous one"
                )

                if reduction_method == "Simple":
                    new_featurelist_name = (
                        f"M{best_model.model_number}_top{len_new_feature_list}"
                        + f"_r{int(ratio * 100)}_iter{runs}"
                    )
                else:
                    new_featurelist_name = (
                        f"DR Reduced M{best_model.model_number}_r{int(ratio * 100)}"
                    )

                try:
                    new_fl = project.create_modeling_featurelist(
                        new_featurelist_name, new_feature_list
                    )

                except dr.errors.ClientError as e:
                    log.info(
                        f"Feature list named {new_featurelist_name} already exists"
                    )
                    log.info(e.json["message"])
                    if (
                        e.json["message"]
                        == f"Feature list named {new_featurelist_name} already exists"
                    ):
                        new_fl = [
                            fl
                            for fl in project.get_modeling_featurelists()
                            if fl.name == new_featurelist_name
                        ][0]
                    else:
                        log.info(e)
                        log.info("Apply ratio decay and try again")
                        ratio *= ratio
                        raise (e)

                log.info(
                    f"Old feature list size: {feature_impacts.shape[0]}, new feature list size: {len_new_feature_list}"
                )

                if restart_autopilot:
                    # ####################################################
                    # #### RESTART AUTOPILOT ON THE NEW FEATURE LIST #####
                    # ####################################################
                    project.start_autopilot(
                        featurelist_id=new_fl.id, mode=dr.enums.AUTOPILOT_MODE.QUICK
                    )
                    log.info("New autopilot is kicked-off")
                    project.wait_for_autopilot()

                else:
                    # ###############################################
                    # #### TRAIN MODEL ON THE NEW FEATURE LIST ######
                    # ###############################################
                    # Train new datetime model
                    if project_partition != "datetime":
                        # log.info(project_partition)
                        # raise NotImplementedError(
                        #     "Datetime partitioned projects support is to be implemented"
                        # )
                        model_job_id = best_model.train(
                            sample_pct=best_model.sample_pct,
                            featurelist_id=new_fl.id,
                        )
                        model_job = dr.ModelJob.get(project.id, model_job_id)
                        model_job.get_result_when_complete()
                    else:
                        try:
                            model_job = best_model.train_datetime(
                                featurelist_id=new_fl.id,
                                training_duration=best_model.training_duration,
                            )
                            model_job.get_result_when_complete()
                        except dr.errors.ClientError as e:
                            if e.json["errorName"] == "JobAlreadyAdded":
                                pass
                            else:
                                log.info(e, "\nThis project type is not supported yet")

                        #                     new_model = dr.DatetimeModel.get(project.id,
                        # model_job.get_result_when_complete().id)

                        log.info("New model training is completed....")

            # method Rank Aggregation
            else:
                #             if runs > 0:
                #                 start_featurelist_name=None
                try:
                    (
                        best_model,
                        new_featurelist_name,
                    ) = self.rank_based_feature_reduction_within_project(
                        n_models=top_n_models,
                        metric=main_scoring_metric,
                        by_partition=partition,
                        feature_list_name=start_featurelist_name,
                        ratio=ratio,
                        model_search_params=best_model_search_params,
                    )
                    start_featurelist_name = new_featurelist_name
                except dr.errors.ClientError as e:
                    log.info(e, "\nTrying again")
                    ratio *= ratio
                    continue

            # #############################
            # #### GET NEW BEST MODEL #####
            # #############################

            new_best_model = self.get_best_models(
                metric=main_scoring_metric,
                by_partition=partition,
                start_featurelist_name=new_featurelist_name,
                model_search_params=model_search_params,
            ).values[0]

            # ################################
            # #### PROSESS STOP CRITERIA #####
            # ################################

            if best_model.id == new_best_model.id:
                # repeat one more time
                lifes -= 1
                ratio *= ratio
                log.info(
                    "new model is worse.\nRepeat again. Decay Ratio for Simple and DR reduced methods"
                )

            if lifes < 0:
                log.info(
                    "new model is worse.\n AUTOMATIC FEATURE SELECTION PROCESS HAS BEEN STOPPED"
                )
                return new_best_model

            runs += 1
            log.info(f"Run {runs} completed")

        return new_best_model

    def get_best_models(
        self,
        metric=None,
        by_partition="validation",
        start_featurelist_name=None,
        model_search_params=None,
    ):
        """
        Gets pd.Series of DR model objects sorted by performance. Excludes blenders,
        frozen models and on DR Reduced FL

        Parameters:
        -----------
        project: DR project object
        metric: str, metric to use for sorting models on lb, if None, default project
            metric will be used. Default None
        by_partiton: boolean, whether to use 'validation' or 'crossValidation' partitioning.
            Default 'validation'
        start_featurelist_name: str, initial featurelist name to get models on. Default None
        model_search_params: dict to pass model search params. Default None

        Returns:
        -----------
        pd.Series of dr.Model objects, not blender, not frozen and not on DR Reduced Feature List
        """

        desc_metric_list = [
            "AUC",
            "Gini Norm",
            "Kolmogorov-Smirnov",
            "Max MCC",
            "Rate@Top5%",
            "Rate@Top10%",
            "Rate@TopTenth%",
            "R Squared",
            "FVE Gamma",
            "FVE Poisson",
            "FVE Tweedie",
            "Accuracy",
            "Balanced Accuracy",
            "FVE Multinomial",
        ]

        if not metric:
            metric = self.metric
            if "Weighted" in metric:
                desc_metric_list = ["Weighted " + metric for metric in desc_metric_list]

        asc_flag = False if metric in desc_metric_list else True
        all_models = self.get_models(
            with_metric=metric, search_params=model_search_params
        )
        if self.is_datetime_partitioned:
            all_datetime_models = {m.id: m for m in self.get_datetime_models()}

        models_df = pd.DataFrame(
            [
                [
                    model.metrics[metric]["crossValidation"],
                    model.metrics[metric]["validation"],
                    model.model_category,
                    model.is_frozen,
                    model.featurelist_name,
                    all_datetime_models[model.id]
                    if self.is_datetime_partitioned
                    else model,
                ]
                for model in all_models
            ],
            columns=[
                "crossValidation",
                "validation",
                "category",
                "is_frozen",
                "featurelist_name",
                "model",
            ],
        ).sort_values([by_partition], ascending=asc_flag, na_position="last")
        # log.info(models_df.head(5))
        if start_featurelist_name:
            return_list = models_df.loc[
                (
                    (models_df.category == "model")
                    & (~models_df.is_frozen)
                    & (models_df.featurelist_name == start_featurelist_name)
                ),
                "model",
            ]
            # log.info(return_list)
            return return_list
        else:
            return models_df.loc[
                (
                    (models_df.category == "model")
                    & (~models_df.is_frozen)
                    & (
                        ~models_df.featurelist_name.str.contains(
                            "DR Reduced Features M"
                        )
                    )
                ),
                "model",
            ]

    def get_simple_feature_list(
        self,
        feature_impacts,
        remove_redundant=True,
        remove_negative_only_first=False,
        ratio=0.99,
    ):
        """
        Simplified version of DR Reduced Feature List; may not produce the exact feature list
        as DR Reduced FL;
        Uses only cumulative feature impact strategy

        Parameters:
        -----------
        feature_impacts: pd.DataFrame of feature impact
        remove_redundant: boolean, default is True - removes redundant first before
        reducing based on cumulative impact
        remove_negative_only_first: boolean, default is False, set True if on every
        iteration to drop features
        with negative impact only
        ratio: float, ratio to be multiplied by total feature impact and choose features that
        possess that much impact

        Returns:
        -----------
        list of str, new feature list to retrain on
        """

        if remove_redundant:
            feature_impacts = feature_impacts.loc[
                feature_impacts["redundantWith"].isna()
            ]

        if remove_negative_only_first:
            new_feature_list = feature_impacts[feature_impacts.impactUnnormalized > 0][
                "featureName"
            ].values.tolist()
            return list(set(new_feature_list))

        # calculate cumulative feature impact and take first features that
        # possess ratio*100 percents of total impact
        feature_impacts["impactCumulative"] = feature_impacts[
            "impactUnnormalized"
        ].cumsum()
        total_impact = feature_impacts["impactCumulative"].max() * ratio

        new_feature_list = feature_impacts[
            feature_impacts.impactCumulative <= total_impact
        ]["featureName"].values.tolist()

        return list(set(new_feature_list))

    def get_dr_reduced_fl(
        self,
        impact,
        min_impact=0.95,
        constant_features=100,
        min_features=25,
        feature_ratio=0.5,
    ):
        """
        CANNOT BE SHARED WITH CUSTOMERS AS IS!

        Returns a list of reduced features; replicates DR Reduced Feature List

        Parameters:
        -----------
        impact: pd.DataFrame of 3 variables, output of pd.DataFrame(model.get_or_request_feature_impact())
        Notes: impact is supposed to be reverse sorted by values, and it is assumed as such
        min_impact: float, default is 0.95, ratio of total impact that is preserved
        constant_features: int, default is 100, additional safeguard in case of too many features, heuristics
        min_features: int, default is 25, additional safeguard in the heuristics
        feature_ration: float, default is 0.5, additional safeguard in the heuristics

        Returns:
        -----------
        new_feature_list: list of feature names to be used for FL creation
        """

        # convert pd.Dataframe to list of 3-tupples of feature name, impact normalized, impact unnormalized
        impact = list(impact.to_records(index=False))

        accumulated_impact = 0.0
        num_features_acc = 0
        # in some cases, the 2nd element of the 3-tuple is not normalized
        # it is safer to directly normalize the other 1st element instead, and use that always
        normalized_impact = np.array([np.float(t[2]) for t in impact])
        normalized_impact /= np.sum(normalized_impact)

        # collect the minimum number of features to meet accumulated impact
        while accumulated_impact <= min_impact and num_features_acc < len(
            normalized_impact
        ):
            accumulated_impact += normalized_impact[num_features_acc]
            num_features_acc += 1

        # number of features used should be minimum of the following numbers
        #     log.info(constant_features, max(min_features, feature_ratio * len(impact)), num_features_acc)

        feat_cnt = min(
            constant_features,
            max(min_features, feature_ratio * len(impact)),
            num_features_acc,
        )

        # index 1 should correspond to name of the feature within tuple
        new_feature_list = [impact[i][1] for i in range(int(feat_cnt))]

        return new_feature_list

    def get_new_feature_list(
        self,
        reduction_method,
        feature_impacts,
        remove_redundant=True,
        remove_negative_only_first=False,
        ratio=0.95,
    ):
        assert reduction_method in ["Simple", "DR Reduced"], "Wrong method"

        if reduction_method == "Simple":
            cnt_negative = feature_impacts[
                feature_impacts.impactUnnormalized <= 0
            ].shape[0]
            if cnt_negative == 0:
                remove_negative_only_first = False

            new_feature_list = self.get_simple_feature_list(
                feature_impacts,
                remove_redundant=remove_redundant,
                remove_negative_only_first=remove_negative_only_first,
                ratio=ratio,
            )

        else:
            new_feature_list = self.get_dr_reduced_fl(feature_impacts, min_impact=ratio)

        return new_feature_list

    # Aggregate Feature Impact Across the Top Performing Models
    # based on:
    # https://github.com/datarobot/data-science-scripts/blob/master/tim-whittaker/mrmg-experiment/PNC%20MRMG%20Experiment.ipynb

    def rank_based_feature_reduction_within_project(
        self,
        n_models=5,
        metric=None,
        by_partition="validation",
        feature_list_name=None,
        ratio=0.95,
        model_search_params=None,
    ):
        """ """

        models = self.get_best_models(
            metric=metric,
            by_partition=by_partition,
            start_featurelist_name=feature_list_name,
            model_search_params=model_search_params,
        )

        models = models.values[:n_models]

        all_impact = pd.DataFrame()

        advanced_options = self.advanced_options
        log.info(
            f"Calculating Feature Impact for {n_models} models on feature list {feature_list_name}"
        )
        if isinstance(advanced_options, dr.AdvancedOptions):
            shap_only_mode = advanced_options.shap_only_mode
        else:
            shap_only_mode = advanced_options["shap_only_mode"]

        if shap_only_mode:
            for model in models:
                try:
                    dr.ShapImpact.create(self.id, model.id)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        else:
            for model in models:
                try:
                    model.request_feature_impact()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        for model in models:
            if shap_only_mode:
                shap_job = dr.ShapImpact.create(self.id, model.id)
                shap_impact = shap_job.get_result_when_complete()

                df = pd.DataFrame.from_records(shap_impact.shap_impacts)
                df = df.rename(
                    columns={
                        "feature_name": "featureName",
                        "impact_normalized": "impactNormalized",
                        "impact_unnormalized": "impactUnnormalized",
                    }
                )
            else:
                # This can take a minute (for each)
                feature_impact = model.get_or_request_feature_impact(max_wait=600)

                # Ready to be converted to DF
                df = pd.DataFrame.from_records(feature_impact)
            # Track model name and ID for bookkeeping purposes
            df["model_type"] = model.model_type
            df["model_id"] = model.id
            # By sorting and re-indexing, the new index becomes our 'ranking'
            df = df.sort_values(by="impactUnnormalized", ascending=False)
            df = df.reset_index(drop=True)
            df["rank"] = df.index.values

            # Add to our master list of all models' feature ranks
            all_impact = pd.concat([all_impact, df], ignore_index=True)

        # use Simple fl function to get number of features to use
        # based on sum of impact
        # log.info(all_impact.head())
        all_impact_agg = (
            all_impact.groupby("featureName")[
                ["impactNormalized", "impactUnnormalized"]
            ]
            .sum()
            .sort_values("impactUnnormalized", ascending=False)
            .reset_index()
        )

        tmp_fl = self.get_simple_feature_list(
            all_impact_agg,
            remove_redundant=False,
            remove_negative_only_first=False,
            ratio=ratio,
        )
        n_feats = len(tmp_fl)

        top_ranked_feats = list(
            all_impact[["featureName", "rank"]]
            .groupby("featureName")
            .median()
            .sort_values("rank")
            .head(n_feats)
            .index.values
        )

        # ## Create new featurelist and run autopilot
        try:
            featurelist = self.create_modeling_featurelist(
                f"Reduced FL by Median Rank, top{n_feats}", top_ranked_feats
            )
            featurelist_id = featurelist.id

        except dr.errors.ClientError as e:
            if e.json["message"] == "Invalid field data":
                return models[0], feature_list_name
            else:
                featurelist = [
                    fl
                    for fl in self.get_modeling_featurelists()
                    if fl.name == f"Reduced FL by Median Rank, top{n_feats}"
                ][0]
                featurelist_id = featurelist.id
                if (
                    len(
                        [
                            m
                            for m in self.get_models()
                            if m.featurelist_id == featurelist_id
                        ]
                    )
                    > 3
                ):
                    log.info("skipping autopilot for " + featurelist.name)
                    return models[0], featurelist.name
        log.info(f"New feature list {featurelist.name} created")
        self.start_autopilot(
            featurelist_id=featurelist_id,
            blend_best_models=False,
            prepare_model_for_deployment=False,
            mode=dr.enums.AUTOPILOT_MODE.QUICK,
        )
        log.info("New autopilot is kicked-off")
        self.wait_for_autopilot()
        # return the previous best model to process the stop criteria
        return models[0], featurelist.name


# usage. Assuming autopilot ran and we have a project_id
# cannon = TimeSeriesIonCannon.aim(project_id)
# cannon.shoot_with_fire()
