from typing import Any, Dict, Optional
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
import logging

from kedro.runner import SequentialRunner
from pluggy import PluginManager

log = logging.getLogger(__name__)


class RunOnlyMissingRunner(SequentialRunner):
    def __init__(self, client_args: Dict[str, Any] = {}, is_async: bool = False):
        """Instantiates the runner by creating a ``distributed.Client``.

        Args:
            client_args: Arguments to pass to the ``distributed.Client``
                constructor.
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
        """
        super().__init__(is_async=is_async)

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager = None,  # type: ignore
        session_id: str = None,  # type: ignore
    ) -> dict[str, Any]:

        # register any datasets that are not already registered
        for dataset in pipeline.data_sets():
            catalog.exists(dataset)
        free_outputs = pipeline.outputs() - set(catalog.list())
        log.info(f"Free outputs: {free_outputs}")
        missing = {ds for ds in catalog.list() if not catalog.exists(ds)}
        existing = set(catalog.list()) - missing
        log.info(f"Missing datasets: {missing}")
        log.info(
            f"Skipping datasets: {[e for e in existing if not e.startswith('params')]}"
        )
        to_build = free_outputs | missing
        log.info(f"Building datasets: {to_build}")
        to_rerun = pipeline.only_nodes_with_outputs(*to_build) + pipeline.from_inputs(
            *to_build
        )
        log.info(f"Running nodes: {to_rerun.nodes}")

        # We also need any missing datasets that are required to run the
        # `to_rerun` pipeline, including any chains of missing datasets.
        unregistered_ds = pipeline.data_sets() - set(catalog.list())
        log.info(f"Unregistered datasets: {unregistered_ds}")
        output_to_unregistered = pipeline.only_nodes_with_outputs(*unregistered_ds)
        log.info(f"Output to unregistered datasets: {output_to_unregistered.nodes}")
        input_from_unregistered = to_rerun.inputs() & unregistered_ds
        log.info(f"Input from unregistered datasets: {input_from_unregistered}")
        to_rerun += output_to_unregistered.to_outputs(*input_from_unregistered)
        log.info(f"Running nodes: {to_rerun.nodes}")

        return SequentialRunner().run(to_rerun, catalog, hook_manager)
