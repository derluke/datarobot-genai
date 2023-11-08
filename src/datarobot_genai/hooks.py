# implement hook to provdie datarobot token and client
import logging
import pdb
import sys
import traceback

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import datarobotx as drx
import urllib3.exceptions
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.framework.project import settings

log = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DataRobotHook:
    def __init__(self):
        pass

    @hook_impl
    def after_context_created(self, context: KedroContext):
        """
        Hook implementation to set up DataRobot client before the node is run.
        """
        conf_path = str(context.project_path / settings.CONF_SOURCE)
        conf_loader = OmegaConfigLoader(conf_source=conf_path, env="local")
        credentials = conf_loader["credentials"]["datarobot"]

        log.info(f"Initializing DataRobot client on endpoint {credentials['endpoint']}")
        _ = dr.Client(
            token=credentials["token"],
            endpoint=credentials["endpoint"],
            ssl_verify=False,
        )
        drx.Context(
            token=credentials["token"],
            endpoint=credentials["endpoint"],
            pred_server_id=dr.PredictionServer.list()[0].id,
        )

        cohere_credentials = conf_loader["credentials"]["cohere"]
        client = dr.client.get_client()
        try:
            res = client.post(
                "credentials",
                json={
                    "name": "cohere_api_token",
                    "description": "Added from kedro hook",
                    "credentialType": "api_token",
                    "apiToken": cohere_credentials["token"],
                },
            )
            if res.status_code == 201:
                log.info("Cohere credentials added to DataRobot")
        except:
            log.info("Cohere credentials already added to DataRobot")


class PDBPipelineDebugHook:
    """A hook class for creating a post mortem debugging with the PDB debugger
    whenever an error is triggered within a pipeline. The local scope from when the
    exception occured is available within this debugging session.
    """

    @hook_impl
    def on_pipeline_error(self):
        # We don't need the actual exception since it is within this stack frame
        _, _, traceback_object = sys.exc_info()

        #  Print the traceback information for debugging ease
        traceback.print_tb(traceback_object)

        # Drop you into a post mortem debugging session
        pdb.post_mortem(traceback_object)
