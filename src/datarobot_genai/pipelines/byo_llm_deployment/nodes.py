import os
import tempfile
import logging


# pyright: reportPrivateImportUsage=false
import datarobot as dr
import datarobotx as drx
import pandas as pd
from datarobot._experimental.models.genai.custom_model_llm_validation import (
    CustomModelLLMValidation,
)
from datarobot._experimental.models.genai.custom_model_validation import (
    CustomModelValidation,
)


log = logging.getLogger(__name__)


def set_credentials(cohere_credentials: dict) -> bool:
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
    except Exception:  # pylint: disable=broad-exception-caught
        log.info("Cohere credentials already added to DataRobot")

    return True


def deploy_custom_llm(custom_py: str, credentials_set: bool) -> drx.Deployment:
    if not credentials_set:
        raise RuntimeError("Cohere credentials not set")
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(f"{tmpdirname}/custom.py", "wt", encoding="utf-8") as f:
            f.write(custom_py)
        os.chdir(tmpdirname)

        deployment = drx.deploy(
            "custom.py",
            target_type="TextGeneration",
            target="resultText",
            environment_id="64c964448dd3f0c07f47d040",
            runtime_parameters=["cohere_api_token"],
            extra_requirements=["cohere==4.27", "datarobot-drum==1.10.10"],
        )
        os.chdir(cwd)

    return deployment


def predict(deployment: drx.Deployment, prompt: str) -> str:

    inp = {
        "context": [""],
        "promptText": [prompt],
    }
    prediction = deployment.predict(pd.DataFrame(inp))
    return prediction["prediction"][0]


def validate_llm_deployment(deployment: drx.Deployment) -> CustomModelValidation:

    custom_model_llm_validation = CustomModelLLMValidation.create(
        prompt_column_name="promptText",
        target_column_name="resultText",
        deployment_id=deployment.dr_deployment.id,
        wait_for_completion=True,
    )
    assert custom_model_llm_validation.validation_status == "PASSED"

    return custom_model_llm_validation
