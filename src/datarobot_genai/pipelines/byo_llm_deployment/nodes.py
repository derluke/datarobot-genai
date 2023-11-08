import os
import tempfile
import time
import logging


# pyright: reportPrivateImportUsage=false
import datarobot as dr
import datarobotx as drx
import pandas as pd

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
    except Exception:  # pylint: disable=bare-except
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


def validate_llm_deployment(deployment: drx.Deployment, genai_api_root: str) -> dict:
    validation_payload = {
        "promptColumnName": "promptText",
        "targetColumnName": "resultText",
        "deploymentId": deployment.dr_deployment.id,
    }

    client = dr.client.get_client()
    response = client.post(
        f"{genai_api_root}/customModelLLMValidations/", json=validation_payload
    )
    status_tracking_url = response.headers.get("Location")
    if not status_tracking_url:
        raise RuntimeError("Failed to get status tracking URL")

    print(f"Response: {response.status_code}")
    print(f"Status tracking URL: {status_tracking_url}")
    validation_id = None

    # Waiting timeout = 60 seconds
    for _ in range(60):
        status_response = client.get(status_tracking_url, allow_redirects=False)

        if status_response.status_code == 303:
            validation_id = response.json()["id"]
            validation = client.get(
                f"{genai_api_root}/customModelLLMValidations/{validation_id}/"
            ).json()
            validation_status = validation["validationStatus"]

            if validation_status == "PASSED":
                print(f"Successful validation ID: {validation['id']}")
                break
            else:
                raise RuntimeError(
                    f"Expected a successful validation, got: {validation_status}"
                )

        time.sleep(1)

    if not validation_id:
        raise RuntimeError("Timed out waiting for custom model validation to succeed")

    return {"validation_id": validation_id}


def add_to_playground(
    playground: dict,
    validation_id: dict,
    genai_api_root: str,
    blueprint_name: str,
    system_prompt: str,
) -> None:
    llm_blueprint_payload = {
        "name": blueprint_name,
        "playgroundId": playground["id"],
        "llmId": "custom-model",
        # Uncomment this if you'd like to use a vector database
        # "vectorDatabaseId": "abcdef0123456789",
        "llmSettings": {
            "systemPrompt": system_prompt,
            "validationId": validation_id["validation_id"],
        },
    }
    client = dr.client.get_client()
    response = client.post(
        f"{genai_api_root}/llmBlueprints/", json=llm_blueprint_payload
    )
    llm_blueprint = response.json()
    log.info(f"LLM Blueprint: {llm_blueprint}")
