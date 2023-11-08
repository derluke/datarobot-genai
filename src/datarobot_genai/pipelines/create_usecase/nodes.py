# pyright: reportPrivateImportUsage=false
from typing import Any
import datarobot as dr
import time
import logging

log = logging.getLogger(__name__)


def create_usecase_object() -> dr.UseCase:
    use_case = dr.UseCase.create(name="GenAI Cert 3 Buzok Use Case")
    return use_case


def create_playground(use_case: dr.UseCase, genai_api_root) -> dict:
    playground_payload = {
        "name": "BYO LLM Playground",
        "description": "This is a playground created using the REST API",
        "useCaseId": use_case.id,
    }

    client = dr.client.get_client()
    response = client.post(f"{genai_api_root}/playgrounds/", json=playground_payload)
    playground = response.json()
    return playground


def create_draft_blueprint(
    playground: dict, genai_api_root: str, llm_blueprint_params: dict[str, Any]
) -> dict:
    # Create a draft LLM blueprint.
    llm_blueprint_name = llm_blueprint_params.get("name", "Draft LLM Blueprint")
    llm_blueprint_llmid = llm_blueprint_params.get(
        "llmId", "azure-openai-gpt-3.5-turbo"
    )
    llm_blueprint_payload = {
        "name": llm_blueprint_name,
        "playgroundId": playground["id"],
        "llmId": llm_blueprint_llmid,
        # Uncomment this if you'd like to use a vector database
        # "vectorDatabaseId": "abcdef0123456789",
    }
    client = dr.client.get_client()
    response = client.post(
        f"{genai_api_root}/llmBlueprints/", json=llm_blueprint_payload
    )
    llm_blueprint = response.json()
    return llm_blueprint


def submit_prompt(llm_blueprint: dict, genai_api_root: str, prompt_text: str) -> str:
    # Submit a prompt
    prompt_payload = {
        "llmBlueprintId": llm_blueprint["id"],
        "text": prompt_text,
    }

    client = dr.client.get_client()

    response = client.post(f"{genai_api_root}/chatPrompts/", json=prompt_payload)

    prompt_status_tracking_url = response.headers["Location"]
    prompt = response.json()
    # prompt

    for _ in range(60):
        prompt_status_response = client.get(
            prompt_status_tracking_url, allow_redirects=False
        )
        if prompt_status_response.status_code == 303:
            log.info("Prompt completed successfully")
            break
        else:
            log.info("Waiting for the prompt to complete...")
            time.sleep(1)
    # Get prompt output.
    response = client.get(f"{genai_api_root}/chatPrompts/{prompt['id']}/")
    completed_prompt = response.json()
    log.info(f"Prompt output: {completed_prompt['resultText']}")

    return completed_prompt["resultText"]
