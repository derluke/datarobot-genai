from kedro.pipeline import Pipeline, node
from .nodes import (
    deploy_custom_llm,
    predict,
    validate_llm_deployment,
    add_to_playground,
)


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=deploy_custom_llm,
                inputs=["custom_py"],
                outputs="byo_llm_deployment",
                name="deploy_custom_llm",
            ),
            node(
                func=predict,
                inputs=["byo_llm_deployment", "params:prompt_text"],
                outputs="byo_llm_response",
                name="predict",
            ),
            node(
                func=validate_llm_deployment,
                inputs=["byo_llm_deployment", "params:genai_api_root"],
                outputs="validation_id",
                name="validate_llm_deployment",
            ),
            node(
                func=add_to_playground,
                inputs=[
                    "playground",
                    "validation_id",
                    "params:genai_api_root",
                    "params:blueprint_name",
                    "params:system_prompt",
                ],
                outputs=None,
                name="add_to_playground",
            ),
        ]
    )
