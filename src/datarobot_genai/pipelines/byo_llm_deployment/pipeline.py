from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (
    set_credentials,
    deploy_custom_llm,
    predict,
    validate_llm_deployment,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=set_credentials,
                inputs=["cohere_credentials"],
                outputs="credentials_set",
                name="set_credentials",
            ),
            node(
                func=deploy_custom_llm,
                inputs=["custom_py", "credentials_set"],
                outputs="byo_llm_deployment",
                name="deploy_custom_llm",
            ),
            node(
                func=predict,
                inputs=["byo_llm_deployment", "params:prompt_text"],
                outputs="byo_llm_deployment_response",
                name="predict",
            ),
            node(
                func=validate_llm_deployment,
                inputs=["byo_llm_deployment"],
                outputs="custom_model_validation",
                name="validate_llm_deployment",
            ),
        ],
        namespace="byo_llm_deployment",
        inputs={"cohere_credentials", "custom_py"},
        outputs="custom_model_validation",
    )
