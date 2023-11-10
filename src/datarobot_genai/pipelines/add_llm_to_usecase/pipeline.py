"""
This is a boilerplate pipeline 'add_llm_to_usecase'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (
    create_byo_llm_blueprint,
    create_pre_baked_blueprint,
    create_vector_database_settings,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=create_vector_database_settings,
                inputs=["params:vector_database_settings"],
                outputs="vector_database_settings",
                name="create_vector_database_settings",
            ),
            node(
                func=create_byo_llm_blueprint,
                inputs=[
                    "playground",
                    "custom_model_validation",
                    "params:byo_llm_blueprint",
                    "params:system_prompt",
                    "vector_database",
                    "vector_database_settings",
                ],
                outputs="byo_llm_blueprint",
                name="add_byo_llm_to_playground",
            ),
            node(
                func=create_pre_baked_blueprint,
                inputs=[
                    "playground",
                    "params:pre_baked_llm_blueprint",
                    "params:system_prompt",
                    "vector_database",
                    "vector_database_settings",
                ],
                outputs="pre_baked_llm_blueprint",
                name="add_pre_baked_llm_to_playground",
            ),
        ],
        namespace="add_llm_to_usecase",
        inputs={
            "playground",
            "custom_model_validation",
            "vector_database",
        },
        outputs={"byo_llm_blueprint", "pre_baked_llm_blueprint"},
        parameters={
            "params:byo_llm_blueprint",
            "params:pre_baked_llm_blueprint",
            "params:system_prompt",
            "params:vector_database_settings",
        },
    )
