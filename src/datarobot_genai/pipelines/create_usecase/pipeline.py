"""
This is a boilerplate pipeline 'create_usecase'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    create_usecase_object,
    create_playground,
    create_draft_blueprint,
    submit_prompt,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(func=create_usecase_object, inputs=None, outputs="use_case"),
            node(
                func=create_playground,
                inputs=["use_case", "params:genai_api_root"],
                outputs="playground",
            ),
            node(
                func=create_draft_blueprint,
                inputs=[
                    "playground",
                    "params:genai_api_root",
                    "params:pre_baked_llm_blueprint",
                ],
                outputs="pre_baked_llm_blueprint",
            ),
            node(
                func=submit_prompt,
                inputs=[
                    "pre_baked_llm_blueprint",
                    "params:genai_api_root",
                    "params:prompt_text",
                ],
                outputs="pre_baked_response",
            ),
        ]
    )
