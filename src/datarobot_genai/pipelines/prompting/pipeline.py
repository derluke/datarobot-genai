"""
This is a boilerplate pipeline 'prompting'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import submit_prompt, save_and_submit_comparison_prompt


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=submit_prompt,
                inputs=[
                    "byo_llm_blueprint",
                    "params:prompt_text",
                ],
                outputs="byo_llm_response",
                name="byo_llm_submit_prompt",
            ),
            node(
                func=submit_prompt,
                inputs=[
                    "pre_baked_llm_blueprint",
                    "params:prompt_text",
                ],
                outputs="pre_baked_response",
                name="pre_baked_submit_prompt",
            ),
            node(
                func=save_and_submit_comparison_prompt,
                inputs=[
                    "byo_llm_blueprint",
                    "pre_baked_llm_blueprint",
                    "params:prompt_text",
                ],
                outputs="comparison_response",
                name="comparison_submit_prompt",
            ),
        ],
        namespace="prompting",
        inputs={
            "byo_llm_blueprint",
            "pre_baked_llm_blueprint",
        },
        outputs={"byo_llm_response", "pre_baked_response", "comparison_response"},
        parameters={"params:prompt_text"},
    )
