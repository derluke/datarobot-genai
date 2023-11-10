"""
This is a boilerplate pipeline 'create_usecase'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (
    create_usecase_object,
    create_playground,
    upload_vector_database,
    add_vector_database,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=create_usecase_object,
                inputs=None,
                outputs="use_case",
                name="create_usecase_object",
            ),
            node(
                func=create_playground,
                inputs=["use_case"],
                outputs="playground",
                name="create_playground",
            ),
            node(
                func=upload_vector_database,
                inputs=["vector_database_raw"],
                outputs="vector_database_dataset",
                name="upload_vector_database",
            ),
            node(
                func=add_vector_database,
                inputs=["vector_database_dataset", "use_case"],
                outputs="vector_database",
                name="add_vector_database",
            ),
        ],
        namespace="create_usecase",
        inputs={"vector_database_raw"},
        outputs={"playground", "vector_database"},
    )
