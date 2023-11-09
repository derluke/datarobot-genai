# pyright: reportPrivateImportUsage=false
import logging
import tempfile
import time

import datarobot as dr
from datarobot._experimental.models.enums import (
    VectorDatabaseChunkingMethod,
    VectorDatabaseEmbeddingModel,
)
from datarobot._experimental.models.genai.playground import Playground
from datarobot._experimental.models.genai.vector_database import (
    ChunkingParameters,
    VectorDatabase,
)

log = logging.getLogger(__name__)


def create_usecase_object() -> dr.UseCase:
    use_case = dr.UseCase.create(name="GenAI Cert 3 Buzok Use Case")
    return use_case


def create_playground(use_case: dr.UseCase) -> Playground:
    playground = Playground.create(
        name="BYO LLM Playground",
        description="This is a playground created using the datarobot_early_access and kedro",
        use_case=use_case,
    )
    return playground


def upload_vector_database(vector_db_zip: bytes) -> dr.Dataset:
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = f"{tmpdirname}/vdb.zip"
        with open(file_name, "wb") as f:
            f.write(vector_db_zip)
        vector_database = dr.Dataset.create_from_file(file_name)
    return vector_database


def add_vector_database(
    vector_database_dataset: dr.Dataset, use_case: dr.UseCase
) -> VectorDatabase:

    chunking_parameters = ChunkingParameters(
        embedding_model=VectorDatabaseEmbeddingModel.JINA_EMBEDDING_T_EN_V1,
        chunking_method=VectorDatabaseChunkingMethod.RECURSIVE,
        chunk_size=512,
        chunk_overlap_percentage=10,
        separators=[],
    )
    vdb = VectorDatabase.create(
        vector_database_dataset.id, chunking_parameters, use_case
    )

    while vdb.execution_status != "COMPLETED":
        vdb = VectorDatabase.get(vdb.id)
        time.sleep(5)
    return vdb
