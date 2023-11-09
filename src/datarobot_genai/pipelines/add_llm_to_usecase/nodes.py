import logging
from typing import Any

from datarobot._experimental.models.genai.llm_blueprint import LLMBlueprint
from datarobot._experimental.models.genai.llm import LLMDefinition
from datarobot._experimental.models.genai.playground import Playground
from datarobot._experimental.models.genai.vector_database import VectorDatabase
from datarobot._experimental.models.genai.llm_blueprint import VectorDatabaseSettings
from datarobot._experimental.models.genai.custom_model_validation import (
    CustomModelValidation,
)

log = logging.getLogger(__name__)


def create_vector_database_settings(
    vector_database_params: dict[str, Any]
) -> VectorDatabaseSettings:
    vector_database_settings = VectorDatabaseSettings(**vector_database_params)
    return vector_database_settings


def create_pre_baked_blueprint(
    playground: Playground,
    blueprint_params: dict[str, Any],
    system_prompt: str,
    vdb: VectorDatabase | None = None,
    vdb_settings: VectorDatabaseSettings | None = None,
) -> LLMBlueprint:
    # Create a draft LLM blueprint.

    llms = LLMDefinition.list(as_dict=False)
    llm = [llm for llm in llms if llm.id == blueprint_params["id"]][0]

    llm_settings = {
        "system_prompt": system_prompt,
    }

    llm_blueprint = LLMBlueprint.create(
        name=blueprint_params["name"],
        playground=playground,
        llm=llm,
        llm_settings=llm_settings,  # type: ignore
        vector_database=vdb,
        vector_database_settings=vdb_settings,
    )
    llm_blueprint = LLMBlueprint.update(llm_blueprint, is_saved=True)

    return llm_blueprint


def create_byo_llm_blueprint(
    playground: Playground,
    validation: CustomModelValidation,
    blueprint_params: dict[str, Any],
    system_prompt: str,
    vdb: VectorDatabase | None = None,
    vdb_settings: VectorDatabaseSettings | None = None,
) -> LLMBlueprint:

    custom_llm = [
        llm_def
        for llm_def in LLMDefinition.list(as_dict=False)
        if llm_def.name == "Custom Model"
    ][0]

    llm_blueprint = LLMBlueprint.create(
        name=blueprint_params["name"],
        playground=playground,
        llm=custom_llm,
        vector_database=vdb,
        vector_database_settings=vdb_settings,
        llm_settings={
            "system_prompt": system_prompt,
            "validation_id": validation.id,
        },
    )

    llm_blueprint = LLMBlueprint.update(llm_blueprint, is_saved=True)
    return llm_blueprint
