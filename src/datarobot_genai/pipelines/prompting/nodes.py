"""
This is a boilerplate pipeline 'prompting'
generated using Kedro 0.18.14
"""
from datarobot._experimental.models.genai.chat_prompt import ChatPrompt
from datarobot._experimental.models.genai.comparison_prompt import ComparisonPrompt
from datarobot._experimental.models.genai.llm_blueprint import LLMBlueprint


def submit_prompt(llm_blueprint: LLMBlueprint, prompt_text: str) -> str:
    # Submit a prompt
    chat_prompt = ChatPrompt.create(
        llm_blueprint.id,
        prompt_text,
        wait_for_completion=True,
    )
    if not chat_prompt.result_text:
        raise ValueError("Prompt failed to generate a result")

    return chat_prompt.result_text


def save_and_submit_comparison_prompt(
    llm_blueprint_1: LLMBlueprint, llm_blueprint_2: LLMBlueprint, prompt_text
) -> str:

    if not llm_blueprint_1.is_saved:
        llm_blueprint_1 = LLMBlueprint.update(llm_blueprint_1, is_saved=True)
    if not llm_blueprint_2.is_saved:
        llm_blueprint_2 = LLMBlueprint.update(llm_blueprint_2, is_saved=True)

    comparison_prompt = ComparisonPrompt.create(
        llm_blueprints=[llm_blueprint_1, llm_blueprint_2],
        text=prompt_text,
        wait_for_completion=True,
    )
    result_text = "\n".join(
        [r.result_text for r in comparison_prompt.results if r.result_text]
    )
    return result_text
