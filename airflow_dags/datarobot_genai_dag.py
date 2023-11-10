from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project


class KedroOperator(BaseOperator):
    @apply_defaults
    def __init__(
        self,
        package_name: str,
        pipeline_name: str,
        node_name: str,
        project_path: str | Path,
        env: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.package_name = package_name
        self.pipeline_name = pipeline_name
        self.node_name = node_name
        self.project_path = project_path
        self.env = env

    def execute(self, context):
        configure_project(self.package_name)
        with KedroSession.create(self.package_name,
                                 self.project_path,
                                 env=self.env) as session:
            session.run(self.pipeline_name, node_names=[self.node_name])


# Kedro settings required to run your pipeline
env = "local"
pipeline_name = "__default__"
project_path = Path.cwd()
package_name = "datarobot_genai"

# Using a DAG context manager, you don't have to specify the dag property of each task
with DAG(
    dag_id="datarobot-genai",
    start_date=datetime(2023,1,1),
    max_active_runs=3,
    # https://airflow.apache.org/docs/stable/scheduler.html#dag-runs
    schedule_interval="@once",
    catchup=False,
    # Default settings applied to all tasks
    default_args=dict(
        owner="airflow",
        depends_on_past=False,
        email_on_failure=False,
        email_on_retry=False,
        retries=1,
        retry_delay=timedelta(minutes=5)
    )
) as dag:
    tasks = {
        "add-llm-to-usecase-create-vector-database-settings": KedroOperator(
            task_id="add-llm-to-usecase-create-vector-database-settings",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="add_llm_to_usecase.create_vector_database_settings",
            project_path=project_path,
            env=env,
        ),
        "byo-llm-deployment-set-credentials": KedroOperator(
            task_id="byo-llm-deployment-set-credentials",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="byo_llm_deployment.set_credentials",
            project_path=project_path,
            env=env,
        ),
        "create-usecase-create-usecase-object": KedroOperator(
            task_id="create-usecase-create-usecase-object",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="create_usecase.create_usecase_object",
            project_path=project_path,
            env=env,
        ),
        "create-usecase-upload-vector-database": KedroOperator(
            task_id="create-usecase-upload-vector-database",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="create_usecase.upload_vector_database",
            project_path=project_path,
            env=env,
        ),
        "byo-llm-deployment-deploy-custom-llm": KedroOperator(
            task_id="byo-llm-deployment-deploy-custom-llm",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="byo_llm_deployment.deploy_custom_llm",
            project_path=project_path,
            env=env,
        ),
        "create-usecase-add-vector-database": KedroOperator(
            task_id="create-usecase-add-vector-database",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="create_usecase.add_vector_database",
            project_path=project_path,
            env=env,
        ),
        "create-usecase-create-playground": KedroOperator(
            task_id="create-usecase-create-playground",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="create_usecase.create_playground",
            project_path=project_path,
            env=env,
        ),
        "add-llm-to-usecase-add-pre-baked-llm-to-playground": KedroOperator(
            task_id="add-llm-to-usecase-add-pre-baked-llm-to-playground",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="add_llm_to_usecase.add_pre_baked_llm_to_playground",
            project_path=project_path,
            env=env,
        ),
        "byo-llm-deployment-predict": KedroOperator(
            task_id="byo-llm-deployment-predict",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="byo_llm_deployment.predict",
            project_path=project_path,
            env=env,
        ),
        "byo-llm-deployment-validate-llm-deployment": KedroOperator(
            task_id="byo-llm-deployment-validate-llm-deployment",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="byo_llm_deployment.validate_llm_deployment",
            project_path=project_path,
            env=env,
        ),
        "add-llm-to-usecase-add-byo-llm-to-playground": KedroOperator(
            task_id="add-llm-to-usecase-add-byo-llm-to-playground",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="add_llm_to_usecase.add_byo_llm_to_playground",
            project_path=project_path,
            env=env,
        ),
        "prompting-pre-baked-submit-prompt": KedroOperator(
            task_id="prompting-pre-baked-submit-prompt",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="prompting.pre_baked_submit_prompt",
            project_path=project_path,
            env=env,
        ),
        "prompting-byo-llm-submit-prompt": KedroOperator(
            task_id="prompting-byo-llm-submit-prompt",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="prompting.byo_llm_submit_prompt",
            project_path=project_path,
            env=env,
        ),
        "prompting-comparison-submit-prompt": KedroOperator(
            task_id="prompting-comparison-submit-prompt",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="prompting.comparison_submit_prompt",
            project_path=project_path,
            env=env,
        ),
    }

    tasks["byo-llm-deployment-set-credentials"] >> tasks["byo-llm-deployment-deploy-custom-llm"]
    tasks["create-usecase-create-usecase-object"] >> tasks["create-usecase-add-vector-database"]
    tasks["create-usecase-create-usecase-object"] >> tasks["create-usecase-create-playground"]
    tasks["create-usecase-upload-vector-database"] >> tasks["create-usecase-add-vector-database"]
    tasks["add-llm-to-usecase-create-vector-database-settings"] >> tasks["add-llm-to-usecase-add-pre-baked-llm-to-playground"]
    tasks["add-llm-to-usecase-create-vector-database-settings"] >> tasks["add-llm-to-usecase-add-byo-llm-to-playground"]
    tasks["create-usecase-create-playground"] >> tasks["add-llm-to-usecase-add-pre-baked-llm-to-playground"]
    tasks["create-usecase-create-playground"] >> tasks["add-llm-to-usecase-add-byo-llm-to-playground"]
    tasks["create-usecase-add-vector-database"] >> tasks["add-llm-to-usecase-add-pre-baked-llm-to-playground"]
    tasks["create-usecase-add-vector-database"] >> tasks["add-llm-to-usecase-add-byo-llm-to-playground"]
    tasks["byo-llm-deployment-deploy-custom-llm"] >> tasks["byo-llm-deployment-predict"]
    tasks["byo-llm-deployment-deploy-custom-llm"] >> tasks["byo-llm-deployment-validate-llm-deployment"]
    tasks["byo-llm-deployment-validate-llm-deployment"] >> tasks["add-llm-to-usecase-add-byo-llm-to-playground"]
    tasks["add-llm-to-usecase-add-pre-baked-llm-to-playground"] >> tasks["prompting-pre-baked-submit-prompt"]
    tasks["add-llm-to-usecase-add-pre-baked-llm-to-playground"] >> tasks["prompting-comparison-submit-prompt"]
    tasks["add-llm-to-usecase-add-byo-llm-to-playground"] >> tasks["prompting-byo-llm-submit-prompt"]
    tasks["add-llm-to-usecase-add-byo-llm-to-playground"] >> tasks["prompting-comparison-submit-prompt"]
