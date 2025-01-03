dvc init --no-scm --force --verbose
dvc import-url gs://ds_project_data/source_data --force
dvc import-url gs://ds_project_data/models --force
dvc remote add -d gcs_remote gs://ds_project_data
dvc pull --remote gcs_remote --force --verbose
mv /home/almasrimilad21/airflow/dags/source_data /home/almasrimilad21/airflow/