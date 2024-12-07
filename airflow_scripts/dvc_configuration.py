from airflow.providers.google.cloud.hooks.secret_manager import GoogleCloudSecretManagerHook
import os

secret_manager_hook = GoogleCloudSecretManagerHook()

secret_name = "airflow_secret"
secret = secret_manager_hook.access_secret(secret_id=secret_name)
secret_data = secret.payload.data.decode("UTF-8")

service_account_file = "/home/almasrimilad21/airflow/ds-airflow-project-3d6d03de0d68.json"

with open(service_account_file, "wb") as f:
    f.write(secret_data.encode("utf-8"))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file