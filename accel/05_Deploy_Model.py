# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will deploy the custom model registered with MLflow in the prior notebook and deploy it to Databricks model serving ([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)).  Databricks model serving provides containerized deployment options for registered models thought which authenticated applications can interact with the model via a REST API.  This provides MLOps teams an easy way to deploy, manage and integrate their models with various applications.

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

import mlflow
import requests
import json
import time
from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

latest_version = mlflow.MlflowClient().get_latest_versions(configs['registered_model_name'], stages=['Production'])[0].version

# gather other inputs the API needs
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()


# COMMAND ----------

# MAGIC %md ##Step 1: Deploy Model Serving Endpoint
# MAGIC
# MAGIC Models may typically be deployed to model sharing endpoints using either the Databricks workspace user-interface or a REST API.  Because our model depends on the deployment of a sensitive environment variable, we will need to leverage a relatively new model serving feature that is currently only available via the REST API.
# MAGIC
# MAGIC See our served model config below and notice the `env_vars` part of the served model config - you can now store a key in a secret scope and pass it to the model serving endpoint as an environment variable.

# COMMAND ----------


served_models = [
  {
    "name": configs['serving_endpoint_name'],
    "config":{
    "served_models": [{
      "model_name": configs['registered_model_name'],
      "model_version": "2",
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      "scale_to_zero_enabled": 'false'}],
    "traffic_config": {"routes": [{"served_model_name": configs['serving_endpoint_name'], "traffic_percentage": "100"}]}
    }
  }
]


# COMMAND ----------

def endpoint_exists():
  """Check if an endpoint with the serving_endpoint_name exists"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{configs['serving_endpoint_name']}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

def wait_for_endpoint():
  """Wait until deployment is ready, then return endpoint config"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{configs['serving_endpoint_name']}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

def create_endpoint():
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {configs['serving_endpoint_name']}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  request_data = served_models[0]
  print(endpoint_url)
  print(json.dumps(request_data))
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.post(endpoint_url, json=request_data, headers=headers)
  #print(response.raise_for_status())
  #wait_for_endpoint()
  print(response.json())
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{configs['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")
  
def update_endpoint():
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {configs['serving_endpoint_name']}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{configs['serving_endpoint_name']}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = served_models[0]
  response = requests.put(endpoint_url, data=json.dumps(request_data), headers=headers)
  #response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{configs['serving_endpoint_name']}" target="_blank">{configs['serving_endpoint_name']}</a> serving endpoint""")


def list_endpoints():
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {configs['serving_endpoint_name']}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(endpoint_url, headers=headers)

  lst = json.loads(response.text)['endpoints']

  for endpoint in lst:
    displayHTML(f'<font face="courier">{endpoint}</font>')

# COMMAND ----------


# list_endpoints()
#kick off endpoint creation/update
if not endpoint_exists():
  create_endpoint()
else:
  update_endpoint()

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

endpoint_url = f"""https://{serving_host}/serving-endpoints/{config['serving_endpoint_name']}/invocations"""


def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(dataset):
    url = endpoint_url
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()

# COMMAND ----------


