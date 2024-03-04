# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Deploy Model
# MAGIC
# MAGIC In this notebook, we will deploy the custom model registered with MLflow in the prior notebook and deploy it to Databricks model serving ([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)).  Databricks model serving provides containerized deployment options for registered models thought which authenticated applications can interact with the model via a REST API.  This provides MLOps teams an easy way to deploy, manage and integrate their models with various applications.
# MAGIC
# MAGIC This notebook relies on GPU model serving which works in [limited regions](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-limits.html)

# COMMAND ----------

# MAGIC %md
# MAGIC Install Libraries

# COMMAND ----------

# MAGIC %pip install mlflow[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

import mlflow
import requests
import json
import time
from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

# MAGIC %md
# MAGIC Get the Latest version from the Model registry

# COMMAND ----------

client = mlflow.MlflowClient()
vallmods = client.search_model_versions(f"name=\'{configs['registered_model_name']}\'", max_results=1000)

latest_version = max([int(ver.version) for ver in vallmods])
print(latest_version)

# COMMAND ----------

client = mlflow.MlflowClient()
#get the latest version of the registered model in MLFlow
if configs['isucregistry'] is True:
  print('Getting model version from UC registry')
  latest_version = client.get_model_version_by_alias(configs['registered_model_name'], 'champion').version

# # gather other inputs the API needs
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()
# print(creds.token)
os.environ['DATABRICKS_HOST']=serving_host
os.environ['DATABRICKS_TOKEN']=creds.token


# COMMAND ----------

# MAGIC %md 
# MAGIC **Step 1**: Deploy Model Serving Endpoint
# MAGIC
# MAGIC Models may typically be deployed to model sharing endpoints using either the Databricks workspace user-interface or a REST API.  Because our model depends on the deployment of a sensitive environment variable, we will need to leverage a relatively new model serving feature that is currently only available via the REST API.
# MAGIC
# MAGIC See our served model config below and notice the `env_vars` part of the served model config - you can now store a key in a secret scope and pass it to the model serving endpoint as an environment variable.
# MAGIC
# MAGIC **Use GPU_MEDIUM for custom models**
# MAGIC
# MAGIC **Use CPU for foundational/external models**
# MAGIC

# COMMAND ----------

#[0] is used for creation with name and config.
#[1] is used for updates.
served_models = [
  {
    "name": configs['serving_endpoint_name'],
    "config":{
    "served_entities": [{
      "name": configs['serving_endpoint_name'],
      "entity_name": configs['registered_model_name'],
      "entity_version": latest_version,
      "workload_type": "CPU", #use GPU_MEDIUM for custom models. use CPU for foundational/external model
      "workload_size": "Small",
      "scale_to_zero_enabled": 'true',
      "environment_vars":{
     }}] 
    }
  }
]
#"DATABRICKS_HOST": f"{{{{secrets/{secret_scope_name}/{secret_key_name}}}}},
#"traffic_config": {"routes": [{"served_model_name": configs['serving_endpoint_name'], "traffic_percentage": "100"}]}

# COMMAND ----------

#Utility functions for deployment using the serving endpoints REST API endpoint.
from mlflow.deployments import get_deploy_client

def endpoint_exists():
  """Check if an endpoint with the serving_endpoint_name exists"""
  try:
    client = get_deploy_client("databricks")
    print(f"Endpoint = {configs['serving_endpoint_name']}")
    endpoint = client.get_endpoint(configs['serving_endpoint_name'])
    print(endpoint)
    return True
  except Exception as ex:
    print(ex.response.text)
    return False

def wait_for_endpoint():
  """Wait until deployment is ready, then return endpoint config"""
  import datetime, json
  try:
    client = get_deploy_client("databricks")
    print(f"Endpoint = {configs['serving_endpoint_name']}")
    response = client.get_endpoint(configs['serving_endpoint_name'])   
    #loop till ready
    while response["state"]["ready"] == "NOT_READY" and response["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
      print(f"Waiting 120s for deployment or update to finish  - {response['state']['ready']}")
      time.sleep(120)
      response = client.get_endpoint(configs['serving_endpoint_name']) 
      now = datetime.datetime.now()
      print(now.time())
      print(response)
  except Exception as ex:
    print(ex)
    return False

  return response

def create_endpoint():
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {configs['serving_endpoint_name']}")
  client = get_deploy_client("databricks")
  try:
    respdict=client.create_endpoint(name=configs['serving_endpoint_name'], config=served_models[0]["config"])
    print(respdict)
    displayHTML(f"""Created the <a href="/#mlflow/endpoints/{configs['serving_endpoint_name']}" target="_blank">{configs['serving_endpoint_name']}</a> serving endpoint""")
  except Exception as ex:
    print(ex)


def update_endpoint():
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {configs['serving_endpoint_name']}")
  client = get_deploy_client("databricks")
  try:
    client.update_endpoint (name=configs['serving_endpoint_name'], config=served_models[0]["config"])
    wait_for_endpoint()
    displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{configs['serving_endpoint_name']}" target="_blank">{configs['serving_endpoint_name']}</a> serving endpoint""")
  except Exception as ex:
    print(ex)


def delete_endpoint():
  """Delete endpoints"""
  client = get_deploy_client("databricks")
  try:
    if endpoint_exists():
      client.delete_endpoint(configs['serving_endpoint_name'])
      print('Endpoint deleted')
    else:
      print('Endpoint doesnt exist')
  except Exception as ex:
    print(ex)

def list_endpoints():
  """List all endpoints"""

  client = get_deploy_client("databricks")
  try:
    respdict = client.list_endpoints()
    for endpt in respdict:
      displayHTML(f"""<font face="courier">Endpoint: {str(endpt)}</font>""")
  except Exception as ex:
    print(ex)


# COMMAND ----------

list_endpoints()

# COMMAND ----------

delete_endpoint()

# COMMAND ----------

endpoint_exists()

# COMMAND ----------

#launch endpoint creation/update
if not endpoint_exists():
  create_endpoint()
  wait_for_endpoint()
else:
  update_endpoint()
  wait_for_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC [Query the logs](https://docs.databricks.com/api/workspace/servingendpoints/logs)
# MAGIC
# MAGIC ```curl -n -X GET https://<tenant>.cloud.databricks.com/api/2.0/serving-endpoints/mfg-llm-qabot-serving-endpoint/served-models/mfg-llm-qabot-6/logs```
