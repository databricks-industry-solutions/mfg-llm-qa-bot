# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Assemble App
# MAGIC
# MAGIC In this notebook, we call the custom MLflow pyfunc wrapper that we created in the previous notebook. We then load in the vectorstore as a retriever and pass the required environment configuration. We then persist the model to MLflow and make the required MLflow API call to register this model in the model registry.
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/MLflow-RAG.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC This notebook was tested on the following infrastructure:
# MAGIC * DBR 13.3ML (GPU)
# MAGIC * g5.2xlarge (AWS) - however comparable infra on Azure should work (A10s)

# COMMAND ----------

# MAGIC %md
# MAGIC Load mlflow pyfunc wrapper 

# COMMAND ----------

# MAGIC %run ./03_Create_ML

# COMMAND ----------

# MAGIC %md
# MAGIC Ensure dependencies are passed to the environment in Mlflow

# COMMAND ----------

# get base environment configuration
conda_env = mlflow.pyfunc.get_default_conda_env()
# define packages required by model

packages = [
  f'langchain==0.1.6',
  f'SQLAlchemy==2.0.27',
  f'databricks-vectorsearch==0.22',
  f'mlflow[databricks]',
  f'xformers==0.0.24',
  f'transformers==4.37.2',
  f'accelerate==0.27.0'
]

# add required packages to environment configuration
conda_env['dependencies'][-1]['pip'] += packages

print(
  conda_env
  )

# COMMAND ----------

# instantiate bot object
mfgsdsbot = MLflowMfgBot()


# COMMAND ----------

# MAGIC %md
# MAGIC For testing locally. hack a context object.

# COMMAND ----------


# context = mlflow.pyfunc.PythonModelContext(
#                                            model_config={"configs":json.dumps(configs),
#                  "automodelconfigs":str(automodelconfigs),
#                  "pipelineconfigs":str(pipelineconfigs)},
#                                            artifacts=None
                                           
#                                            )
# mfgsdsbot.load_context(context)
# # get response to question
# filterdict={'Name':'ACETONE'}
# mfgsdsbot.predict(context, {'questions':['when should OSHA get involved on acetone exposure?'], 'search_kwargs':{"k": 10, "filter":filterdict, "fetch_k":100}})

# COMMAND ----------

# MAGIC %md
# MAGIC Use the wrapper we created from 03_Create_ML to log experiment in MLflow

# COMMAND ----------

# persist model to mlflow
import json
with mlflow.start_run():
  _ = (
    mlflow.pyfunc.log_model(
      python_model=mfgsdsbot,
      code_path=['./utils/stoptoken.py'], #this is not used but shows how additional classes can be included.
      conda_env=conda_env,
      artifact_path='mfgmodel',
      model_config={"configs":json.dumps(configs),
                 "automodelconfigs":str(automodelconfigs),
                 "pipelineconfigs":str(pipelineconfigs)},
      registered_model_name=configs['registered_model_name']
      )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Register model in the [MLflow Model Registry](https://docs.databricks.com/en/mlflow/model-registry.html)
# MAGIC
# MAGIC We do this to help enable CI/CD and for ease of deployment in the next notebook.

# COMMAND ----------

client = mlflow.MlflowClient()

latest_version = client.get_latest_versions(configs['registered_model_name'], stages=['None'])[0].version
print(latest_version)
#transition model to production
client.transition_model_version_stage(
    name=configs['registered_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC Load model from Model Registry

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{configs['registered_model_name']}/Production")

# COMMAND ----------

# MAGIC %md
# MAGIC Verify model from Registry is returning results as expected
# MAGIC
# MAGIC Test model on various queries
# MAGIC
# MAGIC If you see CUDA out of memory errors at this step, restart the compute and try again. This occurs due to leftover processes from running other notebooks.

# COMMAND ----------

import pandas as pd
# construct search
filterdict={'Name':'ACETONE'}
search = {'question':['what are some properties of Acetone?'],'filter':[filterdict]}

# call model
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={'Name':'ACETALDEHYDE'}
search = {'question':['what are some properties of Acetaldehyde?'],'filter':[filterdict]}

y=model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={}
search = {'question':['When is medical attention needed?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)


# COMMAND ----------

filterdict={}
search = {'question':['What is the difference between nuclear fusion and fission?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={}
search = {'question':['What should we do if OSHA get involved in a chemical event?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={}
search = {'question':['What are the exposure limits for acetyl methyl carbinol cause?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={'Name':'ACETYL METHYL CARBINOL'}
search = {'question':['What are the exposure limits for acetyl methyl carbinol cause?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

#check what the split JSON looks like to pass to our predict function.
filterdict={'Name':'ACETALDEHYDE'}
search = {'question':['what are some properties of Acetaldehyde?'],'filter':[filterdict]}
json = pd.DataFrame.from_dict(search).to_json(orient='split')
print(json)

# COMMAND ----------


