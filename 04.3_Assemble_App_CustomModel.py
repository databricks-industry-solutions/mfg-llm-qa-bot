# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Assemble App
# MAGIC
# MAGIC In this notebook, we call the custom MLflow pyfunc wrapper which loads an open source model that we created in the notebook 03.3. In this notebook, we assemble the custom MLflow pyfunc wrapper and store it in the MLFlow Model registry. We persist the model to MLflow and make the required MLflow API call to register this model in the model registry. We change the stage of the model to Production.
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

# MAGIC %run ./03.3_Create_ML_CustomModel

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
# question = {'questions':['what are some properties of Acetaldehyde?'],'filter':[filterdict]}
# preds = mfgsdsbot.predict(context, question)
# print(preds)

# COMMAND ----------


from mlflow.models import infer_signature
import pandas as pd

filterdict={}
question = {'questions':['what are some properties of Acetaldehyde?'],'filter':[filterdict]}

preds = {'query': 'what are some properties of Acetaldehyde?', 'result': 'Clear, colorless liquid or gas, reacts with air, reacts with strong acids, reacts with strong bases, reacts with amines and ketones, reacts with oxidizing agents, isolates in liquid form for up to 150 feet or in a fire for up to half a mile', 'source_documents': ['doc1','doc2']}

signature = infer_signature(pd.DataFrame.from_dict(question), preds)

#convert required fields to false
signlisti = signature.to_dict()['inputs']
print(signlisti)
signlisti = signlisti.replace('"required": true', '"required": false')
signlisto = signature.to_dict()['outputs']
signlisto = signlisto.replace('"required": true', '"required": false')
signature = signature.from_dict({"inputs":signlisti, "outputs":signlisto})
print(signature.to_dict())

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
                 "pipelineconfigs":str(pipelineconfigs)
                 },
      registered_model_name=configs['registered_model_name'],
      signature = signature
      )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Register model in the [MLflow Model Registry](https://docs.databricks.com/en/mlflow/model-registry.html)
# MAGIC
# MAGIC We do this to help enable CI/CD and for ease of deployment in the next notebook.

# COMMAND ----------

client = mlflow.MlflowClient()
v = client.search_model_versions(f"name=\'{configs['registered_model_name']}\'", max_results=1000)
version = max([ver.version for ver in v])
print(version)
client.set_registered_model_alias(configs['registered_model_name'], "champion", version)

# COMMAND ----------

# MAGIC %md
# MAGIC Load model from Model Registry

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{configs['registered_model_name']}@champion")

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
search = {'questions':['what are some properties of Acetone?'],'filter':[filterdict]}

# call model
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={'Name':'ACETALDEHYDE'}
search = {'questions':['what are some properties of Acetaldehyde?'],'filter':[filterdict]}

y=model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={}
search = {'questions':['When is medical attention needed?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)


# COMMAND ----------

filterdict={}
search = {'questions':['What is the difference between nuclear fusion and fission?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={}
search = {'questions':['What should we do if OSHA get involved in a chemical event?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={}
search = {'questions':['What are the exposure limits for acetyl methyl carbinol cause?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

filterdict={'Name':'ACETYL METHYL CARBINOL'}
search = {'questions':['What are the exposure limits for acetyl methyl carbinol cause?'],'filter':[filterdict]}
y = model.predict(pd.DataFrame.from_dict(search))
print(y)

# COMMAND ----------

#check what the split JSON looks like to pass to our predict function.
filterdict={'Name':'ACETALDEHYDE'}
search = {'questions':['what are some properties of Acetaldehyde?'],'filter':[filterdict]}
json = pd.DataFrame.from_dict(search).to_json(orient='split')
print(json)

# COMMAND ----------


