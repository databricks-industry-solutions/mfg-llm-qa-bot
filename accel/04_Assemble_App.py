# Databricks notebook source
# MAGIC %run ./03_Create_ML

# COMMAND ----------


# instantiate bot object
mfgsdsbot = MLflowMfgBot(
        configs,
        automodelconfigs,
        pipelineconfigs,
        dbutils.secrets.get('rkm-scope', 'huggingface'))


#for testing locally
#context = mlflow.pyfunc.PythonModelContext(artifacts={"prompt_template":configs['prompt_template']})
#mfgsdsbot.load_context(context)
# get response to question
#mfgsdsbot.predict(context, {'questions':['when should OSHA get involved?']})

# COMMAND ----------

# get base environment configuration
conda_env = mlflow.pyfunc.get_default_conda_env()
# define packages required by model
packages = [
  f'chromadb==0.3.26',
  f'langchain==0.0.197',
  f'transformers==4.30.1',
  f'accelerate==0.20.3',
  f'bitsandbytes==0.39.0',
  f'einops==0.6.1',
  f'xformers==0.0.20',
  f'typing-inspect==0.8.0',
  f'typing_extensions==4.5.0'
  ]

# add required packages to environment configuration
conda_env['dependencies'][-1]['pip'] += packages

print(
  conda_env
  )

# COMMAND ----------

# persist model to mlflow
with mlflow.start_run():
  _ = (
    mlflow.pyfunc.log_model(
      python_model=mfgsdsbot,
      code_path=['./utils/stoptoken.py'],
      conda_env=conda_env,
      artifact_path='mfgmodel',
      registered_model_name=configs['registered_model_name']
      )
    )

# COMMAND ----------

client = mlflow.MlflowClient()

latest_version = client.get_latest_versions(configs['registered_model_name'], stages=['None'])[0].version
print(latest_version)
client.transition_model_version_stage(
    name=configs['registered_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{configs['registered_model_name']}/Production")


# COMMAND ----------

import pandas as pd
# construct search
search = pd.DataFrame({'questions':['what should we do if OSHA is involved?']})

# call model
y = model.predict(search)
print(y)

# COMMAND ----------

y=model.predict({'questions':['what should we do if OSHA is involved?']})
print(y)

# COMMAND ----------

y=model.predict({'questions':['When is medical attention needed?', 'how long to wait after symptoms appear?']})
print(y)

