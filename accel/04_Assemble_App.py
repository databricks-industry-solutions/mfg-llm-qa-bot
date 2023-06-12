# Databricks notebook source
registered_model_name="rkm_mfg_accel"

# COMMAND ----------

# persist model to mlflow
with mlflow.start_run():
  _ = (
    mlflow.pyfunc.log_model(
      python_model=model,
      extra_pip_requirements=['chromadb==0.3.26', 'langchain==0.0.196', 'transformers==4.30.1', 'accelerate==0.20.3', 'bitsandbytes==0.39.0', 'einops==0.6.1', 'xformers==0.0.20'],
      artifact_path='mfgmodel',
      registered_model_name=registered_model_name
      )
    )

# COMMAND ----------

# connect to mlflow 
client = mlflow.MlflowClient()

# identify latest model version
latest_version = client.get_latest_versions(config['registered_model_name'], stages=['None'])[0].version

# move model into production
client.transition_model_version_stage(
    name=registered_model_name,
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# retrieve model from mlflow
model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/Production")

# assemble question input
queries = pd.DataFrame({'question':[
  "when is medical attention needed?",
  "What should we do when clothing is contaminated?",
  "What kind of hazardous substances exist?"
]})

# get a response
model.predict(queries)
