# Databricks notebook source
# MAGIC %md ##Assemble App
# MAGIC
# MAGIC In this notebook, we call the custom MLflow pyfunc wrapper that we created in the previous notebook. We then load in the vectorstore as a retriever and pass the required environment configuration. We then persist the model to MLflow and make the required MLflow API call to register this model in the model registry.
# MAGIC
# MAGIC
# MAGIC NEED TO UPDATE THE IMAGE - WE WILL LIKELY NEED TO DRAW THE MLFLOW ASPECT IN THIS
# MAGIC
# MAGIC <p>
# MAGIC     <img src="../images/Generate-Embeddings.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %run ./03_Create_ML

# COMMAND ----------

# Chroma
# vectorstore = Chroma(
#         collection_name="mfg_collection",
#         persist_directory=self._configs['chroma_persist_dir'],
#         embedding_function=HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2'))

#FAISS
vector_persist_dir = configs['vector_persist_dir']
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# # Load from FAISS
vectorstore = FAISS.load_local(vector_persist_dir, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": configs['num_similar_docs']}, 
                                                search_type = "similarity") 

# instantiate bot object
mfgsdsbot = MLflowMfgBot(
        configs,
        automodelconfigs,
        pipelineconfigs,
        retriever,
        os.environ['HUGGINGFACEHUB_API_TOKEN'])




# COMMAND ----------

#for testing locally
# context = mlflow.pyfunc.PythonModelContext(artifacts={"prompt_template":configs['prompt_template']})
# mfgsdsbot.load_context(context)
# # get response to question
# filterdict={'Name':'ACETONE'}
# mfgsdsbot.predict(context, {'questions':['when should OSHA get involved on acetone exposure?'], 'search_kwargs':{"k": 10, "filter":filterdict, "fetch_k":100}})

# COMMAND ----------

# get base environment configuration
conda_env = mlflow.pyfunc.get_default_conda_env()
# define packages required by model

packages = [
  f'langchain==0.0.203',
  f'transformers==4.30.1',
  f'accelerate==0.20.3',
  f'einops==0.6.1',
  f'xformers==0.0.20',
  f'sentence-transformers==2.2.2',
  f'typing-inspect==0.8.0',
  f'typing_extensions==4.5.0',
  f'faiss-cpu==1.7.4', 
  f'tiktoken==0.4.0'
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

filterdict={'Name':'ACETALDEHYDE'}
search = {'question':['what are some properties of Acetaldehyde?'],'filter':[filterdict]}
json = pd.DataFrame.from_dict(search).to_json(orient='split')
print(json)
