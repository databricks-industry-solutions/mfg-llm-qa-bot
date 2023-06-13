# Databricks notebook source
# %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

# from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

# MAGIC %md Before setting up the rest of the accelerator, we need set up an [OpenAI key in order to access OpenAI models](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key). Here we demonstrate using the [Databricks Secret Scope](https://docs.databricks.com/security/secrets/secret-scopes.html) for credential management. 
# MAGIC
# MAGIC Copy the block of code below, replace the name the secret scope and fill in the credentials and execute the block. After executing the code, The accelerator notebook will be able to access the credentials it needs.
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC client = NotebookSolutionCompanion().client
# MAGIC try:
# MAGIC   client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/scopes/create", {"scope": "solution-accelerator-cicd"})
# MAGIC except:
# MAGIC   pass
# MAGIC   
# MAGIC client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/put", {
# MAGIC   "scope": "solution-accelerator-cicd",
# MAGIC   "key": "openai_api",
# MAGIC   "string_value": '____' 
# MAGIC })
# MAGIC ```

# COMMAND ----------

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = dbutils.secrets.get('mfg-llm-solution-accel', 'huggingface')
os.environ['HF_HOME'] = '/dbfs/temp/hfmfgcache'

# COMMAND ----------

def dbfsnormalize(path):
  path = path.replace('/dbfs/', 'dbfs:/')
  return path

# COMMAND ----------

configs = {}
configs.update({'chroma_persist_dir' : '/dbfs/Users/ramdas.murali@databricks.com/chromadb'})
configs.update({'data_dir':'/dbfs/Users/ramdas.murali@databricks.com/data/sds_pdf'})
configs.update({'chunk_size':600})
configs.update({'chunk_overlap':20})

configs.update({'temperature':0.8})
configs.update({'max_new_tokens':128})
configs.update({'prompt_template':"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}"""})
configs.update({'num_similar_docs':8})

configs.update({'registered_model_name':'rkm_mfg_accel'})


# COMMAND ----------

# if 'config' not in locals():
#   config = {}

# COMMAND ----------

# DBTITLE 1,Set document path
# config['kb_documents_path'] = "s3://db-gtm-industry-solutions/data/rcg/diy_llm_qa_bot/"
# config['vector_store_path'] = '/dbfs/tmp/qabot/vector_store' # /dbfs/... is a local file system representation

# COMMAND ----------

# DBTITLE 1,Create database
# config['database_name'] = 'qabot'

# # create database if not exists
# _ = spark.sql(f"create database if not exists {config['database_name']}")

# # set current datebase context
# _ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Set Environmental Variables for tokens
# import os

# os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("solution-accelerator-cicd", "openai_api")

# COMMAND ----------

# DBTITLE 1,mlflow settings
# import mlflow
# config['registered_model_name'] = 'databricks_llm_qabot_solution_accelerator'
# config['model_uri'] = f"models:/{config['registered_model_name']}/production"
# username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
# _ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,Set OpenAI model configs
# config['openai_embedding_model'] = 'text-embedding-ada-002'
# config['openai_chat_model'] = "gpt-3.5-turbo"
# config['system_message_template'] = """You are a helpful assistant built by Databricks, you are good at helping to answer a question based on the context provided, the context is a document. If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If you did not find a good answer from the context, just say I don't know. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context to answer the question."""
# config['human_message_template'] = """Given the context: {context}. Answer the question {question}."""
# config['temperature'] = 0.15

# COMMAND ----------

# DBTITLE 1,Set evaluation config
# config["eval_dataset_path"]= "./data/eval_data.tsv"

# COMMAND ----------

# DBTITLE 1,Set deployment configs
# config['openai_key_secret_scope'] = "solution-accelerator-cicd" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope name you actually use 
# config['openai_key_secret_key'] = "openai_api" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope key name you actually use
# config['serving_endpoint_name'] = "llm-qabot-endpoint"
