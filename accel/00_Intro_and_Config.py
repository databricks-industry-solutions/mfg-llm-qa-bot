# Databricks notebook source
# MAGIC %md The purpose of this notebook is to provide access to safety data used by the Product Search solution accelerator.  You may find this notebook on https://github.com/databricks-industry-solutions/product-search.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC The goal of this solution accelerator is to show how we can leverage a large language model in combination with our own data to create an interactive application capable of answering questions specific to a particular domain or subject area.  The core pattern behind this is the delivery of a question along with a document or document fragment that provides relevant context for answering that question to the model.  The model will then respond with an answer that takes into consideration both the question and the context. This notebook builds upon the QA bot accelerator that we previously released and demonstrates how you can easily swap open sourced models from Huggingface and test out their performance.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_flow.png' width=500>
# MAGIC
# MAGIC </p>
# MAGIC To assemble this application, *i.e.* the Q&A Bot, we will need to assemble a series of documents that are relevant to the domain we wish to serve.  We will need to index these using a vector database. We will then need to assemble the core application which combines a question with a document to form a prompt and submits that prompt to a model in order to generate a response. Finally, we'll need to package both the indexed documents and the core application component as a microservice to enable a wide range of deployment options.
# MAGIC
# MAGIC We will tackle these steps across the following notebooks:</p>
# MAGIC
# MAGIC * 01: Create Embeddings
# MAGIC * 02: Define Basic Search
# MAGIC * 03: Create ML
# MAGIC * 04: Assemble App
# MAGIC * 05: Deploy Model
# MAGIC * 06: Example App
# MAGIC </p>

# COMMAND ----------

# MAGIC %md ##Configuration
# MAGIC
# MAGIC The configuration below govern what's being loaded throughout the series of notebooks. If you wish to change the open sourced model type or tokenizer or something else, please change the configs file in /utils to do so. This notebook was tested on the following infrastructure:
# MAGIC * DBR 13.2ML (GPU)
# MAGIC * g5.4xlarge (AWS) - however comparable infra on Azure should work (A10s)

# COMMAND ----------

# DBTITLE 1,Initialize Config Variables
# MAGIC %run ./utils/configs

# COMMAND ----------

print(configs)

# COMMAND ----------

# DBTITLE 1,Databricks url and token
import os
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
config['databricks token'] = ctx.apiToken().getOrElse(None)
config['databricks url'] = ctx.apiUrl().getOrElse(None)
os.environ['DATABRICKS_TOKEN'] = config["databricks token"]
os.environ['DATABRICKS_URL'] = config["databricks url"]

# COMMAND ----------

# DBTITLE 1,MLflow experiment
import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/mfg_llm_sds_search'.format(username))

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  NJ SDS | Right to Know Hazardous Substance Fact Sheets | None  |   https://web.doh.state.nj.us/rtkhsfs/indexfs.aspx |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
