# Databricks notebook source
# MAGIC %md 
# MAGIC # Manufacturing QA over Custom Datasets with 🦜️🔗 LangChain and Open Source LLMs on Hugging Face 🤗
# MAGIC
# MAGIC The purpose of this notebook is to provide a pattern for the building and deployment of LLMs for a Manufacturing use case.  You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Large Language Models for Manufacturers
# MAGIC
# MAGIC The goal of this solution accelerator is to show how we can leverage a large language model in combination with our own data to create an interactive application capable of answering questions specific to manufacturing.  
# MAGIC
# MAGIC In essence the use case is around augmenting the diagnostics capability of field service engineers. Field service engineers are often challenged with accessing tons of documents that are intertwined. Having an LLM to reduce the time taken to diagnose the problem will inadvertently increase efficiencies.
# MAGIC
# MAGIC The core pattern behind this is the delivery of a question along with a document or document fragment that provides relevant context for answering that question to the model.  The model will then respond with an answer that takes into consideration both the question and the context. This notebook builds upon the QA bot accelerator that we previously released and demonstrates how you can easily swap open sourced models from Huggingface and test out their performance.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_flow.png' width=500>
# MAGIC
# MAGIC </p>
# MAGIC To assemble this application, i.e. the Q&A Bot, we will need to assemble a series of documents that are relevant to the domain we wish to serve.  We will need to index these using a vector database. We will then need to assemble the core application which combines a question with a document to form a prompt and submits that prompt to a model in order to generate a response. Finally, we'll need to package both the indexed documents and the core application component as a microservice to enable a wide range of deployment options.
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
# MAGIC The configuration below govern what's being loaded throughout the series of notebooks. If you wish to change the open sourced model type or tokenizer or something else, please change the configs file in `/utils` to do so. This notebook was tested with the infrastructure specified in the RUNME notebook.

# COMMAND ----------

# DBTITLE 1,Initialize Config Variables
# MAGIC %run ./utils/configs

# COMMAND ----------

print(configs)

# COMMAND ----------

def dbfsnormalize(path):
  path = path.replace('/dbfs/', 'dbfs:/')
  return path

# COMMAND ----------

# DBTITLE 1,Set up local copy of source data
dbutils.fs.cp("s3a://db-gtm-industry-solutions/data/MFG/llm_qa/sds_pdf", dbfsnormalize(configs["data_dir"]), True)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  NJ SDS | Right to Know Hazardous Substance Fact Sheets | Public Domain  |   https://web.doh.state.nj.us/rtkhsfs/indexfs.aspx |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | faiss | An open source library for efficient similarity seach and clustering of dense vectors | MIT | https://faiss.ai/ |
# MAGIC | chromadb | An open source embedding database |  Apache 2.0 |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
