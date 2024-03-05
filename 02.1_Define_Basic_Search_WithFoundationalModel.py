# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Define Basic Search
# MAGIC
# MAGIC This is an alternate (easier) way of building a RAG model using a [Foundational model](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) instead of a custom model.
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/Basic-similarity-search.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC This notebook was tested on the following infrastructure:
# MAGIC * DBR 13.3ML
# MAGIC * i3.xlarge (GPUs not needed)

# COMMAND ----------

# MAGIC %md
# MAGIC Install Libraries

# COMMAND ----------

# MAGIC %pip install --upgrade langchain==0.1.6 SQLAlchemy==2.0.27 databricks-vectorsearch==0.22 mlflow[databricks] langchainhub==0.1.14 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch

from langchain.llms import Databricks


# COMMAND ----------

# MAGIC %md
# MAGIC The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:
# MAGIC
# MAGIC * A LLM, in this case it will be defined in the /utils/configs notebook
# MAGIC
# MAGIC * The respective tokenizer for the model.
# MAGIC
# MAGIC We'll explain these as we get to them, let's begin with our model.
# MAGIC
# MAGIC We initialize the model using the externalized configs such as automodelconfigs and pipelineconfigs
# MAGIC
# MAGIC
# MAGIC [Langchain source](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/databricks_vector_search.py)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Huggingface model card will give you info on the tokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC Finally we need to define the _stopping criteria_ of the model. The stopping criteria allows us to specify *when* the model should stop generating text. If we don't provide a stopping criteria the model just goes on a bit of a tangent after answering the initial question.

# COMMAND ----------

def get_retriever():
    '''Get the langchain vector retriever from the Databricks object '''
    vsc = VectorSearchClient() # auth via env vars     
    index = vsc.get_index(endpoint_name=configs['vector_endpoint_name'], 
                          index_name=f"{configs['source_catalog']}.{configs['source_schema']}.{configs['vector_index']}")

    index.describe()
    # Create the langchain retriever. text_columns-> chunks column
    # return columns metadata_name and path along with results.
    # embedding is None for Databricks managed embedding
    vectorstore = DatabricksVectorSearch(
        index, text_column="chunks", embedding=None, columns=['metadata_name', 'path']
    )
    #filter isnt working here
    return vectorstore.as_retriever(search_kwargs={"k": configs["num_similar_docs"]}, search_type = "similarity")


# test our retriever
retriever = get_retriever()
similar_documents = retriever.get_relevant_documents("How can I contact OSHA?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC External models are third-party models hosted outside of Databricks. Supported by Model Serving, external models allow you to streamline the usage and management of various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization. You can also use Databricks Model Serving as a provider to serve custom models, which offers rate limits for those endpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC Create a model serving endpoint for the OpenAI external model
# MAGIC
# MAGIC [Details](https://docs.databricks.com/en/machine-learning/model-serving/create-serving-endpoints-mlflow.html)

# COMMAND ----------

import mlflow.deployments
from mlflow.deployments import get_deploy_client

mlflow_deploy_client = mlflow.deployments.get_deploy_client("databricks")
nameep = f"{configs['serving_endpoint_name']}_rkm"
try:
  openaikey = f"{{{{secrets/solution-accelerator-cicd/openai_api}}}}" #change to your key
  mlflow_deploy_client.create_endpoint(
    name=nameep,
    config={
      "served_entities": [{
          "external_model": {
              "name": "gpt-3.5-turbo-instruct",
              "provider": "openai",
              "task": "llm/v1/completions",
              "openai_config": {
                  "openai_api_key": openaikey
              }
          }
      }]
    }
  )
except Exception as e:
  print(e)

# COMMAND ----------


completions_response = mlflow_deploy_client.predict(
    endpoint=nameep,
    inputs={
        "prompt": "How is ph level calculated",
        "temperature": 0.1,
        "max_tokens": 1000,
        "n": 2
    }
)
print(completions_response)

# COMMAND ----------

# MAGIC %md
# MAGIC External model created above

# COMMAND ----------

llm = Databricks(endpoint_name=f"{configs['serving_endpoint_name']}_rkm", extra_params={"temperature": 0.1, "max_tokens": 1000})

# COMMAND ----------

llm.invoke('How is ph level calculated')

# COMMAND ----------

# MAGIC %md
# MAGIC You can also directly call a [Foundational model](https://www.databricks.com/blog/build-genai-apps-faster-new-foundation-model-capabilities)

# COMMAND ----------

llm = Databricks(endpoint_name=f"databricks-mpt-7b-instruct", extra_params={"temperature": 0.1, "max_tokens": 500})

# COMMAND ----------

# MAGIC %md Test the endpoint

# COMMAND ----------

llm.invoke('How is ph level calculated')

# COMMAND ----------

#Example of retrieving prompt from Hub
#from langchain import hub
#prompthub = hub.pull("rlm/rag-prompt")
#chain_type_kwargs = {"prompt":promptv1, "verbose":True} #change to verbose true for printing out entire prompt 

promptTemplate = PromptTemplate(
        template=configs['prompt_template'], input_variables=["context", "question"])
chain_type_kwargs = {"prompt":promptTemplate, "verbose":False} #change to verbose true for printing out entire prompt 



# metadata filtering logic internal implementation, if interested, in 
# def similarity_search_with_score_by_vector in
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/vectorstores/faiss.py

# To test metadata based filtering.
#filterdict={'Name':'ACETALDEHYDE'}
filterdict={}

#get the langchain wrapper around the databricks Vector search
retriever = get_retriever()

#retriever = vectorstore.as_retriever(search_kwargs={"k": configs['num_similar_docs'], "filter":filterdict}, search_type = "similarity")

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=retriever, 
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs,
                                       verbose=False)

# COMMAND ----------

# MAGIC %md Optionally dynamically pass a filter into the chain to pre-filter docs

# COMMAND ----------

filterdict={'Name':'ACETONE'}
filterdict={}
# fetch_k Amount of documents to pass to search algorithm
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":30}
res = qa_chain.invoke("What issues can acetone exposure cause")
print(res)

print(res['result'])

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":20}
res = qa_chain.invoke({"query":"Explain to me the difference between nuclear fission and fusion."})
res

#print(res['result'])

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":40}
res = qa_chain.invoke({'query':'what should we do if OSHA is involved?'})
res

#print(res['result'])


# COMMAND ----------

# MAGIC %md
# MAGIC Optional Cleanup

# COMMAND ----------

del qa_chain

import gc
gc.collect()

# COMMAND ----------


