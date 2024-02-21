# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Define Basic Search
# MAGIC
# MAGIC In this notebook, we will test out loading the vector database for similarity search. Additionally, we create a simple example of combining the open sourced LLM (defined in the /utils/configs) and the similarity search as a retriever. Think of this as a stand-alone implementation without any MLflow packaging
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/Basic-similarity-search.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC This notebook was tested on the following infrastructure:
# MAGIC * DBR 13.3ML (GPU)
# MAGIC * g5.2xlarge(AWS) - however comparable infra on Azure should work (A10s)

# COMMAND ----------

# MAGIC %md
# MAGIC CUDA [memory management flag](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

# COMMAND ----------

# MAGIC %sh export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# COMMAND ----------

# MAGIC %md
# MAGIC Install Libraries

# COMMAND ----------

# MAGIC %pip install --upgrade langchain==0.1.6 sqlalchemy==2.0.27 transformers==4.37.2 databricks-vectorsearch==0.22 mlflow[databricks] xformers==0.0.24  accelerate==0.27.0 google-search-results wikipedia

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from torch import cuda, bfloat16,float16
import transformers
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
import torch
import gc

# COMMAND ----------

def reconvertVals(configs):
  ''' Convert string values in config to the right types'''
  for key in configs:
    if key in ['trust_remote_code', 'return_full_text', 'low_cpu_mem_usage'] and configs[key] is not None:
      configs[key] = bool(configs[key])
    if key in 'torch_dtype' and isinstance(configs['torch_dtype'], str) and configs['torch_dtype'] in 'torch.bfloat16':
      configs['torch_dtype'] = torch.bfloat16
    if key in 'torch_dtype' and isinstance(configs['torch_dtype'], str) and configs['torch_dtype'] in 'torch.float16':
      configs['torch_dtype'] = torch.float16


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

#configs for the model are externalized in var automodelconfigs
try:
  device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
  reconvertVals(automodelconfigs)
  reconvertVals(pipelineconfigs)
  print(f"{configs['model_name']} using configs {automodelconfigs}")
  #account for small variations in code for loading models between models
  if 'mpt' in configs['model_name']:
    modconfig = transformers.AutoConfig.from_pretrained(configs['model_name'] ,
      trust_remote_code=True
    )
    #modconfig.attn_config['attn_impl'] = 'triton'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        configs['model_name'],
        config=modconfig,
        **automodelconfigs
    )
  elif 'flan' in configs['model_name']:
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        configs['model_name'],
        **automodelconfigs
    )
  else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        configs['model_name'],
        **automodelconfigs
    )

  #  model.to(device) -> `.to` is not supported for `4-bit` or `8-bit` models.
  listmc = automodelconfigs.keys()

  # if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
  #   model.eval()
  #   model.to(device)
  
  if 'RedPajama' in configs['model_name']:
    model.tie_weights()

  print(f"Model loaded on {device}")

except Exception as e:
  print('-----')
  print(e)
  gc.collect()
  torch.cuda.empty_cache()   




# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Huggingface model card will give you info on the tokenizer

# COMMAND ----------

token_model= configs['tokenizer_name']
#load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(token_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally we need to define the _stopping criteria_ of the model. The stopping criteria allows us to specify *when* the model should stop generating text. If we don't provide a stopping criteria the model just goes on a bit of a tangent after answering the initial question.

# COMMAND ----------

#If Stopping Criteria is needed
from transformers import StoppingCriteria, StoppingCriteriaList

# for example. mpt-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
print(stop_token_ids)
print(tokenizer.eos_token)
print(stop_token_ids)

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# COMMAND ----------

# device=device, -> `.to` is not supported for `4-bit` or `8-bit` models.
# accelerate lib prints this
# The model has been loaded with `accelerate` and therefore cannot be moved to a specific device. Please discard the `device` argument when creating your pipeline object.
if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
  generate_text = transformers.pipeline(
      model=model, tokenizer=tokenizer,
      #device=device,
      pad_token_id=tokenizer.eos_token_id,
      #stopping_criteria=stopping_criteria,
      **pipelineconfigs
  )
else:
  generate_text = transformers.pipeline(
      model=model, tokenizer=tokenizer,
      pad_token_id=tokenizer.eos_token_id,
      #stopping_criteria=stopping_criteria,
      **pipelineconfigs
  )       




# COMMAND ----------

def get_retriever():
    '''Get the langchain vector retriever from the Databricks object '''
    vsc = VectorSearchClient(workspace_url=configs["DATABRICKS_URL"], personal_access_token=configs['DATABRICKS_TOKEN'])  
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
# MAGIC Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.
# MAGIC The easiest way to tackle NLP tasks is to use the pipeline function. It connects a model with its necessary pre-processing and post-processing steps. This allows you to directly input any text and get an answer.
# MAGIC
# MAGIC This is the critical element to understand how the Databricks vectorstore is being passed to the QA chain as a retriever (the retrieval augmentation)
# MAGIC
# MAGIC Additional ref docs [here](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)

# COMMAND ----------

llm = HuggingFacePipeline(pipeline=generate_text)

promptTemplate = PromptTemplate(
        template=configs['prompt_template'], input_variables=["context", "question"])
chain_type_kwargs = {"prompt":promptTemplate, "verbose":True} #change to verbose true for printing out entire prompt 

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
                                       return_source_documents=False,
                                       chain_type_kwargs=chain_type_kwargs,
                                       verbose=False)

# COMMAND ----------

# MAGIC %md Optionally dynamically pass a filter into the chain to pre-filter docs

# COMMAND ----------

#filterdict={'Name':'ACETALDEHYDE'} #doesnt work
print(retriever.search_kwargs)
# fetch_k Amount of documents to pass to search algorithm
#retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":30}
question = {"query": "What issues can acetone exposure cause"}
answer = qa_chain.invoke(question)
print(answer)
#print(res['result'])

# COMMAND ----------

#filterdict={'Name':'ACETONE'}

# fetch_k Amount of documents to pass to search algorithm
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":30}
res = qa_chain.invoke({"query":"What issues can acetone exposure cause"})
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
# MAGIC Cleanup(Optional)

# COMMAND ----------

del qa_chain
del tokenizer
del model
with torch.no_grad():
    torch.cuda.empty_cache()
import gc
gc.collect()
