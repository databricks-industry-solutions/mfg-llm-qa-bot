# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Create ML
# MAGIC
# MAGIC In this notebook, we create a custom MLflow pyfunc wrapper to store our langchain model in MLflow. We load and use an open source LLM model. We do this to follow MLOps best practices and simplify the deployment of our application. 
# MAGIC
# MAGIC This continues from notebook 02.3
# MAGIC
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
# MAGIC CUDA [memory management flag](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

# COMMAND ----------

# MAGIC %sh export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# COMMAND ----------

# MAGIC %md
# MAGIC Install Libraries

# COMMAND ----------

# MAGIC %pip install --upgrade langchain==0.1.6 SQLAlchemy==2.0.27 transformers==4.37.2 databricks-vectorsearch==0.22 mlflow[databricks] xformers==0.0.24  accelerate==0.27.0

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
from mlflow.pyfunc import PythonModelContext
from utils.stoptoken import StopOnTokens
import json
import torch
import gc

import logging
import sys



# COMMAND ----------

# MAGIC %md
# MAGIC ##Create wrapper for LLM
# MAGIC
# MAGIC In the code below, we are using the [mlflow.pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models) to create a custom model type in MLflow. We will call this wrapper in the next notebook to save our model to MLflow.
# MAGIC
# MAGIC This approach allows us to customize input and return values to the LLM model
# MAGIC
# MAGIC If you get error messages like
# MAGIC * Some parameters are on the meta device device because they were offloaded to the cpu.
# MAGIC * You can't move a model that has some modules offloaded to cpu or disk.
# MAGIC
# MAGIC if you see above messages then GPU memory may be low

# COMMAND ----------

import json
class MLflowMfgBot(mlflow.pyfunc.PythonModel):

  #constructor with args to pass in during model creation
  def __init__(self):
    pass
  
  def _reconvertVals(self, autoconfig):
    ''' Convert string values in config to the right types'''
    for key in autoconfig:
      if key in ['trust_remote_code', 'return_full_text', 'low_cpu_mem_usage'] and autoconfig[key] is not None:
        autoconfig[key] = bool(autoconfig[key])
      if key in 'torch_dtype' and isinstance(autoconfig['torch_dtype'], str) and autoconfig['torch_dtype'] in 'torch.bfloat16':
        autoconfig['torch_dtype'] = torch.bfloat16
      if key in 'torch_dtype' and isinstance(autoconfig['torch_dtype'], str) and autoconfig['torch_dtype'] in 'torch.float16':
        autoconfig['torch_dtype'] = torch.float16

  def _get_retriever(self):
    '''Get the langchain vector retriever from the Databricks object '''
    logger = logging.getLogger('mlflow.store')
    logger.info('_get_retriever')
    vsc = VectorSearchClient(workspace_url=self._configs["DATABRICKS_URL"], personal_access_token=self._configs['DATABRICKS_TOKEN'])   
    index = vsc.get_index(endpoint_name=self._configs['vector_endpoint_name'], 
                          index_name=f"{self._configs['source_catalog']}.{self._configs['source_schema']}.{self._configs['vector_index']}")

    index.describe()
    # Create the langchain retriever. text_columns-> chunks column
    # return columns metadata_name and path along with results.
    # embedding is None for Databricks managed embedding
    vectorstore = DatabricksVectorSearch(
        index, text_column="chunks", embedding=None, columns=['metadata_name', 'path']
    )
    #filter isnt working here at the moment
    return vectorstore.as_retriever(search_kwargs={"k": self._configs["num_similar_docs"]}, search_type = "similarity")


  #this is not to be confused with load_model. This is private to this class and called internally by load_context.
  def _load_model(self):
    logger = logging.getLogger('mlflow.store')
    try:
      device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
      logger.info(device)
      self._reconvertVals(self._automodelconfigs)
      self._reconvertVals(self._pipelineconfigs)
      print(f"{self._configs['model_name']} using configurations {self._automodelconfigs}")
      #account for small variations in code for loading models between models
      if 'mpt' in self._configs['model_name']:
        modconfig = transformers.AutoConfig.from_pretrained(self._configs['model_name'] ,
          trust_remote_code=True
        )
        #modconfig.attn_config['attn_impl'] = 'triton'
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._configs['model_name'],
            config=modconfig,
            **self._automodelconfigs
        )
      elif 'flan' in self._configs['model_name']:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self._configs['model_name'],
            **self._automodelconfigs
        )
      else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._configs['model_name'],
            **self._automodelconfigs
        )

      #  model.to(device) -> `.to` is not supported for `4-bit` or `8-bit` models.
      listmc = self._automodelconfigs.keys()
      # if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
      #   model.eval()
      #   model.to(device)   
      if 'RedPajama' in self._configs['model_name']:
        model.tie_weights()

      tokenizer = transformers.AutoTokenizer.from_pretrained(self._configs['tokenizer_name'])

      if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            #device=device, #latest accelerate lib doesnt like this.
            pad_token_id=tokenizer.eos_token_id,
            #stopping_criteria=stopping_criteria,
            **self._pipelineconfigs
        )
      else:
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            #stopping_criteria=stopping_criteria,
            **self._pipelineconfigs
        )    
      logger.info('Creating HF Pipeline')
      llm = HuggingFacePipeline(pipeline=generate_text)
      return llm

    except Exception as e:
      logger.info("ErrorDel")
      logger.info(e)
      _qa_chain=None
      gc.collect()
      torch.cuda.empty_cache()  
    

  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
    Args:
        context: MLflow context where the model artifact is stored.
    """

    logger = logging.getLogger('mlflow.store')
    logger.setLevel(logging.INFO)
    #this is passed in  

    logger.info('-1-')
    modconfig = context.model_config
    logger.info('incoming model config ' + str(modconfig))
    self._configs = json.loads(modconfig['configs'])
    self._automodelconfigs = json.loads(modconfig['automodelconfigs'].replace("\'", "\""))
    self._pipelineconfigs = json.loads(modconfig['pipelineconfigs'].replace("\'", "\""))
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = self._configs["HUGGINGFACEHUB_API_TOKEN"]
    os.environ['DATABRICKS_HOST']=self._configs['DATABRICKS_URL']
    os.environ['DATABRICKS_TOKEN']=self._configs['DATABRICKS_TOKEN']
    logger.info('-2-')
    retr = self._get_retriever()
    if retr is None:
      logger.info('Could not load context since Retriever failed')
      return
    self._retriever = retr
    self._qa_chain = None
    logger.info('-3-')
    #have to do a bit of extra initialization with llama-2 models since its by invitation only
    if 'lama-2-' in self._configs['model_name']:
      import subprocess
      retval = subprocess.call(f'huggingface-cli login --token {self._configs["HUGGINGFACEHUB_API_TOKEN"] }', shell=True)
      logger.info(f"{self._configs['model_name']} limited auth is complete-{retval}")    
    logger.info('-4-')
    llm = self._load_model()
    if llm is None:
      logger.info('cannot load context because model was not loaded')
      return
    logger.info('Getting RetrievalQA handle')
    promptTemplate = PromptTemplate(
        template=self._configs['prompt_template'], input_variables=["context", "question"])

    chain_type_kwargs = {"prompt":promptTemplate, "verbose":False}    
    #qa chain is recreated. This is not stored with the model.
    self._qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                          chain_type="stuff", 
                                          retriever=self._retriever, 
                                          return_source_documents=True,
                                          chain_type_kwargs=chain_type_kwargs,
                                          verbose=False)
    logger.info('-5-')



  def predict(self, context, inputs):
    '''Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
    Only one question can be passed in.
    Example of input in json split format thats passed to predict.
    {"dataframe_split":{"columns":["question","filter"],"index":[0],"data":[["what are some properties of Acetaldehyde?",{"Name":"ACETALDEHYDE"}]]}}
    '''
    logger = logging.getLogger('mlflow.store')
    result = {'answer':None, 'source':None, 'output_metadata':None}
    resultQAErr = {'answer':'qa_chain is not initalized!', 'source':'MLFlow Model', 'output_metadata':None}
    resultRetrErr = {'answer':'Retriever is not initalized!', 'source':'MLFlow Model', 'output_metadata':None}
    if self._retriever is None:
      logger.info('retriever is not initialized!')
      return resultRetrErr

    if self._qa_chain is None:
      logger.info('qa_chain is not initialized!')
      return resultQAErr

    if isinstance(inputs, dict):
      question = inputs['questions'][0] 
      filter = {}
      if 'search_kwargs' in inputs:
        filter = inputs['search_kwargs']
    else:
      question = inputs.iloc[0][0]
      filter={}
      filter['k']=6 #num documents to look at for response
      if 'filter' in inputs:
        filter['filter'] = inputs.iloc[0][1]
        filter['fetch_k']=30 #num of documents to get before applying the filter.
      logger.info(question)
      logger.info(filter)

    #get relevant documents
    #inject the filter during every predict.
    self._retriever.search_kwargs = filter #{"k": 6, "filter":filterdict, "fetch_k":20}
    doc = self._qa_chain({'query':question})
    logger.info(doc)
    result['answer'] = doc['result']
    result['source'] = ','.join([ src.metadata['path'] for src in doc['source_documents']])   

    return result


