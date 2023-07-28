# Databricks notebook source
# MAGIC %md ##Create ML
# MAGIC
# MAGIC In this notebook, we create a custom MLflow pyfunc wrapper to store our langchain model in MLflow. We do this to follow MLOps best practices and simplify the deployment of our application
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/MLflow-RAG.png" width="700" />
# MAGIC </p>
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U langchain==0.0.203 transformers==4.30.1 accelerate==0.20.3 einops==0.6.1 xformers==0.0.20 typing-inspect==0.8.0 typing_extensions==4.5.0 faiss-cpu==1.7.4 tiktoken==0.4.0 sentence-transformers==2.2.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

import mlflow
import torch
import transformers

from mlflow.pyfunc import PythonModelContext
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from torch import cuda, bfloat16,float16

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from transformers import StoppingCriteria, StoppingCriteriaList
import gc

# COMMAND ----------

from utils.stoptoken import StopOnTokens
import json

class MLflowMfgBot(mlflow.pyfunc.PythonModel):


  def __init__(self, configs, automodelconfigs, pipelineconfigs, retriever, huggingface_token):
    self._configs = configs
    self._automodelconfigs = automodelconfigs
    self._pipelineconfigs = pipelineconfigs
    self._retriever = retriever
    self._huggingface_token = huggingface_token
    self._qa_chain = None
  
  def __getstate__(self):
    d = dict(self.__dict__).copy()
    del d['_qa_chain']
    return d


  def loadModel(self):
    try:
      print(f'configs {self._configs}' )
      print(f'model configs {self._automodelconfigs}' )   
      print(f'pipeline configs {self._pipelineconfigs}' )        

      device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

      print('Loading Model')
      if 'flan' not in self._configs['model_name']:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._configs['model_name'],
            **self._automodelconfigs
        )
      elif 'mpt' in self._configs['model_name']:
        modconfig = transformers.AutoConfig.from_pretrained(self._configs['model_name'] ,
        trust_remote_code=True
        )
        #modconfig.attn_config['attn_impl'] = 'triton'
        model = transformers.AutoModelForCausalLM.from_pretrained(
        self._configs['model_name'],
        config=modconfig,
        **self._automodelconfigs
        )     
      else:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self._configs['model_name'],
            **self._automodelconfigs
        )      

      #model.to(device) not valid for 4 bit and 8 bit devices
      listmc = self._automodelconfigs.keys()
      if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
        model.eval()
        model.to(device)
      if 'RedPajama' in self._configs['model_name']:
        model.tie_weights()
      
      print(f"Model loaded on {device}")

      print('Loading tokenizer')
      tokenizer = transformers.AutoTokenizer.from_pretrained(self._configs['tokenizer_name'])
      print('in transformers pipeline')
      if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            device=device,
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

      print('Creating HF Pipeline')
      llm = HuggingFacePipeline(pipeline=generate_text)

      return llm
    except Exception as e:
      print("ErrorDel")
      print(e)
      _qa_chain=None
      gc.collect()
      torch.cuda.empty_cache()   
    
  
  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
    Args:
        context: MLflow context where the model artifact is stored.
    """
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = self._huggingface_token
      
    if 'lama-2-' in self._configs['model_name']:
      import subprocess
      retval = subprocess.call(f'huggingface-cli login --token {self._huggingface_token}', shell=True)
      print(f"{self._configs['model_name']} limited auth is complete-{retval}")    

    llm = self.loadModel()
    if llm is None:
      print('cannot load context because model was not loaded')
      return
    print('Getting RetrievalQA handle')
    promptTemplate = PromptTemplate(
        template=self._configs['prompt_template'], input_variables=["context", "question"])
    
    chain_type_kwargs = {"prompt":promptTemplate}    
    self._qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                          chain_type="stuff", 
                                          retriever=self._retriever, 
                                          return_source_documents=True,
                                          chain_type_kwargs=chain_type_kwargs,
                                          verbose=False)
 


  def predict(self, context, inputs):
    result = {'answer':None, 'source':None, 'output_metadata':None}
    resultErr = {'answer':'qa_chain is not initalized!', 'source':'MLFlow Model', 'output_metadata':None}
    if self._qa_chain is None:
      print('qa_chain is not initialized!')
      return resultErr

    question = inputs.iloc[0][0]
    filter={}
    filter['k']=6 #num documents to look at for response
    if 'filter' in inputs:
      filter['filter'] = inputs.iloc[0][1]
      filter['fetch_k']=30 #num of documents to get before applying the filter.
    print(question)
    print(filter)
    #get relevant documents

    self._retriever.search_kwargs = filter #{"k": 6, "filter":filterdict, "fetch_k":20}
    doc = self._qa_chain({'query':question})

    result['answer'] = doc['result']
    result['source'] = ','.join([ src.metadata['source'] for src in doc['source_documents']])   

    return result


