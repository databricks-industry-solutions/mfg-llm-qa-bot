# Databricks notebook source
# MAGIC %pip install -U langchain==0.0.197 transformers==4.30.1 accelerate==0.20.3 einops==0.6.1 xformers==0.0.20 typing-inspect==0.8.0 typing_extensions==4.5.0 faiss-cpu==1.7.4 tiktoken==0.4.0 sentence-transformers

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
      model = transformers.AutoModelForCausalLM.from_pretrained(
          self._configs['model_name'],
          **self._automodelconfigs
      )

      model.eval()
      model.to(device)
      #model.to(device) not valid for 4 bit and 8 bit devices
      if 'RedPajama' in configs['model_name']:
        model.tie_weights()

      print(f"Model loaded on {device}")

      print('Loading tokenizer')
      tokenizer = transformers.AutoTokenizer.from_pretrained(self._configs['tokenizer_name'])
      print('in transformers pipeline')
      generate_text = transformers.pipeline(
          model=model, tokenizer=tokenizer,
          device=device,
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
    llm = self.loadModel()
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
    results=[]
    result = {'answer':None, 'source':None, 'output_metadata':None}
    resultErr = {'answer':'qa_chain is not initalized!', 'source':'MLFlow Model', 'output_metadata':None}
    if self._qa_chain is None:
      print('qa_chain is not initialized!')
      return results.append(resultErr)
    questions = list(inputs['questions'])
    print(questions)
    for question in questions:
      #get relevant documents
      doc = self._qa_chain({'query':question})
      #print(question)
      #print(doc)
      result['answer'] = doc['result']
      result['source'] = ','.join([ src.metadata['source'] for src in doc['source_documents']])   
      results.append(result)
    return results



# COMMAND ----------


