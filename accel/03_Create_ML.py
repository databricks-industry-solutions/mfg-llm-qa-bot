# Databricks notebook source
# MAGIC %pip install -U chromadb==0.3.26 langchain==0.0.197 transformers==4.30.1 accelerate==0.20.3 bitsandbytes==0.39.0 einops==0.6.1 xformers==0.0.20 typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModelContext
import torch
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.prompts import PromptTemplate
import torch
from torch import cuda, bfloat16,float16
import transformers
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import StoppingCriteria, StoppingCriteriaList
import gc

# COMMAND ----------


from utils.stoptoken import StopOnTokens

class MLflowMfgBot(mlflow.pyfunc.PythonModel):


  def __init__(self, configs, automodelconfigs, pipelineconfigs, huggingface_token):
    self._configs = configs
    self._automodelconfigs = automodelconfigs
    self._pipelineconfigs = pipelineconfigs
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
      print('Loading Vectorstore')
      vectorstore = Chroma(
              collection_name="mfg_collection",
              persist_directory=self._configs['chroma_persist_dir'],
              embedding_function=HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2'))

      print('Loading Model')
      model = transformers.AutoModelForCausalLM.from_pretrained(
          self._configs['model_name'],
          **self._automodelconfigs
      )

      model.eval()
      if 'RedPajama' in configs['model_name']:
        model.tie_weights()
        model.to(device)

      print(f"Model loaded on {device}")
      print('Loading tokenizer')
      tokenizer = transformers.AutoTokenizer.from_pretrained(self._configs['tokenizer_name'])
      print('in transformers pipeline')
      generate_text = transformers.pipeline(
          model=model, tokenizer=tokenizer,
          **self._pipelineconfigs
      )

      print('Creating HF Pipeline')
      llm = HuggingFacePipeline(pipeline=generate_text)

      retriever = vectorstore.as_retriever(search_kwargs={"k": self._configs['num_similar_docs']}) #, "search_type" : "similarity" self._num_similar_docs
      print('Completed retriever')
      return (llm, retriever)
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
    llm, retriever = self.loadModel()
    print('Getting RetrievalQA handle')
    promptTemplate = PromptTemplate(
        template=self._configs['prompt_template'], input_variables=["context", "question"])
    
    chain_type_kwargs = {"prompt":promptTemplate}    
    self._qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                          chain_type="stuff", 
                                          retriever=retriever, 
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
      print(question)
      print(doc)
      result['answer'] = doc['result']
      result['source'] = ','.join([ src.metadata['source'] for src in doc['source_documents']])   
      results.append(result)
    return results



# COMMAND ----------


