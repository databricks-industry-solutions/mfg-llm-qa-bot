# Databricks notebook source
# MAGIC %pip install -U chromadb==0.3.26 langchain==0.0.197 transformers==4.30.1 accelerate==0.20.3 bitsandbytes==0.39.0 einops==0.6.1 xformers==0.0.20 typing-inspect==0.8.0 typing_extensions==4.5.0
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %run "./utils/configs"
# MAGIC

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

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


# COMMAND ----------

chroma_persist_dir = configs['chroma_persist_dir']
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings


vectorstore = Chroma(
        collection_name="mfg_collection",
        persist_directory=chroma_persist_dir,
        embedding_function=HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2')
)


def similarity_search(question):
  matched_docs = vectorstore.similarity_search(question, k=12)
  sources = []
  content = []
  for doc in matched_docs:
    sources.append(
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
    )
    content.append(doc.page_content)
    

  return matched_docs, sources, content


matched_docs, sources, content = similarity_search('Who provides recommendations on workspace safety')
print(content)

# COMMAND ----------

embeddings = OpenAIEmbeddings(openai_api_key=dbutils.secrets.get('rkm-scope', 'chatgpt'))
instance = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)

tech_template = """Answer the question at the end using the context provided below.

{context}

Q: {question}
A: """
PROMPT = PromptTemplate(
    template=configs['prompt_template'], input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                                                chain_type="stuff",
                                                retriever=instance.as_retriever(),
                                                chain_type_kwargs={"prompt": PROMPT})

# COMMAND ----------

response=qa.run('what kind of symptoms do you have after a chemical exposure')
print(response)

# COMMAND ----------

class MLflowMfgBotOpen(mlflow.pyfunc.PythonModel):


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
      llm = None

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
    os.environ['OPENAI_API_KEY'] = self._huggingface_token
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

    llm, retriever = self.loadModel()
    print('Getting RetrievalQA handle')
    promptTemplate = PromptTemplate(
        template=self._configs['prompt_template'], input_variables=["context", "question"])
    
    chain_type_kwargs = {"prompt":promptTemplate}    
    self._qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
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

# instantiate bot object
mfgsdsbot = MLflowMfgBotOpen(
        configs,
        automodelconfigs,
        pipelineconfigs,
        dbutils.secrets.get('rkm-scope', 'chatgpt'))


# #for testing locally
# context = mlflow.pyfunc.PythonModelContext(artifacts={"prompt_template":configs['prompt_template']})
# mfgsdsbot.load_context(context)
# # get response to question
# mfgsdsbot.predict(context, {'questions':['when should you do if you get exposed to chemicals?']})

# COMMAND ----------

# get base environment configuration
conda_env = mlflow.pyfunc.get_default_conda_env()
# define packages required by model
packages = [
  f'chromadb==0.3.26',
  f'langchain==0.0.197',
  f'transformers==4.30.1',
  f'accelerate==0.20.3',
  f'bitsandbytes==0.39.0',
  f'einops==0.6.1',
  f'xformers==0.0.20',
  f'typing-inspect==0.8.0',
  f'typing_extensions==4.5.0'
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

y=model.predict({'questions':['what should we do if OSHA is involved?']})
print(y)
