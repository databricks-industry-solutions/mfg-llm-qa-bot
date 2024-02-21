# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Create ML
# MAGIC
# MAGIC In this notebook, we create a custom MLflow pyfunc wrapper which loads a Databricks foundational model in MLflow. This continues from notebook 02.1 We do this to follow MLOps best practices and simplify the deployment of our application. 
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/MLflow-RAG.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC This notebook was tested on the following infrastructure:
# MAGIC * DBR 13.3ML
# MAGIC * i3.xlarge (AWS) GPU not needed

# COMMAND ----------

# MAGIC %md
# MAGIC Install Libraries

# COMMAND ----------

# MAGIC %pip install --upgrade langchain==0.1.6 SQLAlchemy==2.0.27 transformers==4.37.2 databricks-vectorsearch==0.22 mlflow[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.llms import Databricks

import json
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
      #load our databricks foundational model
      llm = Databricks(endpoint_name=f"databricks-mpt-7b-instruct", extra_params={"temperature": 0.1, "max_tokens": 1000})
      return llm

    except Exception as e:
      logger.info("ErrorDel")
      logger.info(e)
    

  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
    Args:
        context: MLflow context where the model artifact is stored.
    """
    logger = logging.getLogger('mlflow.store')
    logger.setLevel(logging.INFO)
    #this is passed in  
    modconfig = context.model_config
    logger.info('incoming model config ' + str(modconfig))
    self._configs = json.loads(modconfig['configs'])
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


