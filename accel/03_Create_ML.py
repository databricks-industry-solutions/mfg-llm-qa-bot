# Databricks notebook source
# MAGIC %pip install -U chromadb==0.3.26 langchain==0.0.197 transformers==4.30.1 accelerate==0.20.3 bitsandbytes==0.39.0 einops==0.6.1 xformers==0.0.20

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

# from langchain import HuggingFacePipeline
# from transformers import AutoTokenizer, pipeline
# from langchain.chains import RetrievalQA
# from langchain import LLMChain
# class MfgLLMWrapper():


#   def __init__(self, llm, retriever, prompt):
#     self.llm = llm
#     self.retriever = retriever
#     self.prompt = prompt

#     chain_type_kwargs = {"prompt":prompt}

#     qa_chain = RetrievalQA.from_chain_type(llm=llm, 
#                                           chain_type="stuff", 
#                                           retriever=retriever, 
#                                           return_source_documents=True,
#                                           chain_type_kwargs=chain_type_kwargs,
#                                           verbose=False)

#   # def _is_good_answer(self, answer):

#   #   ''' check if answer is a valid '''

#   #   result = True # default response

#   #   badanswer_phrases = [ # phrases that indicate model produced non-answer
#   #     "no information", "no context", "don't know", "no clear answer", "sorry", 
#   #     "no answer", "no mention", "reminder", "context does not provide", "no helpful answer", 
#   #     "given context", "no helpful", "no relevant", "no question", "not clear",
#   #     "don't have enough information", " does not have the relevant information", "does not seem to be directly related"
#   #     ]
    
#   #   if answer is None: # bad answer if answer is none
#   #     results = False
#   #   else: # bad answer if contains badanswer phrase
#   #     for phrase in badanswer_phrases:
#   #       if phrase in answer.lower():
#   #         result = False
#   #         break
    
#   #   return result


#   def _get_answer(self, context, question, timeout_sec=60):

#     '''' get answer from llm with timeout handling '''

#     # default result
#     result = None

#     # define end time
#     end_time = time.time() + timeout_sec

#     # try timeout
#     while time.time() < end_time:

#       # attempt to get a response
#       try: 
#         result =  qa_chain.generate([{'context': context, 'question': question}])
#         break # if successful response, stop looping

#       # if rate limit error...
#       except openai.error.RateLimitError as rate_limit_error:
#         if time.time() < end_time: # if time permits, sleep
#           time.sleep(2)
#           continue
#         else: # otherwise, raiser the exception
#           raise rate_limit_error

#       # if other error, raise it
#       except Exception as e:
#         print(f'LLM QA Chain encountered unexpected error: {e}')
#         raise e

#     return result


#   def get_answer(self, question):
#     ''' get answer to provided question '''

#     # default result
#     result = {'answer':None, 'source':None, 'output_metadata':None}

#     # remove common abbreviations from question
#     for abbreviation, full_text in self.abbreviations.items():
#       pattern = re.compile(fr'\b({abbreviation}|{abbreviation.lower()})\b', re.IGNORECASE)
#       question = pattern.sub(f"{abbreviation} ({full_text})", question)

#     # get relevant documents
#     docs = self.retriever.get_relevant_documents(question)

#     # for each doc ...
#     for doc in docs:

#       # get key elements for doc
#       text = doc.page_content
#       source = doc.metadata['source']

#       # get an answer from llm
#       output = self._get_answer(text, question)
 
#       # get output from results
#       generation = output.generations[0][0]
#       answer = generation.text
#       output_metadata = output.llm_output

#       # assemble results if not no_answer
#       if self._is_good_answer(answer):
#         result['answer'] = answer
#         result['source'] = source

#         break # stop looping if good answer
      

#     return result

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModelContext
import torch
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.prompts import PromptTemplate
import torch
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.memory import VectorStoreRetrieverMemory
import gc

# COMMAND ----------


class MLflowMfgBot(mlflow.pyfunc.PythonModel):

  def __init__(self, prompt_template_str, chroma_persist_dir, temperature=0.8, max_new_tokens=128, num_similar_docs=5):
    self._prompt_template_str = prompt_template_str
    self._chroma_persist_dir = chroma_persist_dir
    self._temperature = temperature
    self._max_new_tokens = max_new_tokens
    self._num_similar_docs = num_similar_docs
    self._qa_chain = None
  
  # def __getstate__(self):
  #   d = dict(self.__dict__).copy()
  #   del d['_qa_chain']
  #   return d


  def loadModel(self):
    try:
      print(f'Prompt {self._prompt_template_str}' )
      print(f'Chroma dir {self._chroma_persist_dir}' )   
      print(f'temperature {self._temperature}' )   
      print(f'Max new tokens {self._max_new_tokens}' )   
      print(f'Similar Docs {self._num_similar_docs}' )   

      device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
      print('Loading Vectorstore')
      vectorstore = Chroma(
              collection_name="mfg_collection",
              persist_directory=self._chroma_persist_dir,
              embedding_function=HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2'))

      print('Loading Model')
      model = transformers.AutoModelForCausalLM.from_pretrained(
          'mosaicml/mpt-7b-instruct',
          trust_remote_code=True,
          torch_dtype=bfloat16
      )

      model.eval()
      model.to(device)
      print(f"Model loaded on {device}")
      print('Loading tokenizer')
      tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


      generate_text = transformers.pipeline(
          model=model, tokenizer=tokenizer,
          return_full_text=True,  # langchain expects the full text
          task='text-generation',
          device=device,
          # we pass model parameters here too
          #stopping_criteria=stopping_criteria,  # without this model will ramble
          temperature=self._temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
          top_p=0.80,  # select from top tokens whose probability add up to 15%
          top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
          max_new_tokens=self._max_new_tokens,  # mex number of tokens to generate in the output
          repetition_penalty=1.1 #, without this output begins repeating
          #eos_token_id=tokenizer.eos_token_id          
      )

      print('Creating Pipeline')
      llm = HuggingFacePipeline(pipeline=generate_text)

      retriever = vectorstore.as_retriever(search_kwargs={"k": self._num_similar_docs}) #, "search_type" : "similarity"
      return (llm, retriever)
    except Exception as e:
      print(e)
      _qa_chain=None
      gc.collect()
      torch.cuda.empty_cache()   
    
  
  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
    Args:
        context: MLflow context where the model artifact is stored.
    """
    llm, retriever = self.loadModel()
    print('Getting RetrievalQA handle')
    promptTemplate = PromptTemplate(
        template=self._prompt_template_str, input_variables=["context", "question"])
    
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


# instantiate bot object
mfgsdsbot = MLflowMfgBot(
        configs['prompt_template'], 
        configs['chroma_persist_dir'],
        configs['temperature'], 
        configs['max_new_tokens'],
        configs['num_similar_docs'])

#context = mlflow.pyfunc.PythonModelContext(artifacts={"prompt_template":configs['prompt_template']})

#mfgsdsbot.load_context(context)
# get response to question


# COMMAND ----------



# COMMAND ----------

#mfgsdsbot.predict(context, {'questions':['When is medical attention needed?']})

# COMMAND ----------

# MAGIC %fs ls /Users/ramdas.murali@databricks.com/data/gptmodel

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
  f'xformers==0.0.20'
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
      conda_env=conda_env,
      artifact_path='mfgmodel',
      registered_model_name=configs['registered_model_name']
      )
    )

# COMMAND ----------

print(configs['prompt_template'])

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

import pandas as pd
# construct search
search = pd.DataFrame({'questions':['what should we do if OSHA is involved?']})

# call model
y = model.predict(search)
print(y)

# COMMAND ----------

y=model.predict({'questions':['what should we do if OSHA is involved?']})
print(y)

# COMMAND ----------

y=model.predict({'questions':['When is medical attention needed?']})
print(y)


# COMMAND ----------


