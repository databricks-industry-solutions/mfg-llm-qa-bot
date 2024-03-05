# Databricks notebook source
#%pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

# MAGIC %md Before setting up the rest of the accelerator, we need set up an [HuggingFace access token in order to access Open sourced models on HuggingFace](https://huggingface.co/docs/hub/security-tokens). Here we demonstrate using the [Databricks Secret Scope](https://docs.databricks.com/security/secrets/secret-scopes.html) for credential management. 
# MAGIC
# MAGIC Copy the block of code below, replace the name of the secret scope and fill in the credentials and execute the block. After executing the code, The accelerator notebook will be able to access the credentials it needs.
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC from solacc.companion import NotebookSolutionCompanion
# MAGIC
# MAGIC client = NotebookSolutionCompanion().client
# MAGIC try:
# MAGIC   client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/scopes/create", {"scope": "solution-accelerator-cicd"})
# MAGIC except:
# MAGIC   pass
# MAGIC   
# MAGIC client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/put", {
# MAGIC   "scope": "solution-accelerator-cicd",
# MAGIC   "key": "huggingface",
# MAGIC   "string_value": '____' 
# MAGIC })
# MAGIC ```

# COMMAND ----------

configs={}

# COMMAND ----------

# DBTITLE 1,Databricks url and token
import os

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
configs['DATABRICKS_TOKEN'] = dbutils.secrets.get('solution-accelerator-cicd', 'mfg-sa-key') 
configs['DATABRICKS_URL'] = ctx.apiUrl().getOrElse(None)

# COMMAND ----------

# DBTITLE 1,Huggingface token
import os
hftoken = dbutils.secrets.get('solution-accelerator-cicd', 'huggingface')   
configs['HUGGINGFACEHUB_API_TOKEN'] =  hftoken
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hftoken
os.environ['HF_HOME'] = '/dbfs/temp/hfmfgcache'


# COMMAND ----------

# MAGIC %md
# MAGIC Llama 2 needs [additional licensing](https://huggingface.co/meta-llama) 

# COMMAND ----------

# DBTITLE 1,Llama token
import subprocess
subprocess.call('huggingface-cli login --token $HUGGINGFACEHUB_API_TOKEN', shell=True)

# COMMAND ----------

# DBTITLE 1,OpenAI token
#needed for notebook 2.2 only
try:
  openaitoken = dbutils.secrets.get('solution-accelerator-cicd', 'openai_api') 
except Exception as e:
  print('No OpenAPI token detected')
  openaitoken=''

configs['OPENAI_API_KEY'] = openaitoken
os.environ['OPENAI_API_KEY'] = openaitoken

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

import mlflow

#to use workspace registry - False
#to use uc registry - True
configs['isucregistry']=False

configs['source_catalog'] = "mfg_llm_cat"
configs['source_schema'] = "mfg_llm_schema"
configs['source_sds_table'] = "mfg_llm_sds"

configs['vector_endpoint_name'] = "one-env-shared-endpoint-1"

# Vector index
configs['vector_index'] = "mfg_llm_solnaccel_index"
configs['embedding_model_endpoint'] = "databricks-bge-large-en"


configs['data_dir'] = f'/dbfs/Users/{username}/data/sds_pdf'
configs['chunk_size']=600
configs['chunk_overlap']=20

configs['prompt_template'] = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Return as much information as possible
    Helpful answer:
    """

configs['num_similar_docs']=10


if configs['isucregistry'] is True:
  mlflow.set_registry_uri("databricks-uc")
  configs['registered_model_name'] = f"{configs['source_catalog']}.{configs['source_schema']}.mfg-llm-qabot"
else:
  mlflow.set_registry_uri("databricks")
  configs['registered_model_name'] = f"mfg-llm-qabot" 

configs['serving_endpoint_name'] = 'mfg-llm-qabotse'


#configs['model_name'] = 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1'
#configs['tokenizer_name'] = 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1'
#configs['model_name'] = 'gpt2-xl'
#configs['tokenizer_name'] = 'gpt2-xl'
#configs['model_name']= 'google/flan-t5-xl'
#configs['tokenizer_name']= 'google/flan-t5-xl'
# configs['model_name'] = 'bigscience/bloomz-7b1'
# configs['tokenizer_name'] = 'bigscience/bloomz-7b1'
# configs['model_name']='mosaicml/mpt-7b-instruct'
# configs['tokenizer_name'] = 'EleutherAI/gpt-neox-20b' #for mpt
#configs['model_name'] = 'tiiuae/falcon-7b-instruct'
#configs['tokenizer_name']= 'tiiuae/falcon-7b-instruct'
configs['model_name'] = 'meta-llama/Llama-2-7b-chat-hf'
configs['tokenizer_name'] = 'meta-llama/Llama-2-7b-chat-hf'


#torch_dtype=float16, #for gpu
#torch_dtype=bfloat16, #for cpu
#load_in_8bit=True #, #8 bit stuff needs accelerate which I couldnt get to work with model serving
#max_seq_len=1440 #rkm removed for redpajama

if 'falcon' in configs['model_name']:
  automodelconfigs = {
      'trust_remote_code':'True',
      'device_map':'auto', 
      'torch_dtype':'torch.float16' #torch.float16 changed to bfloat for falcon. changed back to float for falcon
      }
elif 'lama-2-' in configs['model_name']:
  automodelconfigs = {
    'trust_remote_code':'True',
    'device_map':'auto', 
    #'low_cpu_mem_usage':'True',
    'torch_dtype':'torch.float16'
    } 
else:
  automodelconfigs = {
      'trust_remote_code':'True',
      'device_map':'auto', 
      'torch_dtype':'torch.bfloat16'
      }
  

if 'flan' in configs['model_name']:
  pipelineconfigs = {
      'task':'text2text-generation', #'text-generation',
      'temperature':0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
      'top_p':0.80,  # select from top tokens whose probability add up to 80%
      'top_k':8,  # select from top 0 tokens (because zero, relies on top_p)
      'max_new_tokens':60,  # mex number of tokens to generate in the output
      'repetition_penalty':1.1, # without this output begins repeating
      #'return_full_text':True  # langchain expects the full text
  }
elif 'lama-2-' in configs['model_name']:
  pipelineconfigs = {
      'task':'text-generation',
      'temperature':0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
      'top_p':0.80,  # select from top tokens whose probability add up to 80%
      'top_k':0,  # select from top 0 tokens (because zero, relies on top_p)
      'max_new_tokens':400,  # mex number of tokens to generate in the output
      'repetition_penalty':1.1, # without this output begins repeating
      #'max_length':128,
      'return_full_text':'True'  # langchain expects the full text
  }   
else:
  pipelineconfigs = {
      'task':'text-generation',
      'temperature':0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
      'top_p':0.80,  # select from top tokens whose probability add up to 80%
      'top_k':0,  # select from top 0 tokens (because zero, relies on top_p)
      'max_new_tokens':128,  # mex number of tokens to generate in the output
      'repetition_penalty':1.1, # without this output begins repeating
      #'max_answer_len':100, #remove for red pajama & bloom & falcon
      'return_full_text':'True'  # langchain expects the full text
  }  


# COMMAND ----------

# DBTITLE 1,MLflow experiment
import mlflow

mlflow.set_experiment('/Users/{}/mfg_llm_sds_search'.format(username))
