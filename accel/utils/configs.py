# Databricks notebook source
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = dbutils.secrets.get('rkm-scope', 'huggingface')
os.environ['HF_HOME'] = '/dbfs/temp/hfmfgcache'

# COMMAND ----------

def dbfsnormalize(path):
  path = path.replace('/dbfs/', 'dbfs:/')
  return path

# COMMAND ----------

configs = {}
configs.update({'vector_persist_dir' : '/dbfs/Users/ramdas.murali@databricks.com/chromadb'})
configs.update({'data_dir':'/dbfs/Users/ramdas.murali@databricks.com/data/sds_pdf'})
configs.update({'chunk_size':600})
configs.update({'chunk_overlap':20})

#configs.update({'temperature':0.8})
#configs.update({'max_new_tokens':128})
configs.update({'prompt_template':"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}"""})

configs.update({'num_similar_docs':10})
configs.update({'registered_model_name':'mfg_llm_accel'})
configs.update({'HF_key_secret_scope':'mfg-llm-solution-accel'})
configs.update({'HF_key_secret_key':'huggingface'})
configs.update({'serving_endpoint_name':'mfg-llm-qabot-endpoint'})


configs.update({'model_name' : 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1'})
#'togethercomputer/RedPajama-INCITE-Instruct-3B-v1' , "EleutherAI/gpt-neox-20b"
configs.update({'tokenizer_name' : 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1'})


#torch_dtype=float16, #for gpu
#torch_dtype=bfloat16, #for cpu
#load_in_8bit=True #, 
#max_seq_len=1440 #rkm removed for redpajama

import torch
automodelconfigs = {
    'trust_remote_code':True,
    'device_map':'auto', 
    'torch_dtype':torch.float16,
    'load_in_8bit':True 
    }

pipelineconfigs = {
    'task':'text-generation',
    'temperature':0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    'top_p':0.80,  # select from top tokens whose probability add up to 80%
    'top_k':0,  # select from top 0 tokens (because zero, relies on top_p)
    'max_new_tokens':128,  # mex number of tokens to generate in the output
    'repetition_penalty':1.1, # without this output begins repeating
    'return_full_text':True  # langchain expects the full text
}


# COMMAND ----------


