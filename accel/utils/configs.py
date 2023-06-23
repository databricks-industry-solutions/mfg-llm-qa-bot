# Databricks notebook source
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = dbutils.secrets.get('rkm-scope', 'huggingface') #mfg-llm-solution-accel2
os.environ['HF_HOME'] = '/dbfs/temp/hfmfgcache'

# COMMAND ----------

def dbfsnormalize(path):
  path = path.replace('/dbfs/', 'dbfs:/')
  return path

# COMMAND ----------

configs = {}
configs.update({'vector_persist_dir' : '/dbfs/Users/ramdas.murali@databricks.com/chromadb'}) #/dbfs/temp/faissv1
configs.update({'data_dir':'/dbfs/Users/ramdas.murali@databricks.com/data/sds_pdf'})
configs.update({'chunk_size':600})
configs.update({'chunk_overlap':20})

#configs.update({'temperature':0.8})
#configs.update({'max_new_tokens':128})
configs.update({'prompt_template':"""Use the following pieces of context to answer the question. Dont return a blank answer. 

{context}

Question: {question}
Answer:"""})

configs.update({'num_similar_docs':10})
configs.update({'registered_model_name':'mfg_llm_accel'})
configs.update({'HF_key_secret_scope':'mfg-llm-solution-accel'})
configs.update({'HF_key_secret_key':'huggingface'})
configs.update({'serving_endpoint_name':'mfg-llm-qabot-endpoint'})


configs.update({'model_name' : 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1'})
configs.update({'tokenizer_name' : 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1'})
# configs.update({'model_name' : 'gpt2-xl'})
# configs.update({'tokenizer_name' : 'gpt2-xl'})
configs.update({'model_name' : 'google/flan-t5-xl'})
configs.update({'tokenizer_name' : 'google/flan-t5-xl'})
configs.update({'model_name' : 'bigscience/bloom-7b1'})
configs.update({'tokenizer_name' : 'bigscience/bloom-7b1'})


#torch_dtype=float16, #for gpu
#torch_dtype=bfloat16, #for cpu
#load_in_8bit=True #, 
#max_seq_len=1440 #rkm removed for redpajama

import torch
automodelconfigs = {
    'trust_remote_code':True,
    'device_map':'auto', 
    'torch_dtype':torch.float16
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
else:
  pipelineconfigs = {
      'task':'text-generation',
      'temperature':0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
      'top_p':0.80,  # select from top tokens whose probability add up to 80%
      'top_k':8,  # select from top 0 tokens (because zero, relies on top_p)
      'max_new_tokens':80,  # mex number of tokens to generate in the output
      'repetition_penalty':1.1, # without this output begins repeating
      #'max_answer_len':100, #remove for red pajama & bloom
      'return_full_text':True  # langchain expects the full text
  }  


# COMMAND ----------


