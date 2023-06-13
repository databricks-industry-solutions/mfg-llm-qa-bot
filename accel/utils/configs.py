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
configs.update({'chroma_persist_dir' : '/dbfs/Users/ramdas.murali@databricks.com/chromadb'})
configs.update({'data_dir':'/dbfs/Users/ramdas.murali@databricks.com/data/sds_pdf'})
configs.update({'chunk_size':600})
configs.update({'chunk_overlap':20})

configs.update({'temperature':0.8})
configs.update({'max_new_tokens':128})
configs.update({'prompt_template':"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}"""})
configs.update({'num_similar_docs':8})

configs.update({'registered_model_name':'rkm_mfg_accel'})




# COMMAND ----------


