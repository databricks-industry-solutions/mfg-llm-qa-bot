# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Create Embeddings
# MAGIC
# MAGIC So that our qabot application can respond to user questions with relevant answers, we will provide our model with content from documents relevant to the question being asked.  The idea is that the bot will leverage the information in these documents as it formulates a response.
# MAGIC
# MAGIC For our application, we've extracted a series of documents from [New Jersey Chemical Data Fact Sheets](https://web.doh.state.nj.us/rtkhsfs/factsheets.aspx). Using this documentation, we have created a vector database that contains an embedded version of the knowledge stored in these sheets.
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/EntireProcess.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC
# MAGIC In this notebook, we will load these PDF documents, chunk the entire document into pieces and then create embeddings from this.  We will retrieve those documents along with metadata about them and feed that to the Databricks vector store which will create on index enabling fast document search and retrieval.

# COMMAND ----------

# MAGIC %md 
# MAGIC Install Libraries

# COMMAND ----------

# MAGIC %pip install --upgrade PyPDF==3.9.1 pycryptodome==3.18.0 langchain==0.1.6 transformers==4.37.2 databricks-vectorsearch==0.22 mlflow[databricks] xformers==0.0.24  accelerate==0.27.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Configs
# MAGIC %run "./utils/configs"

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# COMMAND ----------

# MAGIC %md
# MAGIC Create the catalogs and schemas for the data and index

# COMMAND ----------

# Need UC permissions to check or create a catalog/schema
# if you dont have create permission then you cant check 'if not exists' either
lstcatalog = spark.catalog.listCatalogs()
exist=False
for cat in lstcatalog:
  if configs["source_catalog"]==cat.name: 
    exist=True

if not exist:
  spark.sql(f'create catalog if not exists {configs["source_catalog"]}')
  spark.sql(f'create schema if not exists {configs["source_catalog"]}.{configs["source_schema"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC Helper routines for extracting metadata

# COMMAND ----------

def dbfsnormalize(path):
  path = path.replace('/dbfs/', 'dbfs:/')
  return path


def extractMetadata(docstr):
  '''
  extracts the common name from the document
  we will use this as metadata for searches
  '''
  dict = {}
  if 'Common Name:' in docstr:
    matches = re.search(r'(?<=Common Name:)(.*?)(?=Synonyms:|Chemical Name:|Date:|CAS Number:|DOT Number:)', docstr)
    if matches is not None and len(matches.groups()) > 0  and matches.groups()[0] is not None :
      dict['Name']=matches.groups()[0].strip()
  return dict

# COMMAND ----------

# MAGIC %md
# MAGIC Use Spark to zip through the docs, read pdf, split pdf and extract metadata from pdf docs.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, ArrayType, MapType, StringType
from pyspark.sql.functions import explode, expr

retschema = StructType([
    StructField("chunks",ArrayType(StringType()), False),
    StructField("metadata", MapType(StringType(), StringType()), False)
])

#use langchains pypdf loader
@udf(returnType=retschema)
def parseAndChunkPDF(pdfPath):
  if not str(pdfPath).endswith('.pdf'):
    return None, None
  pdfurl = pdfPath.replace(':', '')
  pdfurl = '/' + pdfurl
  #use langchains pypdf loader
  loader = PyPDFLoader(pdfurl)
  #list of documents
  pdfdocs = loader.load()
  cleanpdfdocs = []
  metadict={}
  #clean up doc and also extract metadata.
  for doc in pdfdocs:
    doc.page_content=re.sub(r'\n|\uf084', '', doc.page_content)
    if not metadict: #if already extrcacted then use it.
      metadict = extractMetadata(doc.page_content)
    #recreate with cleaned doc
    cleandoc = Document(page_content = doc.page_content, metadata=doc.metadata)
    #append the doc to list
    cleanpdfdocs.append(cleandoc)
  #It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]
  splitter = RecursiveCharacterTextSplitter(chunk_size=configs['chunk_size'], 
                                            chunk_overlap=configs['chunk_overlap'])
  texts = splitter.split_documents(cleanpdfdocs)

  pages_lst=[]

  #add metadata to this block
  for idx2, splitdocs in enumerate(texts):
    pages_lst.append(splitdocs.page_content)  

  #return the chunked docs as a list along with the metadata for the pdf doc
  metaname = metadict['Name'] if 'Name' in metadict else ''
  metadata_i = {'Name':metaname}  
  return pages_lst, metadata_i 


# COMMAND ----------

# MAGIC %md
# MAGIC Use the UDF to parallelize the extract to Delta

# COMMAND ----------


pathlst = dbutils.fs.ls(dbfsnormalize(configs['data_dir']))
df = spark.createDataFrame(pathlst)
df = df.withColumn('chunks', parseAndChunkPDF(df.path))
display(df)



# COMMAND ----------

# MAGIC %md
# MAGIC Explode chunks and metadata to their own row. Also add a guid column as a primary key column
# MAGIC
# MAGIC Requirements for Vector Search
# MAGIC * Unity Catalog enabled workspace.
# MAGIC * Serverless compute enabled.
# MAGIC * Source table must have Change Data Feed enabled.
# MAGIC * CREATE TABLE privileges on catalog schema(s) to create indexes.

# COMMAND ----------

df = df.select('path', 'name', df.chunks.chunks.alias('chunks'), df.chunks.metadata.alias('metadata'))
df = df.select('path', 'name', explode(df.chunks).alias('chunks'), df.metadata.Name.alias('metadata_name')).withColumn('guid', expr("uuid()"))
(df.write
   .format('delta')
   .mode('overwrite').option("overwriteSchema", "true")
   .option("delta.enableChangeDataFeed", "true")
   .saveAsTable(f'{configs["source_catalog"]}.{configs["source_schema"]}.{configs["source_sds_table"]}'))

# COMMAND ----------

# MAGIC %md
# MAGIC Check the table contents

# COMMAND ----------

df = spark.sql(f'select * from {configs["source_catalog"]}.{configs["source_schema"]}.{configs["source_sds_table"]}')
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a Vector Search Client

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
#token and url set in environment var
vsc = VectorSearchClient(workspace_url=configs["DATABRICKS_URL"], personal_access_token=configs['DATABRICKS_TOKEN'])


# COMMAND ----------

# MAGIC %md 
# MAGIC Create the endpoint

# COMMAND ----------

try:
# disable if using a shared vector endpoint to avoid accidentally deleting it.
# vsc.delete_endpoint(name=configs['vector_endpoint_name'])  
  vsc.create_endpoint(
                name=configs['vector_endpoint_name'],
                endpoint_type="STANDARD")
except Exception as e:
  print(e)

# COMMAND ----------

#some time for the operations to complete
import time
time.sleep(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Delete existing indexes and vector search endpoint (optional)

# COMMAND ----------

lemap = vsc.list_endpoints()

lenamelst = [True if le['name']==configs['vector_endpoint_name'] else False for le in lemap.get('endpoints', [])]
if any(lenamelst) is False:
  print(f"{configs['vector_endpoint_name']} Endpoint not found')")
  
limap = vsc.list_indexes(configs['vector_endpoint_name'])
liname = [True if li['name']==f"{configs['source_catalog']}.{configs['source_schema']}.{configs['vector_index']}" else False for li in limap.get('vector_indexes', [])]
if any(liname):
  vsc.delete_index(endpoint_name=configs['vector_endpoint_name'], index_name=f"{configs['source_catalog']}.{configs['source_schema']}.{configs['vector_index']}")


# COMMAND ----------

#some time for the operations to complete
import time
time.sleep(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Get the endpoint details

# COMMAND ----------

endpoint = vsc.get_endpoint(
  name=configs['vector_endpoint_name'])
endpoint

# COMMAND ----------

time.sleep(20)

# COMMAND ----------

# MAGIC %md 
# MAGIC Create the Delta sync index
# MAGIC use the guid as the primary key

# COMMAND ----------

index = vsc.create_delta_sync_index(
    endpoint_name=configs['vector_endpoint_name'],
    source_table_name=f"{configs['source_catalog']}.{configs['source_schema']}.{configs['source_sds_table']}",
    index_name=f"{configs['source_catalog']}.{configs['source_schema']}.{configs['vector_index']}",
    pipeline_type='TRIGGERED',
    primary_key="guid",
    embedding_source_column="chunks",
    embedding_model_endpoint_name=configs['embedding_model_endpoint']
  )


# COMMAND ----------

# MAGIC %md
# MAGIC Query the newly created index

# COMMAND ----------

index = vsc.get_index(endpoint_name=configs['vector_endpoint_name'], index_name=f"{configs['source_catalog']}.{configs['source_schema']}.{configs['vector_index']}")

index.describe()

# COMMAND ----------

# MAGIC %md Check the status of the index and wait till it is **Online**

# COMMAND ----------

import time
while not index.describe().get('status').get('detailed_state').startswith('ONLINE'):
  print("Waiting for index to be ONLINE...")
  time.sleep(15)
print("Index is ONLINE")
retindex = index.describe()
retindex

# COMMAND ----------

time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Issue the command to trigger a sync

# COMMAND ----------

retindex = index.describe()
if retindex['status']['detailed_state'] in 'ONLINE_NO_PENDING_UPDATE':
  print('Syncing...')
  retsync = index.sync()
else:
  print('Cannot Sync Status as ' + retindex['status']['detailed_state'])
  print(retindex)


# COMMAND ----------

# returns [col1, col2, ...]
# this can be set to any subset of the columns
all_columns = spark.table(f"{configs['source_catalog']}.{configs['source_schema']}.{configs['source_sds_table']}").columns
print(all_columns)
results = index.similarity_search(
  query_text="What happens with acetaldehyde chemical exposure?",
  columns=all_columns,
  num_results=10)

results

# COMMAND ----------

# MAGIC %md
# MAGIC Sample query

# COMMAND ----------

results = index.similarity_search(
  query_text="What happens with acetaldehyde chemical exposure?",
  columns=all_columns,
  filters = {"metadata_name": "ACETALDEHYDE"},
  num_results=10 )

results

# COMMAND ----------

results = index.similarity_search(
  query_text="what happens if there are hazardous substances?",
  columns=all_columns,
  filters = {"metadata_name": "ACETONITRILE"},
  num_results=10 )

results

# COMMAND ----------

# MAGIC %md
# MAGIC Extract the response section 

# COMMAND ----------

results['result']['data_array'][2][2]

# COMMAND ----------


